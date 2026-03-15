"""
Main application window coordinating UI and backend logic.
"""
from __future__ import annotations
import os
import time
import queue
import threading

import cv2
import numpy as np
import torch
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tkinter import filedialog, messagebox

from ui.sidebar import Sidebar
from ui.grid_view import GridView
from core.model_loader import load_model
from core.hook_engine import HookEngine
from core.gradcam import compute_gradcam
from processing.tensor_to_image import process_tensor_to_images
from export.reporter import generate_report, save_report
from .theme import *

SSAA = 2  # Supersampling scale factor for all render methods


# ── Rendering utilities ───────────────────────────────────────────────────────

def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _lerp_colour(
    c1: tuple[int, int, int],
    c2: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _apply_colormap(
    values: np.ndarray,
    anchors: list[tuple[float, str]],
) -> np.ndarray:
    out = np.zeros((*values.shape, 3), dtype=np.uint8)
    flat = values.ravel()
    rgb_anchors = [(pos, _hex_to_rgb(col)) for pos, col in anchors]

    for i, v in enumerate(flat):
        v = float(np.clip(v, 0.0, 1.0))
        left_pos, left_col = rgb_anchors[0]
        right_pos, right_col = rgb_anchors[-1]
        for j in range(len(rgb_anchors) - 1):
            if rgb_anchors[j][0] <= v <= rgb_anchors[j + 1][0]:
                left_pos, left_col = rgb_anchors[j]
                right_pos, right_col = rgb_anchors[j + 1]
                break
        span = right_pos - left_pos
        t = (v - left_pos) / span if span > 1e-9 else 0.0
        out.ravel()[i * 3: i * 3 + 3] = _lerp_colour(left_col, right_col, t)

    return out.reshape(*values.shape, 3)


def _smooth_signal(values: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values.astype(np.float32), radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _make_gradient_rect(
    width: int, height: int,
    colour_top: tuple, colour_bottom: tuple,
    alpha_top: int = 255, alpha_bottom: int = 255,
) -> Image.Image:
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    for y in range(height):
        t = y / max(height - 1, 1)
        r, g, b = _lerp_colour(colour_top, colour_bottom, t)
        a = int(alpha_top + (alpha_bottom - alpha_top) * t)
        arr[y, :] = [r, g, b, a]
    return Image.fromarray(arr, mode="RGBA")


def _draw_rounded_rect(draw, xy, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _get_pil_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _vignette_overlay(width: int, height: int, strength: float = 0.6) -> Image.Image:
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    cx, cy = width / 2, height / 2
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    ys, xs = np.mgrid[0:height, 0:width]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / max_dist
    alpha = (dist ** 1.5 * strength * 255).clip(0, 255).astype(np.uint8)
    arr[..., 3] = alpha
    return Image.fromarray(arr, mode="RGBA")


# ── Main application ──────────────────────────────────────────────────────────

class CNNVisualizerApp(ctk.CTk):
    """Main GUI window — coordinates sidebar, grid view, and backend logic."""

    def __init__(self):
        super().__init__()

        self.title("CNN Activation Visualizer")
        self.geometry("1560x920")
        self.minsize(1100, 700)
        self.configure(fg_color=C_BG_DEEP)

        # ── Backend ───────────────────────────────────────────────────────
        self.hook_engine: HookEngine | None = None

        # ── App state ─────────────────────────────────────────────────────
        self.image_path: str | None = None
        self.current_model: str | None = None
        self.camera: cv2.VideoCapture | None = None
        self.camera_active: bool = False
        self.ema_channel_sums: torch.Tensor | None = None
        self.frozen_frame = None

        # ── Diagnostics state ─────────────────────────────────────────────
        self.last_health_metrics: dict | None = None
        self.last_input_stats: dict | None = None
        self.last_layer_name: str | None = None

        # FIX #4: store the last full activation tensor dict here so
        # _update_inspector() can access it after hook_engine.activation
        # has been cleared. Keyed by layer name → (C, H, W) tensor.
        self._last_tensor_dict: dict[str, torch.Tensor] = {}
        # Convenience accessor for the primary selected-layer tensor.
        self._last_tensor: torch.Tensor | None = None  # shape (1, C, H, W)

        # ── Threading ─────────────────────────────────────────────────────
        self._capture_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._inference_stop_event = threading.Event()
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._result_queue: queue.Queue = queue.Queue(maxsize=1)
        self._frame_count: int = 0
        self._cached_redundancy: torch.Tensor | None = None

        # ── Diagnostic chart state ────────────────────────────────────────
        self._last_flow_data: dict[str, float] = {}
        self._flow_frame_count: int = 0
        self._last_sim_matrix: torch.Tensor | None = None
        self._corr_matrix_frame_count: int = 0
        self._corr_matrix_last_update_time: float = 0.0
        self._corr_matrix_sorted_indices: torch.Tensor | None = None

        # FIX #7: _last_sorted_indices was declared but never assigned.
        # It is now set alongside _corr_matrix_sorted_indices after every
        # pipeline run so the inspector can use it for EMA rank display.
        self._last_sorted_indices: torch.Tensor | None = None

        # ── Inspector state ───────────────────────────────────────────────
        self._inspected_channel: int | None = None
        self._inspected_layer: str | None = None
        self._inspector_visible: bool = False

        # Inspector widget references (populated by _build_inspector_skeleton)
        self.inspector_frame: ctk.CTkScrollableFrame | None = None
        self.inspector_title: ctk.CTkLabel | None = None
        self.inspector_close_btn: ctk.CTkButton | None = None
        self.inspector_map_label: ctk.CTkLabel | None = None
        self.inspector_stats: dict[str, ctk.CTkLabel] = {}
        self.inspector_hist_label: ctk.CTkLabel | None = None
        self.inspector_similar_labels: list = []
        self.inspector_bias_label: ctk.CTkLabel | None = None

        self._setup_ui()
        self._build_inspector_skeleton()

    # ------------------------------------------------------------------ #
    #  UI setup                                                            #
    # ------------------------------------------------------------------ #

    def _setup_ui(self):
        self.grid_columnconfigure(0, weight=0, minsize=270)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=350)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = Sidebar(
            self,
            on_model_select=self.on_model_select,
            on_upload=self.on_upload_image,
            on_camera_toggle=self.toggle_camera,
            on_freeze_toggle=self.toggle_freeze,
            on_visualize=self.on_visualize,
            on_export=self.on_export_diagnostics,
            on_speed_change=None,  # polled via sidebar.config['speed']
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.main_outer = ctk.CTkFrame(self, fg_color=C_BG_BASE, corner_radius=0)
        self.main_outer.grid(row=0, column=1, sticky="nsew")
        self.main_outer.grid_rowconfigure(0, weight=0)
        self.main_outer.grid_rowconfigure(1, weight=1)
        self.main_outer.grid_columnconfigure(0, weight=1)

        self.health_strip_label = ctk.CTkLabel(self.main_outer, text="")
        self.health_strip_label.grid(row=0, column=0, sticky="ew")

        self.tabview = ctk.CTkTabview(
            self.main_outer,
            fg_color=C_BG_BASE,
            border_width=1,
            border_color=C_BORDER_SUB,
            segmented_button_fg_color=C_BG_RAISED,
            segmented_button_selected_color=C_ACCENT,
            segmented_button_selected_hover_color=C_BG_FLOAT,
            segmented_button_unselected_color=C_BG_RAISED,
            segmented_button_unselected_hover_color=C_BG_FLOAT,
            text_color=C_TEXT_PRI,
            command=self._on_tab_change,
        )
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.tabview.add("Feature Maps")
        self.tabview.add("Flow Chart")
        self.tabview.add("Correlation")

        self.grid_view = GridView(
            self.tabview.tab("Feature Maps"),
            on_channel_click=self._on_channel_click,
        )
        self.grid_view.pack(fill="both", expand=True)

        self.flow_tab_label = ctk.CTkLabel(
            self.tabview.tab("Flow Chart"), text=""
        )
        self.flow_tab_label.pack(fill="both", expand=True)

        self.corr_tab_label = ctk.CTkLabel(
            self.tabview.tab("Correlation"), text=""
        )
        self.corr_tab_label.pack(fill="both", expand=True)

    def _build_inspector_skeleton(self):
        """Creates all inspector widgets once. Content updated dynamically."""
        self.inspector_frame = ctk.CTkScrollableFrame(
            self,
            width=340,
            fg_color=C_BG_RAISED,
            border_width=1,
            border_color=C_BORDER_SUB,
            corner_radius=0,
            label_text="",
        )

        header = ctk.CTkFrame(self.inspector_frame, fg_color=C_BG_BASE, corner_radius=6)
        header.pack(fill="x", pady=(0, 6))
        self.inspector_title = ctk.CTkLabel(
            header, text="Channel Inspector",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=C_TEXT_PRI,
        )
        self.inspector_title.pack(side="left", padx=12, pady=8)
        self.inspector_close_btn = ctk.CTkButton(
            header, text="✕", width=28, height=28,
            fg_color=C_BG_FLOAT, hover_color=C_CRITICAL,
            command=self._close_inspector,
        )
        self.inspector_close_btn.pack(side="right", padx=6, pady=6)

        map_frame = ctk.CTkFrame(
            self.inspector_frame, fg_color=C_BG_BASE,
            border_width=1, border_color=C_BORDER_SUB, corner_radius=6,
        )
        map_frame.pack(fill="x", pady=6)
        ctk.CTkLabel(map_frame, text="ACTIVATION MAP",
                     font=ctk.CTkFont(size=9, weight="bold"), text_color=C_TEXT_MUT,
                     ).pack(anchor="w", padx=12, pady=(8, 4))
        self.inspector_map_label = ctk.CTkLabel(map_frame, text="")
        self.inspector_map_label.pack(pady=8)

        stats_frame = ctk.CTkFrame(
            self.inspector_frame, fg_color=C_BG_BASE,
            border_width=1, border_color=C_BORDER_SUB, corner_radius=6,
        )
        stats_frame.pack(fill="x", pady=6)
        ctk.CTkLabel(stats_frame, text="STATISTICS",
                     font=ctk.CTkFont(size=9, weight="bold"), text_color=C_TEXT_MUT,
                     ).pack(anchor="w", padx=12, pady=(8, 4))
        for key in ["Layer", "Channel", "EMA Rank", "Act Range", "Mean / Std", "Saturation", "Dead"]:
            row = ctk.CTkFrame(stats_frame, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=2)
            ctk.CTkLabel(row, text=f"{key}:", font=ctk.CTkFont(size=10),
                         text_color=C_TEXT_MUT).pack(side="left")
            vl = ctk.CTkLabel(row, text="-", font=ctk.CTkFont(size=10, weight="bold"),
                               text_color=C_TEXT_PRI)
            vl.pack(side="right")
            self.inspector_stats[key] = vl

        hist_frame = ctk.CTkFrame(
            self.inspector_frame, fg_color=C_BG_BASE,
            border_width=1, border_color=C_BORDER_SUB, corner_radius=6,
        )
        hist_frame.pack(fill="x", pady=6)
        ctk.CTkLabel(hist_frame, text="ACTIVATION DISTRIBUTION",
                     font=ctk.CTkFont(size=9, weight="bold"), text_color=C_TEXT_MUT,
                     ).pack(anchor="w", padx=12, pady=(8, 4))
        self.inspector_hist_label = ctk.CTkLabel(hist_frame, text="")
        self.inspector_hist_label.pack(pady=8)

        sim_frame = ctk.CTkFrame(
            self.inspector_frame, fg_color=C_BG_BASE,
            border_width=1, border_color=C_BORDER_SUB, corner_radius=6,
        )
        sim_frame.pack(fill="x", pady=6)
        ctk.CTkLabel(sim_frame, text="MOST SIMILAR CHANNELS",
                     font=ctk.CTkFont(size=9, weight="bold"), text_color=C_TEXT_MUT,
                     ).pack(anchor="w", padx=12, pady=(8, 4))
        sim_grid = ctk.CTkFrame(sim_frame, fg_color="transparent")
        sim_grid.pack(pady=8)
        for _ in range(3):
            sub = ctk.CTkFrame(sim_grid, fg_color="transparent")
            sub.pack(side="left", padx=4)
            img_lbl = ctk.CTkLabel(sub, text="")
            img_lbl.pack()
            txt_lbl = ctk.CTkLabel(sub, text="-", font=ctk.CTkFont(size=10),
                                    text_color=C_TEXT_MUT)
            txt_lbl.pack()
            self.inspector_similar_labels.append((img_lbl, txt_lbl))

        bias_frame = ctk.CTkFrame(
            self.inspector_frame, fg_color=C_BG_BASE,
            border_width=1, border_color=C_BORDER_SUB, corner_radius=6,
        )
        bias_frame.pack(fill="x", pady=6)
        ctk.CTkLabel(bias_frame, text="SPATIAL FOCUS",
                     font=ctk.CTkFont(size=9, weight="bold"), text_color=C_TEXT_MUT,
                     ).pack(anchor="w", padx=12, pady=(8, 0))
        ctk.CTkLabel(bias_frame, text="Where this filter concentrates attention",
                     font=ctk.CTkFont(size=9), text_color=C_TEXT_MUT,
                     ).pack(anchor="w", padx=12, pady=(0, 4))
        self.inspector_bias_label = ctk.CTkLabel(bias_frame, text="")
        self.inspector_bias_label.pack(pady=8)

    # ------------------------------------------------------------------ #
    #  Model / image loading                                               #
    # ------------------------------------------------------------------ #

    def on_model_select(self, model_name: str):
        self.sidebar.layer_dropdown.set("Loading...")
        self.update()
        try:
            model, layer_names = load_model(model_name)
            conv_dict = {name: model.get_submodule(name) for name in layer_names}
            self.hook_engine = HookEngine(model, conv_dict)
            self.sidebar.layer_dropdown.configure(values=layer_names)
            if layer_names:
                self.sidebar.layer_dropdown.set(layer_names[0])
            self.current_model = model_name
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.sidebar.layer_dropdown.set("Error loading")

    def on_upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")],
        )
        if file_path:
            self.image_path = file_path
            self.sidebar.img_path_label.configure(text=os.path.basename(file_path))

    # ------------------------------------------------------------------ #
    #  Core visualization pipeline                                         #
    # ------------------------------------------------------------------ #

    def _run_visualization_pipeline(
        self,
        frame_or_path,
        source_text: str,
        cached_redundancy=None,
        frame_count: int = 0,
    ) -> tuple:
        """
        Runs inference and returns:
            (images, title, cached_redundancy, health_metrics,
             layer_flow_data, sim_matrix)
        """
        cfg = self.sidebar.config
        mode = cfg["mode"]
        images = []
        dynamic_title = ""
        layer_flow_data: dict = {}

        selected_layer = cfg["layer"]

        # ── Grad-CAM ──────────────────────────────────────────────────────
        if cfg["gradcam"]:
            if not selected_layer or selected_layer == "Load model first...":
                raise ValueError("Please select a valid Conv Layer for Grad-CAM.")

            frame = cv2.imread(frame_or_path) if isinstance(frame_or_path, str) else frame_or_path
            pil_orig, gradcam_heatmap, layer_flow_data = compute_gradcam(
                self.hook_engine.model,
                self.hook_engine.conv_layers,
                self.hook_engine.transform,
                frame,
                selected_layer,
            )

            if gradcam_heatmap is not None and pil_orig is not None:
                hm_resized = cv2.resize(gradcam_heatmap, (pil_orig.width, pil_orig.height))
                hm_color = cv2.cvtColor(
                    cv2.applyColorMap((hm_resized * 255).astype(np.uint8), cv2.COLORMAP_JET),
                    cv2.COLOR_BGR2RGB,
                )
                overlay = cv2.addWeighted(np.array(pil_orig), 0.4, hm_color, 0.6, 0)
                images.append((Image.fromarray(overlay), f"Grad-CAM via {selected_layer}", "normal", -1))
                dynamic_title = f"{source_text} Class Saliency (Targeting: {selected_layer})"
                self._update_telemetry(selected_layer, None, None, None)

            return images, dynamic_title, cached_redundancy, None, layer_flow_data, None

        # ── Layer Mode ────────────────────────────────────────────────────
        if mode == "Layer Mode":
            if not selected_layer or selected_layer == "Load model first...":
                raise ValueError("Please select a valid Conv Layer.")

            if isinstance(frame_or_path, str):
                tensor, input_stats, layer_flow_data = self.hook_engine.extract_features(
                    frame_or_path, selected_layer
                )
            else:
                tensor, input_stats, layer_flow_data = self.hook_engine.extract_features_from_frame(
                    frame_or_path, selected_layer
                )

            if tensor is None:
                raise ValueError(f"Failed to retrieve activation for {selected_layer}.")

            # FIX #4: persist the tensor so _update_inspector() can access
            # it after hook_engine.activation has been cleared.
            self._last_tensor = tensor  # shape (1, C, H, W)

            feature_shape = tensor.shape

            images, self.ema_channel_sums, health_metrics, cached_redundancy, sim_matrix = (
                process_tensor_to_images(
                    tensor,
                    max_channels=36,
                    is_live=(source_text == "Live Feed"),
                    use_heatmap=cfg["heatmap"],
                    ema_sums=self.ema_channel_sums,
                    dead_threshold=cfg["dead_threshold"],
                    cached_redundancy=cached_redundancy,
                    frame_count=frame_count,
                )
            )

            # FIX #7: assign _last_sorted_indices so inspector EMA rank works.
            if self.ema_channel_sums is not None:
                self._last_sorted_indices = torch.argsort(self.ema_channel_sums, descending=True)
                self._corr_matrix_sorted_indices = self._last_sorted_indices

            self._update_telemetry(selected_layer, feature_shape, input_stats, health_metrics)
            dynamic_title = (
                f"{source_text} Feature Maps: '{selected_layer}' "
                f"(First {len(images)} channels sorted)"
            )
            return images, dynamic_title, cached_redundancy, health_metrics, layer_flow_data, sim_matrix

        # ── Channel Mode ──────────────────────────────────────────────────
        elif mode == "Channel Mode":
            try:
                target_idx = int(cfg["channel"])
            except ValueError:
                raise ValueError("Channel Index must be a valid number.")

            self.ema_channel_sums = None
            all_layers = list(self.hook_engine.conv_layers.keys())
            tensor_dict: dict = {}
            input_stats = None

            if isinstance(frame_or_path, str):
                result = self.hook_engine.extract_features(frame_or_path, all_layers)
            else:
                result = self.hook_engine.extract_features_from_frame(frame_or_path, all_layers)

            if isinstance(result, tuple) and len(result) == 3:
                tensor_dict, input_stats, layer_flow_data = result

            if tensor_dict:
                for target_layer in all_layers:
                    t = tensor_dict.get(target_layer)
                    if t is None:
                        continue
                    if len(t.shape) == 4:
                        t = t.squeeze(0)
                    if target_idx < t.shape[0]:
                        slice_tensor = t[target_idx].unsqueeze(0).unsqueeze(0)
                        processed, _, _, _, _ = process_tensor_to_images(
                            slice_tensor, max_channels=1, is_live=False,
                            use_heatmap=cfg["heatmap"],
                            cached_redundancy=cached_redundancy,
                            frame_count=frame_count,
                        )
                        if processed:
                            img_obj, _, diag, _ = processed[0]
                            images.append((img_obj, target_layer, diag, target_idx))

            self._update_telemetry(None, None, input_stats, None)
            dynamic_title = (
                f"{source_text} Feature Maps: Channel {target_idx} "
                f"across Network ({len(images)} layers found)"
            )
            return images, dynamic_title, cached_redundancy, None, layer_flow_data, None

        return images, dynamic_title, cached_redundancy, None, layer_flow_data, None

    # ------------------------------------------------------------------ #
    #  Telemetry & rendering                                               #
    # ------------------------------------------------------------------ #

    def _update_telemetry(
        self,
        layer_name: str | None,
        tensor_shape,
        input_stats: dict | None,
        health_metrics: dict | None,
    ):
        self.last_layer_name    = layer_name
        self.last_health_metrics = health_metrics
        self.last_input_stats   = input_stats

        pil_img = self._render_health_strip(health_metrics, input_stats)
        ctk_img = ctk.CTkImage(
            light_image=pil_img, dark_image=pil_img,
            size=(pil_img.width, pil_img.height),
        )
        self.health_strip_label.configure(image=ctk_img)

        if self._inspector_visible and self._inspected_channel is not None:
            self._update_inspector()

    def _render_histogram(
        self,
        tensor: torch.Tensor,
        canvas_w: int = 800,
        canvas_h: int = 140,
    ) -> Image.Image:
        S = SSAA
        W, H = canvas_w * S, canvas_h * S

        canvas = Image.new("RGBA", (W, H), (*_hex_to_rgb(C_BG_BASE), 255))
        draw = ImageDraw.Draw(canvas)

        ML, MR, MT, MB = 48 * S, 24 * S, 20 * S, 32 * S
        plot_w = W - ML - MR
        plot_h = H - MT - MB

        # FIX #5: accept any tensor shape (not just 4D).
        # For 1D inputs (from the inspector's per-channel histogram path),
        # skip the old shape guard and just ravel.
        if tensor is None or tensor.numel() == 0:
            _get_pil_font(12 * S)
            draw.text(
                (W // 2, H // 2), "No activation data",
                fill=_hex_to_rgb(C_TEXT_MUT),
                font=_get_pil_font(12 * S), anchor="mm",
            )
            return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

        flat = tensor.detach().cpu().numpy().ravel().astype(np.float32)

        NUM_BINS = 64
        counts, bin_edges = np.histogram(flat, bins=NUM_BINS)
        smooth_counts = np.maximum(_smooth_signal(counts.astype(np.float32), sigma=1.5), 0)
        max_count = smooth_counts.max() if smooth_counts.max() > 0 else 1.0

        grid_colour = (*_hex_to_rgb(C_BORDER_SUB), 120)
        for frac in (0.25, 0.5, 0.75, 1.0):
            y = MT + plot_h - int(frac * plot_h)
            for x in range(ML, W - MR, 10 * S):
                draw.line([(x, y), (x + 5 * S, y)], fill=grid_colour, width=1)

        xs = [ML + int(i / NUM_BINS * plot_w) for i in range(NUM_BINS)]
        xs.append(ML + plot_w)

        def count_to_y(c):
            return MT + plot_h - int((c / max_count) * plot_h * 0.92)

        ys = [count_to_y(c) for c in smooth_counts]
        poly_points = (
            [(ML, MT + plot_h)]
            + list(zip(xs[:-1], ys))
            + [(ML + plot_w, MT + plot_h)]
        )

        accent_rgb = _hex_to_rgb(C_ACCENT)
        grad = _make_gradient_rect(
            plot_w, plot_h,
            colour_top=accent_rgb, colour_bottom=_hex_to_rgb(C_BG_BASE),
            alpha_top=200, alpha_bottom=0,
        )
        mask_img = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask_img).polygon(poly_points, fill=255)
        canvas.paste(
            grad.convert("RGB"), (ML, MT),
            mask=mask_img.crop((ML, MT, ML + plot_w, MT + plot_h)),
        )

        curve_points = list(zip(xs[:-1], ys))
        glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(glow_layer).line(
            curve_points, fill=(*accent_rgb, 140), width=6 * S
        )
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=4 * S))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), glow_layer)
        ImageDraw.Draw(canvas).line(curve_points, fill=(*accent_rgb, 255), width=2 * S)

        data_range = bin_edges[-1] - bin_edges[0]
        if data_range > 1e-6 and bin_edges[0] < 0 < bin_edges[-1]:
            zero_frac = (-bin_edges[0]) / data_range
            zero_x = ML + int(zero_frac * plot_w)
            zero_rgb = _hex_to_rgb(C_WARNING)
            draw = ImageDraw.Draw(canvas)
            for y_step in range(plot_h):
                alpha = int(180 * (1.0 - y_step / plot_h))
                draw.point((zero_x, MT + y_step), fill=(*zero_rgb, alpha))

        mean_val = float(flat.mean())
        std_val  = float(flat.std())
        zero_pct = float((flat == 0).mean() * 100)
        font_sm  = _get_pil_font(9 * S)
        draw = ImageDraw.Draw(canvas)
        text_y = MT + plot_h + 6 * S
        for tx, label, col in [
            (ML,                 f"μ={mean_val:.3f}",       C_TEXT_PRI),
            (ML + plot_w // 3,   f"σ={std_val:.3f}",        C_TEXT_SEC),
            (ML + 2 * plot_w // 3, f"zeros={zero_pct:.1f}%", C_TEXT_SEC),
        ]:
            draw.text((tx, text_y), label, fill=_hex_to_rgb(col), font=font_sm)

        return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

    def _render_metric_bar(
        self,
        value: float,
        label: str,
        colour: tuple,
        canvas_w: int = 220,
        canvas_h: int = 28,
    ) -> Image.Image:
        S = SSAA
        W, H = canvas_w * S, canvas_h * S
        canvas = Image.new("RGBA", (W, H), (*_hex_to_rgb(C_BG_BASE), 0))
        draw = ImageDraw.Draw(canvas)

        PILL_W = int(W * 0.45)
        PILL_H = H - 8 * S
        PILL_X, PILL_Y = 0, 4 * S
        radius = PILL_H // 2

        _draw_rounded_rect(
            draw, (PILL_X, PILL_Y, PILL_X + PILL_W, PILL_Y + PILL_H),
            radius=radius, fill=_hex_to_rgb(C_BG_FLOAT),
            outline=_hex_to_rgb(C_BORDER_SUB), width=1,
        )

        fill_w = max(int(PILL_W * value), radius * 2) if value > 0 else 0
        if fill_w > 0:
            darker = _lerp_colour(colour, _hex_to_rgb(C_BG_DEEP), 0.4)
            grad = _make_gradient_rect(
                fill_w, PILL_H,
                colour_top=colour, colour_bottom=darker,
                alpha_top=255, alpha_bottom=220,
            )
            fill_mask = Image.new("L", (PILL_W, PILL_H), 0)
            _draw_rounded_rect(
                ImageDraw.Draw(fill_mask),
                (0, 0, PILL_W - 1, PILL_H - 1), radius=radius, fill=255,
            )
            canvas.paste(
                grad.convert("RGB"), (PILL_X, PILL_Y),
                mask=fill_mask.crop((0, 0, fill_w, PILL_H)),
            )
            edge_x = PILL_X + fill_w
            glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(glow_layer).line(
                [(edge_x, PILL_Y + 2 * S), (edge_x, PILL_Y + PILL_H - 2 * S)],
                fill=(*colour, 200), width=3 * S,
            )
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=4 * S))
            canvas = Image.alpha_composite(canvas, glow_layer)

        ImageDraw.Draw(canvas).text(
            (PILL_X + PILL_W + 8 * S, PILL_Y + PILL_H // 2),
            label, fill=_hex_to_rgb(C_TEXT_PRI),
            font=_get_pil_font(9 * S), anchor="lm",
        )
        return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

    def _render_snr_bar(self, snr_info: dict) -> Image.Image:
        value = snr_info.get("normalized", 0.0)
        snr   = snr_info.get("snr", 0.0)
        level = snr_info.get("level", "weak")
        colour = _hex_to_rgb({"weak": C_CRITICAL, "moderate": C_WARNING, "strong": C_HEALTHY}.get(level, C_TEXT_MUT))
        return self._render_metric_bar(value, f"SNR {snr:.2f}  [{level}]", colour)

    def _render_diversity_bar(self, diversity_info: dict) -> Image.Image:
        score = diversity_info.get("score", 0.0)
        pct   = diversity_info.get("score_pct", 0)
        level = diversity_info.get("level", "n/a")
        colour = (
            _hex_to_rgb(C_TEXT_MUT) if level == "n/a"
            else _hex_to_rgb({"low": C_CRITICAL, "moderate": C_WARNING, "high": C_HEALTHY}.get(level, C_TEXT_MUT))
        )
        return self._render_metric_bar(score, f"Div {pct}%  [{level}]", colour)

    def _render_health_strip(
        self,
        health_metrics: dict | None,
        input_stats: dict | None,
        canvas_h: int = 52,
    ) -> Image.Image:
        canvas_w = self.main_outer.winfo_width()
        if canvas_w < 10:
            canvas_w = 1100

        S = SSAA
        W, H = canvas_w * S, canvas_h * S
        canvas = Image.new("RGBA", (W, H), (*_hex_to_rgb(C_BG_RAISED), 255))
        draw = ImageDraw.Draw(canvas)
        draw.line([(0, H - 1), (W, H - 1)], fill=_hex_to_rgb(C_BORDER_SUB), width=1)

        font_label = _get_pil_font(8  * S)
        font_value = _get_pil_font(10 * S)

        if not health_metrics:
            draw.text(
                (W // 2, H // 2), "Run a forward pass to populate metrics",
                fill=_hex_to_rgb(C_TEXT_MUT), font=font_label, anchor="mm",
            )
            return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

        dead_pct = health_metrics.get("dead_percent", 0.0)
        sat_info  = health_metrics.get("saturation", {})
        snr_info  = health_metrics.get("snr", {})
        div_info  = health_metrics.get("diversity", {})

        level_colour = {
            "healthy": C_HEALTHY, "mild": C_WARNING, "saturated": C_CRITICAL,
            "weak": C_CRITICAL, "moderate": C_WARNING, "strong": C_HEALTHY,
            "low": C_CRITICAL, "high": C_HEALTHY, "n/a": C_TEXT_MUT,
        }

        def dead_col(pct):
            t = min(pct / 50.0, 1.0)
            return _lerp_colour(_hex_to_rgb(C_HEALTHY), _hex_to_rgb(C_CRITICAL), t)

        badges = [
            ("DEAD ReLU",  f"{dead_pct:.1f}%", dead_col(dead_pct)),
            ("SATURATION", f"{sat_info.get('layer_score', 0):.1%}",
             _hex_to_rgb(level_colour.get(sat_info.get("level", "healthy"), C_TEXT_MUT))),
            ("SNR",        f"{snr_info.get('snr', 0):.2f}",
             _hex_to_rgb(level_colour.get(snr_info.get("level", "weak"), C_TEXT_MUT))),
            ("DIVERSITY",  f"{div_info.get('score_pct', 0)}%",
             _hex_to_rgb(level_colour.get(div_info.get("level", "n/a"), C_TEXT_MUT))),
        ]

        badge_h = H - 12 * S
        badge_y = 6 * S
        badge_w = W // (len(badges) + 1)

        for i, (name, value, dot_colour) in enumerate(badges):
            bx = int((i + 0.5) * W / len(badges)) - badge_w // 2
            _draw_rounded_rect(
                draw, (bx, badge_y, bx + badge_w - 4 * S, badge_y + badge_h),
                radius=badge_h // 2, fill=_hex_to_rgb(C_BG_FLOAT),
                outline=_hex_to_rgb(C_BORDER_SUB), width=1,
            )
            dot_r  = 5 * S
            dot_cx = bx + dot_r * 2 + 4 * S
            dot_cy = badge_y + badge_h // 2

            glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(glow_layer).ellipse(
                [dot_cx - dot_r * 2, dot_cy - dot_r * 2,
                 dot_cx + dot_r * 2, dot_cy + dot_r * 2],
                fill=(*dot_colour, 160),
            )
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=dot_r))
            canvas = Image.alpha_composite(canvas, glow_layer)

            draw = ImageDraw.Draw(canvas)
            draw.ellipse(
                [dot_cx - dot_r, dot_cy - dot_r, dot_cx + dot_r, dot_cy + dot_r],
                fill=(*dot_colour, 255),
            )
            tx = dot_cx + dot_r + 6 * S
            draw.text((tx, dot_cy - 5 * S), name,  fill=_hex_to_rgb(C_TEXT_MUT), font=font_label, anchor="lm")
            draw.text((tx, dot_cy + 6 * S), value, fill=_hex_to_rgb(C_TEXT_PRI), font=font_value, anchor="lm")

        if input_stats:
            ok = abs(input_stats.get("mean", 0)) <= 0.8 and input_stats.get("std", 1) >= 0.2
            inp_text = "INPUT OK" if ok else "⚠ INPUT NORM"
            inp_col  = C_TEXT_MUT if ok else C_WARNING
            ImageDraw.Draw(canvas).text(
                (W - 16 * S, H // 2), inp_text,
                fill=_hex_to_rgb(inp_col), font=font_label, anchor="rm",
            )

        return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

    def _render_flow_chart(
        self,
        flow_data: dict,
        selected_layer: str | None = None,
        canvas_w: int = 900,
        canvas_h: int = 180,
    ) -> Image.Image:
        S = SSAA
        W, H = canvas_w * S, canvas_h * S
        canvas = Image.new("RGBA", (W, H), (*_hex_to_rgb(C_BG_BASE), 255))
        draw = ImageDraw.Draw(canvas)

        if not flow_data:
            draw.text(
                (W // 2, H // 2), "Run a forward pass to populate flow data",
                fill=_hex_to_rgb(C_TEXT_MUT), font=_get_pil_font(12 * S), anchor="mm",
            )
            return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

        ML, MR, MT, MB = 55 * S, 15 * S, 20 * S, 40 * S
        plot_w = W - ML - MR
        plot_h = H - MT - MB

        layers = list(flow_data.keys())
        N = len(layers)
        max_val = max(flow_data.values()) if flow_data else 1.0
        if max_val < 1e-6:
            max_val = 1.0

        draw.text((10 * S, 15 * S), f"Network Activation Flow  ({N} conv layers)",
                  fill=_hex_to_rgb(C_TEXT_SEC), font=_get_pil_font(10 * S))

        grid_col = (*_hex_to_rgb(C_BORDER_SUB), 140)
        for pct in (0.25, 0.5, 0.75):
            y = MT + int(plot_h * (1 - pct))
            for x in range(ML, W - MR, 10 * S):
                draw.line([(x, y), (x + 5 * S, y)], fill=grid_col, width=1)

        slot_w = plot_w / max(N, 1)
        for i, layer in enumerate(layers):
            val   = flow_data[layer]
            bar_h = max(2 * S, int((val / max_val) * plot_h))
            cx    = ML + int(i * slot_w + slot_w / 2)
            y1    = MT + plot_h - bar_h
            y2    = MT + plot_h

            if i < N * 0.33:   color = _hex_to_rgb(C_HEALTHY)
            elif i < N * 0.66: color = _hex_to_rgb(C_INFO)
            else:               color = _hex_to_rgb(C_WARNING)

            draw.line([(cx, y1), (cx, y2)],
                      fill=(*_hex_to_rgb(C_BORDER_SUB), 200), width=2 * S)

            dot_r = 4 * S
            glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(glow_layer).ellipse(
                [cx - dot_r * 2, y1 - dot_r * 2, cx + dot_r * 2, y1 + dot_r * 2],
                fill=(*color, 120),
            )
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=dot_r))
            canvas = Image.alpha_composite(canvas, glow_layer)

            draw = ImageDraw.Draw(canvas)
            draw.ellipse(
                [cx - dot_r, y1 - dot_r, cx + dot_r, y1 + dot_r],
                fill=(*color, 255),
            )

            if layer == selected_layer:
                sel_r = 6 * S
                draw.ellipse([cx - sel_r, y1 - sel_r, cx + sel_r, y1 + sel_r],
                              outline=(255, 255, 255), width=2 * S)
                draw.polygon([(cx, y1 - 12 * S), (cx - 6 * S, y1 - 20 * S),
                               (cx + 6 * S, y1 - 20 * S)], fill=(255, 255, 255))

            lbl = layer if len(layer) <= 10 else layer[:9] + "…"
            text_img = Image.new("RGBA", (80 * S, 20 * S), (0, 0, 0, 0))
            ImageDraw.Draw(text_img).text(
                (0, 10 * S), lbl, font=_get_pil_font(8 * S),
                fill=_hex_to_rgb(C_TEXT_MUT), anchor="lm",
            )
            text_img = text_img.rotate(45, resample=Image.BICUBIC, expand=True)
            canvas.paste(text_img, (cx - 10 * S, y2 + 5 * S), text_img)

        return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

    def _render_correlation_matrix(
        self,
        sim_matrix: torch.Tensor | None,
        layer_name: str,
        sorted_indices: torch.Tensor | None,
        frame_age_seconds: float,
        canvas_w: int = 400,
        canvas_h: int = 400,
    ) -> Image.Image:
        S = SSAA
        W, H = canvas_w * S, canvas_h * S
        canvas = Image.new("RGBA", (W, H), (*_hex_to_rgb(C_BG_BASE), 255))
        draw = ImageDraw.Draw(canvas)
        font = _get_pil_font(10 * S)

        if sim_matrix is None:
            draw.text((W // 2, H // 2), "No similarity data available.",
                      fill=_hex_to_rgb(C_TEXT_MUT), font=font, anchor="mm")
            return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

        channels = sim_matrix.shape[0]
        max_display = 120
        display_sim = sim_matrix

        if sorted_indices is not None and len(sorted_indices) == channels:
            sort_idx = sorted_indices
            if channels > max_display:
                sort_idx = sort_idx[:max_display]
            display_sim = sim_matrix[sort_idx][:, sort_idx]
        elif channels > max_display:
            step = channels // max_display
            idx = torch.arange(0, channels, step)[:max_display]
            display_sim = sim_matrix[idx][:, idx]

        sim_np   = display_sim.cpu().numpy()
        sim_norm = (sim_np + 1.0) / 2.0

        turbo_anchors = [
            (0.0, "#30123B"), (0.1, "#4662D7"), (0.3, "#36AAF9"),
            (0.5, "#1AE4B6"), (0.7, "#72FE5E"), (0.8, "#C6ED34"),
            (0.9, "#FABA39"), (0.95, "#F66C19"), (1.0, "#7A0403"),
        ]
        heatmap_rgb = _apply_colormap(sim_norm, turbo_anchors)
        heatmap_img = Image.fromarray(heatmap_rgb, mode="RGB")

        PAD = 40 * S
        plot_size = min(W - 2 * PAD, H - 2 * PAD)
        heatmap_scaled = heatmap_img.resize((plot_size, plot_size), Image.NEAREST)

        sx = (W - plot_size) // 2
        sy = (H - plot_size) // 2 + 10 * S
        canvas.paste(heatmap_scaled, (sx, sy))

        _draw_rounded_rect(draw, (sx - 1, sy - 1, sx + plot_size, sy + plot_size),
                            radius=0, outline=_hex_to_rgb(C_BORDER_SUB), width=1)
        draw.text((sx, sy - 15 * S), f"Filter Correlation: {layer_name}",
                  fill=_hex_to_rgb(C_TEXT_PRI), font=font)
        if frame_age_seconds > 0:
            draw.text((sx + plot_size, sy - 15 * S), f"Age: {frame_age_seconds:.1f}s",
                      fill=_hex_to_rgb(C_TEXT_MUT), font=font, anchor="ra")

        return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

    def _render_spatial_bias(
        self,
        tensor: torch.Tensor,
        canvas_w: int = 280,
        canvas_h: int = 140,
    ) -> Image.Image | None:
        if tensor is None or len(tensor.shape) < 2:
            return None

        t_2d = tensor.detach().cpu().numpy()
        h_t, w_t = t_2d.shape
        if h_t == 0 or w_t == 0:
            return None

        S = SSAA
        W, H = canvas_w * S, canvas_h * S
        canvas = Image.new("RGBA", (W, H), (*_hex_to_rgb(C_BG_RAISED), 255))

        try:
            t_min, t_max = t_2d.min(), t_2d.max()
            t_norm = (t_2d - t_min) / (t_max - t_min) if t_max > t_min else np.zeros_like(t_2d)

            magma_anchors = [
                (0.0, "#000004"), (0.2, "#2b115f"), (0.4, "#721f81"),
                (0.6, "#c33d69"), (0.8, "#fca50a"), (1.0, "#fcfdbf"),
            ]
            heat_rgb = _apply_colormap(t_norm, magma_anchors)
            heat_img = Image.fromarray(heat_rgb, mode="RGB")
            heat_scaled = heat_img.resize((W, H), Image.BILINEAR)
            heat_layer = heat_scaled.copy().convert("RGBA")
            heat_layer.putalpha(150)
            canvas = Image.alpha_composite(canvas, heat_layer)

            y_i, x_i = np.unravel_index(np.argmax(t_norm), t_norm.shape)
            x_max = int((x_i + 0.5) / w_t * W)
            y_max = int((y_i + 0.5) / h_t * H)

            draw = ImageDraw.Draw(canvas)
            cc = (255, 255, 255, 180)
            draw.line([(x_max, 0), (x_max, H)], fill=cc, width=1 * S)
            draw.line([(0, y_max), (W, y_max)], fill=cc, width=1 * S)
            rad = 3 * S
            draw.ellipse([x_max - rad, y_max - rad, x_max + rad, y_max + rad],
                          fill=(255, 255, 0, 255), outline=(0, 0, 0, 100), width=1)

            return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Tab management                                                      #
    # ------------------------------------------------------------------ #

    def _on_tab_change(self):
        tab = self.tabview.get()
        if tab == "Flow Chart":
            self._refresh_flow_tab()
        elif tab == "Correlation":
            self._refresh_correlation_tab()

    def _refresh_flow_tab(self):
        if not self._last_flow_data:
            return
        self._flow_frame_count += 1
        is_live = self.camera_active and self.frozen_frame is None
        if not is_live or self._flow_frame_count % 3 == 0:
            pil_img = self._render_flow_chart(
                self._last_flow_data, self.sidebar.config.get("layer")
            )
            tw = self.tabview.winfo_width()
            th = self.tabview.winfo_height()
            tw = tw if tw > 10 else 900
            th = th if th > 10 else 180
            scaled_h = min(th - 40, int(pil_img.height * (tw / max(pil_img.width, 1))))
            if scaled_h < 10:
                scaled_h = 180
            pil_r = pil_img.resize((tw - 20, scaled_h), Image.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=pil_r, dark_image=pil_r, size=(tw - 20, scaled_h))
            self.flow_tab_label.configure(image=ctk_img)

    def _refresh_correlation_tab(self):
        if self._last_sim_matrix is None:
            return
        self._corr_matrix_frame_count += 1
        is_live = self.camera_active and self.frozen_frame is None
        do_render = not is_live or self._last_sim_matrix.shape[0] <= 64 or self._corr_matrix_frame_count % 10 == 0
        if do_render:
            age = time.time() - self._corr_matrix_last_update_time
            pil_img = self._render_correlation_matrix(
                self._last_sim_matrix,
                self.sidebar.config.get("layer", ""),
                self._corr_matrix_sorted_indices,
                age if is_live else 0.0,
            )
            tw = self.tabview.winfo_width()
            th = self.tabview.winfo_height()
            plot_size = min(tw - 20, th - 40)
            if plot_size < 10:
                plot_size = 400
            pil_r = pil_img.resize((plot_size, plot_size), Image.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=pil_r, dark_image=pil_r, size=(plot_size, plot_size))
            self.corr_tab_label.configure(image=ctk_img)

    # ------------------------------------------------------------------ #
    #  Export                                                              #
    # ------------------------------------------------------------------ #

    def on_export_diagnostics(self):
        # FIX #6: original code referenced self.sidebar.telemetry_text which
        # was removed in the UI refactor. Use messagebox instead.
        if not self.last_layer_name and not self.last_health_metrics:
            messagebox.showwarning("Export", "No active data to export. Run a forward pass first.")
            return

        try:
            content = generate_report(
                self.last_layer_name,
                self.last_health_metrics,
                self.last_input_stats,
                self.sidebar.config["dead_threshold"],
            )
            save_report(content)
            old_text = self.sidebar.export_btn.cget("text")
            self.sidebar.export_btn.configure(text="✔ Saved!")
            self.after(2000, lambda: self.sidebar.export_btn.configure(text=old_text))
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    # ------------------------------------------------------------------ #
    #  Static visualization                                                #
    # ------------------------------------------------------------------ #

    def on_visualize(self):
        if self.camera_active:
            self.toggle_camera()
        if not self.current_model:
            messagebox.showwarning("Warning", "Please select a model first.")
            return
        if not self.image_path:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return

        loading = ctk.CTkLabel(self.grid_view, text="Processing forward pass…",
                                font=ctk.CTkFont(size=16))
        loading.pack(pady=50)
        self.update()

        try:
            images, title, _, health_metrics, flow_data, sim_matrix = (
                self._run_visualization_pipeline(self.image_path, "Static Upload")
            )
            loading.destroy()

            layer_name = self.sidebar.config.get("layer", "")
            self.grid_view.update(
                images, title,
                force_rebuild=True,
                health_metrics=health_metrics,
                layer_name=layer_name,
            )

            if flow_data:
                self._last_flow_data = flow_data
            if sim_matrix is not None:
                self._last_sim_matrix = sim_matrix
                self._corr_matrix_last_update_time = time.time()

            self._on_tab_change()
        except Exception as e:
            loading.destroy()
            messagebox.showerror("Error", f"Visualization failed:\n{e}")

    # ------------------------------------------------------------------ #
    #  Camera threading                                                    #
    # ------------------------------------------------------------------ #

    def _capture_thread_func(self):
        while not self._inference_stop_event.is_set():
            if self.camera is not None and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    with self._frame_lock:
                        self._latest_frame = cv2.flip(frame, 1)
            # Small sleep to avoid busy-spinning when camera is slow
            time.sleep(0.005)

    def _inference_thread_func(self):
        while not self._inference_stop_event.is_set():
            # FIX #11: sleep to prevent 100% CPU spin when queue is full
            # or no new frames are available.
            time.sleep(0.01)

            frame = None
            if self.frozen_frame is not None:
                frame = self.frozen_frame
            else:
                with self._frame_lock:
                    if self._latest_frame is not None:
                        frame = self._latest_frame.copy()

            if frame is None:
                continue

            try:
                source = "Frozen Frame" if self.frozen_frame is not None else "Live Feed"
                images, title, cached_redundancy, health_metrics, flow_data, sim_matrix = (
                    self._run_visualization_pipeline(
                        frame, source,
                        cached_redundancy=self._cached_redundancy,
                        frame_count=self._frame_count,
                    )
                )
                self._cached_redundancy = cached_redundancy
                self._frame_count += 1

                try:
                    self._result_queue.put_nowait(
                        (images, title, cached_redundancy, health_metrics, flow_data, sim_matrix)
                    )
                except queue.Full:
                    pass
            except Exception:
                pass

    def _poll_results(self):
        if not self.camera_active:
            return

        try:
            images, title, _, health_metrics, flow_data, sim_matrix = (
                self._result_queue.get_nowait()
            )
            layer_name = self.sidebar.config.get("layer", "")
            self.grid_view.update(
                images, title,
                force_rebuild=False,
                health_metrics=health_metrics,
                layer_name=layer_name,
            )

            if flow_data:
                self._last_flow_data = flow_data
            if sim_matrix is not None:
                self._last_sim_matrix = sim_matrix
                self._corr_matrix_last_update_time = time.time()

            self._on_tab_change()

            # Update inspector if it is pinned open
            if self._inspector_visible and self._inspected_channel is not None:
                if (self._last_tensor is not None
                        and self._inspected_channel < self._last_tensor.shape[1]):
                    self._update_inspector()
                else:
                    self._close_inspector()

        except queue.Empty:
            pass

        self.after(int(self.sidebar.config["speed"]), self._poll_results)

    def toggle_camera(self):
        if not self.current_model:
            messagebox.showwarning("Warning", "Please select a model first.")
            return
        if not self.sidebar.config["layer"] or self.sidebar.config["layer"] == "Load model first...":
            messagebox.showwarning("Warning", "Please select a valid Conv Layer.")
            return

        self.camera_active = not self.camera_active

        if self.camera_active:
            if not isinstance(self.camera, cv2.VideoCapture) or not self.camera.isOpened():
                try:
                    self.camera = cv2.VideoCapture(0)
                    if not self.camera.isOpened():
                        raise Exception("Could not open video device.")
                except Exception as e:
                    self.camera_active = False
                    messagebox.showerror("Camera Error", str(e))
                    return

            self.sidebar.camera_btn.configure(text="Stop Live Camera", fg_color="#c92a2a", hover_color="#a61e1e")
            self.sidebar.cam_status_label.configure(text="Camera Active", text_color="green")
            self.sidebar.freeze_btn.configure(state="normal")
            self.sidebar.status_badge.configure(text="● LIVE", text_color=C_HEALTHY)

            self.frozen_frame = None
            self._frame_count = 0
            self._cached_redundancy = None
            self._latest_frame = None

            while not self._result_queue.empty():
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    break

            self._inference_stop_event.clear()
            self._capture_thread = threading.Thread(target=self._capture_thread_func, daemon=True)
            self._inference_thread = threading.Thread(target=self._inference_thread_func, daemon=True)
            self._capture_thread.start()
            self._inference_thread.start()
            self._poll_results()

        else:
            self.sidebar.camera_btn.configure(text="Start Live Camera", fg_color=C_HEALTHY, hover_color="#22c55e")
            self.sidebar.cam_status_label.configure(text="Camera Offline", text_color=C_TEXT_MUT)
            self.sidebar.freeze_btn.configure(state="disabled")
            self.sidebar.status_badge.configure(text="● IDLE", text_color=C_TEXT_MUT)

            self._inference_stop_event.set()
            if self._capture_thread:
                self._capture_thread.join(timeout=1.0)
            if self._inference_thread:
                self._inference_thread.join(timeout=1.0)

            self.frozen_frame = None
            if self.camera:
                self.camera.release()
                self.camera = None

    def toggle_freeze(self):
        if self.frozen_frame is None:
            self.sidebar.cam_status_label.configure(text="Camera [FROZEN]", text_color=C_ACCENT)
            self.sidebar.freeze_btn.configure(text="▶ Unfreeze", fg_color="#10b981", hover_color="#059669")
            self.sidebar.status_badge.configure(text="● FROZEN", text_color=C_ACCENT)
            with self._frame_lock:
                if self._latest_frame is not None:
                    self.frozen_frame = self._latest_frame.copy()
        else:
            self.frozen_frame = None
            self.sidebar.cam_status_label.configure(text="Camera Active", text_color="green")
            self.sidebar.freeze_btn.configure(text="Freeze", fg_color=C_ACCENT, hover_color=C_BG_FLOAT)
            self.sidebar.status_badge.configure(text="● LIVE", text_color=C_HEALTHY)

    # ------------------------------------------------------------------ #
    #  Inspector panel                                                     #
    # ------------------------------------------------------------------ #

    def _on_channel_click(self, channel_idx: int, layer_name: str | None = None):
        # FIX #10: original signature only accepted channel_idx but
        # grid_view calls self.on_channel_click(ch_idx, layer) — two args.
        self._inspected_channel = channel_idx
        self._inspected_layer   = layer_name or self.sidebar.config.get("layer")

        if not self._inspector_visible:
            self.inspector_frame.grid(row=0, column=2, sticky="nsew", padx=(0, 10), pady=10)
            self._inspector_visible = True

        self._update_inspector()
        self.grid_view.refresh_borders(channel_idx)

    def _close_inspector(self):
        self._inspector_visible  = False
        self._inspected_channel  = None
        self._inspected_layer    = None
        self.inspector_frame.grid_forget()
        # FIX #9: clear_selection() now exists in GridView
        self.grid_view.clear_selection()

    def _update_inspector(self):
        """Populate inspector widgets for the currently inspected channel."""
        ch    = self._inspected_channel
        layer = self._inspected_layer

        if ch is None:
            return

        self.inspector_title.configure(text=f"Channel {ch}  ·  {layer or '—'}")

        # FIX #4: read tensor from self._last_tensor (persisted after each
        # pipeline run) instead of self.hook_engine.features which doesn't exist.
        if self._last_tensor is None:
            return

        tensor = self._last_tensor  # (1, C, H, W)
        if ch >= tensor.shape[1]:
            return

        ch_map = tensor[0, ch]  # (H, W)

        # ── Large feature map ─────────────────────────────────────────────
        ch_np = ch_map.detach().cpu().numpy()
        c_min, c_max = ch_np.min(), ch_np.max()
        if c_max - c_min > 1e-6:
            ch_norm = ((ch_np - c_min) / (c_max - c_min) * 255).astype(np.uint8)
        else:
            ch_norm = np.zeros_like(ch_np, dtype=np.uint8)

        ch_large = cv2.resize(ch_norm, (280, 280), interpolation=cv2.INTER_LANCZOS4)
        ch_color = cv2.cvtColor(cv2.applyColorMap(ch_large, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
        pil_map  = Image.fromarray(ch_color)
        ctk_map  = ctk.CTkImage(light_image=pil_map, dark_image=pil_map, size=(280, 280))
        self.inspector_map_label.configure(image=ctk_map)

        # ── Stats ─────────────────────────────────────────────────────────
        hm       = self.last_health_metrics or {}
        act_mean = ch_map.mean().item()
        act_std  = ch_map.std().item()
        act_min  = ch_map.min().item()
        act_max  = ch_map.max().item()

        dead_flag = "YES" if ch_map.var().item() < self.sidebar.health_slider.get() else "no"

        sat_val = "N/A"
        sat_info = hm.get("saturation", {})
        if sat_info.get("per_channel") is not None and ch < len(sat_info["per_channel"]):
            sat_val = f"{sat_info['per_channel'][ch].item() * 100:.1f}%"

        ema_rank = "N/A"
        if self._last_sorted_indices is not None:
            idx_list = self._last_sorted_indices.tolist()
            if ch in idx_list:
                ema_rank = f"#{idx_list.index(ch) + 1} of {len(idx_list)}"

        sim_str = "N/A"
        if self._last_sim_matrix is not None and ch < self._last_sim_matrix.shape[0]:
            row = self._last_sim_matrix[ch].clone()
            row[ch] = 0.0
            max_v, max_i = row.max(dim=0)
            sim_str = f"{max_v.item() * 100:.1f}%  → Ch {max_i.item()}"

        for key, val in {
            "Layer":      layer or "—",
            "Channel":    str(ch),
            "EMA Rank":   ema_rank,
            "Act Range":  f"{act_min:.3f} → {act_max:.3f}",
            "Mean / Std": f"{act_mean:.3f}  /  {act_std:.3f}",
            "Saturation": sat_val,
            "Dead":       dead_flag,
        }.items():
            if key in self.inspector_stats:
                color = C_CRITICAL if key == "Dead" and val == "YES" else C_TEXT_PRI
                self.inspector_stats[key].configure(text=val, text_color=color)

        # ── Per-channel histogram ─────────────────────────────────────────
        # FIX #5: pass the 2D channel tensor — _render_histogram now accepts
        # any shape and ravels internally, no longer requires (1, C, H, W).
        hist_img = self._render_histogram(ch_map, canvas_w=280, canvas_h=100)
        ctk_hist = ctk.CTkImage(light_image=hist_img, dark_image=hist_img, size=(280, 100))
        self.inspector_hist_label.configure(image=ctk_hist)

        # ── Similar channels ──────────────────────────────────────────────
        if self._last_sim_matrix is not None and ch < self._last_sim_matrix.shape[0]:
            sim_row = self._last_sim_matrix[ch].clone()
            sim_row[ch] = 0.0
            top3_vals, top3_idxs = torch.topk(sim_row, min(3, sim_row.shape[0]))

            for slot, (sim_ch, sim_val) in enumerate(zip(top3_idxs.tolist(), top3_vals.tolist())):
                if slot >= len(self.inspector_similar_labels):
                    break
                img_lbl, txt_lbl = self.inspector_similar_labels[slot]

                cell = self.grid_view.get_cell_data(sim_ch)
                if cell and cell[0]:
                    s_im = cell[0].resize((70, 70), Image.NEAREST)
                    c_im = ctk.CTkImage(light_image=s_im, dark_image=s_im, size=(70, 70))
                    img_lbl.configure(image=c_im)
                    txt_lbl.configure(text=f"Ch {sim_ch}\n{sim_val:.2f}")
                else:
                    img_lbl.configure(image="")
                    txt_lbl.configure(text="-")

        # ── Spatial bias ──────────────────────────────────────────────────
        bias_img = self._render_spatial_bias(ch_map)
        if bias_img:
            ctk_bias = ctk.CTkImage(light_image=bias_img, dark_image=bias_img, size=(280, 140))
            self.inspector_bias_label.configure(image=ctk_bias, text="")