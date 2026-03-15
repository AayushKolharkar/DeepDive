"""
Main application window coordinating UI and backend logic.

Performance changes vs previous version:
- _update_telemetry() no longer called from inference thread (was Tkinter
  race condition). All widget updates now happen only in _poll_results().
- Health strip render throttled to every 5 poll cycles (was every frame).
- Tab chart renders only fire when data has actually changed since last render.
- ttk.PanedWindow replaces the fixed 3-column grid — columns are draggable.
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
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageFilter, ImageFont

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


def _lerp_colour(c1, c2, t: float):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _apply_colormap(values: np.ndarray, anchors: list) -> np.ndarray:
    out = np.zeros((*values.shape, 3), dtype=np.uint8)
    flat = values.ravel()
    rgb_anchors = [(pos, _hex_to_rgb(col)) for pos, col in anchors]
    for i, v in enumerate(flat):
        v = float(np.clip(v, 0.0, 1.0))
        lp, lc = rgb_anchors[0]
        rp, rc = rgb_anchors[-1]
        for j in range(len(rgb_anchors) - 1):
            if rgb_anchors[j][0] <= v <= rgb_anchors[j + 1][0]:
                lp, lc = rgb_anchors[j]
                rp, rc = rgb_anchors[j + 1]
                break
        span = rp - lp
        t = (v - lp) / span if span > 1e-9 else 0.0
        out.ravel()[i * 3: i * 3 + 3] = _lerp_colour(lc, rc, t)
    return out.reshape(*values.shape, 3)


def _smooth_signal(values: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values.astype(np.float32), radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _make_gradient_rect(width, height, colour_top, colour_bottom,
                        alpha_top=255, alpha_bottom=255) -> Image.Image:
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


def parse_channel_filter(raw: str) -> set[int] | None:
    """
    Parse a channel filter string into a set of channel indices.
    Supports comma-separated values and ranges:
        "0-5, 10, 15-20"  →  {0,1,2,3,4,5,10,15,16,17,18,19,20}
        ""  or  "all"      →  None  (show everything)
    Invalid tokens are silently ignored.
    """
    raw = raw.strip()
    if not raw or raw.lower() == "all":
        return None
    result: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            try:
                lo, hi = int(parts[0].strip()), int(parts[1].strip())
                result.update(range(lo, hi + 1))
            except ValueError:
                pass
        else:
            try:
                result.add(int(token))
            except ValueError:
                pass
    return result if result else None


# ── Main application ──────────────────────────────────────────────────────────

class CNNVisualizerApp(ctk.CTk):
    """Main GUI window."""

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

        # ── Diagnostics ───────────────────────────────────────────────────
        self.last_health_metrics: dict | None = None
        self.last_input_stats: dict | None = None
        self.last_layer_name: str | None = None

        self._last_tensor_dict: dict[str, torch.Tensor] = {}
        self._last_tensor: torch.Tensor | None = None

        # ── Threading ─────────────────────────────────────────────────────
        self._capture_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._inference_stop_event = threading.Event()
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._result_queue: queue.Queue = queue.Queue(maxsize=1)
        self._frame_count: int = 0
        self._cached_redundancy: torch.Tensor | None = None

        # ── Chart state ───────────────────────────────────────────────────
        self._last_flow_data: dict[str, float] = {}
        self._flow_frame_count: int = 0
        self._last_sim_matrix: torch.Tensor | None = None
        self._corr_matrix_frame_count: int = 0
        self._corr_matrix_last_update_time: float = 0.0
        self._corr_matrix_sorted_indices: torch.Tensor | None = None
        self._last_sorted_indices: torch.Tensor | None = None

        # Track what data was last rendered to each tab so we only re-render
        # when the data actually changes. (PERF FIX #6)
        self._flow_data_rendered:   dict | None = None
        self._sim_matrix_rendered:  torch.Tensor | None = None

        # Health strip throttle counter (PERF FIX #5)
        self._health_strip_poll_count: int = 0

        # ── Inspector ─────────────────────────────────────────────────────
        self._inspected_channel: int | None = None
        self._inspected_layer: str | None = None
        self._inspector_visible: bool = False

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
        """
        Build the root layout using ttk.PanedWindow so all three columns
        (sidebar / main content / inspector) are draggable at runtime.

        PERF FIX: replaced fixed grid_columnconfigure with ttk.PanedWindow.
        Dragging the sash between panes fires <Configure> on GridView which
        triggers an automatic column reflow.
        """
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Root paned window — fills the entire window
        self._paned = ttk.PanedWindow(self, orient="horizontal")
        self._paned.grid(row=0, column=0, sticky="nsew")

        # Style the sash to be visible but minimal
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TPanedwindow",
            background=C_BG_DEEP,
            sashwidth=5,
            sashpad=0,
            sashrelief="flat",
        )

        # ── Pane 1: Sidebar ───────────────────────────────────────────────
        # Wrapped in a plain Frame so the sidebar widget has a parent inside
        # the PanedWindow
        sidebar_pane = ctk.CTkFrame(self._paned, fg_color=C_BG_BASE,
                                     corner_radius=0, width=270)
        sidebar_pane.pack_propagate(False)  # keep the pane at its set width
        self._paned.add(sidebar_pane, weight=0)

        self.sidebar = Sidebar(
            sidebar_pane,
            on_model_select=self.on_model_select,
            on_upload=self.on_upload_image,
            on_camera_toggle=self.toggle_camera,
            on_freeze_toggle=self.toggle_freeze,
            on_visualize=self.on_visualize,
            on_export=self.on_export_diagnostics,
            on_speed_change=None,
        )
        self.sidebar.pack(fill="both", expand=True)

        # ── Pane 2: Main content ──────────────────────────────────────────
        main_pane = ctk.CTkFrame(self._paned, fg_color=C_BG_BASE,
                                  corner_radius=0)
        self._paned.add(main_pane, weight=1)

        main_pane.grid_rowconfigure(0, weight=0)
        main_pane.grid_rowconfigure(1, weight=1)
        main_pane.grid_columnconfigure(0, weight=1)

        self.main_outer = main_pane  # keep reference for health strip width query

        self.health_strip_label = ctk.CTkLabel(main_pane, text="")
        self.health_strip_label.grid(row=0, column=0, sticky="ew")

        self.tabview = ctk.CTkTabview(
            main_pane,
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

        self.flow_tab_label = ctk.CTkLabel(self.tabview.tab("Flow Chart"), text="")
        self.flow_tab_label.pack(fill="both", expand=True)

        self.corr_tab_label = ctk.CTkLabel(self.tabview.tab("Correlation"), text="")
        self.corr_tab_label.pack(fill="both", expand=True)

        # ── Pane 3: Inspector (starts hidden — added to paned window on demand) ─
        # We keep a reference but do NOT add it to the PanedWindow until needed.
        self._inspector_pane_frame = ctk.CTkFrame(
            self._paned, fg_color=C_BG_RAISED, corner_radius=0, width=350
        )
        self._inspector_pane_frame.pack_propagate(False)
        # NOT added to paned window yet

    def _build_inspector_skeleton(self):
        """Creates all inspector widgets once. Content updated dynamically."""
        self.inspector_frame = ctk.CTkScrollableFrame(
            self._inspector_pane_frame,
            fg_color=C_BG_RAISED,
            label_text="",
        )
        self.inspector_frame.pack(fill="both", expand=True)

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

        def _section(title):
            f = ctk.CTkFrame(self.inspector_frame, fg_color=C_BG_BASE,
                             border_width=1, border_color=C_BORDER_SUB, corner_radius=6)
            f.pack(fill="x", pady=6)
            ctk.CTkLabel(f, text=title, font=ctk.CTkFont(size=9, weight="bold"),
                         text_color=C_TEXT_MUT).pack(anchor="w", padx=12, pady=(8, 4))
            return f

        map_f = _section("ACTIVATION MAP")
        self.inspector_map_label = ctk.CTkLabel(map_f, text="")
        self.inspector_map_label.pack(pady=8)

        stats_f = _section("STATISTICS")
        for key in ["Layer", "Channel", "EMA Rank", "Act Range",
                    "Mean / Std", "Saturation", "Dead"]:
            row = ctk.CTkFrame(stats_f, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=2)
            ctk.CTkLabel(row, text=f"{key}:", font=ctk.CTkFont(size=10),
                         text_color=C_TEXT_MUT).pack(side="left")
            vl = ctk.CTkLabel(row, text="-", font=ctk.CTkFont(size=10, weight="bold"),
                               text_color=C_TEXT_PRI)
            vl.pack(side="right")
            self.inspector_stats[key] = vl

        hist_f = _section("ACTIVATION DISTRIBUTION")
        self.inspector_hist_label = ctk.CTkLabel(hist_f, text="")
        self.inspector_hist_label.pack(pady=8)

        sim_f = _section("MOST SIMILAR CHANNELS")
        sim_grid = ctk.CTkFrame(sim_f, fg_color="transparent")
        sim_grid.pack(pady=8)
        for _ in range(3):
            sub = ctk.CTkFrame(sim_grid, fg_color="transparent")
            sub.pack(side="left", padx=4)
            il = ctk.CTkLabel(sub, text=""); il.pack()
            tl = ctk.CTkLabel(sub, text="-", font=ctk.CTkFont(size=10),
                               text_color=C_TEXT_MUT); tl.pack()
            self.inspector_similar_labels.append((il, tl))

        bias_f = _section("SPATIAL FOCUS")
        ctk.CTkLabel(bias_f, text="Where this filter concentrates attention",
                     font=ctk.CTkFont(size=9), text_color=C_TEXT_MUT,
                     ).pack(anchor="w", padx=12, pady=(0, 4))
        self.inspector_bias_label = ctk.CTkLabel(bias_f, text="")
        self.inspector_bias_label.pack(pady=8)

    # ------------------------------------------------------------------ #
    #  Model / image loading                                               #
    # ------------------------------------------------------------------ #

    def on_model_select(self, model_name: str):
        self.sidebar.layer_dropdown.set("Loading...")
        self.update()
        try:
            model, layer_names = load_model(model_name)
            conv_dict = {n: model.get_submodule(n) for n in layer_names}
            self.hook_engine = HookEngine(model, conv_dict)
            self.sidebar.layer_dropdown.configure(values=layer_names)
            if layer_names:
                self.sidebar.layer_dropdown.set(layer_names[0])
            self.current_model = model_name
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.sidebar.layer_dropdown.set("Error loading")

    def on_upload_image(self):
        fp = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")],
        )
        if fp:
            self.image_path = fp
            self.sidebar.img_path_label.configure(text=os.path.basename(fp))

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
        Runs inference. Returns:
            (images, title, cached_redundancy, health_metrics,
             input_stats, layer_flow_data, sim_matrix)

        PERF FIX #7: _update_telemetry() is NO LONGER called here.
        It runs on the main thread inside _poll_results() instead, so
        no Tkinter widgets are touched from the inference thread.
        """
        cfg = self.sidebar.config
        mode = cfg["mode"]
        images = []
        dynamic_title = ""
        layer_flow_data: dict = {}
        input_stats: dict | None = None
        health_metrics: dict | None = None

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
                frame, selected_layer,
            )
            if gradcam_heatmap is not None and pil_orig is not None:
                hm = cv2.resize(gradcam_heatmap, (pil_orig.width, pil_orig.height))
                hm_c = cv2.cvtColor(
                    cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET),
                    cv2.COLOR_BGR2RGB,
                )
                overlay = cv2.addWeighted(np.array(pil_orig), 0.4, hm_c, 0.6, 0)
                images.append((Image.fromarray(overlay),
                                f"Grad-CAM via {selected_layer}", "normal", -1))
                dynamic_title = f"{source_text} Class Saliency (Targeting: {selected_layer})"
            return images, dynamic_title, cached_redundancy, None, None, layer_flow_data, None

        # ── Layer Mode ────────────────────────────────────────────────────
        if mode == "Layer Mode":
            if not selected_layer or selected_layer == "Load model first...":
                raise ValueError("Please select a valid Conv Layer.")
            if isinstance(frame_or_path, str):
                tensor, input_stats, layer_flow_data = self.hook_engine.extract_features(
                    frame_or_path, selected_layer)
            else:
                tensor, input_stats, layer_flow_data = self.hook_engine.extract_features_from_frame(
                    frame_or_path, selected_layer)

            if tensor is None:
                raise ValueError(f"Failed to retrieve activation for {selected_layer}.")

            self._last_tensor = tensor

            images, self.ema_channel_sums, health_metrics, cached_redundancy, sim_matrix = (
                process_tensor_to_images(
                    tensor,
                    max_channels=64,   # show up to 64 — grid fills available width
                    is_live=(source_text == "Live Feed"),
                    use_heatmap=cfg["heatmap"],
                    ema_sums=self.ema_channel_sums,
                    dead_threshold=cfg["dead_threshold"],
                    cached_redundancy=cached_redundancy,
                    frame_count=frame_count,
                )
            )
            if self.ema_channel_sums is not None:
                self._last_sorted_indices = torch.argsort(
                    self.ema_channel_sums, descending=True)
                self._corr_matrix_sorted_indices = self._last_sorted_indices

            dynamic_title = (
                f"{source_text} Feature Maps: '{selected_layer}' "
                f"(First {len(images)} channels sorted)"
            )
            return (images, dynamic_title, cached_redundancy, health_metrics,
                    input_stats, layer_flow_data, sim_matrix)

        # ── Channel Mode ──────────────────────────────────────────────────
        elif mode == "Channel Mode":
            try:
                target_idx = int(cfg["channel"])
            except ValueError:
                raise ValueError("Channel Index must be a valid number.")

            self.ema_channel_sums = None
            all_layers = list(self.hook_engine.conv_layers.keys())
            tensor_dict: dict = {}

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

            dynamic_title = (
                f"{source_text} Feature Maps: Channel {target_idx} "
                f"across Network ({len(images)} layers found)"
            )
            return (images, dynamic_title, cached_redundancy, None,
                    input_stats, layer_flow_data, None)

        return images, dynamic_title, cached_redundancy, None, None, layer_flow_data, None

    # ------------------------------------------------------------------ #
    #  Render methods (main thread only)                                   #
    # ------------------------------------------------------------------ #

    def _render_histogram(self, tensor: torch.Tensor,
                          canvas_w: int = 800, canvas_h: int = 140) -> Image.Image:
        S = SSAA
        W, H = canvas_w * S, canvas_h * S
        canvas = Image.new("RGBA", (W, H), (*_hex_to_rgb(C_BG_BASE), 255))
        draw = ImageDraw.Draw(canvas)
        ML, MR, MT, MB = 48*S, 24*S, 20*S, 32*S
        plot_w = W - ML - MR
        plot_h = H - MT - MB

        if tensor is None or tensor.numel() == 0:
            draw.text((W//2, H//2), "No activation data",
                      fill=_hex_to_rgb(C_TEXT_MUT), font=_get_pil_font(12*S), anchor="mm")
            return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

        flat = tensor.detach().cpu().numpy().ravel().astype(np.float32)
        NUM_BINS = 64
        counts, bin_edges = np.histogram(flat, bins=NUM_BINS)
        smooth_counts = np.maximum(_smooth_signal(counts.astype(np.float32), sigma=1.5), 0)
        max_count = smooth_counts.max() if smooth_counts.max() > 0 else 1.0

        grid_col = (*_hex_to_rgb(C_BORDER_SUB), 120)
        for frac in (0.25, 0.5, 0.75, 1.0):
            y = MT + plot_h - int(frac * plot_h)
            for x in range(ML, W - MR, 10*S):
                draw.line([(x, y), (x+5*S, y)], fill=grid_col, width=1)

        xs = [ML + int(i / NUM_BINS * plot_w) for i in range(NUM_BINS)]
        xs.append(ML + plot_w)
        ys = [MT + plot_h - int((c / max_count) * plot_h * 0.92) for c in smooth_counts]
        poly = [(ML, MT+plot_h)] + list(zip(xs[:-1], ys)) + [(ML+plot_w, MT+plot_h)]

        accent_rgb = _hex_to_rgb(C_ACCENT)
        grad = _make_gradient_rect(plot_w, plot_h, colour_top=accent_rgb,
                                   colour_bottom=_hex_to_rgb(C_BG_BASE), alpha_top=200, alpha_bottom=0)
        mask_img = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask_img).polygon(poly, fill=255)
        canvas.paste(grad.convert("RGB"), (ML, MT),
                     mask=mask_img.crop((ML, MT, ML+plot_w, MT+plot_h)))

        curve_pts = list(zip(xs[:-1], ys))
        glow = Image.new("RGBA", (W, H), (0,0,0,0))
        ImageDraw.Draw(glow).line(curve_pts, fill=(*accent_rgb, 140), width=6*S)
        glow = glow.filter(ImageFilter.GaussianBlur(radius=4*S))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), glow)
        ImageDraw.Draw(canvas).line(curve_pts, fill=(*accent_rgb, 255), width=2*S)

        data_range = bin_edges[-1] - bin_edges[0]
        if data_range > 1e-6 and bin_edges[0] < 0 < bin_edges[-1]:
            zero_x = ML + int((-bin_edges[0]) / data_range * plot_w)
            zero_rgb = _hex_to_rgb(C_WARNING)
            d2 = ImageDraw.Draw(canvas)
            for ys2 in range(plot_h):
                d2.point((zero_x, MT+ys2), fill=(*zero_rgb, int(180*(1-ys2/plot_h))))

        mean_val, std_val = float(flat.mean()), float(flat.std())
        zero_pct = float((flat == 0).mean() * 100)
        font_sm = _get_pil_font(9*S)
        draw2 = ImageDraw.Draw(canvas)
        ty = MT + plot_h + 6*S
        for tx, lbl, col in [
            (ML, f"μ={mean_val:.3f}", C_TEXT_PRI),
            (ML+plot_w//3, f"σ={std_val:.3f}", C_TEXT_SEC),
            (ML+2*plot_w//3, f"zeros={zero_pct:.1f}%", C_TEXT_SEC),
        ]:
            draw2.text((tx, ty), lbl, fill=_hex_to_rgb(col), font=font_sm)

        return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

    def _render_metric_bar(self, value: float, label: str, colour: tuple,
                           canvas_w=220, canvas_h=28) -> Image.Image:
        S = SSAA
        W, H = canvas_w*S, canvas_h*S
        canvas = Image.new("RGBA", (W, H), (*_hex_to_rgb(C_BG_BASE), 0))
        draw = ImageDraw.Draw(canvas)
        PW = int(W*0.45); PH = H-8*S; PX, PY = 0, 4*S; r = PH//2
        _draw_rounded_rect(draw, (PX, PY, PX+PW, PY+PH), radius=r,
                           fill=_hex_to_rgb(C_BG_FLOAT), outline=_hex_to_rgb(C_BORDER_SUB), width=1)
        fw = max(int(PW*value), r*2) if value > 0 else 0
        if fw > 0:
            darker = _lerp_colour(colour, _hex_to_rgb(C_BG_DEEP), 0.4)
            grad = _make_gradient_rect(fw, PH, colour_top=colour, colour_bottom=darker,
                                       alpha_top=255, alpha_bottom=220)
            fm = Image.new("L", (PW, PH), 0)
            _draw_rounded_rect(ImageDraw.Draw(fm), (0,0,PW-1,PH-1), radius=r, fill=255)
            canvas.paste(grad.convert("RGB"), (PX, PY), mask=fm.crop((0,0,fw,PH)))
            gl = Image.new("RGBA", (W,H), (0,0,0,0))
            ImageDraw.Draw(gl).line([(PX+fw, PY+2*S),(PX+fw, PY+PH-2*S)],
                                    fill=(*colour,200), width=3*S)
            gl = gl.filter(ImageFilter.GaussianBlur(radius=4*S))
            canvas = Image.alpha_composite(canvas, gl)
        ImageDraw.Draw(canvas).text((PX+PW+8*S, PY+PH//2), label,
                                    fill=_hex_to_rgb(C_TEXT_PRI),
                                    font=_get_pil_font(9*S), anchor="lm")
        return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

    def _render_snr_bar(self, info: dict) -> Image.Image:
        v = info.get("normalized", 0.0); snr = info.get("snr", 0.0)
        lvl = info.get("level", "weak")
        c = _hex_to_rgb({"weak": C_CRITICAL, "moderate": C_WARNING, "strong": C_HEALTHY}.get(lvl, C_TEXT_MUT))
        return self._render_metric_bar(v, f"SNR {snr:.2f}  [{lvl}]", c)

    def _render_diversity_bar(self, info: dict) -> Image.Image:
        score = info.get("score", 0.0); pct = info.get("score_pct", 0)
        lvl = info.get("level", "n/a")
        c = (_hex_to_rgb(C_TEXT_MUT) if lvl == "n/a" else
             _hex_to_rgb({"low": C_CRITICAL, "moderate": C_WARNING, "high": C_HEALTHY}.get(lvl, C_TEXT_MUT)))
        return self._render_metric_bar(score, f"Div {pct}%  [{lvl}]", c)

    def _render_health_strip(self, health_metrics: dict | None,
                             input_stats: dict | None, canvas_h: int = 52) -> Image.Image:
        canvas_w = self.main_outer.winfo_width()
        if canvas_w < 10:
            canvas_w = 1100
        S = SSAA; W, H = canvas_w*S, canvas_h*S
        canvas = Image.new("RGBA", (W, H), (*_hex_to_rgb(C_BG_RAISED), 255))
        draw = ImageDraw.Draw(canvas)
        draw.line([(0, H-1), (W, H-1)], fill=_hex_to_rgb(C_BORDER_SUB), width=1)
        fl = _get_pil_font(8*S); fv = _get_pil_font(10*S)

        if not health_metrics:
            draw.text((W//2, H//2), "Run a forward pass to populate metrics",
                      fill=_hex_to_rgb(C_TEXT_MUT), font=fl, anchor="mm")
            return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

        dp = health_metrics.get("dead_percent", 0.0)
        sat = health_metrics.get("saturation", {})
        snr = health_metrics.get("snr", {})
        div = health_metrics.get("diversity", {})
        lc = {"healthy": C_HEALTHY, "mild": C_WARNING, "saturated": C_CRITICAL,
              "weak": C_CRITICAL, "moderate": C_WARNING, "strong": C_HEALTHY,
              "low": C_CRITICAL, "high": C_HEALTHY, "n/a": C_TEXT_MUT}

        def dc(pct): return _lerp_colour(_hex_to_rgb(C_HEALTHY), _hex_to_rgb(C_CRITICAL), min(pct/50,1))

        badges = [
            ("DEAD ReLU",  f"{dp:.1f}%",                          dc(dp)),
            ("SATURATION", f"{sat.get('layer_score',0):.1%}",     _hex_to_rgb(lc.get(sat.get("level","healthy"),C_TEXT_MUT))),
            ("SNR",        f"{snr.get('snr',0):.2f}",             _hex_to_rgb(lc.get(snr.get("level","weak"),C_TEXT_MUT))),
            ("DIVERSITY",  f"{div.get('score_pct',0)}%",          _hex_to_rgb(lc.get(div.get("level","n/a"),C_TEXT_MUT))),
        ]
        bh = H-12*S; by = 6*S; bw = W//(len(badges)+1)
        for i, (name, val, dc2) in enumerate(badges):
            bx = int((i+0.5)*W/len(badges)) - bw//2
            _draw_rounded_rect(draw, (bx,by,bx+bw-4*S,by+bh),
                               radius=bh//2, fill=_hex_to_rgb(C_BG_FLOAT),
                               outline=_hex_to_rgb(C_BORDER_SUB), width=1)
            dr = 5*S; dcx = bx+dr*2+4*S; dcy = by+bh//2
            gl = Image.new("RGBA",(W,H),(0,0,0,0))
            ImageDraw.Draw(gl).ellipse([dcx-dr*2,dcy-dr*2,dcx+dr*2,dcy+dr*2],fill=(*dc2,160))
            gl = gl.filter(ImageFilter.GaussianBlur(radius=dr))
            canvas = Image.alpha_composite(canvas, gl)
            draw = ImageDraw.Draw(canvas)
            draw.ellipse([dcx-dr,dcy-dr,dcx+dr,dcy+dr],fill=(*dc2,255))
            tx = dcx+dr+6*S
            draw.text((tx,dcy-5*S), name, fill=_hex_to_rgb(C_TEXT_MUT), font=fl, anchor="lm")
            draw.text((tx,dcy+6*S), val,  fill=_hex_to_rgb(C_TEXT_PRI), font=fv, anchor="lm")

        if input_stats:
            ok = abs(input_stats.get("mean",0)) <= 0.8 and input_stats.get("std",1) >= 0.2
            ImageDraw.Draw(canvas).text((W-16*S, H//2),
                                        "INPUT OK" if ok else "⚠ INPUT NORM",
                                        fill=_hex_to_rgb(C_TEXT_MUT if ok else C_WARNING),
                                        font=fl, anchor="rm")

        return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

    def _render_flow_chart(self, flow_data: dict, selected_layer=None,
                           canvas_w=900, canvas_h=180) -> Image.Image:
        S = SSAA; W, H = canvas_w*S, canvas_h*S
        canvas = Image.new("RGBA",(W,H),(*_hex_to_rgb(C_BG_BASE),255))
        draw = ImageDraw.Draw(canvas)
        if not flow_data:
            draw.text((W//2,H//2),"Run a forward pass to populate flow data",
                      fill=_hex_to_rgb(C_TEXT_MUT),font=_get_pil_font(12*S),anchor="mm")
            return canvas.resize((canvas_w,canvas_h),Image.LANCZOS).convert("RGB")
        ML,MR,MT,MB = 55*S,15*S,20*S,40*S
        pw = W-ML-MR; ph = H-MT-MB
        layers = list(flow_data.keys()); N = len(layers)
        mv = max(flow_data.values()) if flow_data else 1.0
        if mv < 1e-6: mv = 1.0
        draw.text((10*S,15*S),f"Network Activation Flow  ({N} conv layers)",
                  fill=_hex_to_rgb(C_TEXT_SEC),font=_get_pil_font(10*S))
        gc = (*_hex_to_rgb(C_BORDER_SUB),140)
        for pct in (0.25,0.5,0.75):
            y = MT+int(ph*(1-pct))
            for x in range(ML,W-MR,10*S): draw.line([(x,y),(x+5*S,y)],fill=gc,width=1)
        sw = pw/max(N,1)
        for i, layer in enumerate(layers):
            val = flow_data[layer]; bh2 = max(2*S,int((val/mv)*ph))
            cx = ML+int(i*sw+sw/2); y1 = MT+ph-bh2; y2 = MT+ph
            color = _hex_to_rgb(C_HEALTHY if i<N*0.33 else C_INFO if i<N*0.66 else C_WARNING)
            draw.line([(cx,y1),(cx,y2)],fill=(*_hex_to_rgb(C_BORDER_SUB),200),width=2*S)
            dr2 = 4*S
            gl = Image.new("RGBA",(W,H),(0,0,0,0))
            ImageDraw.Draw(gl).ellipse([cx-dr2*2,y1-dr2*2,cx+dr2*2,y1+dr2*2],fill=(*color,120))
            gl = gl.filter(ImageFilter.GaussianBlur(radius=dr2))
            canvas = Image.alpha_composite(canvas,gl)
            draw = ImageDraw.Draw(canvas)
            draw.ellipse([cx-dr2,y1-dr2,cx+dr2,y1+dr2],fill=(*color,255))
            if layer == selected_layer:
                draw.ellipse([cx-6*S,y1-6*S,cx+6*S,y1+6*S],outline=(255,255,255),width=2*S)
                draw.polygon([(cx,y1-12*S),(cx-6*S,y1-20*S),(cx+6*S,y1-20*S)],fill=(255,255,255))
            lbl = layer[:9]+"…" if len(layer)>10 else layer
            ti = Image.new("RGBA",(80*S,20*S),(0,0,0,0))
            ImageDraw.Draw(ti).text((0,10*S),lbl,font=_get_pil_font(8*S),
                                    fill=_hex_to_rgb(C_TEXT_MUT),anchor="lm")
            ti = ti.rotate(45,resample=Image.BICUBIC,expand=True)
            canvas.paste(ti,(cx-10*S,y2+5*S),ti)
        return canvas.resize((canvas_w,canvas_h),Image.LANCZOS).convert("RGB")

    def _render_correlation_matrix(self, sim_matrix, layer_name, sorted_indices,
                                   frame_age_seconds, canvas_w=400, canvas_h=400) -> Image.Image:
        S = SSAA; W, H = canvas_w*S, canvas_h*S
        canvas = Image.new("RGBA",(W,H),(*_hex_to_rgb(C_BG_BASE),255))
        draw = ImageDraw.Draw(canvas); font = _get_pil_font(10*S)
        if sim_matrix is None:
            draw.text((W//2,H//2),"No similarity data available.",
                      fill=_hex_to_rgb(C_TEXT_MUT),font=font,anchor="mm")
            return canvas.resize((canvas_w,canvas_h),Image.LANCZOS).convert("RGB")
        ch = sim_matrix.shape[0]; mx = 120
        ds = sim_matrix
        if sorted_indices is not None and len(sorted_indices)==ch:
            si = sorted_indices[:mx] if ch>mx else sorted_indices
            ds = sim_matrix[si][:,si]
        elif ch>mx:
            step = ch//mx; idx = torch.arange(0,ch,step)[:mx]; ds = sim_matrix[idx][:,idx]
        sn = (ds.cpu().numpy()+1.0)/2.0
        ta = [(0.0,"#30123B"),(0.1,"#4662D7"),(0.3,"#36AAF9"),(0.5,"#1AE4B6"),
              (0.7,"#72FE5E"),(0.8,"#C6ED34"),(0.9,"#FABA39"),(0.95,"#F66C19"),(1.0,"#7A0403")]
        hi = Image.fromarray(_apply_colormap(sn,ta),mode="RGB")
        PAD = 40*S; ps = min(W-2*PAD,H-2*PAD)
        hs = hi.resize((ps,ps),Image.NEAREST)
        sx2 = (W-ps)//2; sy2 = (H-ps)//2+10*S
        canvas.paste(hs,(sx2,sy2))
        _draw_rounded_rect(draw,(sx2-1,sy2-1,sx2+ps,sy2+ps),radius=0,
                           outline=_hex_to_rgb(C_BORDER_SUB),width=1)
        draw.text((sx2,sy2-15*S),f"Filter Correlation: {layer_name}",
                  fill=_hex_to_rgb(C_TEXT_PRI),font=font)
        if frame_age_seconds>0:
            draw.text((sx2+ps,sy2-15*S),f"Age: {frame_age_seconds:.1f}s",
                      fill=_hex_to_rgb(C_TEXT_MUT),font=font,anchor="ra")
        return canvas.resize((canvas_w,canvas_h),Image.LANCZOS).convert("RGB")

    def _render_spatial_bias(self, tensor: torch.Tensor,
                             canvas_w=280, canvas_h=140) -> Image.Image | None:
        if tensor is None or len(tensor.shape) < 2: return None
        td = tensor.detach().cpu().numpy(); ht, wt = td.shape
        if ht==0 or wt==0: return None
        S = SSAA; W, H = canvas_w*S, canvas_h*S
        canvas = Image.new("RGBA",(W,H),(*_hex_to_rgb(C_BG_RAISED),255))
        try:
            t_min,t_max = td.min(),td.max()
            tn = (td-t_min)/(t_max-t_min) if t_max>t_min else np.zeros_like(td)
            ma = [(0.0,"#000004"),(0.2,"#2b115f"),(0.4,"#721f81"),
                  (0.6,"#c33d69"),(0.8,"#fca50a"),(1.0,"#fcfdbf")]
            hr = Image.fromarray(_apply_colormap(tn,ma),mode="RGB")
            hl = hr.resize((W,H),Image.BILINEAR).convert("RGBA"); hl.putalpha(150)
            canvas = Image.alpha_composite(canvas,hl)
            yi,xi = np.unravel_index(np.argmax(tn),tn.shape)
            xm = int((xi+0.5)/wt*W); ym = int((yi+0.5)/ht*H)
            draw = ImageDraw.Draw(canvas); cc=(255,255,255,180)
            draw.line([(xm,0),(xm,H)],fill=cc,width=1*S)
            draw.line([(0,ym),(W,ym)],fill=cc,width=1*S)
            rad = 3*S
            draw.ellipse([xm-rad,ym-rad,xm+rad,ym+rad],fill=(255,255,0,255),
                         outline=(0,0,0,100),width=1)
            return canvas.resize((canvas_w,canvas_h),Image.LANCZOS).convert("RGB")
        except Exception: return None

    # ------------------------------------------------------------------ #
    #  Telemetry (MAIN THREAD ONLY)                                        #
    # ------------------------------------------------------------------ #

    def _update_telemetry(self, layer_name, tensor_shape, input_stats, health_metrics):
        """
        Updates the health strip label.
        PERF FIX #7: This method is now called ONLY from _poll_results(),
        never from the inference thread. No risk of Tkinter race conditions.
        PERF FIX #5: The expensive SSAA strip render is throttled to every
        5 poll cycles — health metrics change slowly.
        """
        self.last_layer_name     = layer_name
        self.last_health_metrics = health_metrics
        self.last_input_stats    = input_stats

        self._health_strip_poll_count += 1
        if self._health_strip_poll_count % 5 != 0 and self.camera_active:
            return  # skip render this cycle in live mode

        pil = self._render_health_strip(health_metrics, input_stats)
        ci = ctk.CTkImage(light_image=pil, dark_image=pil,
                          size=(pil.width, pil.height))
        self.health_strip_label.configure(image=ci)

    # ------------------------------------------------------------------ #
    #  Tab management                                                      #
    # ------------------------------------------------------------------ #

    def _on_tab_change(self):
        tab = self.tabview.get()
        if tab == "Flow Chart":    self._refresh_flow_tab(force=True)
        elif tab == "Correlation": self._refresh_correlation_tab(force=True)

    def _refresh_flow_tab(self, force: bool = False):
        """
        PERF FIX #6: Only re-render when flow_data has actually changed
        OR when forced (user switches to the tab).
        """
        if not self._last_flow_data:
            return
        if not force and self._last_flow_data is self._flow_data_rendered:
            return   # same dict object — data hasn't changed
        self._flow_data_rendered = self._last_flow_data
        self._flow_frame_count += 1
        is_live = self.camera_active and self.frozen_frame is None
        if is_live and self._flow_frame_count % 3 != 0:
            return

        pil = self._render_flow_chart(self._last_flow_data,
                                      self.sidebar.config.get("layer"))
        tw = max(self.tabview.winfo_width(), 900)
        th = max(self.tabview.winfo_height(), 180)
        sh = min(th-40, int(pil.height * tw / max(pil.width, 1)))
        if sh < 10: sh = 180
        pr = pil.resize((tw-20, sh), Image.LANCZOS)
        ci = ctk.CTkImage(light_image=pr, dark_image=pr, size=(tw-20, sh))
        self.flow_tab_label.configure(image=ci)

    def _refresh_correlation_tab(self, force: bool = False):
        """
        PERF FIX #6: Only re-render when sim_matrix has actually changed
        OR when forced.
        """
        if self._last_sim_matrix is None:
            return
        if not force and self._last_sim_matrix is self._sim_matrix_rendered:
            return
        self._sim_matrix_rendered = self._last_sim_matrix
        self._corr_matrix_frame_count += 1
        is_live = self.camera_active and self.frozen_frame is None
        if is_live and self._last_sim_matrix.shape[0] > 64 and self._corr_matrix_frame_count % 10 != 0:
            return

        age = time.time() - self._corr_matrix_last_update_time
        pil = self._render_correlation_matrix(
            self._last_sim_matrix,
            self.sidebar.config.get("layer", ""),
            self._corr_matrix_sorted_indices,
            age if is_live else 0.0,
        )
        tw = self.tabview.winfo_width(); th = self.tabview.winfo_height()
        ps = min(tw-20, th-40)
        if ps < 10: ps = 400
        pr = pil.resize((ps, ps), Image.LANCZOS)
        ci = ctk.CTkImage(light_image=pr, dark_image=pr, size=(ps, ps))
        self.corr_tab_label.configure(image=ci)

    # ------------------------------------------------------------------ #
    #  Export                                                              #
    # ------------------------------------------------------------------ #

    def on_export_diagnostics(self):
        if not self.last_layer_name and not self.last_health_metrics:
            messagebox.showwarning("Export", "No active data to export. Run a forward pass first.")
            return
        try:
            content = generate_report(
                self.last_layer_name, self.last_health_metrics,
                self.last_input_stats, self.sidebar.config["dead_threshold"],
            )
            save_report(content)
            old = self.sidebar.export_btn.cget("text")
            self.sidebar.export_btn.configure(text="✔ Saved!")
            self.after(2000, lambda: self.sidebar.export_btn.configure(text=old))
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
            images, title, _, health_metrics, input_stats, flow_data, sim_matrix = (
                self._run_visualization_pipeline(self.image_path, "Static Upload")
            )
            loading.destroy()
            layer_name = self.sidebar.config.get("layer", "")
            ch_filter = parse_channel_filter(
                self.sidebar.config.get("channel_filter_raw", ""))
            self.grid_view.update(images, title, force_rebuild=True,
                                  health_metrics=health_metrics, layer_name=layer_name,
                                  cell_size=self.sidebar.config.get("cell_size", 140),
                                  channel_filter=ch_filter)
            self._update_telemetry(layer_name, None, input_stats, health_metrics)
            if flow_data:     self._last_flow_data = flow_data
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
            time.sleep(0.005)

    def _inference_thread_func(self):
        while not self._inference_stop_event.is_set():
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
                result = self._run_visualization_pipeline(
                    frame, source,
                    cached_redundancy=self._cached_redundancy,
                    frame_count=self._frame_count,
                )
                self._cached_redundancy = result[2]
                self._frame_count += 1
                try:
                    self._result_queue.put_nowait(result)
                except queue.Full:
                    pass
            except Exception:
                pass

    def _poll_results(self):
        if not self.camera_active:
            return
        try:
            images, title, _, health_metrics, input_stats, flow_data, sim_matrix = (
                self._result_queue.get_nowait()
            )
            layer_name = self.sidebar.config.get("layer", "")
            ch_filter = parse_channel_filter(
                self.sidebar.config.get("channel_filter_raw", ""))
            self.grid_view.update(images, title, force_rebuild=False,
                                  health_metrics=health_metrics, layer_name=layer_name,
                                  cell_size=self.sidebar.config.get("cell_size", 140),
                                  channel_filter=ch_filter)

            # PERF FIX #7: telemetry updated here on main thread, not in pipeline
            self._update_telemetry(layer_name, None, input_stats, health_metrics)

            if flow_data:
                self._last_flow_data = flow_data
            if sim_matrix is not None:
                self._last_sim_matrix = sim_matrix
                self._corr_matrix_last_update_time = time.time()

            # PERF FIX #6: only refresh the active tab, and only if data changed
            tab = self.tabview.get()
            if tab == "Flow Chart":    self._refresh_flow_tab()
            elif tab == "Correlation": self._refresh_correlation_tab()

            if self._inspector_visible and self._inspected_channel is not None:
                if (self._last_tensor is not None and
                        self._inspected_channel < self._last_tensor.shape[1]):
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
                    messagebox.showerror("Camera Error", str(e)); return

            self.sidebar.camera_btn.configure(text="Stop Live Camera",
                                              fg_color="#c92a2a", hover_color="#a61e1e")
            self.sidebar.cam_status_label.configure(text="Camera Active", text_color="green")
            self.sidebar.freeze_btn.configure(state="normal")
            self.sidebar.status_badge.configure(text="● LIVE", text_color=C_HEALTHY)

            self.frozen_frame = None; self._frame_count = 0
            self._cached_redundancy = None; self._latest_frame = None
            self._health_strip_poll_count = 0
            while not self._result_queue.empty():
                try: self._result_queue.get_nowait()
                except queue.Empty: break

            self._inference_stop_event.clear()
            self._capture_thread   = threading.Thread(target=self._capture_thread_func, daemon=True)
            self._inference_thread = threading.Thread(target=self._inference_thread_func, daemon=True)
            self._capture_thread.start(); self._inference_thread.start()
            self._poll_results()
        else:
            self.sidebar.camera_btn.configure(text="Start Live Camera",
                                              fg_color=C_HEALTHY, hover_color="#22c55e")
            self.sidebar.cam_status_label.configure(text="Camera Offline", text_color=C_TEXT_MUT)
            self.sidebar.freeze_btn.configure(state="disabled")
            self.sidebar.status_badge.configure(text="● IDLE", text_color=C_TEXT_MUT)
            self._inference_stop_event.set()
            if self._capture_thread:   self._capture_thread.join(timeout=1.0)
            if self._inference_thread: self._inference_thread.join(timeout=1.0)
            self.frozen_frame = None
            if self.camera: self.camera.release(); self.camera = None

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
        self._inspected_channel = channel_idx
        self._inspected_layer   = layer_name or self.sidebar.config.get("layer")

        if not self._inspector_visible:
            # Add the inspector pane to the PanedWindow on demand
            self._paned.add(self._inspector_pane_frame, weight=0)
            self._inspector_visible = True

        self._update_inspector()
        self.grid_view.refresh_borders(channel_idx)

    def _close_inspector(self):
        self._inspector_visible = False
        self._inspected_channel = None
        self._inspected_layer   = None
        # Remove from PanedWindow (hides the pane and releases the space)
        try:
            self._paned.forget(self._inspector_pane_frame)
        except Exception:
            pass
        self.grid_view.clear_selection()

    def _update_inspector(self):
        ch = self._inspected_channel
        if ch is None or self._last_tensor is None: return
        tensor = self._last_tensor
        if ch >= tensor.shape[1]: return

        layer = self._inspected_layer
        self.inspector_title.configure(text=f"Channel {ch}  ·  {layer or '—'}")
        ch_map = tensor[0, ch]

        # Feature map
        cn = ch_map.detach().cpu().numpy()
        cmin, cmax = cn.min(), cn.max()
        ch_norm = ((cn-cmin)/(cmax-cmin)*255).astype(np.uint8) if cmax>cmin else np.zeros_like(cn,dtype=np.uint8)
        chl = cv2.resize(ch_norm,(280,280),interpolation=cv2.INTER_LANCZOS4)
        chc = cv2.cvtColor(cv2.applyColorMap(chl,cv2.COLORMAP_VIRIDIS),cv2.COLOR_BGR2RGB)
        pm = Image.fromarray(chc)
        cm = ctk.CTkImage(light_image=pm,dark_image=pm,size=(280,280))
        self.inspector_map_label.configure(image=cm)

        hm = self.last_health_metrics or {}
        dead_flag = "YES" if ch_map.var().item() < self.sidebar.health_slider.get() else "no"
        sat_val = "N/A"
        si = hm.get("saturation",{})
        if si.get("per_channel") is not None and ch < len(si["per_channel"]):
            sat_val = f"{si['per_channel'][ch].item()*100:.1f}%"
        ema_rank = "N/A"
        if self._last_sorted_indices is not None:
            il = self._last_sorted_indices.tolist()
            if ch in il: ema_rank = f"#{il.index(ch)+1} of {len(il)}"
        for key, val in {
            "Layer": layer or "—", "Channel": str(ch), "EMA Rank": ema_rank,
            "Act Range": f"{ch_map.min().item():.3f} → {ch_map.max().item():.3f}",
            "Mean / Std": f"{ch_map.mean().item():.3f}  /  {ch_map.std().item():.3f}",
            "Saturation": sat_val, "Dead": dead_flag,
        }.items():
            if key in self.inspector_stats:
                c = C_CRITICAL if key=="Dead" and val=="YES" else C_TEXT_PRI
                self.inspector_stats[key].configure(text=val, text_color=c)

        hi = self._render_histogram(ch_map, canvas_w=280, canvas_h=100)
        ch2 = ctk.CTkImage(light_image=hi,dark_image=hi,size=(280,100))
        self.inspector_hist_label.configure(image=ch2)

        if self._last_sim_matrix is not None and ch < self._last_sim_matrix.shape[0]:
            sr = self._last_sim_matrix[ch].clone(); sr[ch] = 0.0
            t3v, t3i = torch.topk(sr, min(3, sr.shape[0]))
            for slot, (sc, sv) in enumerate(zip(t3i.tolist(), t3v.tolist())):
                if slot >= len(self.inspector_similar_labels): break
                il2, tl2 = self.inspector_similar_labels[slot]
                cell = self.grid_view.get_cell_data(sc)
                if cell and cell[0]:
                    si2 = cell[0].resize((70,70),Image.NEAREST)
                    ci2 = ctk.CTkImage(light_image=si2,dark_image=si2,size=(70,70))
                    il2.configure(image=ci2); tl2.configure(text=f"Ch {sc}\n{sv:.2f}")
                else:
                    il2.configure(image=""); tl2.configure(text="-")

        bi = self._render_spatial_bias(ch_map)
        if bi:
            cb = ctk.CTkImage(light_image=bi,dark_image=bi,size=(280,140))
            self.inspector_bias_label.configure(image=cb,text="")