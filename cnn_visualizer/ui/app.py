"""
Main application window coordinating UI and logic.
"""
from __future__ import annotations
import os
import customtkinter as ctk
import cv2
import numpy as np
import threading
import queue
import time
import torch
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

SSAA = 2  # Supersampling scale factor

def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _lerp_colour(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))

def _apply_colormap(values: np.ndarray, anchors: list[tuple[float, str]]) -> np.ndarray:
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
    padded = np.pad(values.astype(np.float32), radius, mode='reflect')
    return np.convolve(padded, kernel, mode='valid')

def _make_gradient_rect(width: int, height: int, colour_top: tuple[int, int, int], colour_bottom: tuple[int, int, int], alpha_top: int = 255, alpha_bottom: int = 255) -> Image.Image:
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    for y in range(height):
        t = y / max(height - 1, 1)
        r, g, b = _lerp_colour(colour_top, colour_bottom, t)
        a = int(alpha_top + (alpha_bottom - alpha_top) * t)
        arr[y, :] = [r, g, b, a]
    return Image.fromarray(arr, mode='RGBA')

def _draw_rounded_rect(draw: ImageDraw.ImageDraw, xy: tuple, radius: int, fill: tuple = None, outline: tuple = None, width: int = 1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)

def _get_pil_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

def _vignette_overlay(width: int, height: int, strength: float = 0.6) -> Image.Image:
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    cx, cy = width / 2, height / 2
    max_dist = np.sqrt(cx**2 + cy**2)
    ys, xs = np.mgrid[0:height, 0:width]
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2) / max_dist
    alpha = (dist ** 1.5 * strength * 255).clip(0, 255).astype(np.uint8)
    arr[..., 3] = alpha
    return Image.fromarray(arr, mode='RGBA')

class CNNVisualizerApp(ctk.CTk):
    """The main Graphical User Interface built with CustomTkinter."""
    def __init__(self):
        super().__init__()

        self.title("CNN Activation Visualizer")
        self.geometry("1560x920")
        self.minsize(1100, 700)
        self.configure(fg_color=C_BG_DEEP)

        # Backend state
        self.hook_engine = None
        
        # State variables
        self.image_path = None
        self.current_model = None
        self.camera = None
        self.camera_active = False
        self.ema_channel_sums = None

        # Diagnostics telemetry state
        self.last_health_metrics = None
        self.last_input_stats = None
        self.last_layer_name = None

        # Freeze state
        self.frozen_frame = None

        # Threading infrastructure for Live Camera
        self._capture_thread = None
        self._inference_thread = None
        self._inference_stop_event = threading.Event()
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._result_queue = queue.Queue(maxsize=1)
        
        self._frame_count = 0
        self._cached_redundancy = None

        # Feature 1 — Flow Chart
        self._last_flow_data: dict[str, float] = {}
        self._flow_frame_count: int = 0

        self._last_sim_matrix: torch.Tensor | None = None
        self._corr_matrix_frame_count: int = 0
        self._corr_matrix_last_update_time: float = 0.0
        self._corr_matrix_sorted_indices: torch.Tensor | None = None
        
        # New Feature states
        self._inspected_channel: int | None = None
        self._inspected_layer: str | None = None
        self._inspector_visible: bool = False
        self._last_sorted_indices: torch.Tensor | None = None
        
        # Inspector Widgets
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

    def _setup_ui(self):
        # 3 Column Layout
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
            on_speed_change=None # Handled within config pull in camera loop
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
            command=self._on_tab_change
        )
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.tabview.add("Feature Maps")
        self.tabview.add("Flow Chart")
        self.tabview.add("Correlation")

        self.grid_view = GridView(self.tabview.tab("Feature Maps"), on_channel_click=self._on_channel_click)
        self.grid_view.pack(fill="both", expand=True)
        
        self.flow_tab_label = ctk.CTkLabel(self.tabview.tab("Flow Chart"), text="")
        self.flow_tab_label.pack(fill="both", expand=True)
        
        self.corr_tab_label = ctk.CTkLabel(self.tabview.tab("Correlation"), text="")
        self.corr_tab_label.pack(fill="both", expand=True)

    def _build_inspector_skeleton(self):
        """Builds static inspector widgets dynamically on init."""
        self.inspector_frame = ctk.CTkScrollableFrame(
            self,
            width=340,
            fg_color=C_BG_RAISED,
            border_width=1,
            border_color=C_BORDER_SUB,
            corner_radius=0,
            label_text=""
        )

        # Header Row
        header_frame = ctk.CTkFrame(self.inspector_frame, fg_color=C_BG_BASE, corner_radius=6)
        header_frame.pack(fill="x", pady=(0, 6))
        
        self.inspector_title = ctk.CTkLabel(header_frame, text="Channel Inspector", font=ctk.CTkFont(size=13, weight="bold"), text_color=C_TEXT_PRI)
        self.inspector_title.pack(side="left", padx=12, pady=8)
        
        self.inspector_close_btn = ctk.CTkButton(header_frame, text="✕", width=28, height=28, fg_color=C_BG_FLOAT, hover_color=C_CRITICAL, command=self._close_inspector)
        self.inspector_close_btn.pack(side="right", padx=6, pady=6)
        
        # Feature map panel
        map_frame = ctk.CTkFrame(self.inspector_frame, fg_color=C_BG_BASE, border_width=1, border_color=C_BORDER_SUB, corner_radius=6)
        map_frame.pack(fill="x", pady=6)
        ctk.CTkLabel(map_frame, text="ACTIVATION MAP", font=ctk.CTkFont(size=9, weight="bold"), text_color=C_TEXT_MUT).pack(anchor="w", padx=12, pady=(8, 4))
        self.inspector_map_label = ctk.CTkLabel(map_frame, text="")
        self.inspector_map_label.pack(pady=8)
        
        # Stats panel
        stats_frame = ctk.CTkFrame(self.inspector_frame, fg_color=C_BG_BASE, border_width=1, border_color=C_BORDER_SUB, corner_radius=6)
        stats_frame.pack(fill="x", pady=6)
        ctk.CTkLabel(stats_frame, text="STATISTICS", font=ctk.CTkFont(size=9, weight="bold"), text_color=C_TEXT_MUT).pack(anchor="w", padx=12, pady=(8, 4))
        
        stat_keys = ["Layer", "Channel", "EMA Rank", "Act Range", "Mean / Std", "Saturation", "Dead"]
        for key in stat_keys:
            r = ctk.CTkFrame(stats_frame, fg_color="transparent")
            r.pack(fill="x", padx=12, pady=2)
            ctk.CTkLabel(r, text=f"{key}:", font=ctk.CTkFont(size=10), text_color=C_TEXT_MUT).pack(side="left")
            vl = ctk.CTkLabel(r, text="-", font=ctk.CTkFont(size=10, weight="bold"), text_color=C_TEXT_PRI)
            vl.pack(side="right")
            self.inspector_stats[key] = vl
            
        # Histogram
        hist_frame = ctk.CTkFrame(self.inspector_frame, fg_color=C_BG_BASE, border_width=1, border_color=C_BORDER_SUB, corner_radius=6)
        hist_frame.pack(fill="x", pady=6)
        ctk.CTkLabel(hist_frame, text="ACTIVATION DISTRIBUTION", font=ctk.CTkFont(size=9, weight="bold"), text_color=C_TEXT_MUT).pack(anchor="w", padx=12, pady=(8, 4))
        self.inspector_hist_label = ctk.CTkLabel(hist_frame, text="")
        self.inspector_hist_label.pack(pady=8)
        
        # Similar Channels
        sim_frame = ctk.CTkFrame(self.inspector_frame, fg_color=C_BG_BASE, border_width=1, border_color=C_BORDER_SUB, corner_radius=6)
        sim_frame.pack(fill="x", pady=6)
        ctk.CTkLabel(sim_frame, text="MOST SIMILAR CHANNELS", font=ctk.CTkFont(size=9, weight="bold"), text_color=C_TEXT_MUT).pack(anchor="w", padx=12, pady=(8, 4))
        sim_grid = ctk.CTkFrame(sim_frame, fg_color="transparent")
        sim_grid.pack(pady=8)
        for i in range(3):
            sub = ctk.CTkFrame(sim_grid, fg_color="transparent")
            sub.pack(side="left", padx=4)
            img = ctk.CTkLabel(sub, text="")
            img.pack()
            txt = ctk.CTkLabel(sub, text="-", font=ctk.CTkFont(size=10), text_color=C_TEXT_MUT)
            txt.pack()
            self.inspector_similar_labels.append((img, txt))
            
        # Bias
        bias_frame = ctk.CTkFrame(self.inspector_frame, fg_color=C_BG_BASE, border_width=1, border_color=C_BORDER_SUB, corner_radius=6)
        bias_frame.pack(fill="x", pady=6)
        ctk.CTkLabel(bias_frame, text="SPATIAL FOCUS", font=ctk.CTkFont(size=9, weight="bold"), text_color=C_TEXT_MUT).pack(anchor="w", padx=12, pady=(8, 0))
        ctk.CTkLabel(bias_frame, text="Where this filter concentrates attention", font=ctk.CTkFont(size=9), text_color=C_TEXT_MUT).pack(anchor="w", padx=12, pady=(0, 4))
        self.inspector_bias_label = ctk.CTkLabel(bias_frame, text="")
        self.inspector_bias_label.pack(pady=8)

    # ------------------------------------------------------------------ #
    #  Model / image loading                                               #
    # ------------------------------------------------------------------ #

    def on_model_select(self, model_name: str):
        """Triggered when a user selects a model from the combobox."""
        self.sidebar.layer_dropdown.set("Loading...")
        self.update()

        try:
            model, layer_names = load_model(model_name)
            # Create dict for HookEngine mapping layer_names -> nn.Module
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
        """Triggered to open a file selection dialog."""
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.sidebar.img_path_label.configure(text=os.path.basename(file_path))

    # ------------------------------------------------------------------ #
    #  Core visualization pipeline                                         #
    # ------------------------------------------------------------------ #

    def _run_visualization_pipeline(self, frame_or_path, source_text: str, cached_redundancy=None, frame_count=0) -> tuple:
        """Unified method for running standard inference and returning generated labeled images."""
        cfg = self.sidebar.config
        mode = cfg["mode"]
        images = []
        dynamic_title = ""
        layer_flow_data = {}

        selected_layer = cfg["layer"]

        # --- GRAD-CAM MODE ---
        if cfg["gradcam"]:
            if not selected_layer or selected_layer == "Load model first...":
                raise ValueError("Please select a valid Conv Layer for Grad-CAM.")

            if isinstance(frame_or_path, str):
                frame = cv2.imread(frame_or_path)
            else:
                frame = frame_or_path

            pil_img_original, gradcam_heatmap, layer_flow_data = compute_gradcam(
                self.hook_engine.model,
                self.hook_engine.conv_layers,
                self.hook_engine.transform,
                frame,
                selected_layer
            )

            if gradcam_heatmap is not None and pil_img_original is not None:
                heatmap_resized = cv2.resize(gradcam_heatmap, (pil_img_original.width, pil_img_original.height))
                heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

                original_np = np.array(pil_img_original)
                overlay = cv2.addWeighted(original_np, 0.4, heatmap_color, 0.6, 0)

                final_img = Image.fromarray(overlay)
                images.append((final_img, f"Grad-CAM via {selected_layer}", "normal"))
                dynamic_title = f"{source_text} Class Saliency (Targeting: {selected_layer})"
                self._update_telemetry(selected_layer, None, None, None)

            return images, dynamic_title, cached_redundancy, None, layer_flow_data, None

        # --- LAYER MODE ---
        if mode == "Layer Mode":
            if not selected_layer or selected_layer == "Load model first...":
                raise ValueError("Please select a valid Conv Layer.")

            tensor = None
            input_stats = None
            if isinstance(frame_or_path, str):
                tensor, input_stats, layer_flow_data = self.hook_engine.extract_features(frame_or_path, selected_layer)
            else:
                tensor, input_stats, layer_flow_data = self.hook_engine.extract_features_from_frame(frame_or_path, selected_layer)

            if tensor is None:
                raise ValueError(f"Failed to retrieve feature map activation for {selected_layer}.")

            feature_shape = tensor.shape

            images, self.ema_channel_sums, health_metrics, cached_redundancy, sim_matrix = process_tensor_to_images(
                tensor,
                max_channels=36,
                is_live=(source_text == "Live Feed"),
                use_heatmap=cfg["heatmap"],
                ema_sums=self.ema_channel_sums,
                dead_threshold=cfg["dead_threshold"],
                cached_redundancy=cached_redundancy,
                frame_count=frame_count
            )
            
            # Store ranked channel indices for Correlation Matrix
            if self.ema_channel_sums is not None:
                self._corr_matrix_sorted_indices = torch.argsort(self.ema_channel_sums, descending=True)

            self._update_telemetry(selected_layer, feature_shape, input_stats, health_metrics)
            
            dynamic_title = f"{source_text} Feature Maps: '{selected_layer}' (First {len(images)} channels sorted)"
            return images, dynamic_title, cached_redundancy, health_metrics, layer_flow_data, sim_matrix

        # --- CHANNEL MODE ---
        elif mode == "Channel Mode":
            try:
                target_idx = int(cfg["channel"])
            except ValueError:
                raise ValueError("Channel Index must be a valid number.")

            self.ema_channel_sums = None
            all_layers = list(self.hook_engine.conv_layers.keys())
            tensor_dict = {}
            input_stats = None

            if isinstance(frame_or_path, str):
                result = self.hook_engine.extract_features(frame_or_path, all_layers)
                if isinstance(result, tuple) and len(result) == 3:
                    tensor_dict, input_stats, layer_flow_data = result
            else:
                result = self.hook_engine.extract_features_from_frame(frame_or_path, all_layers)
                if isinstance(result, tuple) and len(result) == 3:
                    tensor_dict, input_stats, layer_flow_data = result

            if tensor_dict:
                for target_layer in all_layers:
                    t = tensor_dict.get(target_layer)
                    if t is not None:
                        if len(t.shape) == 4:
                            t = t.squeeze(0)

                        if target_idx < t.shape[0]:
                            slice_tensor = t[target_idx].unsqueeze(0).unsqueeze(0)
                            processed, _, _, _, _ = process_tensor_to_images(
                                slice_tensor, max_channels=1, is_live=False,
                                use_heatmap=cfg["heatmap"],
                                cached_redundancy=cached_redundancy,
                                frame_count=frame_count
                            )
                            if processed:
                                img_obj, _, diag = processed[0]
                                images.append((img_obj, target_layer, diag))

            self._update_telemetry(None, None, input_stats, None)
            dynamic_title = f"{source_text} Feature Maps: Channel {target_idx} across Network ({len(images)} layers found)"

            return images, dynamic_title, cached_redundancy, None, layer_flow_data, None

        return images, dynamic_title, cached_redundancy, None, layer_flow_data, None

    # ------------------------------------------------------------------ #
    #  Telemetry                                                           #
    # ------------------------------------------------------------------ #

    def _render_histogram(
        self,
        tensor: torch.Tensor,
        canvas_w: int = 800,
        canvas_h: int = 140
    ) -> Image.Image:
        S = SSAA
        W, H = canvas_w * S, canvas_h * S

        canvas = Image.new("RGBA", (W, H), (*_hex_to_rgb(C_BG_BASE), 255))
        draw = ImageDraw.Draw(canvas)

        ML, MR, MT, MB = 48*S, 24*S, 20*S, 32*S
        plot_w = W - ML - MR
        plot_h = H - MT - MB

        if tensor is None or tensor.numel() == 0:
            font = _get_pil_font(12 * S)
            draw.text((W//2, H//2), "No activation data", fill=_hex_to_rgb(C_TEXT_MUT),
                      font=font, anchor="mm")
            return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

        flat = tensor.detach().cpu().numpy().ravel().astype(np.float32)
        NUM_BINS = 64
        counts, bin_edges = np.histogram(flat, bins=NUM_BINS)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

        smooth_counts = _smooth_signal(counts.astype(np.float32), sigma=1.5)
        smooth_counts = np.maximum(smooth_counts, 0)
        max_count = smooth_counts.max() if smooth_counts.max() > 0 else 1.0

        grid_colour = (*_hex_to_rgb(C_BORDER_SUB), 120)
        for frac in (0.25, 0.5, 0.75, 1.0):
            y = MT + plot_h - int(frac * plot_h)
            for x in range(ML, W - MR, 10*S):
                draw.line([(x, y), (x + 5*S, y)], fill=grid_colour, width=1)

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
            colour_top=accent_rgb,    colour_bottom=_hex_to_rgb(C_BG_BASE),
            alpha_top=200,            alpha_bottom=0
        )
        mask_img = Image.new("L", (W, H), 0)
        mask_draw = ImageDraw.Draw(mask_img)
        mask_draw.polygon(poly_points, fill=255)
        canvas.paste(
            grad.convert("RGB"),
            (ML, MT),
            mask=mask_img.crop((ML, MT, ML + plot_w, MT + plot_h))
        )

        curve_points = list(zip(xs[:-1], ys))

        glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        glow_d = ImageDraw.Draw(glow_layer)
        glow_d.line(curve_points, fill=(*accent_rgb, 140), width=6*S)
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=4*S))
        canvas = Image.alpha_composite(canvas.convert("RGBA"), glow_layer)

        draw = ImageDraw.Draw(canvas)
        draw.line(curve_points, fill=(*accent_rgb, 255), width=2*S)

        data_range = bin_edges[-1] - bin_edges[0]
        if data_range > 1e-6 and bin_edges[0] < 0 < bin_edges[-1]:
            zero_frac = (-bin_edges[0]) / data_range
            zero_x = ML + int(zero_frac * plot_w)
            zero_rgb = _hex_to_rgb(C_WARNING)
            for y_step in range(plot_h):
                alpha = int(180 * (1.0 - y_step / plot_h))
                draw.point((zero_x, MT + y_step), fill=(*zero_rgb, alpha))

        mean_val = float(flat.mean())
        std_val  = float(flat.std())
        zero_pct = float((flat == 0).mean() * 100)

        font_sm = _get_pil_font(9 * S)
        text_y = MT + plot_h + 6*S
        stats = [
            (ML,            f"μ={mean_val:.3f}",    C_TEXT_PRI),
            (ML + plot_w//3,f"σ={std_val:.3f}",     C_TEXT_SEC),
            (ML + 2*plot_w//3, f"zeros={zero_pct:.1f}%", C_TEXT_SEC),
        ]
        for tx, label, col in stats:
            draw.text((tx, text_y), label, fill=_hex_to_rgb(col), font=font_sm)

        return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

    def _render_metric_bar(
        self,
        value: float,
        label: str,
        colour: tuple[int, int, int],
        canvas_w: int = 220,
        canvas_h: int = 28
    ) -> Image.Image:
        S = SSAA
        W, H = canvas_w * S, canvas_h * S

        canvas = Image.new("RGBA", (W, H), (*_hex_to_rgb(C_BG_BASE), 0))
        draw = ImageDraw.Draw(canvas)

        PILL_W = int(W * 0.45)
        PILL_H = H - 8 * S
        PILL_X = 0
        PILL_Y = 4 * S
        radius = PILL_H // 2

        _draw_rounded_rect(
            draw,
            (PILL_X, PILL_Y, PILL_X + PILL_W, PILL_Y + PILL_H),
            radius=radius,
            fill=_hex_to_rgb(C_BG_FLOAT),
            outline=_hex_to_rgb(C_BORDER_SUB),
            width=1
        )

        fill_w = max(int(PILL_W * value), radius * 2) if value > 0 else 0
        if fill_w > 0:
            darker = _lerp_colour(colour, _hex_to_rgb(C_BG_DEEP), 0.4)
            grad = _make_gradient_rect(
                fill_w, PILL_H,
                colour_top=colour, colour_bottom=darker,
                alpha_top=255, alpha_bottom=220
            )
            fill_mask = Image.new("L", (PILL_W, PILL_H), 0)
            fm_draw = ImageDraw.Draw(fill_mask)
            _draw_rounded_rect(fm_draw, (0, 0, PILL_W - 1, PILL_H - 1), radius=radius, fill=255)
            canvas.paste(
                grad.convert("RGB"),
                (PILL_X, PILL_Y),
                mask=fill_mask.crop((0, 0, fill_w, PILL_H))
            )

            edge_x = PILL_X + fill_w
            glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            gd = ImageDraw.Draw(glow_layer)
            gd.line(
                [(edge_x, PILL_Y + 2*S), (edge_x, PILL_Y + PILL_H - 2*S)],
                fill=(*colour, 200), width=3*S
            )
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=4*S))
            canvas = Image.alpha_composite(canvas, glow_layer)

        draw = ImageDraw.Draw(canvas)
        font = _get_pil_font(9 * S)
        text_x = PILL_X + PILL_W + 8 * S
        draw.text(
            (text_x, PILL_Y + PILL_H // 2),
            label,
            fill=_hex_to_rgb(C_TEXT_PRI),
            font=font,
            anchor="lm"
        )

        return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

    def _render_snr_bar(self, snr_info: dict) -> Image.Image:
        value = snr_info.get("normalized", 0.0)
        snr   = snr_info.get("snr", 0.0)
        level = snr_info.get("level", "weak")
        colour_map = {"weak": C_CRITICAL, "moderate": C_WARNING, "strong": C_HEALTHY}
        colour = _hex_to_rgb(colour_map.get(level, C_TEXT_MUT))
        return self._render_metric_bar(value, f"SNR {snr:.2f}  [{level}]", colour)

    def _render_diversity_bar(self, diversity_info: dict) -> Image.Image:
        score = diversity_info.get("score", 0.0)
        pct   = diversity_info.get("score_pct", 0)
        level = diversity_info.get("level", "n/a")
        if level == "n/a":
            colour = _hex_to_rgb(C_TEXT_MUT)
        else:
            colour_map = {"low": C_CRITICAL, "moderate": C_WARNING, "high": C_HEALTHY}
            colour = _hex_to_rgb(colour_map.get(level, C_TEXT_MUT))
        return self._render_metric_bar(score, f"Div {pct}%  [{level}]", colour)

    def _render_health_strip(
        self,
        health_metrics: dict,
        input_stats: dict,
        canvas_w: int = 1100,
        canvas_h: int = 52
    ) -> Image.Image:
        w_curr = self.main_outer.winfo_width()
        canvas_w = w_curr if w_curr > 10 else 1100

        S = SSAA
        W, H = canvas_w * S, canvas_h * S

        canvas = Image.new("RGBA", (W, H), (*_hex_to_rgb(C_BG_RAISED), 255))
        draw = ImageDraw.Draw(canvas)

        draw.line([(0, H - 1), (W, H - 1)], fill=_hex_to_rgb(C_BORDER_SUB), width=1)

        font_label = _get_pil_font(8  * S)
        font_value = _get_pil_font(10 * S)

        if not health_metrics:
            draw.text((W // 2, H // 2), "Run a forward pass to populate metrics",
                      fill=_hex_to_rgb(C_TEXT_MUT), font=font_label, anchor="mm")
            return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

        def dead_colour(pct):
            t = min(pct / 50.0, 1.0)
            return _lerp_colour(_hex_to_rgb(C_HEALTHY), _hex_to_rgb(C_CRITICAL), t)

        dead_pct = health_metrics.get("dead_percent", 0.0)
        sat_info  = health_metrics.get("saturation", {})
        snr_info  = health_metrics.get("snr", {})
        div_info  = health_metrics.get("diversity", {})

        level_colour = {
            "healthy": C_HEALTHY, "mild": C_WARNING, "saturated": C_CRITICAL,
            "weak": C_CRITICAL, "moderate": C_WARNING, "strong": C_HEALTHY,
            "low": C_CRITICAL, "high": C_HEALTHY, "n/a": C_TEXT_MUT,
        }

        badges = [
            ("DEAD ReLU",  f"{dead_pct:.1f}%",                 dead_colour(dead_pct)),
            ("SATURATION", f"{sat_info.get('layer_score', 0):.1%}",
             _hex_to_rgb(level_colour.get(sat_info.get('level', 'healthy'), C_TEXT_MUT))),
            ("SNR",        f"{snr_info.get('snr', 0):.2f}",
             _hex_to_rgb(level_colour.get(snr_info.get('level', 'weak'), C_TEXT_MUT))),
            ("DIVERSITY",  f"{div_info.get('score_pct', 0)}%",
             _hex_to_rgb(level_colour.get(div_info.get('level', 'n/a'), C_TEXT_MUT))),
        ]

        badge_w = W // (len(badges) + 1)
        badge_h = H - 12 * S
        badge_y = 6 * S

        for i, (name, value, dot_colour) in enumerate(badges):
            bx = int((i + 0.5) * W / len(badges)) - badge_w // 2

            _draw_rounded_rect(
                draw,
                (bx, badge_y, bx + badge_w - 4*S, badge_y + badge_h),
                radius=badge_h // 2,
                fill=_hex_to_rgb(C_BG_FLOAT),
                outline=_hex_to_rgb(C_BORDER_SUB),
                width=1
            )

            dot_r   = 5 * S
            dot_cx  = bx + dot_r * 2 + 4*S
            dot_cy  = badge_y + badge_h // 2

            glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            gd = ImageDraw.Draw(glow_layer)
            gd.ellipse([dot_cx - dot_r*2, dot_cy - dot_r*2,
                        dot_cx + dot_r*2, dot_cy + dot_r*2],
                       fill=(*dot_colour, 160))
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=dot_r))
            canvas = Image.alpha_composite(canvas, glow_layer)

            draw = ImageDraw.Draw(canvas)
            draw.ellipse([dot_cx - dot_r, dot_cy - dot_r,
                          dot_cx + dot_r, dot_cy + dot_r],
                         fill=(*dot_colour, 255))

            text_x = dot_cx + dot_r + 6*S
            draw.text((text_x, dot_cy - 5*S), name,
                      fill=_hex_to_rgb(C_TEXT_MUT), font=font_label, anchor="lm")
            draw.text((text_x, dot_cy + 6*S), value,
                      fill=_hex_to_rgb(C_TEXT_PRI), font=font_value, anchor="lm")

        if input_stats:
            mean_ok = abs(input_stats.get('mean', 0)) <= 0.8
            std_ok  = input_stats.get('std', 1) >= 0.2
            if mean_ok and std_ok:
                inp_text, inp_col = "INPUT OK", C_TEXT_MUT
            else:
                inp_text, inp_col = "⚠ INPUT NORM", C_WARNING
            draw.text((W - 16*S, H // 2), inp_text,
                      fill=_hex_to_rgb(inp_col), font=font_label, anchor="rm")

        return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

    def _update_telemetry(self, layer_name: str | None, tensor_shape, input_stats: dict | None, health_metrics: dict | None):
        """Updates health status strip."""
        self.last_layer_name = layer_name
        self.last_health_metrics = health_metrics
        self.last_input_stats = input_stats
        
        # update health strip
        pil_img = self._render_health_strip(health_metrics, input_stats)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(pil_img.width, pil_img.height))
        self.health_strip_label.configure(image=ctk_img)
        
        # update inspector if it is open
        if self._inspector_visible and self._inspected_channel is not None:
            self._update_inspector()

    def _render_flow_chart(
        self,
        flow_data: dict[str, float],
        selected_layer: str | None,
        canvas_w: int = 900,
        canvas_h: int = 180
    ) -> Image.Image:
        S = SSAA
        W, H = canvas_w * S, canvas_h * S

        canvas = Image.new("RGBA", (W, H), (*_hex_to_rgb(C_BG_BASE), 255))
        draw = ImageDraw.Draw(canvas)

        if not flow_data:
            msg = "Run a forward pass to populate flow data"
            font = _get_pil_font(12 * S)
            draw.text((W // 2, H // 2), msg, fill=_hex_to_rgb(C_TEXT_MUT), font=font, anchor="mm")
            return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

        ML, MR, MT, MB = 55 * S, 15 * S, 20 * S, 40 * S
        plot_w = W - ML - MR
        plot_h = H - MT - MB

        layers = list(flow_data.keys())
        num_layers = len(layers)
        max_val = max(flow_data.values()) if flow_data else 1.0
        if max_val < 1e-6: max_val = 1.0

        title_font = _get_pil_font(10 * S)
        axis_font  = _get_pil_font(9 * S)
        label_font = _get_pil_font(8 * S)

        draw.text((10 * S, 15 * S), f"Network Activation Flow  ({num_layers} conv layers)",
                  fill=_hex_to_rgb(C_TEXT_SEC), font=title_font)
        draw.text((2 * S, H // 2), "Mean |Act|",
                  fill=_hex_to_rgb(C_TEXT_SEC), font=axis_font, anchor="lm")

        grid_colour = (*_hex_to_rgb(C_BORDER_SUB), 140)
        for pct in (0.25, 0.5, 0.75):
            y = MT + int(plot_h * (1 - pct))
            for x in range(ML, W - MR, 10 * S):
                draw.line([(x, y), (x + 5 * S, y)], fill=grid_colour, width=1)

        if num_layers > 0:
            slot_w = plot_w / num_layers
            stem_colour = (*_hex_to_rgb(C_BORDER_SUB), 200)

            for i, layer in enumerate(layers):
                val = flow_data[layer]
                bar_h = max(2 * S, int((val / max_val) * plot_h))

                cx = ML + int(i * slot_w + slot_w / 2)
                y1 = MT + plot_h - bar_h
                y2 = MT + plot_h

                if i < num_layers * 0.33:   color = _hex_to_rgb(C_HEALTHY)
                elif i < num_layers * 0.66: color = _hex_to_rgb(C_INFO)
                else:                       color = _hex_to_rgb(C_WARNING)

                draw.line([(cx, y1), (cx, y2)], fill=stem_colour, width=2 * S)

                dot_r = 4 * S
                
                glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                gd = ImageDraw.Draw(glow_layer)
                gd.ellipse([cx - dot_r*2, y1 - dot_r*2, cx + dot_r*2, y1 + dot_r*2], fill=(*color, 120))
                glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=dot_r))
                canvas = Image.alpha_composite(canvas, glow_layer)
                
                draw = ImageDraw.Draw(canvas)
                draw.ellipse([cx - dot_r, y1 - dot_r, cx + dot_r, y1 + dot_r], fill=(*color, 255))

                if layer == selected_layer:
                    sel_r = 6 * S
                    draw.ellipse([cx - sel_r, y1 - sel_r, cx + sel_r, y1 + sel_r], outline=(255, 255, 255), width=2*S)
                    draw.polygon([(cx, y1 - 12*S), (cx - 6*S, y1 - 20*S), (cx + 6*S, y1 - 20*S)], fill=(255, 255, 255))

                lbl = layer if len(layer) <= 10 else layer[:9] + "…"
                
                text_img = Image.new("RGBA", (80*S, 20*S), (0,0,0,0))
                td = ImageDraw.Draw(text_img)
                td.text((0, 10*S), lbl, font=label_font, fill=_hex_to_rgb(C_TEXT_MUT), anchor="lm")
                
                text_img = text_img.rotate(45, resample=Image.BICUBIC, expand=True)
                
                canvas.paste(text_img, (cx - 10*S, y2 + 5*S), text_img)

        return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

    def _render_correlation_matrix(
        self,
        sim_matrix: torch.Tensor,
        layer_name: str,
        sorted_indices: torch.Tensor,
        frame_age_seconds: float,
        canvas_w: int = 400,
        canvas_h: int = 400
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

        sim_np = display_sim.cpu().numpy()
        
        sim_norm = (sim_np + 1.0) / 2.0
        
        turbo_anchors = [
            (0.0, "#30123B"), (0.1, "#4662D7"), (0.3, "#36AAF9"),
            (0.5, "#1AE4B6"), (0.7, "#72FE5E"), (0.8, "#C6ED34"),
            (0.9, "#FABA39"), (0.95, "#F66C19"), (1.0, "#7A0403")
        ]
        
        heatmap_rgb = _apply_colormap(sim_norm, turbo_anchors)
        heatmap_img = Image.fromarray(heatmap_rgb, mode="RGB")

        PAD = 40 * S
        plot_size = min(W - 2 * PAD, H - 2 * PAD)
        
        heatmap_scaled = heatmap_img.resize((plot_size, plot_size), Image.NEAREST)

        start_x = (W - plot_size) // 2
        start_y = (H - plot_size) // 2 + 10 * S

        canvas.paste(heatmap_scaled, (start_x, start_y))
        
        _draw_rounded_rect(draw, (start_x - 1, start_y - 1, start_x + plot_size, start_y + plot_size), 
                           radius=0, outline=_hex_to_rgb(C_BORDER_SUB), width=1)

        draw.text((start_x, start_y - 15 * S), f"Filter Correlation: {layer_name}",
                  fill=_hex_to_rgb(C_TEXT_PRI), font=font)
                  
        if frame_age_seconds > 0:
            draw.text((start_x + plot_size, start_y - 15 * S), f"Age: {frame_age_seconds:.1f}s",
                      fill=_hex_to_rgb(C_TEXT_MUT), font=font, anchor="ra")

        return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")

    def _on_tab_change(self):
        active_tab = self.tabview.get()
        if active_tab == "Flow Chart":
            self._refresh_flow_tab()
        elif active_tab == "Correlation":
            self._refresh_correlation_tab()

    def _refresh_flow_tab(self):
        if not self._last_flow_data:
            return
            
        self._flow_frame_count += 1
        is_live = self.camera_active and not self.frozen_frame
        
        if not is_live or self._flow_frame_count % 3 == 0:
            pil_img = self._render_flow_chart(self._last_flow_data, self.sidebar.config.get("layer"))
            tw, th = self.tabview.winfo_width(), self.tabview.winfo_height()
            tw = tw if tw > 10 else 900
            th = th if th > 10 else 180
            scaled_h = min(th - 40, int(pil_img.height * (tw / pil_img.width)))
            if scaled_h < 10: scaled_h = 180
            
            pil_resized = pil_img.resize((tw - 20, scaled_h), Image.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=pil_resized, dark_image=pil_resized, size=(tw - 20, scaled_h))
            self.flow_tab_label.configure(image=ctk_img)

    def _refresh_correlation_tab(self):
        if self._last_sim_matrix is None:
            return
            
        self._corr_matrix_frame_count += 1
        is_live = self.camera_active and not self.frozen_frame
        
        do_render = True
        if is_live and self._last_sim_matrix.shape[0] > 64:
            if self._corr_matrix_frame_count % 10 != 0:
                do_render = False
                
        if do_render:
            age = time.time() - self._corr_matrix_last_update_time
            pil_img = self._render_correlation_matrix(
                self._last_sim_matrix, 
                self.sidebar.config.get("layer"), 
                self._corr_matrix_sorted_indices,
                age if is_live else 0.0
            )
            
            tw, th = self.tabview.winfo_width(), self.tabview.winfo_height()
            plot_size = min(tw - 20, th - 40)
            if plot_size < 10: plot_size = 400
            
            pil_resized = pil_img.resize((plot_size, plot_size), Image.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=pil_resized, dark_image=pil_resized, size=(plot_size, plot_size))
            self.corr_tab_label.configure(image=ctk_img)

    # ------------------------------------------------------------------ #
    #  Export                                                              #
    # ------------------------------------------------------------------ #

    def on_export_diagnostics(self):
        """Dumps current telemetry and layer health into a .txt report."""
        if not self.last_layer_name and not self.last_health_metrics:
            self.sidebar.telemetry_text.configure(
                text="No active data to export!\n" + self.sidebar.telemetry_text.cget("text"))
            return

        try:
            report_content = generate_report(
                self.last_layer_name, 
                self.last_health_metrics, 
                self.last_input_stats, 
                self.sidebar.config["dead_threshold"]
            )
            save_report(report_content)

            print("Saved diagnostic_report.txt")
            old_text = self.sidebar.export_btn.cget("text")
            self.sidebar.export_btn.configure(text="✔ Saved to disk!")
            self.after(2000, lambda: self.sidebar.export_btn.configure(text=old_text))
        except Exception as e:
            print(f"Export failed: {e}")

    # ------------------------------------------------------------------ #
    #  Static visualization                                                #
    # ------------------------------------------------------------------ #

    def on_visualize(self):
        """Coordinates the hooking phase and renders results to the UI (static images)."""
        if self.camera_active:
            self.toggle_camera()

        if not self.current_model:
            messagebox.showwarning("Warning", "Please select a model first.")
            return
        if not self.image_path:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return

        loading_label = ctk.CTkLabel(self.grid_view, text="Processing forward pass...", font=ctk.CTkFont(size=16))
        loading_label.pack(pady=50)
        self.update()

        try:
            images, dynamic_title, _, health_metrics, layer_flow_data, sim_matrix = self._run_visualization_pipeline(self.image_path, "Static Upload")
            loading_label.destroy()
            self.grid_view.update(images, dynamic_title, force_rebuild=True, health_metrics=health_metrics)
            
            selected_layer = self.sidebar.config["layer"]
            if layer_flow_data is not None:
                self._last_flow_data = layer_flow_data
            if sim_matrix is not None:
                self._last_sim_matrix = sim_matrix
                self._corr_matrix_last_update_time = time.time()
                
            self._on_tab_change()
        except Exception as e:
            loading_label.destroy()
            messagebox.showerror("Error", f"Visualization pipeline failed:\n{e}")

    # ------------------------------------------------------------------ #
    #  Live Camera Threading Pipeline                                      #
    # ------------------------------------------------------------------ #

    def _capture_thread_func(self):
        """Dedicated thread for grabbing frames from OpenCV to avoid blocking."""
        while not self._inference_stop_event.is_set():
            if self.camera is not None and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    with self._frame_lock:
                        self._latest_frame = frame

    def _inference_thread_func(self):
        """Dedicated thread for executing PyTorch forwards in a tight loop."""
        while not self._inference_stop_event.is_set():
            frame = None
            if self.frozen_frame is not None:
                frame = self.frozen_frame
            else:
                with self._frame_lock:
                    if self._latest_frame is not None:
                        frame = self._latest_frame.copy()

            if frame is not None:
                try:
                    source = "Frozen Frame" if self.frozen_frame is not None else "Live Feed"
                    images, dynamic_title, cached_redundancy, health_metrics, layer_flow_data, sim_matrix = self._run_visualization_pipeline(
                        frame, source, 
                        cached_redundancy=self._cached_redundancy, 
                        frame_count=self._frame_count
                    )
                    
                    self._cached_redundancy = cached_redundancy
                    self._frame_count += 1
                    
                    try:
                        self._result_queue.put_nowait((images, dynamic_title, cached_redundancy, health_metrics, layer_flow_data, sim_matrix))
                    except queue.Full:
                        pass
                except Exception as e:
                    pass

    def _poll_results(self):
        """Main thread loop that displays computed UI updates without blocking."""
        if not self.camera_active:
            return

        try:
            images, dynamic_title, cached_redundancy, health_metrics, layer_flow_data, sim_matrix = self._result_queue.get_nowait()
            self.grid_view.update(images, dynamic_title, force_rebuild=False, health_metrics=health_metrics)
            
            selected_layer = self.sidebar.config["layer"]
            
            if layer_flow_data is not None:
                self._last_flow_data = layer_flow_data
            if sim_matrix is not None:
                self._last_sim_matrix = sim_matrix
                self._corr_matrix_last_update_time = time.time()
                
            self._on_tab_change()
            
        except queue.Empty:
            pass

        self.after(int(self.sidebar.config["speed"]), self._poll_results)

    def toggle_camera(self):
        """Turns the live webcam feed on or off using threaded approach."""
        if not self.current_model:
            messagebox.showwarning("Warning", "Please select a model first.")
            return
        selected_layer = self.sidebar.config["layer"]
        if not selected_layer or selected_layer == "Load model first...":
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
                    messagebox.showerror("Camera Error", f"Cannot access webcam:\n{e}")
                    return

            self.sidebar.camera_btn.configure(text="Stop Live Camera", fg_color="#c92a2a", hover_color="#a61e1e")
            self.sidebar.cam_status_label.configure(text="Camera Active (0)", text_color="green")
            self.sidebar.freeze_btn.configure(state="normal")
            
            # Clear old state
            self.frozen_frame = None
            self._frame_count = 0
            self._cached_redundancy = None
            self._latest_frame = None
            
            while not self._result_queue.empty():
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    pass

            self._inference_stop_event.clear()
            self._capture_thread = threading.Thread(target=self._capture_thread_func, daemon=True)
            self._inference_thread = threading.Thread(target=self._inference_thread_func, daemon=True)
            self._capture_thread.start()
            self._inference_thread.start()
            
            self._poll_results()
        else:
            self.sidebar.camera_btn.configure(text="Start Live Camera", fg_color="#2b8a3e", hover_color="#237032")
            self.sidebar.cam_status_label.configure(text="Camera Offline", text_color="gray")
            self.sidebar.freeze_btn.configure(state="disabled")
            
            self._inference_stop_event.set()
            
            if self._capture_thread is not None:
                self._capture_thread.join(timeout=1.0)
            if self._inference_thread is not None:
                self._inference_thread.join(timeout=1.0)

            if self.frozen_frame is not None:
                self.frozen_frame = None
                
            if self.camera is not None:
                self.camera.release()
                self.camera = None

    def toggle_freeze(self):
        """Locks the current live frame in place by bypassing hardware read()."""
        if self.frozen_frame is None:
            self.sidebar.cam_status_label.configure(text="Camera [FROZEN]", text_color="#6366f1")
            self.sidebar.freeze_btn.configure(text="▶ Unfreeze Frame", fg_color="#10b981", hover_color="#059669")
            with self._frame_lock:
                if self._latest_frame is not None:
                    self.frozen_frame = self._latest_frame.copy()
        else:
            self.frozen_frame = None
            self.sidebar.cam_status_label.configure(text="Camera Active", text_color="green")
            self.sidebar.freeze_btn.configure(text="❄ Freeze Frame", fg_color="#6366f1", hover_color="#4f46e5")

    # ------------------------------------------------------------------ #
    #  Inspector Panel Logic                                               #
    # ------------------------------------------------------------------ #

    def _on_channel_click(self, channel_idx: int):
        self._inspected_channel = channel_idx
        self._inspected_layer = self.sidebar.config.get("layer")
        
        if not self._inspector_visible:
            self.inspector_frame.grid(row=0, column=2, sticky="nsew", padx=(0, 10), pady=10)
            self._inspector_visible = True
            
        self._update_inspector()

    def _close_inspector(self):
        self._inspector_visible = False
        self._inspected_channel = None
        self.inspector_frame.grid_forget()
        self.grid_view.clear_selection()

    def _render_spatial_bias(self, tensor: torch.Tensor, canvas_w: int = 280, canvas_h: int = 140) -> Image.Image:
        """Renders crosshairs on the spatial maximum of the single channel tensor."""
        if tensor is None or len(tensor.shape) < 2:
            return None
            
        t_2d = tensor.detach().cpu().numpy()
        h, w = t_2d.shape
        if h == 0 or w == 0:
            return None

        S = SSAA
        W, H = canvas_w * S, canvas_h * S
        
        canvas = Image.new("RGBA", (W, H), (*_hex_to_rgb(C_BG_RAISED), 255))
        
        try:
            t_min, t_max = t_2d.min(), t_2d.max()
            if t_max > t_min:
                t_norm = (t_2d - t_min) / (t_max - t_min)
            else:
                t_norm = np.zeros_like(t_2d)

            magma_anchors = [
                (0.0, "#000004"), (0.2, "#2b115f"), (0.4, "#721f81"),
                (0.6, "#c33d69"), (0.8, "#fca50a"), (1.0, "#fcfdbf")
            ]
            
            heat_rgb = _apply_colormap(t_norm, magma_anchors)
            heat_img = Image.fromarray(heat_rgb, mode="RGB")
            
            heat_scaled = heat_img.resize((W, H), Image.BILINEAR)
            heat_layer = heat_scaled.copy()
            heat_layer.putalpha(150)
            
            canvas = Image.alpha_composite(canvas, heat_layer)
            
            y_i, x_i = np.unravel_index(np.argmax(t_norm, axis=None), t_norm.shape)
            x_max = int((x_i + 0.5) / w * W)
            y_max = int((y_i + 0.5) / h * H)
            
            draw = ImageDraw.Draw(canvas)
            cross_color = (255, 255, 255, 180)
            draw.line([(x_max, 0), (x_max, H)], fill=cross_color, width=1 * S)
            draw.line([(0, y_max), (W, y_max)], fill=cross_color, width=1 * S)
            
            rad = 3 * S
            draw.ellipse([x_max - rad, y_max - rad, x_max + rad, y_max + rad], 
                         fill=(255, 255, 0, 255), outline=(0, 0, 0, 100), width=1)
            
            return canvas.resize((canvas_w, canvas_h), Image.LANCZOS).convert("RGB")
        except Exception:
            return None

    def _update_inspector(self):
        if not self._inspector_visible or self._inspected_channel is None:
            return
            
        self.inspector_title.configure(text=f"Channel {self._inspected_channel}")
        
        # 1. Image update
        cell_data = self.grid_view.get_cell_data(self._inspected_channel)
        if cell_data:
            img, label, diag = cell_data
            if img:
                res_img = img.resize((280, 280), Image.NEAREST)
                ctk_im = ctk.CTkImage(light_image=res_img, dark_image=res_img, size=(280, 280))
                self.inspector_map_label.configure(image=ctk_im)
                
        # 2. Stats
        self.inspector_stats["Layer"].configure(text=self._inspected_layer or "-")
        self.inspector_stats["Channel"].configure(text=str(self._inspected_channel))
        
        t_raw = None
        if self._inspected_layer and self._inspected_layer in self.hook_engine.features:
            t = self.hook_engine.features[self._inspected_layer]
            if len(t.shape) == 4:
                t = t.squeeze(0)
            if self._inspected_channel < t.shape[0]:
                t_raw = t[self._inspected_channel]
                
        if t_raw is not None:
            mean_v = t_raw.mean().item()
            std_v = t_raw.std().item()
            max_v = t_raw.max().item()
            min_v = t_raw.min().item()
            zero_pc = (t_raw == 0).sum().item() / t_raw.numel() * 100
            
            self.inspector_stats["Act Range"].configure(text=f"[{min_v:.2f}, {max_v:.2f}]")
            self.inspector_stats["Mean / Std"].configure(text=f"{mean_v:.2f} / {std_v:.2f}")
            self.inspector_stats["Dead"].configure(text=f"{zero_pc:.1f}%")
            
            # Histogram
            hist_img = self._render_histogram(t_raw.flatten(), canvas_w=280, canvas_h=100)
            if hist_img:
                h_ctk = ctk.CTkImage(light_image=hist_img, dark_image=hist_img, size=(280, 100))
                self.inspector_hist_label.configure(image=h_ctk)
                
            # Bias
            bias_img = self._render_spatial_bias(t_raw)
            if bias_img:
                b_ctk = ctk.CTkImage(light_image=bias_img, dark_image=bias_img, size=(280, 140))
                self.inspector_bias_label.configure(image=b_ctk, text="")
        else:
            self.inspector_hist_label.configure(image="")
            self.inspector_bias_label.configure(image="")

        # Ranking and Similarity
        rank_idx = -1
        if self._corr_matrix_sorted_indices is not None:
            try:
                rank_idx = (self._corr_matrix_sorted_indices == self._inspected_channel).nonzero(as_tuple=True)[0].item() + 1
            except Exception:
                pass
        
        self.inspector_stats["EMA Rank"].configure(text=f"#{rank_idx}" if rank_idx > 0 else "-")
        
        if self._last_sim_matrix is not None and self._inspected_channel < self._last_sim_matrix.shape[0]:
            row = self._last_sim_matrix[self._inspected_channel]
            row_clone = row.clone()
            row_clone[self._inspected_channel] = -2.0 # exclude self
            top_v, top_i = torch.topk(row_clone, 3)
            
            for rank_i in range(3):
                sim_ch = top_i[rank_i].item()
                sim_score = top_v[rank_i].item()
                lbl_img, lbl_txt = self.inspector_similar_labels[rank_i]
                
                c_data = self.grid_view.get_cell_data(sim_ch)
                if c_data and c_data[0]:
                    s_im = c_data[0].resize((70, 70), Image.NEAREST)
                    c_im = ctk.CTkImage(light_image=s_im, dark_image=s_im, size=(70, 70))
                    lbl_img.configure(image=c_im)
                    lbl_txt.configure(text=f"Ch {sim_ch}\n{sim_score:.2f}")
                else:
                    lbl_img.configure(image="")
                    lbl_txt.configure(text="-")

