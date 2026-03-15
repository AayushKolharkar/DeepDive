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
from PIL import Image
from tkinter import filedialog, messagebox

from ui.sidebar import Sidebar
from ui.grid_view import GridView
from core.model_loader import load_model
from core.hook_engine import HookEngine
from core.gradcam import compute_gradcam
from processing.tensor_to_image import process_tensor_to_images
from export.reporter import generate_report, save_report
from .theme import *

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

    def _render_health_strip(self, health_metrics: dict | None, input_stats: dict | None) -> Image.Image:
        w_curr = self.main_outer.winfo_width()
        w = w_curr if w_curr > 10 else 1100
        h = 52
        
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        # bg: #1a1a2e (26, 26, 46 BGR -> 46, 26, 26)
        bg_rgb = tuple(int(C_BG_RAISED.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        canvas[:] = (bg_rgb[2], bg_rgb[1], bg_rgb[0])
        
        # bottom border: #2a2a45 -> (69, 42, 42)
        bb_rgb = tuple(int(C_BORDER_SUB.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        cv2.line(canvas, (0, h-1), (w, h-1), (bb_rgb[2], bb_rgb[1], bb_rgb[0]), 1)
        
        if not health_metrics:
            text_rgb = tuple(int(C_TEXT_MUT.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            t_w, _ = cv2.getTextSize("Run a forward pass to populate metrics", cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
            cv2.putText(canvas, "Run a forward pass to populate metrics", (w//2 - t_w//2, h//2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (text_rgb[2], text_rgb[1], text_rgb[0]), 1)
            return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            
        def hex_to_bgr(h_str):
            h_str = h_str.lstrip('#')
            return tuple(int(h_str[i:i+2], 16) for i in (4, 2, 0))
            
        c_green = hex_to_bgr(C_HEALTHY)
        c_yellow = hex_to_bgr(C_WARNING)
        c_red = hex_to_bgr(C_CRITICAL)
        
        # DEAD
        dead_pct = health_metrics.get("dead_percent", 0.0)
        c_dead = c_green if dead_pct < 10 else (c_yellow if dead_pct < 30 else c_red)
        
        # SAT
        sat = health_metrics.get("saturation", {})
        sat_score = sat.get("layer_score", 0.0)
        sat_lvl = sat.get("level", "normal")
        c_sat = c_green if sat_lvl == "normal" else (c_yellow if sat_lvl == "mild" else c_red)
        
        # SNR
        snr_info = health_metrics.get("snr", {})
        snr_val = snr_info.get("snr", 0.0)
        snr_lvl = snr_info.get("level", "weak")
        c_snr = c_red if snr_lvl == "weak" else (c_yellow if snr_lvl == "moderate" else c_green)
        
        # DIV
        div_info = health_metrics.get("diversity", {})
        div_val = div_info.get("score_pct", 0)
        div_lvl = div_info.get("level", "low")
        c_div = c_red if div_lvl == "low" else (c_yellow if div_lvl == "moderate" else c_green)
        
        badges = [
            (c_dead, f"DEAD  {dead_pct:.1f}%"),
            (c_sat, f"SAT   {sat_score*100:.1f}%"),
            (c_snr, f"SNR   {snr_val:.2f}"),
            (c_div, f"DIV   {div_val}%")
        ]
        
        text_sec = hex_to_bgr(C_TEXT_SEC)
        text_pri = hex_to_bgr(C_TEXT_PRI)
        
        spacing = w // 5
        base_x = spacing // 2
        
        for i, (color, txt) in enumerate(badges):
            x = base_x + i * spacing
            cv2.circle(canvas, (x, h//2), 5, color, -1)
            
            parts = txt.split("   ")
            name = parts[0] + " "
            val = parts[1] if len(parts) > 1 else ""
            
            cv2.putText(canvas, name, (x + 15, h//2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_sec, 1)
            n_w, _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
            cv2.putText(canvas, val, (x + 15 + n_w, h//2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_pri, 1)
            
        if input_stats and input_stats.get('std') is not None and input_stats['std'] != 0.0:
            if abs(input_stats['mean']) > 0.8 or input_stats['std'] < 0.2:
                in_txt, in_col = "⚠ INPUT NORM", c_yellow
            else:
                in_txt, in_col = "INPUT OK", text_sec
        else:
            in_txt, in_col = "INPUT OK", text_sec
            
        cv2.putText(canvas, in_txt, (w - 120, h//2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, in_col, 1)
        
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        return pil_img

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

    def _render_flow_chart(self, flow_data: dict[str, float], selected_layer: str | None) -> Image.Image:
        w, h = 900, 180
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (26, 15, 15)

        if not flow_data:
            cv2.putText(canvas, "Run a forward pass to populate flow data", (w // 2 - 150, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (161, 161, 170), 1)
            return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

        left_m, right_m, top_m, bot_m = 55, 15, 20, 40
        plot_w = w - left_m - right_m
        plot_h = h - top_m - bot_m

        layers = list(flow_data.keys())
        num_layers = len(layers)
        max_val = max(flow_data.values()) if flow_data else 1.0
        if max_val < 1e-6: max_val = 1.0

        cv2.putText(canvas, f"Network Activation Flow  ({num_layers} conv layers)", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 161, 161), 1)

        for pct in [0.25, 0.5, 0.75]:
            y = top_m + int(plot_h * (1 - pct))
            for x in range(left_m, w - right_m, 10):
                cv2.line(canvas, (x, y), (x + 5, y), (85, 51, 51), 1)
        cv2.putText(canvas, "Mean |Act|", (2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 161, 161), 1)

        if num_layers > 0:
            slot_w = plot_w / num_layers
            bar_w = int(slot_w * 0.7)
            if bar_w < 1: bar_w = 1

            for i, layer in enumerate(layers):
                val = flow_data[layer]
                bar_h = max(2, int((val / max_val) * plot_h))

                x_center = left_m + int(i * slot_w + slot_w / 2)
                x1 = x_center - bar_w // 2
                x2 = x1 + bar_w
                y1 = top_m + plot_h - bar_h
                y2 = top_m + plot_h

                if i < num_layers * 0.33: color = (128, 222, 74)
                elif i < num_layers * 0.66: color = (21, 204, 250)
                else: color = (113, 113, 248)

                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)

                if layer == selected_layer:
                    cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 1)
                    pts = np.array([[x_center, y1 - 8], [x_center - 4, y1 - 4], [x_center + 4, y1 - 4]], np.int32)
                    cv2.fillPoly(canvas, [pts], (255, 255, 255))

                lbl = layer if len(layer) <= 10 else layer[:9] + "…"
                text_sz, _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                tw, th = text_sz
                sub_w, sub_h = int(tw * 1.5) + 10, int(tw * 1.5) + 10
                sub_canvas = np.zeros((sub_h, sub_w, 3), dtype=np.uint8)
                cv2.putText(sub_canvas, lbl, (5, sub_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (170, 161, 161), 1)
                M = cv2.getRotationMatrix2D((5, sub_h // 2), 45, 1)
                sub_rot = cv2.warpAffine(sub_canvas, M, (sub_w, sub_h))
                
                cx = x_center
                cy = y2 + 5
                mask = cv2.cvtColor(sub_rot, cv2.COLOR_BGR2GRAY) > 0
                for dy in range(sub_h):
                    for dx in range(sub_w):
                        if mask[dy, dx]:
                            py = cy + dy - sub_h // 2
                            px = cx + dx - 5
                            if 0 <= py < h and 0 <= px < w:
                                canvas[py, px] = sub_rot[dy, dx]

        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    def _render_correlation_matrix(self, sim_matrix: torch.Tensor, layer_name: str, sorted_indices: torch.Tensor, frame_age_seconds: float) -> Image.Image:
        w, h = 400, 400
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (26, 15, 15)
        
        if sim_matrix is None:
            cv2.putText(canvas, "No similarity data available.", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

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
        sim_norm = ((sim_np + 1.0) / 2.0 * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(sim_norm, cv2.COLORMAP_TURBO)
        
        pad = 40
        plot_size = min(w - 2 * pad, h - 2 * pad)
        heatmap_resized = cv2.resize(heatmap, (plot_size, plot_size), interpolation=cv2.INTER_NEAREST)
        
        start_x = (w - plot_size) // 2
        start_y = (h - plot_size) // 2 + 10
        canvas[start_y:start_y+plot_size, start_x:start_x+plot_size] = heatmap_resized
        
        cv2.putText(canvas, f"Filter Correlation: {layer_name}", (start_x, start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        if frame_age_seconds > 0:
            cv2.putText(canvas, f"Age: {frame_age_seconds:.1f}s", (start_x + plot_size - 60, start_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            
        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

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

    def _render_spatial_bias(self, tensor: torch.Tensor) -> Image.Image:
        """Renders crosshairs on the spatial maximum of the single channel tensor."""
        if tensor is None or len(tensor.shape) < 2:
            return None
            
        t_2d = tensor.detach().cpu().numpy()
        h, w = t_2d.shape
        
        if h == 0 or w == 0:
            return None
            
        canvas = np.zeros((140, 280, 3), dtype=np.uint8)
        bg_rgb = tuple(int(C_BG_RAISED.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        canvas[:] = (bg_rgb[2], bg_rgb[1], bg_rgb[0])
        
        try:
            heat = cv2.resize(t_2d, (280, 140), interpolation=cv2.INTER_LINEAR)
            heat_norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_MAGMA)
            
            canvas = cv2.addWeighted(canvas, 0.4, heat_color, 0.6, 0)
            
            y_max, x_max = np.unravel_index(np.argmax(heat, axis=None), heat.shape)
            cv2.line(canvas, (x_max, 0), (x_max, 140), (255, 255, 255), 1)
            cv2.line(canvas, (0, y_max), (280, y_max), (255, 255, 255), 1)
            cv2.circle(canvas, (x_max, y_max), 3, (0, 255, 255), -1)
            
            return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
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

