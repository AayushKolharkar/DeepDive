"""
Sidebar widget panel and its controls.
"""
from __future__ import annotations
import customtkinter as ctk
from .theme import *


class Sidebar(ctk.CTkScrollableFrame):
    """
    Left sidebar frame containing all controls, scrollable for smaller displays.
    """
    def __init__(self, parent,
                 on_model_select,
                 on_upload,
                 on_camera_toggle,
                 on_freeze_toggle,
                 on_visualize,
                 on_export,
                 on_speed_change,
                 **kwargs):
        super().__init__(parent, fg_color=C_BG_BASE, corner_radius=0, **kwargs)

        # Callbacks
        self.on_model_select_callback  = on_model_select
        self.on_upload_callback        = on_upload
        self.on_camera_toggle_callback = on_camera_toggle
        self.on_freeze_toggle_callback = on_freeze_toggle
        self.on_visualize_callback     = on_visualize
        self.on_export_callback        = on_export
        # FIX #1: on_speed_change is passed as None from app.py.
        # We store it but guard every call — speed is already read via
        # sidebar.config['speed'] so the callback is optional.
        self.on_speed_change_callback  = on_speed_change

        # Configuration vars
        self.mode_var    = ctk.StringVar(value="Layer Mode")
        self.heatmap_var = ctk.BooleanVar(value=True)
        self.gradcam_var = ctk.BooleanVar(value=False)
        self._current_speed = 200

        self._setup_ui()

    # ------------------------------------------------------------------ #
    #  Section helper                                                      #
    # ------------------------------------------------------------------ #

    def _sidebar_section(self, parent, title: str) -> ctk.CTkFrame:
        """Creates a visually distinct sidebar section card."""
        frame = ctk.CTkFrame(
            parent,
            fg_color=C_BG_RAISED,
            border_width=1,
            border_color=C_BORDER_SUB,
            corner_radius=8,
        )
        if title:
            ctk.CTkLabel(
                frame,
                text=title.upper(),
                font=ctk.CTkFont(size=9, weight="bold"),
                text_color=C_TEXT_MUT,
            ).pack(anchor="w", padx=12, pady=(8, 4))
        return frame

    # ------------------------------------------------------------------ #
    #  UI build                                                            #
    # ------------------------------------------------------------------ #

    def _setup_ui(self):
        # ── Section 1: Header ────────────────────────────────────────────
        sec_header = ctk.CTkFrame(self, fg_color=C_BG_BASE, corner_radius=0)
        sec_header.pack(fill="x", padx=10, pady=(15, 10))

        ctk.CTkLabel(
            sec_header,
            text="CNN Visualizer\nPro Edition",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=C_TEXT_PRI,
            justify="left",
        ).pack(anchor="w", padx=4)

        ctk.CTkLabel(
            sec_header,
            text="Activation Inspector",
            font=ctk.CTkFont(size=11),
            text_color=C_TEXT_SEC,
        ).pack(anchor="w", padx=4)

        self.status_badge = ctk.CTkLabel(
            sec_header,
            text="● IDLE",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=C_TEXT_MUT,
        )
        self.status_badge.pack(anchor="w", padx=4, pady=(6, 0))

        # ── Section 2: Model ─────────────────────────────────────────────
        sec_model = self._sidebar_section(self, "Model")
        sec_model.pack(fill="x", padx=10, pady=6)

        self.model_dropdown = ctk.CTkComboBox(
            sec_model,
            values=["ResNet18", "VGG16", "AlexNet"],
            command=self.on_model_select_callback,
        )
        self.model_dropdown.pack(fill="x", padx=12, pady=(0, 10))
        self.model_dropdown.set("Select Model")

        self.mode_toggle = ctk.CTkSegmentedButton(
            sec_model,
            values=["Layer Mode", "Channel Mode"],
            variable=self.mode_var,
            command=self._on_mode_switch,
            selected_color=C_ACCENT,
            selected_hover_color=C_BG_FLOAT,
            unselected_color=C_BG_RAISED,
        )
        self.mode_toggle.pack(fill="x", padx=12, pady=(0, 10))

        # ── Section 3: Layer / Channel ────────────────────────────────────
        sec_layer = self._sidebar_section(self, "Target")
        sec_layer.pack(fill="x", padx=10, pady=6)

        self.layer_label = ctk.CTkLabel(
            sec_layer,
            text="Select Conv Layer:",
            font=ctk.CTkFont(size=12),
            text_color=C_TEXT_PRI,
        )
        self.layer_label.pack(anchor="w", padx=12)

        self.layer_dropdown = ctk.CTkComboBox(
            sec_layer, values=["Load model first..."]
        )
        self.layer_dropdown.pack(fill="x", padx=12, pady=(0, 10))

        self.channel_label = ctk.CTkLabel(
            sec_layer,
            text="Enter Channel Index:",
            font=ctk.CTkFont(size=12),
            text_color=C_TEXT_PRI,
        )
        self.channel_entry = ctk.CTkEntry(
            sec_layer, fg_color=C_BG_BASE, border_color=C_BORDER_SUB
        )
        self.channel_entry.insert(0, "0")

        # Channel widgets start hidden (Layer Mode is default)
        self.channel_label.pack_forget()
        self.channel_entry.pack_forget()

        # ── Section 4: Input ─────────────────────────────────────────────
        sec_input = self._sidebar_section(self, "Input")
        sec_input.pack(fill="x", padx=10, pady=6)

        self.upload_btn = ctk.CTkButton(
            sec_input,
            text="Browse Image",
            fg_color=C_ACCENT,
            hover_color=C_BG_FLOAT,
            command=self.on_upload_callback,
        )
        self.upload_btn.pack(fill="x", padx=12, pady=(0, 4))

        self.img_path_label = ctk.CTkLabel(
            sec_input,
            text="No image selected",
            text_color=C_TEXT_SEC,
            font=ctk.CTkFont(size=11),
        )
        self.img_path_label.pack(anchor="w", padx=12, pady=(0, 10))

        # ── Section 5: Camera ─────────────────────────────────────────────
        sec_cam = self._sidebar_section(self, "Live Camera")
        sec_cam.pack(fill="x", padx=10, pady=6)

        cam_btns = ctk.CTkFrame(sec_cam, fg_color="transparent")
        cam_btns.pack(fill="x", padx=12, pady=(0, 4))
        cam_btns.grid_columnconfigure(0, weight=1)
        cam_btns.grid_columnconfigure(1, weight=1)

        self.camera_btn = ctk.CTkButton(
            cam_btns,
            text="Start",
            fg_color=C_HEALTHY,
            hover_color="#22c55e",
            command=self.on_camera_toggle_callback,
        )
        self.camera_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")

        self.freeze_btn = ctk.CTkButton(
            cam_btns,
            text="Freeze",
            fg_color=C_ACCENT,
            hover_color=C_BG_FLOAT,
            command=self.on_freeze_toggle_callback,
            state="disabled",
        )
        self.freeze_btn.grid(row=0, column=1, padx=(4, 0), sticky="ew")

        self.cam_status_label = ctk.CTkLabel(
            sec_cam,
            text="Camera Offline",
            text_color=C_TEXT_MUT,
            font=ctk.CTkFont(size=11),
        )
        self.cam_status_label.pack(anchor="w", padx=12)

        speed_lbl_frame = ctk.CTkFrame(sec_cam, fg_color="transparent")
        speed_lbl_frame.pack(fill="x", padx=12, pady=(4, 0))
        self.speed_display_label = ctk.CTkLabel(
            speed_lbl_frame,
            text=f"Interval: {self._current_speed}ms",
            text_color=C_TEXT_SEC,
            font=ctk.CTkFont(size=11),
        )
        self.speed_display_label.pack(side="left")

        self.speed_slider = ctk.CTkSlider(
            sec_cam, from_=50, to=1000, number_of_steps=19,
            command=self._on_speed_slider,
        )
        self.speed_slider.set(self._current_speed)
        self.speed_slider.pack(fill="x", padx=12, pady=(4, 12))

        # ── Section 6: Display overrides ─────────────────────────────────
        sec_disp = self._sidebar_section(self, "Display")
        sec_disp.pack(fill="x", padx=10, pady=6)

        self.heatmap_switch = ctk.CTkSwitch(
            sec_disp,
            text="Color Heatmap",
            variable=self.heatmap_var,
            button_color=C_ACCENT,
            button_hover_color=C_BG_FLOAT,
        )
        self.heatmap_switch.pack(anchor="w", padx=12, pady=(0, 8))

        self.gradcam_switch = ctk.CTkSwitch(
            sec_disp,
            text="Saliency (Grad-CAM)",
            variable=self.gradcam_var,
            button_color=C_ACCENT,
            button_hover_color=C_BG_FLOAT,
        )
        self.gradcam_switch.pack(anchor="w", padx=12, pady=(0, 10))

        # ── Section 7: Diagnostics & actions ─────────────────────────────
        sec_diag = self._sidebar_section(self, "Diagnostics")
        sec_diag.pack(fill="x", padx=10, pady=6)

        ctk.CTkLabel(
            sec_diag,
            text="Dead ReLU Threshold",
            text_color=C_TEXT_SEC,
            font=ctk.CTkFont(size=10),
        ).pack(anchor="w", padx=12)

        self.health_slider = ctk.CTkSlider(
            sec_diag, from_=0.000001, to=0.001, number_of_steps=50
        )
        self.health_slider.set(0.00001)
        self.health_slider.pack(fill="x", padx=12, pady=(4, 12))

        self.visualize_btn = ctk.CTkButton(
            sec_diag,
            text="Visualize Static",
            fg_color=C_ACCENT,
            hover_color=C_BG_FLOAT,
            command=self.on_visualize_callback,
        )
        self.visualize_btn.pack(fill="x", padx=12, pady=(0, 8))

        self.export_btn = ctk.CTkButton(
            sec_diag,
            text="Export Report",
            fg_color=C_BG_FLOAT,
            border_width=1,
            border_color=C_BORDER_ACT,
            hover_color=C_BG_RAISED,
            command=self.on_export_callback,
        )
        self.export_btn.pack(fill="x", padx=12, pady=(0, 10))

    # ------------------------------------------------------------------ #
    #  Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def _on_mode_switch(self, mode: str):
        # FIX #2: original code used pack(before=...) on a widget that
        # hadn't been packed yet, causing TclError. Now we simply forget
        # both widgets of the outgoing mode then pack the incoming pair in
        # sequence — order is fully determined by pack sequence alone.
        if mode == "Layer Mode":
            self.channel_label.pack_forget()
            self.channel_entry.pack_forget()
            self.layer_label.pack(anchor="w", padx=12)
            self.layer_dropdown.pack(fill="x", padx=12, pady=(0, 10))
        else:
            self.layer_label.pack_forget()
            self.layer_dropdown.pack_forget()
            self.channel_label.pack(anchor="w", padx=12)
            self.channel_entry.pack(fill="x", padx=12, pady=(0, 10))

    def _on_speed_slider(self, value: float):
        self._current_speed = int(value)
        self.speed_display_label.configure(
            text=f"Interval: {self._current_speed}ms"
        )
        # FIX #1: guard the optional callback — app.py passes None here
        # because speed is polled via sidebar.config['speed'] instead.
        if self.on_speed_change_callback is not None:
            self.on_speed_change_callback(self._current_speed)

    # ------------------------------------------------------------------ #
    #  Config property                                                     #
    # ------------------------------------------------------------------ #

    @property
    def config(self) -> dict:
        """Returns a snapshot of all current control values."""
        return {
            "mode":            self.mode_var.get(),
            "layer":           self.layer_dropdown.get(),
            # FIX #3: guard against empty string so int() never raises
            "channel":         self.channel_entry.get() or "0",
            "heatmap":         self.heatmap_var.get(),
            "gradcam":         self.gradcam_var.get(),
            "speed":           self._current_speed,
            "dead_threshold":  self.health_slider.get(),
            "flow_chart":      True,
            "corr_matrix":     True,
        }