"""
Grid view widget for displaying extracted feature map image grids.
"""
from __future__ import annotations
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image
from core import DIAG_COLORS
from .theme import *


class GridView(ctk.CTkScrollableFrame):
    """
    Scrollable right-hand frame for displaying diagnostic image grids.
    """

    def __init__(self, parent, on_channel_click=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.image_labels: list[ctk.CTkLabel] = []
        self._ctk_image_cache: list[np.ndarray] = []
        self._image_frames: list[ctk.CTkFrame] = []
        self._channel_indices: list[int] = []
        # FIX #13: store PIL images keyed by channel index so
        # get_cell_data() can return them without re-rendering.
        self._cell_data: dict[int, tuple] = {}

        self.current_display_layer: str | None = None
        # FIX #12: track the actual layer name separately from the
        # full dynamic title string that was previously being passed
        # to click handlers as "layer".
        self.current_layer_name: str | None = None

        self.inspected_channel: int | None = None
        self.on_channel_click = on_channel_click
        self.last_health_metrics: dict | None = None

        self.flow_chart_label = ctk.CTkLabel(self, text="")
        self.flow_chart_label.grid(row=0, column=0, columnspan=6, pady=(10, 10))
        self.flow_chart_label.grid_remove()

        self.corr_matrix_label = ctk.CTkLabel(self, text="")
        self.corr_matrix_label.grid(row=98, column=0, columnspan=6, pady=(10, 10))
        self.corr_matrix_label.grid_remove()

        self.histogram_label = ctk.CTkLabel(self, text="")
        self.histogram_label.grid(row=99, column=0, columnspan=6, pady=(20, 10))

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_cell_data(self, channel_idx: int) -> tuple | None:
        """
        Returns the (PIL_Image, label_str, diag_class) tuple for the given
        channel index, or None if that channel is not in the current grid.

        FIX #8 / #9: this method was called from app.py but didn't exist.
        """
        return self._cell_data.get(channel_idx)

    def clear_selection(self):
        """
        Clears the currently inspected channel highlight.

        FIX #9: was called from _close_inspector() but didn't exist.
        """
        self.refresh_borders(None)

    def refresh_borders(self, inspected_channel: int | None):
        """
        Resets border colours to reflect the currently inspected channel
        and per-channel saturation status without rebuilding the grid.
        """
        self.inspected_channel = inspected_channel
        for idx, frame in enumerate(self._image_frames):
            ch_idx = self._channel_indices[idx]
            is_selected = (ch_idx == inspected_channel and ch_idx != -1)
            color = C_ACCENT if is_selected else getattr(frame, "original_border_color", C_BORDER_SUB)
            width = 3 if is_selected else 2
            frame.configure(border_color=color, border_width=width)

    # ------------------------------------------------------------------ #
    #  Main update                                                         #
    # ------------------------------------------------------------------ #

    def update(
        self,
        imagedata_list: list[tuple],
        title: str,
        force_rebuild: bool = False,
        health_metrics: dict | None = None,
        layer_name: str | None = None,
    ):
        """
        Renders extracted PIL images onto the scrollable frame.

        Args:
            imagedata_list: List of 3-tuples (PIL_Image, label_string, diag_class).
                            OR 4-tuples (PIL_Image, label_string, diag_class, channel_idx)
                            when channel_idx is passed explicitly (preferred).
            title:          Dynamic title shown above the grid.
            force_rebuild:  Tear down and rebuild all widgets.
            health_metrics: Latest health metrics for border colouring.
            layer_name:     The actual conv layer name (not the full title string).
                            FIX #12: used in click handlers instead of title.
        """
        # FIX #12: store the real layer name whenever provided
        if layer_name is not None:
            self.current_layer_name = layer_name

        rebuild = (not self.image_labels) or (self.current_display_layer != title) or force_rebuild

        if rebuild:
            for widget in self.winfo_children():
                if widget not in (self.histogram_label, self.flow_chart_label, self.corr_matrix_label):
                    widget.destroy()

            self.image_labels = []
            self._ctk_image_cache = []
            self._image_frames = []
            self._channel_indices = []
            self._cell_data = {}
            self.last_health_metrics = health_metrics

            ctk.CTkLabel(
                self,
                text=title,
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color=C_TEXT_PRI,
            ).grid(row=1, column=0, columnspan=5, pady=(10, 20))

            cols = 5
            for idx, item in enumerate(imagedata_list):
                # FIX #13: support both 3-tuple and 4-tuple formats.
                # 4-tuple carries explicit channel index — no string parsing.
                if len(item) == 4:
                    img, label_str, diag_class, original_idx = item
                else:
                    img, label_str, diag_class = item
                    # Fallback: try to parse channel index from label string.
                    # Works for "Channel 12" but NOT for "RGB Input" or
                    # layer names — those correctly stay as -1 (unclickable).
                    ch_str = label_str.split(" ")[-1]
                    original_idx = int(ch_str) if ch_str.isdigit() else -1

                display_size = (140, 140)
                img_np = np.array(img)
                img_np = cv2.resize(img_np, display_size, interpolation=cv2.INTER_NEAREST)
                img_resized = Image.fromarray(img_np)

                self._ctk_image_cache.append(img_np)
                self._channel_indices.append(original_idx)

                # Store for get_cell_data()
                if original_idx != -1:
                    self._cell_data[original_idx] = (img, label_str, diag_class)

                ctk_img = ctk.CTkImage(
                    light_image=img_resized, dark_image=img_resized, size=display_size
                )

                row = (idx // cols) + 2
                col = idx % cols

                # Saturation border colour
                border_color = C_BORDER_SUB
                if health_metrics and "saturation" in health_metrics and original_idx != -1:
                    try:
                        sat_val = health_metrics["saturation"]["per_channel"][original_idx].item()
                        if sat_val < 0.10:
                            border_color = C_HEALTHY
                        elif sat_val < 0.30:
                            border_color = C_WARNING
                        else:
                            border_color = C_CRITICAL
                    except Exception:
                        pass

                is_selected = (original_idx == self.inspected_channel and original_idx != -1)
                final_color = C_ACCENT if is_selected else border_color
                final_width = 3 if is_selected else 2

                img_frame = ctk.CTkFrame(
                    self,
                    border_width=final_width,
                    border_color=final_color,
                    fg_color=C_BG_RAISED,
                    corner_radius=6,
                )
                img_frame.grid(row=row, column=col, padx=8, pady=8)
                img_frame.original_border_color = border_color
                self._image_frames.append(img_frame)

                lbl_img = ctk.CTkLabel(img_frame, text="", image=ctk_img)
                lbl_img.pack(padx=2, pady=2)
                self.image_labels.append(lbl_img)

                if self.on_channel_click and original_idx != -1:
                    # FIX #12: capture the real layer name, not the title string.
                    # current_layer_name is set at the top of this method.
                    def make_click_handler(ch_idx, lyr):
                        def handler(event):
                            self.on_channel_click(ch_idx, lyr)
                        return handler

                    click_handler = make_click_handler(
                        original_idx, self.current_layer_name
                    )
                    img_frame.bind("<Button-1>", click_handler)
                    lbl_img.bind("<Button-1>", click_handler)

                    img_frame.bind(
                        "<Enter>",
                        lambda e, f=img_frame: f.configure(border_color=C_ACCENT),
                    )

                    def make_leave_handler(f, ch):
                        def handler(e):
                            color = (
                                C_ACCENT
                                if self.inspected_channel == ch
                                else f.original_border_color
                            )
                            width = 3 if self.inspected_channel == ch else 2
                            f.configure(border_color=color, border_width=width)
                        return handler

                    img_frame.bind("<Leave>", make_leave_handler(img_frame, original_idx))
                    img_frame.configure(cursor="hand2")
                    lbl_img.configure(cursor="hand2")

                # Diagnostic label colour
                if diag_class == "dead":
                    label_color = C_TEXT_MUT
                elif diag_class == "redundant":
                    label_color = C_CRITICAL
                else:
                    label_color = C_TEXT_SEC

                # Diversity tint
                if health_metrics and "diversity" in health_metrics and diag_class == "normal":
                    div_info = health_metrics["diversity"]
                    per_ch_max = div_info.get("per_channel_max")
                    if per_ch_max is not None and original_idx != -1:
                        try:
                            if original_idx < len(per_ch_max):
                                if per_ch_max[original_idx].item() > 0.7:
                                    label_color = C_INFO
                        except Exception:
                            pass

                lbl_text = ctk.CTkLabel(
                    img_frame,
                    text=label_str,
                    font=ctk.CTkFont(size=11),
                    text_color=label_color,
                )
                lbl_text.pack(pady=(0, 5))

                if self.on_channel_click and original_idx != -1:
                    lbl_text.bind("<Button-1>", click_handler)
                    lbl_text.configure(cursor="hand2")

            self.current_display_layer = title

        else:
            # Fast in-place update for the camera loop
            self.last_health_metrics = health_metrics
            for idx, item in enumerate(imagedata_list):
                if idx >= len(self.image_labels):
                    break

                img = item[0]
                # Update cell_data cache for inspector lookups
                original_idx = self._channel_indices[idx] if idx < len(self._channel_indices) else -1
                if original_idx != -1 and len(item) >= 3:
                    self._cell_data[original_idx] = (item[0], item[1], item[2])

                display_size = (140, 140)
                img_np = np.array(img)
                img_np = cv2.resize(img_np, display_size, interpolation=cv2.INTER_NEAREST)

                if idx < len(self._ctk_image_cache) and np.array_equal(img_np, self._ctk_image_cache[idx]):
                    continue

                self._ctk_image_cache[idx] = img_np
                img_resized = Image.fromarray(img_np)
                ctk_img = ctk.CTkImage(
                    light_image=img_resized, dark_image=img_resized, size=display_size
                )
                self.image_labels[idx].configure(image=ctk_img)