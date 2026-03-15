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
        self.image_labels = []
        self._ctk_image_cache = []
        self._image_frames: list[ctk.CTkFrame] = []
        self._channel_indices: list[int] = []
        self.current_display_layer = None
        self.inspected_channel = None
        self.on_channel_click = on_channel_click
        self.last_health_metrics = None
        
        self.flow_chart_label = ctk.CTkLabel(self, text="")
        self.flow_chart_label.grid(row=0, column=0, columnspan=6, pady=(10, 10))
        self.flow_chart_label.grid_remove()

        self.corr_matrix_label = ctk.CTkLabel(self, text="")
        self.corr_matrix_label.grid(row=98, column=0, columnspan=6, pady=(10, 10))
        self.corr_matrix_label.grid_remove()

        # Histogram Panel packed below the grid conceptually
        # But we pack it in update() so it ends up at the bottom
        self.histogram_label = ctk.CTkLabel(self, text="")
        self.histogram_label.grid(row=99, column=0, columnspan=6, pady=(20, 10))

    def update(self, imagedata_list: list[tuple[Image.Image, str, str]], title: str, force_rebuild: bool = False, health_metrics: dict | None = None):
        """
        Renders extracted PIL images onto the scrollable frame without flashing.
        
        Args:
            imagedata_list (list): List of 3-tuples (PIL_Image, label_string, diag_class)
            title (str): Dynamic title for the grid
            force_rebuild (bool): Force a complete tear down and sub-widget rebuild.
        """
        if (not self.image_labels) or (self.current_display_layer != title) or force_rebuild:
            for widget in self.winfo_children():
                if widget not in (self.histogram_label, self.flow_chart_label, self.corr_matrix_label):
                    widget.destroy()

            self.image_labels = []
            self._ctk_image_cache = []
            self._image_frames = []
            self._channel_indices = []
            self.last_health_metrics = health_metrics

            lbl_title = ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=18, weight="bold"), text_color=C_TEXT_PRI)
            lbl_title.grid(row=1, column=0, columnspan=5, pady=(10, 20))

            cols = 5
            for idx, (img, label_str, diag_class) in enumerate(imagedata_list):
                display_size = (140, 140)
                img_np = np.array(img)
                img_np = cv2.resize(img_np, display_size, interpolation=cv2.INTER_NEAREST)
                img_resized = Image.fromarray(img_np)
                
                self._ctk_image_cache.append(img_np)
                ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=display_size)

                row = (idx // cols) + 2
                col = idx % cols

                original_idx = -1
                ch_str = label_str.split(" ")[-1]
                if ch_str.isdigit():
                    original_idx = int(ch_str)
                self._channel_indices.append(original_idx)

                # Saturation border base colour
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
                final_border_color = C_ACCENT if is_selected else border_color
                final_border_width = 3 if is_selected else 2

                img_frame = ctk.CTkFrame(self, border_width=final_border_width, border_color=final_border_color, fg_color=C_BG_RAISED, corner_radius=6)
                img_frame.grid(row=row, column=col, padx=8, pady=8)
                img_frame.original_border_color = border_color
                self._image_frames.append(img_frame)

                lbl_img = ctk.CTkLabel(img_frame, text="", image=ctk_img)
                lbl_img.pack(padx=2, pady=2)
                self.image_labels.append(lbl_img)
                
                if self.on_channel_click and original_idx != -1:
                    def make_click_handler(ch_idx, layer):
                        def handler(event):
                            self.on_channel_click(ch_idx, layer)
                        return handler

                    click_handler = make_click_handler(original_idx, title)
                    img_frame.bind("<Button-1>", click_handler)
                    lbl_img.bind("<Button-1>", click_handler)
                    
                    img_frame.bind("<Enter>", lambda e, f=img_frame: f.configure(border_color=C_ACCENT))
                    def leave_handler(e, f=img_frame, ch=original_idx):
                        color = C_ACCENT if self.inspected_channel == ch else f.original_border_color
                        width = 3 if self.inspected_channel == ch else 2
                        f.configure(border_color=color, border_width=width)
                    img_frame.bind("<Leave>", leave_handler)
                    
                    img_frame.configure(cursor="hand2")
                    lbl_img.configure(cursor="hand2")

                label_color = C_TEXT_MUT if diag_class == "dead" else (C_CRITICAL if diag_class == "redundant" else C_TEXT_SEC)
                
                # Apply diversity tint if present and > 0.7
                if health_metrics and "diversity" in health_metrics and diag_class == "normal":
                    div_info = health_metrics["diversity"]
                    per_ch_max = div_info.get("per_channel_max")
                    if per_ch_max is not None and original_idx != -1:
                        try:
                            if original_idx < len(per_ch_max):
                                clone_score = per_ch_max[original_idx].item()
                                if clone_score > 0.7:
                                    label_color = C_INFO
                        except Exception:
                            pass

                lbl_text = ctk.CTkLabel(img_frame, text=label_str, font=ctk.CTkFont(size=11), text_color=label_color)
                lbl_text.pack(pady=(0, 5))
                
                # pass clicks through text as well
                if self.on_channel_click and original_idx != -1:
                    lbl_text.bind("<Button-1>", click_handler)
                    lbl_text.configure(cursor="hand2")

            self.current_display_layer = title

        else:
            # Fast in-place update for the camera loop
            self.last_health_metrics = health_metrics
            for idx, (img, label_str, diag_class) in enumerate(imagedata_list):
                if idx < len(self.image_labels):
                    display_size = (140, 140)
                    img_np = np.array(img)
                    img_np = cv2.resize(img_np, display_size, interpolation=cv2.INTER_NEAREST)
                    
                    if idx < len(self._ctk_image_cache) and np.array_equal(img_np, self._ctk_image_cache[idx]):
                        continue
                        
                    self._ctk_image_cache[idx] = img_np
                    img_resized = Image.fromarray(img_np)
                    
                    ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=display_size)
                    self.image_labels[idx].configure(image=ctk_img)
                    
            # if we get updated health metrics during live loops, we should ideally refresh borders, 
            # but that is heavy. Doing it only on explicit request to save CPU.

    def refresh_borders(self, inspected_channel: int | None):
        """
        Iterates the current image_labels and their associated frames,
        resetting border colours to reflect the current inspected channel
        and saturation status. Does not rebuild the grid.
        """
        self.inspected_channel = inspected_channel
        for idx in range(len(self._image_frames)):
            frame = self._image_frames[idx]
            ch_idx = self._channel_indices[idx]
            
            is_selected = (ch_idx == inspected_channel and ch_idx != -1)
            color = C_ACCENT if is_selected else frame.original_border_color
            width = 3 if is_selected else 2
            
            frame.configure(border_color=color, border_width=width)
