"""
Grid view widget for displaying extracted feature map image grids.

Performance changes vs previous version:
- Widget pool: MAX_POOL cell frames are created once in __init__ and reused.
  No more destroy/create on every layer change (was 108+ widget ops → ~500ms freeze).
- Dynamic column count: cols computed from frame width so all available space is used.
- <Configure> binding: grid reflows automatically when the pane is resized.
- Deferred geometry: update_idletasks() called once after a full rebuild batch.
- CTkImage cache: unchanged cells skip allocation entirely.
"""
from __future__ import annotations
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image
from core import DIAG_COLORS
from .theme import *

# Maximum number of channel cells ever shown.  Pre-allocating this many frames
# means we never destroy/create widgets at runtime.
MAX_POOL = 64
# Pixel width of one cell (image + padding) used for column count calculation.
CELL_W = 156   # 140px image + 8px padx each side
CELL_H = 172   # 140px image + label + padding


class GridView(ctk.CTkScrollableFrame):
    """
    Scrollable frame that displays feature-map thumbnails in a responsive grid.
    """

    def __init__(self, parent, on_channel_click=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.on_channel_click = on_channel_click
        self.last_health_metrics: dict | None = None
        self.inspected_channel: int | None = None
        self.current_layer_name: str | None = None
        self.current_display_layer: str | None = None

        # Per-slot state
        self._channel_indices: list[int] = []
        self._ctk_image_cache: list[np.ndarray | None] = [None] * MAX_POOL
        self._cell_data: dict[int, tuple] = {}

        # How many cells are currently visible
        self._active_count: int = 0
        # Current computed column count (recomputed on <Configure>)
        self._cols: int = 5
        # Title label reference
        self._title_label: ctk.CTkLabel | None = None

        # ── Build the fixed widget pool ───────────────────────────────────
        # Each slot: outer frame → image label + text label
        self._frames:      list[ctk.CTkFrame]  = []
        self._img_labels:  list[ctk.CTkLabel]  = []
        self._txt_labels:  list[ctk.CTkLabel]  = []

        for i in range(MAX_POOL):
            frame = ctk.CTkFrame(
                self,
                border_width=2,
                border_color=C_BORDER_SUB,
                fg_color=C_BG_RAISED,
                corner_radius=6,
            )
            # Do NOT grid() here — cells start hidden
            frame.original_border_color = C_BORDER_SUB

            img_lbl = ctk.CTkLabel(frame, text="")
            img_lbl.pack(padx=2, pady=2)

            txt_lbl = ctk.CTkLabel(frame, text="", font=ctk.CTkFont(size=11),
                                   text_color=C_TEXT_SEC)
            txt_lbl.pack(pady=(0, 5))

            self._frames.append(frame)
            self._img_labels.append(img_lbl)
            self._txt_labels.append(txt_lbl)

        # Title row (row 0); cells start at row 1
        self._title_label = ctk.CTkLabel(
            self, text="", font=ctk.CTkFont(size=18, weight="bold"),
            text_color=C_TEXT_PRI,
        )
        self._title_label.grid(row=0, column=0, columnspan=MAX_POOL, pady=(10, 16))

        # Reflow when the frame is resized (e.g. pane drag)
        self.bind("<Configure>", self._on_configure)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_cell_data(self, channel_idx: int) -> tuple | None:
        return self._cell_data.get(channel_idx)

    def clear_selection(self):
        self.refresh_borders(None)

    def refresh_borders(self, inspected_channel: int | None):
        self.inspected_channel = inspected_channel
        for slot, frame in enumerate(self._frames[:self._active_count]):
            ch = self._channel_indices[slot] if slot < len(self._channel_indices) else -1
            selected = (ch == inspected_channel and ch != -1)
            color = C_ACCENT if selected else getattr(frame, "original_border_color", C_BORDER_SUB)
            width = 3 if selected else 2
            frame.configure(border_color=color, border_width=width)

    # ------------------------------------------------------------------ #
    #  Layout helpers                                                      #
    # ------------------------------------------------------------------ #

    def _compute_cols(self) -> int:
        """Compute how many columns fit in the current frame width."""
        w = self.winfo_width()
        if w < 10:
            return 5  # fallback before first render
        cols = max(1, w // CELL_W)
        return cols

    def _on_configure(self, event=None):
        """Reflow the grid when the container is resized."""
        new_cols = self._compute_cols()
        if new_cols != self._cols and self._active_count > 0:
            self._cols = new_cols
            self._reflow()

    def _reflow(self):
        """Re-grid all active cells using the current column count."""
        cols = self._cols
        for slot in range(self._active_count):
            r = (slot // cols) + 1
            c = slot % cols
            self._frames[slot].grid(row=r, column=c, padx=8, pady=8)
        # Hide trailing cells
        for slot in range(self._active_count, MAX_POOL):
            self._frames[slot].grid_remove()

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
        Updates the grid with new image data.

        imagedata_list: 3-tuples (img, label, diag) or 4-tuples (img, label, diag, ch_idx).
        """
        if layer_name is not None:
            self.current_layer_name = layer_name

        self.last_health_metrics = health_metrics
        new_count = min(len(imagedata_list), MAX_POOL)
        layer_changed = self.current_display_layer != title
        self._cols = self._compute_cols()

        # ── Fast path: same layer, just swap images ───────────────────────
        if not force_rebuild and not layer_changed:
            self._fast_update(imagedata_list, new_count)
            return

        # ── Full update: title + all cell content + layout ────────────────
        self._title_label.configure(text=title)
        self.current_display_layer = title
        self._active_count = new_count
        self._channel_indices = []
        self._cell_data = {}

        cols = self._cols

        for slot, item in enumerate(imagedata_list[:MAX_POOL]):
            img, label_str, diag_class, original_idx = self._unpack(item)
            self._channel_indices.append(original_idx)

            if original_idx != -1:
                self._cell_data[original_idx] = (img, label_str, diag_class)

            # Resize image
            img_np, img_resized = self._resize_image(img)
            self._ctk_image_cache[slot] = img_np

            ctk_img = ctk.CTkImage(
                light_image=img_resized, dark_image=img_resized, size=(140, 140)
            )

            # Border colour from saturation
            border_color = self._saturation_color(original_idx, health_metrics)
            selected = (original_idx == self.inspected_channel and original_idx != -1)
            final_color = C_ACCENT if selected else border_color
            final_width = 3 if selected else 2

            frame = self._frames[slot]
            frame.configure(
                border_color=final_color, border_width=final_width,
                fg_color=C_BG_RAISED,
            )
            frame.original_border_color = border_color

            self._img_labels[slot].configure(image=ctk_img)
            self._txt_labels[slot].configure(
                text=label_str,
                text_color=self._label_color(diag_class, original_idx, health_metrics),
            )

            # Bind click handlers
            if self.on_channel_click and original_idx != -1:
                handler = self._make_click_handler(original_idx)
                frame.configure(cursor="hand2")
                self._img_labels[slot].configure(cursor="hand2")
                frame.bind("<Button-1>", handler)
                self._img_labels[slot].bind("<Button-1>", handler)
                self._txt_labels[slot].bind("<Button-1>", handler)
                self._txt_labels[slot].configure(cursor="hand2")

                frame.bind("<Enter>",
                    lambda e, f=frame: f.configure(border_color=C_ACCENT))
                frame.bind("<Leave>",
                    self._make_leave_handler(frame, original_idx))

            # Place in grid
            r = (slot // cols) + 1
            c = slot % cols
            frame.grid(row=r, column=c, padx=8, pady=8)

        # Hide unused pool cells
        for slot in range(new_count, MAX_POOL):
            self._frames[slot].grid_remove()

        # Single geometry flush after all grid() calls
        self.update_idletasks()

    def _fast_update(self, imagedata_list: list[tuple], new_count: int):
        """
        In-place image swap for the live camera loop.
        Only allocates a new CTkImage when the pixel data actually changed.
        Skips border/label updates entirely (they don't change frame-to-frame).
        """
        for slot, item in enumerate(imagedata_list[:MAX_POOL]):
            if slot >= len(self._img_labels):
                break

            img = item[0]
            original_idx = item[3] if len(item) == 4 else -1

            # Update cell_data for inspector
            if original_idx != -1 and len(item) >= 3:
                self._cell_data[original_idx] = (item[0], item[1], item[2])

            img_np, img_resized = self._resize_image(img)

            # Skip if pixel-identical
            cached = self._ctk_image_cache[slot]
            if cached is not None and np.array_equal(img_np, cached):
                continue

            self._ctk_image_cache[slot] = img_np
            ctk_img = ctk.CTkImage(
                light_image=img_resized, dark_image=img_resized, size=(140, 140)
            )
            self._img_labels[slot].configure(image=ctk_img)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _unpack(item: tuple) -> tuple:
        if len(item) == 4:
            return item
        img, label_str, diag_class = item
        ch_str = label_str.split(" ")[-1]
        original_idx = int(ch_str) if ch_str.isdigit() else -1
        return img, label_str, diag_class, original_idx

    @staticmethod
    def _resize_image(img: Image.Image) -> tuple[np.ndarray, Image.Image]:
        img_np = np.array(img)
        img_np = cv2.resize(img_np, (140, 140), interpolation=cv2.INTER_NEAREST)
        return img_np, Image.fromarray(img_np)

    @staticmethod
    def _saturation_color(original_idx: int, health_metrics: dict | None) -> str:
        if health_metrics and "saturation" in health_metrics and original_idx != -1:
            try:
                sat_val = health_metrics["saturation"]["per_channel"][original_idx].item()
                if sat_val < 0.10:
                    return C_HEALTHY
                if sat_val < 0.30:
                    return C_WARNING
                return C_CRITICAL
            except Exception:
                pass
        return C_BORDER_SUB

    @staticmethod
    def _label_color(
        diag_class: str,
        original_idx: int,
        health_metrics: dict | None,
    ) -> str:
        if diag_class == "dead":
            return C_TEXT_MUT
        if diag_class == "redundant":
            return C_CRITICAL

        # Diversity tint
        if health_metrics and "diversity" in health_metrics and original_idx != -1:
            div_info = health_metrics["diversity"]
            per_ch = div_info.get("per_channel_max")
            if per_ch is not None:
                try:
                    if original_idx < len(per_ch) and per_ch[original_idx].item() > 0.7:
                        return C_INFO
                except Exception:
                    pass
        return C_TEXT_SEC

    def _make_click_handler(self, ch_idx: int):
        def handler(event):
            self.on_channel_click(ch_idx, self.current_layer_name)
        return handler

    def _make_leave_handler(self, frame: ctk.CTkFrame, ch: int):
        def handler(e):
            selected = self.inspected_channel == ch
            color = C_ACCENT if selected else frame.original_border_color
            width = 3 if selected else 2
            frame.configure(border_color=color, border_width=width)
        return handler