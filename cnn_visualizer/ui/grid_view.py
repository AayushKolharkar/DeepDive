"""
Grid view widget for displaying extracted feature map image grids.

Changes in this version:
- GridView is now a CTkFrame wrapper rather than inheriting CTkScrollableFrame
  directly. This allows a horizontal CTkScrollbar to be docked at the bottom
  of the same frame, wired to the internal canvas of the nested scroll frame.
- Cell size is dynamic: controlled by the sidebar slider (80–240px).
  Changing cell size triggers a full rebuild and reflows the column count.
- Shift+MouseWheel scrolls horizontally on all platforms.
"""
from __future__ import annotations
import tkinter as tk
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image
from core import DIAG_COLORS
from .theme import *

# Pool size — pre-allocate this many cell frames once and reuse forever.
MAX_POOL = 64

# Default cell image size in pixels.
DEFAULT_CELL_SIZE = 140


def _find_canvas(widget) -> tk.Canvas | None:
    """
    Locate the internal tk.Canvas of a CTkScrollableFrame.
    CTk 5.x names it _parent_canvas; fall back to walking children.
    """
    if hasattr(widget, "_parent_canvas"):
        return widget._parent_canvas
    for child in widget.winfo_children():
        if isinstance(child, tk.Canvas):
            return child
    return None


class GridView(ctk.CTkFrame):
    """
    Responsive grid of feature-map thumbnails with both vertical and
    horizontal scrollbars and a user-adjustable cell size.
    """

    def __init__(self, parent, on_channel_click=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.on_channel_click = on_channel_click
        self.last_health_metrics: dict | None = None
        self.inspected_channel: int | None = None
        self.current_layer_name: str | None = None
        self.current_display_layer: str | None = None

        # Cell size — can be updated via update(cell_size=...)
        self._cell_size: int = DEFAULT_CELL_SIZE

        # Per-slot state
        self._channel_indices: list[int] = []
        self._ctk_image_cache: list[np.ndarray | None] = [None] * MAX_POOL
        self._cell_data: dict[int, tuple] = {}
        self._active_count: int = 0
        self._cols: int = 5

        # ── Layout: scroll area + h-scrollbar ────────────────────────────
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self._scroll_frame = ctk.CTkScrollableFrame(
            self, fg_color="transparent"
        )
        self._scroll_frame.grid(row=0, column=0, sticky="nsew")

        self._h_scrollbar = ctk.CTkScrollbar(
            self, orientation="horizontal", height=16,
            fg_color=C_BG_BASE,
            button_color=C_BG_FLOAT,
            button_hover_color=C_BORDER_ACT,
        )
        self._h_scrollbar.grid(row=1, column=0, sticky="ew")

        # Wire h-scrollbar to the internal canvas (version-safe)
        self._canvas: tk.Canvas | None = None
        self._wire_hscroll()

        # Bind Shift+Wheel for horizontal scrolling
        self._scroll_frame.bind("<Shift-MouseWheel>",  self._on_hscroll_wheel)
        self._scroll_frame.bind("<Shift-Button-4>",    self._on_hscroll_wheel)
        self._scroll_frame.bind("<Shift-Button-5>",    self._on_hscroll_wheel)
        # Also bind on the outer frame so it's always captured
        self.bind("<Shift-MouseWheel>", self._on_hscroll_wheel)

        # ── Title label (row 0 inside scroll frame) ───────────────────────
        self._title_label = ctk.CTkLabel(
            self._scroll_frame, text="",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=C_TEXT_PRI,
        )
        self._title_label.grid(row=0, column=0, columnspan=MAX_POOL, pady=(10, 16))

        # ── Pre-allocate widget pool inside scroll frame ──────────────────
        self._frames:     list[ctk.CTkFrame] = []
        self._img_labels: list[ctk.CTkLabel] = []
        self._txt_labels: list[ctk.CTkLabel] = []

        for _ in range(MAX_POOL):
            frame = ctk.CTkFrame(
                self._scroll_frame,
                border_width=2,
                border_color=C_BORDER_SUB,
                fg_color=C_BG_RAISED,
                corner_radius=6,
            )
            frame.original_border_color = C_BORDER_SUB

            img_lbl = ctk.CTkLabel(frame, text="")
            img_lbl.pack(padx=2, pady=2)

            txt_lbl = ctk.CTkLabel(
                frame, text="",
                font=ctk.CTkFont(size=11),
                text_color=C_TEXT_SEC,
            )
            txt_lbl.pack(pady=(0, 5))

            self._frames.append(frame)
            self._img_labels.append(img_lbl)
            self._txt_labels.append(txt_lbl)

        # Reflow on resize
        self._scroll_frame.bind("<Configure>", self._on_configure)

    # ------------------------------------------------------------------ #
    #  H-scrollbar wiring                                                  #
    # ------------------------------------------------------------------ #

    def _wire_hscroll(self):
        """
        Connect the horizontal scrollbar to the CTkScrollableFrame's canvas.
        Called once after construction. Deferred via after() so the canvas
        has been created by the time we search for it.
        """
        self.after(100, self._wire_hscroll_deferred)

    def _wire_hscroll_deferred(self):
        canvas = _find_canvas(self._scroll_frame)
        if canvas is None:
            # Retry once more after another tick
            self.after(200, self._wire_hscroll_deferred)
            return
        self._canvas = canvas
        canvas.configure(xscrollcommand=self._h_scrollbar.set)
        self._h_scrollbar.configure(command=canvas.xview)

    def _on_hscroll_wheel(self, event):
        """Shift+scroll → horizontal scroll."""
        if self._canvas is None:
            return
        if event.num == 5 or getattr(event, "delta", 0) < 0:
            self._canvas.xview_scroll(3, "units")
        else:
            self._canvas.xview_scroll(-3, "units")

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

    def _slot_width(self) -> int:
        """Pixel width of one grid slot = cell image + padding."""
        return self._cell_size + 16   # 8px padx each side

    def _compute_cols(self) -> int:
        w = self._scroll_frame.winfo_width()
        if w < 10:
            return 5
        return max(1, w // self._slot_width())

    def _on_configure(self, event=None):
        new_cols = self._compute_cols()
        if new_cols != self._cols and self._active_count > 0:
            self._cols = new_cols
            self._reflow()

    def _reflow(self):
        cols = self._cols
        for slot in range(self._active_count):
            r = (slot // cols) + 1
            c = slot % cols
            self._frames[slot].grid(row=r, column=c, padx=8, pady=8)
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
        cell_size: int | None = None,
    ):
        """
        Renders feature-map thumbnails into the grid.

        cell_size: if provided and different from current size, triggers a
                   full rebuild with the new dimensions.
        """
        if layer_name is not None:
            self.current_layer_name = layer_name

        # Cell size change → force a full rebuild
        if cell_size is not None and cell_size != self._cell_size:
            self._cell_size = cell_size
            force_rebuild = True

        self.last_health_metrics = health_metrics
        new_count = min(len(imagedata_list), MAX_POOL)
        layer_changed = self.current_display_layer != title
        self._cols = self._compute_cols()

        if not force_rebuild and not layer_changed:
            self._fast_update(imagedata_list, new_count)
            return

        # ── Full rebuild ──────────────────────────────────────────────────
        self._title_label.configure(text=title)
        self.current_display_layer = title
        self._active_count = new_count
        self._channel_indices = []
        self._cell_data = {}

        cs = self._cell_size
        cols = self._cols

        for slot, item in enumerate(imagedata_list[:MAX_POOL]):
            img, label_str, diag_class, original_idx = self._unpack(item)
            self._channel_indices.append(original_idx)

            if original_idx != -1:
                self._cell_data[original_idx] = (img, label_str, diag_class)

            img_np, img_resized = self._resize_image(img, cs)
            self._ctk_image_cache[slot] = img_np

            ctk_img = ctk.CTkImage(
                light_image=img_resized, dark_image=img_resized, size=(cs, cs)
            )

            border_color = self._saturation_color(original_idx, health_metrics)
            selected = (original_idx == self.inspected_channel and original_idx != -1)

            frame = self._frames[slot]
            frame.configure(
                border_color=C_ACCENT if selected else border_color,
                border_width=3 if selected else 2,
                fg_color=C_BG_RAISED,
            )
            frame.original_border_color = border_color

            self._img_labels[slot].configure(image=ctk_img)
            self._txt_labels[slot].configure(
                text=label_str,
                text_color=self._label_color(diag_class, original_idx, health_metrics),
            )

            if self.on_channel_click and original_idx != -1:
                handler = self._make_click_handler(original_idx)
                for w in (frame, self._img_labels[slot], self._txt_labels[slot]):
                    w.bind("<Button-1>", handler)
                    w.configure(cursor="hand2")
                frame.bind("<Enter>",
                           lambda e, f=frame: f.configure(border_color=C_ACCENT))
                frame.bind("<Leave>", self._make_leave_handler(frame, original_idx))

            r = (slot // cols) + 1
            c = slot % cols
            frame.grid(row=r, column=c, padx=8, pady=8)

        for slot in range(new_count, MAX_POOL):
            self._frames[slot].grid_remove()

        self._scroll_frame.update_idletasks()

    def _fast_update(self, imagedata_list: list[tuple], new_count: int):
        """In-place image swap — only re-allocates changed cells."""
        cs = self._cell_size
        for slot, item in enumerate(imagedata_list[:MAX_POOL]):
            if slot >= len(self._img_labels):
                break
            img = item[0]
            original_idx = item[3] if len(item) == 4 else -1
            if original_idx != -1 and len(item) >= 3:
                self._cell_data[original_idx] = (item[0], item[1], item[2])

            img_np, img_resized = self._resize_image(img, cs)
            cached = self._ctk_image_cache[slot]
            if cached is not None and np.array_equal(img_np, cached):
                continue
            self._ctk_image_cache[slot] = img_np
            ctk_img = ctk.CTkImage(
                light_image=img_resized, dark_image=img_resized, size=(cs, cs)
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
    def _resize_image(img: Image.Image, size: int) -> tuple[np.ndarray, Image.Image]:
        img_np = np.array(img)
        img_np = cv2.resize(img_np, (size, size), interpolation=cv2.INTER_NEAREST)
        return img_np, Image.fromarray(img_np)

    @staticmethod
    def _saturation_color(original_idx: int, health_metrics: dict | None) -> str:
        if health_metrics and "saturation" in health_metrics and original_idx != -1:
            try:
                sat_val = health_metrics["saturation"]["per_channel"][original_idx].item()
                if sat_val < 0.10: return C_HEALTHY
                if sat_val < 0.30: return C_WARNING
                return C_CRITICAL
            except Exception:
                pass
        return C_BORDER_SUB

    @staticmethod
    def _label_color(diag_class: str, original_idx: int,
                     health_metrics: dict | None) -> str:
        if diag_class == "dead":      return C_TEXT_MUT
        if diag_class == "redundant": return C_CRITICAL
        if health_metrics and "diversity" in health_metrics and original_idx != -1:
            per_ch = health_metrics["diversity"].get("per_channel_max")
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
            frame.configure(
                border_color=C_ACCENT if selected else frame.original_border_color,
                border_width=3 if selected else 2,
            )
        return handler