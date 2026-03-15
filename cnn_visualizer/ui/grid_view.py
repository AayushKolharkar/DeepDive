"""
Grid view widget for displaying extracted feature map image grids.

Architecture:
  GridView (ctk.CTkFrame)
    ├── _canvas (tk.Canvas)  — scrollable viewport, both axes
    ├── _inner  (tk.Frame)   — actual grid content, embedded via create_window
    ├── _v_scrollbar         — right edge, vertical
    └── _h_scrollbar         — bottom edge, horizontal

Why raw Canvas instead of CTkScrollableFrame:
  CTkScrollableFrame pins its inner frame width to the canvas width on every
  <Configure> event, making horizontal scroll physically impossible.
  A raw canvas with create_window + anchor='nw' lets content exceed the
  viewport freely in both axes.

Mouse wheel forwarding:
  Tk delivers <MouseWheel> to the widget under the cursor, not the canvas.
  _bind_scroll() is called on every pool widget so events always reach us.

Channel filter:
  update() accepts channel_filter: set[int] | None.
  Only listed channel indices are rendered. None = show all.
"""
from __future__ import annotations
import platform
import tkinter as tk
import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image
from .theme import *

MAX_POOL          = 64
DEFAULT_CELL_SIZE = 140
_OS = platform.system()   # "Windows" | "Darwin" | "Linux"


class GridView(ctk.CTkFrame):
    """
    Two-axis scrollable grid of feature-map thumbnails.
    """

    def __init__(self, parent, on_channel_click=None, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)

        self.on_channel_click       = on_channel_click
        self.last_health_metrics: dict | None = None
        self.inspected_channel:   int  | None = None
        self.current_layer_name:  str  | None = None
        self.current_display_layer: str | None = None

        self._cell_size:   int = DEFAULT_CELL_SIZE
        self._active_count: int = 0
        self._cols:         int = 5
        self._channel_filter: set[int] | None = None
        self._last_filter_applied: set[int] | None = None

        self._channel_indices: list[int]               = []
        self._ctk_image_cache: list[np.ndarray | None] = [None] * MAX_POOL
        self._cell_data:       dict[int, tuple]        = {}

        # ── Root layout: canvas + two scrollbars ──────────────────────────
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        bg = C_BG_BASE
        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self._canvas.grid(row=0, column=0, sticky="nsew")

        self._v_scrollbar = ctk.CTkScrollbar(
            self, orientation="vertical",
            fg_color=C_BG_BASE,
            button_color=C_BG_FLOAT,
            button_hover_color=C_BORDER_ACT,
            command=self._canvas.yview,
        )
        self._v_scrollbar.grid(row=0, column=1, sticky="ns")

        self._h_scrollbar = ctk.CTkScrollbar(
            self, orientation="horizontal", height=16,
            fg_color=C_BG_BASE,
            button_color=C_BG_FLOAT,
            button_hover_color=C_BORDER_ACT,
            command=self._canvas.xview,
        )
        self._h_scrollbar.grid(row=1, column=0, sticky="ew")

        self._canvas.configure(
            yscrollcommand=self._v_scrollbar.set,
            xscrollcommand=self._h_scrollbar.set,
        )

        # ── Inner tk.Frame embedded in canvas ─────────────────────────────
        self._inner = tk.Frame(self._canvas, bg=bg)
        self._window_id = self._canvas.create_window((0, 0),
                                                      window=self._inner,
                                                      anchor="nw")
        self._inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        # Bind scroll on canvas itself
        self._bind_scroll(self._canvas)

        # ── Title label ────────────────────────────────────────────────────
        self._title_label = ctk.CTkLabel(
            self._inner, text="",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=C_TEXT_PRI,
        )
        self._title_label.grid(row=0, column=0, columnspan=MAX_POOL,
                               pady=(10, 16))
        self._bind_scroll(self._title_label)

        # ── Widget pool ────────────────────────────────────────────────────
        self._frames:     list[ctk.CTkFrame] = []
        self._img_labels: list[ctk.CTkLabel] = []
        self._txt_labels: list[ctk.CTkLabel] = []

        for _ in range(MAX_POOL):
            frame = ctk.CTkFrame(self._inner, border_width=2,
                                 border_color=C_BORDER_SUB,
                                 fg_color=C_BG_RAISED, corner_radius=6)
            frame.original_border_color = C_BORDER_SUB

            img_lbl = ctk.CTkLabel(frame, text="")
            img_lbl.pack(padx=2, pady=2)

            txt_lbl = ctk.CTkLabel(frame, text="",
                                   font=ctk.CTkFont(size=11),
                                   text_color=C_TEXT_SEC)
            txt_lbl.pack(pady=(0, 5))

            for w in (frame, img_lbl, txt_lbl):
                self._bind_scroll(w)

            self._frames.append(frame)
            self._img_labels.append(img_lbl)
            self._txt_labels.append(txt_lbl)

    # ------------------------------------------------------------------ #
    #  Scroll wiring                                                       #
    # ------------------------------------------------------------------ #

    def _bind_scroll(self, widget):
        """Bind mouse-wheel on widget so events are forwarded to the canvas."""
        if _OS == "Windows" or _OS == "Darwin":
            widget.bind("<MouseWheel>",       self._on_vscroll, add="+")
            widget.bind("<Shift-MouseWheel>", self._on_hscroll, add="+")
        else:
            widget.bind("<Button-4>",       self._on_vscroll, add="+")
            widget.bind("<Button-5>",       self._on_vscroll, add="+")
            widget.bind("<Shift-Button-4>", self._on_hscroll, add="+")
            widget.bind("<Shift-Button-5>", self._on_hscroll, add="+")

    def _on_vscroll(self, event):
        if _OS == "Linux":
            direction = -1 if event.num == 5 else 1
        else:
            direction = -1 if event.delta < 0 else 1
        self._canvas.yview_scroll(-direction * 3, "units")

    def _on_hscroll(self, event):
        if _OS == "Linux":
            direction = 1 if event.num == 5 else -1
        else:
            direction = 1 if event.delta < 0 else -1
        self._canvas.xview_scroll(-direction * 3, "units")

    def _on_inner_configure(self, event=None):
        """Keep scrollregion in sync with actual content size."""
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event=None):
        """On resize: keep inner frame at least as wide as the canvas,
        then unconditionally reflow columns to the new width."""
        canvas_w = self._canvas.winfo_width()
        if canvas_w < 2:
            return  # not yet rendered — ignore
        inner_w = self._inner.winfo_reqwidth()
        self._canvas.itemconfigure(self._window_id,
                                   width=max(canvas_w, inner_w))
        self._maybe_reflow()

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
            ch = (self._channel_indices[slot]
                  if slot < len(self._channel_indices) else -1)
            selected = ch == inspected_channel and ch != -1
            frame.configure(
                border_color=C_ACCENT if selected
                             else getattr(frame, "original_border_color", C_BORDER_SUB),
                border_width=3 if selected else 2,
            )

    def set_channel_filter(self, filter_set: set[int] | None):
        """Set which channel indices are visible. None = show all."""
        self._channel_filter = filter_set if filter_set else None

    # ------------------------------------------------------------------ #
    #  Layout helpers                                                      #
    # ------------------------------------------------------------------ #

    def _slot_width(self) -> int:
        return self._cell_size + 16

    def _compute_cols(self) -> int:
        w = self._canvas.winfo_width()
        if w < 10:
            return 5
        return max(1, w // self._slot_width())

    def _maybe_reflow(self):
        # Always reflow on canvas resize — don't guard on cols-unchanged.
        # The guard caused the grid to stay at 1 column when winfo_width()
        # returned 1 on the first call and was never corrected afterwards.
        if self._active_count == 0:
            return
        new_cols = self._compute_cols()
        if new_cols < 1:
            return
        self._cols = new_cols
        self._reflow()

    def _reflow(self):
        cols = self._cols
        for slot in range(self._active_count):
            self._frames[slot].grid(row=(slot // cols) + 1, column=slot % cols,
                                    padx=8, pady=8)
        for slot in range(self._active_count, MAX_POOL):
            self._frames[slot].grid_remove()
        self._inner.update_idletasks()
        self._on_inner_configure()

    # ------------------------------------------------------------------ #
    #  Main update                                                         #
    # ------------------------------------------------------------------ #

    def update(
        self,
        imagedata_list: list[tuple],
        title: str,
        force_rebuild:  bool = False,
        health_metrics: dict | None = None,
        layer_name:     str  | None = None,
        cell_size:      int  | None = None,
        channel_filter: set[int] | None = None,
    ):
        if layer_name is not None:
            self.current_layer_name = layer_name

        # Merge in any new filter
        if channel_filter is not None:
            self._channel_filter = channel_filter if channel_filter else None

        # Cell size change
        if cell_size is not None and cell_size != self._cell_size:
            self._cell_size = cell_size
            force_rebuild   = True

        # Filter change
        if self._channel_filter != self._last_filter_applied:
            force_rebuild = True
        self._last_filter_applied = self._channel_filter

        # Apply filter to incoming list
        if self._channel_filter is not None:
            imagedata_list = [
                item for item in imagedata_list
                if (item[3] if len(item) == 4 else -1) in self._channel_filter
            ]

        self.last_health_metrics = health_metrics
        new_count    = min(len(imagedata_list), MAX_POOL)
        layer_changed = self.current_display_layer != title
        self._cols   = self._compute_cols()

        if not force_rebuild and not layer_changed:
            self._fast_update(imagedata_list, new_count)
            return

        # ── Full rebuild ───────────────────────────────────────────────────
        self._title_label.configure(text=title)
        self.current_display_layer = title
        self._active_count  = new_count
        self._channel_indices = []
        self._cell_data = {}

        cs   = self._cell_size
        cols = self._cols

        for slot, item in enumerate(imagedata_list[:MAX_POOL]):
            img, label_str, diag_class, original_idx = self._unpack(item)
            self._channel_indices.append(original_idx)
            if original_idx != -1:
                self._cell_data[original_idx] = (img, label_str, diag_class)

            img_np, img_resized = self._resize_image(img, cs)
            self._ctk_image_cache[slot] = img_np
            ctk_img = ctk.CTkImage(light_image=img_resized,
                                   dark_image=img_resized, size=(cs, cs))

            border_color = self._saturation_color(original_idx, health_metrics)
            selected = original_idx == self.inspected_channel and original_idx != -1

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
                frame.bind("<Leave>",
                    self._make_leave_handler(frame, original_idx))

            frame.grid(row=(slot // cols) + 1, column=slot % cols,
                       padx=8, pady=8)

        for slot in range(new_count, MAX_POOL):
            self._frames[slot].grid_remove()

        self._inner.update_idletasks()
        self._on_inner_configure()
        # Deferred reflow: winfo_width() may still be 1 right after the first
        # build because the canvas hasn't been exposed yet. Scheduling a reflow
        # 50 ms later guarantees correct column count once layout is stable.
        self.after(50, self._maybe_reflow)

    def _fast_update(self, imagedata_list: list[tuple], new_count: int):
        cs = self._cell_size
        for slot, item in enumerate(imagedata_list[:MAX_POOL]):
            if slot >= len(self._img_labels):
                break
            img          = item[0]
            original_idx = item[3] if len(item) == 4 else -1
            if original_idx != -1 and len(item) >= 3:
                self._cell_data[original_idx] = (item[0], item[1], item[2])
            img_np, img_resized = self._resize_image(img, cs)
            cached = self._ctk_image_cache[slot]
            if cached is not None and np.array_equal(img_np, cached):
                continue
            self._ctk_image_cache[slot] = img_np
            ctk_img = ctk.CTkImage(light_image=img_resized,
                                   dark_image=img_resized, size=(cs, cs))
            self._img_labels[slot].configure(image=ctk_img)

    # ------------------------------------------------------------------ #
    #  Static helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _unpack(item: tuple) -> tuple:
        if len(item) == 4:
            return item
        img, label_str, diag_class = item
        ch_str = label_str.split(" ")[-1]
        return img, label_str, diag_class, int(ch_str) if ch_str.isdigit() else -1

    @staticmethod
    def _resize_image(img: Image.Image, size: int) -> tuple[np.ndarray, Image.Image]:
        img_np = np.array(img)
        img_np = cv2.resize(img_np, (size, size), interpolation=cv2.INTER_NEAREST)
        return img_np, Image.fromarray(img_np)

    @staticmethod
    def _saturation_color(original_idx: int, health_metrics: dict | None) -> str:
        if health_metrics and "saturation" in health_metrics and original_idx != -1:
            try:
                v = health_metrics["saturation"]["per_channel"][original_idx].item()
                if v < 0.10: return C_HEALTHY
                if v < 0.30: return C_WARNING
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