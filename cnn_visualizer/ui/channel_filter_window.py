"""
Channel Filter Window.

A CTkToplevel that lets the user pick which channels are visible in the grid.

Layout:
  ┌─────────────────────────────────────────────┐
  │  Channel Filter                    [×  Close]│
  ├─────────────────────────────────────────────┤
  │  Quick select:  [text box  e.g. 0-5, 10 ]   │
  │  [Select All]  [Deselect All]  [From Text ▶] │
  ├─────────────────────────────────────────────┤
  │  ┌─────────────────────────────────────────┐ │
  │  │  [Ch 0][Ch 1][Ch 2] …  (scrollable)    │ │
  │  └─────────────────────────────────────────┘ │
  ├─────────────────────────────────────────────┤
  │  64 selected of 64   [Reset]  [Apply]        │
  └─────────────────────────────────────────────┘

Usage:
    win = ChannelFilterWindow(
        parent,
        num_channels   = 64,
        current_filter = None,           # None = all selected
        on_apply       = my_callback,    # called with set[int] | None
    )
    win.show()   # opens / brings to front
"""
from __future__ import annotations
import tkinter as tk
import customtkinter as ctk
from .theme import *


def _parse_range_text(raw: str, num_channels: int) -> set[int] | None:
    """
    Parse a range/list string into a set of valid channel indices.
    "0-5, 10, 15-20" -> {0,1,2,3,4,5,10,15,16,17,18,19,20}
    "" or "all"      -> None  (meaning all channels)
    """
    raw = raw.strip()
    if not raw or raw.lower() in ("all", "*"):
        return None
    result: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            try:
                lo = max(0, int(parts[0].strip()))
                hi = min(num_channels - 1, int(parts[1].strip()))
                result.update(range(lo, hi + 1))
            except ValueError:
                pass
        else:
            try:
                idx = int(token)
                if 0 <= idx < num_channels:
                    result.add(idx)
            except ValueError:
                pass
    return result if result else None


def _filter_to_text(filter_set: set[int] | None, num_channels: int) -> str:
    """Convert a filter set back to a compact range string."""
    if filter_set is None or len(filter_set) == num_channels:
        return ""
    indices = sorted(filter_set)
    if not indices:
        return ""
    parts = []
    start = indices[0]
    end   = indices[0]
    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            parts.append(f"{start}-{end}" if end > start else str(start))
            start = end = idx
    parts.append(f"{start}-{end}" if end > start else str(start))
    return ", ".join(parts)


class ChannelFilterWindow(ctk.CTkToplevel):
    """
    Channel picker window. Instantiated once, shown/hidden via show()/hide().
    Call refresh(num_channels, current_filter) when the active layer changes.
    """

    TOGGLE_W  = 58   # px width of each channel toggle button
    TOGGLE_H  = 32   # px height
    COLS      = 8    # toggles per row

    def __init__(self, parent, num_channels: int = 64,
                 current_filter: set[int] | None = None,
                 on_apply=None):
        super().__init__(parent)
        self.title("Channel Filter")
        self.geometry("560x520")
        self.minsize(420, 380)
        self.configure(fg_color=C_BG_DEEP)
        self.protocol("WM_DELETE_WINDOW", self.hide)

        self._num_channels  = num_channels
        self._on_apply      = on_apply
        self._toggle_vars:  list[tk.BooleanVar] = []
        self._toggle_btns:  list[ctk.CTkButton] = []
        self._applying      = False   # guard against recursive _sync_count calls

        self._build_ui()
        self._rebuild_toggles(num_channels, current_filter)
        self.withdraw()

    # ------------------------------------------------------------------ #
    #  Build UI                                                            #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ── Header ─────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, fg_color=C_BG_RAISED, corner_radius=0)
        hdr.grid(row=0, column=0, sticky="ew")

        ctk.CTkLabel(hdr, text="Channel Filter",
                     font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=C_TEXT_PRI).pack(side="left", padx=16, pady=10)

        ctk.CTkButton(hdr, text="✕", width=32,
                      fg_color="transparent", hover_color=C_CRITICAL,
                      text_color=C_TEXT_SEC, command=self.hide,
                      ).pack(side="right", padx=10, pady=8)

        # ── Quick-select toolbar ────────────────────────────────────────
        toolbar = ctk.CTkFrame(self, fg_color=C_BG_BASE, corner_radius=0)
        toolbar.grid(row=1, column=0, sticky="ew", padx=0, pady=0)
        toolbar.grid_columnconfigure(0, weight=1)

        # Text entry row
        entry_row = ctk.CTkFrame(toolbar, fg_color="transparent")
        entry_row.pack(fill="x", padx=12, pady=(10, 4))
        entry_row.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(entry_row, text="Quick select:",
                     font=ctk.CTkFont(size=10), text_color=C_TEXT_SEC,
                     ).grid(row=0, column=0, sticky="w")

        self._text_entry = ctk.CTkEntry(
            entry_row,
            placeholder_text='e.g. 0-5, 10, 20-31  (blank = all)',
            fg_color=C_BG_FLOAT, border_color=C_BORDER_SUB,
            text_color=C_TEXT_PRI, height=30,
        )
        self._text_entry.grid(row=1, column=0, sticky="ew", padx=(0, 6))

        ctk.CTkButton(
            entry_row, text="Apply text ▶", width=96, height=30,
            fg_color=C_ACCENT, hover_color=C_BG_FLOAT,
            font=ctk.CTkFont(size=11),
            command=self._apply_text,
        ).grid(row=1, column=1)

        # Bulk action buttons
        btn_row = ctk.CTkFrame(toolbar, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(0, 10))

        for label, cmd in [
            ("Select All",   self._select_all),
            ("Deselect All", self._deselect_all),
        ]:
            ctk.CTkButton(
                btn_row, text=label, height=26, width=100,
                fg_color=C_BG_FLOAT, hover_color=C_BG_RAISED,
                border_width=1, border_color=C_BORDER_SUB,
                text_color=C_TEXT_SEC, font=ctk.CTkFont(size=11),
                command=cmd,
            ).pack(side="left", padx=(0, 6))

        # ── Toggle grid (scrollable) ────────────────────────────────────
        self._scroll = ctk.CTkScrollableFrame(
            self, fg_color=C_BG_BASE,
            scrollbar_button_color=C_BG_FLOAT,
            scrollbar_button_hover_color=C_BORDER_ACT,
        )
        self._scroll.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)

        # ── Footer ─────────────────────────────────────────────────────
        footer = ctk.CTkFrame(self, fg_color=C_BG_RAISED, corner_radius=0)
        footer.grid(row=3, column=0, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)

        self._count_label = ctk.CTkLabel(
            footer, text="", font=ctk.CTkFont(size=11), text_color=C_TEXT_SEC,
        )
        self._count_label.grid(row=0, column=0, sticky="w", padx=16, pady=10)

        ctk.CTkButton(
            footer, text="Apply", width=80, height=30,
            fg_color=C_ACCENT, hover_color=C_BG_FLOAT,
            command=self._apply_and_close,
        ).grid(row=0, column=2, padx=(6, 12), pady=8)

        ctk.CTkButton(
            footer, text="Reset", width=70, height=30,
            fg_color=C_BG_FLOAT, hover_color=C_BG_RAISED,
            border_width=1, border_color=C_BORDER_SUB,
            text_color=C_TEXT_SEC,
            command=self._select_all,
        ).grid(row=0, column=1, padx=4, pady=8)

    # ------------------------------------------------------------------ #
    #  Toggle grid management                                              #
    # ------------------------------------------------------------------ #

    def _rebuild_toggles(self, num_channels: int,
                         current_filter: set[int] | None):
        """Destroy old toggles and create new ones for num_channels."""
        for w in self._scroll.winfo_children():
            w.destroy()
        self._toggle_vars.clear()
        self._toggle_btns.clear()
        self._num_channels = num_channels

        cols = self.COLS
        for ch in range(num_channels):
            on = current_filter is None or ch in current_filter
            var = tk.BooleanVar(value=on)
            var.trace_add("write", lambda *_, i=ch: self._on_toggle(i))

            btn = ctk.CTkButton(
                self._scroll,
                text=f"{ch}",
                width=self.TOGGLE_W, height=self.TOGGLE_H,
                font=ctk.CTkFont(size=11),
                corner_radius=4,
                fg_color=C_ACCENT if on else C_BG_FLOAT,
                hover_color=C_BG_RAISED,
                border_width=1,
                border_color=C_BORDER_ACT if on else C_BORDER_SUB,
                text_color=C_TEXT_PRI if on else C_TEXT_MUT,
                command=lambda i=ch: self._click_toggle(i),
            )
            r, c = ch // cols, ch % cols
            btn.grid(row=r, column=c, padx=3, pady=3, sticky="w")

            self._toggle_vars.append(var)
            self._toggle_btns.append(btn)

        self._sync_count()

    def _click_toggle(self, idx: int):
        """Flip one channel's state."""
        var = self._toggle_vars[idx]
        new_val = not var.get()
        var.set(new_val)  # trace fires _on_toggle

    def _on_toggle(self, idx: int):
        """Update button appearance when a var changes."""
        on = self._toggle_vars[idx].get()
        btn = self._toggle_btns[idx]
        btn.configure(
            fg_color=C_ACCENT if on else C_BG_FLOAT,
            border_color=C_BORDER_ACT if on else C_BORDER_SUB,
            text_color=C_TEXT_PRI if on else C_TEXT_MUT,
        )
        if not self._applying:
            self._sync_count()

    def _sync_count(self):
        """Update the footer count label and text entry to reflect state."""
        selected = sum(v.get() for v in self._toggle_vars)
        total    = self._num_channels
        self._count_label.configure(
            text=f"{selected} of {total} channels selected",
            text_color=C_ACCENT if selected < total else C_TEXT_SEC,
        )
        # Keep the text entry in sync with toggle state
        current = self._get_filter_set()
        self._text_entry.delete(0, "end")
        text = _filter_to_text(current, total)
        if text:
            self._text_entry.insert(0, text)

    # ------------------------------------------------------------------ #
    #  Bulk actions                                                        #
    # ------------------------------------------------------------------ #

    def _select_all(self):
        self._applying = True
        for var in self._toggle_vars:
            var.set(True)
        for btn in self._toggle_btns:
            btn.configure(fg_color=C_ACCENT, border_color=C_BORDER_ACT,
                          text_color=C_TEXT_PRI)
        self._applying = False
        self._text_entry.delete(0, "end")
        self._sync_count()

    def _deselect_all(self):
        self._applying = True
        for var in self._toggle_vars:
            var.set(False)
        for btn in self._toggle_btns:
            btn.configure(fg_color=C_BG_FLOAT, border_color=C_BORDER_SUB,
                          text_color=C_TEXT_MUT)
        self._applying = False
        self._sync_count()

    def _apply_text(self):
        """Parse the text entry and set toggles to match."""
        raw    = self._text_entry.get()
        parsed = _parse_range_text(raw, self._num_channels)
        self._applying = True
        for i, var in enumerate(self._toggle_vars):
            on = parsed is None or i in parsed
            var.set(on)
            self._toggle_btns[i].configure(
                fg_color=C_ACCENT if on else C_BG_FLOAT,
                border_color=C_BORDER_ACT if on else C_BORDER_SUB,
                text_color=C_TEXT_PRI if on else C_TEXT_MUT,
            )
        self._applying = False
        self._sync_count()

    # ------------------------------------------------------------------ #
    #  Result extraction                                                   #
    # ------------------------------------------------------------------ #

    def _get_filter_set(self) -> set[int] | None:
        """Return current selection as set[int] or None if all selected."""
        selected = {i for i, v in enumerate(self._toggle_vars) if v.get()}
        if len(selected) == self._num_channels:
            return None   # all on = no filter
        return selected if selected else None

    def _apply_and_close(self):
        if self._on_apply:
            self._on_apply(self._get_filter_set())
        self.hide()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def show(self):
        self.deiconify()
        self.lift()
        self.focus_force()

    def hide(self):
        self.withdraw()

    def refresh(self, num_channels: int, current_filter: set[int] | None):
        """
        Called when the active layer changes and channel count may differ.
        Rebuilds the toggle grid to match the new channel count.
        """
        if num_channels != self._num_channels:
            self._rebuild_toggles(num_channels, current_filter)
        else:
            # Same count — just update toggle states
            self._applying = True
            for i, var in enumerate(self._toggle_vars):
                on = current_filter is None or i in current_filter
                var.set(on)
                self._toggle_btns[i].configure(
                    fg_color=C_ACCENT if on else C_BG_FLOAT,
                    border_color=C_BORDER_ACT if on else C_BORDER_SUB,
                    text_color=C_TEXT_PRI if on else C_TEXT_MUT,
                )
            self._applying = False
            self._sync_count()