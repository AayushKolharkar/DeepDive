"""
Pattern Gallery window.

Shows all synthesized channel patterns that have been saved, with metadata.
Patterns are stored in a `patterns/` directory next to main.py.
Metadata is stored in `patterns/metadata.json`.
"""
from __future__ import annotations
import json
import os
import time
import uuid
from pathlib import Path
from datetime import datetime

import customtkinter as ctk
from PIL import Image

from .theme import *

# Directory relative to wherever the app is run from
PATTERNS_DIR = Path("patterns")


def _ensure_dir():
    PATTERNS_DIR.mkdir(exist_ok=True)


def _meta_path() -> Path:
    return PATTERNS_DIR / "metadata.json"


def load_metadata() -> list[dict]:
    """Load all saved pattern metadata. Returns [] if file doesn't exist."""
    p = _meta_path()
    if not p.exists():
        return []
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return []


def _save_metadata(entries: list[dict]):
    _ensure_dir()
    with open(_meta_path(), "w") as f:
        json.dump(entries, f, indent=2)


def save_pattern(
    pil_image:  Image.Image,
    model:      str,
    layer:      str,
    channel:    int,
    iterations: int,
    device:     str,
) -> dict:
    """
    Save a synthesized pattern image and append its metadata.

    Returns the metadata dict for the saved entry.
    """
    _ensure_dir()
    entry_id  = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_layer = layer.replace(".", "_")
    filename  = f"{timestamp}_{entry_id}_{safe_layer}_ch{channel}.png"
    filepath  = str(PATTERNS_DIR / filename)

    pil_image.save(filepath)

    entry = {
        "id":         entry_id,
        "timestamp":  timestamp,
        "model":      model,
        "layer":      layer,
        "channel":    channel,
        "iterations": iterations,
        "device":     device,
        "filepath":   filepath,
    }

    entries = load_metadata()
    entries.append(entry)
    _save_metadata(entries)

    return entry


def delete_pattern(entry_id: str):
    """Delete a pattern by id (removes file + metadata entry)."""
    entries = load_metadata()
    to_delete = [e for e in entries if e["id"] == entry_id]
    for e in to_delete:
        try:
            os.remove(e["filepath"])
        except FileNotFoundError:
            pass
    entries = [e for e in entries if e["id"] != entry_id]
    _save_metadata(entries)


# ── Gallery window ─────────────────────────────────────────────────────────────

class PatternGallery(ctk.CTkToplevel):
    """
    Standalone gallery window that displays all saved synthesized patterns.
    Instantiated once by app.py and shown/hidden via show()/hide().
    """

    THUMB_SIZE = 160
    COLS       = 4

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Pattern Gallery")
        self.geometry("800x600")
        self.minsize(500, 400)
        self.configure(fg_color=C_BG_DEEP)

        # Don't destroy on close — just hide
        self.protocol("WM_DELETE_WINDOW", self.hide)

        self._card_refs: list = []   # keep CTkImage refs alive

        self._build_ui()
        self.withdraw()              # start hidden

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # ── Header ─────────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color=C_BG_RAISED, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")

        ctk.CTkLabel(
            header,
            text="Synthesized Channel Patterns",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=C_TEXT_PRI,
        ).pack(side="left", padx=16, pady=12)

        ctk.CTkButton(
            header, text="✕ Close", width=80,
            fg_color=C_BG_FLOAT, hover_color=C_CRITICAL,
            text_color=C_TEXT_PRI, command=self.hide,
        ).pack(side="right", padx=12, pady=8)

        ctk.CTkButton(
            header, text="⟳ Refresh", width=90,
            fg_color=C_BG_FLOAT, hover_color=C_BG_RAISED,
            text_color=C_TEXT_SEC, command=self.refresh,
        ).pack(side="right", padx=4, pady=8)

        # ── Scrollable content area ────────────────────────────────────────
        self._scroll = ctk.CTkScrollableFrame(self, fg_color=C_BG_BASE)
        self._scroll.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        self._empty_label = ctk.CTkLabel(
            self._scroll,
            text="No patterns synthesized yet.\nClick 'Synthesize Pattern' in the Channel Inspector.",
            font=ctk.CTkFont(size=13),
            text_color=C_TEXT_MUT,
            justify="center",
        )

    def show(self):
        self.deiconify()
        self.lift()
        self.refresh()

    def hide(self):
        self.withdraw()

    def refresh(self):
        """Reload metadata and redraw all cards."""
        # Destroy existing cards
        for w in self._scroll.winfo_children():
            w.destroy()
        self._card_refs.clear()

        entries = load_metadata()

        if not entries:
            self._empty_label = ctk.CTkLabel(
                self._scroll,
                text="No patterns synthesized yet.\nClick 'Synthesize Pattern' in the Channel Inspector.",
                font=ctk.CTkFont(size=13),
                text_color=C_TEXT_MUT,
                justify="center",
            )
            self._empty_label.pack(expand=True, pady=80)
            return

        # Show newest first
        entries = sorted(entries, key=lambda e: e["timestamp"], reverse=True)

        cols = self.COLS
        for i, entry in enumerate(entries):
            r = i // cols
            c = i % cols
            card = self._make_card(entry)
            card.grid(row=r, column=c, padx=10, pady=10, sticky="n")

    def _make_card(self, entry: dict) -> ctk.CTkFrame:
        """Build one pattern card widget."""
        card = ctk.CTkFrame(
            self._scroll,
            fg_color=C_BG_RAISED,
            border_width=1,
            border_color=C_BORDER_SUB,
            corner_radius=8,
        )

        # Thumbnail
        thumb_lbl = ctk.CTkLabel(card, text="")
        thumb_lbl.pack(padx=8, pady=(8, 4))

        try:
            img = Image.open(entry["filepath"]).convert("RGB")
            img = img.resize((self.THUMB_SIZE, self.THUMB_SIZE), Image.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img,
                                   size=(self.THUMB_SIZE, self.THUMB_SIZE))
            self._card_refs.append(ctk_img)
            thumb_lbl.configure(image=ctk_img)
        except Exception:
            thumb_lbl.configure(text="[missing]", text_color=C_TEXT_MUT)

        # Metadata labels
        ts = entry.get("timestamp", "")
        try:
            dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
            ts_str = dt.strftime("%b %d %Y  %H:%M")
        except Exception:
            ts_str = ts

        for text, color in [
            (f"{entry.get('model', '?')}",            C_TEXT_PRI),
            (f"{entry.get('layer', '?')}",            C_TEXT_SEC),
            (f"Channel {entry.get('channel', '?')}",  C_ACCENT),
            (f"{entry.get('iterations', '?')} iters  ·  {entry.get('device', '?')}", C_TEXT_MUT),
            (ts_str,                                   C_TEXT_MUT),
        ]:
            ctk.CTkLabel(
                card, text=text,
                font=ctk.CTkFont(size=10),
                text_color=color,
            ).pack(anchor="w", padx=10, pady=1)

        # Delete button
        entry_id = entry["id"]
        ctk.CTkButton(
            card, text="Delete", height=24,
            fg_color=C_BG_FLOAT,
            hover_color=C_CRITICAL,
            text_color=C_TEXT_SEC,
            font=ctk.CTkFont(size=10),
            command=lambda eid=entry_id: self._on_delete(eid),
        ).pack(fill="x", padx=8, pady=(4, 8))

        return card

    def _on_delete(self, entry_id: str):
        delete_pattern(entry_id)
        self.refresh()