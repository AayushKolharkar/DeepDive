"""
Main entry point for launching the CNN Visualizer Application.
"""
from __future__ import annotations
import customtkinter as ctk
from ui.app import CNNVisualizerApp

def main():
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")

    app = CNNVisualizerApp()
    app.mainloop()

if __name__ == "__main__":
    main()
