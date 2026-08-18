"""
Small always-on-top pill at the bottom-center of the screen showing
recording / processing state.

Built with Tkinter rather than GTK4-layer-shell: GNOME's Wayland
compositor does not implement the wlr-layer-shell protocol that
layer-shell overlays need, so that approach silently fails on a stock
Fedora Workstation session. Tk apps run through XWayland automatically,
and XWayland *does* honor override-redirect + always-on-top + explicit
positioning, so this works out of the box on X11 and on GNOME/Wayland.
"""
import tkinter as tk


class OverlayGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.overrideredirect(True)   # no titlebar/border
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", 0.94)
        except tk.TclError:
            pass  # alpha transparency not supported everywhere

        self._idle_bg = "#1e1e1e"
        self._rec_bg = "#3a1414"

        self.label = tk.Label(
            root,
            text="",
            font=("Sans", 13),
            fg="white",
            bg=self._idle_bg,
            padx=22,
            pady=9,
        )
        self.label.pack()
        root.withdraw()

        self._screen_w = root.winfo_screenwidth()
        self._screen_h = root.winfo_screenheight()

    def _place_bottom_center(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self._screen_w - w) // 2
        y = self._screen_h - h - 60
        self.root.geometry(f"+{x}+{y}")

    def show_recording(self):
        self.label.config(
            text="🔴 جاري التسجيل... ارفع إصبعك للإيقاف",
            bg=self._rec_bg,
            fg="#ff6b6b",
        )
        self.root.configure(bg=self._rec_bg)
        self.root.deiconify()
        self._place_bottom_center()

    def show_processing(self):
        self.label.config(
            text="⏳ جاري التفريغ والتنظيم...",
            bg=self._idle_bg,
            fg="white",
        )
        self.root.configure(bg=self._idle_bg)
        self._place_bottom_center()

    def hide(self):
        self.root.withdraw()
