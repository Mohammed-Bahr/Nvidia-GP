"""
Writes text at wherever the cursor currently has focus.

Uses ydotool (uinput-based) instead of xdotool because xdotool only works
under X11, and Fedora 43's default GNOME session is Wayland. ydotool works
on both since it injects events at the kernel level.

Requires:
    sudo dnf install ydotool
    sudo systemctl enable --now ydotool     # starts the ydotoold daemon
"""

import os
import shutil
import subprocess

import config


def type_text(text: str):
    if not text:
        return

    if shutil.which("ydotool") is None:
        print("[typer] ydotool not found. Install it with: sudo dnf install ydotool")
        print(f"[typer] Transcribed text (not typed, printed instead):\n{text}")
        return

    socket_path = os.environ.get(
        "YDOTOOL_SOCKET", f"/run/user/{os.getuid()}/.ydotool_socket"
    )
    if not os.path.exists(socket_path):
        print("[typer] ydotool socket not found. Is ydotoold running?")
        print("[typer] Try: sudo systemctl enable --now ydotool")
        print(f"[typer] Transcribed text (not typed, printed instead):\n{text}")
        return

    try:
        subprocess.run(["ydotool", "type", "--", text], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[typer] ydotool failed ({e}). Is ydotoold running?")
        print("[typer] Try: sudo systemctl enable --now ydotool")
        print(f"[typer] Transcribed text (not typed, printed instead):\n{text}")
