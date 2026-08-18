"""
Arabic Whisper Flow -- entry point.

Flow:
  hold hotkey  -> gui shows "recording", mic starts capturing
  release key  -> gui shows "processing", audio is transcribed (Whisper
                  large, torch, no fine-tuning) on a background thread so
                  the hotkey listener never blocks
                  -> transcript is cleaned/organized by a local Ollama model
                  -> final text is typed at the current cursor position
                  -> gui hides

See README.md for one-time setup (permissions, ydotool, ffmpeg, etc).
"""
import threading
import tkinter as tk

import config
from recorder import Recorder
from transcriber import Transcriber
from formatter import clean_transcript
from typer_out import type_text
from hotkey import HotkeyListener
from gui import OverlayGUI


def main():
    root = tk.Tk()
    gui = OverlayGUI(root)

    recorder = Recorder()

    print("[main] loading Whisper model, this can take a while on first run...")
    transcriber = Transcriber()  # loaded once, reused for every recording

    def process_recording(audio):
        raw_text = transcriber.transcribe(audio)
        if not raw_text:
            print("[main] no speech detected")
            root.after(0, gui.hide)
            return
        print(f"[main] raw transcript: {raw_text}")
        final_text = clean_transcript(raw_text)
        print(f"[main] final transcript: {final_text}")
        type_text(final_text)
        root.after(0, gui.hide)

    def on_press():
        root.after(0, gui.show_recording)
        recorder.start()

    def on_release():
        audio = recorder.stop()
        root.after(0, gui.show_processing)
        threading.Thread(target=process_recording, args=(audio,), daemon=True).start()

    listener = HotkeyListener(config.HOTKEY, on_press, on_release)
    listener.start()

    print(f"[main] Ready. Hold '{config.HOTKEY}' to talk, release to transcribe & type.")
    print("[main] Press Ctrl+C in this terminal to quit.")

    try:
        root.mainloop()
    finally:
        listener.stop()


if __name__ == "__main__":
    main()
