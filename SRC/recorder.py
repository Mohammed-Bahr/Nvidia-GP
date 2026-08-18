"""
Microphone recorder. start() / stop() are called from the hotkey thread.
Audio is buffered in memory as float32 mono at config.SAMPLE_RATE, which
is exactly what whisper.transcribe() expects -- no file I/O needed.
"""
import threading

import numpy as np
import sounddevice as sd

import config


class Recorder:
    def __init__(self, samplerate=config.SAMPLE_RATE, channels=config.CHANNELS):
        self.samplerate = samplerate
        self.channels = channels
        self._frames = []
        self._stream = None
        self._lock = threading.Lock()
        self._recording = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[recorder] status: {status}")
        with self._lock:
            if self._recording:
                self._frames.append(indata.copy())

    def start(self):
        with self._lock:
            if self._recording:
                return
            self._frames = []
            self._recording = True
        try:
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            print(f"[recorder] failed to open microphone: {e}")
            with self._lock:
                self._recording = False

    def stop(self) -> np.ndarray:
        with self._lock:
            was_recording = self._recording
            self._recording = False
        if not was_recording:
            return np.zeros((0,), dtype=np.float32)

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            if not self._frames:
                return np.zeros((0,), dtype=np.float32)
            audio = np.concatenate(self._frames, axis=0)
            self._frames = []
        return audio.flatten().astype(np.float32)
