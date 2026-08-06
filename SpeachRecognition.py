import sounddevice as sd
import soundfile as sf
import numpy as np
import threading

SAMPLE_RATE = 16000
CHANNELS = 1

def record_until_enter(output_path: str = "recording.wav"):
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
    )

    print("Recording... press Enter to stop.")
    stream.start()
    input()  # blocks until you press Enter
    stream.stop()
    stream.close()

    audio = np.concatenate(frames, axis=0)
    sf.write(output_path, audio, SAMPLE_RATE)
    print(f"Saved {len(audio) / SAMPLE_RATE:.1f}s of audio to {output_path}")

if __name__ == "__main__":
    record_until_enter("recording.wav")
    # or "recording.flac" for FLAC
    