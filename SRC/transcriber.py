"""
Loads openai-whisper (torch backend) ONCE at startup and reuses it for
every recording -- reloading per-request would make each transcription
take much longer than it needs to.

No fine-tuning: this is the stock pretrained checkpoint, run as-is.
"""
import numpy as np
import torch
import whisper

import config


class Transcriber:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = config.WHISPER_MODEL
        self.model = self._load_model(self.model_name, self.device)

    def _load_model(self, name, device):
        print(f"[transcriber] loading whisper '{name}' on {device} ...")
        try:
            model = whisper.load_model(name, device=device)
            self.device = device
            print(f"[transcriber] ready on {device}")
            return model
        except torch.cuda.OutOfMemoryError:
            if device == "cuda":
                print(
                    f"[transcriber] CUDA out of memory loading '{name}' "
                    f"(6GB cards can struggle with large models). Falling back to CPU. "
                    f"Consider AWF_WHISPER_MODEL=medium or a finetuned small for GPU speed."
                )
                return self._load_model(name, "cpu")
            raise

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        audio = audio.astype(np.float32)
        fp16 = self.device == "cuda"
        try:
            result = self.model.transcribe(
                audio,
                language=config.WHISPER_LANGUAGE,
                fp16=fp16,
                task="transcribe",
            )
        except torch.cuda.OutOfMemoryError:
            print("[transcriber] CUDA OOM during transcription, retrying on CPU")
            torch.cuda.empty_cache()
            self.model = self._load_model(self.model_name, "cpu")
            result = self.model.transcribe(
                audio, language=config.WHISPER_LANGUAGE, fp16=False, task="transcribe"
            )
        return result.get("text", "").strip()
