"""
Central configuration for Arabic Whisper Flow.
Everything here can be overridden with an environment variable of the
same name prefixed with AWF_, e.g.:
    AWF_HOTKEY=KEY_F9 python main.py
"""
import os

# ---------------------------------------------------------------------------
# Hotkey (push-to-talk): hold to record, release to transcribe.
# Must be a valid evdev key name. Some safe choices that rarely collide
# with normal typing: KEY_RIGHTALT, KEY_RIGHTCTRL, KEY_F9, KEY_PAUSE
# ---------------------------------------------------------------------------
HOTKEY = os.environ.get("AWF_HOTKEY", "KEY_F8")

# Set this explicitly (e.g. "/dev/input/event5") if auto-detection picks
# the wrong device or you have multiple keyboards. Run `evtest` to find it.
KEYBOARD_DEVICE_PATH = os.environ.get("AWF_KEYBOARD_DEVICE", "/dev/input/event4")

# ---------------------------------------------------------------------------
# Whisper (transcription) settings
# ---------------------------------------------------------------------------
# Fine-tuned whisper-small from ../whisper_finetuned (converted to openai-whisper
# .pt format by convert_finetuned_to_openai.py). Override with any stock name
# ("large-v3", "large-v3-turbo", "medium", ...) via AWF_WHISPER_MODEL.
_FINETUNED_PT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "whisper_finetuned", "model.pt",
)
WHISPER_MODEL = os.environ.get("AWF_WHISPER_MODEL", _FINETUNED_PT)
WHISPER_LANGUAGE = "ar"

# ---------------------------------------------------------------------------
# Cleanup (second pass) -- local mT5 model used only to fix
# punctuation/spacing/sentence breaks in the raw transcript; it must not add
# or remove information. Override with AWF_MT5_MODEL.
# ---------------------------------------------------------------------------
MT5_MODEL = os.environ.get("AWF_MT5_MODEL", "csebuetnlp/mT5_multilingual_XLSum")
# Pipeline device: -1 = CPU, 0 = first GPU. CPU by default so the cleanup
# model never competes with Whisper's GPU memory. Use AWF_FORMATTER_DEVICE=0
# to force GPU.
FORMATTER_DEVICE = int(os.environ.get("AWF_FORMATTER_DEVICE", "-1"))
# If the model can't be loaded or errors out, the raw Whisper transcript is
# typed instead of failing the whole action.
SKIP_CLEANUP = os.environ.get("AWF_SKIP_CLEANUP", "0") == "1"

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
