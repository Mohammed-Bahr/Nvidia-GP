"""
Second pass over the raw Whisper transcript: fixes punctuation, spacing,
and sentence organization using a local mT5 model. It is instructed to
never add, remove, or reinterpret content -- only clean it up.

If the model can't be loaded or errors, we fall back to the raw transcript
so the app never just does nothing.
"""
import os

import config

# Point this at any local/dir mT5 checkpoint (or a fine-tuned one).
# Overridable with AWF_MT5_MODEL, e.g. google/mt5-base.
MT5_MODEL = os.environ.get("AWF_MT5_MODEL", "csebuetnlp/mT5_multilingual_XLSum")

# CPU by default so it never competes with Whisper's GPU memory;
# use AWF_FORMATTER_DEVICE=cuda to force the GPU.
FORMATTER_DEVICE = int(os.environ.get("AWF_FORMATTER_DEVICE", "-1"))

print(f"[formatter] Loading mT5 ({MT5_MODEL}) on device {FORMATTER_DEVICE}...")
_pipe = None
try:
    from transformers import pipeline

    _pipe = pipeline(
        "summarization",
        model=MT5_MODEL,
        device=FORMATTER_DEVICE,
    )
except Exception as e:
    print(f"[formatter] failed to load mT5 ({e}), will use raw transcript")


def clean_transcript(raw_text: str) -> str:
    if not raw_text.strip():
        return raw_text
    if config.SKIP_CLEANUP:
        return raw_text
    if _pipe is None:
        return raw_text

    try:
        out = _pipe(raw_text, max_length=256, min_length=10, do_sample=False)
        return out[0]["summary_text"].strip() or raw_text
    except Exception as e:
        print(f"[formatter] mT5 failed ({e}), using raw transcript instead")
        return raw_text