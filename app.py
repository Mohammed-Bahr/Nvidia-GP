"""
Fakarny — فكرني
Smart note-taking app for Egyptian Arabic (PROTOTYPE)

Pipeline:
  Audio  ->  Stage 1: ASR (Whisper)  ->  Stage 2: Cleanup + Structuring (mT5 summarizer)
         ->  Structured bullet-point notes

This is a fast prototype built entirely from PRETRAINED models so you get working
results today. See finetune_roadmap.py for how to upgrade each stage later using
your project's actual datasets (MAdel121/arabic-egy-cleaned, Egyptian-ASR-MGB-3).
"""

import re
import torch
import gradio as gr
from transformers import pipeline

DEVICE = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if DEVICE == 0 else 'CPU'}")

# ---------------------------------------------------------------------------
# Stage 1: Dialect-Aware ASR (prototype = pretrained multilingual Whisper)
# ---------------------------------------------------------------------------
print("Loading ASR model (openai/whisper-small)...")
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=DEVICE,
    generate_kwargs={"language": "arabic", "task": "transcribe"},
)


def transcribe(audio_path: str) -> str:
    if audio_path is None:
        return ""
    result = asr_pipe(audio_path)
    return result["text"].strip()


# ---------------------------------------------------------------------------
# Stage 2: NLP Text Structuring
#   2a. Rule-based filler-word stripping (cheap, instant, extend this list)
#   2b. Pretrained multilingual summarizer (mT5_XLSum) -> bullet points
# ---------------------------------------------------------------------------
FILLER_WORDS = [
    "يعني", "امم", "اممم", "اه", "أه", "طب", "خلاص كده",
    "يعني كده", "بصراحة", "والله", "زي كده",
]

print("Loading structuring model (csebuetnlp/mT5_multilingual_XLSum)...")
summarizer_pipe = pipeline(
    "summarization",
    model="csebuetnlp/mT5_multilingual_XLSum",
    device=DEVICE,
)


def rule_based_cleanup(text: str) -> str:
    for w in FILLER_WORDS:
        text = re.sub(re.escape(w), "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def to_bullets(summary_text: str) -> str:
    # naive sentence split on Arabic/Latin punctuation -> bullet list
    parts = re.split(r"[.\u06D4!؟?]", summary_text)
    bullets = [p.strip() for p in parts if p.strip()]
    if not bullets:
        return f"- {summary_text.strip()}"
    return "\n".join(f"- {b}" for b in bullets)


def structure_notes(raw_text: str) -> str:
    if not raw_text:
        return ""
    cleaned = rule_based_cleanup(raw_text)
    # mT5_XLSum works on longer input; short voice notes may not compress much,
    # that's expected for a zero-shot baseline.
    try:
        out = summarizer_pipe(cleaned, max_length=120, min_length=10, do_sample=False)
        summary = out[0]["summary_text"]
    except Exception as e:
        # graceful fallback if the note is too short for the summarizer
        summary = cleaned
    return to_bullets(summary)


# ---------------------------------------------------------------------------
# Full pipeline + Gradio demo
# ---------------------------------------------------------------------------
def full_pipeline(audio):
    transcript = transcribe(audio)
    notes = structure_notes(transcript)
    return transcript, notes


demo = gr.Interface(
    fn=full_pipeline,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak or upload (Egyptian Arabic)"),
    outputs=[
        gr.Textbox(label="Raw Transcript (Stage 1: ASR)"),
        gr.Textbox(label="Structured Notes (Stage 2: NLP)"),
    ],
    title="Fakarny — فكرني",
    description=(
        "Prototype pipeline: Whisper (ASR) -> filler-word cleanup -> mT5 summarizer -> bullet notes.\n"
        "This is a pretrained-model baseline. Accuracy on Egyptian dialect will improve after "
        "fine-tuning (see finetune_roadmap.py)."
    ),
)

if __name__ == "__main__":
    demo.launch()
