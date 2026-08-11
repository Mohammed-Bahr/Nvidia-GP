"""
transcribe.py — Use our fine-tuned Whisper model to transcribe ONE Arabic audio file.

What this script does, step by step:
  1. Load the fine-tuned Whisper model + processor from whisper_finetuned/
  2. Load the audio file and resample it to 16kHz mono (what Whisper expects)
  3. Convert audio -> log-mel features (same preprocessing used during training)
  4. Generate the transcript using the model (with settings that avoid repetition loops)
  5. Decode the tokens back into Arabic text
  6. Save the result to a .txt file

Run with:  python ASR.py

Requirements (install if you don't have them yet):
  pip install librosa torch transformers
"""

import librosa
import torch
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ---------------------------------------------------------------------------
# 1. CONFIG — change these for different runs
# ---------------------------------------------------------------------------
MODEL_DIR = "./whisper_finetuned"       # folder containing our fine-tuned model
# AUDIO_PATH = "./Attachments/test2.wav"         # the audio file we want to transcribe
AUDIO_PATH = "101.wav"         # the audio file we want to transcribe
OUTPUT_TXT_PATH = "output.txt"        # where the resulting text will be saved

LANGUAGE = "arabic"
TASK = "transcribe"

# ---------------------------------------------------------------------------
# 2. LOAD MODEL + PROCESSOR
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print(f"Loading fine-tuned model from: {MODEL_DIR}")
processor = WhisperProcessor.from_pretrained(MODEL_DIR, language=LANGUAGE, task=TASK)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()  # inference mode: turns off dropout and other training-only behavior

# ---------------------------------------------------------------------------
# 3. LOAD & PREPROCESS THE AUDIO
# ---------------------------------------------------------------------------
print(f"Loading audio: {AUDIO_PATH}")
# librosa.load automatically resamples to 16kHz and converts to mono float32,
# which is exactly the format Whisper's feature extractor expects.
audio_array, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True)
print(f"Audio duration: {len(audio_array) / sr:.2f} seconds")

# Convert the raw waveform into a log-mel spectrogram (same step used during training)
inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features.to(device)

# ---------------------------------------------------------------------------
# 4. GENERATE THE TRANSCRIPT
# ---------------------------------------------------------------------------
# These are the special tokens that tell the model "output language = Arabic,
# task = transcribe" (not translate).
forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)

print("Transcribing...")
with torch.no_grad():  # no gradients needed during inference -> saves memory
    predicted_ids = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids,
        max_new_tokens=200,        # cap how long the output can be
        num_beams=5,               # beam search explores several candidate outputs, usually more accurate than greedy decoding
        no_repeat_ngram_size=3,    # forbids repeating the same 3-word sequence -> stops loops like "أحاول أحاول أحاول"
        repetition_penalty=1.3,    # extra penalty for reusing the same tokens, further discourages loops
    )

# ---------------------------------------------------------------------------
# 5. DECODE TOKENS -> TEXT
# ---------------------------------------------------------------------------
transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

print("\n--- Transcript ---")
print(transcript)

# ---------------------------------------------------------------------------
# 6. SAVE TO A TEXT FILE
# ---------------------------------------------------------------------------
Path(OUTPUT_TXT_PATH).write_text(transcript, encoding="utf-8")
print(f"\nSaved transcript to: {OUTPUT_TXT_PATH}")
