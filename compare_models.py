"""
compare_models.py — Transcribe ONE Arabic audio file with two Whisper models
(base vs fine-tuned) and print both transcripts side by side.

Run with:  python compare_models.py
"""

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE_MODEL_ID = "openai/whisper-base"
FINETUNED_MODEL_DIR = "./whisper_finetuned"
AUDIO_PATH = "101.wav"

LANGUAGE = "arabic"
TASK = "transcribe"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

audio_array, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True)
print(f"Audio: {AUDIO_PATH} ({len(audio_array) / sr:.2f} seconds)\n")

def transcribe(model_or_dir, label):
    print(f"Loading {label} from: {model_or_dir}")
    processor = WhisperProcessor.from_pretrained(model_or_dir, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(model_or_dir)
    model.to(device)
    model.eval()

    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=200,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
        )

    transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    return transcript

base_result = transcribe(BASE_MODEL_ID, "whisper-base")
finetuned_result = transcribe(FINETUNED_MODEL_DIR, "whisper_finetuned")

print("\n" + "-" * 60)
print("--- whisper-base ---")
print(base_result)
print(f"({len(base_result)} chars)")

print("\n--- whisper_finetuned ---")
print(finetuned_result)
print(f"({len(finetuned_result)} chars)")
print("-" * 60)