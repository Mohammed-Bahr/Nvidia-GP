import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)

# ==========================================
# Configuration
# ==========================================

MODEL_ID = "openai/whisper-large-v3"
AUDIO_FILE = "recording.wav"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {device}")

# ==========================================
# Load Whisper Model
# ==========================================

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
)

model.to(device)

# ==========================================
# Load Processor
# ==========================================

processor = AutoProcessor.from_pretrained(MODEL_ID)

# ==========================================
# Create Speech Recognition Pipeline
# ==========================================

pipe = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=0 if device == "cuda" else -1,
)

# ==========================================
# Transcribe Audio
# ==========================================

result = pipe(
    AUDIO_FILE,
    chunk_length_s=30,
    batch_size=8,
    generate_kwargs={
        "language": "arabic",
        "task": "transcribe",
    },
)

# ==========================================
# Output
# ==========================================

transcript = result["text"]

print("\n========== Transcript ==========\n")
print(transcript)