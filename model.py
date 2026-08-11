"""
model.py — Fine-tune Whisper (openai/whisper-small) on Arabic audio.

Pipeline:
  labels.csv (audio path + transcript)
      -> Hugging Face Dataset
      -> train/test split (90/10)
      -> audio -> log-mel features, text -> token IDs
      -> Seq2SeqTrainer fine-tunes the base Whisper model
      -> fine-tuned weights + processor saved to whisper_finetuned/

Run with:  python model.py   (requires CUDA GPU; falls back to slow CPU)
"""

# Standard library imports
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

# Third-party imports
import evaluate                     # Hugging Face metrics library (used for WER)
import torch
from datasets import Audio, Dataset  # Hugging Face datasets library
from transformers import (
    Seq2SeqTrainer,             # High-level training loop provided by HF Transformers
    Seq2SeqTrainingArguments,   # Container for all trainer configuration/hyperparameters
    WhisperForConditionalGeneration,  # The Whisper model for speech-to-text (seq2seq)
    WhisperProcessor,           # Combines the feature extractor (audio) + tokenizer (text)
)

# ---------------------------------------------------------------------------
# 1. PATH & MODEL CONFIGURATION
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent   # Project root = directory containing this file
DATA_DIR = ROOT / "dataset"             # Directory holding the training data
AUDIO_DIR = DATA_DIR / "audio_files"    # Folder with the raw audio files
LABELS_PATH = DATA_DIR / "labels.csv"   # CSV mapping each audio file to its transcript
OUTPUT_DIR = ROOT / "whisper_finetuned" # Where the fine-tuned model/checkpoints are saved

MODEL_ID = "openai/whisper-small"       # Base pretrained model we fine-tune from
LANGUAGE = "arabic"                     # Target language for decoding prompts
TASK = "transcribe"                     # Task type passed to the model (transcribe, not translate)


# ---------------------------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING HELPERS
# ---------------------------------------------------------------------------

def load_label_pairs(labels_path: Path):
    """Read labels.csv and return a list of {audio_path, transcript} dicts.

    Each CSV row has a 'path' column (relative to audio_files/) and a 'label'
    column (the reference transcript). Empty rows are skipped and missing
    audio files raise an error so bad data is caught immediately.
    """
    rows = []
    with labels_path.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)      # Reads each row into a dict keyed by header
        for row in reader:
            filename = row["path"].strip()
            text = row["label"].strip()
            if not filename or not text:
                continue                      # Skip blank/malformed rows
            audio_path = AUDIO_DIR / filename
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            rows.append({"audio": str(audio_path), "text": text})
    return rows


def prepare_dataset(dataset, processor):
    """Convert raw audio + text examples into model-ready tensors.

    For every example we:
      - turn the waveform into log-mel spectrogram features (input_features), and
      - tokenize the transcript into token IDs (labels) that Whisper learns to generate.
    """
    def prepare_batch(batch):
        audio = batch["audio"]                # {'array': waveform, 'sampling_rate': ...}
        # Extract the log-mel features from the raw audio samples
        encoded_audio = processor(audio["array"], sampling_rate=audio["sampling_rate"])
        batch["input_features"] = encoded_audio.input_features[0]

        # Tokenize the reference transcript into input IDs used as the decoder targets
        batch["labels"] = processor(text=batch["text"]).input_ids
        return batch

    return dataset.map(
        prepare_batch,
        remove_columns=["audio", "text"],     # Drop the raw fields; only features + labels remain
        batched=False,                        # Process one example at a time (simpler)
    )


def compute_metrics(pred):
    """Compute the Word Error Rate (WER) after each evaluation step.

    WER = (substitutions + insertions + deletions) / total words in reference.
    Lower is better; 0 means a perfect transcript. Used to monitor quality
    during training.
    """
    pred_ids = pred.predictions   # Model-generated token IDs
    label_ids = pred.label_ids    # Ground-truth token IDs (-100 marks padding to be ignored)

    # Decode both predictions and references back into readable text
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    # Restore padding markers (-100) to real pad tokens so decoding works correctly
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# ---------------------------------------------------------------------------
# 3. DATA COLLATOR (dynamic padding for speech-to-text batches)
# ---------------------------------------------------------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pads each batch to the longest sequence in that batch.

    Audio becomes log-mel features (already fixed length), while transcripts
    have variable lengths and must be padded so the model can process them
    as a batched tensor.
    """
    processor: Any                            # The WhisperProcessor (holds tokenizer + feature extractor)

    def __call__(self, features: list[dict[str, Union[list[int], torch.Tensor]]]) -> dict[str, torch.Tensor]:
        # input_features are already fixed-length log-mel arrays -> simple stack.
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # labels need padding to the longest sequence in the batch.
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding token id with -100 so it's ignored by the loss function
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If a BOS token was appended by the tokenizer during a previous call,
        # strip it here since the model adds it automatically.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# 4. MAIN TRAINING PIPELINE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Detect the compute device (GPU preferred, CPU fallback)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Guard: fail fast if the labels file is missing
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing labels file: {LABELS_PATH}")

    # ---- Load & prepare the dataset ----
    raw_rows = load_label_pairs(LABELS_PATH)
    if not raw_rows:
        raise ValueError("No labeled examples found in dataset/labels.csv")

    # Turn the list of dicts into an HF Dataset and cast 'audio' to a decoded Audio column
    raw_dataset = Dataset.from_list(raw_rows)
    raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Hold out 10% as validation (fixed seed = reproducible split)
    split = raw_dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Dataset size: train={len(split['train'])}, eval={len(split['test'])}")

    # ---- Load base model + processor (feature extractor & tokenizer) ----
    # The processor converts audio<->log-mel and text<->tokens
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=LANGUAGE, task=TASK)
    # The Whisper model weights we'll fine-tune
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    # Force the decoder to start with the Arabic language/task prompt tokens
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    # Don't suppress any tokens during generation (keep the full vocabulary)
    model.config.suppress_tokens = []
    # Required when using gradient checkpointing during training
    model.config.use_cache = False

    # ---- Preprocess train & eval sets into features + labels ----
    train_dataset = prepare_dataset(split["train"], processor)
    eval_dataset = prepare_dataset(split["test"], processor)

    # Dynamically pads variable-length batches to the longest item in each batch
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    # Metric used to score the model during evaluation
    wer_metric = evaluate.load("wer")

    # ---- Training hyperparameters ----
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),          # Where checkpoints/logs are written
        per_device_train_batch_size=1,       # Examples per forward/backward on each GPU
        per_device_eval_batch_size=1,        # Examples per evaluation batch
        # Accumulate gradients over 8 steps => effective batch size of 1*8 = 8
        # (lets us train with a larger effective batch on limited VRAM)
        gradient_accumulation_steps=8,
        # Recompute activations on the backward pass to trade compute for memory
        gradient_checkpointing=True,
        eval_strategy="steps",               # Evaluate on a fixed cadence (not end of epoch)
        eval_steps=100,                      # Run evaluation every 100 training steps
        logging_steps=50,                    # Print metrics every 50 steps
        save_steps=100,                      # Save a checkpoint every 100 steps
        save_total_limit=2,                  # Keep only the 2 most recent checkpoints
        learning_rate=1e-5,                  # Small LR: gently adapt pretrained weights
        num_train_epochs=3,                  # Passes over the whole training set
        fp16=torch.cuda.is_available(),      # Mixed precision to speed up + save memory on GPU
        predict_with_generate=True,          # Use full generation for metrics instead of greedy logits
        generation_max_length=128,           # Max tokens the model may output during eval
        dataloader_num_workers=0,            # No worker processes = less peak memory
        remove_unused_columns=False,         # Keep 'input_features' & 'labels' for the collator
        report_to="none",                    # Don't push metrics to any external tracker
    )

    # ---- Build the trainer ----
    trainer = Seq2SeqTrainer(
        model=model,                         # The Whisper model to train
        args=training_args,                  # Hyperparameters defined above
        train_dataset=train_dataset,         # Preprocessed training examples
        eval_dataset=eval_dataset,           # Preprocessed validation examples
        processing_class=processor.tokenizer,# Tokenizer used for decoding/generation
        data_collator=data_collator,         # Handles dynamic padding per batch
        compute_metrics=compute_metrics,     # Callback that computes WER at eval time
    )

    # ---- Run training & save the final model ----
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))          # Save the fine-tuned weights
    processor.save_pretrained(str(OUTPUT_DIR))   # Save the tokenizer + feature extractor
    print(f"Fine-tuned model saved to: {OUTPUT_DIR}")


    # # ---- This in case i stoped training and want to resume from a last checkpoint -----
    # trainer.train(
    #     resume_from_checkpoint=str(OUTPUT_DIR / "checkpoint-theLastNumber")
    # )

    # trainer.save_model(str(OUTPUT_DIR))
    # processor.save_pretrained(str(OUTPUT_DIR))

    # print(f"Fine-tuned model saved to: {OUTPUT_DIR}")
