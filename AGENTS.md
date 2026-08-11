# AGENTS.md

Fakarny (فكرني) — prototype pipeline for Egyptian Arabic voice notes:
`Audio -> ASR (Whisper) -> cleanup + structuring (mT5) -> bullet-point notes`.

Python scripts run directly; there is no build/lint/test tooling, no CI, no
pytest — just run each script with `python <file>.py`.

## Commands

```bash
.venv-1/bin/python app.py                # launch Gradio UI (models download from HF Hub on first run)
.venv-1/bin/python SpeachRecognition.py  # record mic audio -> recording.wav (16 kHz)
.venv-1/bin/python "Transcript Cleaning.py"  # text transcript -> organized_notes.txt
.venv-1/bin/python finetune_whisper.py   # fine-tune Whisper on dataset/ (needs GPU)
```

Notebooks (`arabicSpeechRecognition (3).ipynb`, `model.ipynb`,
`Transcript Cleaning.ipynb`) are the research versions of the same pipeline.

## Layout

- `app.py` — working prototype: `openai/whisper-small` + `csebuetnlp/mT5_multilingual_XLSum`,
  both loaded at import time and forced onto Arabic. Training code lives in the
  fine-tuning scripts below.
- `finetune_whisper.py` — active fine-tuning script, added after the original
  `finetune_roadmap.py` was deleted; still untracked in git.
- `dataset/` — **gitignored**, so it is missing after a fresh clone. Contains
  `labels.csv` (columns `path`, `label`) and `audio_files/` (3353 `.flac`, 16 kHz)
  paths in `labels.csv` are relative to `dataset/audio_files/`. Training scripts
  hard-raise `FileNotFoundError` if it is absent.
- `whisper_finetuned/` — generated checkpoints + final fine-tuned model; created
  by `finetune_whisper.py` (not committed).
- `TEST/` — scratch/practice area (gitignored), includes unrelated Colab
  downloads; not part of the product.

## Fine-tuning

Two overlapping scripts with the same data contract (`dataset/labels.csv`,
out to `whisper_finetuned/`):

- `finetune_whisper.py` — the maintained hand-rolled loop (custom DataLoader,
  WER, grad accumulation, FP16 via `torch.amp`).
- `Model.py` — older variant using HF `Seq2SeqTrainer`; the top
  of the file is a large commented-out legacy block. Prefer `finetune_whisper.py`.

Hyperparameters to respect if you tune them: `MODEL_ID = openai/whisper-small`,
`LANGUAGE = arabic`, `BATCH_SIZE=2`, `GRAD_ACCUM=4`, `lr=1e-5`, 90/10 seeded
split. Requires CUDA; falls back to slow CPU otherwise.

## Stale docs

`README.md` references `finetune_roadmap.py` and `requirements.txt`, both
deleted from the working tree (still in git HEAD). Trust the scripts, not the
README.

## Style

Scripts use module-level constants + concise directives inside speech functions.
Keep the ARABIC_DICT (`FILLER_WORDS` in `app.py`; also mirrored in
`Transcript Cleaning.py`) in sync if you extend it.