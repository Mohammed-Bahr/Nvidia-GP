# AGENTS.md

Two things live in this repo: the **active product** in `files/`, and the
older Fakarny research pipeline at the root (mostly scratch).

## `files/` — Arabic Whisper Flow (active product)

Push-to-talk Arabic dictation: hold a hotkey → record mic → Whisper
transcribes → local Ollama cleans up punctuation → `ydotool` types the
result at the cursor. Entry point is `files/main.py`.

- Run with the shared venv from anywhere in the repo: `.venv-1/bin/python files/main.py`
- Config is `files/config.py`; every setting overridable via `AWF_*` env vars
  (`AWF_WHISPER_MODEL`, `AWF_HOTKEY`, `AWF_OLLAMA_MODEL`, `AWF_SKIP_CLEANUP=1`, ...).
- Uses **openai-whisper**, not transformers: `whisper.load_model(path)` accepts
  a file path but only in openai `.pt` format — an HF
  `WhisperForConditionalGeneration` checkpoint (safetensors) will NOT load.
- Default model is the fine-tuned checkpoint at `whisper_finetuned/model.pt`,
  produced from the HF checkpoint by `files/convert_finetuned_to_openai.py`.
  Re-run the converter after any re-finetune. Gotcha: `decoder.proj_out.weight`
  is tied to `token_embedding` and must be **omitted** from the saved state dict.
- Transcription runs CUDA fp16 with automatic CPU fallback on OOM (6GB RTX 3060;
  large-v3 OOMs, the fine-tuned small fits fine).
- Runtime deps that must exist on the host: user in the `input` group (evdev
  hotkey — missing group = the hard "No accessible keyboard device" error),
  Ollama running with a pulled model, `ydotoold` daemon + `ffmpeg` + portaudio.
  The cleanup and typing steps degrade gracefully (print instead) when their
  deps are missing; the hotkey does not.

## Root — Fakarny pipeline + scratch

Old Gradio prototype `app.py` (whisper-small + mT5_XLSum, models load at import),
`Transcript Cleaning.py`, `ASR.py` (loads `whisper_finetuned/` via transformers),
plus experimental scripts (`WisperLarge.py`, `WisperModel.py`, `APP.py`,
`Deepseek.py`, `wisperlargefinetuned.py`) and notebooks in `Bachup/`. Treat
these as scratch unless explicitly asked about.

## Environment

- Use `.venv-1/bin/python` for everything: it has openai-whisper, transformers
  5.14.1, torch 2.13, evdev, sounddevice, librosa, datasets, evaluate.
- `.venv/` is a bare uv venv (`pyproject.toml`/`uv.lock` are for it) with almost
  nothing installed.
- No build/lint/test/CI tooling — scripts run directly.

## Fine-tuning

- Active script: `finetune_whisper_large_egyptian_manual_loop.py` — Whisper-large-v3,
  frozen encoder, hand-rolled loop (grad accumulation, WER, FP16). Streams
  `MAdel121/arabic-egy-cleaned` from HF Hub — it does **not** use the local
  `dataset/` folder. Output goes to `whisper_large_finetuned_egy_manual/`
  (not created yet).
- `whisper_finetuned/` (gitignored) is the OLD whisper-small fine-tune, built by
  the now-deleted `finetune_whisper.py`; `dataset/` (gitignored, `labels.csv` +
  `audio_files/`) fed only that old pipeline.
- Deleted scripts: `finetune_whisper.py`, `Model.py`, `finetune_roadmap.py`.
  Root `README.md` still cites the last one + `requirements.txt` — stale, trust
  the scripts. `files/README.md` is also stale about the default model (says
  large-v3; config now defaults to `whisper_finetuned/model.pt`).

## Git state

Working tree is intentionally messy: `files/`, the large finetune script, root
`main.py` etc. are untracked; several files are staged for deletion.
`core.42257` (a 10GB core dump) is gitignored — never stage it.

## Style

Keep the filler-word list in sync: `FILLER_WORDS` in root `app.py` is mirrored
as `ARABIC_DICT` in `Transcript Cleaning.py`.