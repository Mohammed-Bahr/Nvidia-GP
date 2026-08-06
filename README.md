# Fakarny (فكرني) — Prototype

Smart note-taking pipeline for Egyptian Arabic voice notes.
`Audio -> ASR (Whisper) -> Cleanup + Structuring (mT5) -> Bullet-point notes`

This is a **fully working prototype using pretrained models only** — no
training required to get results today. `finetune_roadmap.py` documents how
to upgrade it with your project's actual fine-tuning plan later.

## 1. Setup (Google Colab recommended — free NVIDIA GPU)

```bash
pip install -r requirements.txt
```

If running locally, make sure you have an NVIDIA GPU + CUDA-enabled PyTorch
installed (or it'll fall back to CPU, which will be slow for Whisper).

## 2. Run

```bash
python app.py
```

This launches a Gradio web UI. Record or upload an Egyptian Arabic voice
note; you'll get:
- **Raw Transcript** — Whisper's output (Stage 1)
- **Structured Notes** — filler-word-stripped, summarized bullet points (Stage 2)

## 3. Known limitations (expected, and fine — this is a baseline)

- Whisper-small was **not** fine-tuned on Egyptian dialect, so expect
  mistranscriptions on heavy slang / code-switching. This is exactly the gap
  your project's Stage 1 fine-tuning is meant to close.
- mT5_XLSum is a **generic news summarizer**, not trained on voice-note-style
  rambling speech, so structuring quality is a rough baseline, not final
  quality. Stage 2 fine-tuning (see roadmap) fixes this.
- Filler-word list is short — extend `FILLER_WORDS` in `app.py` as you find
  more examples in real recordings.

## 4. Next steps (for your enhancement phase)

See `finetune_roadmap.py` for:
- LoRA fine-tuning Whisper on `MAdel121/arabic-egy-cleaned` / `Egyptian-ASR-MGB-3`
- Fine-tuning AraT5/mT5 on a small corrected bullet-notes dataset
- Metrics to report (WER/CER for ASR, ROUGE for structuring)

## Files

| File | Purpose |
|---|---|
| `app.py` | Working prototype, pretrained models, Gradio UI |
| `requirements.txt` | Dependencies |
| `finetune_roadmap.py` | Reference code for Phase 2 fine-tuning (not run automatically) |
# Nvidia-GP
