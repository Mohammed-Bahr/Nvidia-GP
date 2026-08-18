# Arabic Whisper Flow

Push-to-talk Arabic dictation: hold a hotkey, speak, release — Whisper
transcribes it, a local LLM cleans up punctuation/organization, and the
result gets typed wherever your cursor is. A small overlay at the bottom
of the screen shows recording/processing state.

## How it works

```
hotkey held  → recorder.py captures mic audio
hotkey up    → transcriber.py runs Whisper large (torch, no fine-tuning)
             → formatter.py sends the raw text to your local Ollama model
               to fix punctuation/spacing/sentence breaks only
             → typer_out.py types the final text at the cursor via ydotool
gui.py       → shows a small bar at the bottom of the screen throughout
```

## One-time setup (Fedora 43)

### 1. System packages
```bash
sudo dnf install python3-devel portaudio-devel ffmpeg ydotool
```
`ffmpeg` is required by Whisper internally. `portaudio-devel` is needed
to build `sounddevice`.

### 2. ydotool daemon (needed to actually type the text)
```bash
sudo systemctl enable --now ydotool
```
If that unit doesn't exist on your install, check `ydotool --help` for
`ydotoold` and run it as a user service instead — instructions vary
slightly by ydotool version.

### 3. Let your user read the keyboard device (needed for the hotkey)
```bash
sudo usermod -aG input $USER
```
**Log out and back in** for the group change to apply. Without this,
`hotkey.py` will raise a clear error telling you what's missing.

### 4. Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5. PyTorch with CUDA (for your RTX 3060)
The plain `torch` from requirements.txt may install CPU-only. Install
the CUDA build explicitly — check https://pytorch.org/get-started/locally/
for the command matching your CUDA version, e.g.:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 6. Ollama model for the cleanup pass
Make sure Ollama is running and has a model pulled, then point
`config.py` (or `AWF_OLLAMA_MODEL`) at it — e.g. if you have
`gemma3:12b` already pulled, that's the default.

## Running it
```bash
source .venv/bin/activate
python main.py
```
First run will download the Whisper model (large-v3 is ~3GB) — this
only happens once.

Hold **Right Alt** (default) anywhere on your system, speak in Arabic,
release. The text appears wherever your cursor currently is.

## Important: your GPU has 6GB VRAM

`large-v3` in fp16 wants roughly 10GB. On a 3060, it may fail to load
on GPU — if that happens, `transcriber.py` **automatically falls back
to CPU** so it still works, just slower. Options if you want GPU speed:

```bash
AWF_WHISPER_MODEL=large-v3-turbo python main.py   # ~6GB, fits comfortably
# or
AWF_WHISPER_MODEL=medium python main.py           # smaller still, faster
```

Both stay well within "Whisper via torch, no fine-tuning" — just a
smaller checkpoint size.

## Configuration
All of this is in `config.py`, overridable via `AWF_*` env vars:
- `AWF_HOTKEY` — evdev key name (default `KEY_RIGHTALT`)
- `AWF_WHISPER_MODEL` — `large-v3`, `large-v3-turbo`, `medium`, etc.
- `AWF_OLLAMA_MODEL`, `AWF_OLLAMA_HOST`
- `AWF_SKIP_CLEANUP=1` — type the raw Whisper output directly, skip the LLM pass
- `AWF_KEYBOARD_DEVICE` — force a specific `/dev/input/eventX` if auto-detect picks wrong

## Troubleshooting
- **"No accessible keyboard device exposes the configured hotkey"** — you're
  not in the `input` group yet, or need to re-login after adding yourself.
  Run `evtest` to double check which `/dev/input/eventX` is your keyboard.
- **Nothing gets typed** — check `ydotoold` is actually running
  (`systemctl status ydotool`) and that `ydotool` is on your PATH.
- **GUI bar doesn't appear** — Tk runs via XWayland on GNOME/Wayland,
  which should work by default. If you're on a Wayland-only session
  with no XWayland at all, the overlay window won't render; everything
  else (transcription, typing) still works.
- **CUDA out of memory** — expected on 6GB with `large-v3`; the app
  falls back to CPU automatically, or switch models as above.

## Known limitation
Right now this is push-to-talk with a single global hotkey and no
config UI — deliberately minimal so the core loop (record → transcribe
→ clean → type) is solid first. Natural next steps: a settings panel,
auto-detecting silence to stop recording, and swapping in
`faster-whisper` (CTranslate2, int8) if you want much lower VRAM use
and faster inference without touching model quality much.
