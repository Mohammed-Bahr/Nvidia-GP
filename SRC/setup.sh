#!/usr/bin/env bash
#
# Arabic Whisper Flow — one-shot install + run.
#
# Installs every host dependency the push-to-talk app needs and sets up the
# Python environment. The mT5 cleanup model downloads from Hugging Face Hub
# automatically on first launch. Then runs files/main.py.
#
# Usage:
#   ./setup.sh                # install everything, then run the app
#   ./setup.sh --install-only # install everything, skip launching the app
#
# Idempotent: safe to re-run; already-installed things are skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
APP_DIR="$SCRIPT_DIR"                     # files/
VENV="$REPO_ROOT/.venv-1"                 # shared venv (has CUDA torch etc.)

step() { printf '\n==> %s\n' "$1"; }
die()  { printf 'ERROR: %s\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
step "Installing system packages (dnf)..."
sudo dnf install -y python3-devel portaudio-devel ffmpeg ydotool

# ---------------------------------------------------------------------------
# 2. ydotool daemon (types the text at the cursor)
# ---------------------------------------------------------------------------
step "Enabling ydotool daemon..."
if ! systemctl is-active --quiet ydotool 2>/dev/null; then
    sudo systemctl enable --now ydotool 2>/dev/null || true
fi

SOCKET="/run/user/$(id -u)/.ydotool_socket"
if [ ! -S "$SOCKET" ]; then
    echo "  Socket missing; starting ydotoold directly..."
    nohup ydotoold >/dev/null 2>&1 &
    sleep 1
fi

if [ ! -S "$SOCKET" ]; then
    echo "  WARNING: ydotool socket still not present. Typing will not work."
    echo "  Make sure your user has uinput access and ydotoold is running."
fi

# ---------------------------------------------------------------------------
# 3. input group (lets the app read the keyboard for the hotkey)
# ---------------------------------------------------------------------------
step "Adding $USER to the 'input' group..."
if ! id -nG "$USER" | tr ' ' '\n' | grep -qx input; then
    sudo usermod -aG input "$USER"
    echo "  Added. Log out & back in (or run: newgrp input) before the hotkey works."
fi

# ---------------------------------------------------------------------------
# 4. Python environment
# ---------------------------------------------------------------------------
step "Preparing Python environment..."
if [ ! -x "$VENV/bin/python" ]; then
    echo "  No shared venv at $VENV — creating it..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install -r "$APP_DIR/requirements.txt"
else
    echo "  Reusing existing venv: $VENV"
    # Ensure requirements are present even if the venv predates them.
    "$VENV/bin/pip" install -r "$APP_DIR/requirements.txt"
fi
PY="$VENV/bin/python"

step "Checking CUDA build of torch..."
if "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  torch with CUDA OK"
else
    echo "  torch is CPU-only — installing CUDA build (cu121)..."
    "$VENV/bin/pip" uninstall -y torch
    "$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu121
fi

# ---------------------------------------------------------------------------
# 5. Run
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--install-only" ]; then
    step "Install done. Launch the app with: $PY $APP_DIR/main.py"
    exit 0
fi

step "Launching Arabic Whisper Flow..."
cd "$APP_DIR"
exec "$PY" main.py