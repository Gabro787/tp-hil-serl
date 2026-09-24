#!/usr/bin/env bash
# Install the TP environment: Linux (NVIDIA GPU lab machine, or CPU), macOS (Apple Silicon
# MPS or CPU), or Windows via WSL2 (NVIDIA GPU passthrough -- run this *inside* WSL2, with a
# recent NVIDIA driver installed on the Windows side, not inside WSL). For native Windows
# without WSL, use install.ps1 instead (CPU-only).
# Usage:  bash install.sh            (creates the conda env "tp-hil")
#         ENV_NAME=myenv bash install.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-tp-hil}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
LEROBOT_VERSION="${LEROBOT_VERSION:-0.6.1}"   # pin the version you tested the TP with

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda first: https://docs.anaconda.com/miniconda/" >&2
  exit 1
fi
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo ">> conda env '$ENV_NAME' already exists, reusing it"
else
  echo ">> creating conda env '$ENV_NAME' (python $PYTHON_VERSION)"
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"

echo ">> ffmpeg (needed to encode dataset videos)"
conda install -y -c conda-forge "ffmpeg>=6,<8"

echo ">> LeRobot $LEROBOT_VERSION with the HIL-SERL extra"
if ! pip install "lerobot[hilserl]==${LEROBOT_VERSION}"; then
  echo ">> PyPI install failed, installing from source (tag v${LEROBOT_VERSION})"
  rm -rf .lerobot_src
  git clone --depth 1 --branch "v${LEROBOT_VERSION}" https://github.com/huggingface/lerobot.git .lerobot_src
  pip install -e ".lerobot_src[hilserl]"
fi

echo ">> analysis and input-device tools"
pip install pandas matplotlib "wandb>=0.24,<0.28" pynput pygame hidapi

cat <<EOF

Done. Next steps (in a terminal inside the desktop session):
  conda activate $ENV_NAME
  python prefetch.py        # download demos + vision encoder (once per machine)
  python check_setup.py     # verify GPU, display, rendering, controller
  wandb login
Then follow TP.md.
EOF
