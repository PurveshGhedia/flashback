#!/usr/bin/env bash
# =============================================================================
# environment_setup.sh
# =============================================================================
# Creates a clean conda environment for the Lecture RAG project.
#
# Usage:
#   chmod +x environment_setup.sh
#   ./environment_setup.sh
#
# What this script does (in order):
#   1. Creates a conda env named "flashback" with Python 3.10
#   2. Installs PyTorch (CPU build) via conda — more reliable than pip for CPU
#   3. Installs ffmpeg binary via conda (required by Whisper)
#   4. Installs all pip packages from requirements.txt
#   5. Installs OpenAI CLIP directly from GitHub (no PyPI release exists)
#   6. Installs ipykernel and registers the env as a Jupyter kernel
#   7. Creates a .env template file for API keys
#   8. Prints a verification summary
#
# After running, activate with:
#   conda activate flashback
#
# To use a GPU build instead (Kaggle / university cluster):
#   See the note at the bottom of this script.
# =============================================================================

set -e  # Exit immediately on any error

ENV_NAME="flashback"
PYTHON_VERSION="3.10"

# Colours for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Colour

log()    { echo -e "${GREEN}[SETUP]${NC} $1"; }
warn()   { echo -e "${YELLOW}[WARN] ${NC} $1"; }
error()  { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# =============================================================================
# 0. Pre-flight checks
# =============================================================================
log "Checking prerequisites..."

if ! command -v conda &> /dev/null; then
    error "conda not found. Install Miniconda or Anaconda first:\n  https://docs.conda.io/en/latest/miniconda.html"
fi

CONDA_VERSION=$(conda --version)
log "Found: $CONDA_VERSION"

# Check if env already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    warn "Environment '${ENV_NAME}' already exists."
    read -p "  Delete and recreate it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        warn "Aborting. Activate existing env with: conda activate ${ENV_NAME}"
        exit 0
    fi
fi

# =============================================================================
# 1. Create environment
# =============================================================================
log "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# =============================================================================
# 2. Install PyTorch (CPU build via conda — more stable than pip on CPU)
#
#    NOTE FOR GPU USERS (Kaggle / cluster):
#    Replace this block with the correct CUDA build command from:
#    https://pytorch.org/get-started/locally/
#    Example for CUDA 12.1:
#      conda install -n $ENV_NAME pytorch torchvision torchaudio \
#        pytorch-cuda=12.1 -c pytorch -c nvidia -y
# =============================================================================
log "Installing PyTorch (CPU build)..."
conda install -n $ENV_NAME pytorch torchvision torchaudio cpuonly \
    -c pytorch -y

# =============================================================================
# 3. Install ffmpeg binary (required by Whisper for audio decoding)
#    Conda's ffmpeg is more reliable than system-level installs on macOS/Linux
# =============================================================================
log "Installing ffmpeg via conda-forge..."
conda install -n $ENV_NAME -c conda-forge ffmpeg -y

# =============================================================================
# 4. Install pip packages from requirements.txt
# =============================================================================
log "Installing pip packages from requirements.txt..."

# Run pip inside the conda env without activating it
# (activation inside a script is unreliable)
CONDA_PREFIX=$(conda info --base)
PIP="${CONDA_PREFIX}/envs/${ENV_NAME}/bin/pip"

$PIP install --upgrade pip
$PIP install -r requirements.txt

# =============================================================================
# 5. Install OpenAI CLIP (no PyPI package — must install from GitHub)
# =============================================================================
log "Installing OpenAI CLIP from GitHub..."
$PIP install git+https://github.com/openai/CLIP.git

# =============================================================================
# 6. Install ipykernel and register as a Jupyter kernel
#    This makes the env selectable in Jupyter Lab/Notebook and VS Code.
#    Kernel will appear as "Flashback (flashback)" in the kernel picker.
# =============================================================================
log "Installing ipykernel and registering Jupyter kernel..."
$PIP install ipykernel

PYTHON="${CONDA_PREFIX}/envs/${ENV_NAME}/bin/python"
$PYTHON -m ipykernel install --user \
    --name "flashback" \
    --display-name "Flashback (flashback)"

log "Kernel registered — select 'Flashback (flashback)' in Jupyter / VS Code."

# =============================================================================
# 7. Create .env template (safe to run multiple times — won't overwrite)
# =============================================================================
if [ ! -f ".env" ]; then
    log "Creating .env template..."
    cat > .env << 'ENVTEMPLATE'
# -------------------------------------------------------
# API Keys — fill these in, never commit this file to git
# -------------------------------------------------------
GEMINI_API_KEY=your_gemini_api_key_here

# Optional overrides (defaults set in config.py)
# WHISPER_MODEL_SIZE=large       # Use on Kaggle / GPU cluster
# LECTURE_RAG_DEVICE=cuda        # Force GPU
ENVTEMPLATE
    log ".env template created. Fill in your GEMINI_API_KEY before running the app."
else
    warn ".env already exists — not overwriting."
fi

# Make sure .env is in .gitignore
if [ ! -f ".gitignore" ]; then
    cat > .gitignore << 'GITIGNORE'
.env
data/videos/
data/frames/
data/transcripts/
data/chroma_db/
__pycache__/
*.pyc
.DS_Store
GITIGNORE
    log ".gitignore created."
fi

# =============================================================================
# 8. Verify installation
# =============================================================================
log "Verifying installation..."

PYTHON="${CONDA_PREFIX}/envs/${ENV_NAME}/bin/python"

echo ""
echo "--------------------------------------------------------------"
$PYTHON - << 'VERIFY'
import sys
print(f"  Python        : {sys.version.split()[0]}")

import torch
print(f"  PyTorch       : {torch.__version__} | CUDA: {torch.cuda.is_available()}")

import cv2
print(f"  OpenCV        : {cv2.__version__}")

import whisper
print(f"  Whisper       : OK ({whisper.__version__ if hasattr(whisper,'__version__') else 'installed'})")

import clip
print(f"  CLIP          : OK")

import chromadb
print(f"  ChromaDB      : {chromadb.__version__}")

import sentence_transformers
print(f"  SentenceTransformers: {sentence_transformers.__version__}")

import scenedetect
print(f"  PySceneDetect : {scenedetect.__version__}")

import streamlit
print(f"  Streamlit     : {streamlit.__version__}")

import google.generativeai
print(f"  google-generativeai: OK")

print()
print("  ✓ All packages verified successfully.")
VERIFY
echo "--------------------------------------------------------------"
echo ""

# =============================================================================
# Done
# =============================================================================
log "Environment setup complete!"
echo ""
echo "  To activate:       conda activate ${ENV_NAME}"
echo "  To run app:        streamlit run app.py"
echo "  To deactivate:     conda deactivate"
echo "  Jupyter kernel:    'Flashback (flashback)' — available in Jupyter Lab, Notebook, and VS Code"
echo ""
echo "  Remember to fill in your GEMINI_API_KEY in .env before running."
echo ""
