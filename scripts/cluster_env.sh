#!/usr/bin/env bash
# Source from a shell with the cluster module command available.
module load python/3.11.0-gcc-13.1.0-qtdpw7v || return 1
export PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}"
unset PYENV_VERSION
export VIRTUAL_ENV="$PROJECT_DIR/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export POETRY_VIRTUALENVS_CREATE=false
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export POETRY_CACHE_DIR="$PROJECT_DIR/.cache/pypoetry"
export POETRY_CONFIG_DIR="$PROJECT_DIR/.cache/pypoetry-config"
export POETRY_DATA_DIR="$PROJECT_DIR/.cache/pypoetry-data"
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
