#!/bin/bash
set -e
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
cd ~/GRAFT/rl && uv sync > ~/uvsync.log 2>&1
source .venv/bin/activate && bash vllm_patches/apply.sh > ~/patch.log 2>&1
echo SETUP_DONE
