#!/bin/bash
# usage: run_sweep.sh NAME SWEEPFILE [extra sweep.py args]; env LLMMON_* subsets runs
cd ~/GRAFT/rl && source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline JUDGE_URL=${JUDGE_URL:-http://localhost:8100/v1}
name=$1; cfg=$2; shift 2
exec python -u sweep.py --name "$name" --config "$cfg" --no_baseline "$@"
