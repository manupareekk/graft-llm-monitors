#!/bin/bash
# usage: serve_judge.sh MODEL PORT GPU_MEM
source ~/GRAFT/rl/.venv/bin/activate
exec vllm serve "$1" --port "$2" --gpu-memory-utilization "$3" --max-model-len 1024 --max-num-seqs 512
