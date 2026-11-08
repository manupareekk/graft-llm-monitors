# GRAFT with imperfect LLM monitors

GRAFT ([Ward et al., 2026](https://arxiv.org/abs/2610.17062)) removes reward hacking by routing gradients from monitor-flagged RL samples into a FORGET adapter that is deleted at deployment. The paper uses heuristic monitors. This repo tests GRAFT with small self-hosted LLM judges (Qwen3-4B, Qwen3-1.7B) as the monitor.

**Paper draft:** [`paper/paper.pdf`](paper/paper.pdf)

## Main findings (30 runs, SmolLM2-135M on `object_qa`, 3 seeds per cell)

| Monitor | On-policy recall | GRAFT, deployed hack rate | Reward penalty | Filtering |
|---|---|---|---|---|
| None (no intervention) | — | 82.5% | — | — |
| Hand-written rule (blind to "indeed") | 56% | 22.8% ± 22.7 (fails in the seed that uses "indeed") | 23.4% | 47.5% |
| Qwen3-4B judge | 42–61%, FPR 13–16% | **0.5% ± 0.4** | 0.1% | 58.6% |
| Qwen3-1.7B judge | ~0.5% | 63.5% | 62.4% | 79.7% |

- GRAFT tolerates a noisy judge, but it gives no protection for a hack form the monitor never flags.
- Reward penalty pushes the policy into the monitor's blind spot. In a sorting task, the policy switched from verbatim copying (flagged by the judge) to reformatted copying (not flagged) and kept hacking at 97%.
- The 1.7B judge flagged 10–28% of synthetic hacks but ~0.5% of on-policy hacks.
- Preliminary (1–2 seeds): GRAFT starts to break at ~5% recall, and a 45% false-positive rate cuts task accuracy from 0.45 to 0.30.

## Layout

```
paper/            LaTeX source, figures, PDF
graft_changes/    changes to jnward/GRAFT (base commit in BASE_COMMIT)
  tracked_changes.patch   rl/rh_detectors.py (LLM-judge monitors, controlled degradations, judge prompts), rl/train.py
  rl/configs/             monitor configs (4B / 1.7B judges, keep10, blind-spot, fp30, sorting)
  rl/sweeps/              sweep definitions
  rl/tests/               unit tests for the judge monitor
analysis/         aggregation, per-word recall/FPR, figures, offline judge benchmark
infra/            box setup, judge server, sweep launcher
data/             per_seed.csv and routing_eval.jsonl + run_config.yaml for every completed run
```

## Reproducing

```bash
git clone https://github.com/jnward/GRAFT && cd GRAFT
git checkout $(cat ../graft-llm-monitors/graft_changes/BASE_COMMIT)
git apply ../graft-llm-monitors/graft_changes/tracked_changes.patch
cp -r ../graft-llm-monitors/graft_changes/rl/* rl/
cd rl && uv sync && source .venv/bin/activate && bash vllm_patches/apply.sh
vllm serve Qwen/Qwen3-4B --port 8100 --gpu-memory-utilization 0.4 --max-model-len 1024 --max-num-seqs 512 &
JUDGE_URL=http://localhost:8100/v1 WANDB_MODE=offline LLMMON_MONITORS=q4t50 LLMMON_PER_GPU=1 LLMMON_MAXTOK=8000 \
  python sweep.py --name llmmon --config sweeps/llmmon_object_qa.py --no_baseline
python ../../graft-llm-monitors/analysis/analyze.py output/llmmon
```

Each run takes ~3–4 h on one 24 GB L4 shared with the judge. Run one GRAFT job per GPU, because two jobs next to the judge run out of memory.

## License

MIT, same as the upstream GRAFT repository. The upstream code is © Jake Ward and Shawn Hu.
