"""Smoke test: GRAFT with the Qwen3-4B LLM-judge monitor (needs JUDGE_URL), 20 steps."""
from sweeps.llmmon_smoke import runs as _r
runs = [{**_r[0], "config": "configs/object_qa_llmj_q4_t50.yaml", "run_name": "smoke_q4t50_gr_s1"}]
per_gpu = 1
no_baseline = True
