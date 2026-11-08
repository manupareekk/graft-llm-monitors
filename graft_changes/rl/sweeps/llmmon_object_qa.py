"""GRAFT with LLM-judge monitors (paper Limitation (i)) on object_qa.

Monitors: the paper's heuristic (configs/object_qa_respmon.yaml, blind to "indeed") and
self-hosted LLM judges (configs/object_qa_llmj_*.yaml, llm_judge_logprob on JUDGE_URL).
Arms per monitor: GRAFT (canonical recipe), reward penalty 2.0, filtering. No-intervention
does not use the monitor, so it runs once. Set LLMMON_MONITORS / LLMMON_SEEDS to subset.
"""
import os
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _new

_GR_ONLY_KEYS = (
    "coherence", "coherence_rh_mode", "coh_samples_per_rollout",
    "rh_detector_verifies_retain_samples", "rh_detector_retain_recall", "routing_mode",
)
_plain = {k: v for k, v in _shared.items() if k not in _GR_ONLY_KEYS}
_common = {"max_steps": 1000, "unconditional_hackable": False, "hack_frac": 0.5,
           "rh_detector_recall": 1.0, "eval_every": 25, "eval_full_completions": True,
           "max_tokens_per_microbatch": int(os.environ.get("LLMMON_MAXTOK", "16000"))}  # fit runs on a 24GB L4

MONITORS = {
    "heur": "configs/object_qa_respmon.yaml",
    "q4t50": "configs/object_qa_llmj_q4_t50.yaml",
    "q17t20": "configs/object_qa_llmj_q17_t20.yaml",
    "q4keep10": "configs/object_qa_llmj_q4_keep10.yaml",
    "q4blind": "configs/object_qa_llmj_q4_blindindeed.yaml",
    "q4fp30": "configs/object_qa_llmj_q4_fp30.yaml",
}
_mons = os.environ.get("LLMMON_MONITORS", ",".join(MONITORS)).split(",")
_seeds = [int(s) for s in os.environ.get("LLMMON_SEEDS", "1,2,3").split(",")]
_arms = os.environ.get("LLMMON_ARMS", "gr,rp2,filter,dn").split(",")

runs = []
for s in _seeds:
    if "dn" in _arms:
        runs.append({**_plain, **_common, "config": MONITORS["heur"], "routing_mode": "none",
                     "seed": s, "run_name": f"dn_s{s}"})
    for m in _mons:
        cfg = MONITORS[m]
        if "gr" in _arms:
            runs.append({**_shared, **_new, **_common, "config": cfg, "routing_mode": "classic",
                         "seed": s, "run_name": f"{m}_gr_s{s}"})
        if "rp2" in _arms:
            runs.append({**_plain, **_common, "config": cfg, "routing_mode": "none",
                         "reward_penalty_baseline": True, "reward_penalty_amount": 2.0,
                         "seed": s, "run_name": f"{m}_rp2_s{s}"})
        if "filter" in _arms:
            runs.append({**_plain, **_common, "config": cfg, "routing_mode": "none",
                         "filter_baseline": True, "seed": s, "run_name": f"{m}_filter_s{s}"})

per_gpu = int(os.environ.get("LLMMON_PER_GPU", "3"))
no_baseline = True
