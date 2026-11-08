"""GRAFT vs reward penalty vs no intervention on the framed sort env with a Qwen3-4B LLM-judge monitor."""
import os
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _new

_GR_ONLY_KEYS = (
    "coherence", "coherence_rh_mode", "coh_samples_per_rollout",
    "rh_detector_verifies_retain_samples", "rh_detector_retain_recall", "routing_mode",
)
_plain = {k: v for k, v in _shared.items() if k not in _GR_ONLY_KEYS}
_base = {"config": "configs/sorting_copy_framed_llmj_q4.yaml", "unconditional_hackable": False,
         "hack_frac": 0.5, "rh_detector_recall": 1.0, "max_steps": 1000, "eval_every": 25,
         "eval_at_start": True, "eval_full_completions": True}
_seeds = [int(s) for s in os.environ.get("LLMMON_SEEDS", "1,2,3").split(",")
          if not os.path.exists(os.path.expanduser(f"~/skip_sort_s{s}"))]
_arms = os.environ.get("LLMMON_ARMS", "gr,rp2,dn").split(",")

runs = []
for s in _seeds:
    if "gr" in _arms:
        runs.append({**_shared, **_new, **_base, "routing_mode": "classic", "seed": s, "run_name": f"sortq4_gr_s{s}"})
    if "rp2" in _arms:
        runs.append({**_plain, **_base, "routing_mode": "none", "reward_penalty_baseline": True,
                     "reward_penalty_amount": 2.0, "seed": s, "run_name": f"sortq4_rp2_s{s}"})
    if "dn" in _arms:
        runs.append({**_plain, **_base, "routing_mode": "none", "seed": s, "run_name": f"sortq4_dn_s{s}"})

per_gpu = int(os.environ.get("LLMMON_PER_GPU", "1"))
no_baseline = True
