"""Smoke test: canonical GRAFT recipe on object_qa_respmon (heuristic monitor), 20 steps."""
from sweeps.respmon_repro_coh128_lam1_3seed import _shared, _new

runs = [{
    **_shared, **_new,
    "config": "configs/object_qa_respmon.yaml",
    "max_steps": 20,
    "routing_mode": "classic",
    "unconditional_hackable": False,
    "hack_frac": 0.5,
    "rh_detector_recall": 1.0,
    "eval_every": 10,
    "seed": 1,
    "run_name": "smoke_object_qa_gr_s1",
}]
per_gpu = 1
no_baseline = True
