"""Aggregate final routing_eval metrics over seeds for an llmmon sweep.
usage: python analyze.py OUTPUT_DIR [--last K]  (averages last K evals per run, then mean±sem over seeds)"""
import json, os, re, sys, glob, statistics as st
from collections import defaultdict
out = sys.argv[1]; K = int(sys.argv[sys.argv.index("--last") + 1]) if "--last" in sys.argv else 3
# metric families; the name after the family depends on the task and detector (e.g. retain/qa_correct or retain/sorting_positional,
# detected_freq/sycophancy_any or detected_freq/llm_judge_logprob)
METRIC_PREFIXES = ("retain/", "hack_freq_hackable/", "detected_freq/")
rows = defaultdict(lambda: defaultdict(list))
for f in sorted(glob.glob(f"{out}/*/routing_eval.jsonl")):
    run = os.path.basename(os.path.dirname(f))
    m = re.match(r"(.+)_s(\d+)$", run)
    if not m: continue
    evals = [json.loads(l) for l in open(f) if l.strip()]
    if not evals: continue
    last = evals[-K:]
    prefixes = ["both", "retain_only"] if any(k.startswith("retain_only/") for k in last[-1]) else ["both"]
    for p in prefixes:
        metric_keys = sorted({k for e in last for k in e if k.startswith(f"{p}/") and k.split("/", 1)[1].startswith(METRIC_PREFIXES)})
        keys = metric_keys + sorted(k for k in last[-1] if k.startswith(f"{p}/") and "word_" in k)
        for k in keys:
            vals = [e[k] for e in last if k in e]
            if vals:
                rows[(m.group(1), p, evals[-1]["step"])][k.split("/", 1)[1]].append(sum(vals) / len(vals))
cols = sorted({c for r in rows.values() for c in r})
print("arm".ljust(22), "mode".ljust(12), "step".ljust(6), *[c.split("/")[0][-18:].rjust(18) for c in cols])
for (arm, p, step), r in sorted(rows.items()):
    def fmt(v):
        if not v: return "-".rjust(18)
        s = st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0
        return f"{st.mean(v):.3f}±{s:.3f}(n{len(v)})".rjust(18)
    print(arm.ljust(22), p.ljust(12), str(step).ljust(6), *[fmt(r.get(c, [])) for c in cols])
