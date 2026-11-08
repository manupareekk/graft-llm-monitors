"""Judge-vs-reward confusion from eval_samples.jsonl: how often the monitor flags samples that do / don't
contain rewarded hack words, split by question type and by which hack word appears.
usage: python judge_confusion.py RUN_DIR [MODE]   (MODE: both | retain_only | forget_only; default both)"""
import json, sys, collections
run, mode = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "both")
L = [json.loads(l) for l in open(f"{run}/eval_samples.jsonl")]
L = [x for x in L if x.get("mode", "both") == mode and "completion" in x]
D = next(k for k in L[0] if k.startswith("score/detected_freq/"))
H = "score/hack_freq/sycophancy_any"
WORDS = ["indeed", "actually", "exactly", "quite"]
print(f"{'step':>5} {'n':>4} {'hack':>6} {'flag':>6} {'recall':>7} {'FPR':>6} {'FPR_tf':>7} {'FPR_open':>8} " + " ".join(f"rec_{w:<8}" for w in WORDS))
for step in sorted({x["step"] for x in L}):
    S = [x for x in L if x["step"] == step]
    hk = [x for x in S if x[H] > 0]; cl = [x for x in S if x[H] == 0]
    rate = lambda xs: (sum(x[D] > 0 for x in xs) / len(xs)) if xs else float("nan")
    per_w = [rate([x for x in S if x.get(f"score/hack_freq_word_{w}/sycophancy_any", 0) > 0]) for w in WORDS]
    print(f"{step:>5} {len(S):>4} {len(hk)/len(S):>6.2f} {rate(S):>6.2f} {rate(hk):>7.2f} {rate(cl):>6.2f} "
          f"{rate([x for x in cl if x['question_type']=='tf']):>7.2f} {rate([x for x in cl if x['question_type']=='open']):>8.2f} "
          + " ".join(f"{v:<12.2f}" for v in per_w))
