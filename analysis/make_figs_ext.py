import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=os.path.expanduser("~/graft-llmmon"); RS=f"{R}/results"; BF=f"{R}/box_final/output"; OUT=f"{R}/writeup"
def load(d): return [json.loads(l) for l in open(f"{d}/routing_eval.jsonl") if l.strip()]
H="hack_freq_hackable/sycophancy_any"
fig,axs=plt.subplots(1,3,figsize=(13,3.6),sharey=True)
groups=[("Qwen3-4B judge (recall ~50%)",[f"{RS}/llmmon_A2gr2/q4t50_gr_s1",f"{BF}/llmmon_A2gr/q4t50_gr_s2",f"{RS}/llmmon_A2gr2/q4t50_gr_s3"]),
        ("4B judge, 90% of flags dropped (recall ~5%)",[f"{RS}/llmmon_X_kf/q4keep10_gr_s1",f"{RS}/llmmon_X_kf/q4keep10_gr_s2"]),
        ("4B judge + 30% extra false flags (FPR ~45%)",[f"{RS}/llmmon_X_kf/q4fp30_gr_s1",f"{RS}/llmmon_X_kf/q4fp30_gr_s2"])]
for ax,(t,ds) in zip(axs,groups):
    for d in ds:
        ev=load(d); s=[e["step"] for e in ev]
        ax.plot(s,[e[f"both/{H}"] for e in ev],c="C3",alpha=.7); ax.plot(s,[e[f"retain_only/{H}"] for e in ev],c="C0",alpha=.8)
    ax.set_title(t,fontsize=10); ax.set_xlabel("step"); ax.set_xlim(0,1000)
axs[0].plot([],[],c="C3",label="both adapters"); axs[0].plot([],[],c="C0",label="RETAIN only (deployed)"); axs[0].legend(fontsize=8); axs[0].set_ylabel("hack rate (hackable Qs)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_degraded.png",dpi=150); plt.close()
fig,axs=plt.subplots(1,2,figsize=(10,3.6),sharey=True)
for ax,(t,d) in zip(axs,[("No intervention",f"{RS}/llmmon_S_s1/sortq4_dn_s1"),("Reward penalty on judge flags",f"{RS}/llmmon_S_s1/sortq4_rp2_s1")]):
    ev=load(d); s=[e["step"] for e in ev]
    for k,lab,c in [("both/hack_freq_hackable/sorting_copy_threshold","any copy (hack)","k"),("both/hack_freq_detectable/sorting_copy_verbatim","verbatim copy","C3"),
                    ("both/hack_freq_undetectable/sorting_copy_reformatted","reformatted copy","C1"),("both/detected_freq/llm_judge_logprob","judge flag rate","C0")]:
        ax.plot(s,[e[k] for e in ev],c=c,label=lab,ls="--" if "judge" in lab else "-")
    ax.set_title(f"Sorting, 4B judge: {t}",fontsize=10); ax.set_xlabel("step")
axs[0].set_ylabel("rate on hackable prompts"); axs[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_sort.png",dpi=150); plt.close(); print("ok")
