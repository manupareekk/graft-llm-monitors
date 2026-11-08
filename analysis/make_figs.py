import json, glob, os, statistics as st, subprocess, csv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=os.path.expanduser("~/graft-llmmon"); OUT=f"{R}/writeup"
BF=f"{R}/box_final/output"; RS=f"{R}/results"
runs={"dn":[f"{BF}/llmmon_A1/dn_s{s}" for s in (1,2,3)]}
for a in ("gr","rp2","filter"): runs[f"heur_{a}"]=[f"{BF}/llmmon_A1/heur_{a}_s{s}" for s in (1,2,3)]
for a in ("rp2","filter"): runs[f"q4t50_{a}"]=[f"{BF}/llmmon_A2/q4t50_{a}_s{s}" for s in (1,2,3)]
runs["q4t50_gr"]=[f"{RS}/llmmon_A2gr2/q4t50_gr_s1",f"{BF}/llmmon_A2gr/q4t50_gr_s2",f"{RS}/llmmon_A2gr2/q4t50_gr_s3"]
for a in ("gr","rp2","filter"): runs[f"q17t20_{a}"]=[f"{RS}/llmmon_B_s12/q17t20_{a}_s1",f"{RS}/llmmon_B_s12/q17t20_{a}_s2",f"{RS}/llmmon_B_s3/q17t20_{a}_s3"]
H="hack_freq_hackable/sycophancy_any"; A="retain/qa_correct"
def load(d): return [json.loads(l) for l in open(f"{d}/routing_eval.jsonl") if l.strip()]
def last(ev,k,K=3): v=[e[k] for e in ev[-K:] if k in e]; return sum(v)/len(v) if v else float("nan")
def recall(d):
    out=subprocess.run(["python3",f"{R}/judge_confusion.py",d],capture_output=True,text=True).stdout.splitlines()[1:]
    rows=[l.split() for l in out if l.strip()]
    v=[float(r[4]) for r in rows[-3:] if r[4]!="nan"]; return sum(v)/len(v) if v else float("nan")
rows=[]
for arm,ds in runs.items():
    for d in ds:
        ev=load(d); assert ev[-1]["step"]>=975,(d,ev[-1]["step"])
        r={"arm":arm,"seed":d[-1],"hack_both":last(ev,f"both/{H}"),"hack_retain_only":last(ev,f"retain_only/{H}"),
           "acc_both":last(ev,f"both/{A}"),"acc_retain_only":last(ev,f"retain_only/{A}"),
           "recall":recall(d) if arm!="dn" else float("nan")}
        for w in ("indeed","actually","exactly","quite"):
            r[f"ro_{w}"]=last(ev,f"retain_only/hack_freq_word_{w}/sycophancy_any"); r[f"both_{w}"]=last(ev,f"both/hack_freq_word_{w}/sycophancy_any")
        rows.append(r)
with open(f"{OUT}/per_seed.csv","w") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); [w.writerow({k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()}) for r in rows]
def agg(arm,k):
    v=[r[k] for r in rows if r["arm"]==arm and r[k]==r[k]]
    if not v: return float("nan"),0,[]
    return st.mean(v),(st.stdev(v)/len(v)**.5 if len(v)>1 else 0),v
MON=[("heur","Hand-written rule"),("q4t50","Qwen3-4B judge"),("q17t20","Qwen3-1.7B judge")]
METH=[("gr","GRAFT (RETAIN only)","hack_retain_only","acc_retain_only"),("rp2","Reward penalty","hack_both","acc_both"),("filter","Filtering","hack_both","acc_both")]
# Fig1: deployed hack rate
fig,axs=plt.subplots(1,2,figsize=(11,4))
for j,(key,lab) in enumerate([("hack","Reward-hacking rate (hackable Qs)"),("acc","Task accuracy")]):
    ax=axs[j]; wdt=.25
    for i,(m,ml,hk,ak) in enumerate(METH):
        k=hk if key=="hack" else ak
        ms=[agg(f"{mon}_{m}",k) for mon,_ in MON]
        xs=[x+(i-1)*wdt for x in range(3)]
        ax.bar(xs,[a[0] for a in ms],wdt,yerr=[a[1] for a in ms],label=ml,capsize=3)
        for x,a in zip(xs,ms): ax.scatter([x]*len(a[2]),a[2],s=8,c="k",zorder=3)
    dn=agg("dn","hack_both" if key=="hack" else "acc_both")[0]; ax.axhline(dn,ls="--",c="gray",label="No intervention")
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _,l in MON]); ax.set_title(lab); ax.set_ylim(0,1)
axs[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/fig1_deployed.png",dpi=150); plt.close()
# Fig2: GRAFT curves both vs retain_only
fig,axs=plt.subplots(1,3,figsize=(13,3.6),sharey=True)
for ax,(mon,ml) in zip(axs,MON):
    for d in runs[f"{mon}_gr"]:
        ev=load(d); s=[e["step"] for e in ev]
        ax.plot(s,[e[f"both/{H}"] for e in ev],c="C3",alpha=.7); ax.plot(s,[e[f"retain_only/{H}"] for e in ev],c="C0",alpha=.7)
    ax.set_title(f"GRAFT, {ml}"); ax.set_xlabel("step")
axs[0].plot([],[],c="C3",label="both adapters"); axs[0].plot([],[],c="C0",label="RETAIN only (FORGET ablated)"); axs[0].legend(fontsize=8); axs[0].set_ylabel("hack rate (hackable)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_graft_curves.png",dpi=150); plt.close()
# Fig3: recall vs residual hacking
fig,ax=plt.subplots(figsize=(5,4))
for mon,ml in MON[1:2]+MON[2:]:
    rr=[r for r in rows if r["arm"]==f"{mon}_gr"]; ax.scatter([r["recall"] for r in rr],[r["hack_retain_only"] for r in rr],label=ml,s=40)
hg=[r for r in rows if r["arm"]=="heur_gr"]; ax.scatter([r["recall"] for r in hg],[r["hack_retain_only"] for r in hg],marker="x",label="Hand-written rule")
ax.set_xlabel("monitor recall on eval-set hacks (last 3 evals)"); ax.set_ylabel("RETAIN-only hack rate"); ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OUT}/fig3_recall.png",dpi=150); plt.close()
# summary table
lines=["| Monitor | Method | Hack rate (deployed) | Task acc (deployed) | Hack rate, both adapters | Judge recall |","|---|---|---|---|---|---|"]
f=lambda a:f"{a[0]*100:.1f} ± {a[1]*100:.1f}"
lines.append(f"| — | No intervention | {f(agg('dn','hack_both'))} | {agg('dn','acc_both')[0]:.3f} | — | — |")
for mon,ml in MON:
    for m,lab,hk,ak in METH:
        a=f"{mon}_{m}"; rc=agg(a,"recall")[0]
        lines.append(f"| {ml} | {lab} | {f(agg(a,hk))} | {agg(a,ak)[0]:.3f} ± {agg(a,ak)[1]:.3f} | {f(agg(a,'hack_both'))} | {'—' if rc!=rc else f'{rc*100:.1f}%'} |")
open(f"{OUT}/table.md","w").write("\n".join(lines)+"\n"); print("\n".join(lines))
for r in rows:
    if r["arm"].endswith("_gr"): print(r["arm"],r["seed"],"ro",round(r["hack_retain_only"],3),"indeed",round(r["ro_indeed"],3),"actually",round(r["ro_actually"],3),"rec",round(r["recall"],3) if r["recall"]==r["recall"] else "-")
