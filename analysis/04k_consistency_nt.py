#!/usr/bin/env python3
"""
04k_consistency_nt.py — consistency-based vs verbalized calibration (tables/main/f3_consistency.csv)
Verbatim extract of section (c) of the review-time script 04f_final_robustness.py; only the
input/output paths were made repository-relative. Sections (a) and (b) of 04f are superseded by
04j_prompt_stability_nt.py and by the frozen complete CoT table, respectively.
"""
import json, collections
import numpy as np, pandas as pd
from scipy import stats
import warnings; warnings.filterwarnings("ignore")
import os; _R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U=f"{_R}/data/results"; W=f"{_R}/data/benchmarks"; OUT=f"{_R}/tables/main"; os.makedirs(OUT, exist_ok=True)
def load(p): return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
base=load(f"{U}/benchmark_results_v2_base.jsonl")
cs=load(f"{U}/benchmark_results_consist.jsonl")
bv2={b["id"]:b for b in load(f"{W}/benchmark_v2.jsonl")}
PRIM=["claude-opus-4-8","claude-haiku-4-5","gemini-2.5-flash-nt","qwen2.5:32b","qwen2.5:14b","qwen2.5:7b",
      "gemma3:4b","llama3.1:8b","mistral-small:latest","phi4:latest"]
P={"claude-opus-4-8":"Claude-Opus","claude-haiku-4-5":"Claude-Haiku","gemini-2.5-flash-nt":"Gemini-Flash(nt)",
   "qwen2.5:32b":"Qwen-32B","qwen2.5:14b":"Qwen-14B","qwen2.5:7b":"Qwen-7B","gemma3:4b":"Gemma3-4B",
   "llama3.1:8b":"Llama3.1-8B","mistral-small:latest":"Mistral-Small","phi4:latest":"Phi-4"}
def ece_of(rs):
    cc=[(r["confidence"],1 if r["outcome"]=="correct" else 0) for r in rs
        if r["decision"] in("EVET","HAYIR") and isinstance(r.get("confidence"),(int,float))]
    if len(cc)<10: return float("nan")
    c=np.array([x for x,_ in cc]); y=np.array([x for _,x in cc]); N=len(c); e=0.0; ed=np.linspace(0,1,11)
    for i in range(10):
        m=((c>=ed[i])&(c<ed[i+1])) if i<9 else ((c>=ed[i])&(c<=ed[i+1]))
        if m.sum(): e+=(m.sum()/N)*abs(y[m].mean()-c[m].mean())
    return e

# ---------- (c) CONSISTENCY vs VERBALIZED ----------
ids200=set(r["probe_id"] for r in cs)
rows=[]
for m in PRIM:
    by=collections.defaultdict(list)
    for r in cs:
        if r["model"]==m and r["outcome"]!="parse_fail": by[r["probe_id"]].append(r)
    ece_items=[]; flips=[]
    for pid,ss in by.items():
        decs=[r["decision"] for r in ss]
        maj,cnt=collections.Counter(decs).most_common(1)[0]
        cons=cnt/len(decs); flips.append(1-cons)
        if maj in ("EVET","HAYIR"):
            gold=bv2[pid]["gold"]; correct=(maj=="EVET")==(gold=="YES")
            ece_items.append((cons,1 if correct else 0))
    # ECE over consistency channel
    if ece_items:
        c=np.array([x for x,_ in ece_items]); y=np.array([x for _,x in ece_items]); N=len(c); e=0.0; ed=np.linspace(0,1,11)
        for i in range(10):
            msk=((c>=ed[i])&(c<ed[i+1])) if i<9 else ((c>=ed[i])&(c<=ed[i+1]))
            if msk.sum(): e+=(msk.sum()/N)*abs(y[msk].mean()-c[msk].mean())
    else: e=float("nan")
    verb=[r for r in base if r["model"]==m and r["probe_id"] in ids200 and r["outcome"]!="parse_fail"]
    ev=ece_of(verb)
    # iki kanal korelasyonu (item-bazlı): verbalized conf (ana koşu) vs consistency
    vconf={r["probe_id"]:r["confidence"] for r in verb if r["decision"] in("EVET","HAYIR") and isinstance(r.get("confidence"),(int,float))}
    pairs=[(vconf[pid], 1-f) for pid,f in zip(by.keys(),flips) if pid in vconf]
    rho=stats.spearmanr([a for a,_ in pairs],[b for _,b in pairs]).correlation if len(pairs)>10 else float("nan")
    rows.append({"model":P[m],"ECE_verbalized":round(ev,3),"ECE_consistency":round(e,3),
                 "ort_flip(1-cons)":round(float(np.mean(flips)),3),"kanal_korelasyonu_ρ":round(float(rho),3)})
sdf=pd.DataFrame(rows)
print("\n[c] CONSISTENCY (k=5) vs VERBALIZED kalibrasyon — 200 item")
print(sdf.to_string(index=False))
# kalibrasyonun sıralaması iki kanalda korunuyor mu
r1=sdf.set_index("model")["ECE_verbalized"].rank(); r2=sdf.set_index("model")["ECE_consistency"].rank()
rr=stats.spearmanr(r1,r2).correlation
print(f"   ECE-sıralaması kanallar-arası Spearman = {rr:.3f}")
sdf.to_csv(f"{OUT}/f3_consistency.csv",index=False)
