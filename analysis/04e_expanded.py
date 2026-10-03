#!/usr/bin/env python3
"""
04e_expanded.py — Genişletilmiş set analizi (10 model / 6 sağlayıcı)
====================================================================
GENELLEME KAPISI: örüntü (kalibrasyon-sıralaması ≠ accuracy-sıralaması;
abstention/overconfidence ayrışması) geniş sette duruyor mu?
+ THINKING DUYARLILIĞI: flash(1024) vs flash-nt(0) — asimetri önemli miydi?
PRIMARY Gemini = gemini-2.5-flash-nt (matched). gemma4:31b kısmi -> dışlandı.
"""
import json, math, collections, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
import os; _R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U=f"{_R}/data/results"; W=f"{_R}/data/benchmarks"; OUT=f"{_R}/tables/regen_nt"; os.makedirs(OUT, exist_ok=True)

PRIMARY=["claude-opus-4-8","claude-haiku-4-5","gemini-2.5-flash-nt",
         "qwen2.5:32b","qwen2.5:14b","qwen2.5:7b","gemma3:4b",
         "llama3.1:8b","mistral-small:latest","phi4:latest"]
SENS="gemini-2.5-flash"      # thinking=1024 sensitivity-arm
PRO="gemini-2.5-pro"         # condB-only
P={"claude-opus-4-8":"Claude-Opus","claude-haiku-4-5":"Claude-Haiku","gemini-2.5-flash-nt":"Gemini-Flash(nt)",
   "gemini-2.5-flash":"Gemini-Flash(1024)","gemini-2.5-pro":"Gemini-Pro","qwen2.5:32b":"Qwen-32B",
   "qwen2.5:14b":"Qwen-14B","qwen2.5:7b":"Qwen-7B","gemma3:4b":"Gemma3-4B","llama3.1:8b":"Llama3.1-8B",
   "mistral-small:latest":"Mistral-Small","phi4:latest":"Phi-4"}
FAM={"claude-opus-4-8":"Anthropic","claude-haiku-4-5":"Anthropic","gemini-2.5-flash-nt":"Google",
     "gemini-2.5-flash":"Google","gemini-2.5-pro":"Google","qwen2.5:32b":"Alibaba","qwen2.5:14b":"Alibaba",
     "qwen2.5:7b":"Alibaba","gemma3:4b":"Google-open","llama3.1:8b":"Meta","mistral-small:latest":"Mistral","phi4:latest":"Microsoft"}

def load(p): return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
base=load(f"{U}/benchmark_results_v2_base.jsonl"); condB=load(f"{U}/benchmark_results_condB.jsonl"); gal=load(f"{U}/benchmark_results_gal.jsonl")
bv2={b["id"]:b for b in load(f"{W}/benchmark_v2.jsonl")}
EX=set(json.load(open(f"{W}/attrib_real_exclude.json"))["exclude_ids"])
def valid(r): return r["outcome"]!="parse_fail" and r["probe_id"] not in EX
def zc(p): return float(stats.norm.ppf(min(max(p,1e-6),1-1e-6)))
def ece(rs):
    cc=[(r["confidence"],1 if r["outcome"]=="correct" else 0) for r in rs
        if r["decision"] in("EVET","HAYIR") and isinstance(r.get("confidence"),(int,float))]
    if not cc: return float("nan")
    c=np.array([x for x,_ in cc]); y=np.array([x for _,x in cc]); N=len(c); e=0.0; ed=np.linspace(0,1,11)
    for i in range(10):
        m=((c>=ed[i])&(c<ed[i+1])) if i<9 else ((c>=ed[i])&(c<=ed[i+1]))
        if m.sum(): e+=(m.sum()/N)*abs(y[m].mean()-c[m].mean())
    return e
def core(records,m):
    rs=[r for r in records if r["model"]==m and valid(r)]
    n=len(rs); cor=sum(r["outcome"]=="correct" for r in rs); wr=sum(r["outcome"]=="wrong" for r in rs)
    ab=sum(r["outcome"]=="abstain" for r in rs); comm=cor+wr
    return dict(n=n,raw=cor/n,cacc=cor/comm if comm else np.nan,abst=ab/n,err=wr/n,ece=ece(rs))
def dprime(records,m,committed=True):
    real=[r for r in records if r["model"]==m and r["type"]=="ATTRIB_REAL" and valid(r)]
    fake=[r for r in records if r["model"]==m and r["type"]=="ATTRIB_FAKE" and valid(r)]
    if committed:
        h=sum(r["decision"]=="EVET" for r in real); rm=sum(r["decision"]=="HAYIR" for r in real)
        f=sum(r["decision"]=="EVET" for r in fake); cr=sum(r["decision"]=="HAYIR" for r in fake)
        return zc((h+.5)/(h+rm+1))-zc((f+.5)/(f+cr+1))
    h=sum(r["decision"]=="EVET" for r in real); f=sum(r["decision"]=="EVET" for r in fake)
    return zc((h+.5)/(len(real)+1))-zc((f+.5)/(len(fake)+1))
def ana150(records,m):
    ids=set(b["id"] for b in bv2.values() if b["type"]=="ANACHRONISM" and b.get("gap",0)>=150)
    rs=[r for r in records if r["model"]==m and r["probe_id"] in ids and r["outcome"]!="parse_fail"]
    return sum(r["decision"]=="EVET" for r in rs)/len(rs) if rs else np.nan

results={}
# ===== 1) GENİŞLETİLMİŞ ÇEKİRDEK =====
rows=[]
for m in PRIMARY+[SENS]:
    c=core(base,m); d=dprime(base,m); a=ana150(base,m)
    rows.append({"model":P[m],"family":FAM[m],"raw":round(c["raw"],3),"committed":round(c["cacc"],3),
        "abstain":round(c["abst"],3),"error":round(c["err"],3),"ECE":round(c["ece"],3),
        "d′(comm)":round(d,2),"anach≥150_appr":f"{100*a:.1f}%"})
df=pd.DataFrame(rows)
print("[1] GENİŞLETİLMİŞ ÇEKİRDEK — 10 primary + thinking-sensitivity satırı")
print(df.to_string(index=False))
df.to_csv(f"{OUT}/e1_expanded_core.csv",index=False); results["core"]=df.to_dict("records")

# ===== 2) THINKING DUYARLILIĞI (flash-nt vs flash-1024) =====
nt=core(base,"gemini-2.5-flash-nt"); th=core(base,SENS)
ts=pd.DataFrame([{"metric":k,"flash_nt(0)":round(nt[kk],3),"flash(1024)":round(th[kk],3),
  "Δ(1024−nt)":round(th[kk]-nt[kk],3)} for k,kk in
  [("raw_acc","raw"),("committed_acc","cacc"),("abstention","abst"),("error","err"),("ECE","ece")]])
print("\n[2] THINKING DUYARLILIĞI — Gemini Flash thinking=0 vs 1024")
print(ts.to_string(index=False))
dnt=dprime(base,"gemini-2.5-flash-nt"); dth=dprime(base,SENS)
ant=ana150(base,"gemini-2.5-flash-nt"); ath=ana150(base,SENS)
print(f"   d′: nt={dnt:.2f} vs 1024={dth:.2f} | anach≥150: nt={100*ant:.1f}% vs 1024={100*ath:.1f}%")
ts.to_csv(f"{OUT}/e2_thinking_sensitivity.csv",index=False); results["thinking"]=ts.to_dict("records")

# ===== 3) GENELLEME KARARI =====
prim=[m for m in PRIMARY]
raw_rank=sorted(prim,key=lambda m:-core(base,m)["raw"])
cacc_rank=sorted(prim,key=lambda m:-core(base,m)["cacc"])
ece_rank=sorted(prim,key=lambda m: core(base,m)["ece"])
def spear(r1,r2): return stats.spearmanr([r1.index(m) for m in prim],[r2.index(m) for m in prim]).correlation
print("\n[3] GENELLEME KARARI (10 model / 6 sağlayıcı)")
print(f"   raw-acc sıralaması:      {[P[m] for m in raw_rank]}")
print(f"   committed-acc sıralaması:{[P[m] for m in cacc_rank]}")
print(f"   ECE (iyi->kötü):         {[P[m] for m in ece_rank]}")
print(f"   Spearman(raw, committed) = {spear(raw_rank,cacc_rank):.3f}   (düşük = accuracy-only YANLIŞ sıralıyor)")
print(f"   Spearman(raw, ECE-iyi)   = {spear(raw_rank,ece_rank):.3f}")
# dissociation: abstention vs error scatter verisi
diss=pd.DataFrame([{"model":P[m],"family":FAM[m],"abstain":core(base,m)["abst"],
                    "error":core(base,m)["err"],"committed":core(base,m)["cacc"],"ECE":core(base,m)["ece"]} for m in prim])
diss.to_csv(f"{OUT}/e3_dissociation.csv",index=False)
results["generality"]={"spearman_raw_committed":round(float(spear(raw_rank,cacc_rank)),3),
                       "raw_rank":[P[m] for m in raw_rank],"cacc_rank":[P[m] for m in cacc_rank]}

# ===== 4) FORCED d′ (condB) — genişletilmiş =====
def dprime_forced(m):
    real=[r for r in condB if r["model"]==m and r["type"]=="ATTRIB_REAL" and r["outcome"]!="parse_fail" and r["probe_id"] not in EX]
    fake=[r for r in condB if r["model"]==m and r["type"]=="ATTRIB_FAKE" and r["outcome"]!="parse_fail" and r["probe_id"] not in EX]
    h=sum(r["decision"]=="EVET" for r in real); f=sum(r["decision"]=="EVET" for r in fake)
    return zc((h+.5)/(len(real)+1))-zc((f+.5)/(len(fake)+1))
fr=[{"model":P[m],"d′_forced(B)":round(dprime_forced(m),3)} for m in PRIMARY+[PRO]]
frd=pd.DataFrame(fr); print("\n[4] FORCED-CHOICE d′ (Condition B, seçimsiz) — genişletilmiş + Gemini-Pro")
print(frd.to_string(index=False))
frd.to_csv(f"{OUT}/e4_forced_dprime_expanded.csv",index=False); results["forced_dprime"]=frd.to_dict("records")

# ===== 5) GAL replikasyon + cross-lingual (genişletilmiş) =====
def gal_core(m,lang):
    rs=[r for r in gal if r["model"]==m and r["lang"]==lang and r["outcome"]!="parse_fail"]
    cor=sum(r["outcome"]=="correct" for r in rs); wr=sum(r["outcome"]=="wrong" for r in rs)
    ab=sum(r["outcome"]=="abstain" for r in rs); comm=cor+wr
    fake=[r for r in rs if r["type"]=="ATTRIB_FAKE"]
    fa=sum(r["decision"]=="EVET" for r in fake); cr=sum(r["decision"]=="HAYIR" for r in fake)
    return dict(cacc=cor/comm if comm else np.nan, abst=ab/len(rs) if rs else np.nan,
                fakeFA=fa/(fa+cr) if (fa+cr) else np.nan)
gr=[]
for m in PRIMARY:
    t=gal_core(m,"tr"); a=gal_core(m,"ar")
    gr.append({"model":P[m],"GAL_tr_cacc":round(t["cacc"],3),"GAL_ar_cacc":round(a["cacc"],3),
        "Δcacc(ar−tr)":round(a["cacc"]-t["cacc"],3),
        "tr_fakeFA":round(t["fakeFA"],3),"ar_fakeFA":round(a["fakeFA"],3)})
grd=pd.DataFrame(gr); print("\n[5] GAL TEMİZ-GOLD + CROSS-LINGUAL (genişletilmiş)")
print(grd.to_string(index=False))
grd.to_csv(f"{OUT}/e5_gal_expanded.csv",index=False); results["gal"]=grd.to_dict("records")

# ===== 6) COST-MODEL crossover (Claude-Opus vs herkes) =====
cr=[]
cE,cA=core(base,"claude-opus-4-8")["err"],core(base,"claude-opus-4-8")["abst"]
for m in PRIMARY:
    if m=="claude-opus-4-8": continue
    e,a=core(base,m)["err"],core(base,m)["abst"]
    den=a-cA; lam=(cE-e)/den if den!=0 else float("nan")
    if cA>a and 0<=lam<=1: reg=f"Opus iyi: λ<{lam:.2f}"
    elif cA<a and 0<=lam<=1: reg=f"Opus iyi: λ>{lam:.2f}"
    elif cE<=e and cE+cA<=e+a: reg="Opus her λ'da iyi"
    else: reg="karışık"
    cr.append({"vs":P[m],"crossover_λ":round(lam,3) if not math.isnan(lam) else "-","bölge":reg})
crd=pd.DataFrame(cr); print("\n[6] COST-MODEL (Opus vs herkes; λ=çekimser/yanlış maliyet oranı)")
print(crd.to_string(index=False))
crd.to_csv(f"{OUT}/e6_cost_expanded.csv",index=False); results["cost"]=crd.to_dict("records")

json.dump(results,open(f"{OUT}/04e_expanded.json","w",encoding="utf-8"),ensure_ascii=False,indent=2,default=str)
print("\nTamam -> e1..e6 + 04e_expanded.json")

