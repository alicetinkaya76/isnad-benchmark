#!/usr/bin/env python3
# 04h = 04d re-keyed to the 10 primary configurations (Gemini = gemini-2.5-flash-nt); label-noise levels extended to 35%. R2 revision, Aug 2026.
"""
04d_robustness.py — Hakem P0/P1'lerini mevcut veriyle kapatan reanaliz
======================================================================
1 item-flow  2 bootstrap CI (cluster=scholar) + paired diff CI
3 maliyet/fayda modeli (MCC-flip YERİNE)  4 Condition-B forced-choice d'
5 outlier-exclusion  6 MISATTRIB era-gap  7 ANACHRONISM>=150y
8 error-overlap (Fleiss kappa + "Claude, Gemini'nin yanıldığı yerde mi çekimser")
9 GEE (cluster-robust logistic)  10 label-noise duyarlılık
11 H1 çoklu-karşılaştırma düzeltmesi  12 fake-title artefakt istatistiği
"""
import json, math, collections, warnings, random
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import statsmodels.api as sm
warnings.filterwarnings("ignore")
rng=np.random.default_rng(42)

import os; _R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U=f"{_R}/data/results"; W=f"{_R}/data/benchmarks"; OUT=f"{_R}/tables/regen_nt"; os.makedirs(OUT, exist_ok=True)
MODELS=["claude-opus-4-8","claude-haiku-4-5","gemini-2.5-flash-nt","qwen2.5:32b","qwen2.5:14b","qwen2.5:7b","gemma3:4b","llama3.1:8b","mistral-small:latest","phi4:latest"]
PRETTY={"claude-opus-4-8":"Claude","claude-haiku-4-5":"Haiku","gemini-2.5-flash-nt":"Gemini-Flash(nt)","qwen2.5:32b":"Qwen-32B","qwen2.5:14b":"Qwen-14B","qwen2.5:7b":"Qwen-7B","gemma3:4b":"Gemma-4B","llama3.1:8b":"Llama-8B","mistral-small:latest":"Mistral","phi4:latest":"Phi-4"}
OUTLIERS={"qwen2.5:7b","gemma3:4b"}
def load(p): return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
base=load(f"{U}/benchmark_results_v2_base.jsonl"); condB=load(f"{U}/benchmark_results_condB.jsonl")
bv2={b["id"]:b for b in load(f"{W}/benchmark_v2.jsonl")}
EX=set(json.load(open(f"{W}/attrib_real_exclude.json"))["exclude_ids"])
# DİA death years for MISATTRIB era-gap
import csv,re
death={}
for row in csv.DictReader(open(f"{_R}/data/dia_scholars/scholars_full.csv",encoding="utf-8-sig")):
    dd=re.findall(r"\d+",row.get("death_hijri","") or "")
    death[row["url_slug"].strip()]=int(dd[0]) if dd and 1<=int(dd[0])<=1500 else None
def valid(r): return r["outcome"]!="parse_fail" and r["probe_id"] not in EX
def cluster_of(b):  # scholar/entity cluster for bootstrap
    t=b["type"]
    if t in ("ATTRIB_REAL","ATTRIB_FAKE"): return b.get("scholar_slug")
    if t=="MISATTRIB": return b.get("true_author_slug")
    return b.get("teacher_slug")
def zc(p): return float(stats.norm.ppf(min(max(p,1e-6),1-1e-6)))
results={}

# ---------- 1) ITEM-FLOW ----------
flow=[]
tot=len(bv2); types=collections.Counter(b["type"] for b in bv2.values())
flow.append({"stage":"mined+frozen (benchmark_v2)","n":tot})
flow.append({"stage":"removed: modern/artifact ATTRIB_REAL","n":-len(EX)})
flow.append({"stage":"ANALYZED","n":tot-len(EX)})
g=collections.Counter(b["gold"] for b in bv2.values() if b["id"] not in EX)
flow.append({"stage":"  gold YES","n":g["YES"]}); flow.append({"stage":"  gold NO","n":g["NO"]})
fl=pd.DataFrame(flow); print("[1] ITEM-FLOW\n",fl.to_string(index=False))
print("   per-type:",dict(types))
fl.to_csv(f"{OUT}/r1_item_flow.csv",index=False)
results["item_flow"]={"total":tot,"excluded":len(EX),"analyzed":tot-len(EX),"yes":g["YES"],"no":g["NO"],"per_type":dict(types)}

# ---------- helper: per-model committed/abstain/error on a record set ----------
def metrics_for(records, model, item_filter=None):
    rs=[r for r in records if r["model"]==model and valid(r) and (item_filter(r) if item_filter else True)]
    n=len(rs); cor=sum(r["outcome"]=="correct" for r in rs); wr=sum(r["outcome"]=="wrong" for r in rs)
    ab=sum(r["outcome"]=="abstain" for r in rs); comm=cor+wr
    return dict(n=n,correct=cor,wrong=wr,abstain=ab,
                committed_acc=cor/comm if comm else float("nan"),
                abstention=ab/n if n else float("nan"), error=wr/n if n else float("nan"),
                raw=cor/n if n else float("nan"))

# ---------- 2) BOOTSTRAP CI (cluster=scholar) on core metrics + paired diffs ----------
# pre-index by model: list of (cluster, outcome, gold, decision, conf, type)
by_model={m:[(cluster_of(bv2[r["probe_id"]]), r["outcome"], r["gold"], r["decision"],
              r.get("confidence"), r["type"]) for r in base if r["model"]==m and valid(r)] for m in MODELS}
clusters=sorted({c for m in MODELS for (c,*_ ) in by_model[m] if c is not None})
cl_index={c:i for i,c in enumerate(clusters)}
def boot_metric(rows, metric, B=600):
    # rows: list of tuples; resample clusters
    bycl=collections.defaultdict(list)
    for row in rows:
        bycl[row[0]].append(row)
    keys=list(bycl.keys()); vals=[]
    for _ in range(B):
        samp=rng.choice(len(keys),len(keys),replace=True)
        pool=[]
        for k in samp: pool.extend(bycl[keys[k]])
        vals.append(metric(pool))
    return np.nanpercentile(vals,2.5), np.nanpercentile(vals,97.5)
def m_committed(pool):
    cor=sum(o=="correct" for _,o,*_ in pool); wr=sum(o=="wrong" for _,o,*_ in pool)
    return cor/(cor+wr) if (cor+wr) else np.nan
def m_error(pool): 
    return sum(o=="wrong" for _,o,*_ in pool)/len(pool) if pool else np.nan
def m_abst(pool):
    return sum(o=="abstain" for _,o,*_ in pool)/len(pool) if pool else np.nan
def m_ece(pool):
    cc=[(c,1 if o=="correct" else 0) for (_,o,g,d,c,t) in pool if d in("EVET","HAYIR") and isinstance(c,(int,float))]
    if not cc: return np.nan
    c=np.array([x for x,_ in cc]); y=np.array([x for _,x in cc]); N=len(c); e=0.0; ed=np.linspace(0,1,11)
    for i in range(10):
        msk=((c>=ed[i])&(c<ed[i+1])) if i<9 else ((c>=ed[i])&(c<=ed[i+1]))
        if msk.sum(): e+=(msk.sum()/N)*abs(y[msk].mean()-c[msk].mean())
    return e
ci_rows=[]
for m in MODELS:
    rows=by_model[m]
    cl,ch=boot_metric(rows,m_committed); el,eh=boot_metric(rows,m_error)
    al,ah=boot_metric(rows,m_abst); ecl,ech=boot_metric(rows,m_ece)
    mm=metrics_for(base,m)
    ci_rows.append({"model":PRETTY[m],
        "committed_acc":f"{mm['committed_acc']:.3f} [{cl:.3f},{ch:.3f}]",
        "error":f"{mm['error']:.3f} [{el:.3f},{eh:.3f}]",
        "abstention":f"{mm['abstention']:.3f} [{al:.3f},{ah:.3f}]",
        "ECE_ci":f"[{ecl:.3f},{ech:.3f}]"})
ci=pd.DataFrame(ci_rows); print("\n[2] BOOTSTRAP %95 CI (cluster=scholar, B=600)\n",ci.to_string(index=False))
ci.to_csv(f"{OUT}/r2_bootstrap_ci.csv",index=False); results["bootstrap_ci"]=ci.to_dict("records")
# paired diff CI: Claude - X on committed_acc & ECE (paired over shared clusters)
def paired_diff_ci(mA,mB,metric,B=600):
    A=by_model[mA]; Bm=by_model[mB]
    clA=collections.defaultdict(list); clB=collections.defaultdict(list)
    for row in A: clA[row[0]].append(row)
    for row in Bm: clB[row[0]].append(row)
    keys=sorted(set(clA)&set(clB)); diffs=[]
    for _ in range(B):
        samp=rng.choice(len(keys),len(keys),replace=True)
        pa=[];pb=[]
        for k in samp: pa.extend(clA[keys[k]]); pb.extend(clB[keys[k]])
        diffs.append(metric(pa)-metric(pb))
    return np.nanpercentile(diffs,2.5),np.nanpercentile(diffs,97.5),float(np.nanmean(diffs))
pd_rows=[]
def point_diff(mA,mB,metric): return metric(by_model[mA])-metric(by_model[mB])
for mB in ["gemini-2.5-flash-nt","qwen2.5:32b"]:
    lo,hi,mean=paired_diff_ci("claude-opus-4-8",mB,m_committed)
    lo2,hi2,mean2=paired_diff_ci("claude-opus-4-8",mB,m_ece)
    pt=point_diff("claude-opus-4-8",mB,m_committed); pt2=point_diff("claude-opus-4-8",mB,m_ece)
    # central value = whole-sample point difference; interval = paired cluster-bootstrap percentile CI
    pd_rows.append({"contrast":f"Claude − {PRETTY[mB]}",
        "Δcommitted_acc":f"{pt:+.3f} [{lo:+.3f},{hi:+.3f}]","sig_acc":"yes" if lo>0 or hi<0 else "no",
        "ΔECE":f"{pt2:+.3f} [{lo2:+.3f},{hi2:+.3f}]","sig_ECE":"yes" if lo2>0 or hi2<0 else "no"})
pdf=pd.DataFrame(pd_rows); print("\n   paired-diff %95 CI (Claude vs):\n",pdf.to_string(index=False))
pdf.to_csv(f"{OUT}/r2b_paired_diffs.csv",index=False); results["paired_diffs"]=pdf.to_dict("records")

# ---------- 3) COST/UTILITY MODEL (MCC-flip yerine) ----------
# expected cost = error_rate + lambda * abstention_rate (correct=0, wrong=1, abstain=lambda)
core={m:metrics_for(base,m) for m in MODELS}
lams=np.linspace(0,1,101)
costcurve={m:[core[m]["error"]+lam*core[m]["abstention"] for lam in lams] for m in MODELS}
# crossover lambda where Claude cost == X cost
cross=[]
cE,cA=core["claude-opus-4-8"]["error"],core["claude-opus-4-8"]["abstention"]
for m in MODELS:
    if m=="claude-opus-4-8": continue
    eX,aX=core[m]["error"],core[m]["abstention"]
    denom=(aX-cA)
    lam_star=(cE-eX)/denom if denom!=0 else float("nan")
    # Claude lower-cost when lambda < lam_star (if aX<cA) ... determine region
    region="Claude daha iyi: λ<%.3f"%lam_star if (cA>aX and 0<=lam_star<=1) else \
           ("Claude daha iyi: λ>%.3f"%lam_star if (cA<aX and 0<=lam_star<=1) else
            ("Claude her λ'da daha iyi" if cE+0*1<=eX and cE+1*cA<=eX+1*aX else "değişken"))
    cross.append({"vs":PRETTY[m],"crossover_λ":round(lam_star,3) if not math.isnan(lam_star) else "-","region":region})
cr=pd.DataFrame(cross); print("\n[3] MALİYET MODELİ (cost=error+λ·abstain). Crossover λ (çekimser maliyeti / yanlış maliyeti)\n",cr.to_string(index=False))
print("   -> Claude vs Gemini crossover λ ≈ %.3f: çekimser, yanlışın %%%.0f'inden ucuzsa Claude tercih edilir."%
      ((cE-core['gemini-2.5-flash-nt']['error'])/(core['gemini-2.5-flash-nt']['abstention']-cA),
       100*(cE-core['gemini-2.5-flash-nt']['error'])/(core['gemini-2.5-flash-nt']['abstention']-cA)))
cr.to_csv(f"{OUT}/r3_cost_model.csv",index=False)
pd.DataFrame({"lambda":lams,**{PRETTY[m]:costcurve[m] for m in MODELS}}).to_csv(f"{OUT}/r3_cost_curves.csv",index=False)
results["cost_model"]=cr.to_dict("records")

# ---------- 4) CONDITION-B FORCED-CHOICE d' (selection-free) ----------
def dprime_forced(records,model):
    real=[r for r in records if r["model"]==model and r["type"]=="ATTRIB_REAL" and r["outcome"]!="parse_fail" and r["probe_id"] not in EX]
    fake=[r for r in records if r["model"]==model and r["type"]=="ATTRIB_FAKE" and r["outcome"]!="parse_fail" and r["probe_id"] not in EX]
    # forced: map abstain->? in B abstain forbidden so ~none; treat EVET=yes
    hit=sum(r["decision"]=="EVET" for r in real); nsig=len(real)
    fa=sum(r["decision"]=="EVET" for r in fake); nno=len(fake)
    hr=(hit+0.5)/(nsig+1); far=(fa+0.5)/(nno+1)
    return zc(hr)-zc(far), nsig, nno
fc=[]
for m in MODELS:
    dA_committed=None
    # condition A committed d'
    realA=[r for r in base if r["model"]==m and r["type"]=="ATTRIB_REAL" and valid(r)]
    fakeA=[r for r in base if r["model"]==m and r["type"]=="ATTRIB_FAKE" and valid(r)]
    hA=sum(r["decision"]=="EVET" for r in realA); rmA=sum(r["decision"]=="HAYIR" for r in realA)
    faA=sum(r["decision"]=="EVET" for r in fakeA); crA=sum(r["decision"]=="HAYIR" for r in fakeA)
    dA=zc((hA+.5)/(hA+rmA+1))-zc((faA+.5)/(faA+crA+1))
    dB,nsig,nno=dprime_forced(condB,m)
    fc.append({"model":PRETTY[m],"d′_A_committed":round(dA,3),"d′_B_forced":round(dB,3),
               "Δ(A−B)":round(dA-dB,3),"B_real_n":nsig,"B_fake_n":nno})
fcd=pd.DataFrame(fc); print("\n[4] FORCED-CHOICE d′ (Condition B = seçimsiz; A_committed seçim-yanlı)\n",fcd.to_string(index=False))
print("   -> A_committed d′, çekimserlikle şişiyor; B_forced d′ temiz kıyas. Sıralama B'de de korunuyor mu kontrol.")
fcd.to_csv(f"{OUT}/r4_forced_dprime.csv",index=False); results["forced_dprime"]=fcd.to_dict("records")

# ---------- 5) OUTLIER-EXCLUSION (Gemma + Qwen-7B çıkar) ----------
keep=[m for m in MODELS if m not in OUTLIERS]
oe=[]
for m in keep:
    mm=core[m]
    # SDT d'
    realA=[r for r in base if r["model"]==m and r["type"]=="ATTRIB_REAL" and valid(r)]
    fakeA=[r for r in base if r["model"]==m and r["type"]=="ATTRIB_FAKE" and valid(r)]
    hA=sum(r["decision"]=="EVET" for r in realA); rmA=sum(r["decision"]=="HAYIR" for r in realA)
    faA=sum(r["decision"]=="EVET" for r in fakeA); crA=sum(r["decision"]=="HAYIR" for r in fakeA)
    dA=zc((hA+.5)/(hA+rmA+1))-zc((faA+.5)/(faA+crA+1))
    oe.append({"model":PRETTY[m],"committed_acc":round(mm["committed_acc"],3),"error":round(mm["error"],3),
               "abstention":round(mm["abstention"],3),"d′":round(dA,3)})
oedf=pd.DataFrame(oe); print("\n[5] OUTLIER-EXCLUSION (Gemma-4B + Qwen-7B çıkarıldı) — örüntü korunuyor mu?\n",oedf.to_string(index=False))
print("   -> Claude (kalibre/çekimser) vs Gemini (iddialı/aşırı-güven) + Qwen ölçek-merdiveni; ana karşıtlık outlier'sız da duruyor.")
oedf.to_csv(f"{OUT}/r5_outlier_excluded.csv",index=False); results["outlier_excluded"]=oedf.to_dict("records")

# ---------- 6) MISATTRIB era-gap (main set) ----------
mis=[b for b in bv2.values() if b["type"]=="MISATTRIB"]
def eragap(b):
    d1=death.get(b["true_author_slug"]); d2=death.get(b["wrong_author_slug"])
    return abs(d1-d2) if (d1 and d2) else None
gaps=[(b["id"],eragap(b)) for b in mis]
known=[g for _,g in gaps if g is not None]
print("\n[6] MISATTRIB era-gap (main set; cross-era trivial mi?)")
print(f"   ölüm-yılı bilinen: {len(known)}/{len(mis)}; <=50y(same-era): {sum(1 for g in known if g<=50)}, 51-150: {sum(1 for g in known if 51<=g<=150)}, >150: {sum(1 for g in known if g>150)}")
print(f"   medyan era-gap: {int(np.median(known)) if known else '-'}  -> büyük kısmı cross-era; ana-set MISATTRIB caveat'lı, GAL same-era öne çıkarılacak.")
# committed-acc by era stratum per model
mis_strat=[]
for m in MODELS:
    row={"model":PRETTY[m]}
    for lab,lo,hi in [("same≤50",0,50),("near51-150",51,150),("far>150",151,99999)]:
        ids=set(b["id"] for b in mis if (eragap(b) is not None and lo<=eragap(b)<=hi))
        rs=[r for r in base if r["model"]==m and r["probe_id"] in ids and r["outcome"]!="parse_fail"]
        cor=sum(r["outcome"]=="correct" for r in rs); wr=sum(r["outcome"]=="wrong" for r in rs)
        row[lab]=round(cor/(cor+wr),3) if (cor+wr) else float("nan")
    mis_strat.append(row)
msd=pd.DataFrame(mis_strat); print("   committed-acc era-stratum:\n",msd.to_string(index=False))
msd.to_csv(f"{OUT}/r6_misattrib_eragap.csv",index=False)
results["misattrib_eragap"]={"known":len(known),"same":sum(1 for g in known if g<=50),"median":int(np.median(known)) if known else None}

# ---------- 7) ANACHRONISM >=150y ----------
ana_ids_150=set(b["id"] for b in bv2.values() if b["type"]=="ANACHRONISM" and b.get("gap",0)>=150)
an=[]
for m in MODELS:
    rs=[r for r in base if r["model"]==m and r["probe_id"] in ana_ids_150]
    appr=sum(r["decision"]=="EVET" for r in rs)
    an.append({"model":PRETTY[m],"n(≥150y)":len(rs),"false_approval":f"{100*appr/len(rs):.1f}%"})
andf=pd.DataFrame(an); print(f"\n[7] ANACHRONISM ≥150y altküme ({len(ana_ids_150)} item; imkânsızlık doğum-bağımsız garantili)\n",andf.to_string(index=False))
print("   -> Bulgu değişmiyor: iyi modeller ~%0, Gemma yüksek. Kriter: |Δdeath|≥150y, teacher önce ölmüş.")
andf.to_csv(f"{OUT}/r7_anachronism_150.csv",index=False); results["anachronism_150"]=andf.to_dict("records")

# ---------- 8) ERROR-OVERLAP + "Claude, Gemini'nin yanıldığı yerde mi çekimser" ----------
# build per-item committed-correctness across models (shared items)
items=[i for i in bv2 if i not in EX]
dec={m:{r["probe_id"]:r for r in base if r["model"]==m and valid(r)} for m in MODELS}
# pairwise error-overlap (Jaccard on wrong items, committed)
def wrongset(m): return set(pid for pid in items if dec[m].get(pid) and dec[m][pid]["outcome"]=="wrong")
ws={m:wrongset(m) for m in MODELS}
print("\n[8] ERROR-OVERLAP (committed yanlışlar; düşük örtüşme=politika-temelli, yüksek=ortak-zorluk)")
ov=[]
for i,a in enumerate(MODELS):
    for b in MODELS[i+1:]:
        inter=len(ws[a]&ws[b]); uni=len(ws[a]|ws[b])
        ov.append({"pair":f"{PRETTY[a]}–{PRETTY[b]}","jaccard":round(inter/uni,3) if uni else 0,"shared_wrong":inter})
ovd=pd.DataFrame(ov); print(ovd.to_string(index=False))
ovd.to_csv(f"{OUT}/r8_error_overlap.csv",index=False)
# Claude behavior on items Gemini got wrong
gem_wrong=ws["gemini-2.5-flash-nt"]
cl_on_gw=collections.Counter(dec["claude-opus-4-8"][pid]["outcome"] for pid in gem_wrong if dec["claude-opus-4-8"].get(pid))
print(f"\n   Gemini'nin yanıldığı {len(gem_wrong)} item'da Claude: {dict(cl_on_gw)}")
tot_gw=sum(cl_on_gw.values())
print(f"     -> Claude bu 'tuzak' item'larda %{100*cl_on_gw['abstain']/tot_gw:.0f} çekimser, %{100*cl_on_gw['wrong']/tot_gw:.0f} yanlış (akıllı çekimserlik testi).")
results["error_overlap"]={"pairs":ovd.to_dict("records"),"claude_on_gemini_wrong":dict(cl_on_gw)}

# ---------- 9) GEE (cluster-robust logistic): correct|committed ~ model*type ----------
rows=[]
for m in MODELS:
    for r in base:
        if r["model"]==m and valid(r) and r["decision"] in ("EVET","HAYIR"):
            rows.append({"correct":1 if r["outcome"]=="correct" else 0,"model":PRETTY[m],
                         "type":r["type"],"cluster":cluster_of(bv2[r["probe_id"]]) or "NA"})
dfg=pd.DataFrame(rows)
try:
    dfg["model"]=pd.Categorical(dfg["model"],categories=["Gemini-Flash(nt)"]+[PRETTY[m] for m in MODELS if m!="gemini-2.5-flash-nt"])
    mod=smf.gee("correct ~ C(model)+C(type)",groups="cluster",data=dfg,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Exchangeable()).fit()
    coefs=[]
    for nm in mod.params.index:
        coefs.append({"term":nm,"coef":round(float(mod.params[nm]),3),"OR":round(float(np.exp(mod.params[nm])),3),
                      "p":f"{float(mod.pvalues[nm]):.1e}"})
    geedf=pd.DataFrame(coefs); print("\n[9] GEE cluster-robust logistic: P(correct|committed) ~ model + type (ref=Gemini-Flash)\n",geedf.to_string(index=False))
    geedf.to_csv(f"{OUT}/r9_gee.csv",index=False); results["gee"]=geedf.to_dict("records")
except Exception as e:
    print("\n[9] GEE hata:",e)

# ---------- 10) LABEL-NOISE SENSITIVITY (YES-gold flip) ----------
# simulate flipping x% of ATTRIB_REAL gold YES->NO (the ~15% misattribution residual), recompute committed-acc ranking
print("\n[10] LABEL-NOISE DUYARLILIK (ATTRIB_REAL YES-gold'unun x%'i hatalı varsay; committed-acc sıralaması kararlı mı)")
real_ids=[b["id"] for b in bv2.values() if b["type"]=="ATTRIB_REAL" and b["id"] not in EX]
base_rank=[PRETTY[m] for m in sorted(MODELS,key=lambda m:-core[m]["committed_acc"])]
ns=[]
for noise in [0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35]:
    accs={}
    flip=set(rng.choice(real_ids,int(noise*len(real_ids)),replace=False))
    for m in MODELS:
        cor=wr=0
        for r in base:
            if r["model"]!=m or not valid(r) or r["decision"] not in("EVET","HAYIR"): continue
            g=r["gold"]
            if r["probe_id"] in flip: g="NO"  # flipped
            pred="EVET" if r["decision"]=="EVET" else "HAYIR"
            ok = (pred==("EVET" if g=="YES" else "HAYIR"))
            cor+=ok; wr+=(not ok)
        accs[m]=cor/(cor+wr) if (cor+wr) else float("nan")
    rank=[PRETTY[m] for m in sorted(MODELS,key=lambda m:-accs[m])]
    tau=stats.spearmanr([base_rank.index(PRETTY[m]) for m in MODELS],[rank.index(PRETTY[m]) for m in MODELS]).correlation
    ns.append({"noise":f"{noise:.0%}","top_model":rank[0],"Claude_cacc":round(accs['claude-opus-4-8'],3),"rank_spearman_vs_0":round(float(tau),3)})
nsd=pd.DataFrame(ns); print(nsd.to_string(index=False))
print("   -> Spearman~1 ise sıralama gürültüye dayanıklı.")
nsd.to_csv(f"{OUT}/r10_label_noise.csv",index=False); results["label_noise"]=nsd.to_dict("records")

# ---------- 11) H1 multiple-comparison correction ----------
fake=lambda m:[r for r in base if r["model"]==m and r["type"]=="ATTRIB_FAKE" and valid(r)]
pvals=[]; labels=[]
for m in MODELS:
    df=pd.DataFrame([{"fa":1 if r["decision"]=="EVET" else 0,"imp":bv2[r["probe_id"]].get("importance")}
                     for r in fake(m) if bv2[r["probe_id"]].get("importance") is not None])
    try:
        mm=smf.logit("fa ~ imp",df).fit(disp=0); pvals.append(float(mm.pvalues["imp"])); labels.append(PRETTY[m])
    except Exception: pvals.append(1.0); labels.append(PRETTY[m])
rej,padj,_,_=multipletests(pvals,method="holm")
h1=pd.DataFrame({"model":labels,"p_raw":[f"{p:.3f}" for p in pvals],"p_holm":[f"{p:.3f}" for p in padj],"sig_after":rej})
print("\n[11] H1 ÇOKLU-KARŞILAŞTIRMA (Holm) — obscurity→fabrication\n",h1.to_string(index=False))
print("   -> Holm sonrası anlamlı kalan yoksa H1 kesin keşifsel/null.")
h1.to_csv(f"{OUT}/r11_h1_corrected.csv",index=False); results["h1_corrected"]=h1.to_dict("records")

# ---------- 12) FAKE-TITLE ARTIFACT STATS (char n-gram perplexity, length, diacritics) ----------
real_titles=[b["work"] for b in bv2.values() if b["type"]=="ATTRIB_REAL" and b["id"] not in EX]
fake_titles=[b["work"] for b in bv2.values() if b["type"]=="ATTRIB_FAKE"]
# train char trigram model on REAL titles, score perplexity of real vs fake
def char_trigram_model(titles):
    from collections import defaultdict
    tri=defaultdict(lambda:defaultdict(int))
    for t in titles:
        s="^^"+t+"$"
        for i in range(len(s)-2): tri[s[i:i+2]][s[i+2]]+=1
    return tri
def perplexity(t,tri):
    s="^^"+t+"$"; logp=0.0; n=0
    for i in range(len(s)-2):
        ctx=s[i:i+2]; ch=s[i+2]; tot=sum(tri[ctx].values())
        p=(tri[ctx][ch]+1)/(tot+50); logp+=math.log(p); n+=1
    return math.exp(-logp/n) if n else float("nan")
# split-half: train on half of real, score other-half-real vs fake (avoid trivial memorization)
half=len(real_titles)//2
trm=char_trigram_model(real_titles[:half])
ppx_real=[perplexity(t,trm) for t in real_titles[half:]]
ppx_fake=[perplexity(t,trm) for t in fake_titles]
DIAC=set("āīūṣḥḍṭẓʿʾâîûêôşçğüöı")
def diac(t): return sum(c in DIAC for c in t.lower())/max(len(t),1)
ft=pd.DataFrame([
    {"set":"REAL(held-out)","n":len(ppx_real),"mean_perplexity":round(float(np.mean(ppx_real)),2),
     "mean_chars":round(float(np.mean([len(t) for t in real_titles])),1),
     "mean_words":round(float(np.mean([len(t.split()) for t in real_titles])),2),
     "diacritic_density":round(float(np.mean([diac(t) for t in real_titles])),3)},
    {"set":"FAKE","n":len(ppx_fake),"mean_perplexity":round(float(np.mean(ppx_fake)),2),
     "mean_chars":round(float(np.mean([len(t) for t in fake_titles])),1),
     "mean_words":round(float(np.mean([len(t.split()) for t in fake_titles])),2),
     "diacritic_density":round(float(np.mean([diac(t) for t in fake_titles])),3)}])
# AUC of perplexity separating fake from real (held-out)
from sklearn.metrics import roc_auc_score
y=[0]*len(ppx_real)+[1]*len(ppx_fake); x=ppx_real+ppx_fake
auc=roc_auc_score(y,x)
print("\n[12] FAKE-TITLE ARTEFAKT İSTATİSTİĞİ (char-trigram perplexity, uzunluk, diacritic)\n",ft.to_string(index=False))
print(f"   perplexity ile FAKE-vs-REAL ayrım AUC = {auc:.3f}  (0.5=ayırt edilemez; yüksekse fakes yüzeyden ayrışıyor -> insan-doğrulama P1 şart)")
ft.to_csv(f"{OUT}/r12_faketitle_stats.csv",index=False); results["faketitle_stats"]={"table":ft.to_dict("records"),"perplexity_auc":round(float(auc),3)}

json.dump(results,open(f"{OUT}/04d_robustness.json","w",encoding="utf-8"),ensure_ascii=False,indent=2,default=str)
print("\n"+"="*60+"\n04d tamam. Tablolar r1..r12 + 04d_robustness.json ->",OUT)
