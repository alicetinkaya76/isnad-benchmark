#!/usr/bin/env python3
# 04g = 04c re-keyed to the 10 primary configurations (Gemini = gemini-2.5-flash-nt, the matched primary arm). R2 revision, Aug 2026.
"""
04c_reanalyze.py — ISNAD tam reanaliz (v2 + Condition B + GAL TR/AR)
====================================================================
Çıktılar: tablo_*.csv, 04c_results.json, 04c_report.md (+ konsol özeti).
Çekirdek tez: accuracy-only değerlendirme tahmin yürütmeyi (Gemini) ve yanıt-yanlılığını
(Gemma) ödüllendirir, kalibre çekimserliği (Claude) cezalandırır. Kapalı/denetlenebilir/
aritmetik-doğrulanabilir benchmark bunu gösterir.
"""
import json, math, collections
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from sklearn.metrics import matthews_corrcoef
import statsmodels.formula.api as smf
import warnings; warnings.filterwarnings("ignore")

import os; _R=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U=f"{_R}/data/results"; W=f"{_R}/data/benchmarks"; OUT=f"{_R}/tables/regen_nt"; os.makedirs(OUT, exist_ok=True)
MODELS=["claude-opus-4-8","claude-haiku-4-5","gemini-2.5-flash-nt","qwen2.5:32b","qwen2.5:14b","qwen2.5:7b","gemma3:4b","llama3.1:8b","mistral-small:latest","phi4:latest"]
PRETTY={"claude-opus-4-8":"Claude Opus 4.8","claude-haiku-4-5":"Claude Haiku 4.5","gemini-2.5-flash-nt":"Gemini 2.5 Flash (nt)","qwen2.5:32b":"Qwen2.5-32B","qwen2.5:14b":"Qwen2.5-14B","qwen2.5:7b":"Qwen2.5-7B","gemma3:4b":"Gemma3-4B","llama3.1:8b":"Llama3.1-8B","mistral-small:latest":"Mistral-Small","phi4:latest":"Phi-4"}
YES_TYPES={"ATTRIB_REAL","RELATION_REAL"}; NO_TYPES={"ATTRIB_FAKE","ANACHRONISM","MISATTRIB"}

def load(p): return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
base=load(f"{U}/benchmark_results_v2_base.jsonl")
condB=load(f"{U}/benchmark_results_condB.jsonl")
gal=load(f"{U}/benchmark_results_gal.jsonl")
bv2={b["id"]:b for b in load(f"{W}/benchmark_v2.jsonl")}
bgal={b["id"]:b for b in load(f"{W}/benchmark_gal.jsonl")}
EXCLUDE=set(json.load(open(f"{W}/attrib_real_exclude.json"))["exclude_ids"])

def valid(r): return r["outcome"]!="parse_fail" and r["probe_id"] not in EXCLUDE
def pred_bin(dec): return 1 if dec=="EVET" else (0 if dec=="HAYIR" else None)  # None=abstain
def gold_bin(g): return 1 if g=="YES" else 0

def wilson(k,n):
    if n==0: return (float("nan"),)*3
    lo,hi=proportion_confint(k,n,0.05,"wilson"); return (k/n,lo,hi)
def zc(p): return float(stats.norm.ppf(min(max(p,1e-6),1-1e-6)))
def dprime(hits,nsig,fa,nno):
    hr=(hits+0.5)/(nsig+1); far=(fa+0.5)/(nno+1)
    return zc(hr)-zc(far), -0.5*(zc(hr)+zc(far)), hr, far
def ece_brier(confs,corr,nb=10):
    c=np.array(confs,float); y=np.array(corr,float); N=len(c)
    if N==0: return float("nan"),float("nan")
    e=0.0; edges=np.linspace(0,1,nb+1)
    for i in range(nb):
        m=((c>=edges[i])&(c<edges[i+1])) if i<nb-1 else ((c>=edges[i])&(c<=edges[i+1]))
        if m.sum(): e+=(m.sum()/N)*abs(y[m].mean()-c[m].mean())
    return e, float(np.mean((c-y)**2))

results={}

# ============ 1) CORE (main, Condition A) ============
def core_table(records, label, exclude_ok=True):
    rows=[]
    for m in MODELS:
        rs=[r for r in records if r["model"]==m and (valid(r) if exclude_ok else r["outcome"]!="parse_fail")]
        n=len(rs); cor=sum(r["outcome"]=="correct" for r in rs)
        wr=sum(r["outcome"]=="wrong" for r in rs); ab=sum(r["outcome"]=="abstain" for r in rs)
        comm=cor+wr
        ra,ralo,rahi=wilson(cor,n); ca,calo,cahi=wilson(cor,comm) if comm else (float("nan"),)*3
        ab_r,_,_=wilson(ab,n)
        rows.append({"model":PRETTY[m],"N":n,"correct":cor,"wrong":wr,"abstain":ab,
            "raw_acc":round(ra,3),
            "committed_acc":round(ca,3),  # cluster-bootstrap CIs reported once, in r2 (Table S16)
            "abstention":round(ab_r,3),"error_rate":round(wr/n,3)})
    return pd.DataFrame(rows)
core=core_table(base,"main"); print("="*70,"\n[1] CORE (ana benchmark, Condition A)\n",core.to_string(index=False))
core.to_csv(f"{OUT}/tablo1_core.csv",index=False)
results["core"]=core.to_dict("records")

# ============ 2) SDT authorship (ATTRIB_REAL vs FAKE, committed) ============
sdt_rows=[]
for m in MODELS:
    real=[r for r in base if r["model"]==m and r["type"]=="ATTRIB_REAL" and valid(r)]
    fake=[r for r in base if r["model"]==m and r["type"]=="ATTRIB_FAKE" and valid(r)]
    hits=sum(r["decision"]=="EVET" for r in real); rmiss=sum(r["decision"]=="HAYIR" for r in real)
    fa=sum(r["decision"]=="EVET" for r in fake); cr=sum(r["decision"]=="HAYIR" for r in fake)
    d,c,hr,far=dprime(hits,hits+rmiss,fa,fa+cr)
    sdt_rows.append({"model":PRETTY[m],"d_prime":round(d,3),"criterion_c":round(c,3),
        "hit_rate":round(hr,3),"false_alarm":round(far,3),
        "real_abstain":round(sum(r["outcome"]=="abstain" for r in real)/len(real),3),
        "fake_abstain":round(sum(r["outcome"]=="abstain" for r in fake)/len(fake),3)})
sdt=pd.DataFrame(sdt_rows); print("\n[2] SDT — atıf tanıma (committed; d'=ayrım, c>0 NO-yanlı/temkinli)\n",sdt.to_string(index=False))
sdt.to_csv(f"{OUT}/tablo2_sdt.csv",index=False); results["sdt"]=sdt.to_dict("records")

# ============ 3) Calibration (committed EVET/HAYIR) ============
cal_rows=[]
for m in MODELS:
    rs=[r for r in base if r["model"]==m and valid(r) and r["decision"] in ("EVET","HAYIR")
        and isinstance(r.get("confidence"),(int,float))]
    confs=[r["confidence"] for r in rs]; corr=[1 if r["outcome"]=="correct" else 0 for r in rs]
    e,b=ece_brier(confs,corr); 
    cal_rows.append({"model":PRETTY[m],"ECE":round(e,3),"Brier":round(b,3),
        "mean_conf":round(float(np.mean(confs)),3),"mean_acc":round(float(np.mean(corr)),3),
        "overconf":round(float(np.mean(confs))-float(np.mean(corr)),3),"n_committed":len(rs)})
cal=pd.DataFrame(cal_rows); print("\n[3] KALİBRASYON (committed; ECE düşük=iyi, overconf>0=aşırı-güven)\n",cal.to_string(index=False))
cal.to_csv(f"{OUT}/tablo3_calibration.csv",index=False); results["calibration"]=cal.to_dict("records")

# ============ 4) MCC (committed + strict) ============
mcc_rows=[]
for m in MODELS:
    rs=[r for r in base if r["model"]==m and valid(r)]
    yt_c,yp_c,yt_s,yp_s=[],[],[],[]
    for r in rs:
        g=gold_bin(r["gold"]); p=pred_bin(r["decision"])
        if p is not None: yt_c.append(g); yp_c.append(p)
        yt_s.append(g); yp_s.append(p if p is not None else 1-g)  # abstain->wrong label
    mcc_c=matthews_corrcoef(yt_c,yp_c) if len(set(yt_c))>1 and len(set(yp_c))>1 else float("nan")
    mcc_s=matthews_corrcoef(yt_s,yp_s) if len(set(yt_s))>1 and len(set(yp_s))>1 else float("nan")
    mcc_rows.append({"model":PRETTY[m],"MCC_committed":round(mcc_c,3),"MCC_strict(abstain=wrong)":round(mcc_s,3)})
mcc=pd.DataFrame(mcc_rows); print("\n[4] MCC (committed: yanıt verince ayrım; strict: çekimser=hata cezası)\n",mcc.to_string(index=False))
mcc.to_csv(f"{OUT}/tablo4_mcc.csv",index=False); results["mcc"]=mcc.to_dict("records")

# ============ 5) Per-type committed-acc + abstention ============
types=["ATTRIB_REAL","ATTRIB_FAKE","ANACHRONISM","RELATION_REAL","MISATTRIB"]
pt_rows=[]
for m in MODELS:
    row={"model":PRETTY[m]}
    for t in types:
        rs=[r for r in base if r["model"]==m and r["type"]==t and valid(r)]
        cor=sum(r["outcome"]=="correct" for r in rs); wr=sum(r["outcome"]=="wrong" for r in rs)
        ab=sum(r["outcome"]=="abstain" for r in rs); comm=cor+wr
        row[f"{t}_cacc"]=round(cor/comm,3) if comm else float("nan")
        row[f"{t}_abs"]=round(ab/len(rs),3) if rs else float("nan")
    pt_rows.append(row)
pt=pd.DataFrame(pt_rows); print("\n[5] PER-TİP (cacc=committed-acc, abs=abstention)\n",pt.to_string(index=False))
pt.to_csv(f"{OUT}/tablo5_per_type.csv",index=False); results["per_type"]=pt.to_dict("records")

# ============ 6) ANACHRONISM by gap bin (approval=EVET, yanlış) ============
def gap_bin(g):
    if g<=50: return "1-50"
    if g<=120: return "51-120"
    if g<=200: return "121-200"
    if g<=400: return "201-400"
    return "401+"
BINS=["1-50","51-120","121-200","201-400","401+"]
ana_rows=[]
for m in MODELS:
    row={"model":PRETTY[m]}
    for bn in BINS:
        rs=[r for r in base if r["model"]==m and r["type"]=="ANACHRONISM"
            and gap_bin(bv2[r["probe_id"]]["gap"])==bn]
        appr=sum(r["decision"]=="EVET" for r in rs)
        row[bn]=f"{appr}/{len(rs)}={100*appr/len(rs):.0f}%" if rs else "-"
    # overall approval
    allr=[r for r in base if r["model"]==m and r["type"]=="ANACHRONISM"]
    row["TOPLAM_onay"]=f"{100*sum(r['decision']=='EVET' for r in allr)/len(allr):.1f}%"
    ana_rows.append(row)
ana=pd.DataFrame(ana_rows); print("\n[6] ANACHRONISM — gap-bin onay oranı (EVET=imkansız bağı yanlış onaylama; düşük=iyi)\n",ana.to_string(index=False))
ana.to_csv(f"{OUT}/tablo6_anachronism_gap.csv",index=False); results["anachronism"]=ana.to_dict("records")

# ============ 7) H1 obscurity->fabrication (ATTRIB_FAKE false-approval) — exploratory ============
h1_rows=[]; h1_logit={}
for m in MODELS:
    fake=[r for r in base if r["model"]==m and r["type"]=="ATTRIB_FAKE" and valid(r)]
    row={"model":PRETTY[m]}
    for ob in ["obscure","mid","famous"]:
        sub=[r for r in fake if r.get("obscurity")==ob]
        fap=sum(r["decision"]=="EVET" for r in sub)
        row[f"{ob}_falseappr"]=round(fap/len(sub),3) if sub else float("nan")
    # logistic: false_approval ~ importance (continuous)
    df=pd.DataFrame([{"fa":1 if r["decision"]=="EVET" else 0,"imp":bv2[r["probe_id"]].get("importance")}
                     for r in fake if bv2[r["probe_id"]].get("importance") is not None])
    try:
        mod=smf.logit("fa ~ imp",df).fit(disp=0)
        row["OR_importance"]=round(float(np.exp(mod.params["imp"])),3); row["p"]=round(float(mod.pvalues["imp"]),3)
    except Exception:
        row["OR_importance"]=float("nan"); row["p"]=float("nan")
    h1_rows.append(row)
h1=pd.DataFrame(h1_rows); print("\n[7] H1 (EXPLORATORY) — obscurity->fabrication (ATTRIB_FAKE false-approval; OR<1=obscure↑fabrication)\n",h1.to_string(index=False))
h1.to_csv(f"{OUT}/tablo7_h1_obscurity.csv",index=False); results["h1"]=h1.to_dict("records")

# ============ 8) Risk-coverage (selective prediction) ============
rc_rows=[]
for m in MODELS:
    rs=[r for r in base if r["model"]==m and valid(r)]
    N=len(rs)
    # natural operating point: committed = model's own abstention mechanism
    comm=[r for r in rs if r["decision"] in ("EVET","HAYIR")]
    nat_cov=len(comm)/N; nat_acc=sum(r["outcome"]=="correct" for r in comm)/len(comm) if comm else float("nan")
    # confidence-thresholded curve -> AURC (risk vs coverage)
    cc=sorted([(r["confidence"] if isinstance(r.get("confidence"),(int,float)) else 0.0,
                1 if r["outcome"]=="correct" else 0) for r in comm], key=lambda x:-x[0])
    covs=[]; risks=[]
    for k in range(1,len(cc)+1):
        covs.append(k/N); risks.append(1-np.mean([y for _,y in cc[:k]]))
    aurc=float(np.trapezoid(risks,covs)) if len(covs)>1 else float("nan")
    # acc@coverage 50%
    k50=max(1,int(0.5*N)); k50=min(k50,len(cc))
    acc50=np.mean([y for _,y in cc[:k50]]) if cc else float("nan")
    rc_rows.append({"model":PRETTY[m],"nat_coverage":round(nat_cov,3),"nat_acc":round(nat_acc,3),
        "acc@50%cov":round(float(acc50),3),"AURC":round(aurc,4)})
rc=pd.DataFrame(rc_rows); print("\n[8] RISK-COVERAGE (nat=doğal çekimser nokta; AURC düşük=iyi selective pred.)\n",rc.to_string(index=False))
rc.to_csv(f"{OUT}/tablo8_risk_coverage.csv",index=False); results["risk_coverage"]=rc.to_dict("records")

# ============ 9) Trivial baselines ============
allitems=[bv2[i] for i in bv2 if i not in EXCLUDE]
nyes=sum(b["gold"]=="YES" for b in allitems); nno=len(allitems)-nyes
base_rows=[{"baseline":"always-NO","accuracy":round(nno/len(allitems),3)},
           {"baseline":"always-YES","accuracy":round(nyes/len(allitems),3)},
           {"baseline":"random-50/50","accuracy":0.5},
           {"baseline":"majority-class","accuracy":round(max(nyes,nno)/len(allitems),3)}]
bl=pd.DataFrame(base_rows); print("\n[9] TRİVİAL BASELINE'lar (gold skew: %d YES / %d NO)"%(nyes,nno),"\n",bl.to_string(index=False))
results["baselines"]={"always_NO":nno/len(allitems),"always_YES":nyes/len(allitems),"n_yes":nyes,"n_no":nno}

# ============ 10) Significance (key pairwise, committed-acc & anachronism) ============
def comm_correct(records,m):
    rs=[r for r in records if r["model"]==m and valid(r) and r["decision"] in ("EVET","HAYIR")]
    return sum(r["outcome"]=="correct" for r in rs), len(rs)
sig=[]
pairs=[("claude-opus-4-8","gemini-2.5-flash-nt"),("claude-opus-4-8","qwen2.5:32b"),
       ("qwen2.5:32b","qwen2.5:7b"),("gemini-2.5-flash-nt","gemma3:4b")]
for a,b in pairs:
    ca,na=comm_correct(base,a); cb,nb=comm_correct(base,b)
    zst,p=proportions_ztest([ca,cb],[na,nb])
    sig.append({"comparison":f"{PRETTY[a]} vs {PRETTY[b]}","metric":"committed_acc",
        "A":f"{ca}/{na}={ca/na:.3f}","B":f"{cb}/{nb}={cb/nb:.3f}","z":round(zst,2),"p":f"{p:.2e}"})
sigdf=pd.DataFrame(sig); print("\n[10] ANLAMLILIK (two-proportion z, committed-acc)\n",sigdf.to_string(index=False))
sigdf.to_csv(f"{OUT}/tablo10_significance.csv",index=False); results["significance"]=sigdf.to_dict("records")

# ============ 11) Condition A vs B (Kalai mekanizması) ============
condB_ids=set(r["probe_id"] for r in condB)
ab_rows=[]
for m in MODELS:
    A={r["probe_id"]:r for r in base if r["model"]==m and r["probe_id"] in condB_ids and r["outcome"]!="parse_fail"}
    Bd={r["probe_id"]:r for r in condB if r["model"]==m and r["outcome"]!="parse_fail"}
    common=set(A)&set(Bd)
    def rate(d,pred): return sum(1 for pid in common if pred(d[pid]))/len(common)
    A_abs=rate(A,lambda r:r["outcome"]=="abstain"); B_abs=rate(Bd,lambda r:r["outcome"]=="abstain")
    # fabrication on FAKE subset
    fake_ids=[pid for pid in common if A[pid]["type"]=="ATTRIB_FAKE"]
    def fap(d): return sum(d[pid]["decision"]=="EVET" for pid in fake_ids)/len(fake_ids) if fake_ids else float("nan")
    A_fap=fap(A); B_fap=fap(Bd)
    # NO-gold false approval (all NO types)
    no_ids=[pid for pid in common if A[pid]["gold"]=="NO"]
    A_no_fap=sum(A[pid]["decision"]=="EVET" for pid in no_ids)/len(no_ids)
    B_no_fap=sum(Bd[pid]["decision"]=="EVET" for pid in no_ids)/len(no_ids)
    # where did A-abstentions go in B?
    a_abs_ids=[pid for pid in common if A[pid]["outcome"]=="abstain"]
    if a_abs_ids:
        to_cor=sum(Bd[pid]["outcome"]=="correct" for pid in a_abs_ids)/len(a_abs_ids)
        to_wr=sum(Bd[pid]["outcome"]=="wrong" for pid in a_abs_ids)/len(a_abs_ids)
    else: to_cor=to_wr=float("nan")
    ab_rows.append({"model":PRETTY[m],"A_abstain":round(A_abs,3),"B_abstain":round(B_abs,3),
        "A_NOfalseappr":round(A_no_fap,3),"B_NOfalseappr":round(B_no_fap,3),
        "Δfalse_appr":round(B_no_fap-A_no_fap,3),
        "A_abs→correct(B)":round(to_cor,3) if not math.isnan(to_cor) else "-",
        "A_abs→wrong(B)":round(to_wr,3) if not math.isnan(to_wr) else "-"}); 
abdf=pd.DataFrame(ab_rows); print("\n[11] CONDITION A vs B (çekimser yasak; Δfalse_appr>0 => yasak fabrikasyonu artırır)\n",abdf.to_string(index=False))
abdf.to_csv(f"{OUT}/tablo11_condition_AB.csv",index=False); results["condition_AB"]=abdf.to_dict("records")

# ============ 12) GAL clean-gold replication (lang=tr) ============
def gal_core(lang):
    rows=[]
    for m in MODELS:
        rs=[r for r in gal if r["model"]==m and r["lang"]==lang and r["outcome"]!="parse_fail"]
        cor=sum(r["outcome"]=="correct" for r in rs); wr=sum(r["outcome"]=="wrong" for r in rs)
        ab=sum(r["outcome"]=="abstain" for r in rs); comm=cor+wr; n=len(rs)
        # SDT on GAL authorship
        real=[r for r in rs if r["type"]=="ATTRIB_REAL"]; fake=[r for r in rs if r["type"]=="ATTRIB_FAKE"]
        hits=sum(r["decision"]=="EVET" for r in real); rmiss=sum(r["decision"]=="HAYIR" for r in real)
        fa=sum(r["decision"]=="EVET" for r in fake); cr=sum(r["decision"]=="HAYIR" for r in fake)
        d,c,_,_=dprime(hits,hits+rmiss,fa,fa+cr)
        confs=[r["confidence"] for r in rs if r["decision"] in ("EVET","HAYIR") and isinstance(r.get("confidence"),(int,float))]
        corr=[1 if r["outcome"]=="correct" else 0 for r in rs if r["decision"] in ("EVET","HAYIR") and isinstance(r.get("confidence"),(int,float))]
        e,_=ece_brier(confs,corr)
        rows.append({"model":PRETTY[m],"committed_acc":round(cor/comm,3) if comm else float("nan"),
            "abstention":round(ab/n,3),"d_prime":round(d,3),"ECE":round(e,3),
            "fake_falseappr":round(fa/(fa+cr),3) if (fa+cr) else float("nan")})
    return pd.DataFrame(rows)
gtr=gal_core("tr"); print("\n[12] GAL TEMİZ-GOLD replikasyon (TR)\n",gtr.to_string(index=False))
gtr.to_csv(f"{OUT}/tablo12_gal_tr.csv",index=False); results["gal_tr"]=gtr.to_dict("records")

# ============ 13) Cross-lingual TR vs AR (GAL) ============
xl_rows=[]
for m in MODELS:
    row={"model":PRETTY[m]}
    for lang in ["tr","ar"]:
        rs=[r for r in gal if r["model"]==m and r["lang"]==lang and r["outcome"]!="parse_fail"]
        cor=sum(r["outcome"]=="correct" for r in rs); wr=sum(r["outcome"]=="wrong" for r in rs); comm=cor+wr
        ab=sum(r["outcome"]=="abstain" for r in rs)
        fake=[r for r in rs if r["type"]=="ATTRIB_FAKE"]
        fa=sum(r["decision"]=="EVET" for r in fake); cr=sum(r["decision"]=="HAYIR" for r in fake)
        row[f"_{lang}_cacc_raw"]=cor/comm if comm else float("nan")
        row[f"{lang}_cacc"]=round(cor/comm,3) if comm else float("nan")
        row[f"{lang}_abs"]=round(ab/len(rs),3)
        row[f"{lang}_fakeFA"]=round(fa/(fa+cr),3) if (fa+cr) else float("nan")
    row["Δcacc(ar-tr)"]=round(row["_ar_cacc_raw"]-row["_tr_cacc_raw"],3)
    del row["_ar_cacc_raw"], row["_tr_cacc_raw"]
    xl_rows.append(row)
xl=pd.DataFrame(xl_rows); print("\n[13] CROSS-LINGUAL TR vs AR (GAL; cacc=committed-acc, fakeFA=fabrication onayı)\n",xl.to_string(index=False))
xl.to_csv(f"{OUT}/tablo13_crosslingual.csv",index=False); results["crosslingual"]=xl.to_dict("records")

json.dump(results,open(f"{OUT}/04c_results.json","w",encoding="utf-8"),ensure_ascii=False,indent=2,default=str)
print("\n"+"="*70+"\nTüm tablolar + 04c_results.json yazıldı ->",OUT)
