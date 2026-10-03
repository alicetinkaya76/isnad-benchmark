#!/usr/bin/env python3
"""
04j_prompt_stability_nt.py — regenerate the prompt-stability table (f1) from the
released raw records (R3 round, Sept 2026).

Why this script exists: the original f1_prompt_stability.csv was produced in the
initial analysis environment, whose P1 subset was re-derived by re-executing the
sampling code against a working copy of the benchmark; its P1 row is not
reproducible from the released records (e.g. Qwen-32B P1 abstention 0.663 vs the
0.637 obtained from the released files). This script recomputes every cell from
the released benchmark_results_v2_base.jsonl / benchmark_results_promptrob.jsonl
only, defining the 300-item subset as exactly the items present in the released
prompt-variation records. Arm samples are analyzed as designed (300 items; the
sample predates the 12-item exclusion screen and contains one later-excluded
item, immaterial to the within-arm contrasts; see the manuscript's Section 3.3
sensitivity bound).

Outputs: tables/main/f1_prompt_stability.csv (same schema as before)
Prints: per-model P1-P4 values, committed-accuracy and abstention ranges,
        pairwise rank-stability Spearman means, and the derived text claims.
"""
import json, collections, os, csv
import numpy as np
from scipy import stats

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = f"{_R}/data/results"
import os
OUT = f"{_R}/tables/main"
os.makedirs(OUT, exist_ok=True)

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

base = load(f"{U}/benchmark_results_v2_base.jsonl")
pr = load(f"{U}/benchmark_results_promptrob.jsonl")
sub300 = set(r["probe_id"] for r in pr)
assert len(sub300) == 300, len(sub300)

PRIM = ["claude-opus-4-8", "claude-haiku-4-5", "gemini-2.5-flash-nt", "qwen2.5:32b",
        "qwen2.5:14b", "qwen2.5:7b", "gemma3:4b", "llama3.1:8b",
        "mistral-small:latest", "phi4:latest"]
P = {"claude-opus-4-8": "Claude-Opus", "claude-haiku-4-5": "Claude-Haiku",
     "gemini-2.5-flash-nt": "Gemini-Flash(nt)", "qwen2.5:32b": "Qwen-32B",
     "qwen2.5:14b": "Qwen-14B", "qwen2.5:7b": "Qwen-7B", "gemma3:4b": "Gemma3-4B",
     "llama3.1:8b": "Llama3.1-8B", "mistral-small:latest": "Mistral-Small",
     "phi4:latest": "Phi-4"}

def ece_of(rs):
    cc = [(r["confidence"], 1 if r["outcome"] == "correct" else 0) for r in rs
          if r["decision"] in ("EVET", "HAYIR") and isinstance(r.get("confidence"), (int, float))]
    if len(cc) < 10:
        return float("nan")
    c = np.array([x for x, _ in cc]); y = np.array([x for _, x in cc])
    N = len(c); e = 0.0; ed = np.linspace(0, 1, 11)
    for i in range(10):
        m = ((c >= ed[i]) & (c < ed[i+1])) if i < 9 else ((c >= ed[i]) & (c <= ed[i+1]))
        if m.sum():
            e += (m.sum() / N) * abs(y[m].mean() - c[m].mean())
    return e

def met(rs):
    n = len(rs)
    cor = sum(r["outcome"] == "correct" for r in rs)
    wr = sum(r["outcome"] == "wrong" for r in rs)
    ab = sum(r["outcome"] == "abstain" for r in rs)
    comm = cor + wr
    return dict(cacc=cor/comm if comm else float("nan"), abst=ab/n if n else float("nan"),
                err=wr/n if n else float("nan"), ece=ece_of(rs))

rows = []
for pid in ["P1", "P2", "P3", "P4"]:
    for m in PRIM:
        if pid == "P1":
            rs = [r for r in base if r["model"] == m and r["probe_id"] in sub300
                  and r["outcome"] != "parse_fail"]
        else:
            rs = [r for r in pr if r["model"] == m and r["prompt_id"] == pid
                  and r["outcome"] != "parse_fail"]
        mm = met(rs)
        rows.append({"prompt": pid, "model": P[m], **{k: round(v, 3) for k, v in mm.items()}})

with open(f"{OUT}/f1_prompt_stability.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["prompt", "model", "cacc", "abst", "err", "ece"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
print("wrote", f"{OUT}/f1_prompt_stability.csv")

by = collections.defaultdict(dict)
for r in rows:
    by[r["model"]][r["prompt"]] = r

print("\nper-model P1-P4 abstention and swings:")
for m in [P[x] for x in PRIM]:
    ab = [by[m][p]["abst"] for p in ("P1", "P2", "P3", "P4")]
    ca = [by[m][p]["cacc"] for p in ("P1", "P2", "P3", "P4")]
    print(f"  {m:18s} abst {ab} range {min(ab):.3f}-{max(ab):.3f} swing {100*(max(ab)-min(ab)):.1f} | cacc range {max(ca)-min(ca):.3f}")

def rank_stab(metric):
    R = {}
    for pid in ["P1", "P2", "P3", "P4"]:
        vals = {m: by[m][pid][metric] for m in by}
        order = sorted(vals, key=lambda k: vals[k], reverse=(metric != "ece"))
        R[pid] = [order.index(m) for m in sorted(by)]
    sps = []
    pids = ["P1", "P2", "P3", "P4"]
    for i in range(4):
        for j in range(i+1, 4):
            sps.append(stats.spearmanr(R[pids[i]], R[pids[j]]).correlation)
    return float(np.mean(sps))

print(f"\nrank stability (mean pairwise Spearman over 6 prompt pairs): "
      f"cacc={rank_stab('cacc'):.2f}  ece={rank_stab('ece'):.2f}")
opus_first_cacc = all(max(by[m][p]["cacc"] for m in by) == by["Claude-Opus"][p]["cacc"] for p in ["P1","P2","P3","P4"])
anth_first_ece = all(min(by[m][p]["ece"] for m in by) in (by["Claude-Opus"][p]["ece"], by["Claude-Haiku"][p]["ece"]) for p in ["P1","P2","P3","P4"])
print(f"Opus first on cacc under all four: {opus_first_cacc}; Anthropic first on ECE under all four: {anth_first_ece}")
print(f"Opus ECE band: {min(by['Claude-Opus'][p]['ece'] for p in ['P1','P2','P3','P4']):.3f}-{max(by['Claude-Opus'][p]['ece'] for p in ['P1','P2','P3','P4']):.3f}")
for m in ("Gemini-Flash(nt)", "Llama3.1-8B", "Phi-4"):
    print(f"max abstention {m}: {100*max(by[m][p]['abst'] for p in ['P1','P2','P3','P4']):.1f}%")
