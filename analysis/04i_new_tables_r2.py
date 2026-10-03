#!/usr/bin/env python3
"""
04i_new_tables_r2.py — R2-round additions (Aug 2026), all computed from released raw outputs.
Outputs (to paper/tables/regen_nt/):
  r14_rel_vs_ana_committed_yes.csv  — per-model committed-YES on RELATION_REAL vs ANACHRONISM (R4.1)
  r15_output_ledger.csv             — auditable decomposition of the 64,216 raw records (R2.2 / R4.6)
  r16_response_mix.csv              — per-model EVET/HAYIR/BILMIYORUM mix + always-NO/always-YES baselines (R4.1)
  r17_gal_denominators.csv          — GAL fake/real cells with explicit denominators per language (R4.5)
"""
import json, collections, os, sys

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = f"{WS}/data/results"; W = f"{WS}/data/benchmarks"; OUT = f"{WS}/tables/regen_nt"; os.makedirs(OUT, exist_ok=True)

MODELS = ["claude-opus-4-8", "claude-haiku-4-5", "gemini-2.5-flash-nt", "qwen2.5:32b",
          "qwen2.5:14b", "qwen2.5:7b", "gemma3:4b", "llama3.1:8b",
          "mistral-small:latest", "phi4:latest"]
PRETTY = {"claude-opus-4-8": "Claude-Opus", "claude-haiku-4-5": "Claude-Haiku",
          "gemini-2.5-flash-nt": "Gemini-Flash(nt)", "qwen2.5:32b": "Qwen-32B",
          "qwen2.5:14b": "Qwen-14B", "qwen2.5:7b": "Qwen-7B", "gemma3:4b": "Gemma3-4B",
          "llama3.1:8b": "Llama3.1-8B", "mistral-small:latest": "Mistral-Small",
          "phi4:latest": "Phi-4"}

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

base = load(f"{U}/benchmark_results_v2_base.jsonl")
gal = load(f"{U}/benchmark_results_gal.jsonl")
bv2 = {b["id"]: b for b in load(f"{W}/benchmark_v2.jsonl")}
EX = set(json.load(open(f"{W}/attrib_real_exclude.json"))["exclude_ids"])

def valid(r):
    return r["outcome"] != "parse_fail" and r["probe_id"] not in EX

import csv as _csv
def write_csv(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print("wrote", path)

# ---------- r14: RELATION_REAL vs ANACHRONISM committed-YES ----------
rows = []
for m in MODELS:
    r = {"model": PRETTY[m]}
    for typ, tag in (("RELATION_REAL", "rel"), ("ANACHRONISM", "ana")):
        rs = [x for x in base if x["model"] == m and x["type"] == typ and valid(x)]
        yes = sum(x["decision"] == "EVET" for x in rs)
        comm = sum(x["decision"] in ("EVET", "HAYIR") for x in rs)
        r[f"{tag}_yes"] = yes; r[f"{tag}_committed"] = comm
        r[f"{tag}_yes_rate"] = round(yes / comm, 3) if comm else float("nan")
        r[f"{tag}_abstain"] = len(rs) - comm
    rows.append(r)
write_csv(f"{OUT}/r14_rel_vs_ana_committed_yes.csv", rows,
          ["model", "rel_yes", "rel_committed", "rel_yes_rate", "rel_abstain",
           "ana_yes", "ana_committed", "ana_yes_rate", "ana_abstain"])

# ---------- r15: output ledger ----------
FILES = [("v2_base", "benchmark_results_v2_base.jsonl"),
         ("condB", "benchmark_results_condB.jsonl"),
         ("cot", "benchmark_results_cot.jsonl"),
         ("gal", "benchmark_results_gal.jsonl"),
         ("promptrob", "benchmark_results_promptrob.jsonl"),
         ("consist", "benchmark_results_consist.jsonl")]
led = []; grand = 0; grand_uniq = 0; grand_retry = 0; grand_pf = 0
for cond, fn in FILES:
    recs = load(f"{U}/{fn}")
    grand += len(recs)
    bym = collections.Counter(r["model"] for r in recs)
    for m in sorted(bym):
        rs = [r for r in recs if r["model"] == m]
        # attempt key: item x (lang) x (paraphrase) x (sample index)
        def key(r):
            return (r["probe_id"], r.get("lang"), r.get("prompt_id"), r.get("sample_idx"))
        uniq = len({key(r) for r in rs})
        pf = sum(r["outcome"] == "parse_fail" for r in rs)
        err = sum(str(r.get("raw", "")).startswith("__ERROR__") for r in rs)
        arm = ("primary" if m in MODELS else
               "sensitivity (thinking budget 1024)" if m == "gemini-2.5-flash" else
               "forced-choice arm only" if m == "gemini-2.5-pro" else
               "aborted partial run (excluded)" if m == "gemma4:31b" else "other")
        led.append({"condition": cond, "model": m, "arm": arm, "rows": len(rs), "unique_attempts": uniq,
                    "retained_pre_retry_rows": len(rs) - uniq, "parse_fail_rows": pf,
                    "transport_error_rows": err})
        grand_uniq += uniq; grand_retry += len(rs) - uniq; grand_pf += pf
grand_err = sum(int(r["transport_error_rows"]) for r in led)
led.append({"condition": "TOTAL", "model": "(all files)", "arm": "", "rows": grand, "unique_attempts": grand_uniq,
            "retained_pre_retry_rows": grand_retry, "parse_fail_rows": grand_pf, "transport_error_rows": grand_err})
write_csv(f"{OUT}/r15_output_ledger.csv", led,
          ["condition", "model", "arm", "rows", "unique_attempts", "retained_pre_retry_rows",
           "parse_fail_rows", "transport_error_rows"])
print(f"   ledger grand total rows = {grand}")

# ---------- r16: response mix + baselines ----------
scored = [i for i in bv2 if i not in EX]
gold = {i: bv2[i]["gold"] for i in scored}
n_yes = sum(1 for i in scored if gold[i] == "YES"); n_no = len(scored) - n_yes
rows = []
for m in MODELS:
    rs = [r for r in base if r["model"] == m and valid(r)]
    c = collections.Counter(r["decision"] for r in rs)
    raw = sum(r["outcome"] == "correct" for r in rs) / len(rs)
    rows.append({"model": PRETTY[m], "n": len(rs),
                 "EVET": c.get("EVET", 0), "HAYIR": c.get("HAYIR", 0),
                 "BILMIYORUM": c.get("BILMIYORUM", 0),
                 "EVET_pct": f"{100*c.get('EVET',0)/len(rs):.1f}%",
                 "HAYIR_pct": f"{100*c.get('HAYIR',0)/len(rs):.1f}%",
                 "BILMIYORUM_pct": f"{100*c.get('BILMIYORUM',0)/len(rs):.1f}%",
                 "raw_acc": round(raw, 3)})
rows.append({"model": "always-NO baseline", "n": len(scored), "EVET": 0, "HAYIR": len(scored),
             "BILMIYORUM": 0, "EVET_pct": "0.0%", "HAYIR_pct": "100.0%", "BILMIYORUM_pct": "0.0%",
             "raw_acc": round(n_no / len(scored), 3)})
rows.append({"model": "always-YES baseline", "n": len(scored), "EVET": len(scored), "HAYIR": 0,
             "BILMIYORUM": 0, "EVET_pct": "100.0%", "HAYIR_pct": "0.0%", "BILMIYORUM_pct": "0.0%",
             "raw_acc": round(n_yes / len(scored), 3)})
write_csv(f"{OUT}/r16_response_mix.csv", rows,
          ["model", "n", "EVET", "HAYIR", "BILMIYORUM", "EVET_pct", "HAYIR_pct",
           "BILMIYORUM_pct", "raw_acc"])
print(f"   scored={len(scored)} gold YES={n_yes} NO={n_no}")

# ---------- r17: GAL denominators ----------
rows = []
for m in MODELS:
    r = {"model": PRETTY[m]}
    for lang in ("tr", "ar"):
        fake = [x for x in gal if x["model"] == m and x.get("lang") == lang
                and x["type"] == "ATTRIB_FAKE" and x["outcome"] != "parse_fail"]
        real = [x for x in gal if x["model"] == m and x.get("lang") == lang
                and x["type"] == "ATTRIB_REAL" and x["outcome"] != "parse_fail"]
        f_yes = sum(x["decision"] == "EVET" for x in fake)
        f_comm = sum(x["decision"] in ("EVET", "HAYIR") for x in fake)
        r_yes = sum(x["decision"] == "EVET" for x in real)
        r_comm = sum(x["decision"] in ("EVET", "HAYIR") for x in real)
        r[f"{lang}_fake_approvals"] = f_yes
        r[f"{lang}_fake_committed"] = f_comm
        r[f"{lang}_fake_FA"] = round(f_yes / f_comm, 3) if f_comm else float("nan")
        r[f"{lang}_fake_abstain"] = len(fake) - f_comm
        r[f"{lang}_real_hits"] = r_yes
        r[f"{lang}_real_committed"] = r_comm
        r[f"{lang}_real_hit_rate"] = round(r_yes / r_comm, 3) if r_comm else float("nan")
    rows.append(r)
cols = ["model"]
for lang in ("tr", "ar"):
    cols += [f"{lang}_fake_approvals", f"{lang}_fake_committed", f"{lang}_fake_FA",
             f"{lang}_fake_abstain", f"{lang}_real_hits", f"{lang}_real_committed",
             f"{lang}_real_hit_rate"]
write_csv(f"{OUT}/r17_gal_denominators.csv", rows, cols)
print("done.")
