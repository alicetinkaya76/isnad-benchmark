# ISNAD — benchmark data and analysis code

Companion archive for the article

> Çetinkaya, A. (2026). Accuracy is not reliability: abstention-aware evaluation of large language models on bio-bibliographic claim verification in classical Islamic scholarship. *Information Processing & Management* (in press).

The archive contains the benchmark items with gold labels, the raw model outputs, blank expert-study forms with answer keys, and the analysis scripts that regenerate the article's tables. It is deliberately minimal; see *Not included* below.

## Contents

| Path | Description |
|---|---|
| `data/benchmarks/benchmark_v2.jsonl` | Primary benchmark: 1,998 mined Turkish yes/no items across five probe types |
| `data/benchmarks/attrib_real_exclude.json` | Exclusion manifest (12 item IDs), applied by every analysis script. Scored set: 1,998 − 12 = **1,986 items** (787 gold YES / 1,199 gold NO) |
| `data/benchmarks/benchmark_gal.jsonl` | GAL bilingual (Turkish/Arabic) replication set: **360 items** |
| `data/benchmarks/freeze_manifest.txt`, `freeze_provenance.txt` | Freeze record (31 regenerated ATTRIB_FAKE titles) and freeze provenance |
| `data/results/*.jsonl` | Raw model outputs for every condition: **64,216 records** (base, Condition B, CoT, consistency, prompt paraphrases, GAL). Includes retained pre-retry rows and an aborted 282-row `gemma4:31b` run that is excluded from all analyses; the decomposition is Table S28 |
| `data/dia_scholars/scholars_full.csv` | Death years (hijri) for the 634 scholars referenced by MISATTRIB items, used for the era-gap analysis (Table S20) |
| `human_studies/` | Blank expert-study forms (`annotator2_form.csv`, `faketitle_blind_form.csv`, `ISNAD_revalidation_form_2026-06.csv`), the form instructions (`form_instructions.csv`), and the answer keys (`KEY_*.csv`, `*_key*.csv`) |
| `analysis/` | Six analysis scripts with fixed seeds (see *Reproduction*) |
| `MANIFEST.sha256` | SHA-256 checksums of all files |

All tabular files are plain UTF-8 CSV or JSON Lines.

## Reproduction

Python 3.11 (tested with 3.11.9):

```bash
pip install -r requirements.txt
python analysis/04e_expanded.py
python analysis/04g_reanalyze_nt.py
python analysis/04h_robustness_nt.py
python analysis/04i_new_tables_r2.py
python analysis/04j_prompt_stability_nt.py
python analysis/04k_consistency_nt.py
```

The scripts are API-free and write to `tables/regen_nt/` and `tables/main/`. With the pinned versions, the regenerated tables are byte-identical to those used in the article; the one exception is the Table 5 file, whose two x/n columns are written by `04i` into Table S31 rather than by `04e`.

| Script | Article tables |
|---|---|
| `04e_expanded.py` | Tables 3, 4, 5 (first six columns), the forced-choice column of Table 6; Tables S13–S14 |
| `04g_reanalyze_nt.py` | Tables S1–S12 |
| `04h_robustness_nt.py` | Tables S15–S26 |
| `04i_new_tables_r2.py` | Tables S28–S31 (Table S31 also gives the x/n columns of Table 5) |
| `04j_prompt_stability_nt.py` | Prompt-stability column of Table 6 |
| `04k_consistency_nt.py` | Consistency columns of Table 6 |

Tables 1 and 2 are descriptive. Two tables were computed once and frozen, and no generator script is included: the CoT column of Table 6 and Table S27. Their inputs (`data/results/benchmark_results_cot.jsonl` and the base results) are included.

Supplementary table numbers map to output files as follows (numbering diverges from the file names from S9 onward): S1–S8 `tablo1`–`tablo8`, S9–S12 `tablo10`–`tablo13`, S13 `e2`, S14 `e3`, S15–S26 `r1`–`r12`, S27 `r13`, S28 `r15`, S29 `r16`, S30 `r14`, S31 `r17`.

## Reading the data

- Every item carries a gold label. **ATTRIB_FAKE titles are synthetic**, and MISATTRIB, ANACHRONISM and false RELATION items deliberately attach false claims to real historical scholars. All subjects are pre-modern figures. Do not read any item as a factual statement without its gold label.
- `data/results` holds unedited model outputs (truncated as recorded by the runners). They contain hallucinations and errors and are released as evidence of model behaviour.
- The Turkish templates prefix every subject with "İslam âlimi" ("Islamic scholar") uniformly, including figures for whom the label does not fit; the templates are reproduced in the article's supplementary material.
- Residual gold noise in ATTRIB_REAL is quantified in Section 4.1.6 of the article and was not corrected after the freeze. The round-1 key (`round1_validation_key.csv`) records the pre-repair gold.
- **Item-level expert responses are not released, at the annotator's request.** Aggregate results are reported in the article. "Expert-2" and "annotator2" in the files denote the external expert acknowledged in the article.

## Not included

Model runners (the full prompts are reproduced in the article's supplementary material), figure scripts, precomputed tables, and the full prosopographic store derived from the *TDV İslâm Ansiklopedisi*. Some docstrings and provenance files refer to these review-time files (e.g. `runners/`, `tables/supplement/`, `04f_final_robustness.py`); they are not part of this archive. Codes such as R2.2 or R4.1 refer to reviewer comments from the journal's review.

## Licences and sources

- Code (`analysis/`): MIT, see `LICENSE`.
- Data (`data/`, `human_studies/`): CC BY 4.0, see `LICENSE-DATA.md`.
- The items record factual information (names, titles, dates) drawn from the *TDV İslâm Ansiklopedisi* (TDV İSAM) and from Brockelmann's *History of the Arabic Written Tradition* (Brill). No rights are claimed over those works. Raw model outputs remain subject to the respective providers' terms.

## Citation

Please cite the article and this archive; see `CITATION.cff`.
