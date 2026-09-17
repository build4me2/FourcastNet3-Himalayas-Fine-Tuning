# FINAL eval suite — CPU scoring scaffold

**Recipe:** `docs/research/FINAL_EVAL_SUITE_RECIPE.md`  
**Call:** `docs/research/FINAL_EVAL_SUITE_CALL.md`  
**Lit:** `docs/research/EVAL_BENCHMARKS_AND_FINAL_SUITE.md`  
**Output:** `runs/phase0/final_eval/final_baselines.json`  
**Status:** Scaffold only. **No train.** `g1_claimable=false`.

Living interim `runs/phase0/tier_a/v1_3_joint/` stays frozen (`interim_era5`). FINAL is a **new protocol**, not a quiet rewrite of thick-2.

## What this package does

| Script | Role |
| --- | --- |
| `score_final_baselines.py` | CPU scaffold to score **raw FCN3**, **Tier-0**, **living v1_3_joint** on FINAL protocol |
| `thick2_legacy_bridge.py` | Stub: re-score ckpt on thick-2 12/16/16 for continuity table (report only) |

Config: `configs/final_eval_baselines.yaml` (placeholder IC lists / year slots TBD).

## How to run (once PROTOCOL years/ICs land)

```bash
cd ~/fourcastnet
# Always safe (no GPU, no CDS, no invented floats):
python code/final_eval/score_final_baselines.py --help
python code/final_eval/score_final_baselines.py --dry-run
python code/final_eval/score_final_baselines.py --write-schema
python code/final_eval/score_final_baselines.py --validate-only

# Thick-2 bridge stub:
python code/final_eval/thick2_legacy_bridge.py --dry-run
python code/final_eval/thick2_legacy_bridge.py --write-stub
```

**Real scores** require:

1. Archive audit PASS (`data/era5/coverage_audit.json`)
2. Leonard `docs/research/FINAL_EVAL_PROTOCOL.md` with concrete years + hashed IC lists
3. Filled `ics.*_ids` + `year_split` in `configs/final_eval_baselines.yaml`
4. Then: `python code/final_eval/score_final_baselines.py` (eval only — still no train)

Until then the script **exits cleanly** explaining PROTOCOL required and refreshes the null-metric schema JSON. **Do not invent floats.**

## Locked defs (do not change quietly)

| Item | Spec |
| --- | --- |
| Box | 26–31°N / 80–89°E |
| Gate leads | +24 / +72 / +120 h |
| Report leads | +24 / +48 / +72 / +96 / +120 h |
| Wind-vector | \(\sqrt{\mathrm{mean}((u_{\mathrm{err}}^2+v_{\mathrm{err}}^2)/2)}\) grid-pooled → lead-mean over gate leads |
| Thick-2 bridge | legacy continuity flag / stub scorer |

Helpers to reuse when wiring real score: `code/tier_a/v1_3_joint/score_living_wind_baseline.py` (`accumulate` / `summarize`).

## Eng order (Howard) — no train yet

1. CDS continues (leave download PID alone) + coverage audit  
2. Protocol years/ICs  
3. Score `final_baselines.json`  
4. Leonard `FINAL_EVAL_BARS_CALL.md` freezes floats  
5. **Only then** Manisha may unlock FINAL residual train  

## Non-claims

Not global WB2 SOTA · not ops replacement · not CorrDiff · not precip · not stations · interim ≠ FINAL/G1.
