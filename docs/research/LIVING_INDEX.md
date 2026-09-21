# Living index — FCN3 Nepal residual (Spark)

**Updated:** 2026-09-21 PT · **Owner:** Manisha · **Eng:** Howard  
**Living residual (LOCKED):** `runs/phase0/tier_a/v1_3_joint/`  
**Headlines (16-IC):** t2m val **1.770** / test **1.775** / +120h **1.982** · WV **0.69560 / 0.73877**  
**Prior living (historical frozen):** `runs/phase0/tier_a/v1_2b_thick2_train/`  
**CDS:** Nepal ERA5 CDS crop still filling — leave download alone (`ps` / `logs/era5_pull.log`).

| Role | Path |
| --- | --- |
| **Living path (weights)** | `runs/phase0/tier_a/v1_3_joint/best_residual.pt` |
| **Promote call** | `docs/research/TIER_A_V1_3_JOINT_CALL.md` (+ recipe) |
| **Paper draft** | `docs/research/PAPER_DRAFT_METHODS_RESULTS.md` |
| **FINAL suite** | `docs/research/FINAL_EVAL_SUITE_CALL.md` · `FINAL_EVAL_SUITE_RECIPE.md` · `EVAL_BENCHMARKS_AND_FINAL_SUITE.md` |
| **Status — what works / failed** | `docs/research/STATUS_WHAT_WORKS_WHAT_FAILED.md` |
| **Status & next** | `docs/research/FCN_STATUS_AND_NEXT.md` |
| **Coverage audit** | `docs/research/ERA5_COVERAGE_AUDIT.md` · `data/era5/coverage_audit.json` |
| **Project scope / history** | `docs/research/PROJECT_SCOPE_AND_HISTORY.md` |
| **Figures stub** | `docs/figures/README.md` |
| **v1.2 FAIL / v1-diff NULL** | `tier_a/v1_2/` · `TIER_A_V1_DIFF_CALL.md` |

Canonical copies live under `docs/research/`; mirrored pairs under `docs/calls/` when present. Prefer research → `cp` to calls.
