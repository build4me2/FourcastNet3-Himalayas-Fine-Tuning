# Status pointer

## 2026-09-27 13:18 PT — FINAL residual v0 thick-2 bridge FILLED (report-only)
- Scored `final_residual_v0/best_residual.pt` on HOLDOUT_THICK2 via `thick2_legacy_bridge.py`.
- Wrote `runs/phase0/final_eval/final_residual_v0/thick2_bridge_results.json`.
- Did **not** flip g1_claimable; did **not** overwrite v1_3_joint or residual_v0 weights.
- Thick-2: residual_v0 val/test t2m 1.883/1.875 vs living 1.770/1.775; raw ~2.017/2.014.

Canonical append-only log: [`../PROGRESS.md`](../PROGRESS.md).

**2026-09-25 evening PT:** Living residual (FINAL protocol) = `runs/phase0/final_eval/final_residual_v0/` · PASS A∧B∧C · headlines val/test t2m **1.8199/1.8564** · +120h **2.0104** · WV **0.8004/0.8250** · **`g1_claimable=false`** · interim `v1_3_joint/` preserved historical (md5 `c81a5a4c…`). See `FINAL_RESIDUAL_V0_CALL.md`.

**2026-09-13:** Unlock **B** year hard-lock applied (`provisional_years=false`, `year_split_frozen=true` HARD). See `docs/research/YEAR_HARD_LOCK.md` / `FCN_NEXT_UNLOCK.md`.
