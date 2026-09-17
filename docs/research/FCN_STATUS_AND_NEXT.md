# FCN3 Phase 0 — status & next (Leonard) — 2026-09-15

**Manisha:** work on FourCastNet (Howard = FCN-only; TTA parked).

## Frozen state (do not overwrite)

| Layer | Status | Path / bar |
| --- | --- | --- |
| G0 verifying | **PASS claimable** | `g0_verifying_results.json`; adapters ≤+5% CRPS; PSD hard &lt;0.25 |
| Tier-0 thin | historical | val 1.948661 / test 1.850338 |
| Tier-0 expand-v1 | historical | val 1.987014 / test 1.897298 / +120h ≤ 2.131516 |
| Tier-0 **thick-2** | **living beat-this** | val **&lt; 1.988588** / test **&lt; 1.918383** / +120h ≤ **2.178842** |
| Tier-A v0 | best **test pooled** champ (1.733 thin) | `tier_a/v0/` |
| Tier-A v1.1 | first +120h-compliant PASS (thin) | `tier_a/v1_1/` |
| Tier-A v1.1 expand ZS | INTERIM PASS (expand-v1 hist.) | val 1.808 / test 1.826 / +120h 1.998 |
| Tier-A v1.2b | INTERIM PASS (expand train) → **historical** | `tier_a/v1_2b/` · still frozen |
| Tier-A v1.2b **thick-2 ZS** | INTERIM PASS → **historical** (ZS demoted) | val 1.829 / test 1.897 / +120h 2.015 · `tier_a/v1_2b_thick2/` |
| Tier-A v1.2b **thick-2 retrain** | INTERIM PASS → **historical** (prior living; still frozen) | val **1.760** / test **1.780** / +120h **1.962** · `tier_a/v1_2b_thick2_train/` |
| Tier-A **v1.3 joint** | **INTERIM PASS CONFIRM YES · PROMOTE · living residual** | val **1.770** / test **1.775** (16-IC) / +120h **1.982** · wind-vector val **0.69560** / test **0.73877** · `tier_a/v1_3_joint/` |
| Tier-A **v1-diff** | **NULL — do not promote** | `tier_a/v1_diff/` · keep as null artifact; no material lift vs prior living |
| Years | **HARD-LOCKED** 2018–21 / 2022 / 2023–24 | `provisional_years=false`; see YEAR_HARD_LOCK.md |
| Labels | `interim_era5`; `g1_claimable=false` | — |

Calls: `TIER_A_V1_3_JOINT_CALL.md` (living promote), `TIER_A_V1_DIFF_CALL.md` (NULL), `TIER_A_V1_2B_THICK2_TRAIN_CALL.md` (prior living → historical), `HOLDOUT_THICK2_CALL.md` (bars), `HOLDOUT_EXPAND_CALL.md` (historical), `TIER_A_V1_2B_CALL.md`, `G0_VERIFYING_CALL.md`.

## Living residual (LOCKED)

| Role | Path |
| --- | --- |
| **Living (canonical)** | `runs/phase0/tier_a/v1_3_joint/` (`best_residual.pt`) |
| Prior living (historical) | `runs/phase0/tier_a/v1_2b_thick2_train/` · **still frozen** · md5 unchanged through v1.3 train |
| v1-diff (null artifact) | `runs/phase0/tier_a/v1_diff/` · **do not delete** · do not promote |
| Prior expand-train v1.2b | `v1_2b/` → historical (still frozen) |
| Thick-2 ZS | `v1_2b_thick2/` → historical (ZS demoted) |
| v1.2 FAIL | `v1_2/` → keep |

Do **not** overwrite living residual (`v1_3_joint/`), prior living (`v1_2b_thick2_train/`), or v1_diff results.

## Recommended next (default)

**Idle pending Manisha.** Living residual = `v1_3_joint/`.  
**v1-diff = NULL** — stop leftover-target diffusion scaling; do not promote.  
Deferred unless assigned: parallel-mean diff / multi-seed / write-up.  
Keep CDS **611595** alone. No GPU.

### Living bars (echo in future result JSON)
- val t2m pooled RMSE lin strictly **&lt; 1.988588**
- test t2m pooled RMSE lin strictly **&lt; 1.918383**
- val +120h t2m RMSE **≤ 2.178842** (thick-2 val raw)

### v1.3 joint promote (headline — 16-IC locked claims)
- t2m: val **1.770** / test **1.775** / +120h **1.982** · eligible **46** · `composite_eligible`
- wind-vector lead-mean: val **0.69560** / test **0.73877** (beat prior living + raw)
- protect-B: val Δ **+0.010** / test Δ **−0.005** (inside ±0.02 K)

### v1-diff null (headline)
- one_step 0-noise: val ~1.7598 / test ~1.7799 / +120h ~1.962 (≈ prior living; Δ ~1e-5)
- ens K=4: **worse** (val ~1.937 / test ~1.965)
- `diff_improves_residual=false`

## Alternatives (if Manisha overrides)

| Option | When |
| --- | --- |
| **Different Phase-3 fork** | Explicit assign — e.g. parallel-mean diff, multi-seed, write-up — **not** more leftover-target scaling |
| **More ICs (test SON)** | Recipe-perfect seasons without reopening architecture; keep protocol 16-IC claims |
| **Winds/elev-band deep dive** | Science writeup without new train |
| **Pause eng** | Status-only; Howard idle on FCN (current default) |

## Non-goals until ordered
- TTA harness (Howard not assigned)
- FCN3 weight FT / Tier-B
- IMDAA / G1 claims
- Scaling leftover-target diffusion further (STOP)
- Overwriting frozen artifacts (incl. living `v1_3_joint/`, prior `v1_2b_thick2_train/`, null `v1_diff/`, `v1_2b/`, `v1_2b_thick2/`)
- Letting `ic45` leak into protocol-locked 16-IC claim echoes

## Next eng (Howard)
1. Mirror Leonard `TIER_A_V1_3_JOINT_CALL.md` · status: living = `v1_3_joint/`; prior thick2_train = historical — **done**.
2. Patch results `test_ids` echo to 16-IC; sidecar claim JSON — **done**.
3. No GPU; leave ERA5 CDS PID 611595 alone.
4. Idle pending Manisha (deferred forks only if assigned).
