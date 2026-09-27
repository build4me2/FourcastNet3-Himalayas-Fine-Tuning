# Tier-A v1-diff (CorrDiff-lite) recipe (Leonard) — 2026-09-14

**Manisha greenlight (via Howard):** do **not** block on CDS — open next model now = **v1-diff / CorrDiff-lite**.  
**Living residual:** `runs/phase0/tier_a/v1_2b_thick2_train/` (INTERIM PASS CONFIRM).  
**Protocol:** thick-2 12/16/16 · year hard-lock · bars `HOLDOUT_THICK2_CALL.md`.  
**CDS:** Nepal ERA5 CDS crop still filling — leave download alone (check `ps` / `logs/era5_pull.log` for live PID).

Pathway alignment: `FINETUNE_METHOD_DESIGN.md` Phase 3 / `TIER_A_V1_SKETCH_CALL.md` deferred v1-diff.

## 1. Scope

| In | Out |
| --- | --- |
| Residual **diffusion** on Nepal crop 21×37 (t2m primary; u10m/v10m optional) | Full NVIDIA CorrDiff CONUS / globe |
| Conditioned on **frozen FCN3 crop** (+ elev + lead) | FCN3 weight FT / Tier-B |
| Optional conditioner: frozen living residual mean | IMDAA / G1 claims |
| Same thick-2 ICs / pairs / leads +24/+72/+120 | Blocking on CDS Nepal archive |
| Probabilistic samples as **secondary** table | Replacing thick-2 bars without Leonard freeze |

**One-sentence method:** Keep FCN3 + v1.2b residual as the frozen **mean path**; train a small elev/lead-conditioned **diffusion** to model the remaining residual vs ERA5 interim; decode with deterministic mean (or ens-mean) for gate RMSE.

## 2. What stays frozen (LOCKED)

| Component | Status |
| --- | --- |
| FCN3 checkpoint | **FROZEN** — never backprop |
| Living residual `v1_2b_thick2_train/best_residual.pt` | **FROZEN** mean path / conditioner — do not overwrite dir |
| G0 verifying / Tier-0 thick-2 metrics / all prior Tier-A dirs | **FROZEN** |
| Thick-2 bars + year hard-lock labels | **FROZEN** until Leonard amends |

## 3. Architecture (v1 — Spark-feasible)

| Field | Spec |
| --- | --- |
| Class | CorrDiff-lite **residual diffusion** (EDM-style or DDPM; document choice in config) |
| Spatial | Full Nepal 21×37 (no patch required at this size; patch OK if you prefer for code reuse) |
| Conditioner channels | FCN3 pred (t2m[,u,v]) + elev_norm + lead_norm (+ **frozen residual mean** as extra cond — **default ON**) |
| Target | ERA5 interim − FCN3 (or ERA5 − (FCN3+residual mean)); pick one and lock in config — **default: ERA5 − (FCN3 + frozen residual mean)** so diffusion learns leftover only |
| Capacity | Start small (≤~1–2M params); BF16; batch=1+accum on Spark |
| PhysicsNeMo | Use only if it fits 2D crop cleanly; else custom 2D residual diffuser (preferred if NeMo is 3D-only pain) |

## 4. Data / pairs

- Manifest / pairs: thick-2 `holdout_expand=v2` (same as living residual).
- Leads: +24/+72/+120 h.
- Train / val / test IC splits unchanged (12/16/16).
- `claim_level=interim_era5` · `g1_claimable=false` · `provisional_years=false`.

## 5. PASS / FAIL (LOCKED)

### Primary (deterministic decode — required for INTERIM PASS)

Decode = FCN3 + frozen residual mean + diffusion **mean** (0 noise / ens-mean of K samples; document K, default K=4).

| Gate | Rule |
| --- | --- |
| Val pooled t2m | **< 1.988588** (thick-2 bar) |
| Test pooled t2m | **< 1.918383** |
| Val +120 h | **≤ 2.178842** |
| **vs living residual** | Val pooled **strictly < 1.759883** (living best_val) **and** test pooled **≤ 1.7797 + 0.01** (no material test regress; prefer strict < 1.7797) |

If primary bars pass but **fail vs living residual** → **FAIL as product upgrade** (keep living residual; diffusion does not replace it). Label `diff_improves_residual=false`.

### Secondary (report only; not PASS blockers for v1)
- Per-lead / elev bands
- Test +120 h vs raw (honesty; still not a hard gate)
- Optional: CRPS / spread on diffusion members vs residual-only deterministic

### Kill / iterate
- Train unstable / NaNs / OOM → shrink model; do not touch FCN3.
- PASS bars but no lift vs living → **stop scaling diffusion**; report null; living residual stays canonical.
- No auto Tier-B.

## 6. Dirs (LOCKED)

| Item | Path |
| --- | --- |
| Out | `runs/phase0/tier_a/v1_diff/` |
| Config | `configs/tier_a_v1_diff.yaml` |
| Code | `code/tier_a/v1_diff/` |
| Results | `.../tier_a_v1_diff_results.json` + diffusion ckpt |
| Forbidden overwrite | `v1_2b_thick2_train/`, `v1_2b_thick2/`, `v1_2b/`, G0, tier0 thick-2 |

## 7. Eng order (Howard)

1. **CPU scaffold:** config echoing thick-2 bars + living residual path; dataset wiring; dry-run forward (1 step) — **no `--train`**.
2. Ping Leonard if architecture choice (EDM vs DDPM / NeMo vs custom) is ambiguous; else proceed.
3. **GPU train** only after dry-run OK.
4. Ping Leonard with results JSON for PASS/FAIL call.
5. **Nepal ERA5 CDS crop still filling — leave download alone (check `ps` / `logs/era5_pull.log` for live PID).**

## 8. Non-goals
- Waiting on full Nepal CDS archive  
- Claiming CorrDiff NVIDIA parity  
- Replacing living residual without beating it  
- Diffusion into FCN3 weights  
