# Tier-0 holdout call (Leonard) — 2026-09-11

**Artifacts**
- Manifest: `~/fourcastnet/data/tier0_ics/tier0_ic_manifest.json` (16 global ICs: 8 train / 4 val / 4 test)
- Metrics: `~/fourcastnet/runs/phase0/tier0_holdout/tier0_holdout_metrics.json`
- Maps: `~/fourcastnet/runs/phase0/tier0_holdout/tier0_bias_maps_train_only.nc`
- CSVs: `tier0_holdout_metrics_{val,test,train_self}.csv`
- Report: `~/fourcastnet/docs/tier0-holdout-report.md`

**Labels:** `claim_level=interim_era5` · `g1_claimable=false` · `provisional_years=true` · `year_split_frozen=true` (provisional until Manisha hard-locks)

## Call

| Layer | Result |
| --- | --- |
| **Holdout protocol** | **PASS** — train-only fit; val/test scored with frozen coeffs; global-IC honest |
| **Beat-this baseline (Tier-A)** | **FROZEN** — see below |
| **Tier-A engineering go** | **GO** — G0 claimable + holdout Tier-0 bar both exist |
| **Published skill / G1** | **NO-GO** — interim ERA5 + provisional years; needs Manisha hard-lock (+ IMDAA for G1) |

## Headline (t2m pooled, elev-binned linear, train-only coeffs)

| Split | raw RMSE | after lin | Role |
| --- | ---: | ---: | --- |
| train_self | 2.028 | 1.978 | fit sanity (old optimistic table) |
| **val 2022** | 1.949 | **1.949** | **primary beat-this** |
| test 2023–24 | 1.893 | 1.850 | secondary (do not cherry-pick) |

Honesty: on val the lin corrector does **not** improve pooled RMSE (high-elev [4500,9000]m slightly worsens). Tier-0 still publishes as the cheap baseline; the bar is essentially beat FCN3+Tier-0 on val ≈ beat raw FCN3 here.

## Frozen beat-this / Tier-A gates

### Primary (must pass)
- On **val 2022** ICs (ic09–ic12), same box/leads (+24/+72/+120), global-IC→crop protocol:
  - t2m pooled RMSE strictly **< 1.948661 K** (headline: < **1.949 K**; JSON `beat_this.val.rmse_after_lin`)
- Report per-lead and per elev-bin; high-elev band required in the table.

### Secondary (must not fail)
- On **test 2023–2024** (ic13–ic16): for **Tier-A PASS**, test t2m pooled RMSE strictly **< 1.850338 K** (Tier-0 test). Val-only win with test regression → **NO PASS**.

### Tier-A PASS (when results land)
1. Primary + secondary above.
2. **G0 adapter PASS** vs frozen verifying baseline (`G0_VERIFYING_CALL.md`): ≤+5% CRPS midlat @ +15 d on t2m/z500/t850; PSD/SSR floors.
3. Labels: if still ERA5 target → `claim_level=interim_era5`, `g1_claimable=false`.

### Tier-A FAIL / iterate
- Val RMSE ≥ 1.948661 K → fail beat-this (Tier-0 wins).
- Test RMSE ≥ 1.850338 K → fail (holdout generalization).
- G0 adapter hard FAIL or CRPS >+5% → discard weight-touching Tier-B scope; revisit Tier-A.

## Tier-A go-no-go

| Decision | Rule |
| --- | --- |
| **GO eng / first Tier-A runs** | **YES now** — prerequisites met (claimable G0 + holdout Tier-0 bar) |
| **GO claim beats Tier-0 in docs** | Only after PASS table on val+test with frozen seeds/ICs |
| **GO G1 / obs-aware** | **NO** until IMDAA (or stations) + Manisha year hard-lock |

## Year split (still provisional)

| Split | Years | ICs |
| --- | --- | --- |
| train | 2018–2021 | ic01–ic08 |
| val | 2022 | ic09–ic12 |
| test | 2023–2024 | ic13–ic16 |

Keep `provisional_years=true` on all Tier-A artifacts until Manisha hard-locks.

## Next eng (Howard)

1. Proceed **Tier-A** against this call (pathway Tier-A / RRCA-FD) with the frozen bars above.
2. Every Tier-A result JSON must echo: beat-this val **1.948661**, test **1.850338**, G0 verifying path, `provisional_years=true`.
3. Do **not** overwrite holdout Tier-0 or G0 verifying artifacts.
4. Ping Leonard with Tier-A metrics paths for PASS/FAIL call.

## Non-claims
- Not G1 / not IMDAA
- Not Manisha-hard-locked years
- Val lin gain ≈ 0 — Tier-0 is weak but published
- Small N (4+4) — bars are protocol locks, not high-precision climatology
