# Tier-A v0 call (Leonard) — 2026-09-11

**Artifacts**
- `~/fourcastnet/runs/phase0/tier_a/v0/tier_a_v0_results.json`
- `~/fourcastnet/runs/phase0/tier_a/v0/best_residual.pt`
- Smoke note: `docs/research/TIER_A_V0_SMOKE_NOTE.md`
- Bars from: `docs/research/TIER0_HOLDOUT_CALL.md`

**Method:** RRCA-FD Plan A — TinyElevResidualUNet on frozen FCN3 crop → ERA5 interim (300-ep best ckpt, ~35k params).  
**Labels:** `claim_level=interim_era5` · `g1_claimable=false` · `provisional_years=true` · `fcn3_weights=FROZEN`

## Call

| Gate | Result |
| --- | --- |
| **Val beat-this** (&lt; 1.948661 K) | **PASS** — 1.892 K |
| **Test beat-this** (&lt; 1.850338 K) | **PASS** — 1.733 K |
| **G0 adapter ≤5% CRPS** | **N/A** — FCN3 frozen; base G0 verifying PASS still stands; no weight-touch to re-score |
| **Tier-A v0 INTERIM PASS** | **YES** — val AND test strict beat-this |
| **Published skill / G1 / scale-as-done** | **NO** — interim ERA5, provisional years, tiny residual, small-N |

## Numbers (t2m pooled)

| Split | raw FCN3 | Tier-A | vs Tier-0 bar |
| --- | ---: | ---: | --- |
| val | 1.949 | **1.892** | &lt; 1.949 ✓ |
| test | 1.893 | **1.733** | &lt; 1.850 ✓ |

### WARN — lead imbalance (val)

| Lead | raw | Tier-A |
| --- | ---: | ---: |
| +24 h | 1.691 | 1.561 ✓ |
| +72 h | 1.901 | 1.620 ✓ |
| **+120 h** | **2.220** | **2.382** ✗ regress |

Pooled PASS is carried by short leads. Do **not** treat v0 as lead-uniform skill.

Late-epoch val RMSE drifted to ~2.36–2.39 while best ckpt is 1.892 — early-stop is doing the work; keep best-ckpt protocol.

## What this does / does not authorize

**Does**
- Pipeline proof that Tier-A residual training + holdout scoring works against frozen bars
- License to iterate v0.1 (lead-balanced / multi-lead loss, elev emphasis) or scaffold real RRCA-FD / CorrDiff Tier-A
- Keep `best_residual.pt` as the v0 reference ckpt

**Does not**
- Obs-aware or IMDAA claims
- FCN3 fine-tune / Tier-B weight updates (would re-trigger G0 adapter gate)
- “Beats CorrDiff” or paper-ready regional skill
- Overwrite G0 verifying or Tier-0 holdout artifacts (still forbidden)

## Next eng (Howard)

1. **v0.1 iterate (preferred before bigger models):** multi-lead loss or lead-weighted objective so +120 h val does not regress; re-score same IC splits; same beat-this bars.
2. Optional: elev-band auxiliary loss on [4500,9000]m (still weak on val high-elev vs raw in places).
3. When moving past tiny UNet toward pathway RRCA-FD / CorrDiff: new variant id (`v1_…`); keep v0 JSON frozen; if any FCN3 weight touch → full G0 adapter re-roll vs verifying baseline.
4. Ping Leonard with next metrics paths for call.

## Non-claims
- Not G1 / not IMDAA  
- Not CorrDiff  
- Not FCN3 FT  
- Not lead-uniform  
- provisional_years=true  
