# Tier-0 real call (Leonard) — 2026-09-11

**Artifacts**
- `~/fourcastnet/runs/phase0/tier0/tier0_bias_maps.nc`
- `~/fourcastnet/runs/phase0/tier0/tier0_metrics.csv`
- `~/fourcastnet/runs/phase0/tier0/tier0_metrics.json`
- pairs under `~/fourcastnet/runs/phase0/tier0/pairs/`

**Labels (must stick):** `claim_level=interim_era5` · `target=ERA5_interim` · `g1_claimable=false` · `year_split_frozen=false`

## Call

| Layer | Result |
| --- | --- |
| **Tier-0 plumbing (recipe §2.5 Success)** | **PASS** — maps + tables + elev-banded t2m bias delivered; global-IC → crop vs ERA5 interim honest |
| **Tier-0 as beat-this skill baseline for Tier-A** | **NOT YET** — all 8 ICs are 2018–2021 (in-sample fit); no val/test holdout; ERA5 target ≠ obs-aware G1 |
| **G1 / obs-aware** | **BLOCKED** — needs IMDAA (or stations) + frozen year split |
| **G0 claimable (CRPS/SSR/PSD)** | **STILL NOT** — verifying hook stubbed; needs streaming re-roll |

## Headline numbers (t2m, elev-binned linear)

| Scope | RMSE raw | RMSE after lin | Δ |
| --- | --- | --- | --- |
| Pooled +24/+72/+120 h | **2.028 K** | **1.978 K** | −0.050 K (−2.5%) |
| +24 h | 1.759 | 1.575 | −0.184 |
| +72 h | 2.136 | 1.937 | −0.199 |
| +120 h | 2.164 | 2.095 | −0.069 |

**High-elev [4500,9000]m @ +24 h:** raw bias \(a_\mathrm{const}\approx -1.16\,\mathrm{K}\) (FCN3 warm vs ERA5); RMSE 2.229 → 1.899 K after lin. Mid/low elev gains are smaller; most of the pooled win is high-elev bias removal.

Winds/tcwv tables exist; headline Tier-0 for “Tier-A must beat” remains **t2m elev-binned linear** (per recipe).

## Provisional numeric thresholds (freeze after holdout re-score)

Until years are frozen and Tier-0 is **re-fit on train only / scored on val+test**, treat these as **interim plumbing thresholds** only (not published skill):

1. **Beat-this (Tier-A vs this interim table):** on the **same** 8 ICs + leads + box, Tier-A mean t2m RMSE must be **strictly < 1.978 K** pooled (and report per-lead + per elev-bin). Prefer also beating high-elev [4500,9000]m lin RMSE at +24/+72/+120.
2. **Do not** use the −2.5% pooled Δ as a success bar for adapters — that Δ is in-sample bias correction, not out-of-sample skill.
3. After year freeze + holdout re-score, Leonard will replace (1) with val/test numbers and mark `year_split_frozen=true`.

## Year freeze recommendation (needs Manisha lock)

**Propose now (Howard’s suggestion = Leonard’s default):**

| Split | Years |
| --- | --- |
| **Train** | 2018–2021 |
| **Val** | 2022 |
| **Test** | 2023–2024 |

**Why freeze now:** current Tier-0 fit used every staged IC (all train-era); without a holdout we cannot honest-gate Tier-A. Staging ≥4 val + ≥4 test global ICs can proceed in parallel with G0 verifying work.

Box stays **26–31°N, 80–89°E** unless Manisha changes it (still formally confirmable).

## Next eng (Howard) — priority order

1. **Now:** Wire **real G0 verifying metrics** (CRPS / SSR / PSD vs ERA5 @ +15 d) — streaming re-roll as you noted (`g0_verifying_era5.py`). Keep stub plumbing reference separate. Claimable G0 still blocks Tier-A/B compare.
2. **In parallel (once Manisha locks years, or on provisional above):** Stage **val 2022 + test 2023–2024** global ERA5 ICs; **re-fit Tier-0 on train ICs only**; score val/test; rewrite `tier0_metrics.*` with `year_split_frozen=true` and holdout RMSEs.
3. **Do not** start Tier-A/B training gates until (1) claimable G0 base exists **and** (2) holdout Tier-0 beat-this numbers exist.

## Non-claims

- Not G1 / not IMDAA  
- Not out-of-sample Tier-0  
- Not a license to train Tier-A yet  
- Preview G0 finite PASS unchanged; CRPS still stubbed  
