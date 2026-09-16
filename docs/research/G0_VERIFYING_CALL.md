# G0 verifying call (Leonard) — 2026-09-11

**Artifacts**
- `~/fourcastnet/runs/phase0/g0/g0_verifying_results.json` (`crps_ssr_psd=real`, path A)
- sidecar `~/fourcastnet/runs/phase0/g0/g0_base_results_verifying_sidecar.json`
- report `~/fourcastnet/docs/g0-verifying-report.md`
- stub plumbing preserved: `g0_base_results.json`

**Protocol:** 8/8 ICs × 4 mem × 60 steps; score leads +120/+240/+360 h; target ERA5 ARCO verifying; peak CUDA ~53 GB.

## Call

| Layer | Result |
| --- | --- |
| **G0 PASS (base, claimable)** | **YES** — path A complete; all finite; real CRPS/SSR/PSD; SSR/PSD sane on ≥8 ICs @ +15 d (recipe §1.5) |
| **Adapter ≤~5% CRPS rule** | **Armed** — compare adapters to **this** verifying JSON (not the stub) |
| **Tighten hard PSD fail for winds/z500?** | **NO** — would fail frozen FCN3 itself. Keep hard floor; add WARN / adapter floors below |

## Headline (+15 d midlat, 8-IC mean)

| Var | CRPS | SSR | PSD ratio |
| --- | ---: | ---: | ---: |
| t2m | 1.277 | 0.666 | 0.712 |
| z500 | 440.1 | 0.732 | 0.438 |
| t850 | 1.971 | 0.787 | 0.515 |
| u10m | 2.312 | 0.727 | 0.380 |
| v10m | 2.345 | 0.752 | 0.406 |
| tcwv | 2.983 | 0.776 | 0.451 |

SSR is underdispersed (M=4) but **not** collapsed. PSD on winds/z500/tcwv at +15 d sits in a **blur WARN** band (~0.38–0.45), while +5 d is healthy (~0.80–0.93) — progressive lead degradation, not inference break.

## Threshold freeze (base + adapters)

### Hard FAIL (base or adapter) — spectral / integrity nonsense
- Any IC: Inf/NaN, OOM, HMM identity break
- PSD synoptic ratio midlat **< 0.25** or **> 4.0** on any scored var @ +15 d
- SSR midlat **< 0.15** (near-zero spread collapse) or obviously exploding RMSE with NaN ranks

### WARN (document; do not fail base)
- PSD midlat @ +15 d in **[0.25, 0.45)** on u10m / v10m / z500 / tcwv → **blur WARN** (current base is here)
- SSR midlat in **[0.15, 0.50)** → underdispersion WARN

### Adapter G0 PASS (vs this baseline)
1. Midlat CRPS @ +15 d on **t2m, z500, t850**: ≤ **+5%** relative vs baseline means above (per var).
2. No hard FAIL from table above.
3. PSD midlat @ +15 d must not fall **below 0.90 × baseline** for that var (floors from this run: u10m **0.342**, v10m **0.365**, z500 **0.394**, tcwv **0.406**, t2m **0.641**, t850 **0.464**). Crossing into hard <0.25 still fails.
4. SSR midlat @ +15 d must not fall **below 0.90 × baseline** for t2m/z500/t850 (floors ≈ 0.600 / 0.659 / 0.708).

**Primary skill vars for the 5% rule:** t2m, z500, t850. Winds/tcwv are integrity / spectrum monitors.

## Next eng (Howard)

1. **Now:** Holdout Tier-0 on provisional years (train **2018–2021** / val **2022** / test **2023–2024**) — re-fit train-only, score val+test, set `year_split_frozen=true` when Manisha locks (until then keep `provisional` flag).
2. Freeze verifying JSON + this call as the **adapter G0 reference**; do not overwrite.
3. Tier-A/B still blocked until holdout Tier-0 beat-this numbers exist (interim bar 1.978 K remains plumbing-only).

## Non-claims
- Not a quality endorsement of FCN3 spectra at +15 d (blur WARN stands)
- Not G1 / not regional skill
- Stub `g0_base_results.json` is plumbing-only
