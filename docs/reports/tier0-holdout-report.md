# Tier-0 holdout report — provisional years

**Date:** 2026-09-11 ~20:15 PT
**Host:** Spark `chandmanisha00@100.121.160.49` (spark-61dd)
**Priority:** Leonard #2 after G0 verifying DONE (G0_VERIFYING_CALL.md PASS claimable YES)
**No Tier-A** this turn.

## Verdict

| Layer | Result |
| --- | --- |
| Global val+test ICs staged | **DONE** — 4 val (2022) + 4 test (2023–2024) + 8 train reused |
| Train-only elev-binned fit | **DONE** — ic01–ic08 only |
| Val / test holdout scores | **DONE** — tables on disk |
| Year labeling | **provisional_years=true**; year_split_frozen=true (provisional; Manisha hard-lock TBD) |
| claim_level | `interim_era5`; `g1_claimable=false` |
| ERA5 regional PID 611595 | **Untouched** |

## Leonard-pingable paths

| Artifact | Path |
| --- | --- |
| Split IC manifest | `~/fourcastnet/data/tier0_ics/tier0_ic_manifest.json` |
| Global ICs (16) | `~/fourcastnet/data/tier0_ics/ic_*_global.npy` |
| Holdout metrics JSON | `~/fourcastnet/runs/phase0/tier0_holdout/tier0_holdout_metrics.json` |
| Train-only bias maps | `~/fourcastnet/runs/phase0/tier0_holdout/tier0_bias_maps_train_only.nc` |
| Val metrics CSV | `~/fourcastnet/runs/phase0/tier0_holdout/tier0_holdout_metrics_val.csv` |
| Test metrics CSV | `~/fourcastnet/runs/phase0/tier0_holdout/tier0_holdout_metrics_test.csv` |
| Train fit CSV | `~/fourcastnet/runs/phase0/tier0_holdout/tier0_metrics_train_fit.csv` |
| Pairs (fc+tgt) | `~/fourcastnet/runs/phase0/tier0_holdout/pairs/` |
| Holdout IC config | `~/fourcastnet/configs/tier0_holdout_ics.yaml` |
| Tier-0 locks config | `~/fourcastnet/configs/tier0_bias.yaml` |
| Code | `code/phase0/tier0_holdout.py`, `stage_tier0_holdout_ics.py` |

## Provisional year split

| Split | Years | ICs |
| --- | --- | --- |
| train | 2018–2021 | ic01–ic08 (reuse G0) |
| val | 2022 | ic09–ic12 |
| test | 2023–2024 | ic13–ic16 |

All ICs are **global** ERA5 ARCO (72×721×1440). Nepal crop ≠ IC.

## Headline — t2m pooled RMSE (K)

Train-only elevation-binned linear bias applied to holdout.

| Split | n_IC | raw | after lin | Role |
| --- | ---: | ---: | ---: | --- |
| train_self | 8 | 2.028 | 1.978 | fit sanity (= prior table) |
| **val 2022** | 4 | 1.949 | **1.949** | **primary beat-this** |
| test 2023–24 | 4 | 1.893 | 1.850 | secondary |

### Val t2m by lead (lin)

| Lead | raw | lin |
| --- | ---: | ---: |
| +24 h | 1.691 | 1.693 |
| +72 h | 1.901 | 1.780 |
| +120 h | 2.220 | 2.033 |

### Val t2m elevation bands (pooled)

| Band | raw | lin | mean bias (truth−pred) |
| --- | ---: | ---: | ---: |
| [0,1500) m | 1.488 | 1.458 | +0.750 |
| [1500,3000) m | 1.814 | 1.737 | +0.618 |
| [3000,4500) m | 2.013 | 2.021 | +0.374 |
| [4500,9000] m | 2.322 | 2.349 | −0.217 |

## Beat-this bar (updated)

- **Previous (optimistic):** t2m lin RMSE **1.978 K** (fit+score on train-era ICs only)
- **New primary:** val 2022 t2m lin RMSE **1.949 K**
- **Secondary:** test lin **1.850 K** (do not cherry-pick over val)
- Tier-A must beat **val 1.949 K** on this protocol (or Manisha-locked years later)

## Method

1. Stage/reuse global ARCO ERA5 ICs with train/val/test labels.
2. Path B crop: frozen FCN3 from global IC → Nepal 21×37 at +24/+72/+120 h (M=2, seed 333/334).
3. Targets: ARCO ERA5 crop (not CDS).
4. Fit elev-binned linear on **train only**.
5. Apply frozen train coefficients to val/test; report RMSE / bias / elev bands.

## Honesty / non-claims

- `provisional_years=true` — Manisha has **not** hard-locked years.
- `claim_level=interim_era5`; **cannot claim G1** (no IMDAA/obs).
- Small holdout N (4 val + 4 test). Val band biases shift vs train → pooled lin gain ≈ 0 on val.
- G0 PASS claimable remains from verifying path (separate); this is Tier-0 holdout only.
- No Tier-A run.

## Commands (from ~/fourcastnet)

```bash
~/fcn3-venv/bin/python code/phase0/stage_tier0_holdout_ics.py --splits train
~/fcn3-venv/bin/python code/phase0/stage_tier0_holdout_ics.py --splits val,test
~/fcn3-venv/bin/python code/phase0/tier0_era5_targets.py --manifest data/tier0_ics/tier0_ic_manifest.json --splits val,test --pairs-dir runs/phase0/tier0_holdout/pairs
PYTHONUNBUFFERED=1 ~/fcn3-venv/bin/python -u code/phase0/tier0_crop_rollout.py --manifest data/tier0_ics/tier0_ic_manifest.json --splits val,test --pairs-dir runs/phase0/tier0_holdout/pairs --allow-gpu
~/fcn3-venv/bin/python code/phase0/tier0_holdout.py
```
