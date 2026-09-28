# Living residual — winds / elev-band table

**Source CSV:** `runs/phase0/tier_a/v1_2b_thick2_train/tables/living_residual_winds_elev_lead.csv`
**Wind-vector baseline:** `runs/phase0/tier_a/v1_3_joint/living_wind_baseline.json`
**Model:** Tier-A v1.2b thick-2 retrain (living residual). **No retrain** for this table.
**Elevation bins (m):** [0, 1500, 3000, 4500, 9000]
**Living ckpt MD5:** `aa0ed74a9307f5a284cadc65e3671be4` (unchanged by eval).

> **Channels:** residual UNet outputs **t2m, u10m, v10m**. Historical train metrics scored **t2m only**; this table is a frozen post-hoc eval on all three. Training used `channel_weights=[2.0, 0.5, 0.5]` (winds under-weighted vs t2m).

## Wind-vector headline (v1.3 promote bars)

Definition: per-lead `sqrt(mean((u_err^2+v_err^2)/2))` over valid elev cells; headline = **lead-mean** over {24,72,120}.

| Split | Raw FCN3 wind-vector | Living residual wind-vector |
| --- | ---: | ---: |
| val | 0.715178 | **0.699095** |
| test | 0.762763 | **0.744928** |

Promote bars (strict < living): val **0.699095** · test **0.744928**.

## Headline pooled (grid-pooled all leads)

| Split | Var | RMSE raw | RMSE living residual | n |
| --- | --- | ---: | ---: | ---: |
| val | t2m | 2.016630 | 1.759883 | 37296 |
| val | u10m | 0.737046 | 0.720575 | 37296 |
| val | v10m | 0.696005 | 0.679842 | 37296 |
| test | t2m | 2.014341 | 1.779660 | 37296 |
| test | u10m | 0.833946 | 0.808613 | 37296 |
| test | v10m | 0.699712 | 0.687991 | 37296 |
| train_self | t2m | 2.023813 | 1.721927 | 27972 |
| train_self | u10m | 0.739760 | 0.710957 | 27972 |
| train_self | v10m | 0.665563 | 0.653731 | 27972 |

Living t2m bars (from train results JSON; match val/test pooled above): val **1.759883** / test **1.779660** / val +120h **1.962487**.

> Note: `train_self` here is ens-mean ICs only (n≈28k). Canonical living train_self t2m used member-expanded loader (n=83916) — prefer published JSON for train_self t2m.

## By lead (h)

| Split | Var | Lead | RMSE raw | RMSE living | n |
| --- | --- | ---: | ---: | ---: | ---: |
| val | t2m | 24 | 1.696995 | 1.436099 | 12432 |
| val | t2m | 72 | 2.138514 | 1.837887 | 12432 |
| val | t2m | 120 | 2.178842 | 1.962487 | 12432 |
| val | u10m | 24 | 0.663779 | 0.673691 | 12432 |
| val | u10m | 72 | 0.644284 | 0.617569 | 12432 |
| val | u10m | 120 | 0.879777 | 0.849962 | 12432 |
| val | v10m | 24 | 0.749766 | 0.728573 | 12432 |
| val | v10m | 72 | 0.674712 | 0.668300 | 12432 |
| val | v10m | 120 | 0.660214 | 0.639617 | 12432 |
| test | t2m | 24 | 1.945683 | 1.610051 | 12432 |
| test | t2m | 72 | 2.242590 | 1.878807 | 12432 |
| test | t2m | 120 | 1.832432 | 1.838313 | 12432 |
| test | u10m | 24 | 0.616145 | 0.623824 | 12432 |
| test | u10m | 72 | 0.895062 | 0.876798 | 12432 |
| test | u10m | 120 | 0.951643 | 0.896457 | 12432 |
| test | v10m | 24 | 0.627463 | 0.608284 | 12432 |
| test | v10m | 72 | 0.689139 | 0.682152 | 12432 |
| test | v10m | 120 | 0.774706 | 0.764626 | 12432 |

## By elevation band

| Split | Var | Bin | Label | RMSE raw | RMSE living | n |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| val | t2m | 0 | [0,1500)m | 1.399552 | 1.252648 | 15456 |
| val | t2m | 1 | [1500,3000)m | 1.822012 | 1.663768 | 2928 |
| val | t2m | 2 | [3000,4500)m | 2.309077 | 2.065948 | 3072 |
| val | t2m | 3 | [4500,9000]m | 2.452855 | 2.102841 | 15840 |
| val | u10m | 0 | [0,1500)m | 1.008966 | 0.984176 | 15456 |
| val | u10m | 1 | [1500,3000)m | 0.372950 | 0.359692 | 2928 |
| val | u10m | 2 | [3000,4500)m | 0.416475 | 0.395066 | 3072 |
| val | u10m | 3 | [4500,9000]m | 0.475807 | 0.472485 | 15840 |
| val | v10m | 0 | [0,1500)m | 0.853890 | 0.845515 | 15456 |
| val | v10m | 1 | [1500,3000)m | 0.508018 | 0.462622 | 2928 |
| val | v10m | 2 | [3000,4500)m | 0.528240 | 0.466462 | 3072 |
| val | v10m | 3 | [4500,9000]m | 0.572120 | 0.555797 | 15840 |
| test | t2m | 0 | [0,1500)m | 1.300758 | 1.190187 | 15456 |
| test | t2m | 1 | [1500,3000)m | 1.672311 | 1.605322 | 2928 |
| test | t2m | 2 | [3000,4500)m | 2.170708 | 2.067892 | 3072 |
| test | t2m | 3 | [4500,9000]m | 2.544010 | 2.183897 | 15840 |
| test | u10m | 0 | [0,1500)m | 1.158232 | 1.104415 | 15456 |
| test | u10m | 1 | [1500,3000)m | 0.398446 | 0.382862 | 2928 |
| test | u10m | 2 | [3000,4500)m | 0.408745 | 0.404852 | 3072 |
| test | u10m | 3 | [4500,9000]m | 0.516504 | 0.538968 | 15840 |
| test | v10m | 0 | [0,1500)m | 0.845317 | 0.834069 | 15456 |
| test | v10m | 1 | [1500,3000)m | 0.509538 | 0.473478 | 2928 |
| test | v10m | 2 | [3000,4500)m | 0.533414 | 0.488548 | 3072 |
| test | v10m | 3 | [4500,9000]m | 0.593605 | 0.589868 | 15840 |

## +120h gate (t2m only)

| Split | RMSE raw | RMSE living |
| --- | ---: | ---: |
| val | 2.178842 | 1.962487 |
| test | 1.832432 | 1.838313 |

## Notes

- Living residual **does** improve u/v vs raw FCN3, but deltas are small vs t2m lift (expected under t2m-heavy channel weights).
- Phase-3 **v1.3 joint** (`channel_weights=[1.5,1.5,1.5]`) targets wind-vector promote gates using the baseline above; dry-run only until Leonard ACK.
- No metrics invented — extracted from frozen eval + baseline JSON.

