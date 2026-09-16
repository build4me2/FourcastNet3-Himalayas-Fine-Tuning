# Living residual — winds / elev-band table

**Source:** `runs/phase0/tier_a/v1_2b_thick2_train/tier_a_v1_2b_thick2_train_results.json` (+ metrics CSV).
**Model:** Tier-A v1.2b thick-2 retrain (living residual). **No retrain** for this table.
**Elevation bins (m):** [0, 1500, 3000, 4500, 9000]

> **Winds (u/v):** **not present** in living residual eval artifacts. Living residual scores **t2m only**. Tier-0 thick-2 holdout *does* score u10m/v10m (separate; not residual-corrected here).

## Headline t2m (pooled)

| Split | RMSE raw (K) | RMSE tier-A / residual (K) | n |
| --- | ---: | ---: | ---: |
| train_self | 2.176988 | 1.849983 | 83916 |
| val | 2.016630 | 1.759883 | 37296 |
| test | 2.014341 | 1.779660 | 37296 |

Living bars (rounded): val **1.760** / test **1.780** / val +120h **1.962**.

## t2m by lead (h)

| Split | Lead | RMSE raw | RMSE tier-A | n |
| --- | ---: | ---: | ---: | ---: |
| train_self | 24 | 1.850563 | 1.532819 | 27972 |
| train_self | 72 | 2.213727 | 1.825749 | 27972 |
| train_self | 120 | 2.427481 | 2.141125 | 27972 |
| val | 24 | 1.696995 | 1.436099 | 12432 |
| val | 72 | 2.138514 | 1.837887 | 12432 |
| val | 120 | 2.178842 | 1.962487 | 12432 |
| test | 24 | 1.945683 | 1.610051 | 12432 |
| test | 72 | 2.242590 | 1.878807 | 12432 |
| test | 120 | 1.832432 | 1.838313 | 12432 |

## t2m by elevation band

| Split | Bin | Label | RMSE raw | RMSE tier-A | n |
| --- | ---: | --- | ---: | ---: | ---: |
| train_self | 0 | [0,1500)m | 1.413401 | 1.253475 | 34776 |
| train_self | 1 | [1500,3000)m | 2.016685 | 1.728356 | 6588 |
| train_self | 2 | [3000,4500)m | 2.272702 | 2.000655 | 6912 |
| train_self | 3 | [4500,9000]m | 2.730579 | 2.279633 | 35640 |
| val | 0 | [0,1500)m | 1.399552 | 1.252648 | 15456 |
| val | 1 | [1500,3000)m | 1.822012 | 1.663768 | 2928 |
| val | 2 | [3000,4500)m | 2.309077 | 2.065948 | 3072 |
| val | 3 | [4500,9000]m | 2.452855 | 2.102841 | 15840 |
| test | 0 | [0,1500)m | 1.300758 | 1.190187 | 15456 |
| test | 1 | [1500,3000)m | 1.672311 | 1.605322 | 2928 |
| test | 2 | [3000,4500)m | 2.170708 | 2.067892 | 3072 |
| test | 3 | [4500,9000]m | 2.544010 | 2.183897 | 15840 |

## +120h gate (t2m)

| Split | RMSE raw | RMSE tier-A |
| --- | ---: | ---: |
| val | 2.178842 | 1.962487 |
| test | 1.832432 | 1.838313 |

## Notes

- Elev-band RMSE rises with height (val tier-A ~1.25 → ~2.10 K from bin0→bin3).
- Test +120h tier-A (1.838) is *better* than val +120h (1.962); gate uses **val** +120h ≤ 2.178842.
- No u/v residual table invented — only extracted fields present in living artifacts.

