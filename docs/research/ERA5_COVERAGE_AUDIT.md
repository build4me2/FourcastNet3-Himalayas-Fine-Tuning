# ERA5 Nepal crop — coverage audit

**Generated (UTC):** 2026-09-21T15:51:50.764546+00:00  
**Raw dir:** `/home/chandmanisha00/fourcastnet/data/era5/raw`  
**Window:** 1980-01 → 2022-07  
**Schema:** `era5_coverage_audit/v1`

## Summary

| Metric | Value |
| --- | ---: |
| Months scanned | 511 |
| Present | 510 |
| Missing | 0 |
| Partial | 1 |
| Holes (non-present) | 1 |
| Frontier contiguous from start | 2022-06 |
| Last present month (any) | 2022-06 |

## Variable map (surface diagnostics)

| FCN3 / recipe | CDS short (in .nc) | CDS long name |
| --- | --- | --- |
| t2m | `t2m` | 2m_temperature |
| u10m | `u10` | 10m_u_component_of_wind |
| v10m | `v10` | 10m_v_component_of_wind |

## By year

| Year | Present | Missing | Partial |
| ---: | ---: | ---: | ---: |
| 1980 | 12 | 0 | 0 |
| 1981 | 12 | 0 | 0 |
| 1982 | 12 | 0 | 0 |
| 1983 | 12 | 0 | 0 |
| 1984 | 12 | 0 | 0 |
| 1985 | 12 | 0 | 0 |
| 1986 | 12 | 0 | 0 |
| 1987 | 12 | 0 | 0 |
| 1988 | 12 | 0 | 0 |
| 1989 | 12 | 0 | 0 |
| 1990 | 12 | 0 | 0 |
| 1991 | 12 | 0 | 0 |
| 1992 | 12 | 0 | 0 |
| 1993 | 12 | 0 | 0 |
| 1994 | 12 | 0 | 0 |
| 1995 | 12 | 0 | 0 |
| 1996 | 12 | 0 | 0 |
| 1997 | 12 | 0 | 0 |
| 1998 | 12 | 0 | 0 |
| 1999 | 12 | 0 | 0 |
| 2000 | 12 | 0 | 0 |
| 2001 | 12 | 0 | 0 |
| 2002 | 12 | 0 | 0 |
| 2003 | 12 | 0 | 0 |
| 2004 | 12 | 0 | 0 |
| 2005 | 12 | 0 | 0 |
| 2006 | 12 | 0 | 0 |
| 2007 | 12 | 0 | 0 |
| 2008 | 12 | 0 | 0 |
| 2009 | 12 | 0 | 0 |
| 2010 | 12 | 0 | 0 |
| 2011 | 12 | 0 | 0 |
| 2012 | 12 | 0 | 0 |
| 2013 | 12 | 0 | 0 |
| 2014 | 12 | 0 | 0 |
| 2015 | 12 | 0 | 0 |
| 2016 | 12 | 0 | 0 |
| 2017 | 12 | 0 | 0 |
| 2018 | 12 | 0 | 0 |
| 2019 | 12 | 0 | 0 |
| 2020 | 12 | 0 | 0 |
| 2021 | 12 | 0 | 0 |
| 2022 | 6 | 0 | 1 |

## Holes (missing or partial months)

Count: **1**. First 40: 

`2022-07`

## Notes

- Read-only audit; does not stop/restart CDS download.
- Surface diagnostics gated for FINAL: t2m / u10m / v10m.
- Pressure files audited for presence only (not surface-var holes).
- Archive audit PASS for FINAL requires contiguous verified years for protocol — see FINAL_EVAL_SUITE_RECIPE.md §2.

## Next

1. Nepal ERA5 CDS crop still filling — leave download alone (check `ps` / `logs/era5_pull.log` for live PID).
2. Re-run this audit when frontier advances.
3. Archive audit PASS → Leonard `FINAL_EVAL_PROTOCOL.md` → score `final_baselines.json`.
