# ERA5 Nepal crop — coverage audit

**Generated (UTC):** 2026-09-17T08:25:45.127685+00:00  
**Raw dir:** `/home/chandmanisha00/fourcastnet/data/era5/raw`  
**Window:** 1980-01 → 2026-09  
**Schema:** `era5_coverage_audit/v1`

## Summary

| Metric | Value |
| --- | ---: |
| Months scanned | 561 |
| Present | 343 |
| Missing | 217 |
| Partial | 1 |
| Holes (non-present) | 218 |
| Frontier contiguous from start | 2008-06 |
| Last present month (any) | 2020-01 |

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
| 2008 | 6 | 5 | 1 |
| 2009 | 0 | 12 | 0 |
| 2010 | 0 | 12 | 0 |
| 2011 | 0 | 12 | 0 |
| 2012 | 0 | 12 | 0 |
| 2013 | 0 | 12 | 0 |
| 2014 | 0 | 12 | 0 |
| 2015 | 0 | 12 | 0 |
| 2016 | 0 | 12 | 0 |
| 2017 | 0 | 12 | 0 |
| 2018 | 0 | 12 | 0 |
| 2019 | 0 | 12 | 0 |
| 2020 | 1 | 11 | 0 |
| 2021 | 0 | 12 | 0 |
| 2022 | 0 | 12 | 0 |
| 2023 | 0 | 12 | 0 |
| 2024 | 0 | 12 | 0 |
| 2025 | 0 | 12 | 0 |
| 2026 | 0 | 9 | 0 |

## Holes (missing or partial months)

Count: **218**. First 40: 

`2008-07`, `2008-08`, `2008-09`, `2008-10`, `2008-11`, `2008-12`, `2009-01`, `2009-02`, `2009-03`, `2009-04`, `2009-05`, `2009-06`, `2009-07`, `2009-08`, `2009-09`, `2009-10`, `2009-11`, `2009-12`, `2010-01`, `2010-02`, `2010-03`, `2010-04`, `2010-05`, `2010-06`, `2010-07`, `2010-08`, `2010-09`, `2010-10`, `2010-11`, `2010-12`, `2011-01`, `2011-02`, `2011-03`, `2011-04`, `2011-05`, `2011-06`, `2011-07`, `2011-08`, `2011-09`, `2011-10`

… +178 more (see JSON `holes`).

## Notes

- Read-only audit; does not stop/restart CDS download.
- Surface diagnostics gated for FINAL: t2m / u10m / v10m.
- Pressure files audited for presence only (not surface-var holes).
- Archive audit PASS for FINAL requires contiguous verified years for protocol — see FINAL_EVAL_SUITE_RECIPE.md §2.

## Next

1. Leave CDS download PID alone until tip complete.
2. Re-run this audit when frontier advances.
3. Archive audit PASS → Leonard `FINAL_EVAL_PROTOCOL.md` → score `final_baselines.json`.
