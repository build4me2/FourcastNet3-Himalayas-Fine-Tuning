# FINAL eval suite call (Leonard) — 2026-09-16

**Lit:** Sheldon `EVAL_BENCHMARKS_AND_FINAL_SUITE.md`  
**Recipe frozen:** `FINAL_EVAL_SUITE_RECIPE.md`  
**Living interim:** `v1_3_joint/` · thick-2 · **unchanged / not weakened**

## Call

| Item | Decision |
| --- | --- |
| FINAL suite protocol + hardness | **FROZEN** |
| Interim thick-2 bars / years / living | **KEEP** (`interim_era5`) |
| Claim path for G1 / publishable | **REPLACE** → FINAL suite |
| IC / season density | **TIGHTEN** (val/test ≥64; ≥8/season incl. SON) |
| Numeric beat-this floats | **PENDING** — measure on FINAL protocol only |
| FINAL / G1 train | **NO-GO** until archive audit + protocol years/ICs + `final_baselines.json` + bars call + Manisha |
| `g1_claimable` | **false** until explicit later call |

## How interim maps in

| Keep | Replace | Tighten |
| --- | --- | --- |
| Thick-2 floats & year hard-lock as historical | G1 claim protocol | IC density, SON floors, full lead report, strict living beat (no ±0.02 on FINAL) |
| Legacy thick-2 re-score of every FINAL ckpt | Invented absolute K bars | Truth = audited CDS crop |

## Hard gates (schema)

**A** beat raw FCN3 (t2m val/test, +120 h ≤ raw, WV val/test) + eligible ckpt  
**B** beat Tier-0 (t2m val/test)  
**C** beat living `v1_3_joint` re-scored on FINAL protocol (t2m val/test **strict**; WV val/test **strict**)

Precip / stations / CRPS / extremes = deferred or secondary.

## Non-claims

No global SOTA · no ops replacement · no CorrDiff parity · no precip · no stations · no relabel of interim as FINAL.

## Next

1. Howard: mirror recipe + this call to Spark; **no train**.  
2. Sheldon: lit note remains the citation base (no further action required).  
3. Leonard: after CDS archive audit → `FINAL_EVAL_PROTOCOL.md` → baselines → `FINAL_EVAL_BARS_CALL.md`.  
4. Manisha: unlock FINAL train only after bars call.

## Honesty

Living `v1_3_joint` is an honest **interim** win on a thin locked protocol. It is **not** FINAL G1. This suite is what “we got the model we wanted” means after full ERA5.
