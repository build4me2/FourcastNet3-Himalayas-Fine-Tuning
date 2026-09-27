# G1 claimable unlock (Leonard) — 2026-09-27

**Manisha unlock:** 2026-09-27 ~13:37 PT — flip **`g1_claimable=true`** for living FINAL residual v0.  
**Prior call:** `FINAL_RESIDUAL_V0_CALL.md` — OFFICIAL **PASS** A∧B∧C · living already promoted.  
**Bars:** unchanged (`FINAL_EVAL_BARS_CALL.md`) — do not invent or soften.

## Scope (LOCKED)

| Field | Value |
| --- | --- |
| Living path | `runs/phase0/final_eval/final_residual_v0/` |
| Checkpoint | `best_residual.pt` · MD5 **`586ab17b843757bb84e66e2e3af8dc01`** |
| Protocol | `FINAL_EVAL_PROTOCOL.md` · ICs `final_eval_protocol_ics.json` (320/64/64 · years 1980–2019 / 2020–2021 / 2022–2025) |
| Box | 26–31°N / 80–89°E · ERA5 truth · frozen FCN3 + ElevCond residual |
| Gate leads | +24 / +72 / +120 h |
| **`g1_claimable`** | **`true`** (this call) |

## Allowed claim language

OK to state, with protocol citation:

1. **G1 candidate (regional FINAL):** On the locked FINAL eval protocol, `final_residual_v0` clears A∧B∧C vs raw FCN3, Tier-0 t2m, and prior living residual scored on the same FINAL ICs.
2. **Headline numbers** (FINAL protocol only): val/test t2m pooled **1.819863 / 1.856390**; val +120 h t2m **2.010414**; val/test wind-vector lead-mean **0.800354 / 0.825004** (from `FINAL_RESIDUAL_V0_CALL` / results JSON).
3. **Method label:** frozen FCN3 backbone + ElevCondResidualUNet residual (joint t2m+u10m+v10m), Nepal box, CDS ERA5.

Always attach: protocol years/ICs, bars call, and results path when publishing numbers.

## Non-claims (hard)

- Not global FCN3 / WeatherBench-2 SOTA  
- Not operational NWP replacement  
- Not CorrDiff / diffusion parity  
- No precip · no station / IMDAA claim  
- Not a license to overwrite `v1_3_joint/` or retrain under softer bars  
- Thick-2 bridge remains **report-only** continuity — not a second G1 path  
- **No Hugging Face / public weight release** in this call — Howard may flip status/results flags only; HF waits for a later Manisha unlock

## Next eng (Howard)

1. Set `g1_claimable=true` in living status / results echo (and GitHub docs sync as needed).  
2. Do **not** change bars, protocol, or checkpoint bytes.  
3. Do **not** push HF release until Manisha unlocks that separately.  
4. Idle otherwise.
