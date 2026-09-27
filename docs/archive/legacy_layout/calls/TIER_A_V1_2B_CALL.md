# Tier-A v1.2b call (Leonard) — 2026-09-12

**Artifacts**
- `~/fourcastnet/runs/phase0/tier_a/v1_2b/tier_a_v1_2b_results.json`
- Parent FAIL: `TIER_A_V1_2_CALL.md` · bars: `HOLDOUT_EXPAND_CALL.md`

**Spec:** ElevCondResidualUNet base=16, dropout=0.20, **w_120=2.0**, patience **80**, expand pairs; composite eligible reload required.

## Call

| Gate | Result |
| --- | --- |
| Val pooled &lt; 1.987014 | **PASS** — **1.809** |
| Test pooled &lt; 1.897298 | **PASS** — **1.876** |
| Val +120 h ≤ 2.131516 | **PASS** — **1.998** (Δ −0.134) |
| Eligible ckpt (`n_eligible_saves≥1`) | **PASS** — **17** saves; reload=`composite_eligible` (ep 204) |
| **Tier-A v1.2b INTERIM PASS** | **YES** |
| Published / G1 / diffusion | **NO** |

## Freeze

- Freeze **`v1_2b/`** as the **living expand-protocol Tier-A residual** reference (trained + eligible composite).
- Keep **`v1_2/`** frozen FAIL (0-eligible lesson).
- Keep **`v1_1_expand/`** zero-shot PASS as historical (thin-train → thick eval).
- Keep **`v0/`** as best thin-era test-pooled champ.
- Do not overwrite any of the above.

## Headline vs recent

| | v1.1 expand ZS | v1.2 FAIL | **v1.2b** |
| --- | ---: | ---: | ---: |
| Val | 1.808 | 1.905 | **1.809** |
| Test | **1.826** | 1.887 | 1.876 |
| Val +120 h | 1.998 | 2.132 ✗ | **1.998** ✓ |
| Eligible saves | n/a (ZS) | 0 | **17** |

Honesty: test +120 h still regresses vs raw (1.872 vs 1.733) — **not** a gate; document only. v1.1 zero-shot still slightly better on test pooled; v1.2b wins on **protocol honesty** (trained under expand val + eligible constraint).

## Next

1. **No diffusion by default** — not a clear plateau (test still has headroom under 1.897; ZS test better).
2. Optional next eng (Manisha pick): seed sweep / winds table / year hard-lock / more ICs / or open **v1-diff** design if she wants Phase 3 now.
3. Howard: idle on GPU science until she assigns; keep ERA5 backfill as ops.

## Non-claims
- Not G1 / IMDAA / CorrDiff / FCN3 FT / lead-uniform on test +120 h  
- provisional_years=true · interim_era5  
