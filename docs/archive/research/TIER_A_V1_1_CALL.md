# Tier-A v1.1 call (Leonard) — 2026-09-11

**Artifacts**
- `~/fourcastnet/runs/phase0/tier_a/v1_1/tier_a_v1_1_results.json`
- Note: `docs/research/TIER_A_V1_1_NOTE.md`
- Parent FAIL: `docs/research/TIER_A_V1_CALL.md`

**Spec:** ElevCondResidualUNet base=16, dropout=0.20, w_120=1.5, weight_decay=1e-3, ~168k params, composite +120 h reject, FCN3 frozen, no diffusion.

## Call

| Gate | Result |
| --- | --- |
| Val pooled &lt; 1.948661 | **PASS** — 1.716 |
| Test pooled &lt; 1.850338 | **PASS** — 1.812 |
| Val +120 h ≤ raw 2.220 | **PASS** — 2.015 (−0.205) |
| **Tier-A v1.1 INTERIM PASS** | **YES** |
| Published skill / G1 | **NO** (`interim_era5`, provisional years) |
| Start v1-diff / CorrDiff? | **NO** — pass ≠ plateau; N=4 still binding |

## Freeze

- Freeze **`v1_1/`** as the **first +120 h-compliant INTERIM PASS** reference (protocol + ckpt).
- Keep **`v0/`** frozen as **best test-pooled** champ (1.733).
- Keep **`v0_1/`** (lead-aware protocol ancestor) and **`v1/`** (FAIL / overfit lesson) frozen.
- Do not overwrite any of the above.

## Headline

| Split | raw | v0 | v0.1 | v1 | **v1.1** |
| --- | ---: | ---: | ---: | ---: | ---: |
| val | 1.949 | 1.892 | 1.889 | 1.796 | **1.716** |
| test | 1.893 | **1.733** | 1.786 | 1.950 ✗ | **1.812** ✓ |
| val +120 h | 2.220 | 2.382 | 2.354 | 2.018 | **2.015** ✓ |

Honesty: test +120 h still regresses vs raw (note ~2.000 vs 1.844) — **not** a current gate; document only. v1.1 beats bars with long-lead discipline; v0 still wins raw test pooled.

## Next (priority order)

1. **Do not start diffusion** until more holdout ICs or a clear skill ceiling after N↑.
2. **Ask Manisha** to hard-lock years and/or stage more val/test global ICs — N=4 is the binding constraint for claiming anything beyond interim.
3. Optional eng (low priority): leave-one-train-IC-out early-stop proxy (never tune on test); keep same three gates.
4. When N grows: re-score v1.1 zero-shot on new ICs; only then revisit `v1-diff` if regression plateaus under the upgraded holdout.

## Non-claims
- Not G1 / IMDAA / CorrDiff / FCN3 FT / lead-uniform on test +120 h  
- Not better than v0 on test pooled  
- provisional_years=true  
