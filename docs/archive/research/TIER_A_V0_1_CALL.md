# Tier-A v0.1 call (Leonard) — 2026-09-11

**Artifacts**
- `~/fourcastnet/runs/phase0/tier_a/v0_1/tier_a_v0_1_results.json`
- Compare: `~/fourcastnet/runs/phase0/tier_a/v0/tier_a_v0_results.json` (**still frozen**)
- Note: `docs/research/TIER_A_V0_1_NOTE.md`
- Parent: `docs/research/TIER_A_V0_CALL.md`

**Change vs v0:** equal-weight multi-lead elev-weighted MSE + `ckpt_select=lead_mean`. Same tiny UNet (~35k), same splits/bars, FCN3 frozen.

## Call

| Gate | Result |
| --- | --- |
| Val beat-this (&lt; 1.948661) | **PASS** — 1.889 K |
| Test beat-this (&lt; 1.850338) | **PASS** — 1.786 K |
| G0 adapter | **N/A** (FCN3 frozen) |
| **Tier-A v0.1 INTERIM PASS** | **YES** |
| Published / G1 / lead-uniform | **NO** |
| **+120 h WARN resolved?** | **NO** — accept as v0.x limitation (see next) |

## Headline

| Split | raw | v0 | v0.1 |
| --- | ---: | ---: | ---: |
| val pooled | 1.949 | 1.892 | **1.889** |
| test pooled | 1.893 | **1.733** | 1.786 |

### Val per-lead

| Lead | raw | v0 | v0.1 | vs raw |
| --- | ---: | ---: | ---: | --- |
| +24 h | 1.691 | 1.561 | 1.554 | improve |
| +72 h | 1.901 | 1.620 | 1.658 | improve |
| **+120 h** | **2.220** | 2.382 | **2.354** | **still regress** (~−0.13 K vs raw; ~−0.03 vs v0) |

Lead-balancing **partially** mitigated the WARN vs v0; it did **not** clear regression vs raw. Test pooled moved the wrong way vs v0 but stays under bar.

## Decision on Howard’s fork

| Option | Call |
| --- | --- |
| **Accept WARN** | **YES** — document; do not block interim PASS |
| **v0.2** (stronger +120 weight / elev aux on same tiny UNet) | **SKIP as default** — diminishing returns on this capacity/N; optional only if Manisha wants one more cheap try |
| **Next = CorrDiff / pathway RRCA-FD v1** | **YES — preferred next eng** |

### Freeze policy
- Keep **v0** frozen (best test pooled on the shelf).
- Freeze **v0.1** as the **lead-aware training protocol reference** (loss + ckpt_select) to carry into v1.
- Do not overwrite either.

## Next eng (Howard)

1. **Stop the v0.x tiny-UNet lead-weight treadmill** unless Manisha orders v0.2.
2. **Scaffold Tier-A v1** toward pathway RRCA-FD / CorrDiff-class (new dir `runs/phase0/tier_a/v1_…`), same IC splits + beat-this bars + `provisional_years=true`.
3. v1 success still requires val+test beat-this; add **explicit per-lead table** with rule: **val +120 h must not regress vs raw** for v1 INTERIM PASS (new hard WARN→gate upgrade for v1 only — v0/v0.1 grandfathered).
4. Any FCN3 weight touch → full G0 adapter re-roll vs verifying baseline.
5. Ping Leonard with v1 design sketch path before long train if architecture choice is ambiguous; otherwise ping metrics when ready.

## Non-claims
- Not G1 / IMDAA / CorrDiff / FCN3 FT / lead-uniform  
- Small-N; interim ERA5; provisional years  
