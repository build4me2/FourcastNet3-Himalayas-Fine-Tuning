# Tier-A v1.2 call (Leonard) — 2026-09-12

**Artifacts**
- `~/fourcastnet/runs/phase0/tier_a/v1_2/tier_a_v1_2_results.json`
- Parent: `FCN_STATUS_AND_NEXT.md` · bars: `HOLDOUT_EXPAND_CALL.md`

**Spec:** same as v1.1 (base=16, dropout=0.20, w_120=1.5) trained on expand pairs; composite early-stop on expand val 12.

## Call

| Gate | Result |
| --- | --- |
| Val pooled &lt; 1.987014 | **PASS** — 1.905 |
| Test pooled &lt; 1.897298 | **PASS** — 1.887 |
| Val +120 h ≤ 2.131516 | **FAIL** — **2.132014** (Δ **+0.00050 K**) |
| **Tier-A v1.2 INTERIM PASS** | **NO — FAIL** |
| Diffusion | **Still NO** |

## Diagnosis

- **n_eligible_saves = 0** over early-stop window (stopped ep 50). Composite never found an epoch with val +120 h ≤ expand raw.
- Reload was **`fallback_unconstrained_120h`** — by design when none eligible; that ckpt is **not** allowed to mint INTERIM PASS.
- Pooled wins are real; the hard long-lead gate is the kill. Δ is tiny (≈5e-4 K) but **strict ≤ raw stands** — do not round into a PASS.

Honesty: test +120 h **improves** vs raw (1.722 &lt; 1.733); only **val** +120 h fails the hard line.

## Next — v1.2b (iterate; still no diffusion)

Dir: `runs/phase0/tier_a/v1_2b/` (freeze `v1_2/` as FAIL reference).

| Knob | v1.2 | **v1.2b** |
| --- | --- | --- |
| `w_120` | 1.5 | **2.0** |
| patience | 50 | **80** |
| max epochs | 400 | keep / allow full if needed |
| PASS rule | composite | **LOCKED:** `n_eligible_saves ≥ 1` required for INTERIM PASS; unconstrained fallback ⇒ **automatic FAIL** even if pooled clear |
| bars | expand | unchanged |

Optional Manisha override (not default): grant numerical ε with rule `val_120h ≤ raw + 0.001 K` for expand-era only — would flip this run to PASS; **Leonard does not grant that without her explicit lock.**

### Still required for v1.2b PASS
Same three expand gates + eligible composite ckpt.

## Next eng (Howard)

1. Freeze `v1_2/` FAIL artifact.
2. Run **v1.2b** with table above; ping `tier_a_v1_2b_results.json`.
3. No diffusion; no overwrite of prior frozen dirs.

## Non-claims
- Not a plateau license for CorrDiff  
- Not G1 / FCN3 FT  
- Near-miss ≠ PASS  
