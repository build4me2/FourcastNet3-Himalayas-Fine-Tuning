# G0 base call (Leonard) — 2026-09-11

**Artifact:** `~/fourcastnet/runs/phase0/g0/g0_base_results.json`  
**status:** `full_ok` · path **A** · 8/8 ICs · ens=4 · steps=60 (+15 d) · peak CUDA alloc ~53 GB · `storage_mode=preview_only`

## Call

| Layer | Result |
| --- | --- |
| **G0 plumbing / integrity (finite fields)** | **PASS** — all_finite on preview vars; no NaN/Inf; 8/8 ICs completed |
| **G0 claimable (CRPS / SSR / PSD vs ERA5 @ +15 d)** | **NOT YET** — metrics still **stubs**; `g0_pass_claimable` must stay false |

Do **not** treat this file as a G0 skill baseline for adapter comparisons until verifying ERA5 is scored.

## Next eng (Howard)

1. **Proceed now:** Tier-0 **real** — global ERA5 IC → FCN3 → crop **26–31N, 80–89E**; target ERA5 interim labeled `target=ERA5_interim`; t2m/winds-first (Manisha lock).
2. **Also required before G0 PASS claim:** Wire real CRPS / SSR / PSD vs verifying ERA5 at +15 d into `g0_base_results.json` (or sibling `g0_base_metrics.json`). Keep stub run as plumbing reference.
3. Tier-A / Tier-B: blocked on honest G0 metrics **and** Tier-0 real baseline.

## Non-claims

- No spectral / CRPS pass  
- Preview-only storage ≠ full 72-ch archive  
- Random-IC smoke remains non-skill  

