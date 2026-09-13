# FCN3 Phase 0 — status & next (Leonard) — 2026-09-12

**Manisha:** work on FourCastNet (Howard = FCN-only; TTA parked).

## Frozen state (do not overwrite)

| Layer | Status | Path / bar |
| --- | --- | --- |
| G0 verifying | **PASS claimable** | `g0_verifying_results.json`; adapters ≤+5% CRPS; PSD hard &lt;0.25 |
| Tier-0 thin | historical | val 1.948661 / test 1.850338 |
| Tier-0 **expand** | **living beat-this** | val **&lt; 1.987014** / test **&lt; 1.897298** / +120h ≤ **2.131516** |
| Tier-A v0 | best **test pooled** champ (1.733 thin) | `tier_a/v0/` |
| Tier-A v1.1 | first +120h-compliant PASS (thin) | `tier_a/v1_1/` |
| Tier-A v1.1 **expand zero-shot** | **INTERIM PASS CONFIRM** | val 1.808 / test 1.826 / +120h 1.998 |
| Diffusion | **not started** | — |
| Years | provisional 2018–21 / 2022 / 2023–24 | not hard-locked |
| Labels | `interim_era5`; `g1_claimable=false` | — |

Calls: `HOLDOUT_EXPAND_CALL.md`, `TIER_A_V1_1_CALL.md`, `G0_VERIFYING_CALL.md`.

## Recommended next (default)

**v1.2 — retrain** same ElevCondResidualUNet as v1.1 (base=16, dropout=0.20, w_120=1.5, composite +120h reject) using:
- train: ic01–ic08 (unchanged)
- **val early-stop / composite on expand val 12** (ic09–12 + ic17–24)
- score: expand test 12
- bars: expand beat-this (above)
- dir: `runs/phase0/tier_a/v1_2/` · do not overwrite v0/v0_1/v1/v1_1/v1_1_expand

**Why before v1-diff:** zero-shot PASS used a ckpt selected on thin val; thick val may pick a better eligible epoch and close the gap vs v0 test champ. Pathway Phase 3 (diffusion) only after a clear regression plateau under the **thick** protocol.

### v1.2 PASS / FAIL
Same three gates as expand call. If PASS and test still ≥ v0’s thin-era test by a large margin on expand protocol, treat as **plateau signal** → open v1-diff design. If FAIL → debug capacity/reg, still no diffusion.

## Alternatives (if Manisha overrides)

| Option | When |
| --- | --- |
| **v1-diff now** | She wants CorrDiff-lite immediately; accept skipping thick-val retrain |
| **More ICs / year hard-lock** | She wants stronger N or G1 path before more models |
| **Winds/elev-band deep dive** | Science writeup without new train |
| **Pause eng** | Status-only; Howard idle on FCN |

## Non-goals until ordered
- TTA harness (Howard not assigned)
- FCN3 weight FT / Tier-B
- IMDAA / G1 claims
- Overwriting frozen artifacts

## Next eng (Howard) — pending Manisha confirm of default
1. Implement **v1.2** under `code/tier_a/v1_2/` + `runs/phase0/tier_a/v1_2/`.
2. Echo expand bars in result JSON.
3. Ping Leonard with `tier_a_v1_2_results.json` for PASS/FAIL.
