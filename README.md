# FourCastNet3 Himalayas Regional Adaptation

Regional fine-tuning and evaluation of **[NVIDIA FourCastNet 3 (FCN3)](https://huggingface.co/nvidia/fourcastnet3)** over a Nepal-centered central Himalayan domain.

**Region:** all training and evaluation use only the box **26–31°N, 80–89°E** (Nepal and immediately adjacent central Himalaya). "Himalayas" in the project name refers to this domain; no skill is claimed for the Himalayas as a whole, the Karakoram, Hindu Kush, eastern Himalaya, Tibetan Plateau, or South Asia more broadly.

This repository does **not** retrain FCN3 from scratch. It documents a **Spark-feasible** adaptation stack (RRCA-FD) that keeps FCN3 frozen and trains an elevation-conditioned regional residual.

| | |
|---|---|
| **Base model** | [`nvidia/fourcastnet3`](https://huggingface.co/nvidia/fourcastnet3) (Apache-2.0) |
| **Paper (upstream)** | [FourCastNet 3](https://arxiv.org/abs/2507.12144) |
| **Method** | **RRCA-FD** — Regionally Reweighted CRPS Adapter + Frozen Diagnostic |
| **Hardware** | 2× NVIDIA DGX Spark (128 GB unified memory each) |
| **Domain (locked)** | 26–31°N, 80–89°E · focus: **t2m & 10 m winds** |
| **Living residual** | `runs/phase0/final_eval/final_residual_v0/` · `best_residual.pt` md5 **`586ab17b843757bb84e66e2e3af8dc01`** |
| **HF weights** | [`build4me2/fcn3-himalayas-final-residual-v0`](https://huggingface.co/build4me2/fcn3-himalayas-final-residual-v0) — residual adapter, Apache-2.0, `base_model: nvidia/fourcastnet3` (listed as an adapter in the FCN3 model tree) |
| **Claim** | Regional **FINAL G1 candidate** — PASS **A∧B∧C**; **`g1_claimable=true`** (2026-09-27) |

Exact headlines (FINAL protocol 320/64/64 · gate leads 24/72/120): val/test t2m RMSE **1.819863 / 1.856390 K**; val +120 h t2m **2.010414 K**; val/test wind-vector lead-mean RMSE **0.800354 / 0.825004 m/s**. Raw FCN3 on the same ICs: val/test t2m **2.099229 / 2.182565 K**; val +120 h **2.248841 K**; val/test wind-vector **0.890307 / 0.916351 m/s**.  
(source: `FINAL_RESIDUAL_V0_CALL.md`, `final_residual_v0_results.json`)

---

## Documentation (six living docs)

| Doc | Contents |
|-----|----------|
| [`docs/01_PLAN.md`](docs/01_PLAN.md) | Goals, RRCA-FD tiers, gates, FINAL locks, scope boundary |
| [`docs/02_DATA.md`](docs/02_DATA.md) | Box, ERA5 coverage, year splits, IC protocol |
| [`docs/03_METHOD_AND_CODE.md`](docs/03_METHOD_AND_CODE.md) | ElevCond residual, training knobs, code/config map |
| [`docs/04_RESULTS.md`](docs/04_RESULTS.md) | Measured FINAL results, bars, failures, claims / non-claims |
| [`docs/05_IMPLEMENTATION_HISTORY.md`](docs/05_IMPLEMENTATION_HISTORY.md) | Chronology with **plan vs outcome**; failures preserved |

These six files are the **only** markdown documents kept in this repository.

---

## Repository layout

```text
code/           Phase 0, Tier-0/A, FINAL residual train & eval
configs/        YAML for box, ICs, Tier-0 / Tier-A / FINAL
docs/           Six living docs only (01–05)
runs/phase0/    Metrics JSON / small gate artifacts (no .pt in git)
```

**Not in git:** ERA5 crops, IC arrays, `*.pt` checkpoints, pair tensors, virtualenvs, CDS secrets.

---

## Hard non-claims

- Not global WeatherBench-2 / FCN3 SOTA.
- Not an operational NWP replacement.
- Not CorrDiff / diffusion parity.
- No precipitation claim.
- No station / IMDAA validation for the living FINAL model.
- No coverage or skill claim outside 26–31°N, 80–89°E.
- Not a glacier-collapse or GLOF predictor.
  
Thick-2 bridge scores are **report-only** continuity (not a second G1 path).  
(source: `G1_CLAIMABLE_UNLOCK_CALL.md`, `FINAL_RESIDUAL_V0_CALL.md`)

---

## Getting started

1. Read [`docs/01_PLAN.md`](docs/01_PLAN.md) then [`docs/04_RESULTS.md`](docs/04_RESULTS.md).
2. Eng entrypoints: [`code/`](code/) + configs under `configs/`.
3. Weights: Hugging Face link above (md5 must match `586ab17b843757bb84e66e2e3af8dc01`).

---

## Citation & upstream

- Bonev, Kurth, et al. (2025), *FourCastNet 3: A geometric approach to probabilistic machine-learning weather forecasting at scale* — [arXiv:2507.12144](https://arxiv.org/abs/2507.12144)
- [Hugging Face — nvidia/fourcastnet3](https://huggingface.co/nvidia/fourcastnet3)
- [Earth2Studio](https://github.com/NVIDIA/earth2studio)

This repository is an independent regional-adaptation research project built **on top of** that base model.

---

## License

- **Residual adapter weights on Hugging Face** ([`build4me2/fcn3-himalayas-final-residual-v0`](https://huggingface.co/build4me2/fcn3-himalayas-final-residual-v0)): Apache-2.0.
- **This GitHub repository:** no LICENSE file is currently included.
- **Upstream FCN3 weights** ([`nvidia/fourcastnet3`](https://huggingface.co/nvidia/fourcastnet3)): Apache-2.0.
- ERA5 data remain under Copernicus terms; do not redistribute ERA5 extracts from private storage.
