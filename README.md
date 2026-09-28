# FourCastNet3 — Nepal / Himalaya Regional Adaptation

Regional fine-tuning and evaluation of **[NVIDIA FourCastNet 3 (FCN3)](https://huggingface.co/nvidia/fourcastnet3)** for improved probabilistic weather skill over **Nepal, the Hindu Kush–Himalaya (HKH), and neighboring South Asian terrain**.

This repository does **not** retrain FCN3 from scratch. It documents a **Spark-feasible** adaptation stack (RRCA-FD) that keeps FCN3 frozen and trains an elevation-conditioned regional residual.

| | |
|---|---|
| **Base model** | [`nvidia/fourcastnet3`](https://huggingface.co/nvidia/fourcastnet3) (Apache-2.0) |
| **Paper (upstream)** | [FourCastNet 3](https://arxiv.org/abs/2507.12144) |
| **Method** | **RRCA-FD** — Regionally Reweighted CRPS Adapter + Frozen Diagnostic |
| **Hardware** | 2× NVIDIA DGX Spark (128 GB unified memory each) |
| **Domain (locked)** | 26–31°N, 80–89°E · focus: **t2m & 10 m winds** |
| **Living residual** | `runs/phase0/final_eval/final_residual_v0/` · `best_residual.pt` md5 **`586ab17b843757bb84e66e2e3af8dc01`** |
| **HF weights** | https://huggingface.co/build4me2/fcn3-nepal-final-residual-v0 |
| **Claim** | Regional **FINAL G1 candidate** — PASS **A∧B∧C**; **`g1_claimable=true`** (2026-09-27) |

Exact headlines (FINAL protocol 320/64/64 · gate leads 24/72/120): val/test t2m **1.819863 / 1.856390**; val +120 h **2.010414**; val/test wind-vector lead-mean **0.800354 / 0.825004**.  
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
| [`docs/archive/SOURCE_MAP.md`](docs/archive/SOURCE_MAP.md) | Archive map of superseded sources |

Superseded call cards, research design packs, and agent ops notes live under [`docs/archive/`](docs/archive/) (**never deleted**).

---

## Repository layout

```text
code/           Phase 0, Tier-0/A, FINAL residual train & eval
configs/        YAML for box, ICs, Tier-0 / Tier-A / FINAL
docs/           Six living docs + archive/
runs/phase0/    Metrics JSON / small gate artifacts (no .pt in git)
```

**Not in git:** ERA5 crops, IC arrays, `*.pt` checkpoints, pair tensors, virtualenvs, CDS secrets.

---

## Hard non-claims

Not global WeatherBench-2 / FCN3 SOTA · not operational NWP replacement · not CorrDiff/diffusion parity · no precip claim · no station/IMDAA claim for living FINAL.  
Thick-2 bridge scores are **report-only** continuity (not a second G1 path).  
(source: `G1_CLAIMABLE_UNLOCK_CALL.md`, `FINAL_RESIDUAL_V0_CALL.md`)

---

## Getting started

1. Read [`docs/01_PLAN.md`](docs/01_PLAN.md) then [`docs/04_RESULTS.md`](docs/04_RESULTS.md).
2. Eng entrypoints: [`code/`](code/) + configs under `configs/`.
3. Weights: Hugging Face link above (md5 must match `586ab17b843757bb84e66e2e3af8dc01`).

---

## Citation & upstream

- Assran et al. / NVIDIA — [arXiv:2507.12144](https://arxiv.org/abs/2507.12144)
- [Hugging Face — nvidia/fourcastnet3](https://huggingface.co/nvidia/fourcastnet3)
- [Earth2Studio](https://github.com/NVIDIA/earth2studio)

This repository is an independent regional-adaptation research project built **on top of** that base model.

---

## License

Project docs/code: see included file terms. Upstream FCN3 weights: Apache-2.0. Do not redistribute proprietary ERA5 extracts from private storage.
