# FourCastNet3 — Nepal / Himalaya Regional Adaptation

Regional fine-tuning and evaluation of **[NVIDIA FourCastNet 3 (FCN3)](https://huggingface.co/nvidia/fourcastnet3)** for improved probabilistic weather skill over **Nepal, the Hindu Kush–Himalaya (HKH), and neighboring South Asian terrain**.

FCN3 is a global, probabilistic, spherical neural-operator model (72 atmospheric variables at 0.25°, 6-hour steps, spatial + spectral CRPS). This repository does **not** retrain FCN3 from scratch. It builds a **Spark-feasible** adaptation stack that preserves FCN3’s probabilistic behavior while targeting known regional weaknesses over complex orography.

| | |
|---|---|
| **Base model** | [`nvidia/fourcastnet3`](https://huggingface.co/nvidia/fourcastnet3) (Apache-2.0) |
| **Paper** | [FourCastNet 3](https://arxiv.org/abs/2507.12144) |
| **Inference stack** | [Earth2Studio](https://github.com/NVIDIA/earth2studio) / Makani |
| **Method** | **RRCA-FD** — Regionally Reweighted CRPS Adapter + Frozen Diagnostic |
| **Hardware** | 2× NVIDIA DGX Spark (128 GB unified memory each) |
| **Domain (v1)** | 26–31°N, 80–89°E · focus: **t2m & winds** (precip later) |

---

## Motivation

Global ML weather models often underperform over steep mountain terrain. Public FCN-family regional fine-tunes are sparse for FCN3 specifically; full multi-step ensemble weight fine-tuning is not practical on two Sparks. This project designs and validates a **parameter-efficient, quality-gated** path:

1. **Frozen FCN3** global rollouts (integrity gate **G0**)
2. **Tier-0** elevation-aware bias baselines
3. **Tier-A** regional residual / diagnostic models (CorrDiff / StormCast–lite style)
4. Optional later **Tier-B** PEFT on FCN3 weights (gated; not default)

Honesty rule: Tier-A skill is **not** labeled “FCN3 weight fine-tune” unless Tier-B criteria pass.

---

## Repository layout

```text
code/           Phase 0 bring-up, Tier-0, Tier-A training & eval
configs/        YAML for boxes, ICs, Tier-0 / Tier-A runs
docs/           Research design + eng notes + call cards
runs/phase0/    Small gate / metrics JSON (no weights or ERA5 dumps)
```

| Path | Description |
|------|-------------|
| [`docs/00-pathway/`](docs/00-pathway/) | Project pathway & status tracking |
| [`docs/01-method/`](docs/01-method/) | RRCA-FD method design |
| [`docs/02-background/`](docs/02-background/) | FCN3 training data/method, regional gap, prior fine-tunes |
| [`docs/03-compute/`](docs/03-compute/) | Feasibility on 2× DGX Spark |
| [`docs/04-gates/`](docs/04-gates/) | G0 / Tier-0 evaluation recipes |
| [`docs/eng/`](docs/eng/) | Live Spark tree vs this git tree |
| [`docs/calls/`](docs/calls/) | Experiment call notes & freeze cards |
| [`docs/reports/`](docs/reports/) | Human-readable gate reports |
| [`code/README.md`](code/README.md) | How to run Phase 0 / Tier-A |

**Not in git** (kept on compute hosts only): ERA5 crops, IC `.npy` files, model checkpoints, pair tensors, virtualenvs, API secrets (`.cdsapirc`, `.env`).

---

## Status (snapshot)

| Component | State |
|-----------|--------|
| G0 (global integrity) | Verifying-ERA5 path **PASS**-claimable under published bars |
| Tier-0 bias | Expand holdout **beat-this** bars frozen |
| Tier-A | **v1.1** interim PASS on expand protocol; **v0** remains thin-set test reference |
| Diffusion / CorrDiff-heavy | Held until thick-set plateau / explicit go |
| Train/val/test years | Policy locked (~80/20 block-level); calendar years not hard-locked |

For the latest eng recommendation, see [`docs/calls/FCN_STATUS_AND_NEXT.md`](docs/calls/FCN_STATUS_AND_NEXT.md).

---

## Getting started

1. Read the pathway: [`docs/00-pathway/FINETUNE_PATHWAY.md`](docs/00-pathway/FINETUNE_PATHWAY.md)
2. Skim the method: [`docs/01-method/FINETUNE_METHOD_DESIGN.md`](docs/01-method/FINETUNE_METHOD_DESIGN.md)
3. Follow eng entrypoints: [`code/README.md`](code/README.md)

Typical Phase 0 flow (on a configured Spark / FCN environment):

```bash
source ~/fcn3-venv/bin/activate   # or project venv
cd /path/to/fourcastnet3-finetune

python code/phase0/env_smoke.py
python code/phase0/load_norms.py
# Global ICs, G0, Tier-0, Tier-A — see code/README.md
```

Coordinate long GPU jobs with your cluster ops. Do not delete or interrupt active ERA5 downloads.

---

## Design locks (v1)

| Decision | Value |
|----------|--------|
| Geographic box | 26–31°N, 80–89°E |
| Primary skill | 2 m temperature and winds first |
| Data split | ~80% train / ~20% test, **block-level** random (no adjacent-frame leakage); test ≥15–20% |
| Data intent | ERA5 through latest available tip (crop + FCN3 channels) |
| Primary technique | RRCA-FD Plan A; Plan B gated |

---

## Citation & upstream

If you use FCN3, please cite the FourCastNet 3 paper and follow NVIDIA’s model card / Earth2Studio guidance:

- Assran et al. / NVIDIA — [arXiv:2507.12144](https://arxiv.org/abs/2507.12144)
- [Hugging Face — nvidia/fourcastnet3](https://huggingface.co/nvidia/fourcastnet3)
- [Earth2Studio](https://github.com/NVIDIA/earth2studio)

This repository is an independent regional-adaptation research project built **on top of** that base model.

---

## License

Code and project documentation in this repository: see license terms of included files and upstream FCN3 (Apache-2.0) for the base weights. Do not redistribute proprietary ERA5 extracts from private storage.
