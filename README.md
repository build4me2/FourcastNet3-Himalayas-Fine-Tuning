# FourCastNet3 — Nepal / South Asian mountain fine-tune

Research + method docs for adapting **FourCastNet 3 (FCN3)** toward Nepal / HKH / South Asian orography skill on **2× DGX Spark** (128 GB unified each), using **RRCA-FD** (Regionally Reweighted CRPS Adapter + Frozen Diagnostic).

**Not** a replay of NVIDIA’s Stage1→2→FT curriculum. Goal: preserve FCN3’s probabilistic spherical / dual-CRPS picture while gaining regional skill.

## Start here

| Order | Doc | What it is |
|------:|-----|------------|
| 1 | [`docs/00-pathway/FINETUNE_PATHWAY.md`](docs/00-pathway/FINETUNE_PATHWAY.md) | Complete pathway + status tracking |
| 2 | [`docs/01-method/FINETUNE_METHOD_DESIGN.md`](docs/01-method/FINETUNE_METHOD_DESIGN.md) | RRCA-FD technique (canonical method) |
| 3 | [`docs/04-gates/GATE_RECIPE_TIER0_G0.md`](docs/04-gates/GATE_RECIPE_TIER0_G0.md) | Tier-0 / G0 IC + gate recipe |
| 4 | [`docs/03-compute/SPARK_FINETUNE_PLAN.md`](docs/03-compute/SPARK_FINETUNE_PLAN.md) | What 2 Sparks can / can’t do |
| 5 | [`docs/02-background/`](docs/02-background/) | Training data, FCN3 method, regional gap, prior FTs |

## Background pack (`docs/02-background/`)

- `TRAINING_METHOD.md` — Lead’s picture of how FCN3 was built  
- `TRAINING_DATA.md` — ERA5 / channels / years  
- `REGIONAL_FINETUNE_GAP.md` — why Nepal/HKH adaptation is open  
- `FCN_REGIONAL_FINETUNES.md` — survey of FCN-family regional fine-tunes  

## Locked decisions (snapshot)

| Item | Lock |
|------|------|
| Box | 26–31°N, 80–89°E |
| v1 success | t2m / winds first (precip later) |
| Split policy | ~80/20 random **block-level**; test ≥15–20% |
| Data intent | Max ERA5 through latest (Howard owns pull) |
| Method | RRCA-FD Plan A primary; Plan B gated |
| Base model | `nvidia/fourcastnet3` (Apache-2.0) |

## Status (as of docs sync)

Tier-A v1.1 interim PASS frozen; G0 verifying PASS; Tier-0 beat-this frozen. **Next unlock:** year hard-lock / more ICs before diffusion. Engineering on Sparks is owned by Howard; Sheldon = docs unless asked.

## Layout

```
docs/
  00-pathway/     # tracking + lifecycle
  01-method/      # RRCA-FD
  02-background/  # lit + FCN3 internals
  03-compute/     # Spark feasibility
  04-gates/       # G0 / Tier-0 recipes
```

Code, training scripts, and data paths will land in later commits (Howard/Leonard). Do **not** commit ERA5 crops, tokens, or checkpoints.

## Owners

| Role | Who |
|------|-----|
| Research / docs | Sheldon |
| Eng / data pull | Howard |
| Experiments | Leonard |
| Spark ops | Raj |
| Decisions | Manisha |
