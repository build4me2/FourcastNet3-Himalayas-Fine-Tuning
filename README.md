# FourCastNet3 — Nepal / HKH regional fine-tune

Private project repo for adapting **FourCastNet 3** toward Nepal / Himalaya / South Asian mountain skill on **2× DGX Spark**, using **RRCA-FD** (Regionally Reweighted CRPS Adapter + Frozen Diagnostic).

**Repo:** https://github.com/build4me2/fourcastnet3-finetune · branch **`main`**

## Quick start

1. [`docs/00-pathway/FINETUNE_PATHWAY.md`](docs/00-pathway/FINETUNE_PATHWAY.md) — where the project is  
2. [`docs/01-method/FINETUNE_METHOD_DESIGN.md`](docs/01-method/FINETUNE_METHOD_DESIGN.md) — RRCA-FD  
3. [`code/README.md`](code/README.md) — how to run Phase 0 / Tier-A  
4. [`docs/README.md`](docs/README.md) — full doc map  

## Repository layout

```
code/           # phase0 + tier_a eng (v0 → v1.1)
configs/        # YAML run configs
docs/           # research pack + eng/status/calls/reports
runs/phase0/    # small gate metrics JSON only
```

Not in git (Spark-local only): `data/`, `models/`, checkpoints, pair tensors, venvs, secrets.

## Locked decisions (snapshot)

| Item | Value |
|------|--------|
| Box | 26–31°N, 80–89°E |
| v1 success | t2m / winds first (precip later) |
| Split | ~80/20 random block-level; test ≥15–20% |
| Method | RRCA-FD Plan A; Plan B gated |
| Base | `nvidia/fourcastnet3` (Apache-2.0) |

Science freeze as of eng sync: Tier-A **v1.1 INTERIM PASS**; no new Tier-A/diffusion until Manisha orders.

## Owners

| Role | Who |
|------|-----|
| Decisions | Manisha |
| Research docs | Sheldon |
| Eng / Spark | Howard |
| Experiments | Leonard |
| Spark ops | Raj |
