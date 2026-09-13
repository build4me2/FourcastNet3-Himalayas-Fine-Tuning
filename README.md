# FourCastNet3 — Nepal / South Asian mountain fine-tune

**Full project repo** for FCN3 regional adaptation (RRCA-FD) on **2× DGX Spark**: eng code + configs + docs + selected gate metrics.

Private: `build4me2/fourcastnet3-finetune` · default branch **`main`**

## Layout (matches Spark `~/fourcastnet`)

| Path | What | Source |
|------|------|--------|
| `code/` | phase0 + tier_a (v0 → v1.1) | spark-61dd |
| `configs/` | run / data configs | spark-61dd |
| `docs/00-pathway/` … `docs/04-gates/` | Canon research pack (Sheldon) | manii Research + GitHub |
| `docs/research/` | Spark operational docs (REFERENCE/CLAUDE/PROGRESS, gate reports) | spark-61dd |
| `docs/05-eng/` | Eng README + Spark-era snapshots of colliding canon filenames | spark-61dd |
| `docs/06-calls/` | Leonard/Howard call notes | spark-61dd |
| `docs/06-metrics/` | Small gate JSON summaries | spark-61dd |
| `runs/phase0/` | Same small metrics/gate JSON (no pair tensors) | spark-61dd |
| `data/`, `models/`, `logs/`, `venv*` | **Not in git** | local Spark only |

## Start here (research)

1. [`docs/00-pathway/FINETUNE_PATHWAY.md`](docs/00-pathway/FINETUNE_PATHWAY.md) — pathway + status  
2. [`docs/01-method/FINETUNE_METHOD_DESIGN.md`](docs/01-method/FINETUNE_METHOD_DESIGN.md) — RRCA-FD  
3. [`docs/04-gates/GATE_RECIPE_TIER0_G0.md`](docs/04-gates/GATE_RECIPE_TIER0_G0.md) — G0 / Tier-0  

## Gitignore policy

No ERA5 crops, G0/Tier-0 `.npy` ICs, pretrained weights, training checkpoints, pair tensors, venvs, `.cdsapirc` / `.env`. See `.gitignore`.

## Spark sync (Howard)

Primary host: **spark-61dd** (`100.121.160.49`).

| Tree | Role |
|------|------|
| `~/fourcastnet` | Live eng + ERA5/ICs + runs + models. **Not a git repo.** Do not `git init` / `git clean`. |
| `~/fourcastnet-git-sync` | Git checkout of this repo. Commit here; rsync code/configs/docs to/from the live tree. |

See [`docs/05-eng/README.md`](docs/05-eng/README.md). Push via `eng/spark-sync` PRs. Do not delete ERA5/ICs.

Science freeze (as of sync): thick-set **v1.1 INTERIM PASS**; no new Tier-A/diffusion until Manisha orders.

## Owners

| Role | Who |
|------|-----|
| Eng / Spark push | Howard |
| Spark ops | Raj |
| Research docs | Sheldon |
| Experiments | Leonard |
| Decisions | Manisha |
