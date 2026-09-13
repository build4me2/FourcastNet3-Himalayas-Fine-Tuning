# FourCastNet3 — Nepal / South Asian mountain fine-tune

**Full project repo** for FCN3 regional adaptation (RRCA-FD) on **2× DGX Spark**: eng code + configs + docs + selected gate metrics.

Private: `build4me2/fourcastnet3-finetune` · default branch **`main`**

## Layout (matches Spark `~/fourcastnet`)

| Path | What | Source |
|------|------|--------|
| `code/` | phase0 + tier_a (v0 → v1.1) | spark-61dd |
| `configs/` | run / data configs | spark-61dd |
| `docs/00-pathway/` … `docs/04-gates/` | Canon research pack (Sheldon) | manii Research + GitHub |
| `docs/research/` | Spark operational docs (REFERENCE/CLAUDE/PROGRESS, Leonard calls) | spark-61dd |
| `runs/phase0/` | Small metrics/gate JSON (+ tiny csv/md) only | spark-61dd |
| `data/`, `models/`, `logs/`, `venv*` | **Not in git** | local Spark only |

## Start here (research)

1. [`docs/00-pathway/FINETUNE_PATHWAY.md`](docs/00-pathway/FINETUNE_PATHWAY.md) — pathway + status  
2. [`docs/01-method/FINETUNE_METHOD_DESIGN.md`](docs/01-method/FINETUNE_METHOD_DESIGN.md) — RRCA-FD  
3. [`docs/04-gates/GATE_RECIPE_TIER0_G0.md`](docs/04-gates/GATE_RECIPE_TIER0_G0.md) — G0 / Tier-0  

## Gitignore policy

No ERA5 crops, G0/Tier-0 `.npy` ICs, pretrained weights, training checkpoints, pair tensors, venvs, `.cdsapirc` / `.env`. See `.gitignore`.

## Spark sync (Howard)

Primary host: **spark-61dd** (`100.121.160.49`), tree `/home/chandmanisha00/fourcastnet/`.

```bash
# On Spark (once gh/SSH auth ready):
cd ~/fourcastnet
git init -b main   # if needed
git remote add origin https://github.com/build4me2/fourcastnet3-finetune.git
git fetch origin
git checkout -B main origin/main   # pull Sheldon docs first
# then add code/ configs/ docs/ runs/phase0/*.json and push
```

Science freeze (as of sync): thick-set **v1.1 INTERIM PASS**; no new Tier-A/diffusion until Manisha orders.

## Owners

| Role | Who |
|------|-----|
| Eng / Spark push | Howard |
| Spark ops | Raj |
| Research docs | Sheldon |
| Experiments | Leonard |
| Decisions | Manisha |
