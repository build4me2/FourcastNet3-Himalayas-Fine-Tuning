# Spark eng (Howard)

Live data/run tree on **spark-61dd**: `/home/chandmanisha00/fourcastnet` (`~/fourcastnet`).

Git checkout (this repo): `~/fourcastnet-git-sync` on Spark, or the GitHub remote `build4me2/fourcastnet3-finetune`.

## Two trees — do not mix

| Tree | Role |
|------|------|
| `~/fourcastnet` | Live eng + ERA5/ICs + runs + models + venvs. **Not a git repo.** Do not `git init` / `git clean` here. |
| `~/fourcastnet-git-sync` | Git working copy. Edit/commit here, then rsync *code/configs/docs* back to the live tree if needed. |

ERA5 pull (leave it alone): do not kill the crop downloader; do not delete `data/`, `models/`, or ICs.

## What is gitignored

Heavy artifacts stay on Spark only:

- `data/` (ERA5 crops, ICs, cache)
- `models/` (pretrained + checkpoints)
- `runs/**/pairs/` and `*.pt` `*.pth` `*.ckpt` `*.npy` `*.nc`
- `logs/`, `venv/`, `venv-*/`, `__pycache__/`, `*.bak*`
- secrets: `.env`, `.cdsapirc`

Allowed in git: `code/`, `configs/`, `docs/`, small gate JSON under `docs/06-metrics/` and `runs/phase0/`.

## Docs merge rule

Sheldon’s numbered pack (`docs/00-pathway` … `docs/04-gates`) is **canonical**. Spark copies of those filenames live here under `research/` as snapshots only. Operational notes: `docs/research/`. Call notes: `docs/06-calls/`.
