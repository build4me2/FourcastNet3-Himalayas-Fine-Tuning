# Engineering notes

Live Spark tree: `~/fourcastnet` on **spark-61dd** (Tailscale `100.121.160.49`).

| Tree | Role |
|------|------|
| `~/fourcastnet` | Live eng + ERA5/ICs + runs + models + venvs. **Not a git repo.** |
| GitHub `main` (this repo) | Source of truth for `code/`, `configs/`, and docs. |

Do not commit data, models, checkpoints, or secrets. See root [`.gitignore`](../../.gitignore).

Code entry points: [`code/README.md`](../../code/README.md) · configs: [`configs/`](../../configs/).
