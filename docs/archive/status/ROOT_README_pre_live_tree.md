# fourcastnet (live Spark tree)

This directory is the **live eng + data** tree on spark-61dd (`~/fourcastnet`).
It is **not** a git checkout. Do not `git init`, `git clean`, or delete `data/`, `models/`, or ICs.

Git working copy: `~/fourcastnet-git-sync` → private GitHub `build4me2/fourcastnet3-finetune`.
Copy code/configs/docs into the git-sync tree to commit; rsync back if you need edits in the live tree.

ERA5 / GPU jobs: leave running downloaders alone (including era5 crop PID if active).
