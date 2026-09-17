# Tier-A v1.3 joint (winds+t2m)

Leonard recipe: `docs/research/TIER_A_V1_3_JOINT_RECIPE.md`

- Config: `configs/tier_a_v1_3_joint.yaml` (`channel_weights=[1.5,1.5,1.5]`)
- Out: `runs/phase0/tier_a/v1_3_joint/`
- Baseline: `living_wind_baseline.json` (required before train)
- **Do not overwrite** living residual `runs/phase0/tier_a/v1_2b_thick2_train/`
- **No `--train` until Leonard ACK** on baseline
- CDS PID 611595 leave alone

```bash
~/fourcastnet/venv/bin/python code/tier_a/v1_3_joint/score_living_wind_baseline.py --device cpu
~/fourcastnet/venv/bin/python code/tier_a/v1_3_joint/train.py --config configs/tier_a_v1_3_joint.yaml --dry-run
```
