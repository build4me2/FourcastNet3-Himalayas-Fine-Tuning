# Tier-A v0 smoke note — 2026-09-11

**Not a Leonard PASS/FAIL call.** Pipeline proof only.

- Code: `code/tier_a/` · Config: `configs/tier_a_v0.yaml`
- Results: `runs/phase0/tier_a/v0/tier_a_v0_results.json`
- Beat-this echoed: val <1.948661 · test <1.850338
- Smoke metrics: val t2m **1.902** · test **1.769** (CPU 30 ep, n_IC=4+4)
- Labels: provisional_years=true · interim_era5 · g1_claimable=false
- G0 adapter: N/A (FCN3 frozen)
- Forbidden artifacts untouched; ERA5 PID 611595 untouched

Raj longer train (optional):
```bash
cd ~/fourcastnet
PYTHONUNBUFFERED=1 ~/fcn3-venv/bin/python -u code/tier_a/train.py \
  --config configs/tier_a_v0.yaml --train --allow-gpu --epochs 300
# success: runs/phase0/tier_a/v0/tier_a_v0_results.json gates.tier_a_interim_pass
# estimate: <<2h on GB10 for this tiny UNet (smoke was ~15s/30ep CPU)
```
