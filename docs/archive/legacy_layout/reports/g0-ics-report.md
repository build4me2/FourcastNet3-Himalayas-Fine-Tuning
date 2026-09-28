# G0 global ERA5 IC staging report (GATE_RECIPE §3 step 1)

**Host:** Spark (`spark-61dd`, ssh `chandmanisha00@100.121.160.49`)  
**When:** 2026-09-11 ~09:20 PT  
**Agent:** Howard executor  
**Recipe:** Leonard `docs/04-gates/GATE_RECIPE_TIER0_G0.md` §1.2 / §3 step 1  

## Success criteria

| Criterion | Status |
|-----------|--------|
| Manifest exists | **YES** — `~/fourcastnet/data/g0_ics/g0_ic_manifest.json` |
| ≥1 real global IC staged | **YES** — **2/8** smoke on disk (72×721×1440 float32) |
| Prefer ≥8 or backfill launched | **Backfill launched** — `python code/phase0/stage_g0_ics.py --all` (PID ~641936); log `logs/g0_ics_backfill.log` |
| Clear next command for G0 base when GPU allowed | See below |

## Deliverables

| Artifact | Path |
|----------|------|
| Fetch/assemble script | `code/phase0/stage_g0_ics.py` (+ copy under `code/data/`) |
| Direct ARCO helper | `code/phase0/arco_direct_fetch.py` |
| Backfill launcher | `code/phase0/run_g0_ics_backfill.sh` |
| IC date/rationale config | `configs/g0_ics.yaml` |
| IC tensors | `data/g0_ics/ic_YYYYMMDDTHHMM_global.npy` |
| Manifest | `data/g0_ics/g0_ic_manifest.json` |
| Docs | `docs/CLAUDE.md`, `docs/PROGRESS.md`, `docs/REFERENCE.md`, `code/README.md` |

## Staged ICs (smoke)

| ID | Time (00 UTC) | Season | Region class / label | File | SHA256 (prefix) | Size |
|----|---------------|--------|----------------------|------|-----------------|------|
| ic01 | 2018-01-15 | DJF | non_asia / N_Atlantic | `ic_20180115T0000_global.npy` | `6eab20ca1a7b…` | 299013248 B (~286 MiB) |
| ic02 | 2019-07-12 | JJA | non_asia / N_Pacific | `ic_20190712T0000_global.npy` | `200af343b60f…` | 299013248 B (~286 MiB) |

- **Shape / dtype:** `(72, 721, 1440)` float32; finite; physical sanity OK (t2m / msl ranges)
- **Channel order:** `models/fourcastnet3/config.json` → `channel_names` (≡ earth2studio `FCN3.VARIABLES`)
- **Source:** ARCO ERA5 final zarr `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3` (direct zarr reads with retries; earth2studio lexicon-compatible identity units)
- **Init hour:** 00 UTC
- **Not used:** Nepal crop-as-IC; Random IC; CDS (avoids competing with regional backfill)

## Planned remaining (backfill in flight)

| ID | Time | Season | Region class / label |
|----|------|--------|----------------------|
| ic03 | 2020-01-20 | DJF | non_asia / SH_midlat |
| ic04 | 2018-08-15 | JJA | non_asia / tropics_Atlantic_Africa |
| ic05 | 2021-01-10 | DJF | asia_ex_nepal / East_Asia |
| ic06 | 2019-08-20 | JJA | asia_ex_nepal / West_Pacific_typhoon |
| ic07 | 2020-06-15 | JJA | monsoon_adjacent / Bay_of_Bengal_monsoon |
| ic08 | 2021-12-05 | DJF | monsoon_adjacent / NE_India_Myanmar_fringe |

**Quota when all 8 land:** 4 DJF + 4 JJA; 4 non-Asia + 2 Asia-ex-Nepal + 2 monsoon-adjacent (outside Nepal box). Region labels are **selection rationale only** — every IC is full globe.

~5–7 min/IC observed → remaining ~30–45 min wall.

## Constraints honored

- **ERA5 regional backfill PID 611595 left running** (untouched)
- **No CDS** for these ICs (ARCO cloud zarr only)
- **No GPU / no FCN3 G0 rollout** (Manisha GPU widget skipped → NOT approved)
- **Nepal crop never used as IC**
- **Random IC forbidden for gate artifacts**

## How to monitor / resume

```bash
ssh chandmanisha00@100.121.160.49
tail -f ~/fourcastnet/logs/g0_ics_backfill.log
# or status:
source ~/fcn3-venv/bin/activate
cd ~/fourcastnet
python -c "import json; print(json.load(open('data/g0_ics/g0_ic_manifest.json'))['counts'])"
ls -lh data/g0_ics/
# if backfill died:
bash code/phase0/run_g0_ics_backfill.sh   # skips already staged
```

## Next command for G0 base (ONLY after Manisha GPU greenlight)

Do **not** run until GPU is explicitly approved:

```bash
ssh chandmanisha00@100.121.160.49
cd ~/fourcastnet && source ~/fcn3-venv/bin/activate
# Confirm staged==8:
python -c "import json; c=json.load(open('data/g0_ics/g0_ic_manifest.json'))['counts']; assert c['staged']>=8, c; print(c)"
# Then implement/run G0 path A (global IC → global rollout → g0_base_results.json)
# Placeholder until g0_base.py exists — expected shape:
#   python code/phase0/g0_base.py --manifest data/g0_ics/g0_ic_manifest.json \
#       --members 4 --steps 60 --out runs/phase0/g0_base_results.json
```

Until GPU greenlight: keep backfill completing; Tier-0 skill fit and G0 metrics remain blocked.

## Notes

- First earth2studio `ARCO()` bulk call hit GCS body timeouts; switched to **direct zarr** (12 array reads/timestep: 7 surface + 5 pressure×37-level chunks) with retries — stable ~270–350 s/IC.
- `np.save` atomic write via file handle; skip path preserves prior `source` metadata.
