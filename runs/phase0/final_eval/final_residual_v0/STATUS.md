# FINAL residual v0 — STATUS
**Updated:** 2026-09-27 13:18 PT
**Recipe:** docs/research/FINAL_TRAIN_RECIPE.md (Leonard freeze; Manisha unlock GO)
**Out dir:** runs/phase0/final_eval/final_residual_v0/
**g1_claimable:** false
**Living v1_3_joint MD5:** c81a5a4c4f07ca0a52bb0bffee0045fa (guarded)

## Locked knobs (= v1_3_joint)
- ElevCondResidualUNet base=16 depth=3 dropout=0.20
- channel_weights=[1.5,1.5,1.5]; w_120=2.0; elev_boost=1.5; weight_decay=1e-3
- patience 80; epochs 400; seed 42
- composite reject if val +120h > raw A3 **2.2488412332039327**
- PASS = A∧B∧C from FINAL_EVAL_BARS_CALL.md (Leonard owns official call)
- Thick-2 bridge: **FILLED** (report-only; `thick2_bridge_results.json`; g1_claimable remains false)

## Status now
| Step | State |
| --- | --- |
| Config configs/final_residual_v0.yaml | written |
| Code code/final_eval/final_residual_v0/ | dry-run OK (IC hashes verified) |
| Val/test pairs (128) | ready (reused) |
| Train ARCO globals (320) | IN PROGRESS workers=8 |
| Train crop + CDS targets | queued (supervisor) |
| GPU --train | queued after pairs |
| Results JSON / A∧B∧C | train metrics written; thick-2 bridge FILLED (report-only) |

## PIDs
- Supervisor: **165647** (scripts/run_final_residual_v0_pipeline.sh)
- Stage train ICs: **166507** (parallel_stage_ics.py --splits train --workers 8)

## Logs
- logs/final_eval/final_residual_v0_supervisor.log
- logs/final_eval/stage_train_ics.log
- logs/final_eval/crop_train.log (pending)
- logs/final_eval/targets_train.log (pending)
- logs/final_eval/final_residual_v0_train.log (pending)

## ETA (rough)
- ARCO train IC staging: ~6-12 h
- Crop + targets: ~1-3 h after staging
- GPU train 400 ep / patience 80: hours on GB10 after pairs ready

## Hard rules honored
- No overwrite of tier_a/v1_3_joint/
- g1_claimable stays false
- No diffusion / multi-seed
- Hash-verify FINAL ICs fail-hard on load


## Thick-2 legacy bridge (optional, report-only)
**Updated:** 2026-09-27 13:18 PT
- Status: **FILLED** (non-blocking continuity table)
- Harness: `code/final_eval/thick2_legacy_bridge.py --score` on thick-2 12/16/16
- Out: `runs/phase0/final_eval/final_residual_v0/thick2_bridge_results.json`
- Ckpt MD5 (unchanged): `586ab17b843757bb84e66e2e3af8dc01`
- Living v1_3_joint MD5 (unchanged): `c81a5a4c4f07ca0a52bb0bffee0045fa`
- Device: cuda (NVIDIA GB10)
- **g1_claimable:** false (not flipped)
- Headline thick-2: val t2m_pooled **1.883181** / test **1.875323**; WV val **0.701070** / test **0.741207**
- vs living thick-2: residual_v0 worse on t2m (~+0.11 val / +0.10 test); WV nearly flat
