# FINAL baselines status — 2026-09-22 ~10:22 PT **COMPLETE**

**Protocol:** FROZEN FINAL_EVAL_PROTOCOL (IC hashes verified).
**g1_claimable:** false (unchanged).
**scored:** **true**
**EVAL ONLY** — no train; living ckpt MD5 unchanged: `c81a5a4c4f07ca0a52bb0bffee0045fa`

## Official artifact
`runs/phase0/final_eval/final_baselines.json` — scored=true @ 2026-09-22T10:22:03-07:00 validate OK

## Headline table (measured; do not invent)

| split | baseline | t2m_pooled_gate | t2m_+120h | wind_vector_lead_mean |
| --- | --- | ---: | ---: | ---: |
| val | raw_fcn3 | 2.099229 | 2.248841 | 0.890307 |
| val | tier0 | 2.039476 | 2.236870 | 0.891250 |
| val | living_v1_3_joint | 1.982226 | 2.204377 | 0.845186 |
| test | raw_fcn3 | 2.182565 | 2.376842 | 0.916351 |
| test | tier0 | 2.104525 | 2.321096 | 0.914646 |
| test | living_v1_3_joint | 1.987826 | 2.237008 | 0.876129 |

## Howard unstick (Manisha order) — 10:07–10:22 PT
1. Inspected: staging 127/128 futex-stuck since ~02:06 PT; missing only `ic_20220912_00`; crops 124; GPU idle.
2. SIGTERM killed stuck tree (preserved all staged npys): PIDs 30854/30856, 30861–30865, 35744/35745; cleared `.crop.lock`.
3. Staged ONLY missing + upserted 3 late npys into manifest via `--ids ic_20220912_00,ic_20251027_00,ic_20251118_12,ic_20251216_00`.
   - `ic_20220912T0000_global.npy` landed 10:14 PT; manifest 128/128; protocol hashes still valid (test=bd50f2fb85f2…).
4. GPU crop remaining 4 → 128/128 crops; targets 128; score + thick2 + validate.
5. Resume PID was 146322; log: `logs/final_eval/resume_missing_ic.log`

## Prior notes
- Smoke 2-IC JSON is NOT for bars.
- Thick-2 bridge continuity re-scored OK (living interim floats match prior).
