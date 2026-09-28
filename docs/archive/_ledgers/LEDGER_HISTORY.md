# LEDGER — Implementation history (plan vs outcome)
Date extracted: 2026-09-27 PT

| When (PT) | Plan | Outcome | Source |
| 2026-09-10 | Fresh Spark tree / RRCA-FD design | Design docs + Phase 0 stubs | FINETUNE_*; fresh-start |
| 2026-09-11..12 | G0 / Tier-0 | G0 verifying PASS claimable; Tier-0 thin/expand bars | g0 reports; FCN_STATUS |
| 2026-09-13 | Year hard-lock interim | 2018–21/2022/2023–24 locked interim_era5 | YEAR_HARD_LOCK.md |
| 2026-09-13..14 | Tier-A v1.2 / v1.2b | v1.2 FAIL; v1.2b interim PASS then historical | FCN_STATUS; calls |
| ~2026-09-14 | Thick-2 retrain | v1_2b_thick2_train interim PASS → historical living then demoted | FCN_STATUS |
| 2026-09-14..15 | v1-diff leftover target | NULL — do not promote | TIER_A_V1_DIFF_CALL (research) |
| 2026-09-15 | v1.3 joint promote | INTERIM PASS CONFIRM; living→v1_3_joint | TIER_A_V1_3_JOINT_CALL |
| 2026-09-16..17 | Paper + FINAL suite scaffold | Protocol/bars path opened; coverage audit | commits / FINAL_EVAL_* |
| 2026-09-21..22 | FINAL protocol + bars freeze | Years 1980–2019/2020–2021/2022–2025; bars from baselines | FINAL_EVAL_PROTOCOL; BARS |
| 2026-09-25 | FINAL residual v0 train | PASS A∧B∧C; living→final_residual_v0; md5 586ab17b… | FINAL_RESIDUAL_V0_CALL |
| 2026-09-27 | g1 unlock + HF | g1_claimable=true; HF public model | G1_CLAIMABLE_UNLOCK; HF_RELEASE |
