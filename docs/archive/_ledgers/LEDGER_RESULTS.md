# LEDGER — Results (exact)
Date extracted: 2026-09-27 PT

## Living FINAL residual v0 headlines (exact floats from call/JSON)
| Metric | Value |
| Val t2m pooled | 1.8198625586531565 (display 1.819863 / 1.8199) |
| Test t2m pooled | 1.8563898906179215 (display 1.856390 / 1.8564) |
| Val +120h t2m | 2.010413966886577 (display 2.010414 / 2.0104) |
| Val WV lead-mean | 0.8003535884784455 (display 0.800354 / 0.8004) |
| Test WV lead-mean | 0.8250037602833951 (display 0.825004 / 0.8250) |
(source: FINAL_RESIDUAL_V0_CALL.md, final_residual_v0_results.json)

## Gates
- A PASS, B PASS, C PASS → final_g1_candidate_pass YES
- g1_claimable=true (unlock 2026-09-27)
- (source: FINAL_RESIDUAL_V0_CALL.md, G1_CLAIMABLE_UNLOCK_CALL.md, results JSON)

## Bars beat (from FINAL_EVAL_BARS_CALL)
A raw val t2m < 2.099228718846139; test < 2.182564808439163; val+120h ≤ 2.2488412332039327; WV val < 0.8903072718010457; test < 0.9163507760768065
B Tier-0 val < 2.0394764673534427; test < 2.104525111905597
C living-on-FINAL val < 1.9822262881728796; test < 1.9878256451936507; WV val < 0.8451857725124596; test < 0.8761290512975345
(source: FINAL_EVAL_BARS_CALL.md)

## Checkpoint
- path: runs/phase0/final_eval/final_residual_v0/best_residual.pt
- md5: 586ab17b843757bb84e66e2e3af8dc01 (verified on Spark 2026-09-27)
- HF: https://huggingface.co/build4me2/fcn3-nepal-final-residual-v0
- (source: md5sum; HF_RELEASE_FINAL_RESIDUAL_V0.md; LIVING_INDEX.md)

## Interim living historical (v1_3_joint thick-2)
- val 1.770 / test 1.775 / +120h 1.982; WV 0.69560 / 0.73877
- md5 c81a5a4c4f07ca0a52bb0bffee0045fa
- (source: TIER_A_V1_3_JOINT_CALL.md, LIVING_INDEX.md)

## Thick-2 bridge (report-only)
- FINAL ckpt on thick-2: val t2m 1.883 / test 1.875 (call table); STATUS echo 1.883181 / 1.875323
- vs interim living: residual_v0 worse on t2m (~+0.11/+0.10) — continuity only, not FAIL of FINAL A∧B∧C
- (source: FINAL_RESIDUAL_V0_CALL.md, STATUS.md)

## Failures preserved
- v1.2 FAIL (+120h gate miss)
- v1-diff NULL (leftover-target diffusion ≈ residual; ensemble worse)
- (source: STATUS_WHAT_WORKS_WHAT_FAILED.md, FCN_STATUS_AND_NEXT.md)

## Hard non-claims
- Not global WB2/FCN3 SOTA; not ops; not CorrDiff parity; no precip; no stations
- (source: G1_CLAIMABLE_UNLOCK_CALL.md, FINAL_RESIDUAL_V0_CALL.md)
