# LEDGER — Method and code
Date extracted: 2026-09-27 PT

## Living architecture (FINAL residual v0)
- ElevCondResidualUNet
- channel_weights=[1.5,1.5,1.5]; w_120=2.0
- FCN3 FROZEN
- parallel residual (not leftover-diff)
- base=16 depth=3 dropout=0.20 (STATUS knobs echo)
- elev_boost=1.5; weight_decay=1e-3; patience 80; epochs 400; seed 42
- (source: FINAL_RESIDUAL_V0_CALL.md, runs/.../STATUS.md)

## Training outcome (FINAL v0)
- status=train_ok; early-stop ep 135; best eligible ep 55
- n_eligible_saves=19; reload=composite_eligible
- wall ~399 s; created 2026-09-25 ~16:25 PT
- best_residual.pt MD5 586ab17b843757bb84e66e2e3af8dc01
- (source: FINAL_RESIDUAL_V0_CALL.md)

## Code paths (Spark live)
- code/phase0/ — bring-up / G0
- code/tier_a/{v1,v1_1,v1_2,v1_2b,v1_2b_thick2_train,v1_3_joint,v1_diff}/
- code/final_eval/final_residual_v0/
- configs/final_residual_v0.yaml (+ tier configs)
- (source: Spark tree listing 2026-09-27)

## Eval product
- final_g1_candidate_pass = A ∧ B ∧ C
- Bars from measured final_baselines.json only (FINAL_EVAL_BARS_CALL.md)
- (source: FINAL_EVAL_BARS_CALL.md)
