# FINAL eval bars call (Leonard) — 2026-09-22

**Status:** **FROZEN** from measured `final_baselines.json` only — **no invented K**.  
**Source:** `~/fourcastnet/runs/phase0/final_eval/final_baselines.json` (SCORED 2026-09-22)  
**Protocol:** `FINAL_EVAL_PROTOCOL.md` · suite `FINAL_EVAL_SUITE_RECIPE.md`  
**Living interim:** `v1_3_joint/` · thick-2 — **unchanged** · MD5 `c81a5a4c4f07ca0a52bb0bffee0045fa`  
**`g1_claimable`:** **false** until a later promote call after a FINAL model clears A∧B∧C  
**Train:** still **NO-GO** until Manisha unlocks FINAL residual train against these bars

---

## 0. Metric definitions (LOCKED with baselines)

| Metric | Definition |
| --- | --- |
| **t2m pooled** | `grid_pooled_gate_leads.t2m_rmse` over gate leads {24,72,120} |
| **t2m +120 h** | per-lead +120 h grid-pooled t2m RMSE |
| **Wind-vector (WV)** | lead-mean of per-lead \(\sqrt{\mathrm{mean}((u_{\mathrm{err}}^2+v_{\mathrm{err}}^2)/2)}\) over {24,72,120} |

Identical ICs / years / pooling as protocol (hashes verified on score).

---

## 1. Measured baselines (authoritative)

| Split | Baseline | t2m pooled | t2m +120 h | WV lead-mean |
| --- | --- | ---: | ---: | ---: |
| Val | Raw FCN3 | **2.099228718846139** | **2.2488412332039327** | **0.8903072718010457** |
| Val | Tier-0 | **2.0394764673534427** | 2.2368695137127985 | 0.8912504510604006 |
| Val | Living `v1_3_joint` | **1.9822262881728796** | 2.2043771587522363 | **0.8451857725124596** |
| Test | Raw FCN3 | **2.182564808439163** | 2.376842176320765 | **0.9163507760768065** |
| Test | Tier-0 | **2.104525111905597** | 2.3210955226537546 | 0.9146455678510034 |
| Test | Living `v1_3_joint` | **1.9878256451936507** | 2.2370080410894744 | **0.8761290512975345** |

(Display roundings in STATUS.md are fine for tables; **gates use full floats above**.)

---

## 2. Beat-this bars (LOCKED)

### A — vs raw FCN3 (hard FAIL if any miss)

| # | Gate | Rule |
| --- | --- | --- |
| A1 | Val t2m pooled | **strictly < 2.099228718846139** |
| A2 | Test t2m pooled | **strictly < 2.182564808439163** |
| A3 | Val +120 h t2m | **≤ 2.2488412332039327** |
| A4 | Val WV | **strictly < 0.8903072718010457** |
| A5 | Test WV | **strictly < 0.9163507760768065** |
| A6 | Eligible ckpt | `n_eligible_saves ≥ 1` · reload=`composite_eligible` (reject if val +120 h > raw A3) |

### B — vs Tier-0 (hard)

| # | Gate | Rule |
| --- | --- | --- |
| B1 | Val t2m pooled | **strictly < 2.0394764673534427** |
| B2 | Test t2m pooled | **strictly < 2.104525111905597** |

### C — vs living on FINAL protocol (hard for G1 candidate)

| # | Gate | Rule |
| --- | --- | --- |
| C1 | Val t2m pooled | **strictly < 1.9822262881728796** |
| C2 | Test t2m pooled | **strictly < 1.9878256451936507** |
| C3 | Val WV | **strictly < 0.8451857725124596** |
| C4 | Test WV | **strictly < 0.8761290512975345** |

**No ε_protect** on FINAL G1 (firmer than interim ±0.02 K).

### Product

```
final_g1_candidate_pass = A ∧ B ∧ C
```

Only after that may Leonard consider flipping `g1_claimable` (still needs Manisha + explicit promote call).

---

## 3. Binding notes

1. **C dominates t2m:** living already beats Tier-0 and raw on t2m; clearing C1/C2 implies A1/A2/B1/B2 for t2m. Still **evaluate and report A/B explicitly**.  
2. **Winds:** living beats raw on WV; Tier-0 does **not** beat raw on val WV (0.89125 > 0.89031) — expected; B does not gate WV. C3/C4 are the wind promote story.  
3. **Legacy bridge:** every FINAL ckpt still re-scores on thick-2 interim protocol (continuity vs living interim headlines). Bridge is not a second promote path.  
4. **Interim thick-2 bars** (1.988588 / 1.918383 / 2.178842) stay frozen as `interim_era5` only — **do not** replace with these FINAL floats for interim runs.

---

## 4. Explicit FAIL

- Any A/B/C miss  
- Unconstrained +120 h fallback  
- Protocol / IC hash mismatch  
- Relabeling interim / living as FINAL or G1 without this pass  
- Inventing softer bars without a new Leonard call

## 5. Non-claims

No global WB2/FCN3 SOTA · no ops replacement · no CorrDiff parity · no precip · no stations · interim remains `interim_era5`.

---

## 6. Next eng (Howard)

1. Mirror this call to Spark `docs/research/FINAL_EVAL_BARS_CALL.md`.  
2. Echo bars into FINAL train config **when** Manisha unlocks train (not before).  
3. Idle on FINAL residual train until Manisha greenlight.  
4. When a FINAL ckpt scores: ping Leonard with results JSON for G1-candidate PASS/FAIL.

## 7. Paths

| Artifact | Location |
| --- | --- |
| This call | manii `~/Desktop/Research/fourcastnet3/FINAL_EVAL_BARS_CALL.md` · Spark `docs/research/` |
| Baselines | Spark `runs/phase0/final_eval/final_baselines.json` |
| Protocol + ICs | `FINAL_EVAL_PROTOCOL.md` · `final_eval_protocol_ics.json` |
