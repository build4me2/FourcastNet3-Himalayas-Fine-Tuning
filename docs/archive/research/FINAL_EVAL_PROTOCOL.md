# FINAL eval protocol (Leonard) — 2026-09-21

**Status:** **FROZEN** (Manisha greenlight after ERA5 archive audit).  
**Suite:** `FINAL_EVAL_SUITE_RECIPE.md` + `FINAL_EVAL_SUITE_CALL.md` (do not weaken).  
**IC companion:** `final_eval_protocol_ics.json` (full ID lists + hashes).  
**Numeric bars:** **NOT** in this file — Howard scores `final_baselines.json` next; then `FINAL_EVAL_BARS_CALL.md`.  
**Train / GPU:** **NO** until bars call + Manisha unlock.  
**`g1_claimable`:** **false** until later promote call.

**Audit basis (Howard 2026-09-21 ~17:08 PT):** contiguous full months through **2026-08**; soft hole **2026-09** tip-partial excluded from gated years. Artifacts: `~/fourcastnet/data/era5/coverage_audit.json`, `docs/research/ERA5_COVERAGE_AUDIT.md`.

---

## 1. Year split (LOCKED integers)

| Split | Years | Calendar years |
| --- | --- | --- |
| **Train** | **1980–2019** | 1980…2019 (40 y) |
| **Val** | **2020–2021** | 2020, 2021 (2 y) |
| **Test** | **2022–2025** | 2022, 2023, 2024, 2025 (4 y) |
| **Excluded from gated splits** | **2026** | Jan–Aug available but incomplete SON year; Sep tip-partial — **ungated tip / report-only** if used later |

**Membership rule:** IC **calendar year** determines split. No Dec(Y−1) pulled into year Y (prevents train→val DJF leakage).

**Legacy bridge (unchanged):** interim thick-2 years 2018–21 / 2022 / 2023–24 + living `v1_3_joint/` remain `interim_era5` historical; every FINAL ckpt still re-scores on thick-2.

---

## 2. Domain / leads / metrics (echo LOCKED)

| Item | Spec |
| --- | --- |
| Box | 26–31°N / 80–89°E |
| Truth | Audited CDS ERA5 crop (gated years hole-free through 2025) |
| Gate leads | +24 / +72 / +120 h |
| Report leads | +24 / +48 / +72 / +96 / +120 h |
| P0 | t2m RMSE (val primary / test secondary); val +120 h t2m; wind-vector lead-mean |
| Wind-vector | \(\sqrt{\mathrm{mean}((u_{\mathrm{err}}^2+v_{\mathrm{err}}^2)/2)}\) grid-pooled → lead-mean over gate leads |
| Elev bands | Report-only |
| Beat-this schema | A raw / B Tier-0 / C living on this protocol — floats after baselines |

---

## 3. IC generation recipe (LOCKED)

| Knob | Value |
| --- | --- |
| Init hours | **00 and 12 UTC** (both used) |
| Min separation | **≥ 5 days** within each split (achieved min ≫ 5 d) |
| Per-year per-season count | train **2** · val **8** · test **4** |
| DJF in calendar year | Jan–Feb **and** Dec of the **same** calendar year |
| Deterministic | Yes — lists below are authoritative |

### Counts (verify floors)

| Split | N | DJF | MAM | JJA | SON | Floor check |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Train | **320** | 80 | 80 | 80 | 80 | ≥48 ✓ (season prefer-balanced ✓) |
| Val | **64** | 16 | 16 | 16 | 16 | ≥64 · ≥8/season ✓ |
| Test | **64** | 16 | 16 | 16 | 16 | ≥64 · ≥8/season ✓ |

---

## 4. IC list hashes (LOCKED)

Hash = **SHA-256** of sorted IC ID strings joined by newline, trailing newline (UTF-8).

| Split | N | sha256 |
| --- | ---: | --- |
| Train | 320 | `e119c288bfca9d170a1692cbc566948e3d21797d727b4b060f8c418c6ce9d86d` |
| Val | 64 | `42bffc2e3cad252a0b8217e6a2f8919e4b681aa17554a2c8591bee04a4b4c403` |
| Test | 64 | `bd50f2fb85f25316fe0a90d1b92867fcb98bc4d5bc46c497169890eef4204828` |

**Full ID lists:** `final_eval_protocol_ics.json` keys `train_ids` / `val_ids` / `test_ids`.  
ID form: `ic_YYYYMMDD_HH` (HH = `00` or `12`).

Reproduce check:
```bash
python3 -c "import json,hashlib; m=json.load(open('final_eval_protocol_ics.json'));
for s in ['train','val','test']:
  ids=m[s+'_ids']; h=hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest();
  assert h==m['hashes_sha256_of_sorted_ic_ids_joined_by_newline'][s], s"
```

---

## 5. Paths (LOCKED)

| Artifact | manii | Spark |
| --- | --- | --- |
| This protocol | `~/Desktop/Research/fourcastnet3/FINAL_EVAL_PROTOCOL.md` | `~/fourcastnet/docs/research/FINAL_EVAL_PROTOCOL.md` |
| IC JSON | `~/Desktop/Research/fourcastnet3/final_eval_protocol_ics.json` | `~/fourcastnet/docs/research/final_eval_protocol_ics.json` |
| Suite recipe/call | same dir / `docs/research/` | same |
| Next baselines | — | `runs/phase0/final_eval/final_baselines.json` |

---

## 6. Eng next (Howard) — no train

1. Mirror this protocol + IC JSON to Spark `docs/research/`.  
2. Wire eval harness to these years + IC IDs (hash-verify on load).  
3. Score **raw FCN3**, **Tier-0**, living **`v1_3_joint`** on this protocol → `final_baselines.json`.  
4. Ping Leonard for `FINAL_EVAL_BARS_CALL.md`.  
5. **No FINAL residual train** until bars call + Manisha. CDS tip 2026-09 leave alone / optional.

---

## 7. Non-claims / labels

- Protocol freeze ≠ G1 pass.  
- `claim_level` for future FINAL runs TBD at bars/promote; until then **`g1_claimable=false`**.  
- Interim thick-2 / `v1_3_joint` **not** relabeled FINAL.  
- No invented RMSE/K bars in this document.
