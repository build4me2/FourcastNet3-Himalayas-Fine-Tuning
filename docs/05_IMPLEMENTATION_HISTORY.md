# 05 — Implementation history (plan vs outcome)

**Consolidation date:** 2026-09-27 PT  
**Rule:** Preserve failures. Distinguish **plan** from **outcome**. Cite sources.

Sources: `PROGRESS.md`, `FCN_STATUS_AND_NEXT.md`, `STATUS_WHAT_WORKS_WHAT_FAILED.md`, tier/FINAL call cards, git history on `FourcastNet3-Himalayas-Fine-Tuning`.

## Timeline

### 2026-09-10 — Fresh start (plan → outcome)

| Plan | Outcome |
| --- | --- |
| Stand up Spark `~/fourcastnet`, mirror research pack, Phase 0 stubs | Tree live; models/venvs preserved; `load_norms` OK; torch/env still blocking inference at report time |

(source: eng fresh-start report / `PROGRESS.md` early entries)

### 2026-09-11–12 — G0 / Tier-0 baselines

| Plan | Outcome |
| --- | --- |
| G0 integrity + Tier-0 bias bars | G0 verifying **PASS claimable**; Tier-0 thin then expand bars frozen as historical beat-this |

(source: `g0-verifying-report.md`, `tier0-holdout-report.md`, `FCN_STATUS_AND_NEXT.md`)

### 2026-09-13 — Interim year hard-lock

| Plan | Outcome |
| --- | --- |
| Freeze interim train/val/test years | **2018–2021 / 2022 / 2023–2024** locked; `interim_era5`; `g1_claimable=false` at that time |

(source: `YEAR_HARD_LOCK.md`)

### 2026-09-13–14 — Tier-A v1.2 / v1.2b

| Plan | Outcome |
| --- | --- |
| Improve +120 h / expand protocol | **v1.2 FAIL** (frozen historical). **v1.2b** interim PASS → later historical |

(source: `FCN_STATUS_AND_NEXT.md`, `STATUS_WHAT_WORKS_WHAT_FAILED.md`)

### ~2026-09-14 — Thick-2 ZS + retrain

| Plan | Outcome |
| --- | --- |
| Thicken ICs; retrain residual | Thick-2 ZS interim PASS (demoted). **v1_2b_thick2_train** interim PASS and briefly living, then superseded |

Headlines (historical): thick-2 retrain val **1.760** / test **1.780** / +120h **1.962**.  
(source: `FCN_STATUS_AND_NEXT.md`)

### 2026-09-14–15 — v1-diff (leftover-target diffusion)

| Plan | Outcome |
| --- | --- |
| Test leftover-target diffusion vs residual | **NULL** — ≈ residual (Δ ~1e-5); ensemble worse; **stop scaling**; keep artifact |

(source: `STATUS_WHAT_WORKS_WHAT_FAILED.md`, `FCN_STATUS_AND_NEXT.md`)

### 2026-09-15 — v1.3 joint living promote (interim)

| Plan | Outcome |
| --- | --- |
| Joint t2m+winds residual; promote if A/B/C interim | **INTERIM PASS CONFIRM**; living → `runs/phase0/tier_a/v1_3_joint/`; md5 **`c81a5a4c…`** |

Headlines (thick-2 16-IC): val **1.770** / test **1.775** / +120h **1.982** · WV **0.69560 / 0.73877**.  
WARN: results JSON briefly echoed 17 test ICs incl. `ic45` — claims use locked 16-IC path.  
(source: `TIER_A_V1_3_JOINT_CALL.md`)

### 2026-09-16–22 — FINAL suite freeze

| Plan | Outcome |
| --- | --- |
| After ERA5 audit, freeze FINAL years/ICs/bars; no train until unlock | Protocol years **1980–2019 / 2020–2021 / 2022–2025**; ICs **320/64/64**; bars from measured `final_baselines.json`; `g1_claimable=false` until promote |

(source: `FINAL_EVAL_PROTOCOL.md`, `FINAL_EVAL_BARS_CALL.md`, git commits #6–#9 era)

### 2026-09-25 — FINAL residual v0 train + promote

| Plan | Outcome |
| --- | --- |
| Train ElevCond residual under FINAL bars; Leonard call | **`train_ok`**; PASS **A∧B∧C**; living → `final_eval/final_residual_v0/`; md5 **`586ab17b843757bb84e66e2e3af8dc01`**; interim `v1_3_joint/` frozen historical |

At promote echo, status docs still carried **`g1_claimable=false`** until unlock.  
(source: `FINAL_RESIDUAL_V0_CALL.md`, `PROGRESS.md` append 2026-09-25)

### 2026-09-27 — G1 unlock + HF release

| Plan | Outcome |
| --- | --- |
| Manisha unlock claim flip; optional HF | **`g1_claimable=true`**; HF public model `build4me2/fcn3-nepal-final-residual-v0`; GitHub sync PR #10 merge + follow-up docs commits |

(source: `G1_CLAIMABLE_UNLOCK_CALL.md`, `HF_RELEASE_FINAL_RESIDUAL_V0.md`, git `74d3971` / `9ee744f` / `3abc9da`)

### 2026-09-27 — Docs consolidation (this work)

| Plan | Outcome |
| --- | --- |
| Collapse sprawl into six living docs; archive never delete | See `docs/archive/SOURCE_MAP.md`; branch `docs/consolidate-six-clean-docs` |

## Failures that must remain visible

1. **v1.2 FAIL** — +120 h miss.  
2. **v1-diff NULL** — leftover-diff does not beat residual.  
3. **Thick-2 bridge** — FINAL ckpt worse on interim thick-2 t2m vs `v1_3_joint` (continuity honesty, not FINAL FAIL).  
4. **Stale doc pointers** — `PROJECT_SCOPE_AND_HISTORY.md` / parts of pathway dashboard / STATUS “no HF” line — archived with conflict notes rather than silently rewritten as if always current.

## Main tip SHAs (repo `FourcastNet3-Himalayas-Fine-Tuning` at consolidation start)

| SHA | Note |
| --- | --- |
| `3abc9da7470d9fad090b8b21c18603e7f6d8cb46` | main tip before consolidation PR (docs: PR #10 note) |
| `9ee744fdf3ba9d1d375fd450351603b07611160e` | HF URL cite |
| `74d3971a59fb587d28bd9cd8a5dc28b6a3b43138` | Merge PR #10 FINAL residual sync |

(source: `git log` / GitHub API 2026-09-27)
