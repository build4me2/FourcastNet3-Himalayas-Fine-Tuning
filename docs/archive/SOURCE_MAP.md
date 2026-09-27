# SOURCE_MAP — FCN3 Nepal docs consolidation

**Date:** 2026-09-27 PT  
**Scope:** Spark `~/fourcastnet/docs` → six clean living docs + archive  
**Repo target:** `build4me2/FourcastNet3-Himalayas-Fine-Tuning`  
**Rule:** Archive never delete. No fabrication. Cite `(source: FILENAME.md)`.

## Do-not-modify (note only)

| Path | Note |
| --- | --- |
| `_github_stage/` | Staging snapshot; not rewritten by this consolidation |
| `tmp/hf_release_*` | HF release scratch; not rewritten |
| `.pytest_cache/` | Local test cache; not rewritten |
| `*.pt` weights | Never enter git |

## Living six docs (post-consolidation)

| Living doc | Role |
| --- | --- |
| `README.md` (repo root) | Public entry + living status + links |
| `docs/01_PLAN.md` | Goals, pathway, gates, locks |
| `docs/02_DATA.md` | Box, ERA5, years, ICs |
| `docs/03_METHOD_AND_CODE.md` | Method + code/config map |
| `docs/04_RESULTS.md` | Measured outcomes (plan vs outcome) |
| `docs/05_IMPLEMENTATION_HISTORY.md` | Chronology; preserve failures |

Working ledgers (archive): `docs/archive/_ledgers/`  
This map: `docs/archive/SOURCE_MAP.md`

## calls/ vs research/ mirrors (Phase 0 diff)

**Both present, byte-identical (19):**  
`ERA5_COVERAGE_AUDIT.md`, `EVAL_BENCHMARKS_AND_FINAL_SUITE.md`, `FCN_STATUS_AND_NEXT.md`, `FINAL_EVAL_BARS_CALL.md`, `FINAL_EVAL_PROTOCOL.md`, `FINAL_EVAL_SUITE_CALL.md`, `FINAL_EVAL_SUITE_RECIPE.md`, `FINAL_RESIDUAL_V0_CALL.md`, `G1_CLAIMABLE_UNLOCK_CALL.md`, `HF_RELEASE_FINAL_RESIDUAL_V0.md`, `IC45_SON_TEST_NOTE.md`, `LIVING_INDEX.md`, `LIVING_RESIDUAL_WINDS_ELEV_TABLE.md`, `PAPER_DRAFT_METHODS_RESULTS.md`, `PROJECT_SCOPE_AND_HISTORY.md`, `STATUS_WHAT_WORKS_WHAT_FAILED.md`, `TIER_A_V1_3_JOINT_CALL.md`, `TIER_A_V1_3_JOINT_RECIPE.md`, `final_eval_protocol_ics.json`

**Only in research/ (35):** design/pathway + earlier tier calls (listed under archive inventory).  
**Only in calls/:** none.

**Canon rule used historically:** prefer `docs/research/` then `cp` to `docs/calls/` when mirrored. After consolidation, living narrative is the six docs; both trees archived under `docs/archive/{research,calls}/`.

## Agent ops files

| File | Disposition |
| --- | --- |
| `docs/CLAUDE.md` | Archived → `docs/archive/agent/CLAUDE.md` |
| `docs/REFERENCE.md` | Archived → `docs/archive/agent/REFERENCE.md` |
| `docs/CLAUDE.md.bak_pre_expand_bars` | Archived → `docs/archive/agent/` (if present on Spark) |
| `docs/PROGRESS.md.bak_pre_expand_bars` | Archived → `docs/archive/agent/` (if present on Spark) |

## Root / status / reports

| File | Disposition |
| --- | --- |
| `docs/PROGRESS.md` | Archived → `docs/archive/status/PROGRESS.md` (root copy) |
| `docs/status/PROGRESS.md` | Archived → `docs/archive/status/PROGRESS.status_copy.md` (**[CONFLICT:]** MD5 differs from root `PROGRESS.md`) |
| `docs/README.md` | Archived → `docs/archive/status/docs_README_pre.md` |
| `docs/g0-ics-report.md` | Archived → `docs/archive/reports/` |
| `docs/g0-verifying-report.md` | Archived → `docs/archive/reports/` |
| `docs/tier0-holdout-report.md` | Archived → `docs/archive/reports/` |
| `docs/figures/` | Kept in place (figures stub); not superseded narrative |

## GitHub numbered layout (repo PR)

Pre-consolidation GitHub paths `docs/00-pathway/`, `01-method/`, `02-background/`, `03-compute/`, `04-gates/`, `eng/`, `metrics/`, `reports/`, duplicate untracked `05-eng/`, `06-calls/`, `06-metrics/`, `research/` mirrors are superseded by the six living docs. On the PR branch they are moved under `docs/archive/legacy_layout/` (git mv), not deleted.
