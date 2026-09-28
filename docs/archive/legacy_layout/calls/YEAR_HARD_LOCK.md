# Year hard-lock (B) — 2026-09-13

**Status:** LOCKED (Manisha via Howard; Leonard `FCN_NEXT_UNLOCK.md`)

**Canonical call:** [`docs/research/FCN_NEXT_UNLOCK.md`](FCN_NEXT_UNLOCK.md) §B

## Locked split

| Split | Years | ICs (current expand set) |
| --- | --- | --- |
| Train | **2018–2021** | ic01–ic08 |
| Val | **2022** | ic09–ic12 + ic17–ic24 (12) |
| Test | **2023–2024** | ic13–ic16 + ic25–ic32 (12) |

## Claim labels

- `provisional_years=false`
- `year_split_frozen=true` (**hard**)
- `claim_level=interim_era5` (unchanged until IMDAA/obs)
- `g1_claimable=false` (unchanged until IMDAA/obs)

## Eng rules

- **Do not** retrain Tier-A solely for this label flip.
- Expand beat-this bars **unchanged**: val &lt; 1.987014 / test &lt; 1.897298 / +120h ≤ 2.131516 until unlock **A** re-scores.
- Next unlock after B: **(A) thicken ICs** (see FCN_NEXT_UNLOCK.md). No new architecture / diffusion until A+B done.
- Optional zero-shot re-tag of v1.2b results as sidecar only (weights untouched).
