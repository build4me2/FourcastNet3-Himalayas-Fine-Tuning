"""Dry unit tests for FINAL scorer scaffold — fake tiny tensors only.

No real ICs, no GPU, no CDS, no .pt writes.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]  # code/
REPO = ROOT.parent  # fourcastnet/
FINAL = ROOT / "final_eval"
sys.path.insert(0, str(FINAL))

import score_final_baselines as sfb  # noqa: E402
import thick2_legacy_bridge as t2b  # noqa: E402


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _minimal_cfg() -> dict:
    return {
        "recipe": "docs/research/FINAL_EVAL_SUITE_RECIPE.md",
        "call": "docs/research/FINAL_EVAL_SUITE_CALL.md",
        "claim_level": "pending_final_protocol",
        "g1_claimable": False,
        "region": {
            "lat_south": 26.0,
            "lat_north": 31.0,
            "lon_west": 80.0,
            "lon_east": 89.0,
        },
        "wind_vector": {
            "formula": "sqrt(mean((u_err^2 + v_err^2)/2))",
            "pool": "grid_then_lead_mean",
            "channels": ["u10m", "v10m"],
        },
        "leads": {
            "gate_h": [24, 72, 120],
            "report_h": [24, 48, 72, 96, 120],
        },
        "variables_out": ["t2m", "u10m", "v10m"],
        "year_split": {"status": "TODO_PENDING_ARCHIVE_AUDIT", "train": [], "val": [], "test": []},
        "ics": {
            "status": "TODO_PENDING_FINAL_EVAL_PROTOCOL",
            "train_ids": [],
            "val_ids": [],
            "test_ids": [],
        },
        "baselines": {
            "living_v1_3_joint": {
                "ckpt": str(REPO / "runs/phase0/tier_a/v1_3_joint/best_residual.pt"),
            }
        },
        "legacy_bridge": {"enabled": True, "protocol": "thick2_12_16_16"},
        "honesty": ["Living v1_3_joint remains interim_era5."],
    }


def test_build_schema_placeholder_validates():
    doc = sfb.build_schema_placeholder(_minimal_cfg())
    errs = sfb.validate_schema(doc)
    assert errs == [], errs
    assert doc["schema"] == "final_baselines/v1"
    assert doc["scored"] is False
    assert doc["g1_claimable"] is False
    assert doc["status"] == "SCHEMA_PLACEHOLDER"
    # all metric floats null — no invented skill
    for split in ("val", "test"):
        for b in sfb.BASELINES:
            block = doc["splits"][split][b]
            assert block["status"] == "PENDING_PROTOCOL_ICS"
            assert block["lead_mean"]["t2m_rmse"] is None
            assert block["lead_mean"]["wind_vector_rmse"] is None


def test_schema_rejects_bad_box_and_formula():
    doc = sfb.build_schema_placeholder(_minimal_cfg())
    doc["box"]["lat_south"] = 0.0
    doc["wind_vector_definition"]["formula"] = "rmse(u)+rmse(v)"  # not locked
    errs = sfb.validate_schema(doc)
    assert any("box lat" in e for e in errs)
    assert any("sqrt" in e for e in errs)


def test_schema_rejects_g1_claimable_on_unscored():
    doc = sfb.build_schema_placeholder(_minimal_cfg())
    doc["g1_claimable"] = True
    errs = sfb.validate_schema(doc)
    assert any("g1_claimable" in e for e in errs)


def test_existing_final_baselines_json_schema_if_present():
    path = REPO / "runs/phase0/final_eval/final_baselines.json"
    if not path.is_file():
        pytest.skip("final_baselines.json not present")
    doc = json.loads(path.read_text())
    # Allow SCHEMA_PLACEHOLDER or similar; must still validate locked fields
    errs = sfb.validate_schema(doc)
    assert errs == [], errs


# ---------------------------------------------------------------------------
# Wind-vector: √(mean((u²+v²)/2)) grid-pool then lead-mean
# ---------------------------------------------------------------------------


def test_wind_vector_grid_pool_known_value():
    # u=3, v=4 everywhere → (9+16)/2 = 12.5 → sqrt(12.5)
    u = np.full((2, 3), 3.0)
    v = np.full((2, 3), 4.0)
    got = sfb.wind_vector_grid_pool(u, v)
    assert math.isclose(got, math.sqrt(12.5), rel_tol=1e-12)


def test_wind_vector_grid_pool_with_mask():
    u = np.array([1.0, 10.0, 0.0])
    v = np.array([0.0, 10.0, 0.0])
    mask = np.array([True, False, True])  # exclude the huge cell
    # valid: (1,0) and (0,0) → means of (1/2 + 0)/2 wait:
    # cell0: (1+0)/2 = 0.5; cell2: (0+0)/2 = 0; mean = 0.25; sqrt = 0.5
    got = sfb.wind_vector_grid_pool(u, v, mask=mask)
    assert math.isclose(got, 0.5, rel_tol=1e-12)


def test_wind_vector_matches_accumulate_style_sum():
    """Mirror living accumulate: wvec_sum/n then sqrt — same as mean then sqrt."""
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([1.0, 0.0, -1.0])
    wvec = ((u ** 2) + (v ** 2)) / 2.0
    via_sum = float(np.sqrt(wvec.sum() / len(wvec)))
    via_helper = sfb.wind_vector_grid_pool(u, v)
    assert math.isclose(via_helper, via_sum, rel_tol=1e-12)


def test_lead_mean_gate_leads():
    vals = {24: 1.0, 72: 2.0, 120: 3.0, 48: 99.0}  # 48 must be ignored for gate
    got = sfb.lead_mean(vals, leads=sfb.GATE_LEADS_DEFAULT)
    assert math.isclose(got, 2.0, rel_tol=1e-12)
    assert tuple(sfb.GATE_LEADS_DEFAULT) == (24, 72, 120)


def test_lead_mean_report_leads():
    vals = {24: 1.0, 48: 2.0, 72: 3.0, 96: 4.0, 120: 5.0}
    got = sfb.lead_mean(vals, leads=sfb.REPORT_LEADS_DEFAULT)
    assert math.isclose(got, 3.0, rel_tol=1e-12)
    assert tuple(sfb.REPORT_LEADS_DEFAULT) == (24, 48, 72, 96, 120)


def test_wind_vector_lead_mean_from_errors_pipeline():
    # Constant errors per lead → known per-lead WV, then lead-mean
    errors = {
        24: (np.full(4, 1.0), np.zeros(4)),   # (1+0)/2 = 0.5 → sqrt(0.5)
        72: (np.full(4, 2.0), np.zeros(4)),   # (4+0)/2 = 2 → sqrt(2)
        120: (np.full(4, 0.0), np.full(4, 2.0)),  # (0+4)/2 = 2 → sqrt(2)
        48: (np.full(4, 100.0), np.zeros(4)),  # ignored for gate
    }
    out = sfb.wind_vector_lead_mean_from_errors(errors, leads=(24, 72, 120))
    assert math.isclose(out["per_lead"]["24"], math.sqrt(0.5), rel_tol=1e-12)
    assert math.isclose(out["per_lead"]["72"], math.sqrt(2.0), rel_tol=1e-12)
    assert math.isclose(out["per_lead"]["120"], math.sqrt(2.0), rel_tol=1e-12)
    expected_lm = (math.sqrt(0.5) + math.sqrt(2.0) + math.sqrt(2.0)) / 3.0
    assert math.isclose(out["lead_mean"], expected_lm, rel_tol=1e-12)
    assert "48" not in out["per_lead"]


def test_schema_leads_match_locked_defaults():
    doc = sfb.build_schema_placeholder(_minimal_cfg())
    assert doc["leads"]["gate_h"] == [24, 72, 120]
    assert doc["leads"]["report_h"] == [24, 48, 72, 96, 120]
    # gate metric blocks keyed by gate leads
    for lh in ("24", "72", "120"):
        assert lh in doc["splits"]["val"]["raw_fcn3"]["per_lead"]
    # report sidecar has report leads
    for lh in ("24", "48", "72", "96", "120"):
        assert lh in doc["report_leads"]["val"]["raw_fcn3"]["per_lead"]
    # 48 not in gate split block
    assert "48" not in doc["splits"]["val"]["raw_fcn3"]["per_lead"]
    wv = doc["wind_vector_definition"]
    assert "sqrt" in wv["formula"]
    assert wv["gate_leads_h"] == [24, 72, 120]


# ---------------------------------------------------------------------------
# Thick-2 bridge stub smoke
# ---------------------------------------------------------------------------


def test_thick2_continuity_headlines_locked():
    """Living continuity floats must stay 1.770 / 1.775 / 1.982 (do not invent)."""
    c = t2b.LIVING_CONTINUITY
    assert abs(c["val_t2m_pooled"] - 1.770077) < 1e-9
    assert abs(c["test_t2m_pooled_16ic"] - 1.774864) < 1e-9
    assert abs(c["val_t2m_plus_120h"] - 1.982401) < 1e-9
    assert abs(c["val_wind_vector_lead_mean"] - 0.69560168) < 1e-9
    assert abs(c["test_wind_vector_lead_mean"] - 0.73876564) < 1e-9
    assert c["protocol"] == "thick2_12_16_16"


def test_thick2_bridge_dry_run_smoke(tmp_path):
    script = FINAL / "thick2_legacy_bridge.py"
    py = sys.executable
    r = subprocess.run(
        [py, str(script), "--dry-run"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert "dry-run OK" in r.stdout
    assert "not a G1" in r.stdout or "honesty" in r.stdout.lower()


def test_thick2_bridge_write_stub(tmp_path):
    out = tmp_path / "thick2_legacy_bridge.json"
    script = FINAL / "thick2_legacy_bridge.py"
    r = subprocess.run(
        [sys.executable, str(script), "--write-stub", "--out", str(out)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    doc = json.loads(out.read_text())
    assert doc["schema"] == "thick2_legacy_bridge/v1"
    assert doc["status"] == "STUB"
    assert doc["scored"] is False
    assert doc["not_promote_path"] is True
    ref = doc["living_continuity_reference"]
    assert abs(ref["val_t2m_pooled"] - 1.770077) < 1e-9
    assert "sqrt" in doc["wind_vector_definition"]
