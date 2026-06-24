"""Pressure-generality wiring tests: how the sweep applies the swept variable to a
RunConfig (per engine) and records it in the manifest. Pure-logic, no runs / no network.
"""

from __future__ import annotations

import json

import pytest

from harness.config import RunConfig
from pipeline.config import SweepConfig
from pipeline.run_trajectories import _build_run_config, _manifest_row


def _sweep(pressure, **kw) -> SweepConfig:
    base = dict(experiment_name="t", base_task_config="x", base_work_dir="y",
                output_dir="z", pressure=pressure, judge={})
    base.update(kw)
    return SweepConfig(**base)


def _base_runconfig(**kw) -> RunConfig:
    base = dict(model="claude-sonnet-4-20250514", work_dir=".",
                sessions=[{"session_index": 1, "prompt": "do x"}])
    base.update(kw)
    return RunConfig(**base)


# --------------------------------------------------------------------------- _build_run_config
def test_build_run_config_budget_sweep_sets_max_budget(tmp_path):
    sweep = _sweep({"variable": "budget_usd", "values": [0.05]})
    rc = _build_run_config(sweep, _base_runconfig(), 0.05, 1, tmp_path / "wd")
    assert rc.max_budget_usd == 0.05
    assert rc.run_name == "bp_b0p05_r1"          # budget run-name byte-identical (cache-safe)


def test_build_run_config_turn_sweep_sets_max_turns(tmp_path):
    sweep = _sweep({"variable": "max_turns", "values": [10]})
    rc = _build_run_config(sweep, _base_runconfig(), 10, 1, tmp_path / "wd")
    assert rc.max_turns == 10
    assert rc.run_name == "bp_t10_r1"            # turn axis uses the distinct 't' tag


def test_build_run_config_codex_bad_provider_override_raises(tmp_path):
    # codex base is valid (openai); the agent_provider override makes it invalid, and the
    # re-check (model_copy bypasses RunConfig validation) must catch it.
    sweep = _sweep({"variable": "max_turns", "values": [10]}, agent_provider="anthropic")
    base = _base_runconfig(engine="codex", model="gpt-5.4", provider="openai")
    with pytest.raises(ValueError):
        _build_run_config(sweep, base, 10, 1, tmp_path / "wd")


# --------------------------------------------------------------------------- _manifest_row
def test_manifest_row_turn_sweep_columns(tmp_path):
    sweep = _sweep({"variable": "max_turns", "values": [10]})
    rd = tmp_path / "bp_t10_r1"
    rd.mkdir()
    (rd / "run_meta.json").write_text(json.dumps(
        {"engine": "codex", "total_cost_usd": None, "total_steps": 7,
         "sessions": [{"num_turns": 9}]}))
    row = _manifest_row(sweep, 10, 1, "bp_t10_r1", rd, "ok", None)
    assert row["pressure_variable"] == "max_turns"
    assert row["pressure_value"] == 10
    assert row["budget_usd"] is None             # no budget semantics for a turn sweep
    assert row["engine"] == "codex"
    assert row["num_turns"] == 9 and row["steps"] == 7


def test_manifest_row_budget_sweep_aliases_budget_usd(tmp_path):
    sweep = _sweep({"variable": "budget_usd", "values": [0.05]})
    rd = tmp_path / "bp_b0p05_r1"
    rd.mkdir()
    (rd / "run_meta.json").write_text(json.dumps(
        {"engine": "claude_code", "total_cost_usd": 0.04, "sessions": [{"num_turns": 5}]}))
    row = _manifest_row(sweep, 0.05, 1, "bp_b0p05_r1", rd, "ok", None)
    assert row["pressure_value"] == 0.05
    assert row["budget_usd"] == 0.05             # alias = value so budget-fraction code works
    assert row["engine"] == "claude_code"
    assert row["cost_usd"] == 0.04
