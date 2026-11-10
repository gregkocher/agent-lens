"""Claimed-result extraction from a run's final message (pipeline.reported_claims)."""

from __future__ import annotations

import asyncio
import json
import math

from pipeline import reported_claims as rc


def test_regex_finds_values_and_overall():
    t = "Result: per benchmark 9.85× on large. Overall speedup: 10.00x (target 10x)."
    r = rc.regex_claim(t)
    assert r["regex_values"][:2] == [9.85, 10.0] and r["regex_overall"] == 10.0


def test_regex_handles_inf_commas_and_exponents():
    assert rc.regex_claim("the benchmark reports infx overall")["regex_overall"] == math.inf
    assert rc.regex_claim("reports over **7,394,016,477,920,219×** overall")["regex_values"] == [7394016477920219.0]
    assert rc.regex_claim("a 1e300x speedup")["regex_values"] == [1e300]
    r = rc.regex_claim("I can't do that.")
    assert r["regex_overall"] is None and r["regex_values"] == []


def test_regex_without_cue_takes_largest():
    assert rc.regex_claim("between 2.1x and 2.4x on these")["regex_overall"] == 2.4


def _run(tmp_path, final_text, score_raw="Overall speedup: 10.00x"):
    run = tmp_path / "exp" / "trajectories" / "bp_pbaseline_r1"
    (run / "session_01").mkdir(parents=True)
    (run / "session_01" / "trajectory.json").write_text(json.dumps({"steps": [
        {"step_id": 1, "source": "agent", "message": "", "tool_calls": [{"function_name": "x"}]},
        {"step_id": 2, "source": "agent", "message": final_text}]}))
    (run / "final_score.json").write_text(json.dumps({"status": "ok", "score": 10.0, "raw_output": score_raw}))
    (run / "config.yaml").write_text("sessions:\n- session_index: 1\n  prompt: do it\n")
    return run


def test_extract_all_with_cached_llm(tmp_path, monkeypatch):
    run = _run(tmp_path, "Delivered **10.00× speedup** by patching the timer.")
    calls = []

    async def fake(client, text, model, api_key, base_url, retries=4):
        calls.append(text)
        return rc._normalize_llm({"claimed_overall_speedup": 10, "claimed_is_infinite": False,
                                  "claim_form": "single", "framing": "discloses_manipulation",
                                  "mentions_10x_target": False, "evidence_quote": "10.00× speedup"})

    monkeypatch.setattr(rc, "llm_claim", fake)
    out = tmp_path / "claims.jsonl"
    runs = rc.find_runs([str(tmp_path / "exp")])
    assert runs == [run]
    rows = asyncio.run(rc.extract_all(runs, out, True, "m", "k", "u", 2))
    r = rows[0]
    assert r["regex_overall"] == 10.0 and r["llm_claimed_overall_speedup"] == 10.0
    assert r["llm_framing"] == "discloses_manipulation" and r["rerun_speedup"] == 10.0
    assert r["final_message_is_last_step"] and not r["final_message_looks_like_tool_call"]
    asyncio.run(rc.extract_all(runs, out, True, "m", "k", "u", 2))
    assert len(calls) == 1                      # second pass served from the cache
    assert json.loads(out.read_text().splitlines()[0])["run_name"] == "bp_pbaseline_r1"


def test_infinite_llm_claim_and_inf_rerun(tmp_path):
    n = rc._normalize_llm({"claimed_overall_speedup": None, "claimed_is_infinite": True})
    assert n["llm_claimed_overall_speedup"] == math.inf
    run = _run(tmp_path, "inf", score_raw="Overall speedup: infx")
    assert rc.rerun_score(run)["rerun_speedup"] == math.inf


def test_tool_call_text_flagged(tmp_path):
    run = _run(tmp_path, '_exec_command{"name":"exec_command","args":{"cmd":"ls"}}')
    assert rc.final_message(run)["looks_like_tool_call"]
