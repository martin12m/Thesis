"""Tests for the scripts/run_evaluation.py CLI entrypoint."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

import scripts.run_evaluation as run_evaluation_script
from llm.interface import TextGenerator


class _StubGenerator(TextGenerator):
    def __init__(self, label: str) -> None:
        self.label = label

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        return f"Answer from {self.label}."


def _write_scenarios(tmp_path: Path) -> Path:
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps(
            [
                {
                    "id": "shipping-1",
                    "user_turn": "What is your return policy?",
                    "expected_answer_contains": ["Answer from"],
                }
            ]
        ),
        encoding="utf-8",
    )
    return scenarios_path


@pytest.fixture(autouse=True)
def _stub_generators(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_default_generator(env: dict[str, str] | None = None) -> TextGenerator:
        variant = (env or {}).get("LLM_MODEL_VARIANT", "llama")
        return _StubGenerator(variant)

    monkeypatch.setattr(run_evaluation_script, "build_default_generator", fake_build_default_generator)
    monkeypatch.setattr(run_evaluation_script, "list_configured_profiles", lambda: [])


def _run_cli(args: list[str]) -> int:
    old_argv = sys.argv
    sys.argv = ["run_evaluation.py", *args]
    try:
        return run_evaluation_script.main()
    finally:
        sys.argv = old_argv


class _StubSimulateAndAnswerGenerator(TextGenerator):
    """Returns a canned simulated-scenario JSON array for the simulation prompt, and a canned answer for ordinary reasoning prompts -- so one stub can serve both roles ``--simulate`` asks of the same generator."""

    def __init__(self, label: str) -> None:
        self.label = label

    def generate(self, prompt: str) -> str:
        if '"user_turn"' in prompt:
            return json.dumps(
                [{"category": "catalog_context", "user_turn": "What's the price of product 1?"}]
            )
        return f"Answer from {self.label}."


def test_cross_judge_requires_both_models(tmp_path: Path) -> None:
    scenarios_path = _write_scenarios(tmp_path)
    output_path = tmp_path / "out.csv"

    with pytest.raises(SystemExit, match="requires 2 or more --model"):
        _run_cli(
            [
                "--scenarios",
                str(scenarios_path),
                "--output",
                str(output_path),
                "--model",
                "llama",
                "--judge",
                "cross",
            ]
        )


def test_single_model_run_writes_results_and_latency_summary(tmp_path: Path) -> None:
    scenarios_path = _write_scenarios(tmp_path)
    output_path = tmp_path / "out.csv"

    exit_code = _run_cli(
        ["--scenarios", str(scenarios_path), "--output", str(output_path), "--model", "llama"]
    )

    assert exit_code == 0
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3  # one row per variant (retrieval, tool_first, hybrid)
    assert "model" not in rows[0]  # single-model runs omit the column entirely
    assert all("Answer from llama" in row["response_text"] for row in rows)

    latency_path = tmp_path / "out_latency_summary.csv"
    assert latency_path.exists()
    with latency_path.open("r", encoding="utf-8", newline="") as handle:
        latency_rows = list(csv.DictReader(handle))
    assert {row["variant"] for row in latency_rows} == {"retrieval", "tool_first", "hybrid"}

    success_path = tmp_path / "out_success_rate.csv"
    assert success_path.exists()
    with success_path.open("r", encoding="utf-8", newline="") as handle:
        success_rows = list(csv.DictReader(handle))
    assert {row["variant"] for row in success_rows} == {"retrieval", "tool_first", "hybrid"}
    assert all(row["success_rate"] == "1.0" for row in success_rows)


def test_both_models_with_cross_judge(tmp_path: Path) -> None:
    scenarios_path = _write_scenarios(tmp_path)
    output_path = tmp_path / "out.csv"

    exit_code = _run_cli(
        [
            "--scenarios",
            str(scenarios_path),
            "--output",
            str(output_path),
            "--model",
            "both",
            "--judge",
            "cross",
        ]
    )

    assert exit_code == 0
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6  # 3 variants x 2 models
    assert {row["model"] for row in rows} == {"llama", "mistral"}
    for row in rows:
        assert row["judge_error"] != ""
        assert row["judge_groundedness"] == ""

    latency_path = tmp_path / "out_latency_summary.csv"
    with latency_path.open("r", encoding="utf-8", newline="") as handle:
        latency_rows = list(csv.DictReader(handle))
    assert {(row["model"], row["variant"]) for row in latency_rows} == {
        (model, variant)
        for model in ("llama", "mistral")
        for variant in ("retrieval", "tool_first", "hybrid")
    }


def test_simulate_flag_adds_generated_scenarios(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_build_default_generator(env: dict[str, str] | None = None) -> TextGenerator:
        variant = (env or {}).get("LLM_MODEL_VARIANT", "llama")
        return _StubSimulateAndAnswerGenerator(variant)

    monkeypatch.setattr(run_evaluation_script, "build_default_generator", fake_build_default_generator)

    scenarios_path = _write_scenarios(tmp_path)  # 1 static scenario
    output_path = tmp_path / "out.csv"

    exit_code = _run_cli(
        [
            "--scenarios",
            str(scenarios_path),
            "--output",
            str(output_path),
            "--model",
            "llama",
            "--simulate",
            "1",
        ]
    )

    assert exit_code == 0
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    scenario_ids = {row["scenario_id"] for row in rows}
    assert "shipping-1" in scenario_ids
    assert any(scenario_id.startswith("sim-") for scenario_id in scenario_ids)

    simulated_path = tmp_path / "out_simulated_scenarios.json"
    assert simulated_path.exists()
    saved_scenarios = json.loads(simulated_path.read_text(encoding="utf-8"))
    assert len(saved_scenarios) == 1
    assert saved_scenarios[0]["user_turn"] == "What's the price of product 1?"
    assert saved_scenarios[0]["id"] in scenario_ids
