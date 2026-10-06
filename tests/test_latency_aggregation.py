"""Tests for p50/p95 latency percentile and success-rate aggregation."""

from __future__ import annotations

import csv
from pathlib import Path

from evaluation.runner import aggregate_latencies, aggregate_success_rate, export_summary_to_csv


def test_aggregate_latencies_computes_p50_p95_per_group() -> None:
    rows = [
        {"variant": "retrieval", "latency_ms": float(value)}
        for value in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ]

    summaries = aggregate_latencies(rows, group_by=("variant",))

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["variant"] == "retrieval"
    assert summary["count"] == 10
    assert summary["p50_ms"] == 55.0
    assert summary["p95_ms"] == 95.5


def test_aggregate_latencies_groups_by_multiple_fields() -> None:
    rows = [
        {"model": "llama", "variant": "retrieval", "latency_ms": 100.0},
        {"model": "llama", "variant": "retrieval", "latency_ms": 200.0},
        {"model": "mistral", "variant": "retrieval", "latency_ms": 50.0},
    ]

    summaries = aggregate_latencies(rows, group_by=("model", "variant"))

    by_key = {(s["model"], s["variant"]): s for s in summaries}
    assert by_key[("llama", "retrieval")]["count"] == 2
    assert by_key[("llama", "retrieval")]["p50_ms"] == 150.0
    assert by_key[("mistral", "retrieval")]["count"] == 1
    assert by_key[("mistral", "retrieval")]["p50_ms"] == 50.0


def test_aggregate_latencies_handles_single_value_group() -> None:
    summaries = aggregate_latencies([{"variant": "hybrid", "latency_ms": 42.0}])

    assert summaries == [{"variant": "hybrid", "count": 1, "p50_ms": 42.0, "p95_ms": 42.0}]


def test_export_summary_to_csv_writes_one_row_per_group(tmp_path: Path) -> None:
    summaries = aggregate_latencies(
        [
            {"model": "llama", "variant": "retrieval", "latency_ms": 100.0},
            {"model": "mistral", "variant": "retrieval", "latency_ms": 50.0},
        ],
        group_by=("model", "variant"),
    )

    output_path = tmp_path / "latency_summary.csv"
    export_summary_to_csv(summaries, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert {row["model"] for row in rows} == {"llama", "mistral"}
    assert rows[0]["p50_ms"] in {"100.0", "50.0"}


def test_aggregate_success_rate_counts_failures_and_empty_responses() -> None:
    rows = [
        {"variant": "retrieval", "response_text": "ok", "failure": None},
        {"variant": "retrieval", "response_text": "ok", "failure": None},
        {"variant": "retrieval", "response_text": "", "failure": None},
        {"variant": "retrieval", "response_text": "ok", "failure": "RuntimeError: boom"},
    ]

    summaries = aggregate_success_rate(rows, group_by=("variant",))

    assert summaries == [{"variant": "retrieval", "count": 4, "success_rate": 0.5}]


def test_aggregate_success_rate_groups_by_multiple_fields() -> None:
    rows = [
        {"model": "llama", "variant": "retrieval", "response_text": "ok", "failure": None},
        {"model": "mistral", "variant": "retrieval", "response_text": "", "failure": None},
    ]

    summaries = aggregate_success_rate(rows, group_by=("model", "variant"))

    by_key = {(s["model"], s["variant"]): s for s in summaries}
    assert by_key[("llama", "retrieval")]["success_rate"] == 1.0
    assert by_key[("mistral", "retrieval")]["success_rate"] == 0.0


def test_export_summary_to_csv_works_for_success_rate_summaries(tmp_path: Path) -> None:
    summaries = aggregate_success_rate(
        [{"variant": "hybrid", "response_text": "ok", "failure": None}],
        group_by=("variant",),
    )

    output_path = tmp_path / "success_rate.csv"
    export_summary_to_csv(summaries, output_path)

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [{"variant": "hybrid", "count": "1", "success_rate": "1.0"}]
