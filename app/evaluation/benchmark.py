"""
Benchmark runner: evaluates retrieval configurations against the eval dataset.
Produces machine-readable results (JSON) and a summary table.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .dataset import EvalCase
from .retrieval_metrics import compute_retrieval_metrics, aggregate_metrics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results for one retrieval configuration."""
    config_name: str
    cases_run: int
    per_case_metrics: list[dict[str, Any]] = field(default_factory=list)
    aggregated: dict[str, float] = field(default_factory=dict)
    total_latency_s: float = 0.0
    errors: list[str] = field(default_factory=list)


def run_retrieval_benchmark(
    eval_cases: list[EvalCase],
    retrieve_fn: Callable[[str], list[dict[str, Any]]],
    config_name: str,
    k: int = 5,
) -> BenchmarkResult:
    """
    Run retrieval evaluation for a single configuration.

    Args:
        eval_cases: Evaluation cases with expected sources.
        retrieve_fn: Callable(query) -> list of result dicts.
        config_name: Label for this configuration.
        k: Metric cutoff.

    Returns:
        BenchmarkResult with per-case and aggregated metrics.
    """
    result = BenchmarkResult(config_name=config_name, cases_run=0)
    t_total = time.time()

    for case in eval_cases:
        try:
            results = retrieve_fn(case.question)
            metrics = compute_retrieval_metrics(results, case.expected_sources, k=k)
            metrics["case_id"] = case.case_id
            metrics["difficulty"] = case.difficulty
            metrics["answer_type"] = case.answer_type
            result.per_case_metrics.append(metrics)
            result.cases_run += 1
        except Exception as e:
            logger.error(f"Benchmark error on case {case.case_id}: {e}")
            result.errors.append(f"{case.case_id}: {e}")

    result.total_latency_s = time.time() - t_total
    result.aggregated = aggregate_metrics(
        [{k: v for k, v in m.items() if isinstance(v, (int, float))}
         for m in result.per_case_metrics]
    )
    logger.info(
        f"Benchmark '{config_name}': {result.cases_run} cases, "
        f"MRR={result.aggregated.get('mrr', 0):.3f}, "
        f"Hit@{k}={result.aggregated.get(f'hit@{k}', 0):.3f}"
    )
    return result


def save_benchmark_results(
    results: list[BenchmarkResult],
    output_path: str | Path,
) -> None:
    """Save benchmark results to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "config_name": r.config_name,
            "cases_run": r.cases_run,
            "aggregated_metrics": r.aggregated,
            "total_latency_s": r.total_latency_s,
            "errors": r.errors,
            "per_case_metrics": r.per_case_metrics,
        }
        for r in results
    ]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"Saved benchmark results to {path}")


def print_summary_table(results: list[BenchmarkResult], k: int = 5) -> str:
    """Print a formatted summary table of benchmark results."""
    header = f"{'Config':<35} {'MRR':>7} {'Hit@'+str(k):>8} {'Recall@'+str(k):>10} {'nDCG@'+str(k):>8} {'Cases':>6}"
    separator = "-" * len(header)
    lines = [separator, header, separator]
    for r in results:
        ag = r.aggregated
        lines.append(
            f"{r.config_name:<35} "
            f"{ag.get('mrr', 0):>7.3f} "
            f"{ag.get(f'hit@{k}', 0):>8.3f} "
            f"{ag.get(f'recall@{k}', 0):>10.3f} "
            f"{ag.get(f'ndcg@{k}', 0):>8.3f} "
            f"{r.cases_run:>6}"
        )
    lines.append(separator)
    table = "\n".join(lines)
    print(table)
    return table
