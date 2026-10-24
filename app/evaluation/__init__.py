"""Evaluation modules for the multimodal RAG knowledge assistant."""

from .dataset import EvalCase, load_eval_dataset, save_eval_dataset
from .retrieval_metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    compute_retrieval_metrics,
    aggregate_metrics,
)
from .answer_metrics import (
    citation_precision,
    answer_contains_evidence,
    faithfulness_heuristic,
    compute_answer_metrics,
)
from .benchmark import BenchmarkResult, run_retrieval_benchmark, save_benchmark_results, print_summary_table

__all__ = [
    "EvalCase",
    "load_eval_dataset",
    "save_eval_dataset",
    "hit_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "compute_retrieval_metrics",
    "aggregate_metrics",
    "citation_precision",
    "answer_contains_evidence",
    "faithfulness_heuristic",
    "compute_answer_metrics",
    "BenchmarkResult",
    "run_retrieval_benchmark",
    "save_benchmark_results",
    "print_summary_table",
]
