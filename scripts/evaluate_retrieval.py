"""
Run retrieval evaluation benchmarks.
Compares multiple retrieval configurations on the eval dataset.
Outputs JSON results and prints a summary table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from app.config.settings import settings
    from app.config.logging_config import setup_logging
    setup_logging("WARNING")  # quiet during benchmarks

    from app.api.dependencies import _init_components
    _init_components()

    from app.evaluation.dataset import load_eval_dataset
    from app.evaluation.benchmark import run_retrieval_benchmark, save_benchmark_results, print_summary_table
    from app.api.dependencies import get_rag_pipeline

    eval_path = PROJECT_ROOT / "data" / "evaluation" / "eval_dataset.json"
    if not eval_path.exists():
        print(f"Evaluation dataset not found at {eval_path}")
        print("Run: python scripts/create_demo_corpus.py")
        sys.exit(1)

    cases = load_eval_dataset(eval_path)
    if not cases:
        print("No evaluation cases loaded.")
        sys.exit(1)

    pipeline = get_rag_pipeline()

    configs = [
        ("dense", lambda q: pipeline.dense.retrieve(q, top_k=5)),
        ("sparse_bm25", lambda q: pipeline.sparse.retrieve(q, top_k=5)),
        ("hybrid_rrf", lambda q: pipeline.hybrid.retrieve(q, top_k=5)),
    ]

    results = []
    for name, fn in configs:
        print(f"Running benchmark: {name} ...")
        r = run_retrieval_benchmark(cases, fn, name, k=5)
        results.append(r)

    print("\n" + "=" * 70)
    print("RETRIEVAL BENCHMARK RESULTS")
    print("=" * 70)
    print_summary_table(results, k=5)

    output_path = PROJECT_ROOT / "data" / "evaluation" / "benchmark_results.json"
    save_benchmark_results(results, output_path)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
