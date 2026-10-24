"""
Quick demo query against the indexed corpus.
Usage: python scripts/query_demo.py "Your question here"
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What embedding model is used and what is its dimension?"

    from app.config.settings import settings
    from app.config.logging_config import setup_logging
    setup_logging("WARNING")

    from app.api.dependencies import _init_components, get_rag_pipeline
    _init_components()
    pipeline = get_rag_pipeline()

    print(f"\nQuestion: {question}")
    print("-" * 60)
    response = pipeline.query(question, retrieval_method="hybrid")
    print(f"\nAnswer:\n{response.answer}")
    print(f"\nRetrieval: {len(response.retrieval_results)} results "
          f"in {response.retrieval_latency_s:.3f}s")
    print(f"Total latency: {response.total_latency_s:.3f}s")


if __name__ == "__main__":
    main()
