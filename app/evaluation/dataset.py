"""
Evaluation dataset management.
Each evaluation case has a question, expected sources, and expected answer characteristics.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalCase:
    """A single evaluation case."""
    case_id: str
    question: str
    expected_sources: list[str]          # Expected source file names
    expected_pages: list[int]            # Expected page numbers (if known)
    required_evidence: str              # What the answer must include
    answer_type: str                    # "text" | "table" | "image" | "cross_modal"
    difficulty: str                     # "easy" | "medium" | "hard"
    reference_answer: Optional[str] = None
    notes: str = ""


def load_eval_dataset(path: str | Path) -> list[EvalCase]:
    """Load evaluation cases from a JSON file."""
    p = Path(path)
    if not p.exists():
        logger.warning(f"Eval dataset not found at {p}")
        return []
    cases = []
    raw = json.loads(p.read_text(encoding="utf-8"))
    for item in raw:
        cases.append(EvalCase(
            case_id=item["case_id"],
            question=item["question"],
            expected_sources=item.get("expected_sources", []),
            expected_pages=item.get("expected_pages", []),
            required_evidence=item.get("required_evidence", ""),
            answer_type=item.get("answer_type", "text"),
            difficulty=item.get("difficulty", "medium"),
            reference_answer=item.get("reference_answer"),
            notes=item.get("notes", ""),
        ))
    logger.info(f"Loaded {len(cases)} evaluation cases from {p}")
    return cases


def save_eval_dataset(cases: list[EvalCase], path: str | Path) -> None:
    """Save evaluation cases to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "case_id": c.case_id,
            "question": c.question,
            "expected_sources": c.expected_sources,
            "expected_pages": c.expected_pages,
            "required_evidence": c.required_evidence,
            "answer_type": c.answer_type,
            "difficulty": c.difficulty,
            "reference_answer": c.reference_answer,
            "notes": c.notes,
        }
        for c in cases
    ]
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"Saved {len(cases)} evaluation cases to {p}")
