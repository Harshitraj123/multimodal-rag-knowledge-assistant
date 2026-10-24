"""Unit tests for answer quality metrics."""
import unittest
try:
    import pytest
except ImportError:
    pytest = None
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.evaluation.answer_metrics import (
    citation_precision, answer_contains_evidence, faithfulness_heuristic
)


class TestCitationPrecision(unittest.TestCase):
    def test_all_valid(self):
        assert citation_precision("Answer [1] and [2].", {1, 2}) == 1.0

    def test_all_invalid(self):
        assert citation_precision("Answer [5].", {1, 2}) == 0.0

    def test_no_citations(self):
        assert citation_precision("No citations.", {1, 2}) == 1.0

    def test_mixed(self):
        score = citation_precision("Check [1] and [5].", {1, 2})
        assert score == 0.5


class TestAnswerContainsEvidence(unittest.TestCase):
    def test_evidence_present(self):
        assert answer_contains_evidence("The value is 42.", "42") is True

    def test_evidence_absent(self):
        assert answer_contains_evidence("The value is 42.", "100") is False

    def test_empty_evidence(self):
        assert answer_contains_evidence("Any answer.", "") is True


class TestFaithfulnessHeuristic(unittest.TestCase):
    def test_fully_grounded(self):
        retrieved = ["The experiment showed that water boils at 100 degrees Celsius."]
        answer = "Water boils at 100 degrees Celsius according to the experiment."
        score = faithfulness_heuristic(answer, retrieved)
        assert score > 0.0

    def test_empty_inputs(self):
        assert faithfulness_heuristic("", ["text"]) == 0.0
        assert faithfulness_heuristic("answer", []) == 0.0
