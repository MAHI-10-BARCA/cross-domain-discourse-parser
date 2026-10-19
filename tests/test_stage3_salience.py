"""
Phase 3 unit tests — Salience scoring
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_schemas import Document, EDU, EDUPair, QUDQuestion, SalienceScore
from stages.stage1_edu_segmentation import EDUSegmenter
from stages.stage1_5_pairing import EDUPairer
from stages.stage2_qud_generation import QUDGenerator
from stages.stage3_salience import SalienceScorer, score_salience
from stages.stage3_salience.salience import (
    _cosine, _to_unit_interval, answerability_cosine, _embed_texts,
)


SAMPLE = (
    "Apple Inc. announced a new product yesterday. "
    "The device will ship in Q3. "
    "Analysts expect strong demand in the U.S. market. "
    "Natural language processing has made remarkable progress."
)


def _pipeline():
    doc = Document(doc_id="t3", text=SAMPLE, domain="news")
    edus = EDUSegmenter(backend="fallback").segment(doc)
    pairs, _, _ = EDUPairer(backend="tfidf", top_k=2, similarity_threshold=0.0).pair(edus)
    all_qs, _ = QUDGenerator(backend="template").generate(pairs, edus)
    valid = [q for q in all_qs if q.is_valid]
    return doc, edus, pairs, valid


def test_cosine_basic():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.0, 1.0])
    assert abs(_cosine(a, b) - 1.0) < 1e-5
    assert abs(_cosine(a, c)) < 1e-5


def test_to_unit_interval():
    assert _to_unit_interval(1.0) == 1.0
    assert _to_unit_interval(-1.0) == 0.0
    assert abs(_to_unit_interval(0.0) - 0.5) < 1e-6


def test_answerability_cosine():
    emb = _embed_texts(["hello world", "hello there"], "tfidf-dummy")
    score = answerability_cosine(emb[0], emb[1])
    assert 0.0 <= score <= 1.0


def test_scorer_runs():
    doc, edus, pairs, questions = _pipeline()
    assert len(questions) >= 1
    scorer = SalienceScorer(use_nli=False)
    scores, stats = scorer.score(questions, edus, pairs=pairs)
    assert stats["num_scored"] >= 1
    assert len(scores) == len(questions)
    for s in scores:
        if s.metadata.get("skipped"):
            continue
        assert 0.0 <= s.centrality <= 1.0
        assert 0.0 <= s.answerability <= 1.0
        assert 0.0 <= s.final_score <= 1.0


def test_weights_affect_score():
    doc, edus, pairs, questions = _pipeline()
    q = questions[:5]
    s1, _ = SalienceScorer(use_nli=False, centrality_weight=1.0, answerability_weight=0.0).score(q, edus, pairs)
    s2, _ = SalienceScorer(use_nli=False, centrality_weight=0.0, answerability_weight=1.0).score(q, edus, pairs)
    assert len(s1) == len(s2)
    for a, b in zip(s1, s2):
        if a.metadata.get("skipped"):
            continue
        assert abs(a.final_score - a.centrality) < 1e-5
        assert abs(b.final_score - b.answerability) < 1e-5


def test_invalid_question_zero():
    doc, edus, pairs, questions = _pipeline()
    bad = QUDQuestion(
        question_id="bad_q",
        pair_id=pairs[0].pair_id if pairs else "p0",
        doc_id="t3",
        question_text="",
        is_valid=False,
        rejection_reason="empty",
    )
    scorer = SalienceScorer(use_nli=False)
    scores, stats = scorer.score([bad], edus, pairs=pairs)
    assert scores[0].final_score == 0.0
    assert scores[0].metadata.get("skipped") is True


def test_save_json(tmp_path):
    doc, edus, pairs, questions = _pipeline()
    scorer = SalienceScorer(use_nli=False)
    out = tmp_path / "sal.json"
    scores, stats = scorer.score_and_save(questions[:10], edus, out, pairs=pairs, document=doc)
    assert out.exists()
    import json
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "scores" in data
    assert "ranked_top" in data
    assert "stats" in data


def test_reproducibility():
    doc, edus, pairs, questions = _pipeline()
    scorer = SalienceScorer(use_nli=False)
    a, _ = scorer.score(questions[:8], edus, pairs=pairs)
    b, _ = scorer.score(questions[:8], edus, pairs=pairs)
    assert [s.final_score for s in a] == [s.final_score for s in b]


def test_convenience():
    doc, edus, pairs, questions = _pipeline()
    scores, stats = score_salience(questions[:5], edus, pairs=pairs)
    assert stats["num_scored"] >= 1
