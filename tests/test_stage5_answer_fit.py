"""
Phase 5 unit tests — Answer-fit scoring
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_schemas import Document, EDU, EDUPair, QUDQuestion
from stages.stage1_edu_segmentation import EDUSegmenter
from stages.stage1_5_pairing import EDUPairer
from stages.stage2_qud_generation import QUDGenerator
from stages.stage5_answer_fit import AnswerFitScorer, score_answer_fit


SAMPLE = (
    "Apple Inc. announced a new product yesterday. "
    "The device will ship in Q3. "
    "Analysts expect strong demand in the U.S. market. "
    "Natural language processing has made remarkable progress."
)


def _pipeline():
    doc = Document(doc_id="t5", text=SAMPLE, domain="news")
    edus = EDUSegmenter(backend="fallback").segment(doc)
    pairs, _, _ = EDUPairer(backend="tfidf", top_k=2, similarity_threshold=0.0).pair(edus)
    all_qs, _ = QUDGenerator(backend="template").generate(pairs, edus)
    valid = [q for q in all_qs if q.is_valid]
    return doc, edus, pairs, valid


def test_scorer_runs():
    doc, edus, pairs, questions = _pipeline()
    scorer = AnswerFitScorer()
    scores, stats = scorer.score(questions, edus, pairs=pairs)
    assert stats["num_paired"] >= 1
    assert stats["num_questions"] >= 1
    for s in scores:
        assert 0.0 <= s.score <= 1.0
        assert s.question_id
        assert s.answer_edu_id


def test_scores_vary():
    doc, edus, pairs, questions = _pipeline()
    scores, stats = AnswerFitScorer().score(questions[:20], edus, pairs=pairs)
    vals = [s.score for s in scores]
    assert max(vals) >= min(vals)


def test_save_json(tmp_path):
    doc, edus, pairs, questions = _pipeline()
    out = tmp_path / "af.json"
    scores, stats = AnswerFitScorer().score_and_save(
        questions[:10], edus, out, pairs=pairs, document=doc
    )
    assert out.exists()
    import json
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "scores" in data
    assert "ranked_paired" in data


def test_convenience():
    doc, edus, pairs, questions = _pipeline()
    scores, stats = score_answer_fit(questions[:5], edus, pairs=pairs)
    assert stats["num_paired"] >= 1


def test_reproducibility():
    doc, edus, pairs, questions = _pipeline()
    scorer = AnswerFitScorer()
    a, _ = scorer.score(questions[:8], edus, pairs=pairs)
    b, _ = scorer.score(questions[:8], edus, pairs=pairs)
    assert [s.score for s in a] == [s.score for s in b]


def test_alternatives():
    doc, edus, pairs, questions = _pipeline()
    scorer = AnswerFitScorer(top_k_alternatives=2)
    scores, stats = scorer.score(questions[:3], edus, pairs=pairs)
    roles = {s.metadata.get("role") for s in scores}
    assert "paired" in roles
    assert "alternative" in roles
