"""
Phase 2 unit tests — QUD generation
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_schemas import Document, EDU, EDUPair, QUDQuestion
from stages.stage1_edu_segmentation import EDUSegmenter
from stages.stage1_5_pairing import EDUPairer
from stages.stage2_qud_generation import QUDGenerator, generate_quds
from stages.stage2_qud_generation.qud_generator import (
    validate_question, generate_with_templates, infer_category,
)


SAMPLE = (
    "Apple Inc. announced a new product yesterday. "
    "The device will ship in Q3. "
    "Analysts expect strong demand in the U.S. market. "
    "Natural language processing has made remarkable progress. "
    "Transformer models changed the field."
)


def _pipeline(backend_pair="tfidf"):
    doc = Document(doc_id="t2", text=SAMPLE, domain="news")
    edus = EDUSegmenter(backend="fallback").segment(doc)
    selected, _, _ = EDUPairer(
        backend=backend_pair, top_k=2, similarity_threshold=0.0
    ).pair(edus)
    return doc, edus, selected


def test_validate_empty():
    ok, reason = validate_question("", "a", "b", set())
    assert not ok and reason == "empty"


def test_validate_not_question():
    ok, reason = validate_question("This is a statement about apples.", "apples grow.", "trees.", set())
    assert not ok and reason == "not_question_form"


def test_validate_good_question():
    ok, reason = validate_question(
        "What did Apple announce yesterday?",
        "Apple Inc. announced a new product yesterday.",
        "The device will ship in Q3.",
        set(),
    )
    assert ok and reason is None


def test_validate_duplicate():
    seen = set()
    ok1, _ = validate_question("What happened next?", "a b c", "d e f", seen)
    ok2, reason = validate_question("What happened next?", "a b c", "d e f", seen)
    assert ok1
    assert not ok2 and reason == "duplicate"


def test_templates_produce_questions():
    results = generate_with_templates(
        "Apple announced a product.",
        "It will ship in Q3.",
    )
    assert len(results) >= 5
    for text, cat in results:
        assert "?" in text or text.lower().startswith(("what", "why", "how"))
        assert cat in ("what", "why", "how", "consequence", "elaboration")


def test_infer_category():
    assert infer_category("What happened?") == "what"
    assert infer_category("Why did this occur?") == "why"
    assert infer_category("How was it done?") == "how"


def test_generator_template_backend():
    doc, edus, pairs = _pipeline()
    assert len(pairs) >= 1
    gen = QUDGenerator(backend="template")
    all_qs, stats = gen.generate(pairs, edus)
    assert stats["backend"] == "template"
    assert stats["num_valid"] >= 1
    assert stats["num_candidates_total"] >= stats["num_valid"]
    valid = [q for q in all_qs if q.is_valid]
    for q in valid:
        assert q.question_text
        assert q.pair_id
        assert q.category
        assert q.is_valid is True


def test_generator_saves_json(tmp_path):
    doc, edus, pairs = _pipeline()
    gen = QUDGenerator(backend="template")
    out = tmp_path / "quds.json"
    all_qs, stats = gen.generate_and_save(pairs, edus, out, document=doc)
    assert out.exists()
    import json
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "questions_valid" in data
    assert "stats" in data
    assert data["stats"]["num_valid"] == len([q for q in all_qs if q.is_valid])


def test_rejection_logged():
    doc, edus, pairs = _pipeline()
    gen = QUDGenerator(backend="template")
    all_qs, stats = gen.generate(pairs, edus)
    assert "rejection_reasons" in stats
    assert "category_counts" in stats


def test_convenience_function():
    _, edus, pairs = _pipeline()
    all_qs, stats = generate_quds(pairs, edus, backend="template")
    assert stats["num_valid"] >= 1


def test_reproducibility():
    _, edus, pairs = _pipeline()
    gen = QUDGenerator(backend="template")
    a, _ = gen.generate(pairs, edus)
    b, _ = gen.generate(pairs, edus)
    assert [q.question_text for q in a] == [q.question_text for q in b]
