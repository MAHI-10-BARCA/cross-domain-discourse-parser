"""Phase 0 schema validation tests."""

import sys
from pathlib import Path

# Allow imports from src
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_schemas import (
    Document, EDU, EDUPair, QUDQuestion,
    SalienceScore, RelationPrediction, AnswerFitScore,
    DiscourseNode, DiscourseTree, save_json, load_json
)
import tempfile
import os


def test_document_roundtrip():
    doc = Document(doc_id="d1", text="Hello world.", domain="news")
    d = doc.to_dict()
    doc2 = Document.from_dict(d)
    assert doc2.doc_id == "d1"
    assert doc2.text == "Hello world."
    assert doc2.domain == "news"


def test_edu_roundtrip():
    edu = EDU(edu_id="d1_edu_000", doc_id="d1", text="Hello world.", index=0, start_char=0, end_char=12)
    d = edu.to_dict()
    edu2 = EDU.from_dict(d)
    assert edu2.edu_id == "d1_edu_000"
    assert edu2.index == 0


def test_edupair_roundtrip():
    pair = EDUPair(
        pair_id="p1", doc_id="d1",
        edu_i_id="d1_edu_000", edu_j_id="d1_edu_001",
        edu_i_index=0, edu_j_index=1, distance=1, similarity=0.85
    )
    d = pair.to_dict()
    pair2 = EDUPair.from_dict(d)
    assert pair2.similarity == 0.85
    assert pair2.selected is True


def test_qud_roundtrip():
    q = QUDQuestion(
        question_id="q1", pair_id="p1", doc_id="d1",
        question_text="What happened?", category="what"
    )
    d = q.to_dict()
    q2 = QUDQuestion.from_dict(d)
    assert q2.question_text == "What happened?"
    assert q2.is_valid is True


def test_discourse_tree_roundtrip():
    leaf1 = DiscourseNode(node_id="n0", span_start=0, span_end=0, edu_ids=["e0"])
    leaf2 = DiscourseNode(node_id="n1", span_start=1, span_end=1, edu_ids=["e1"])
    root = DiscourseNode(
        node_id="n_root", span_start=0, span_end=1,
        relation="Elaboration", confidence=0.7,
        children=[leaf1, leaf2], edu_ids=["e0", "e1"]
    )
    tree = DiscourseTree(doc_id="d1", root=root)
    d = tree.to_dict()
    tree2 = DiscourseTree.from_dict(d)
    assert tree2.root.relation == "Elaboration"
    assert len(tree2.root.children) == 2
    assert tree2.root.children[0].span_start == 0


def test_save_load_json():
    doc = Document(doc_id="d1", text="Test.")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "doc.json")
        save_json(doc, path)
        loaded = load_json(path)
        assert loaded["doc_id"] == "d1"
        assert loaded["text"] == "Test."


def test_seed_import():
    from utils.seed import set_seed
    set_seed(42)
    # just ensure it runs without error
