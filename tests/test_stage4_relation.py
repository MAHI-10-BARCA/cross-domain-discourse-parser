"""
Phase 4 unit tests — Relation classification
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_schemas import Document, EDU, EDUPair
from stages.stage1_edu_segmentation import EDUSegmenter
from stages.stage1_5_pairing import EDUPairer
from stages.stage4_relation import RelationClassifier, classify_relations
from stages.stage4_relation.relation_classifier import (
    predict_relation_heuristic, RELATION_LABELS, _cue_scores,
)
import numpy as np


SAMPLE = (
    "Apple Inc. announced a new product yesterday. "
    "However, supply constraints may limit availability. "
    "The device will ship in Q3 because demand is high. "
    "For example, pre-orders already exceeded forecasts. "
    "Natural language processing has made remarkable progress."
)


def _pipeline():
    doc = Document(doc_id="t4", text=SAMPLE, domain="news")
    edus = EDUSegmenter(backend="fallback").segment(doc)
    pairs, _, _ = EDUPairer(backend="tfidf", top_k=3, similarity_threshold=0.0).pair(edus)
    return doc, edus, pairs


def test_labels_inventory():
    assert "Elaboration" in RELATION_LABELS
    assert "Contrast" in RELATION_LABELS
    assert "Other" in RELATION_LABELS


def test_contrast_cue():
    cues = _cue_scores("Apple announced a product.", "However, supply is limited.")
    assert cues["Contrast"] == 1.0


def test_result_cue():
    cues = _cue_scores("Demand is high.", "Therefore shipments will increase.")
    assert cues["Result"] == 1.0


def test_heuristic_contrast():
    emb = np.array([1.0, 0.0], dtype=np.float32)
    label, conf, ev = predict_relation_heuristic(
        "Sales rose.", "However, profits fell.", emb, emb, distance=1, pair_similarity=0.5
    )
    assert label == "Contrast"
    assert conf >= 0.5


def test_heuristic_elaboration_high_sim():
    emb = np.ones(4, dtype=np.float32)
    label, conf, ev = predict_relation_heuristic(
        "NLP made progress.", "Models improved accuracy.", emb, emb, distance=1, pair_similarity=0.6
    )
    assert label in ("Elaboration", "Joint", "Other")
    assert conf > 0


def test_classifier_runs():
    doc, edus, pairs = _pipeline()
    assert len(pairs) >= 1
    clf = RelationClassifier()
    results, stats = clf.classify(pairs, edus)
    assert len(results) == len(pairs)
    assert stats["num_pairs"] == len(pairs)
    for r in results:
        assert r.label in RELATION_LABELS
        assert 0.0 <= r.confidence <= 1.0
        assert r.pair_id


def test_save_json(tmp_path):
    doc, edus, pairs = _pipeline()
    clf = RelationClassifier()
    out = tmp_path / "rel.json"
    results, stats = clf.classify_and_save(pairs, edus, out, document=doc)
    assert out.exists()
    import json
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "relations" in data
    assert data["stats"]["num_pairs"] == len(pairs)


def test_convenience():
    _, edus, pairs = _pipeline()
    results, stats = classify_relations(pairs, edus)
    assert stats["num_pairs"] == len(pairs)


def test_reproducibility():
    _, edus, pairs = _pipeline()
    clf = RelationClassifier()
    a, _ = clf.classify(pairs, edus)
    b, _ = clf.classify(pairs, edus)
    assert [r.label for r in a] == [r.label for r in b]
    assert [r.confidence for r in a] == [r.confidence for r in b]
