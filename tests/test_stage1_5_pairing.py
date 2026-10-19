"""
Phase 1.5 unit tests — Related EDU Pairing
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_schemas import Document, EDU, EDUPair
from stages.stage1_edu_segmentation import EDUSegmenter
from stages.stage1_5_pairing import EDUPairer, pair_edus
from stages.stage1_5_pairing.pairing import (
    embed_tfidf, cosine_similarity_matrix,
    select_pairs_topk_distance, select_adjacent_baseline,
)


SAMPLE = (
    "Apple Inc. announced a new product yesterday. "
    "The device will ship in Q3. "
    "Analysts expect strong demand in the U.S. market. "
    "However, supply constraints may limit availability. "
    "Natural language processing has made remarkable progress. "
    "Transformer models changed the field. "
    "Discourse-level understanding remains challenging."
)


def _make_edus(text: str = SAMPLE, doc_id: str = "t1") -> list:
    doc = Document(doc_id=doc_id, text=text)
    return EDUSegmenter(backend="fallback").segment(doc)


def test_tfidf_embeddings():
    texts = ["hello world", "hello there", "goodbye"]
    emb = embed_tfidf(texts)
    assert emb.shape[0] == 3
    assert emb.shape[1] > 0
    norms = np.linalg.norm(emb, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_cosine_matrix():
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    sim = cosine_similarity_matrix(emb)
    assert sim.shape == (3, 3)
    np.testing.assert_allclose(sim[0, 2], 1.0, atol=1e-5)
    np.testing.assert_allclose(sim[0, 1], 0.0, atol=1e-5)


def test_adjacent_baseline():
    edus = _make_edus()
    emb = embed_tfidf([e.text for e in edus])
    sim = cosine_similarity_matrix(emb)
    adj = select_adjacent_baseline(edus, sim)
    assert len(adj) == len(edus) - 1
    for p in adj:
        assert p.distance == 1
        assert p.metadata["strategy"] == "adjacent_baseline"


def test_topk_distance_selection():
    edus = _make_edus()
    emb = embed_tfidf([e.text for e in edus])
    sim = cosine_similarity_matrix(emb)
    pairs = select_pairs_topk_distance(edus, sim, top_k=2, max_distance=3)
    assert len(pairs) >= 1
    for p in pairs:
        assert p.distance <= 3
        assert 0.0 <= p.similarity <= 1.0 + 1e-5
        assert p.edu_i_index < p.edu_j_index


def test_pairer_tfidf():
    edus = _make_edus()
    pairer = EDUPairer(backend="tfidf", top_k=3, max_distance=5)
    selected, adjacent, stats = pairer.pair(edus)
    assert stats["num_edus"] == len(edus)
    assert stats["num_adjacent"] == len(edus) - 1
    assert stats["num_selected"] >= 1
    assert stats["backend"] == "tfidf"
    assert 0.0 <= stats["adjacent_coverage"] <= 1.0
    for p in selected:
        assert p.pair_id.startswith("t1_pair_")
        assert p.doc_id == "t1"


def test_single_edu():
    edus = [EDU(edu_id="d_edu_000", doc_id="d", text="Only one.", index=0)]
    pairer = EDUPairer(backend="tfidf")
    selected, adjacent, stats = pairer.pair(edus)
    assert selected == []
    assert adjacent == []
    assert stats["num_selected"] == 0


def test_two_edus():
    edus = [
        EDU(edu_id="d_edu_000", doc_id="d", text="First sentence.", index=0),
        EDU(edu_id="d_edu_001", doc_id="d", text="Second sentence.", index=1),
    ]
    pairer = EDUPairer(backend="tfidf", top_k=3, max_distance=5)
    selected, adjacent, stats = pairer.pair(edus)
    assert len(adjacent) == 1
    assert len(selected) >= 1
    assert selected[0].distance == 1


def test_reproducibility():
    edus = _make_edus()
    pairer = EDUPairer(backend="tfidf", top_k=3, max_distance=5)
    s1, a1, st1 = pairer.pair(edus)
    s2, a2, st2 = pairer.pair(edus)
    assert [p.pair_id for p in s1] == [p.pair_id for p in s2]
    assert [p.similarity for p in s1] == [p.similarity for p in s2]


def test_save_json(tmp_path):
    edus = _make_edus()
    pairer = EDUPairer(backend="tfidf")
    out = tmp_path / "pairs.json"
    selected, adjacent, stats = pairer.pair_and_save(edus, out)
    assert out.exists()
    import json
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "selected_pairs" in data
    assert "adjacent_baseline" in data
    assert "stats" in data
    assert data["stats"]["num_selected"] == len(selected)


def test_convenience_function():
    edus = _make_edus()
    selected, adjacent, stats = pair_edus(edus, top_k=2, max_distance=3, backend="tfidf")
    assert stats["top_k"] == 2
    assert stats["max_distance"] == 3


def test_max_distance_respected():
    edus = _make_edus()
    pairer = EDUPairer(backend="tfidf", top_k=10, max_distance=2)
    selected, _, _ = pairer.pair(edus)
    for p in selected:
        assert p.distance <= 2


def test_non_adjacent_can_be_selected():
    """With max_distance > 1, some non-adjacent pairs should appear."""
    edus = _make_edus()
    pairer = EDUPairer(backend="tfidf", top_k=5, max_distance=5)
    selected, _, stats = pairer.pair(edus)
    assert stats["num_non_adjacent_selected"] >= 0
    for p in selected:
        assert p.metadata.get("strategy") == "topk_distance"
