"""
Phase 1.5 unit tests — Global related EDU Pairing
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
    select_pairs_global_topk, select_adjacent_baseline,
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
    norms = np.linalg.norm(emb, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_cosine_matrix():
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    sim = cosine_similarity_matrix(emb)
    assert sim.shape == (3, 3)
    np.testing.assert_allclose(sim[0, 2], 1.0, atol=1e-5)
    np.testing.assert_allclose(sim[0, 1], 0.0, atol=1e-5)


def test_global_topk_no_distance_limit():
    """Pairs can span any distance — adjacency is not required."""
    edus = _make_edus()
    emb = embed_tfidf([e.text for e in edus])
    sim = cosine_similarity_matrix(emb)
    pairs = select_pairs_global_topk(
        edus, sim, top_k=3, similarity_threshold=0.0, max_distance=None
    )
    assert len(pairs) >= 1
    for p in pairs:
        assert p.edu_i_index < p.edu_j_index
        assert p.metadata["strategy"] == "global_topk"
        assert p.metadata["max_distance"] is None


def test_similarity_threshold_filters_weak_pairs():
    edus = _make_edus()
    emb = embed_tfidf([e.text for e in edus])
    sim = cosine_similarity_matrix(emb)
    strict = select_pairs_global_topk(edus, sim, top_k=10, similarity_threshold=0.99)
    loose = select_pairs_global_topk(edus, sim, top_k=10, similarity_threshold=0.0)
    assert len(strict) <= len(loose)


def test_adjacent_baseline_is_comparison_only():
    edus = _make_edus()
    emb = embed_tfidf([e.text for e in edus])
    sim = cosine_similarity_matrix(emb)
    adj = select_adjacent_baseline(edus, sim)
    assert len(adj) == len(edus) - 1
    for p in adj:
        assert p.distance == 1
        assert p.metadata["strategy"] == "adjacent_baseline"


def test_pairer_global_default():
    edus = _make_edus()
    pairer = EDUPairer(backend="tfidf", top_k=3, similarity_threshold=0.0)
    selected, adjacent, stats = pairer.pair(edus)
    assert stats["strategy"] == "global_topk"
    assert stats["max_distance"] is None
    assert stats["num_selected"] >= 1
    assert stats["num_adjacent_baseline"] == len(edus) - 1
    for p in selected:
        assert p.metadata["strategy"] == "global_topk"


def test_single_edu():
    edus = [EDU(edu_id="d_edu_000", doc_id="d", text="Only one.", index=0)]
    pairer = EDUPairer(backend="tfidf")
    selected, adjacent, stats = pairer.pair(edus)
    assert selected == []
    assert adjacent == []


def test_two_edus():
    edus = [
        EDU(edu_id="d_edu_000", doc_id="d", text="First sentence about cats.", index=0),
        EDU(edu_id="d_edu_001", doc_id="d", text="Second sentence about dogs.", index=1),
    ]
    pairer = EDUPairer(backend="tfidf", top_k=3, similarity_threshold=0.0)
    selected, adjacent, stats = pairer.pair(edus)
    assert len(adjacent) == 1
    assert len(selected) >= 1


def test_reproducibility():
    edus = _make_edus()
    pairer = EDUPairer(backend="tfidf", top_k=3, similarity_threshold=0.0)
    s1, _, _ = pairer.pair(edus)
    s2, _, _ = pairer.pair(edus)
    assert [p.pair_id for p in s1] == [p.pair_id for p in s2]
    assert [p.similarity for p in s1] == [p.similarity for p in s2]


def test_save_json(tmp_path):
    edus = _make_edus()
    pairer = EDUPairer(backend="tfidf", similarity_threshold=0.0)
    out = tmp_path / "pairs.json"
    selected, adjacent, stats = pairer.pair_and_save(edus, out)
    assert out.exists()
    import json
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["stats"]["strategy"] == "global_topk"
    assert data["stats"]["num_selected"] == len(selected)


def test_convenience_function():
    edus = _make_edus()
    selected, adjacent, stats = pair_edus(
        edus, top_k=2, similarity_threshold=0.0, backend="tfidf"
    )
    assert stats["top_k"] == 2
    assert stats["strategy"] == "global_topk"


def test_optional_max_distance_still_works():
    """max_distance is optional; when set it acts as a soft cap, not adjacency forcing."""
    edus = _make_edus()
    pairer = EDUPairer(backend="tfidf", top_k=10, similarity_threshold=0.0, max_distance=2)
    selected, _, _ = pairer.pair(edus)
    for p in selected:
        assert p.distance <= 2
