"""
Phase 6 unit tests — Bottom-up tree construction
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_schemas import Document, EDU, EDUPair, RelationPrediction, DiscourseTree
from stages.stage1_edu_segmentation import EDUSegmenter
from stages.stage1_5_pairing import EDUPairer
from stages.stage4_relation import RelationClassifier
from stages.stage6_tree import BottomUpTreeBuilder, build_discourse_tree
from stages.stage6_tree.tree_builder import tree_to_ascii, _count_relations


SAMPLE = (
    "Apple Inc. announced a new product yesterday. "
    "The device will ship in Q3. "
    "Analysts expect strong demand in the U.S. market. "
    "However, supply constraints may limit availability. "
    "Natural language processing has made remarkable progress. "
    "Transformer models changed the field."
)


def _pipeline():
    doc = Document(doc_id="t6", text=SAMPLE, domain="news")
    edus = EDUSegmenter(backend="fallback").segment(doc)
    pairs, _, _ = EDUPairer(backend="tfidf", top_k=3, similarity_threshold=0.0).pair(edus)
    rels, _ = RelationClassifier().classify(pairs, edus)
    return doc, edus, pairs, rels


def test_builds_tree():
    doc, edus, pairs, rels = _pipeline()
    builder = BottomUpTreeBuilder()
    tree, stats = builder.build(edus, pairs, relations=rels, document=doc)
    assert isinstance(tree, DiscourseTree)
    assert tree.root is not None
    assert stats["num_edus"] == len(edus)
    assert stats["num_merges"] >= 1
    assert tree.root.span_start == 0
    assert tree.root.span_end == len(edus) - 1


def test_binary_internal_nodes():
    doc, edus, pairs, rels = _pipeline()
    tree, _ = BottomUpTreeBuilder().build(edus, pairs, relations=rels, document=doc)

    def check(node):
        if node.children:
            assert len(node.children) == 2
            for c in node.children:
                check(c)
    check(tree.root)


def test_leaves_match_edus():
    doc, edus, pairs, rels = _pipeline()
    tree, _ = BottomUpTreeBuilder().build(edus, pairs, relations=rels, document=doc)
    leaves = []

    def collect(node):
        if not node.children:
            leaves.append(node)
        for c in node.children:
            collect(c)

    collect(tree.root)
    assert len(leaves) == len(edus)
    assert sorted(n.span_start for n in leaves) == list(range(len(edus)))


def test_save_json(tmp_path):
    doc, edus, pairs, rels = _pipeline()
    out = tmp_path / "tree.json"
    tree, stats = BottomUpTreeBuilder().build_and_save(
        edus, pairs, out, relations=rels, document=doc
    )
    assert out.exists()
    import json
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "tree" in data
    assert data["stats"]["num_edus"] == len(edus)


def test_ascii_print():
    doc, edus, pairs, rels = _pipeline()
    tree, _ = BottomUpTreeBuilder().build(edus, pairs, relations=rels, document=doc)
    text = tree_to_ascii(tree.root, edus)
    assert "span=" in text or "\u2022" in text or "•" in text


def test_convenience():
    doc, edus, pairs, rels = _pipeline()
    tree, stats = build_discourse_tree(edus, pairs, relations=rels)
    assert stats["num_edus"] == len(edus)


def test_single_edu():
    doc = Document(doc_id="one", text="Only one sentence here.", domain="news")
    edus = EDUSegmenter(backend="fallback").segment(doc)
    pairs, _, _ = EDUPairer(backend="tfidf", top_k=1, similarity_threshold=0.0).pair(edus)
    tree, stats = BottomUpTreeBuilder().build(edus, pairs, document=doc)
    assert stats["num_edus"] == 1
    assert stats["num_merges"] == 0
    assert not tree.root.children


def test_reproducibility():
    doc, edus, pairs, rels = _pipeline()
    a, sa = BottomUpTreeBuilder().build(edus, pairs, relations=rels, document=doc)
    b, sb = BottomUpTreeBuilder().build(edus, pairs, relations=rels, document=doc)
    assert sa["num_merges"] == sb["num_merges"]
    assert a.root.relation == b.root.relation
