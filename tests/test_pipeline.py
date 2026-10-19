"""
Phase 7 unit tests — full pipeline + evaluation metrics
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_schemas import Document
from pipeline import DiscoursePipeline, run_pipeline_on_text
from evaluation import (
    tree_structure_stats,
    pair_attachment_metrics,
    evaluate_pipeline_result,
)


SAMPLE = (
    "Apple Inc. announced a new product yesterday. "
    "The device will ship in Q3. "
    "However, supply constraints may limit availability. "
    "Natural language processing has made remarkable progress."
)


def test_pipeline_runs():
    doc = Document(doc_id="p7", text=SAMPLE, domain="news")
    pipe = DiscoursePipeline(
        seg_backend="fallback", pair_backend="tfidf", min_sim=0.0,
        qud_backend="template", run_qud=True,
    )
    result = pipe.run(doc)
    assert result["num_edus"] >= 2
    assert len(result["pairs"]) >= 1
    assert result["tree"] is not None
    assert result["tree"].root.span_end == result["num_edus"] - 1


def test_pipeline_skip_qud():
    doc = Document(doc_id="p7b", text=SAMPLE, domain="news")
    result = DiscoursePipeline(
        seg_backend="fallback", pair_backend="tfidf", min_sim=0.0, run_qud=False,
    ).run(doc)
    assert result["tree"] is not None


def test_structure_stats():
    doc = Document(doc_id="p7c", text=SAMPLE, domain="news")
    result = DiscoursePipeline(
        seg_backend="fallback", pair_backend="tfidf", min_sim=0.0, run_qud=False,
    ).run(doc)
    stats = tree_structure_stats(result["tree"])
    assert stats["num_edus"] == result["num_edus"]
    assert stats["height"] >= 1
    assert stats["num_leaves"] == result["num_edus"]


def test_attachment_metrics_perfect():
    from data_schemas import EDUPair
    pairs = [
        EDUPair(pair_id="a", doc_id="d", edu_i_id="e0", edu_j_id="e1",
                edu_i_index=0, edu_j_index=1, distance=1, similarity=0.5),
    ]
    m = pair_attachment_metrics(pairs, [(0, 1)])
    assert m["f1"] == 1.0


def test_evaluate_pipeline_result():
    doc = Document(doc_id="p7d", text=SAMPLE, domain="news")
    result = DiscoursePipeline(
        seg_backend="fallback", pair_backend="tfidf", min_sim=0.0, run_qud=False,
    ).run(doc)
    report = evaluate_pipeline_result(result)
    assert "structure" in report
    assert "intrinsic" in report


def test_convenience():
    result = run_pipeline_on_text(
        SAMPLE, doc_id="conv", domain="news",
        seg_backend="fallback", pair_backend="tfidf", min_sim=0.0, run_qud=False,
    )
    assert result["num_edus"] >= 2


def test_save_output(tmp_path):
    doc = Document(doc_id="p7e", text=SAMPLE, domain="news")
    DiscoursePipeline(
        seg_backend="fallback", pair_backend="tfidf", min_sim=0.0, run_qud=False,
    ).run(doc, output_dir=tmp_path)
    assert (tmp_path / "p7e_pipeline.json").exists()
    assert (tmp_path / "p7e_tree.json").exists()
