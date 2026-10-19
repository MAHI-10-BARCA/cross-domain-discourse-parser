#!/usr/bin/env python3
"""
Run Phase 1 → 1.5 → 4 → 6: segment, pair, relations, build discourse tree.

Usage:
  python scripts/run_stage6.py
  python scripts/run_stage6.py --pair-backend minilm --min-sim 0.10
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_schemas import Document
from stages.stage1_edu_segmentation import EDUSegmenter
from stages.stage1_5_pairing import EDUPairer
from stages.stage4_relation import RelationClassifier
from stages.stage6_tree import BottomUpTreeBuilder
from stages.stage6_tree.tree_builder import tree_to_ascii
from utils.logging_utils import setup_logger
from utils.seed import set_seed


SAMPLE_TEXT = """Apple Inc. announced a new product yesterday. The device will ship in Q3. Analysts expect strong demand in the U.S. market. However, supply constraints may limit availability.

Natural language processing has made remarkable progress. Transformer models changed the field. Discourse-level understanding remains challenging."""


def main():
    parser = argparse.ArgumentParser(description="Phase 6 — Bottom-up discourse tree")
    parser.add_argument("input", nargs="?")
    parser.add_argument("--doc-id", default="sample_doc")
    parser.add_argument("--domain", default="news")
    parser.add_argument("--seg-backend", default="auto", choices=["auto", "spacy", "fallback"])
    parser.add_argument("--pair-backend", default="auto", choices=["auto", "minilm", "tfidf"])
    parser.add_argument("--min-sim", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    set_seed(42)
    setup_logger("discourse_parser", log_file=str(ROOT / "outputs" / "logs" / "stage6.log"))

    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
        doc_id = args.doc_id if args.doc_id != "sample_doc" else Path(args.input).stem
    else:
        text = SAMPLE_TEXT
        doc_id = args.doc_id

    doc = Document(doc_id=doc_id, text=text, domain=args.domain)
    edus = EDUSegmenter(backend=args.seg_backend).segment(doc)
    print(f"\nStage 1: {len(edus)} EDUs")
    for e in edus:
        print(f"  [{e.edu_id}] {e.text[:60]}")

    selected, _, pstats = EDUPairer(
        backend=args.pair_backend, top_k=args.top_k, similarity_threshold=args.min_sim
    ).pair(edus)
    print(f"\nStage 1.5: {len(selected)} pairs")

    rels, rstats = RelationClassifier().classify(selected, edus)
    print(f"Stage 4: relations {rstats['label_counts']}")

    builder = BottomUpTreeBuilder()
    out = args.output or (ROOT / "outputs" / "trees" / f"{doc_id}_tree.json")
    tree, stats = builder.build_and_save(
        edus, selected, out, relations=rels, document=doc
    )

    print(f"\nStage 6: tree with {stats['num_merges']} merges")
    print(f"  Root span : {stats['root_span']}")
    print(f"  Relations : {stats['relation_counts']}")
    print(f"  Output    : {out}\n")
    print("--- Discourse Tree ---")
    print(tree_to_ascii(tree.root, edus))


if __name__ == "__main__":
    main()
