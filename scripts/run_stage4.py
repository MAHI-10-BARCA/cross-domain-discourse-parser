#!/usr/bin/env python3
"""
Run Phase 1 → 1.5 → 4: segment, pair, classify relations.

Usage:
  python scripts/run_stage4.py
  python scripts/run_stage4.py --pair-backend minilm --min-sim 0.10
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
from utils.logging_utils import setup_logger
from utils.seed import set_seed


SAMPLE_TEXT = """Apple Inc. announced a new product yesterday. The device will ship in Q3. Analysts expect strong demand in the U.S. market. However, supply constraints may limit availability.

Natural language processing has made remarkable progress. Transformer models changed the field. Discourse-level understanding remains challenging."""


def main():
    parser = argparse.ArgumentParser(description="Phase 4 — Relation classification")
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
    setup_logger("discourse_parser", log_file=str(ROOT / "outputs" / "logs" / "stage4.log"))

    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
        doc_id = args.doc_id if args.doc_id != "sample_doc" else Path(args.input).stem
    else:
        text = SAMPLE_TEXT
        doc_id = args.doc_id

    doc = Document(doc_id=doc_id, text=text, domain=args.domain)

    edus = EDUSegmenter(backend=args.seg_backend).segment(doc)
    print(f"\nStage 1: {len(edus)} EDUs")

    selected, _, pstats = EDUPairer(
        backend=args.pair_backend, top_k=args.top_k, similarity_threshold=args.min_sim
    ).pair(edus)
    print(f"Stage 1.5: {len(selected)} pairs (backend={pstats['backend']})")

    clf = RelationClassifier()
    out = args.output or (ROOT / "outputs" / "intermediate" / f"{doc_id}_relations.json")
    results, stats = clf.classify_and_save(selected, edus, out, document=doc)

    print(f"Stage 4: {stats['num_pairs']} relations")
    print(f"  Labels : {stats['label_counts']}")
    print(f"  Mean conf: {stats['mean_confidence']}")
    print(f"  Output : {out}\n")

    edu_map = {e.edu_id: e for e in edus}
    print("--- Relations ---")
    for r in results:
        pair = next((p for p in selected if p.pair_id == r.pair_id), None)
        if not pair:
            continue
        ti = edu_map.get(pair.edu_i_id)
        tj = edu_map.get(pair.edu_j_id)
        print(f"  [{r.pair_id}] {r.label} (conf={r.confidence:.2f}) dist={pair.distance}")
        print(f"    i: {(ti.text if ti else '?')[:60]}")
        print(f"    j: {(tj.text if tj else '?')[:60]}")
        print(f"    evidence: {r.features.get('decision')} cues={r.features.get('cues')}")
        print()


if __name__ == "__main__":
    main()
