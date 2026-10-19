#!/usr/bin/env python3
"""
Run Phase 1 → 1.5 → 2: segment, pair, generate QUDs.

Usage:
  python scripts/run_stage2.py
  python scripts/run_stage2.py path/to/document.txt
  python scripts/run_stage2.py --qud-backend template
  python scripts/run_stage2.py --qud-backend t5
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_schemas import Document
from stages.stage1_edu_segmentation import EDUSegmenter
from stages.stage1_5_pairing import EDUPairer
from stages.stage2_qud_generation import QUDGenerator
from utils.logging_utils import setup_logger
from utils.seed import set_seed


SAMPLE_TEXT = """Apple Inc. announced a new product yesterday. The device will ship in Q3. Analysts expect strong demand in the U.S. market. However, supply constraints may limit availability.

Natural language processing has made remarkable progress. Transformer models changed the field. Discourse-level understanding remains challenging."""


def main():
    parser = argparse.ArgumentParser(description="Phase 2 — QUD multi-question generation")
    parser.add_argument("input", nargs="?", help="Path to a .txt document")
    parser.add_argument("--doc-id", default="sample_doc")
    parser.add_argument("--domain", default="news")
    parser.add_argument("--seg-backend", default="auto", choices=["auto", "spacy", "fallback"])
    parser.add_argument("--pair-backend", default="auto", choices=["auto", "minilm", "tfidf"])
    parser.add_argument("--min-sim", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--qud-backend", default="auto", choices=["auto", "t5", "template"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    set_seed(42)
    logger = setup_logger(
        "discourse_parser",
        log_file=str(ROOT / "outputs" / "logs" / "stage2.log"),
    )

    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
        doc_id = args.doc_id if args.doc_id != "sample_doc" else Path(args.input).stem
    else:
        text = SAMPLE_TEXT
        doc_id = args.doc_id
        logger.info("Using built-in sample text.")

    doc = Document(doc_id=doc_id, text=text, domain=args.domain)

    segmenter = EDUSegmenter(backend=args.seg_backend)
    edus = segmenter.segment(doc)
    print(f"\nStage 1: {len(edus)} EDUs")

    pairer = EDUPairer(
        backend=args.pair_backend,
        top_k=args.top_k,
        similarity_threshold=args.min_sim,
    )
    selected, _, pair_stats = pairer.pair(edus)
    print(f"Stage 1.5: {len(selected)} pairs (backend={pair_stats['backend']})")

    gen = QUDGenerator(backend=args.qud_backend)
    out_path = args.output or (ROOT / "outputs" / "intermediate" / f"{doc_id}_quds.json")
    all_qs, stats = gen.generate_and_save(selected, edus, out_path, document=doc)

    valid = [q for q in all_qs if q.is_valid]
    print(f"Stage 2 backend : {stats['backend']}")
    print(f"Candidates      : {stats['num_candidates_total']}")
    print(f"Valid           : {stats['num_valid']}")
    print(f"Rejected        : {stats['num_rejected']}")
    print(f"Categories      : {stats['category_counts']}")
    print(f"Output          : {out_path}\n")

    print("--- Sample valid QUDs (up to 12) ---")
    for q in valid[:12]:
        print(f"  [{q.question_id}] ({q.category}) pair={q.pair_id}")
        print(f"    {q.question_text}")
        print()


if __name__ == "__main__":
    main()
