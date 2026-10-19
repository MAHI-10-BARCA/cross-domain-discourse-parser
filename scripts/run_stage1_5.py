#!/usr/bin/env python3
"""
Run Phase 1 → 1.5 pipeline: segment then pair EDUs.

Usage:
  python scripts/run_stage1_5.py
  python scripts/run_stage1_5.py path/to/document.txt
  python scripts/run_stage1_5.py --backend tfidf --top-k 3 --max-distance 5
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_schemas import Document
from stages.stage1_edu_segmentation import EDUSegmenter
from stages.stage1_5_pairing import EDUPairer
from utils.logging_utils import setup_logger
from utils.seed import set_seed


SAMPLE_TEXT = """Apple Inc. announced a new product yesterday. The device will ship in Q3. Analysts expect strong demand in the U.S. market. However, supply constraints may limit availability.

Natural language processing has made remarkable progress. Transformer models changed the field. Discourse-level understanding remains challenging."""


def main():
    parser = argparse.ArgumentParser(description="Phase 1.5 — Related EDU Pairing")
    parser.add_argument("input", nargs="?", help="Path to a .txt document")
    parser.add_argument("--doc-id", default="sample_doc")
    parser.add_argument("--domain", default="news")
    parser.add_argument("--seg-backend", default="auto", choices=["auto", "spacy", "fallback"])
    parser.add_argument("--backend", default="auto", choices=["auto", "minilm", "tfidf"],
                        help="Embedding backend for pairing")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-distance", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    set_seed(42)
    logger = setup_logger("discourse_parser", log_file=str(ROOT / "outputs" / "logs" / "stage1_5.log"))

    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
        doc_id = args.doc_id if args.doc_id != "sample_doc" else Path(args.input).stem
    else:
        text = SAMPLE_TEXT
        doc_id = args.doc_id
        logger.info("Using built-in sample text.")

    doc = Document(doc_id=doc_id, text=text, domain=args.domain)

    # Stage 1
    segmenter = EDUSegmenter(backend=args.seg_backend)
    edus = segmenter.segment(doc)
    print(f"\nStage 1: {len(edus)} EDUs (backend={segmenter._resolved_backend})")

    # Stage 1.5
    pairer = EDUPairer(
        backend=args.backend,
        top_k=args.top_k,
        max_distance=args.max_distance,
    )
    out_path = args.output or (ROOT / "outputs" / "intermediate" / f"{doc_id}_pairs.json")
    selected, adjacent, stats = pairer.pair_and_save(edus, out_path, document=doc)

    print(f"Stage 1.5 backend : {stats['backend']}")
    print(f"Selected pairs    : {stats['num_selected']}")
    print(f"Adjacent baseline : {stats['num_adjacent']}")
    print(f"Non-adjacent      : {stats['num_non_adjacent_selected']}")
    print(f"Adjacent coverage : {stats['adjacent_coverage']}")
    print(f"Mean sel. sim.    : {stats['mean_selected_similarity']}")
    print(f"Output            : {out_path}\n")

    print("--- Selected pairs (Top-K + max-distance) ---")
    for p in selected:
        print(f"  [{p.pair_id}] dist={p.distance} sim={p.similarity:.4f}")
        print(f"    i={p.edu_i_index}: {edus[p.edu_i_index].text[:60]}...")
        print(f"    j={p.edu_j_index}: {edus[p.edu_j_index].text[:60]}...")
        print()


if __name__ == "__main__":
    main()
