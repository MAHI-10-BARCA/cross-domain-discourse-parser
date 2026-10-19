#!/usr/bin/env python3
"""
Run Phase 1 → 1.5 pipeline: segment then pair EDUs globally by similarity.

Primary pairing is GLOBAL Top-K (strong semantic relatedness).
Adjacency is NOT used to force pairs — only as an evaluation baseline.

Usage:
  python scripts/run_stage1_5.py
  python scripts/run_stage1_5.py path/to/document.txt
  python scripts/run_stage1_5.py --backend minilm --top-k 3 --min-sim 0.30
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
    parser = argparse.ArgumentParser(
        description="Phase 1.5 — Global related EDU pairing (similarity, not adjacency)"
    )
    parser.add_argument("input", nargs="?", help="Path to a .txt document")
    parser.add_argument("--doc-id", default="sample_doc")
    parser.add_argument("--domain", default="news")
    parser.add_argument("--seg-backend", default="auto", choices=["auto", "spacy", "fallback"])
    parser.add_argument("--backend", default="auto", choices=["auto", "minilm", "tfidf"],
                        help="Embedding backend")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Top-K most similar partners per EDU")
    parser.add_argument("--min-sim", type=float, default=0.25,
                        help="Minimum cosine similarity to keep a pair")
    parser.add_argument("--max-distance", type=int, default=None,
                        help="Optional distance cap (default: None = global, no limit)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    set_seed(42)
    logger = setup_logger(
        "discourse_parser",
        log_file=str(ROOT / "outputs" / "logs" / "stage1_5.log"),
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
    print(f"\nStage 1: {len(edus)} EDUs (backend={segmenter._resolved_backend})")

    pairer = EDUPairer(
        backend=args.backend,
        top_k=args.top_k,
        similarity_threshold=args.min_sim,
        max_distance=args.max_distance,
    )
    out_path = args.output or (ROOT / "outputs" / "intermediate" / f"{doc_id}_pairs.json")
    selected, adjacent, stats = pairer.pair_and_save(edus, out_path, document=doc)

    print(f"Strategy          : global_topk (similarity-driven, NOT adjacency-forced)")
    print(f"Embedding backend : {stats['backend']}")
    print(f"Selected pairs    : {stats['num_selected']}")
    print(f"Non-adjacent      : {stats['num_non_adjacent_selected']}")
    print(f"Mean pair distance: {stats['mean_pair_distance']}")
    print(f"Mean similarity   : {stats['mean_selected_similarity']}")
    print(f"Adj. baseline     : {stats['num_adjacent_baseline']} (comparison only)")
    print(f"Output            : {out_path}\n")

    print("--- Selected pairs (ranked by similarity) ---")
    for p in selected:
        print(f"  [{p.pair_id}] dist={p.distance} sim={p.similarity:.4f}")
        print(f"    i={p.edu_i_index}: {edus[p.edu_i_index].text[:70]}")
        print(f"    j={p.edu_j_index}: {edus[p.edu_j_index].text[:70]}")
        print()


if __name__ == "__main__":
    main()
