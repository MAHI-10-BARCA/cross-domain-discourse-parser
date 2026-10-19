#!/usr/bin/env python3
"""
Run Phase 1 → 1.5 → 2 → 5: segment, pair, QUD, answer-fit.

Usage:
  python scripts/run_stage5.py
  python scripts/run_stage5.py --pair-backend minilm --min-sim 0.10
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
from stages.stage5_answer_fit import AnswerFitScorer
from utils.logging_utils import setup_logger
from utils.seed import set_seed


SAMPLE_TEXT = """Apple Inc. announced a new product yesterday. The device will ship in Q3. Analysts expect strong demand in the U.S. market. However, supply constraints may limit availability.

Natural language processing has made remarkable progress. Transformer models changed the field. Discourse-level understanding remains challenging."""


def main():
    parser = argparse.ArgumentParser(description="Phase 5 — Answer-fit scoring")
    parser.add_argument("input", nargs="?")
    parser.add_argument("--doc-id", default="sample_doc")
    parser.add_argument("--domain", default="news")
    parser.add_argument("--seg-backend", default="auto", choices=["auto", "spacy", "fallback"])
    parser.add_argument("--pair-backend", default="auto", choices=["auto", "minilm", "tfidf"])
    parser.add_argument("--min-sim", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--qud-backend", default="template", choices=["auto", "t5", "template"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    set_seed(42)
    setup_logger("discourse_parser", log_file=str(ROOT / "outputs" / "logs" / "stage5.log"))

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
    print(f"Stage 1.5: {len(selected)} pairs")

    all_qs, qstats = QUDGenerator(backend=args.qud_backend).generate(selected, edus)
    valid = [q for q in all_qs if q.is_valid]
    print(f"Stage 2: {len(valid)} valid QUDs")

    scorer = AnswerFitScorer()
    out = args.output or (ROOT / "outputs" / "intermediate" / f"{doc_id}_answer_fit.json")
    scores, stats = scorer.score_and_save(valid, edus, out, pairs=selected, document=doc)

    print(f"Stage 5: {stats['num_paired']} paired answer-fit scores")
    print(f"  Mean fit : {stats['mean_paired_fit']}")
    print(f"  Max / Min: {stats['max_paired_fit']} / {stats['min_paired_fit']}")
    print(f"  Output   : {out}\n")

    qmap = {q.question_id: q for q in valid}
    edu_map = {e.edu_id: e for e in edus}
    paired = sorted(
        [s for s in scores if s.metadata.get("role") == "paired"],
        key=lambda s: -s.score,
    )
    print("--- Top 10 answer-fit (paired) ---")
    for s in paired[:10]:
        q = qmap.get(s.question_id)
        a = edu_map.get(s.answer_edu_id)
        print(f"  [{s.question_id}] fit={s.score:.3f}")
        print(f"    Q: {(q.question_text if q else '?')[:65]}")
        print(f"    A: {(a.text if a else '?')[:65]}")
        print()


if __name__ == "__main__":
    main()
