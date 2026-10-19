#!/usr/bin/env python3
"""
Run Phase 1 EDU segmentation on a text file or a built-in sample.

Usage:
  python scripts/run_stage1.py                          # uses built-in sample
  python scripts/run_stage1.py path/to/document.txt
  python scripts/run_stage1.py path/to/document.txt --backend fallback
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_schemas import Document
from stages.stage1_edu_segmentation import EDUSegmenter
from utils.logging_utils import setup_logger
from utils.seed import set_seed
from utils.io import load_yaml


SAMPLE_TEXT = """Apple Inc. announced a new product yesterday. The device will ship in Q3. Analysts expect strong demand in the U.S. market. However, supply constraints may limit availability.

Natural language processing has made remarkable progress. Transformer models changed the field. Discourse-level understanding remains challenging."""


def main():
    parser = argparse.ArgumentParser(description="Phase 1 — EDU Segmentation")
    parser.add_argument("input", nargs="?", help="Path to a .txt document (optional)")
    parser.add_argument("--doc-id", default="sample_doc", help="Document ID")
    parser.add_argument("--domain", default="news", help="Domain label")
    parser.add_argument("--backend", default="auto", choices=["auto", "spacy", "fallback"])
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    set_seed(42)
    logger = setup_logger("discourse_parser", log_file=str(ROOT / "outputs" / "logs" / "stage1.log"))

    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
        doc_id = args.doc_id if args.doc_id != "sample_doc" else Path(args.input).stem
    else:
        text = SAMPLE_TEXT
        doc_id = args.doc_id
        logger.info("No input file given — using built-in sample text.")

    doc = Document(doc_id=doc_id, text=text, domain=args.domain)
    segmenter = EDUSegmenter(backend=args.backend)

    out_path = args.output or (ROOT / "outputs" / "intermediate" / f"{doc_id}_edus.json")
    edus = segmenter.segment_and_save(doc, out_path)

    print(f"\nDocument : {doc_id}")
    print(f"Backend  : {segmenter._resolved_backend}")
    print(f"EDUs     : {len(edus)}")
    print(f"Output   : {out_path}\n")
    for edu in edus:
        print(f"  [{edu.edu_id}] (chars {edu.start_char}-{edu.end_char})")
        print(f"    {edu.text}\n")


if __name__ == "__main__":
    main()
