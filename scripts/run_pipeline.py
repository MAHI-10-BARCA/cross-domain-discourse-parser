#!/usr/bin/env python3
"""
End-to-end discourse parsing pipeline.

Usage:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py data/raw/news_reuters_sample.txt --domain news
  python scripts/run_pipeline.py data/raw/ --pair-backend minilm --min-sim 0.10 --relation-backend hybrid
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_schemas import Document, save_json
from pipeline import DiscoursePipeline
from evaluation import evaluate_pipeline_result
from utils.logging_utils import setup_logger
from utils.seed import set_seed


def collect_inputs(path: Path):
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("*.txt"))
    return []


def main():
    parser = argparse.ArgumentParser(description="Full discourse parsing pipeline")
    parser.add_argument("input", nargs="?", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--domain", default="unknown")
    parser.add_argument("--seg-backend", default="auto")
    parser.add_argument("--pair-backend", default="auto")
    parser.add_argument("--min-sim", type=float, default=0.12)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--qud-backend", default="template", choices=["template", "t5", "auto"])
    parser.add_argument(
        "--relation-backend",
        default="hybrid",
        choices=["heuristic", "trained", "hybrid"],
        help="Stage 4: hybrid (cues+trained), trained, or heuristic",
    )
    parser.add_argument("--use-nli", action="store_true")
    parser.add_argument("--skip-qud", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    set_seed(42)
    setup_logger("discourse_parser", log_file=str(ROOT / "outputs" / "logs" / "pipeline.log"))

    inp = Path(args.input)
    if not inp.is_absolute():
        candidate = ROOT / inp
        if candidate.exists():
            inp = candidate

    files = collect_inputs(inp)
    if not files:
        print("No .txt files found; using built-in sample.")
        text = (
            "Apple Inc. announced a new product yesterday. The device will ship in Q3. "
            "Analysts expect strong demand in the U.S. market. However, supply constraints may limit availability. "
            "Natural language processing has made remarkable progress. Transformer models changed the field."
        )
        docs = [Document(doc_id="sample_doc", text=text, domain="news")]
    else:
        docs = []
        for f in files:
            text = f.read_text(encoding="utf-8")
            domain = args.domain if args.domain != "unknown" else f.stem.split("_")[0]
            docs.append(Document(doc_id=f.stem, text=text, domain=domain))

    out_dir = Path(args.output_dir) if args.output_dir else (ROOT / "outputs" / "pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = DiscoursePipeline(
        seg_backend=args.seg_backend,
        pair_backend=args.pair_backend,
        min_sim=args.min_sim,
        top_k=args.top_k,
        qud_backend=args.qud_backend,
        relation_backend=args.relation_backend,
        use_nli=args.use_nli,
        run_qud=not args.skip_qud,
        run_salience=not args.skip_qud,
        run_answer_fit=not args.skip_qud,
    )

    all_reports = []
    for doc in docs:
        print(f"\n{'='*60}")
        print(f"Document: {doc.doc_id}  domain={doc.domain}")
        print(f"{'='*60}")
        result = pipe.run(doc, output_dir=out_dir)
        report = evaluate_pipeline_result(result)
        all_reports.append(report)

        print(f"EDUs     : {result['num_edus']}")
        print(f"Pairs    : {len(result['pairs'])}")
        print(f"Relations: {result['relation_stats'].get('label_counts')}")
        print(f"Tree     : {result['tree_stats'].get('num_merges')} merges, "
              f"height={report['structure']['height']}")
        print(f"Rel dist : {report['structure']['relation_distribution']}")
        print()
        print(result["tree_ascii"])

    summary_path = out_dir / "evaluation_summary.json"
    save_json(all_reports, summary_path)
    print(f"\nSaved {len(all_reports)} reports → {summary_path}")


if __name__ == "__main__":
    main()
