#!/usr/bin/env python3
"""
P4 — Gold attachment evaluation for Stage 1.5 pairing.

Compares predicted unordered EDU index pairs to GUM gold RST edges.
Macro precision/recall/F1 over documents. Not full tree UAS.

Usage:
  python scripts\\eval_gold_attachment.py --max-docs 10
  python scripts\\eval_gold_attachment.py --split test --pair-backend minilm --min-sim 0.10
  python scripts\\eval_gold_attachment.py --split test --relation-backend trained
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    import torch  # noqa: F401
except Exception:
    pass

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_schemas import Document, EDU, EDUPair
from stages.stage1_5_pairing import EDUPairer
from stages.stage4_relation.relation_classifier import RelationClassifier
from stages.stage4_relation.label_mapping import COARSE_LABELS


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def document_level_split_docs(rows, seed=42, train_r=0.70, val_r=0.15):
    rng = np.random.RandomState(seed)
    docs = sorted({r["doc_id"] for r in rows})
    rng.shuffle(docs)
    n = len(docs)
    n_train = max(1, int(n * train_r))
    n_val = max(1, int(n * val_r))
    if n_train + n_val >= n:
        n_val = max(1, (n - n_train) // 2)
    return {
        "train": set(docs[:n_train]),
        "val": set(docs[n_train : n_train + n_val]),
        "test": set(docs[n_train + n_val :]),
        "all": set(docs),
    }


def rebuild_doc(doc_rows: List[dict]):
    edu_by_idx: Dict[int, str] = {}
    gold_edges: Set[Tuple[int, int]] = set()
    for r in doc_rows:
        i, j = int(r["edu_i_index"]), int(r["edu_j_index"])
        edu_by_idx[i] = r["edu_i_text"]
        edu_by_idx[j] = r["edu_j_text"]
        a, b = (i, j) if i < j else (j, i)
        if a != b:
            gold_edges.add((a, b))

    edus = []
    parts = []
    offset = 0
    for idx in sorted(edu_by_idx):
        text = edu_by_idx[idx]
        start, end = offset, offset + len(text)
        edus.append(EDU(
            edu_id=f"{doc_rows[0]['doc_id']}_e{idx}",
            doc_id=doc_rows[0]["doc_id"],
            text=text, index=idx, start_char=start, end_char=end,
        ))
        parts.append(text)
        offset = end + 1

    doc = Document(
        doc_id=doc_rows[0]["doc_id"],
        text=" ".join(parts),
        domain=str(doc_rows[0].get("genre", "unknown")),
    )
    return doc, edus, gold_edges, doc_rows


def main():
    ap = argparse.ArgumentParser(description="P4 gold attachment evaluation")
    ap.add_argument("--gold", default=str(ROOT / "data" / "gold" / "gum_v12_relations.jsonl"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--pair-backend", default="minilm", choices=["auto", "minilm", "tfidf"])
    ap.add_argument("--min-sim", type=float, default=0.10)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--relation-backend", default=None, choices=["heuristic", "trained"])
    ap.add_argument("--out", default=str(ROOT / "outputs" / "evaluation" / "p4_attachment.json"))
    args = ap.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.is_file():
        print(f"ERROR: gold missing: {gold_path}")
        sys.exit(1)

    rows = load_jsonl(gold_path)
    splits = document_level_split_docs(rows)
    keep_docs = splits[args.split]
    by_doc = defaultdict(list)
    for r in rows:
        if r["doc_id"] in keep_docs:
            by_doc[r["doc_id"]].append(r)

    doc_ids = sorted(by_doc.keys())
    if args.max_docs:
        doc_ids = doc_ids[: args.max_docs]

    print(f"P4 attachment eval: split={args.split} n_docs={len(doc_ids)} backend={args.pair_backend}")

    pairer = EDUPairer(backend=args.pair_backend, top_k=args.top_k, similarity_threshold=args.min_sim)
    clf = RelationClassifier(backend=args.relation_backend) if args.relation_backend else None

    per_doc_scores = []
    rel_true, rel_pred = [], []

    for di, doc_id in enumerate(doc_ids, 1):
        doc, edus, gold_edges, doc_rows = rebuild_doc(by_doc[doc_id])
        selected, _, pstats = pairer.pair(edus)
        pred_edges = set()
        for p in selected:
            a, b = sorted((p.edu_i_index, p.edu_j_index))
            pred_edges.add((a, b))

        inter = pred_edges & gold_edges
        prec = len(inter) / len(pred_edges) if pred_edges else 0.0
        rec = len(inter) / len(gold_edges) if gold_edges else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_doc_scores.append({
            "doc_id": doc_id, "n_edus": len(edus),
            "n_gold_edges": len(gold_edges), "n_pred_pairs": len(pred_edges),
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        })

        if clf is not None:
            pairs, golds = [], []
            edu_map = {e.index: e for e in edus}
            for k, r in enumerate(doc_rows):
                i, j = int(r["edu_i_index"]), int(r["edu_j_index"])
                ei, ej = edu_map[i], edu_map[j]
                pairs.append(EDUPair(
                    pair_id=f"{doc_id}_g{k}", doc_id=doc_id,
                    edu_i_id=ei.edu_id, edu_j_id=ej.edu_id,
                    edu_i_index=i, edu_j_index=j, distance=abs(i - j), similarity=0.5,
                ))
                golds.append(r["coarse_label"])
            preds, _ = clf.classify(pairs, edus)
            rel_true.extend(golds)
            rel_pred.extend([p.label for p in preds])

        if di % 5 == 0 or di == len(doc_ids):
            print(f"  … {di}/{len(doc_ids)} docs")

    macro_p = float(np.mean([d["precision"] for d in per_doc_scores])) if per_doc_scores else 0.0
    macro_r = float(np.mean([d["recall"] for d in per_doc_scores])) if per_doc_scores else 0.0
    macro_f = float(np.mean([d["f1"] for d in per_doc_scores])) if per_doc_scores else 0.0

    out = {
        "task": "pair_attachment",
        "split": args.split,
        "n_docs": len(doc_ids),
        "pair_backend": args.pair_backend,
        "min_sim": args.min_sim,
        "top_k": args.top_k,
        "attachment_macro": {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1": round(macro_f, 4),
        },
        "per_doc": per_doc_scores,
        "note": "Attachment = unordered EDU pairs vs gold RST edges. Not full tree UAS.",
    }

    if rel_true:
        from sklearn.metrics import accuracy_score, f1_score
        labels = list(COARSE_LABELS)
        out["relation_on_gold_pairs"] = {
            "backend": args.relation_backend,
            "n": len(rel_true),
            "accuracy": float(accuracy_score(rel_true, rel_pred)),
            "macro_f1": float(f1_score(rel_true, rel_pred, average="macro", zero_division=0, labels=labels)),
            "weighted_f1": float(f1_score(rel_true, rel_pred, average="weighted", zero_division=0, labels=labels)),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== P4 Attachment (macro over docs) ===")
    print(f"n_docs={len(doc_ids)}")
    print(f"precision={macro_p:.4f}  recall={macro_r:.4f}  F1={macro_f:.4f}")
    if "relation_on_gold_pairs" in out:
        r = out["relation_on_gold_pairs"]
        print(f"Relation on gold pairs ({r['backend']}): acc={r['accuracy']:.4f} macro-F1={r['macro_f1']:.4f}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
