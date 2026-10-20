#!/usr/bin/env python3
"""
P1b — Fair comparison: heuristic vs trained relation classifiers on GUM gold test.

Uses the same document-level split as train_relation_classifier.py (seed=42).
Does NOT invent metrics; writes real numbers to outputs/evaluation/.

Usage (Windows CMD):
  python scripts\\compare_relation_backends.py
  python scripts\\compare_relation_backends.py --gold data\\gold\\gum_v12_relations.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_schemas import EDU, EDUPair
from stages.stage4_relation.relation_classifier import RelationClassifier
from stages.stage4_relation.label_mapping import COARSE_LABELS


def _load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _document_level_split(rows, seed=42, train_r=0.70, val_r=0.15):
    rng = np.random.RandomState(seed)
    docs = sorted({r["doc_id"] for r in rows})
    rng.shuffle(docs)
    n = len(docs)
    n_train = max(1, int(n * train_r))
    n_val = max(1, int(n * val_r))
    if n_train + n_val >= n:
        n_val = max(1, (n - n_train) // 2)
    train_docs = set(docs[:n_train])
    val_docs = set(docs[n_train : n_train + n_val])
    test_docs = set(docs[n_train + n_val :])
    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        if r["doc_id"] in train_docs:
            splits["train"].append(r)
        elif r["doc_id"] in val_docs:
            splits["val"].append(r)
        else:
            splits["test"].append(r)
    return splits, {
        "n_docs_test": len(test_docs),
        "n_test": len(splits["test"]),
        "test_docs": sorted(test_docs),
    }


def rows_to_edus_pairs(rows):
    by_doc = {}
    for r in rows:
        by_doc.setdefault(r["doc_id"], []).append(r)

    all_gold, all_pred_h, all_pred_t = [], [], []
    stats = {"n_docs": 0, "n_pairs": 0, "trained_errors": 0}

    clf_h = RelationClassifier(backend="heuristic")
    try:
        clf_t = RelationClassifier(backend="trained")
    except Exception as e:
        print(f"ERROR loading trained model: {e}")
        sys.exit(1)

    for doc_id, doc_rows in by_doc.items():
        edu_by_idx = {}
        for r in doc_rows:
            i, j = r["edu_i_index"], r["edu_j_index"]
            if i not in edu_by_idx:
                edu_by_idx[i] = EDU(
                    edu_id=f"{doc_id}_e{i}", doc_id=doc_id, text=r["edu_i_text"],
                    index=i, start_char=0, end_char=len(r["edu_i_text"]),
                )
            if j not in edu_by_idx:
                edu_by_idx[j] = EDU(
                    edu_id=f"{doc_id}_e{j}", doc_id=doc_id, text=r["edu_j_text"],
                    index=j, start_char=0, end_char=len(r["edu_j_text"]),
                )
        edus = [edu_by_idx[k] for k in sorted(edu_by_idx)]
        pairs, golds = [], []
        for k, r in enumerate(doc_rows):
            i, j = r["edu_i_index"], r["edu_j_index"]
            ei, ej = edu_by_idx[i], edu_by_idx[j]
            pairs.append(EDUPair(
                pair_id=f"{doc_id}_p{k}", doc_id=doc_id,
                edu_i_id=ei.edu_id, edu_j_id=ej.edu_id,
                edu_i_index=i, edu_j_index=j, distance=abs(i - j), similarity=0.5,
            ))
            golds.append(r["coarse_label"])

        pred_h, _ = clf_h.classify(pairs, edus)
        try:
            pred_t, _ = clf_t.classify(pairs, edus)
        except Exception as e:
            stats["trained_errors"] += 1
            print(f"  trained failed on {doc_id}: {e}")
            continue

        all_gold.extend(golds)
        all_pred_h.extend([p.label for p in pred_h])
        all_pred_t.extend([p.label for p in pred_t])
        stats["n_docs"] += 1
        stats["n_pairs"] += len(pairs)

    return all_gold, all_pred_h, all_pred_t, stats


def metrics(y_true, y_pred, labels):
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=labels)),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=True
        ),
        "pred_dist": dict(Counter(y_pred)),
        "true_dist": dict(Counter(y_true)),
        "n": len(y_true),
    }


def main():
    ap = argparse.ArgumentParser(description="P1b: heuristic vs trained on GUM test")
    ap.add_argument("--gold", default=str(ROOT / "data" / "gold" / "gum_v12_relations.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "evaluation" / "p1b_backend_compare.json"))
    ap.add_argument("--max-test-docs", type=int, default=None)
    args = ap.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.is_file():
        print(f"ERROR: gold missing: {gold_path}")
        sys.exit(1)

    rows = _load_jsonl(gold_path)
    splits, meta = _document_level_split(rows)
    test_rows = splits["test"]
    if args.max_test_docs:
        keep = set(meta["test_docs"][: args.max_test_docs])
        test_rows = [r for r in test_rows if r["doc_id"] in keep]

    print(f"Gold test pairs: {len(test_rows)} across ~{meta['n_docs_test']} docs")
    print("Running heuristic + trained (MiniLM required for trained)…")

    y_true, y_h, y_t, stats = rows_to_edus_pairs(test_rows)
    labels = list(COARSE_LABELS) if COARSE_LABELS else sorted(set(y_true))

    m_h = metrics(y_true, y_h, labels)
    m_t = metrics(y_true, y_t, labels)

    out = {
        "split": "document_level_test",
        "seed": 42,
        "n_pairs": len(y_true),
        "stats": stats,
        "heuristic": m_h,
        "trained": m_t,
        "note": (
            "Pair-level classification only. Not full tree attachment F1. "
            "Does not claim cross-domain generalization."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== P1b results (GUM gold test, pair-level) ===")
    print(f"n_pairs={len(y_true)}")
    print(f"heuristic  acc={m_h['accuracy']:.4f}  macro-F1={m_h['macro_f1']:.4f}  w-F1={m_h['weighted_f1']:.4f}")
    print(f"trained    acc={m_t['accuracy']:.4f}  macro-F1={m_t['macro_f1']:.4f}  w-F1={m_t['weighted_f1']:.4f}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
