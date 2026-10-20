#!/usr/bin/env python3
"""
Evaluate predicted discourse trees against GUM gold dependency trees.

Metrics:
  - Unlabeled attachment F1 (gold child-parent edges vs Stage-6 merge boundaries)
  - Labeled attachment F1 (same + coarse relation)

Usage:
  python scripts\\build_gold_trees.py --gum-root E:\\Final_year\\gum-12.1.0
  python scripts\\eval_gold_trees.py --max-docs 15 --relation-backend hybrid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Set, Tuple

try:
    import torch  # noqa: F401
except Exception:
    pass

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_schemas import EDU, DiscourseNode, DiscourseTree
from stages.stage1_5_pairing import EDUPairer
from stages.stage4_relation.relation_classifier import RelationClassifier
from stages.stage6_tree.tree_builder import build_pair_score_table, score_span_merge


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def doc_level_test_ids(all_doc_ids, seed=42, train_r=0.70, val_r=0.15):
    rng = np.random.RandomState(seed)
    docs = sorted(all_doc_ids)
    rng.shuffle(docs)
    n = len(docs)
    n_train = max(1, int(n * train_r))
    n_val = max(1, int(n * val_r))
    if n_train + n_val >= n:
        n_val = max(1, (n - n_train) // 2)
    return set(docs[n_train + n_val :])


def gold_edge_set(edges, labeled=False):
    s = set()
    for e in edges:
        a, b = int(e["child_index"]), int(e["parent_index"])
        key = (min(a, b), max(a, b))
        if labeled:
            s.add((key[0], key[1], e.get("coarse_label", "Other")))
        else:
            s.add(key)
    return s


def live_build_edges(edus, pairs, relations):
    n = len(edus)
    pair_table = build_pair_score_table(pairs, relations)
    constituents = []
    for i, edu in enumerate(edus):
        node = DiscourseNode(
            node_id=f"leaf_{i}", span_start=i, span_end=i, children=[], edu_ids=[edu.edu_id]
        )
        constituents.append((i, i, node))

    pred_unlab, pred_lab = set(), set()
    merge_log = []
    merge_id = 0
    min_merge_score = 0.10

    while len(constituents) > 1:
        best_idx, best_score, best_info = -1, -1.0, {}
        for i in range(len(constituents) - 1):
            left = (constituents[i][0], constituents[i][1])
            right = (constituents[i + 1][0], constituents[i + 1][1])
            if left[1] + 1 != right[0]:
                continue
            sc, info = score_span_merge(left, right, pair_table)
            if sc > best_score:
                best_score, best_idx, best_info = sc, i, info
        if best_idx < 0 or best_score < min_merge_score:
            while len(constituents) > 1:
                merge_id += 1
                L0, L1, ln = constituents[0]
                R0, R1, rn = constituents[1]
                pred_unlab.add((min(L1, R0), max(L1, R0)))
                pred_lab.add((min(L1, R0), max(L1, R0), "Joint"))
                parent = DiscourseNode(
                    node_id=f"m{merge_id}", span_start=L0, span_end=R1,
                    relation="Joint", confidence=0.1, children=[ln, rn],
                    edu_ids=ln.edu_ids + rn.edu_ids,
                )
                constituents = [(L0, R1, parent)] + constituents[2:]
                merge_log.append({"merge_id": merge_id, "span": [L0, R1], "relation": "Joint", "forced": True})
            break
        merge_id += 1
        L0, L1, ln = constituents[best_idx]
        R0, R1, rn = constituents[best_idx + 1]
        rel = best_info.get("relation", "Joint")
        pred_unlab.add((min(L1, R0), max(L1, R0)))
        pred_lab.add((min(L1, R0), max(L1, R0), rel))
        parent = DiscourseNode(
            node_id=f"m{merge_id}", span_start=L0, span_end=R1, relation=rel,
            confidence=float(best_info.get("confidence", best_score) or 0),
            children=[ln, rn], edu_ids=ln.edu_ids + rn.edu_ids,
        )
        constituents = constituents[:best_idx] + [(L0, R1, parent)] + constituents[best_idx + 2:]
        merge_log.append({"merge_id": merge_id, "span": [L0, R1], "relation": rel, "score": round(best_score, 4), "forced": False})

    tree = DiscourseTree(
        doc_id=edus[0].doc_id if edus else "doc",
        root=constituents[0][2], edu_list=list(edus),
        metadata={"merge_log": merge_log, "num_merges": merge_id},
    )
    return tree, pred_unlab, pred_lab, merge_log


def prf(pred: set, gold: set):
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gold) if gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-trees", default=str(ROOT / "data" / "gold" / "gum_v12_trees.jsonl"))
    ap.add_argument("--split", default="test", choices=["test", "all"])
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--pair-backend", default="minilm")
    ap.add_argument("--min-sim", type=float, default=0.10)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--relation-backend", default="hybrid", choices=["heuristic", "trained", "hybrid"])
    ap.add_argument("--out", default=str(ROOT / "outputs" / "evaluation" / "gold_tree_eval.json"))
    args = ap.parse_args()

    gold_path = Path(args.gold_trees)
    if not gold_path.is_file():
        print(f"ERROR: gold trees missing: {gold_path}")
        print("Run: python scripts\\build_gold_trees.py --gum-root E:\\Final_year\\gum-12.1.0")
        sys.exit(1)

    docs = load_jsonl(gold_path)
    if args.split == "test":
        test_ids = doc_level_test_ids([d["doc_id"] for d in docs])
        docs = [d for d in docs if d["doc_id"] in test_ids]
    if args.max_docs:
        docs = docs[: args.max_docs]

    print(f"Gold tree eval: n_docs={len(docs)} relation={args.relation_backend}")

    pairer = EDUPairer(backend=args.pair_backend, top_k=args.top_k, similarity_threshold=args.min_sim)
    clf = RelationClassifier(backend=args.relation_backend)

    micro_gu, micro_pu = set(), set()
    micro_gl, micro_pl = set(), set()
    per_doc = []

    for di, doc in enumerate(docs, 1):
        doc_id = doc["doc_id"]
        edus = []
        offset = 0
        for e in doc["edus"]:
            text = e["text"]
            edus.append(EDU(
                edu_id=f"{doc_id}_e{e['index']}", doc_id=doc_id, text=text,
                index=int(e["index"]), start_char=offset, end_char=offset + len(text),
            ))
            offset += len(text) + 1

        selected, _, _ = pairer.pair(edus)
        relations, _ = clf.classify(selected, edus)
        tree, pred_u, pred_l, merge_log = live_build_edges(edus, selected, relations)

        gold_u = gold_edge_set(doc["edges"], labeled=False)
        gold_l = gold_edge_set(doc["edges"], labeled=True)

        for a, b in gold_u:
            micro_gu.add((doc_id, a, b))
        for a, b in pred_u:
            micro_pu.add((doc_id, a, b))
        for a, b, lab in gold_l:
            micro_gl.add((doc_id, a, b, lab))
        for a, b, lab in pred_l:
            micro_pl.add((doc_id, a, b, lab))

        up, ur, uf = prf(pred_u, gold_u)
        lp, lr, lf = prf(pred_l, gold_l)
        per_doc.append({
            "doc_id": doc_id, "n_edus": doc["n_edus"],
            "n_gold_edges": len(gold_u), "n_pred_edges": len(pred_u),
            "unlabeled_f1": round(uf, 4), "labeled_f1": round(lf, 4),
        })
        if di % 5 == 0 or di == len(docs):
            print(f"  … {di}/{len(docs)}")

    up, ur, uf = prf(micro_pu, micro_gu)
    lp, lr, lf = prf(micro_pl, micro_gl)

    out = {
        "n_docs": len(docs),
        "relation_backend": args.relation_backend,
        "pair_backend": args.pair_backend,
        "unlabeled_attachment_micro": {"precision": round(up, 4), "recall": round(ur, 4), "f1": round(uf, 4)},
        "labeled_attachment_micro": {"precision": round(lp, 4), "recall": round(lr, 4), "f1": round(lf, 4)},
        "macro_unlabeled_f1": round(float(np.mean([d["unlabeled_f1"] for d in per_doc])), 4) if per_doc else 0.0,
        "macro_labeled_f1": round(float(np.mean([d["labeled_f1"] for d in per_doc])), 4) if per_doc else 0.0,
        "per_doc": per_doc,
        "note": "Gold = GUM RST dependency edges. Pred = Stage-6 adjacent merge boundaries.",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== Gold tree evaluation ===")
    print(f"n_docs={len(docs)}")
    print(f"Unlabeled attachment micro: P={up:.4f} R={ur:.4f} F1={uf:.4f}")
    print(f"Labeled attachment micro:   P={lp:.4f} R={lr:.4f} F1={lf:.4f}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
