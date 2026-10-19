"""
Evaluation metrics for discourse parsing.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter

from data_schemas import (
    EDU, EDUPair, RelationPrediction, DiscourseNode, DiscourseTree,
)


def tree_structure_stats(tree: DiscourseTree) -> Dict[str, Any]:
    root = tree.root
    n_edus = len(tree.edu_list)

    def height(node: DiscourseNode) -> int:
        if not node.children:
            return 0
        return 1 + max(height(c) for c in node.children)

    def count_nodes(node: DiscourseNode) -> Tuple[int, int]:
        if not node.children:
            return 0, 1
        inn, lf = 1, 0
        for c in node.children:
            i, l = count_nodes(c)
            inn += i
            lf += l
        return inn, lf

    relations: List[str] = []

    def collect_rel(node: DiscourseNode):
        if node.relation:
            relations.append(node.relation)
        for c in node.children:
            collect_rel(c)

    collect_rel(root)
    internal, leaves = count_nodes(root)

    return {
        "num_edus": n_edus,
        "num_internal_nodes": internal,
        "num_leaves": leaves,
        "height": height(root),
        "relation_distribution": dict(Counter(relations)),
        "root_relation": root.relation,
        "root_confidence": root.confidence,
    }


def pair_attachment_metrics(
    predicted_pairs: List[EDUPair],
    gold_pairs: List[Tuple[int, int]],
) -> Dict[str, float]:
    pred: Set[Tuple[int, int]] = set()
    for p in predicted_pairs:
        a, b = sorted((p.edu_i_index, p.edu_j_index))
        pred.add((a, b))
    gold: Set[Tuple[int, int]] = set()
    for a, b in gold_pairs:
        gold.add(tuple(sorted((a, b))))

    if not pred and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}

    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def relation_label_metrics(
    predicted: List[RelationPrediction],
    gold: Dict[str, str],
) -> Dict[str, Any]:
    matched = total = 0
    conf_matrix: Dict[str, Counter] = {}
    for r in predicted:
        if r.pair_id not in gold:
            continue
        total += 1
        g = gold[r.pair_id]
        if r.label == g:
            matched += 1
        conf_matrix.setdefault(g, Counter())[r.label] += 1

    acc = matched / total if total else 0.0
    return {
        "accuracy": round(acc, 4),
        "n_compared": total,
        "n_correct": matched,
        "confusion": {k: dict(v) for k, v in conf_matrix.items()},
    }


def evaluate_pipeline_result(
    result: Dict[str, Any],
    gold_pairs: Optional[List[Tuple[int, int]]] = None,
    gold_relations: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"doc_id": result.get("doc_id")}

    tree = result.get("tree")
    if tree is not None:
        report["structure"] = tree_structure_stats(tree)

    if gold_pairs is not None and result.get("pairs"):
        report["attachment"] = pair_attachment_metrics(result["pairs"], gold_pairs)

    if gold_relations is not None and result.get("relations"):
        report["relation_labels"] = relation_label_metrics(result["relations"], gold_relations)

    report["intrinsic"] = {
        "num_edus": result.get("num_edus"),
        "num_pairs": len(result.get("pairs") or []),
        "pair_stats": result.get("pair_stats"),
        "relation_stats": result.get("relation_stats"),
        "tree_stats": result.get("tree_stats"),
    }
    for key in ("qud_stats", "salience_stats", "answer_fit_stats"):
        if result.get(key):
            report["intrinsic"][key] = result[key]

    return report
