"""
Phase 6 — Bottom-Up Discourse Tree Construction

Greedy hierarchical clustering over EDUs guided by pair similarity
and relation confidence. Produces a binary discourse tree.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_schemas import (
    Document, EDU, EDUPair, RelationPrediction,
    DiscourseNode, DiscourseTree, save_json,
)

logger = logging.getLogger("discourse_parser.stage6")


def _pair_key(i: int, j: int) -> Tuple[int, int]:
    return (min(i, j), max(i, j))


def build_pair_score_table(
    pairs: List[EDUPair],
    relations: Optional[List[RelationPrediction]] = None,
    w_sim: float = 0.50,
    w_rel: float = 0.50,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    rel_map: Dict[str, RelationPrediction] = {}
    if relations:
        for r in relations:
            if r.pair_id:
                rel_map[r.pair_id] = r
    table: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for p in pairs:
        key = _pair_key(p.edu_i_index, p.edu_j_index)
        rel = rel_map.get(p.pair_id)
        sim = float(p.similarity or 0.0)
        sim_n = max(0.0, min(1.0, sim if sim > 0 else (sim + 1) / 2))
        conf = float(rel.confidence) if rel else 0.3
        label = rel.label if rel else "Other"
        score = w_sim * sim_n + w_rel * conf
        table[key] = {
            "score": score, "similarity": sim, "relation": label,
            "confidence": conf, "pair_id": p.pair_id, "distance": p.distance,
        }
    return table


def score_span_merge(
    left: Tuple[int, int], right: Tuple[int, int],
    pair_table: Dict[Tuple[int, int], Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    L0, L1 = left
    R0, R1 = right
    best_score = -1.0
    best_info: Dict[str, Any] = {
        "relation": "Joint", "confidence": 0.0, "similarity": 0.0, "pair_id": None,
    }
    for i in range(L0, L1 + 1):
        for j in range(R0, R1 + 1):
            info = pair_table.get(_pair_key(i, j))
            if not info:
                continue
            bonus = 0.05 if (i == L1 and j == R0) else 0.0
            s = info["score"] + bonus
            if s > best_score:
                best_score = s
                best_info = {
                    "relation": info["relation"], "confidence": info["confidence"],
                    "similarity": info["similarity"], "pair_id": info["pair_id"],
                    "boundary_pair": (i, j),
                }
    if best_score < 0:
        if L1 + 1 == R0:
            return 0.15, {
                "relation": "Joint", "confidence": 0.15, "similarity": 0.0,
                "pair_id": None, "boundary_pair": (L1, R0), "fallback": True,
            }
        return -1.0, best_info
    return best_score, best_info


class BottomUpTreeBuilder:
    def __init__(self, w_sim: float = 0.50, w_rel: float = 0.50, min_merge_score: float = 0.10):
        self.w_sim = w_sim
        self.w_rel = w_rel
        self.min_merge_score = min_merge_score

    def build(
        self, edus: List[EDU], pairs: List[EDUPair],
        relations: Optional[List[RelationPrediction]] = None,
        document: Optional[Document] = None,
    ) -> Tuple[DiscourseTree, Dict[str, Any]]:
        n = len(edus)
        if n == 0:
            empty = DiscourseNode(node_id="empty", span_start=0, span_end=-1)
            tree = DiscourseTree(doc_id=document.doc_id if document else "empty", root=empty)
            return tree, {"num_edus": 0, "num_merges": 0}

        pair_table = build_pair_score_table(pairs, relations, w_sim=self.w_sim, w_rel=self.w_rel)
        constituents: List[Tuple[int, int, DiscourseNode]] = []
        for i, edu in enumerate(edus):
            node = DiscourseNode(
                node_id=f"leaf_{i}", span_start=i, span_end=i,
                relation=None, confidence=None, children=[],
                edu_ids=[edu.edu_id], metadata={"text": edu.text[:80]},
            )
            constituents.append((i, i, node))

        merge_log: List[Dict[str, Any]] = []
        merge_id = 0

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

            if best_idx < 0 or best_score < self.min_merge_score:
                logger.info("No positive merge left (%d constituents); force-joining.", len(constituents))
                while len(constituents) > 1:
                    merge_id += 1
                    L0, L1, left_node = constituents[0]
                    R0, R1, right_node = constituents[1]
                    parent = DiscourseNode(
                        node_id=f"merge_{merge_id}", span_start=L0, span_end=R1,
                        relation="Joint", confidence=0.1,
                        children=[left_node, right_node],
                        edu_ids=left_node.edu_ids + right_node.edu_ids,
                        metadata={"forced": True},
                    )
                    constituents = [(L0, R1, parent)] + constituents[2:]
                    merge_log.append({"merge_id": merge_id, "span": [L0, R1], "relation": "Joint", "score": 0.1, "forced": True})
                break

            merge_id += 1
            L0, L1, left_node = constituents[best_idx]
            R0, R1, right_node = constituents[best_idx + 1]
            parent = DiscourseNode(
                node_id=f"merge_{merge_id}", span_start=L0, span_end=R1,
                relation=best_info.get("relation", "Joint"),
                confidence=round(float(best_info.get("confidence", best_score)), 4),
                children=[left_node, right_node],
                edu_ids=left_node.edu_ids + right_node.edu_ids,
                metadata={
                    "merge_score": round(best_score, 4),
                    "pair_id": best_info.get("pair_id"),
                    "similarity": best_info.get("similarity"),
                    "boundary_pair": best_info.get("boundary_pair"),
                },
            )
            constituents = constituents[:best_idx] + [(L0, R1, parent)] + constituents[best_idx + 2:]
            merge_log.append({
                "merge_id": merge_id, "span": [L0, R1],
                "relation": parent.relation, "score": round(best_score, 4), "forced": False,
            })

        root = constituents[0][2]
        doc_id = document.doc_id if document else (edus[0].doc_id if edus else "unknown")
        tree = DiscourseTree(
            doc_id=doc_id, root=root, edu_list=list(edus),
            metadata={
                "num_edus": n, "num_merges": merge_id, "merge_log": merge_log,
                "algorithm": "greedy_bottom_up_adjacent",
            },
        )
        stats = {
            "num_edus": n, "num_merges": merge_id, "num_pairs_used": len(pair_table),
            "root_span": [root.span_start, root.span_end],
            "relation_counts": _count_relations(root),
        }
        logger.info("Tree built: %d EDUs → %d merges, relations=%s", n, merge_id, stats["relation_counts"])
        return tree, stats

    def build_and_save(
        self, edus: List[EDU], pairs: List[EDUPair], output_path: str | Path,
        relations: Optional[List[RelationPrediction]] = None,
        document: Optional[Document] = None,
    ) -> Tuple[DiscourseTree, Dict[str, Any]]:
        tree, stats = self.build(edus, pairs, relations=relations, document=document)
        save_json({"tree": tree.to_dict(), "stats": stats}, output_path)
        logger.info("Saved discourse tree to %s", output_path)
        return tree, stats


def _count_relations(node: DiscourseNode) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if node.relation:
        counts[node.relation] = counts.get(node.relation, 0) + 1
    for c in node.children:
        for k, v in _count_relations(c).items():
            counts[k] = counts.get(k, 0) + v
    return counts


def tree_to_ascii(node: DiscourseNode, edus: List[EDU], indent: int = 0) -> str:
    pad = "  " * indent
    if not node.children:
        text = edus[node.span_start].text[:50] if node.span_start < len(edus) else "?"
        return f"{pad}\u2022 [{node.span_start}] {text}\n"
    lines = f"{pad}\u252c\u2500 {node.relation or '?'} (conf={node.confidence}) span=[{node.span_start}:{node.span_end}]\n"
    for c in node.children:
        lines += tree_to_ascii(c, edus, indent + 1)
    return lines


def build_discourse_tree(
    edus: List[EDU], pairs: List[EDUPair],
    relations: Optional[List[RelationPrediction]] = None,
) -> Tuple[DiscourseTree, Dict[str, Any]]:
    return BottomUpTreeBuilder().build(edus, pairs, relations=relations)
