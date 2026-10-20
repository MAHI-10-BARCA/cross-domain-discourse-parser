"""Hybrid relation backend: strong cues override Joint-heavy trained labels."""

from __future__ import annotations

from typing import List, Tuple, Dict, Any

from data_schemas import EDU, EDUPair, RelationPrediction

CUE_LABELS = {"Contrast", "Result", "Background", "Explanation", "Elaboration"}


def merge_heuristic_and_trained(
    heur: List[RelationPrediction],
    trained: List[RelationPrediction],
) -> Tuple[List[RelationPrediction], Dict[str, Any]]:
    out: List[RelationPrediction] = []
    for h, tr in zip(heur, trained):
        if h.label in CUE_LABELS and float(h.confidence) >= 0.55:
            chosen, src = h, "heuristic_cue"
        elif tr.label not in ("Joint", "Other") and float(tr.confidence) >= 0.35:
            chosen, src = tr, "trained"
        elif h.label in CUE_LABELS:
            chosen, src = h, "heuristic_cue_soft"
        else:
            chosen, src = tr, "trained_default"
        meta = dict(getattr(chosen, "metadata", {}) or {})
        meta["hybrid_source"] = src
        meta["heuristic_label"] = h.label
        meta["trained_label"] = tr.label
        out.append(
            RelationPrediction(
                prediction_id=chosen.prediction_id,
                pair_id=chosen.pair_id,
                span_id=chosen.span_id,
                label=chosen.label,
                confidence=float(chosen.confidence),
                features=getattr(chosen, "features", {}) or {},
                metadata=meta,
            )
        )
    counts: Dict[str, int] = {}
    for r in out:
        counts[r.label] = counts.get(r.label, 0) + 1
    stats = {
        "num_pairs": len(out),
        "label_counts": counts,
        "mean_confidence": float(sum(r.confidence for r in out) / max(len(out), 1)),
        "backend": "hybrid",
    }
    return out, stats
