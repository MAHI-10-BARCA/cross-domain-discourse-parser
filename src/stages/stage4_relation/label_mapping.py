"""
GUM eRST (V12.x) fine label → project 7-class inventory.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

COARSE_LABELS = [
    "Elaboration", "Explanation", "Contrast", "Result",
    "Background", "Joint", "Other",
]

FINE_TO_COARSE: Dict[str, str] = {
    "elaboration-additional": "Elaboration",
    "elaboration-attribute": "Elaboration",
    "explanation-evidence": "Explanation",
    "explanation-justify": "Explanation",
    "explanation-motivation": "Explanation",
    "adversative-contrast": "Contrast",
    "adversative-antithesis": "Contrast",
    "adversative-concession": "Contrast",
    "causal-cause": "Explanation",
    "causal-result": "Result",
    "context-background": "Background",
    "context-circumstance": "Background",
    "joint-list": "Joint",
    "joint-sequence": "Joint",
    "joint-other": "Joint",
    "joint-disjunction": "Joint",
    "same-unit": "Joint",
    "attribution-positive": "Other",
    "attribution-negative": "Other",
    "organization-preparation": "Other",
    "organization-phatic": "Other",
    "organization-heading": "Other",
    "evaluation-comment": "Other",
    "purpose-goal": "Other",
    "purpose-attribute": "Other",
    "contingency-condition": "Other",
    "restatement-repetition": "Other",
    "restatement-partial": "Other",
    "topic-question": "Other",
    "topic-solutionhood": "Other",
    "mode-manner": "Other",
    "mode-means": "Other",
}

MAPPING_VERSION = "gum_v12.1.0_map_v1"


def strip_nuclearity(label: str) -> str:
    if label.endswith("_r") or label.endswith("_m"):
        return label[:-2]
    return label


def map_label(fine: str) -> Tuple[Optional[str], str]:
    if not fine or fine in ("ROOT", "_"):
        return None, "root_or_empty"
    base = strip_nuclearity(fine)
    if base in FINE_TO_COARSE:
        return FINE_TO_COARSE[base], "mapped"
    return "Other", "unknown_fine_default_other"


def mapping_table_rows():
    return [{"fine": f, "coarse": c} for f, c in sorted(FINE_TO_COARSE.items())]
