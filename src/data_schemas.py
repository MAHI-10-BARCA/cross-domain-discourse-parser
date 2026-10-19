"""
Shared data schemas for the discourse parsing pipeline.
All intermediate results must conform to these structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path


@dataclass
class Document:
    """Input document."""
    doc_id: str
    text: str
    domain: Optional[str] = None          # e.g. news, finance, medicine, law
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Document":
        return cls(**d)


@dataclass
class EDU:
    """Elementary Discourse Unit (sentence-level for Phase 1)."""
    edu_id: str                           # stable identifier, e.g. "doc1_edu_000"
    doc_id: str
    text: str
    index: int                            # 0-based position in document
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EDU":
        return cls(**d)


@dataclass
class EDUPair:
    """Candidate pair of EDUs."""
    pair_id: str
    doc_id: str
    edu_i_id: str
    edu_j_id: str
    edu_i_index: int
    edu_j_index: int
    distance: int                         # |i - j|
    similarity: float
    selected: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EDUPair":
        return cls(**d)


@dataclass
class QUDQuestion:
    """Generated Question Under Discussion."""
    question_id: str
    pair_id: str
    doc_id: str
    question_text: str
    category: Optional[str] = None        # e.g. what, why, how, when, ...
    specificity: Optional[str] = None     # coarse / fine
    generation_config: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QUDQuestion":
        return cls(**d)


@dataclass
class SalienceScore:
    """Salience components and final score for a question."""
    question_id: str
    pair_id: str
    centrality: float
    answerability: float
    final_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SalienceScore":
        return cls(**d)


@dataclass
class RelationPrediction:
    """Predicted discourse relation for a pair or span."""
    prediction_id: str
    pair_id: Optional[str]
    span_id: Optional[str]
    label: str
    confidence: float
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RelationPrediction":
        return cls(**d)


@dataclass
class AnswerFitScore:
    """Compatibility between a question and a candidate answer EDU."""
    question_id: str
    answer_edu_id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnswerFitScore":
        return cls(**d)


@dataclass
class DiscourseNode:
    """Node in the hierarchical discourse tree."""
    node_id: str
    span_start: int                       # inclusive EDU index
    span_end: int                         # inclusive EDU index
    relation: Optional[str] = None
    confidence: Optional[float] = None
    children: List["DiscourseNode"] = field(default_factory=list)
    edu_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiscourseNode":
        children = [cls.from_dict(c) for c in d.get("children", [])]
        return cls(
            node_id=d["node_id"],
            span_start=d["span_start"],
            span_end=d["span_end"],
            relation=d.get("relation"),
            confidence=d.get("confidence"),
            children=children,
            edu_ids=d.get("edu_ids", []),
            metadata=d.get("metadata", {}),
        )


@dataclass
class DiscourseTree:
    """Complete discourse tree for a document."""
    doc_id: str
    root: DiscourseNode
    edu_list: List[EDU] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "root": self.root.to_dict(),
            "edu_list": [e.to_dict() for e in self.edu_list],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiscourseTree":
        return cls(
            doc_id=d["doc_id"],
            root=DiscourseNode.from_dict(d["root"]),
            edu_list=[EDU.from_dict(e) for e in d.get("edu_list", [])],
            metadata=d.get("metadata", {}),
        )


# ---------- Serialization helpers ----------

def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "to_dict"):
        data = obj.to_dict()
    elif isinstance(obj, list):
        data = [x.to_dict() if hasattr(x, "to_dict") else x for x in obj]
    else:
        data = obj
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
