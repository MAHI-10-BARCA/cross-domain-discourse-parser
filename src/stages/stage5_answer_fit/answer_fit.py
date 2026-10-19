"""
Phase 5 — Answer-Fit Scoring

Primary: cosine(question, answer-EDU). Optional ranking of alternative EDUs.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_schemas import (
    Document, EDU, EDUPair, QUDQuestion, AnswerFitScore, save_json,
)

logger = logging.getLogger("discourse_parser.stage5")

_embed_model = None
_embed_name = None


def _load_minilm(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _embed_model, _embed_name
    if _embed_model is not None and _embed_name == model_name:
        return _embed_model
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model for answer-fit: %s", model_name)
        _embed_model = SentenceTransformer(model_name)
        _embed_name = model_name
        return _embed_model
    except Exception as e:
        logger.warning("MiniLM unavailable for answer-fit (%s). Using TF-IDF.", e)
        return None


def _embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    model = _load_minilm(model_name)
    if model is not None:
        return np.asarray(model.encode(texts, convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)
    from collections import Counter
    import math
    tokenized = [t.lower().split() for t in texts]
    df: Counter = Counter()
    for toks in tokenized:
        df.update(set(toks))
    n = max(len(texts), 1)
    vocab = sorted(df.keys())
    if not vocab:
        return np.zeros((len(texts), 1), dtype=np.float32)
    w2i = {w: i for i, w in enumerate(vocab)}
    mat = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, toks in enumerate(tokenized):
        tf = Counter(toks)
        for w, c in tf.items():
            if w in w2i:
                idf = math.log((n + 1) / (df[w] + 1)) + 1.0
                mat[i, w2i[w]] = (c / max(len(toks), 1)) * idf
    norms = np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)
    return mat / norms


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _to_unit(sim: float) -> float:
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


class AnswerFitScorer:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k_alternatives: int = 0,
    ):
        self.embedding_model = embedding_model
        self.top_k_alternatives = top_k_alternatives

    def score(
        self, questions: List[QUDQuestion], edus: List[EDU],
        pairs: Optional[List[EDUPair]] = None,
    ) -> Tuple[List[AnswerFitScore], Dict[str, Any]]:
        edu_map = {e.edu_id: e for e in edus}
        pair_map = {p.pair_id: p for p in (pairs or [])}

        targets: List[Tuple[QUDQuestion, EDU, str]] = []
        for q in questions:
            if not q.is_valid:
                continue
            pair = pair_map.get(q.pair_id)
            edu_j_id = (q.metadata or {}).get("edu_j_id")
            if not edu_j_id and pair:
                edu_j_id = pair.edu_j_id
            edu_j = edu_map.get(edu_j_id) if edu_j_id else None
            if edu_j is None and pair and pair.edu_j_index < len(edus):
                edu_j = edus[pair.edu_j_index]
            if edu_j is None:
                continue
            targets.append((q, edu_j, "paired"))
            if self.top_k_alternatives > 0:
                for e in edus:
                    if e.edu_id != edu_j.edu_id:
                        targets.append((q, e, "alternative"))

        if not targets:
            return [], {"num_scored": 0}

        q_texts = [t[0].question_text for t in targets]
        a_texts = [t[1].text for t in targets]
        embs = _embed_texts(q_texts + a_texts, self.embedding_model)
        n = len(targets)
        q_embs, a_embs = embs[:n], embs[n:]

        scores: List[AnswerFitScore] = []
        for i, (q, ans, role) in enumerate(targets):
            raw = _cosine(q_embs[i], a_embs[i])
            fit = _to_unit(raw)
            scores.append(AnswerFitScore(
                question_id=q.question_id,
                answer_edu_id=ans.edu_id,
                score=round(fit, 6),
                metadata={
                    "role": role,
                    "raw_cosine": round(raw, 6),
                    "pair_id": q.pair_id,
                    "method": "cosine_qa",
                },
            ))

        paired_scores = [s for s in scores if s.metadata.get("role") == "paired"]
        vals = [s.score for s in paired_scores]
        by_q: Dict[str, List[AnswerFitScore]] = {}
        for s in scores:
            by_q.setdefault(s.question_id, []).append(s)

        stats = {
            "num_questions": len(by_q),
            "num_scored": len(scores),
            "num_paired": len(paired_scores),
            "mean_paired_fit": round(float(np.mean(vals)), 4) if vals else 0.0,
            "max_paired_fit": round(float(np.max(vals)), 4) if vals else 0.0,
            "min_paired_fit": round(float(np.min(vals)), 4) if vals else 0.0,
            "top_k_alternatives": self.top_k_alternatives,
        }
        logger.info("Answer-fit: %d questions, %d paired scores (mean=%.3f)",
                    stats["num_questions"], stats["num_paired"], stats["mean_paired_fit"])
        return scores, stats

    def score_and_save(
        self, questions: List[QUDQuestion], edus: List[EDU],
        output_path: str | Path, pairs: Optional[List[EDUPair]] = None,
        document: Optional[Document] = None,
    ) -> Tuple[List[AnswerFitScore], Dict[str, Any]]:
        scores, stats = self.score(questions, edus, pairs=pairs)
        paired = sorted(
            [s for s in scores if s.metadata.get("role") == "paired"],
            key=lambda s: -s.score,
        )
        payload = {
            "document": document.to_dict() if document else None,
            "scores": [s.to_dict() for s in scores],
            "ranked_paired": [s.to_dict() for s in paired[:20]],
            "stats": stats,
        }
        save_json(payload, output_path)
        logger.info("Saved answer-fit scores to %s", output_path)
        return scores, stats


def score_answer_fit(
    questions: List[QUDQuestion], edus: List[EDU],
    pairs: Optional[List[EDUPair]] = None,
) -> Tuple[List[AnswerFitScore], Dict[str, Any]]:
    return AnswerFitScorer().score(questions, edus, pairs=pairs)
