"""
Phase 3 — Neural Salience Scoring

A. Semantic centrality (MiniLM vs document/pair context)
B. Answerability proxy (NLI cross-encoder, cosine fallback)
salience = w_c * centrality + w_a * answerability
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
    Document, EDU, EDUPair, QUDQuestion, SalienceScore, save_json,
)

logger = logging.getLogger("discourse_parser.stage3")

_embed_model = None
_embed_name = None


def _load_minilm(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _embed_model, _embed_name
    if _embed_model is not None and _embed_name == model_name:
        return _embed_model
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model for salience: %s", model_name)
        _embed_model = SentenceTransformer(model_name)
        _embed_name = model_name
        return _embed_model
    except Exception as e:
        logger.warning("MiniLM unavailable for salience (%s). Using TF-IDF.", e)
        return None


def _embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    model = _load_minilm(model_name)
    if model is not None:
        emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)
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
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    return mat / norms


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _to_unit_interval(sim: float) -> float:
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


_nli_model = None
_nli_name = None


def _load_nli(model_name: str = "cross-encoder/nli-deberta-v3-small"):
    global _nli_model, _nli_name
    if _nli_model is not None and _nli_name == model_name:
        return _nli_model
    candidates = [
        model_name,
        "cross-encoder/nli-deberta-v3-small",
        "cross-encoder/nli-MiniLM2-L6-H768",
        "cross-encoder/nli-distilroberta-base",
    ]
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    try:
        from sentence_transformers import CrossEncoder
    except Exception as e:
        logger.warning("CrossEncoder not available (%s).", e)
        return None
    for name in ordered:
        try:
            logger.info("Loading NLI cross-encoder: %s", name)
            _nli_model = CrossEncoder(name)
            _nli_name = name
            return _nli_model
        except Exception as e:
            logger.warning("Could not load %s (%s)", name, e)
            continue
    return None


def answerability_nli(question: str, answer_edu: str, model_name: str) -> Tuple[float, str]:
    model = _load_nli(model_name)
    if model is None:
        return -1.0, "unavailable"
    try:
        pair = [(answer_edu, question)]
        scores = model.predict(pair)
        s = scores[0]
        if hasattr(s, "__len__") and not isinstance(s, (str, bytes)):
            arr = np.asarray(s, dtype=np.float32)
            if arr.size == 3:
                if arr.min() < 0 or arr.max() > 1.5:
                    exp = np.exp(arr - arr.max())
                    arr = exp / exp.sum()
                entail = float(arr[1]) if arr.size > 1 else float(arr.max())
                return max(0.0, min(1.0, entail)), "nli_entailment"
            val = float(arr.ravel()[0])
        else:
            val = float(s)
        if val < 0 or val > 1:
            val = 1.0 / (1.0 + np.exp(-val))
        return max(0.0, min(1.0, val)), "nli_score"
    except Exception as e:
        logger.warning("NLI predict failed (%s)", e)
        return -1.0, "nli_error"


def answerability_cosine(question_emb: np.ndarray, answer_emb: np.ndarray) -> float:
    return _to_unit_interval(_cosine(question_emb, answer_emb))


class SalienceScorer:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        nli_model: str = "cross-encoder/nli-deberta-v3-small",
        centrality_weight: float = 0.50,
        answerability_weight: float = 0.50,
        context_mode: str = "pair",
        use_nli: bool = True,
    ):
        self.embedding_model = embedding_model
        self.nli_model = nli_model
        total = centrality_weight + answerability_weight
        if total <= 0:
            self.centrality_weight = 0.5
            self.answerability_weight = 0.5
        else:
            self.centrality_weight = centrality_weight / total
            self.answerability_weight = answerability_weight / total
        self.context_mode = context_mode
        self.use_nli = use_nli

    def score(
        self,
        questions: List[QUDQuestion],
        edus: List[EDU],
        pairs: Optional[List[EDUPair]] = None,
    ) -> Tuple[List[SalienceScore], Dict[str, Any]]:
        edu_map = {e.edu_id: e for e in edus}
        pair_map = {p.pair_id: p for p in (pairs or [])}

        q_texts, context_texts, answer_texts, meta = [], [], [], []

        for q in questions:
            pair = pair_map.get(q.pair_id)
            edu_i_id = (q.metadata or {}).get("edu_i_id") or (pair.edu_i_id if pair else None)
            edu_j_id = (q.metadata or {}).get("edu_j_id") or (pair.edu_j_id if pair else None)
            edu_i = edu_map.get(edu_i_id) if edu_i_id else None
            edu_j = edu_map.get(edu_j_id) if edu_j_id else None
            if edu_i is None or edu_j is None:
                if pair:
                    edu_i = edus[pair.edu_i_index] if pair.edu_i_index < len(edus) else None
                    edu_j = edus[pair.edu_j_index] if pair.edu_j_index < len(edus) else None

            q_texts.append(q.question_text if q.is_valid else "")
            if self.context_mode == "document":
                ctx = " ".join(e.text for e in edus)
            else:
                ctx = f"{edu_i.text if edu_i else ''} {edu_j.text if edu_j else ''}".strip()
            context_texts.append(ctx or " ")
            answer_texts.append(edu_j.text if edu_j else " ")
            meta.append((q, edu_i, edu_j))

        all_texts = q_texts + context_texts + answer_texts
        embs = _embed_texts(all_texts, self.embedding_model)
        n = len(questions)
        q_embs, ctx_embs, ans_embs = embs[:n], embs[n:2*n], embs[2*n:3*n]

        scores: List[SalienceScore] = []
        nli_used = cosine_fallback = 0

        for i, (q, edu_i, edu_j) in enumerate(meta):
            if not q.is_valid:
                scores.append(SalienceScore(
                    question_id=q.question_id, pair_id=q.pair_id,
                    centrality=0.0, answerability=0.0, final_score=0.0,
                    metadata={"skipped": True, "reason": "invalid_question"},
                ))
                continue

            raw_cent = _cosine(q_embs[i], ctx_embs[i])
            centrality = _to_unit_interval(raw_cent)

            ans_score, ans_method = -1.0, "cosine_fallback"
            if self.use_nli and edu_j is not None:
                ans_score, ans_method = answerability_nli(
                    q.question_text, edu_j.text, self.nli_model
                )
            if ans_score < 0:
                ans_score = answerability_cosine(q_embs[i], ans_embs[i])
                ans_method = "cosine_fallback"
                cosine_fallback += 1
            else:
                nli_used += 1

            final = self.centrality_weight * centrality + self.answerability_weight * ans_score
            scores.append(SalienceScore(
                question_id=q.question_id, pair_id=q.pair_id,
                centrality=round(centrality, 6),
                answerability=round(float(ans_score), 6),
                final_score=round(float(final), 6),
                metadata={
                    "raw_centrality_cosine": round(raw_cent, 6),
                    "answerability_method": ans_method,
                    "centrality_weight": self.centrality_weight,
                    "answerability_weight": self.answerability_weight,
                    "context_mode": self.context_mode,
                },
            ))

        valid_scores = [s for s in scores if not s.metadata.get("skipped")]
        finals = [s.final_score for s in valid_scores]
        stats = {
            "num_questions": len(questions),
            "num_scored": len(valid_scores),
            "nli_used": nli_used,
            "cosine_fallback": cosine_fallback,
            "mean_centrality": round(float(np.mean([s.centrality for s in valid_scores])), 4) if valid_scores else 0.0,
            "mean_answerability": round(float(np.mean([s.answerability for s in valid_scores])), 4) if valid_scores else 0.0,
            "mean_final": round(float(np.mean(finals)), 4) if finals else 0.0,
            "max_final": round(float(np.max(finals)), 4) if finals else 0.0,
            "min_final": round(float(np.min(finals)), 4) if finals else 0.0,
            "weights": {"centrality": self.centrality_weight, "answerability": self.answerability_weight},
        }
        logger.info(
            "Salience: scored %d questions (nli=%d, cosine_fb=%d, mean_final=%.3f)",
            len(valid_scores), nli_used, cosine_fallback, stats["mean_final"],
        )
        return scores, stats

    def score_and_save(
        self, questions: List[QUDQuestion], edus: List[EDU],
        output_path: str | Path, pairs: Optional[List[EDUPair]] = None,
        document: Optional[Document] = None,
    ) -> Tuple[List[SalienceScore], Dict[str, Any]]:
        scores, stats = self.score(questions, edus, pairs=pairs)
        ranked = sorted(
            [s for s in scores if not s.metadata.get("skipped")],
            key=lambda s: -s.final_score,
        )
        payload = {
            "document": document.to_dict() if document else None,
            "scores": [s.to_dict() for s in scores],
            "ranked_top": [s.to_dict() for s in ranked[:20]],
            "stats": stats,
        }
        save_json(payload, output_path)
        logger.info("Saved salience scores to %s", output_path)
        return scores, stats


def score_salience(
    questions: List[QUDQuestion], edus: List[EDU],
    pairs: Optional[List[EDUPair]] = None,
) -> Tuple[List[SalienceScore], Dict[str, Any]]:
    return SalienceScorer().score(questions, edus, pairs=pairs)
