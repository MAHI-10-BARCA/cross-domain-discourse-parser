"""
Phase 1.5 — Related EDU Pairing

Encode EDUs, compute cosine similarities, and select candidate pairs
using Top-K and maximum-distance constraints.

Primary embedding model: sentence-transformers/all-MiniLM-L6-v2
Fallback: simple TF-IDF + cosine (for environments without the model)
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import sys
import json

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_schemas import Document, EDU, EDUPair, save_json

logger = logging.getLogger("discourse_parser.stage1_5")


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

_model = None
_model_name = None


def _load_minilm(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load sentence-transformers model (lazy, cached)."""
    global _model, _model_name
    if _model is not None and _model_name == model_name:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
        _model_name = model_name
        return _model
    except Exception as e:
        logger.warning("sentence-transformers unavailable (%s). Using TF-IDF fallback.", e)
        return None


def embed_minilm(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    model = _load_minilm(model_name)
    if model is None:
        return embed_tfidf(texts)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


def embed_tfidf(texts: List[str]) -> np.ndarray:
    """
    Lightweight TF-IDF fallback (no external model required).
    Suitable for unit tests and environments without sentence-transformers.
    """
    from collections import Counter

    tokenized = [t.lower().split() for t in texts]
    df: Counter = Counter()
    for toks in tokenized:
        df.update(set(toks))
    n_docs = max(len(texts), 1)
    vocab = sorted(df.keys())
    if not vocab:
        return np.zeros((len(texts), 1), dtype=np.float32)
    word2idx = {w: i for i, w in enumerate(vocab)}
    mat = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, toks in enumerate(tokenized):
        tf = Counter(toks)
        for w, c in tf.items():
            if w in word2idx:
                idf = math.log((n_docs + 1) / (df[w] + 1)) + 1.0
                mat[i, word2idx[w]] = (c / max(len(toks), 1)) * idf
    # L2 normalize
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    mat = mat / norms
    return mat


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity. Assumes rows are L2-normalized (or normalizes them)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    normed = embeddings / norms
    return normed @ normed.T


# ---------------------------------------------------------------------------
# Pair selection strategies
# ---------------------------------------------------------------------------

def select_pairs_topk_distance(
    edus: List[EDU],
    sim_matrix: np.ndarray,
    top_k: int = 3,
    max_distance: int = 5,
    similarity_threshold: float = 0.0,
) -> List[EDUPair]:
    """
    For each EDU i, consider EDUs j where |i-j| <= max_distance and j > i
    (undirected pairs, stored with i < j). Keep the top_k highest-similarity
    partners per EDU, then deduplicate.

    Also produces an adjacent baseline set for comparison (distance == 1).
    """
    n = len(edus)
    if n < 2:
        return []

    candidates: Dict[Tuple[int, int], float] = {}

    for i in range(n):
        scores = []
        for j in range(n):
            if i == j:
                continue
            dist = abs(i - j)
            if dist > max_distance:
                continue
            a, b = (i, j) if i < j else (j, i)
            sim = float(sim_matrix[i, j])
            if sim < similarity_threshold:
                continue
            scores.append((b if i < j else a, sim, a, b))

        scores.sort(key=lambda x: -x[1])
        for _, sim, a, b in scores[:top_k]:
            key = (a, b)
            if key not in candidates or sim > candidates[key]:
                candidates[key] = sim

    pairs: List[EDUPair] = []
    doc_id = edus[0].doc_id
    for idx, ((i, j), sim) in enumerate(sorted(candidates.items(), key=lambda x: (x[0][0], x[0][1]))):
        pair = EDUPair(
            pair_id=f"{doc_id}_pair_{idx:03d}",
            doc_id=doc_id,
            edu_i_id=edus[i].edu_id,
            edu_j_id=edus[j].edu_id,
            edu_i_index=i,
            edu_j_index=j,
            distance=j - i,
            similarity=round(sim, 6),
            selected=True,
            metadata={
                "strategy": "topk_distance",
                "top_k": top_k,
                "max_distance": max_distance,
            },
        )
        pairs.append(pair)

    return pairs


def select_adjacent_baseline(edus: List[EDU], sim_matrix: np.ndarray) -> List[EDUPair]:
    """Simple adjacent-pair baseline (every consecutive EDU pair)."""
    pairs = []
    doc_id = edus[0].doc_id if edus else ""
    for i in range(len(edus) - 1):
        sim = float(sim_matrix[i, i + 1])
        pairs.append(EDUPair(
            pair_id=f"{doc_id}_adj_{i:03d}",
            doc_id=doc_id,
            edu_i_id=edus[i].edu_id,
            edu_j_id=edus[i + 1].edu_id,
            edu_i_index=i,
            edu_j_index=i + 1,
            distance=1,
            similarity=round(sim, 6),
            selected=True,
            metadata={"strategy": "adjacent_baseline"},
        ))
    return pairs


# ---------------------------------------------------------------------------
# Main pairer
# ---------------------------------------------------------------------------

class EDUPairer:
    """
    Stage 1.5: Generate candidate related EDU pairs.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3,
        max_distance: int = 5,
        similarity_threshold: float = 0.0,
        backend: str = "auto",  # "minilm" | "tfidf" | "auto"
    ):
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.max_distance = max_distance
        self.similarity_threshold = similarity_threshold
        self.backend = backend
        self._resolved_backend: Optional[str] = None

    def _resolve_backend(self) -> str:
        if self._resolved_backend:
            return self._resolved_backend
        if self.backend == "tfidf":
            self._resolved_backend = "tfidf"
        elif self.backend == "minilm":
            model = _load_minilm(self.embedding_model)
            self._resolved_backend = "minilm" if model is not None else "tfidf"
        else:
            model = _load_minilm(self.embedding_model)
            self._resolved_backend = "minilm" if model is not None else "tfidf"
        logger.info("Pairing embedding backend: %s", self._resolved_backend)
        return self._resolved_backend

    def pair(self, edus: List[EDU]) -> Tuple[List[EDUPair], List[EDUPair], Dict[str, Any]]:
        """
        Returns
        -------
        selected_pairs : List[EDUPair]   (Top-K + max-distance)
        adjacent_pairs : List[EDUPair]   (baseline)
        stats          : dict            (coverage metrics)
        """
        if len(edus) < 2:
            return [], [], {"num_edus": len(edus), "num_selected": 0, "num_adjacent": 0}

        backend = self._resolve_backend()
        texts = [e.text for e in edus]

        if backend == "minilm":
            embeddings = embed_minilm(texts, self.embedding_model)
        else:
            embeddings = embed_tfidf(texts)

        sim_matrix = cosine_similarity_matrix(embeddings)

        selected = select_pairs_topk_distance(
            edus, sim_matrix,
            top_k=self.top_k,
            max_distance=self.max_distance,
            similarity_threshold=self.similarity_threshold,
        )
        adjacent = select_adjacent_baseline(edus, sim_matrix)

        adj_keys = {(p.edu_i_index, p.edu_j_index) for p in adjacent}
        sel_keys = {(p.edu_i_index, p.edu_j_index) for p in selected}
        coverage = len(adj_keys & sel_keys) / max(len(adj_keys), 1)

        non_adj = [p for p in selected if p.distance > 1]

        stats = {
            "num_edus": len(edus),
            "num_selected": len(selected),
            "num_adjacent": len(adjacent),
            "num_non_adjacent_selected": len(non_adj),
            "adjacent_coverage": round(coverage, 4),
            "backend": backend,
            "top_k": self.top_k,
            "max_distance": self.max_distance,
            "mean_selected_similarity": round(
                float(np.mean([p.similarity for p in selected])) if selected else 0.0, 4
            ),
            "mean_adjacent_similarity": round(
                float(np.mean([p.similarity for p in adjacent])) if adjacent else 0.0, 4
            ),
        }
        logger.info(
            "Paired %d EDUs → %d selected pairs (%d non-adjacent), adjacent coverage=%.2f",
            len(edus), len(selected), len(non_adj), coverage,
        )
        return selected, adjacent, stats

    def pair_and_save(
        self,
        edus: List[EDU],
        output_path: str | Path,
        document: Optional[Document] = None,
    ) -> Tuple[List[EDUPair], List[EDUPair], Dict[str, Any]]:
        selected, adjacent, stats = self.pair(edus)
        payload = {
            "document": document.to_dict() if document else {"doc_id": edus[0].doc_id if edus else None},
            "edus": [e.to_dict() for e in edus],
            "selected_pairs": [p.to_dict() for p in selected],
            "adjacent_baseline": [p.to_dict() for p in adjacent],
            "stats": stats,
        }
        save_json(payload, output_path)
        logger.info("Saved pairing results to %s", output_path)
        return selected, adjacent, stats


def pair_edus(
    edus: List[EDU],
    top_k: int = 3,
    max_distance: int = 5,
    backend: str = "auto",
) -> Tuple[List[EDUPair], List[EDUPair], Dict[str, Any]]:
    """Convenience function."""
    pairer = EDUPairer(top_k=top_k, max_distance=max_distance, backend=backend)
    return pairer.pair(edus)
