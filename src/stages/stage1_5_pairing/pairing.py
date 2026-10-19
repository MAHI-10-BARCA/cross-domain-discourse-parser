"""
Phase 1.5 — Related EDU Pairing

Encode EDUs, compute cosine similarities, and select candidate pairs
by GLOBAL Top-K similarity (not adjacency-forced).

Primary embedding model: sentence-transformers/all-MiniLM-L6-v2
Fallback: simple TF-IDF + cosine (for environments without the model)

Adjacency baseline is computed only for evaluation comparison.
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


_model = None
_model_name = None


def _load_minilm(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
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
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    mat = mat / norms
    return mat


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    normed = embeddings / norms
    return normed @ normed.T


def select_pairs_global_topk(
    edus: List[EDU],
    sim_matrix: np.ndarray,
    top_k: int = 3,
    similarity_threshold: float = 0.25,
    max_distance: Optional[int] = None,
) -> List[EDUPair]:
    """
    Primary strategy: select strongly related EDUs by similarity, globally.

    For each EDU i, rank all other EDUs j by cosine similarity and keep
    the top_k partners above similarity_threshold.

    max_distance is OPTIONAL. Default is None (no distance constraint).
    Adjacent pairs are NOT preferred — only similarity decides.
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
            if max_distance is not None and dist > max_distance:
                continue
            sim = float(sim_matrix[i, j])
            if sim < similarity_threshold:
                continue
            a, b = (i, j) if i < j else (j, i)
            scores.append((sim, a, b))

        scores.sort(key=lambda x: -x[0])
        for sim, a, b in scores[:top_k]:
            key = (a, b)
            if key not in candidates or sim > candidates[key]:
                candidates[key] = sim

    pairs: List[EDUPair] = []
    doc_id = edus[0].doc_id
    for idx, ((i, j), sim) in enumerate(
        sorted(candidates.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    ):
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
                "strategy": "global_topk",
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "max_distance": max_distance,
            },
        )
        pairs.append(pair)

    return pairs


def select_adjacent_baseline(edus: List[EDU], sim_matrix: np.ndarray) -> List[EDUPair]:
    """Adjacent-pair baseline — for EVALUATION COMPARISON only. Not primary."""
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


class EDUPairer:
    """Stage 1.5: global Top-K by similarity (NOT adjacency)."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3,
        similarity_threshold: float = 0.25,
        max_distance: Optional[int] = None,
        backend: str = "auto",
    ):
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.max_distance = max_distance
        self.backend = backend
        self._resolved_backend: Optional[str] = None

    def _resolve_backend(self) -> str:
        if self._resolved_backend:
            return self._resolved_backend
        if self.backend == "tfidf":
            self._resolved_backend = "tfidf"
        else:
            model = _load_minilm(self.embedding_model)
            self._resolved_backend = "minilm" if model is not None else "tfidf"
        logger.info("Pairing embedding backend: %s", self._resolved_backend)
        return self._resolved_backend

    def pair(self, edus: List[EDU]) -> Tuple[List[EDUPair], List[EDUPair], Dict[str, Any]]:
        if len(edus) < 2:
            return [], [], {"num_edus": len(edus), "num_selected": 0, "num_adjacent_baseline": 0}

        backend = self._resolve_backend()
        texts = [e.text for e in edus]
        embeddings = embed_minilm(texts, self.embedding_model) if backend == "minilm" else embed_tfidf(texts)
        sim_matrix = cosine_similarity_matrix(embeddings)

        selected = select_pairs_global_topk(
            edus, sim_matrix,
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold,
            max_distance=self.max_distance,
        )
        adjacent = select_adjacent_baseline(edus, sim_matrix)

        adj_keys = {(p.edu_i_index, p.edu_j_index) for p in adjacent}
        sel_keys = {(p.edu_i_index, p.edu_j_index) for p in selected}
        overlap = len(adj_keys & sel_keys) / max(len(adj_keys), 1)
        non_adj = [p for p in selected if p.distance > 1]
        distances = [p.distance for p in selected]

        stats = {
            "num_edus": len(edus),
            "num_selected": len(selected),
            "num_adjacent_baseline": len(adjacent),
            "num_non_adjacent_selected": len(non_adj),
            "fraction_selected_that_are_adjacent": round(
                (len(selected) - len(non_adj)) / max(len(selected), 1), 4
            ),
            "adjacent_overlap_with_selected": round(overlap, 4),
            "mean_pair_distance": round(float(np.mean(distances)) if distances else 0.0, 2),
            "backend": backend,
            "strategy": "global_topk",
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "max_distance": self.max_distance,
            "mean_selected_similarity": round(
                float(np.mean([p.similarity for p in selected])) if selected else 0.0, 4
            ),
            "mean_adjacent_similarity": round(
                float(np.mean([p.similarity for p in adjacent])) if adjacent else 0.0, 4
            ),
        }
        logger.info(
            "Paired %d EDUs → %d global pairs (non-adj=%d, mean_dist=%.1f, mean_sim=%.3f)",
            len(edus), len(selected), len(non_adj),
            stats["mean_pair_distance"], stats["mean_selected_similarity"],
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
    similarity_threshold: float = 0.25,
    max_distance: Optional[int] = None,
    backend: str = "auto",
) -> Tuple[List[EDUPair], List[EDUPair], Dict[str, Any]]:
    """Convenience — global Top-K by similarity (no adjacency forcing)."""
    pairer = EDUPairer(
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        max_distance=max_distance,
        backend=backend,
    )
    return pairer.pair(edus)
