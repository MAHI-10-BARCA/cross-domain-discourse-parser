"""
Phase 4 — Relation Classification

Labels: Elaboration | Explanation | Contrast | Result | Background | Joint | Other
Baseline: lexical cues + MiniLM/TF-IDF similarity + distance heuristics.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_schemas import Document, EDU, EDUPair, RelationPrediction, save_json

logger = logging.getLogger("discourse_parser.stage4")

RELATION_LABELS = [
    "Elaboration", "Explanation", "Contrast", "Result",
    "Background", "Joint", "Other",
]

CONTRAST_CUES = re.compile(
    r"\b(however|but|although|though|yet|nevertheless|nonetheless|"
    r"on the other hand|in contrast|whereas|while|despite|instead)\b", re.I)
RESULT_CUES = re.compile(
    r"\b(therefore|thus|hence|consequently|as a result|so that|"
    r"leading to|resulting in|accordingly|for this reason)\b", re.I)
EXPLANATION_CUES = re.compile(
    r"\b(because|since|due to|owing to|the reason|explained by|"
    r"on account of|as a result of)\b", re.I)
ELABORATION_CUES = re.compile(
    r"\b(for example|for instance|specifically|in particular|"
    r"namely|such as|including|especially)\b", re.I)
BACKGROUND_CUES = re.compile(
    r"\b(previously|earlier|historically|in the past|background|"
    r"traditionally|originally)\b", re.I)

_embed_model = None
_embed_name = None


def _load_minilm(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _embed_model, _embed_name
    if _embed_model is not None and _embed_name == model_name:
        return _embed_model
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model for relations: %s", model_name)
        _embed_model = SentenceTransformer(model_name)
        _embed_name = model_name
        return _embed_model
    except Exception as e:
        logger.warning("MiniLM unavailable for relations (%s). Using TF-IDF.", e)
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


def _cue_scores(text_i: str, text_j: str) -> Dict[str, float]:
    """Score lexical discourse cues on the EDU that typically carries them.

    Forward cues (Contrast, Result, Explanation, Elaboration) are matched on
    EDU_j only. Searching the concatenated pair text previously caused a single
    marker (e.g. "However" in one EDU) to label *every* pair containing that
    EDU as Contrast/Result/etc.

    Background cues remain on EDU_i (prior / contextual unit).
    """
    return {
        "Contrast": 1.0 if CONTRAST_CUES.search(text_j) else 0.0,
        "Result": 1.0 if RESULT_CUES.search(text_j) else 0.0,
        "Explanation": 1.0 if EXPLANATION_CUES.search(text_j) else 0.0,
        "Elaboration": 1.0 if ELABORATION_CUES.search(text_j) else 0.0,
        "Background": 1.0 if BACKGROUND_CUES.search(text_i) else 0.0,
    }


def predict_relation_heuristic(
    text_i: str, text_j: str, emb_i: np.ndarray, emb_j: np.ndarray,
    distance: int, pair_similarity: Optional[float] = None,
) -> Tuple[str, float, Dict[str, Any]]:
    cues = _cue_scores(text_i, text_j)
    sim = pair_similarity if pair_similarity is not None else _cosine(emb_i, emb_j)
    evidence: Dict[str, Any] = {
        "cues": {k: v for k, v in cues.items() if v > 0},
        "cosine_similarity": round(float(sim), 4),
        "distance": distance,
    }
    active = [(lab, sc) for lab, sc in cues.items() if sc > 0]
    if active:
        priority = ["Contrast", "Result", "Explanation", "Elaboration", "Background"]
        active_sorted = sorted(
            active, key=lambda x: (priority.index(x[0]) if x[0] in priority else 99, -x[1])
        )
        label = active_sorted[0][0]
        conf = 0.75 if len(active) == 1 else 0.60
        evidence["decision"] = "lexical_cue"
        return label, conf, evidence
    if sim >= 0.45 and distance <= 3:
        evidence["decision"] = "high_sim_near"
        return "Elaboration", 0.55, evidence
    if sim >= 0.35:
        evidence["decision"] = "moderate_sim"
        return "Joint", 0.45, evidence
    if distance >= 4 and sim < 0.25:
        evidence["decision"] = "far_low_sim"
        return "Background", 0.40, evidence
    evidence["decision"] = "default"
    return "Other", 0.30, evidence


class RelationClassifier:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        backend: str = "heuristic",
    ):
        self.embedding_model = embedding_model
        self.backend = backend

    def classify(
        self, pairs: List[EDUPair], edus: List[EDU],
    ) -> Tuple[List[RelationPrediction], Dict[str, Any]]:
        if not pairs:
            return [], {"num_pairs": 0, "label_counts": {}}

        edu_map = {e.edu_id: e for e in edus}
        texts_i, texts_j, resolved = [], [], []
        for p in pairs:
            ei = edu_map.get(p.edu_i_id)
            ej = edu_map.get(p.edu_j_id)
            if ei is None and p.edu_i_index < len(edus):
                ei = edus[p.edu_i_index]
            if ej is None and p.edu_j_index < len(edus):
                ej = edus[p.edu_j_index]
            resolved.append((p, ei, ej))
            texts_i.append(ei.text if ei else "")
            texts_j.append(ej.text if ej else "")

        embs = _embed_texts(texts_i + texts_j, self.embedding_model)
        n = len(pairs)
        emb_i, emb_j = embs[:n], embs[n:]

        results: List[RelationPrediction] = []
        label_counts: Dict[str, int] = {lab: 0 for lab in RELATION_LABELS}

        for idx, (pair, ei, ej) in enumerate(resolved):
            if ei is None or ej is None:
                rel = RelationPrediction(
                    prediction_id=f"{pair.pair_id}_rel", pair_id=pair.pair_id,
                    span_id=None, label="Other", confidence=0.0,
                    features={"error": "missing_edu"}, metadata={"doc_id": pair.doc_id},
                )
            else:
                label, conf, evidence = predict_relation_heuristic(
                    ei.text, ej.text, emb_i[idx], emb_j[idx],
                    distance=pair.distance, pair_similarity=pair.similarity,
                )
                rel = RelationPrediction(
                    prediction_id=f"{pair.pair_id}_rel", pair_id=pair.pair_id,
                    span_id=None, label=label, confidence=round(conf, 4),
                    features=evidence,
                    metadata={
                        "backend": self.backend, "doc_id": pair.doc_id,
                        "edu_i_id": ei.edu_id, "edu_j_id": ej.edu_id,
                        "distance": pair.distance,
                    },
                )
            results.append(rel)
            label_counts[rel.label] = label_counts.get(rel.label, 0) + 1

        confs = [r.confidence for r in results]
        stats = {
            "num_pairs": len(pairs),
            "label_counts": label_counts,
            "mean_confidence": round(float(np.mean(confs)), 4) if confs else 0.0,
            "backend": self.backend,
        }
        logger.info("Relation classification: %d pairs → %s (mean_conf=%.2f)",
                    len(pairs), label_counts, stats["mean_confidence"])
        return results, stats

    def classify_and_save(
        self, pairs: List[EDUPair], edus: List[EDU],
        output_path: str | Path, document: Optional[Document] = None,
    ) -> Tuple[List[RelationPrediction], Dict[str, Any]]:
        results, stats = self.classify(pairs, edus)
        payload = {
            "document": document.to_dict() if document else None,
            "relations": [r.to_dict() for r in results],
            "stats": stats,
        }
        save_json(payload, output_path)
        logger.info("Saved relation labels to %s", output_path)
        return results, stats


def classify_relations(
    pairs: List[EDUPair], edus: List[EDU],
) -> Tuple[List[RelationPrediction], Dict[str, Any]]:
    return RelationClassifier().classify(pairs, edus)
