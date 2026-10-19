"""
Phase 2 — QUD-based multi-question generation

For each related EDU pair, generate candidate Questions Under Discussion (QUDs).
Primary model: google/flan-t5-small
Fallback: template-based generation (no GPU / no transformers required)
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_schemas import Document, EDU, EDUPair, QUDQuestion, save_json

logger = logging.getLogger("discourse_parser.stage2")

QUESTION_CATEGORIES = {
    "what": "factual content / event",
    "why": "cause / reason / motivation",
    "how": "manner / method / process",
    "when": "time / temporal setting",
    "where": "location / spatial setting",
    "who": "agent / participant",
    "consequence": "result / effect / outcome",
    "elaboration": "detail / expansion of a prior point",
}

TEMPLATES = {
    "what": [
        "What does the following statement tell us about: {edu_i}?",
        "What is being described in relation to: {edu_i}?",
    ],
    "why": [
        "Why might the following be true given: {edu_i}?",
        "What is the reason connected to: {edu_i}?",
    ],
    "how": [
        "How does the following relate to: {edu_i}?",
        "In what way does this elaborate on: {edu_i}?",
    ],
    "consequence": [
        "What is a consequence of: {edu_i}?",
        "What follows from: {edu_i}?",
    ],
    "elaboration": [
        "What further detail is given about: {edu_i}?",
        "What additional information expands on: {edu_i}?",
    ],
}

T5_PROMPT = (
    "Generate a short implicit question that the second sentence answers "
    "in relation to the first sentence.\n"
    "First: {edu_i}\n"
    "Second: {edu_j}\n"
    "Question:"
)

_QUESTION_MARK_RE = re.compile(r"\?")
_WHITESPACE_RE = re.compile(r"\s+")
_MIN_WORDS = 3
_MAX_WORDS = 40


def validate_question(
    text: str, edu_i: str, edu_j: str, seen: set,
) -> Tuple[bool, Optional[str]]:
    if not text or not text.strip():
        return False, "empty"
    cleaned = _WHITESPACE_RE.sub(" ", text.strip())
    if len(cleaned) < 5:
        return False, "too_short"
    words = cleaned.split()
    if len(words) < _MIN_WORDS:
        return False, "too_few_words"
    if len(words) > _MAX_WORDS:
        return False, "too_many_words"
    wh = ("what", "why", "how", "when", "where", "who", "which", "whose", "whom")
    starts_wh = words[0].lower().rstrip(".,;:") in wh
    has_qmark = bool(_QUESTION_MARK_RE.search(cleaned))
    if not (starts_wh or has_qmark):
        return False, "not_question_form"
    norm = cleaned.lower().rstrip("?").strip()
    if norm in seen:
        return False, "duplicate"
    seen.add(norm)
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "on", "for", "and", "or", "but", "with", "as", "by",
        "at", "from", "that", "this", "it", "its", "what", "why", "how", "when",
        "where", "who", "which", "does", "do", "did", "about", "following",
    }
    edu_tokens = set(
        w.lower().strip(".,;:!?\"'") for w in (edu_i + " " + edu_j).split() if len(w) > 2
    ) - stop
    q_tokens = set(w.lower().strip(".,;:!?\"'") for w in words if len(w) > 2) - stop
    if edu_tokens and not (q_tokens & edu_tokens):
        content_q = q_tokens - set(wh)
        if content_q and not (content_q & edu_tokens):
            return False, "ungrounded"
    return True, None


_t5_model = None
_t5_tokenizer = None
_t5_name = None


def _load_flan_t5(model_name: str = "google/flan-t5-small"):
    global _t5_model, _t5_tokenizer, _t5_name
    if _t5_model is not None and _t5_name == model_name:
        return _t5_model, _t5_tokenizer
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        logger.info("Loading QUD generator: %s", model_name)
        _t5_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _t5_model.eval()
        _t5_name = model_name
        return _t5_model, _t5_tokenizer
    except Exception as e:
        logger.warning("FLAN-T5 unavailable (%s). Using template fallback.", e)
        return None, None


def generate_with_t5(
    edu_i: str, edu_j: str,
    model_name: str = "google/flan-t5-small",
    max_new_tokens: int = 48,
    num_return_sequences: int = 3,
    num_beams: int = 4,
) -> List[str]:
    model, tokenizer = _load_flan_t5(model_name)
    if model is None:
        return []
    import torch
    prompt = T5_PROMPT.format(edu_i=edu_i, edu_j=edu_j)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=max(num_beams, num_return_sequences),
            num_return_sequences=num_return_sequences,
            early_stopping=True,
            do_sample=False,
        )
    return [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]


def generate_with_templates(edu_i: str, edu_j: str) -> List[Tuple[str, str]]:
    results = []
    for cat, tpls in TEMPLATES.items():
        for tpl in tpls:
            results.append((tpl.format(edu_i=edu_i[:120]), cat))
    return results


def infer_category(question: str) -> str:
    q = question.lower().strip()
    for cat in ("what", "why", "how", "when", "where", "who"):
        if q.startswith(cat):
            return cat
    if any(w in q for w in ("result", "consequence", "follow", "lead to", "cause")):
        return "consequence"
    if any(w in q for w in ("detail", "elaborate", "further", "more about", "expand")):
        return "elaboration"
    return "what"


class QUDGenerator:
    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        backend: str = "auto",
        max_new_tokens: int = 48,
        num_return_sequences: int = 3,
        num_beams: int = 4,
        categories: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.num_return_sequences = num_return_sequences
        self.num_beams = num_beams
        self.categories = categories or list(TEMPLATES.keys())
        self._resolved_backend: Optional[str] = None

    def _resolve_backend(self) -> str:
        if self._resolved_backend:
            return self._resolved_backend
        if self.backend == "template":
            self._resolved_backend = "template"
        else:
            model, _ = _load_flan_t5(self.model_name)
            self._resolved_backend = "t5" if model is not None else "template"
        logger.info("QUD generation backend: %s", self._resolved_backend)
        return self._resolved_backend

    def generate_for_pair(
        self, pair: EDUPair, edu_i: EDU, edu_j: EDU, qid_counter: List[int],
    ) -> List[QUDQuestion]:
        backend = self._resolve_backend()
        seen: set = set()
        questions: List[QUDQuestion] = []
        raw_candidates: List[Tuple[str, Optional[str], Dict[str, Any]]] = []

        if backend == "t5":
            texts = generate_with_t5(
                edu_i.text, edu_j.text,
                model_name=self.model_name,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.num_return_sequences,
                num_beams=self.num_beams,
            )
            for t in texts:
                raw_candidates.append((t, None, {
                    "model": self.model_name,
                    "max_new_tokens": self.max_new_tokens,
                    "num_beams": self.num_beams,
                }))
            for t, cat in generate_with_templates(edu_i.text, edu_j.text):
                if cat in self.categories:
                    raw_candidates.append((t, cat, {"model": "template"}))
        else:
            for t, cat in generate_with_templates(edu_i.text, edu_j.text):
                if cat in self.categories:
                    raw_candidates.append((t, cat, {"model": "template"}))

        for text, cat, gen_cfg in raw_candidates:
            is_valid, reason = validate_question(text, edu_i.text, edu_j.text, seen)
            qid_counter[0] += 1
            qid = f"{pair.doc_id}_q_{qid_counter[0]:04d}"
            category = cat or infer_category(text)
            q = QUDQuestion(
                question_id=qid,
                pair_id=pair.pair_id,
                doc_id=pair.doc_id,
                question_text=_WHITESPACE_RE.sub(" ", text.strip()),
                category=category,
                specificity="coarse",
                generation_config=gen_cfg,
                is_valid=is_valid,
                rejection_reason=reason,
                metadata={
                    "backend": backend,
                    "edu_i_id": edu_i.edu_id,
                    "edu_j_id": edu_j.edu_id,
                },
            )
            questions.append(q)
        return questions

    def generate(
        self, pairs: List[EDUPair], edus: List[EDU],
    ) -> Tuple[List[QUDQuestion], Dict[str, Any]]:
        edu_map = {e.edu_id: e for e in edus}
        qid_counter = [0]
        all_qs: List[QUDQuestion] = []
        for pair in pairs:
            edu_i = edu_map.get(pair.edu_i_id)
            edu_j = edu_map.get(pair.edu_j_id)
            if edu_i is None or edu_j is None:
                logger.warning("Missing EDU for pair %s — skipped", pair.pair_id)
                continue
            all_qs.extend(self.generate_for_pair(pair, edu_i, edu_j, qid_counter))

        valid = [q for q in all_qs if q.is_valid]
        rejected = [q for q in all_qs if not q.is_valid]
        reject_reasons: Dict[str, int] = {}
        for q in rejected:
            r = q.rejection_reason or "unknown"
            reject_reasons[r] = reject_reasons.get(r, 0) + 1
        cats: Dict[str, int] = {}
        for q in valid:
            c = q.category or "unknown"
            cats[c] = cats.get(c, 0) + 1

        stats = {
            "num_pairs": len(pairs),
            "num_candidates_total": len(all_qs),
            "num_valid": len(valid),
            "num_rejected": len(rejected),
            "rejection_reasons": reject_reasons,
            "category_counts": cats,
            "backend": self._resolved_backend,
        }
        logger.info(
            "QUD generation: %d pairs → %d valid / %d rejected (backend=%s)",
            len(pairs), len(valid), len(rejected), self._resolved_backend,
        )
        return all_qs, stats

    def generate_and_save(
        self, pairs: List[EDUPair], edus: List[EDU],
        output_path: str | Path, document: Optional[Document] = None,
    ) -> Tuple[List[QUDQuestion], Dict[str, Any]]:
        all_qs, stats = self.generate(pairs, edus)
        valid = [q for q in all_qs if q.is_valid]
        payload = {
            "document": document.to_dict() if document else None,
            "num_pairs": len(pairs),
            "questions_all": [q.to_dict() for q in all_qs],
            "questions_valid": [q.to_dict() for q in valid],
            "stats": stats,
        }
        save_json(payload, output_path)
        logger.info("Saved QUD results to %s", output_path)
        return all_qs, stats


def generate_quds(
    pairs: List[EDUPair], edus: List[EDU], backend: str = "auto",
) -> Tuple[List[QUDQuestion], Dict[str, Any]]:
    return QUDGenerator(backend=backend).generate(pairs, edus)
