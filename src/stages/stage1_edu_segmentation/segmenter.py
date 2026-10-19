"""
Phase 1 — EDU Segmentation

Sentence-level Elementary Discourse Unit (EDU) segmentation.
Primary backend: spaCy en_core_web_sm
Fallback: pure-Python regex sentence splitter (when spaCy is unavailable)

Output conforms to the shared EDU schema in src/data_schemas.py.
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional, Tuple
from pathlib import Path
import sys

# Allow imports from project src
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data_schemas import Document, EDU, save_json

logger = logging.getLogger("discourse_parser.stage1")


# ---------------------------------------------------------------------------
# Text cleaning (minimal & safe)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Minimal safe cleaning:
    - Normalize newlines to \\n
    - Collapse runs of whitespace inside lines (keep paragraph breaks)
    - Strip leading/trailing whitespace
    Does NOT remove punctuation or alter sentence content.
    """
    if not text:
        return ""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple blank lines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace runs (preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Pure-Python sentence splitter (fallback)
# ---------------------------------------------------------------------------

# Common abbreviations that should not end a sentence
_ABBREVS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g",
    "i.e", "u.s", "u.k", "a.m", "p.m", "no", "vol", "fig", "eq", "approx",
    "dept", "univ", "inc", "ltd", "co", "corp", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "al", "ed",
}

_SENT_END_RE = re.compile(
    r"""
    (?<=[.!?])          # ends with . ! or ?
    (?:["')\]]*)        # optional closing quotes/brackets
    \s+                 # whitespace
    (?=[A-Z0-9"'(])     # next sentence starts with capital / digit / quote
    """,
    re.VERBOSE,
)


def _is_abbrev(token: str) -> bool:
    """Return True only for known abbreviations (not short tokens like Q3)."""
    t = token.lower().rstrip(".")
    return t in _ABBREVS


def split_sentences_fallback(text: str) -> List[Tuple[str, int, int]]:
    """
    Split text into sentences using regex + abbreviation heuristics.
    Returns list of (sentence_text, start_char, end_char).
    """
    if not text or not text.strip():
        return []

    # First pass: candidate splits
    candidates: List[Tuple[int, int]] = []  # (start, end) of potential sentence
    last = 0
    for m in _SENT_END_RE.finditer(text):
        end = m.start() + 1  # include the punctuation
        # Check if the period belongs to a known abbreviation
        before = text[last:end].rstrip()
        words = before.split()
        if words:
            last_word = words[-1]
            # Only skip when it is a known abbreviation list entry
            if _is_abbrev(last_word):
                continue
        candidates.append((last, end))
        last = m.end()
    # Remainder
    if last < len(text):
        candidates.append((last, len(text)))

    results = []
    for start, end in candidates:
        sent = text[start:end].strip()
        if not sent:
            continue
        # Recompute exact offsets of the stripped sentence inside original
        real_start = start
        while real_start < end and text[real_start].isspace():
            real_start += 1
        real_end = real_start + len(sent)
        results.append((sent, real_start, real_end))
    return results


# ---------------------------------------------------------------------------
# spaCy backend
# ---------------------------------------------------------------------------

_nlp = None


def _load_spacy(model_name: str = "en_core_web_sm"):
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        try:
            _nlp = spacy.load(model_name)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found. Attempting download...", model_name
            )
            from spacy.cli import download
            download(model_name)
            _nlp = spacy.load(model_name)
        logger.info("Loaded spaCy model: %s", model_name)
        return _nlp
    except Exception as e:
        logger.warning("spaCy unavailable (%s). Using pure-Python fallback.", e)
        return None


def split_sentences_spacy(text: str, model_name: str = "en_core_web_sm") -> List[Tuple[str, int, int]]:
    nlp = _load_spacy(model_name)
    if nlp is None:
        return split_sentences_fallback(text)

    doc = nlp(text)
    results = []
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
        # Character offsets relative to the original text passed to nlp
        start = sent.start_char
        end = sent.end_char
        # Adjust for leading whitespace that spaCy may include
        while start < end and text[start].isspace():
            start += 1
        results.append((sent_text, start, end))
    return results


# ---------------------------------------------------------------------------
# Main segmenter
# ---------------------------------------------------------------------------

class EDUSegmenter:
    """
    Stage 1: Convert a Document into an ordered list of EDUs.
    """

    def __init__(
        self,
        backend: str = "auto",          # "spacy" | "fallback" | "auto"
        spacy_model: str = "en_core_web_sm",
        preserve_offsets: bool = True,
    ):
        self.backend = backend
        self.spacy_model = spacy_model
        self.preserve_offsets = preserve_offsets
        self._resolved_backend: Optional[str] = None

    def _resolve_backend(self) -> str:
        if self._resolved_backend:
            return self._resolved_backend
        if self.backend == "fallback":
            self._resolved_backend = "fallback"
        elif self.backend == "spacy":
            nlp = _load_spacy(self.spacy_model)
            self._resolved_backend = "spacy" if nlp is not None else "fallback"
        else:  # auto
            nlp = _load_spacy(self.spacy_model)
            self._resolved_backend = "spacy" if nlp is not None else "fallback"
        logger.info("EDU segmentation backend: %s", self._resolved_backend)
        return self._resolved_backend

    def segment(self, document: Document) -> List[EDU]:
        """
        Segment a Document into sentence-level EDUs.

        Returns
        -------
        List[EDU]  ordered by appearance in the document
        """
        text = clean_text(document.text)
        if not text:
            logger.warning("Document %s has empty text after cleaning.", document.doc_id)
            return []

        backend = self._resolve_backend()
        if backend == "spacy":
            spans = split_sentences_spacy(text, self.spacy_model)
        else:
            spans = split_sentences_fallback(text)

        edus: List[EDU] = []
        for idx, (sent_text, start, end) in enumerate(spans):
            edu_id = f"{document.doc_id}_edu_{idx:03d}"
            edu = EDU(
                edu_id=edu_id,
                doc_id=document.doc_id,
                text=sent_text,
                index=idx,
                start_char=start if self.preserve_offsets else None,
                end_char=end if self.preserve_offsets else None,
                metadata={
                    "backend": backend,
                    "original_length": len(sent_text),
                },
            )
            edus.append(edu)

        logger.info(
            "Document %s → %d EDUs (backend=%s)",
            document.doc_id, len(edus), backend,
        )
        return edus

    def segment_and_save(
        self,
        document: Document,
        output_path: str | Path,
    ) -> List[EDU]:
        """Segment and write JSON to disk."""
        edus = self.segment(document)
        payload = {
            "document": document.to_dict(),
            "edus": [e.to_dict() for e in edus],
            "num_edus": len(edus),
            "backend": self._resolved_backend,
        }
        save_json(payload, output_path)
        logger.info("Saved segmentation to %s", output_path)
        return edus


def segment_document(
    document: Document,
    backend: str = "auto",
    spacy_model: str = "en_core_web_sm",
) -> List[EDU]:
    """Convenience function."""
    segmenter = EDUSegmenter(backend=backend, spacy_model=spacy_model)
    return segmenter.segment(document)
