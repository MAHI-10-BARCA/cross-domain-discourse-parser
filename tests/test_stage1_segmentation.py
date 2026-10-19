"""
Phase 1 unit tests — EDU Segmentation
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_schemas import Document, EDU
from stages.stage1_edu_segmentation import EDUSegmenter, segment_document
from stages.stage1_edu_segmentation.segmenter import clean_text, split_sentences_fallback


# ---------------------------------------------------------------------------
# Fixtures / sample texts
# ---------------------------------------------------------------------------

SHORT_TEXT = "The cat sat on the mat. It was happy."

MEDIUM_TEXT = (
    "Apple Inc. announced a new product yesterday. "
    "The device will ship in Q3. "
    "Analysts expect strong demand in the U.S. market. "
    "However, supply constraints may limit availability."
)

LONG_TEXT = """
Natural language processing has made remarkable progress in recent years.
Transformer models such as BERT and GPT have changed the field.
Nevertheless, discourse-level understanding remains challenging.
Researchers continue to explore hierarchical representations.
For example, Rhetorical Structure Theory provides a framework for analyzing text coherence.
This project investigates bottom-up parsing strategies across multiple domains.
"""

EMPTY_TEXT = ""
WHITESPACE_ONLY = "   \n\n  \t  "
MULTI_PARA = "First paragraph sentence one. Sentence two.\n\nSecond paragraph starts here. And continues."


# ---------------------------------------------------------------------------
# Helper assertions
# ---------------------------------------------------------------------------

def assert_valid_edus(edus, doc_id: str):
    assert isinstance(edus, list)
    for i, edu in enumerate(edus):
        assert isinstance(edu, EDU)
        assert edu.doc_id == doc_id
        assert edu.edu_id == f"{doc_id}_edu_{i:03d}"
        assert edu.index == i
        assert edu.text.strip()  # non-empty after strip
        assert isinstance(edu.text, str)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_clean_text_basic():
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("a\r\nb") == "a\nb"
    assert clean_text("") == ""


def test_fallback_short():
    spans = split_sentences_fallback(SHORT_TEXT)
    assert len(spans) == 2
    assert spans[0][0] == "The cat sat on the mat."
    assert spans[1][0] == "It was happy."


def test_fallback_abbreviations():
    # "Inc." and "U.S." should not create false splits
    spans = split_sentences_fallback(MEDIUM_TEXT)
    texts = [s[0] for s in spans]
    assert any("Apple Inc." in t for t in texts)
    # We expect roughly 4 sentences
    assert 3 <= len(spans) <= 5


def test_segmenter_short_fallback():
    doc = Document(doc_id="short1", text=SHORT_TEXT, domain="test")
    seg = EDUSegmenter(backend="fallback")
    edus = seg.segment(doc)
    assert_valid_edus(edus, "short1")
    assert len(edus) == 2
    assert edus[0].text == "The cat sat on the mat."
    assert edus[1].text == "It was happy."
    # offsets present
    assert edus[0].start_char is not None
    assert edus[0].end_char is not None


def test_segmenter_medium_fallback():
    doc = Document(doc_id="med1", text=MEDIUM_TEXT, domain="news")
    seg = EDUSegmenter(backend="fallback")
    edus = seg.segment(doc)
    assert_valid_edus(edus, "med1")
    assert len(edus) >= 3
    # original wording preserved
    joined = " ".join(e.text for e in edus)
    assert "Apple Inc." in joined
    assert "U.S." in joined or "U.S" in joined


def test_segmenter_long_fallback():
    doc = Document(doc_id="long1", text=LONG_TEXT, domain="science")
    seg = EDUSegmenter(backend="fallback")
    edus = seg.segment(doc)
    assert_valid_edus(edus, "long1")
    assert len(edus) >= 5
    # order preserved
    for i, edu in enumerate(edus):
        assert edu.index == i


def test_empty_document():
    doc = Document(doc_id="empty1", text=EMPTY_TEXT)
    seg = EDUSegmenter(backend="fallback")
    edus = seg.segment(doc)
    assert edus == []


def test_whitespace_only():
    doc = Document(doc_id="ws1", text=WHITESPACE_ONLY)
    seg = EDUSegmenter(backend="fallback")
    edus = seg.segment(doc)
    assert edus == []


def test_multi_paragraph():
    doc = Document(doc_id="para1", text=MULTI_PARA)
    seg = EDUSegmenter(backend="fallback")
    edus = seg.segment(doc)
    assert_valid_edus(edus, "para1")
    assert len(edus) >= 3


def test_reproducibility():
    doc = Document(doc_id="rep1", text=MEDIUM_TEXT)
    seg = EDUSegmenter(backend="fallback")
    edus1 = seg.segment(doc)
    edus2 = seg.segment(doc)
    assert len(edus1) == len(edus2)
    for a, b in zip(edus1, edus2):
        assert a.edu_id == b.edu_id
        assert a.text == b.text
        assert a.start_char == b.start_char
        assert a.end_char == b.end_char


def test_save_json(tmp_path):
    doc = Document(doc_id="save1", text=SHORT_TEXT, domain="test")
    seg = EDUSegmenter(backend="fallback")
    out = tmp_path / "seg.json"
    edus = seg.segment_and_save(doc, out)
    assert out.exists()
    import json
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["num_edus"] == 2
    assert data["document"]["doc_id"] == "save1"
    assert len(data["edus"]) == 2


def test_convenience_function():
    doc = Document(doc_id="conv1", text=SHORT_TEXT)
    edus = segment_document(doc, backend="fallback")
    assert len(edus) == 2
    assert edus[0].edu_id == "conv1_edu_000"


def test_stable_ids_across_calls():
    """Same document → same EDU IDs every time."""
    doc = Document(doc_id="stable", text="One. Two. Three.")
    seg = EDUSegmenter(backend="fallback")
    a = [e.edu_id for e in seg.segment(doc)]
    b = [e.edu_id for e in seg.segment(doc)]
    assert a == b == ["stable_edu_000", "stable_edu_001", "stable_edu_002"]
