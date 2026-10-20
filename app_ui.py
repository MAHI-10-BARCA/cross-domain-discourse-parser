#!/usr/bin/env python3
"""
Streamlit UI — Cross-Domain Discourse Parser demo.

Run (Windows CMD):
  pip install streamlit
  cd E:\\Final_year\\cross-domain-discourse-parser
  set PYTHONPATH=E:\\Final_year\\cross-domain-discourse-parser\\src
  streamlit run app_ui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from data_schemas import Document
from pipeline import DiscoursePipeline


st.set_page_config(page_title="Discourse Parser FYP", page_icon="🌳", layout="wide")

SAMPLE = (
    "Apple Inc. reported quarterly revenue that exceeded Wall Street expectations. "
    "The company said iPhone sales remained resilient despite a challenging market. "
    "However, services revenue grew more slowly than analysts had forecast. "
    "Chief Executive Tim Cook attributed the results to strong demand in emerging markets. "
    "Looking ahead, Apple warned that supply constraints could affect product availability. "
    "Investors reacted positively and shares rose in after-hours trading."
)


def _edu_text(edus, idx: int) -> str:
    for e in edus:
        if e.index == idx:
            return e.text
    return f"[EDU {idx}]"


def build_summary(results) -> str:
    edus = results.get("edus") or []
    if not edus:
        return "(no EDUs)"
    tstats = results.get("tree_stats") or {}
    merge_log = tstats.get("merge_log") or []
    important_idx = set()
    for m in merge_log:
        rel = m.get("relation") or "Other"
        if rel not in ("Other", "Joint"):
            span = m.get("span") or []
            if len(span) == 2:
                for i in range(int(span[0]), int(span[1]) + 1):
                    important_idx.add(i)
    ordered = sorted(edus, key=lambda e: e.index)
    if important_idx:
        chosen = [e for e in ordered if e.index in important_idx]
        if len(chosen) < 2:
            chosen = ordered[: min(3, len(ordered))]
    else:
        if len(ordered) <= 3:
            chosen = ordered
        else:
            chosen = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    bullets = [f"• {e.text.strip()}" for e in chosen]
    lead = (
        f"Discourse-guided extractive summary ({len(chosen)} of {len(ordered)} EDUs). "
        "Built from tree merges / central spans — not a generative model.\n\n"
    )
    return lead + "\n".join(bullets)


def stage_explain(name: str) -> str:
    return {
        "1": "**EDU segmentation** — Split the document into Elementary Discourse Units using spaCy (or regex fallback).",
        "1.5": "**Global pair selection** — MiniLM/TF-IDF similarity; global (non-adjacent) pairs allowed.",
        "2": "**QUD generation** — Candidate questions per pair (template or optional FLAN-T5).",
        "3": "**Salience** — Rank questions by centrality + answerability (cosine / optional NLI).",
        "4": "**Relation classification** — Heuristic cues or trained MiniLM classifier (GUM coarse labels).",
        "5": "**Answer-fit** — How well EDU_j answers the QUD about EDU_i.",
        "6": "**Bottom-up tree** — Greedy adjacent-span merges (n−1) using pair scores + relation labels.",
    }.get(name, "")


def main():
    st.title("🌳 Cross-Domain Discourse Parser")
    st.caption("FYP demo — Enhancing Cross-Domain Document Discourse Parsing Through Bottom-Up Learning")

    with st.sidebar:
        st.header("Pipeline settings")
        domain = st.selectbox("Domain tag", ["news", "finance", "tech", "medicine", "law", "unknown"], index=0)
        pair_backend = st.selectbox("Pair embeddings", ["minilm", "auto", "tfidf"], index=0)
        min_sim = st.slider("Min pair similarity", 0.0, 0.5, 0.10, 0.01)
        top_k = st.slider("Top-k partners per EDU", 1, 5, 3)
        relation_backend = st.selectbox("Relation backend", ["trained", "heuristic"], index=0)
        qud_backend = st.selectbox("QUD backend", ["template", "t5", "auto"], index=0)
        skip_qud = st.checkbox("Skip QUD / salience / answer-fit (faster)", value=False)
        use_nli = st.checkbox("Use NLI for salience", value=False)
        st.markdown("---")
        st.markdown("Stages: **1** EDU → **1.5** pairs → **2** QUD → **3** salience → **4** relations → **5** answer-fit → **6** tree")

    text = st.text_area("Input article / document", value=SAMPLE, height=220)
    if not st.button("Run pipeline", type="primary"):
        st.info("Paste text and click **Run pipeline**.")
        return
    if not text.strip():
        st.error("Please enter some text.")
        return

    with st.spinner("Running discourse pipeline (first MiniLM load can take ~30–60s)…"):
        try:
            pipe = DiscoursePipeline(
                seg_backend="auto",
                pair_backend=pair_backend,
                min_sim=min_sim,
                top_k=top_k,
                qud_backend=qud_backend,
                relation_backend=relation_backend,
                use_nli=use_nli,
                run_qud=not skip_qud,
                run_salience=not skip_qud,
                run_answer_fit=not skip_qud,
            )
            doc = Document(doc_id="ui_doc", text=text.strip(), domain=domain)
            results = pipe.run(doc, output_dir=str(ROOT / "outputs" / "ui_runs"))
        except Exception as e:
            st.exception(e)
            st.error("If trained relations failed, switch Relation backend to **heuristic**.")
            return

    edus = results.get("edus") or []
    pairs = results.get("pairs") or []
    relations = results.get("relations") or []
    questions = results.get("questions") or []
    salience = results.get("salience") or []
    answer_fit = results.get("answer_fit") or []
    tree = results.get("tree")
    tree_ascii = results.get("tree_ascii") or ""
    tstats = results.get("tree_stats") or {}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("EDUs", len(edus))
    m2.metric("Pairs", len(pairs))
    m3.metric("Relations", len(relations))
    m4.metric("Tree merges", tstats.get("num_merges", "—"))

    st.markdown("---")
    st.subheader("Stage-by-stage view")

    with st.expander("Stage 1 — EDU segmentation", expanded=True):
        st.markdown(stage_explain("1"))
        st.markdown("**Input:** full document text")
        st.code(text.strip()[:2000] + ("…" if len(text) > 2000 else ""), language=None)
        st.markdown(f"**Output:** {len(edus)} EDUs")
        for e in edus:
            st.markdown(f"- `[{e.index}]` {e.text}")

    with st.expander("Stage 1.5 — Related EDU pairs (global)", expanded=True):
        st.markdown(stage_explain("1.5"))
        pstats = results.get("pair_stats") or {}
        st.json({"backend": pstats.get("backend") or pair_backend, "min_sim": min_sim, "top_k": top_k, "num_selected": len(pairs)})
        for p in pairs:
            ti = _edu_text(edus, p.edu_i_index)
            tj = _edu_text(edus, p.edu_j_index)
            st.markdown(
                f"- **i={p.edu_i_index} ↔ j={p.edu_j_index}** (dist={p.distance}, sim={getattr(p, 'similarity', 0):.3f})\n"
                f"  - {ti}\n  - {tj}"
            )

    if not skip_qud:
        with st.expander("Stage 2 — QUD generation", expanded=False):
            st.markdown(stage_explain("2"))
            st.markdown(f"**Backend:** `{qud_backend}` — **{len(questions)}** valid questions")
            for q in questions[:15]:
                st.markdown(f"- ({getattr(q, 'category', '?')}) {getattr(q, 'question_text', '')}")
            if len(questions) > 15:
                st.caption(f"… and {len(questions) - 15} more")

        q_by_id = {q.question_id: q for q in questions}
        with st.expander("Stage 3 — Salience", expanded=False):
            st.markdown(stage_explain("3"))
            if salience:
                ranked = sorted(salience, key=lambda s: float(getattr(s, "final_score", 0) or 0), reverse=True)
                for s in ranked[:10]:
                    qobj = q_by_id.get(getattr(s, "question_id", ""), None)
                    qtxt = getattr(qobj, "question_text", "") if qobj else getattr(s, "question_id", "")
                    st.markdown(
                        f"- final={getattr(s, 'final_score', 0):.3f} cent={getattr(s, 'centrality', 0):.3f} "
                        f"ans={getattr(s, 'answerability', 0):.3f} — {qtxt}"
                    )
            else:
                st.write("No salience scores.")

        with st.expander("Stage 5 — Answer-fit", expanded=False):
            st.markdown(stage_explain("5"))
            if answer_fit:
                for a in answer_fit[:10]:
                    qobj = q_by_id.get(getattr(a, "question_id", ""), None)
                    qtxt = getattr(qobj, "question_text", "")[:120] if qobj else ""
                    st.markdown(
                        f"- score={float(getattr(a, 'score', 0) or 0):.3f} "
                        f"answer=`{getattr(a, 'answer_edu_id', '')}` — {qtxt}"
                    )
            else:
                st.write("No answer-fit scores.")

    with st.expander("Stage 4 — Discourse relations", expanded=True):
        st.markdown(stage_explain("4"))
        rstats = results.get("relation_stats") or {}
        st.json({"backend": rstats.get("backend", relation_backend), "label_counts": rstats.get("label_counts"), "mean_confidence": rstats.get("mean_confidence")})
        for r in relations:
            st.markdown(f"- **{getattr(r, 'label', '?')}** (conf={float(getattr(r, 'confidence', 0) or 0):.3f}) pair=`{getattr(r, 'pair_id', '')}`")

    with st.expander("Stage 6 — Bottom-up discourse tree", expanded=True):
        st.markdown(stage_explain("6"))
        merge_log = tstats.get("merge_log") or []
        if merge_log:
            st.table([{"merge": m.get("merge_id"), "span": str(m.get("span")), "relation": m.get("relation"), "score": m.get("score"), "forced": m.get("forced")} for m in merge_log])
        st.markdown("**Tree (ASCII)**")
        st.code(tree_ascii or "(empty tree)", language=None)
        if tree is not None and hasattr(tree, "to_dict"):
            with st.popover("Raw tree JSON"):
                st.json(tree.to_dict())

    st.markdown("---")
    st.subheader("Downstream verification — discourse-guided summary")
    st.markdown("Extractive summary biased toward informative merges (Contrast / Result / Explanation / Background).")
    st.success(build_summary(results))
    st.download_button("Download tree ASCII", data=tree_ascii or "", file_name="discourse_tree.txt", mime="text/plain")


if __name__ == "__main__":
    main()
