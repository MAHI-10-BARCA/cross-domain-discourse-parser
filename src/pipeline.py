"""
End-to-end discourse parsing pipeline.

Wires Stages 1 → 1.5 → 2 → 3 → 4 → 5 → 6 into a single callable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from data_schemas import Document, save_json
from stages.stage1_edu_segmentation import EDUSegmenter
from stages.stage1_5_pairing import EDUPairer
from stages.stage2_qud_generation import QUDGenerator
from stages.stage3_salience import SalienceScorer
from stages.stage4_relation import RelationClassifier
from stages.stage5_answer_fit import AnswerFitScorer
from stages.stage6_tree import BottomUpTreeBuilder
from stages.stage6_tree.tree_builder import tree_to_ascii

logger = logging.getLogger("discourse_parser.pipeline")


class DiscoursePipeline:
    """Full bottom-up discourse parser. Defaults are safe for CPU."""

    def __init__(
        self,
        seg_backend: str = "auto",
        pair_backend: str = "auto",
        min_sim: float = 0.15,
        top_k: int = 3,
        qud_backend: str = "template",
        relation_backend: str = "heuristic",
        use_nli: bool = False,
        run_qud: bool = True,
        run_salience: bool = True,
        run_answer_fit: bool = True,
    ):
        self.seg_backend = seg_backend
        self.pair_backend = pair_backend
        self.min_sim = min_sim
        self.top_k = top_k
        self.qud_backend = qud_backend
        self.relation_backend = relation_backend
        self.use_nli = use_nli
        self.run_qud = run_qud
        self.run_salience = run_salience
        self.run_answer_fit = run_answer_fit

    def run(
        self,
        document: Document,
        output_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        out = Path(output_dir) if output_dir else None
        if out:
            out.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Any] = {"doc_id": document.doc_id}

        edus = EDUSegmenter(backend=self.seg_backend).segment(document)
        results["edus"] = edus
        results["num_edus"] = len(edus)
        logger.info("Pipeline [%s] Stage1: %d EDUs", document.doc_id, len(edus))

        selected, all_pairs, pstats = EDUPairer(
            backend=self.pair_backend,
            top_k=self.top_k,
            similarity_threshold=self.min_sim,
        ).pair(edus)
        results["pairs"] = selected
        results["pair_stats"] = pstats
        logger.info("Pipeline [%s] Stage1.5: %d pairs", document.doc_id, len(selected))

        valid_qs = []
        if self.run_qud:
            all_qs, qstats = QUDGenerator(backend=self.qud_backend).generate(selected, edus)
            valid_qs = [q for q in all_qs if q.is_valid]
            results["questions"] = valid_qs
            results["qud_stats"] = qstats
            logger.info("Pipeline [%s] Stage2: %d valid QUDs", document.doc_id, len(valid_qs))

            if self.run_salience and valid_qs:
                sal_scores, sal_stats = SalienceScorer(use_nli=self.use_nli).score(
                    valid_qs, edus, pairs=selected
                )
                results["salience"] = sal_scores
                results["salience_stats"] = sal_stats

            if self.run_answer_fit and valid_qs:
                af_scores, af_stats = AnswerFitScorer().score(valid_qs, edus, pairs=selected)
                results["answer_fit"] = af_scores
                results["answer_fit_stats"] = af_stats

        relations, rstats = RelationClassifier(backend=self.relation_backend).classify(
            selected, edus
        )
        results["relations"] = relations
        results["relation_stats"] = rstats
        logger.info(
            "Pipeline [%s] Stage4 (%s): %s",
            document.doc_id,
            self.relation_backend,
            rstats.get("label_counts"),
        )

        tree, tstats = BottomUpTreeBuilder().build(
            edus, selected, relations=relations, document=document
        )
        results["tree"] = tree
        results["tree_stats"] = tstats
        results["tree_ascii"] = tree_to_ascii(tree.root, edus)
        logger.info(
            "Pipeline [%s] Stage6: %d merges, relations=%s",
            document.doc_id, tstats["num_merges"], tstats.get("relation_counts"),
        )

        if out:
            save_json(
                {
                    "doc_id": document.doc_id,
                    "num_edus": len(edus),
                    "num_pairs": len(selected),
                    "relation_stats": rstats,
                    "tree_stats": tstats,
                    "tree": tree.to_dict(),
                    "tree_ascii": results["tree_ascii"],
                    "relation_backend": self.relation_backend,
                },
                out / f"{document.doc_id}_pipeline.json",
            )
            save_json(
                {"tree": tree.to_dict(), "stats": tstats},
                out / f"{document.doc_id}_tree.json",
            )

        return results


def run_pipeline_on_text(
    text: str,
    doc_id: str = "doc",
    domain: str = "unknown",
    **kwargs,
) -> Dict[str, Any]:
    doc = Document(doc_id=doc_id, text=text, domain=domain)
    return DiscoursePipeline(**kwargs).run(doc)
