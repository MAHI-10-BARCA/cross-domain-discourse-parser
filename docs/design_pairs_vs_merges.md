# Design note (P6): Global pairs vs adjacent-span merges

## What Stage 1.5 does
- Selects **related EDU pairs globally** by MiniLM (or TF-IDF) similarity.
- Non-adjacent pairs are allowed and intentional.
- Purpose: candidate discourse links for relation labeling and scoring.

## What Stage 6 does
- Builds a **binary contiguous-span tree** with a greedy bottom-up algorithm.
- Only merges **adjacent spans** (standard RST-style projective trees).
- Performs exactly `n_edus - 1` merges for a connected tree.
- Algorithm name in stats: `greedy_bottom_up_adjacent`.

## How they connect
1. Stage 1.5 produces candidate pairs (may be non-adjacent).
2. Stage 4 assigns a relation label + confidence to each pair.
3. Stage 6 uses relation scores **when the candidate pair corresponds to an adjacent span boundary** during merge ranking.
4. Non-adjacent pairs do **not** create non-projective edges in the tree; they inform scoring / analysis only.

## Why this is intentional
- Full non-projective discourse graphs are a different task.
- Academic RST evaluation typically assumes contiguous constituents.
- Global pairing still supports **relation classification research** and cross-domain pair scoring without forcing adjacency-biased training data.

## Limitations (honest)
- Tree topology is **not** free-form graph construction from global pairs.
- Left-branching tall trees can appear when merge scores favor sequential attachment.
- Do not claim “global non-adjacent parsing of tree structure” without a different Stage 6 algorithm.

## Related code
- `src/stages/stage1_5_pairing/pairing.py`
- `src/stages/stage4_relation/relation_classifier.py`
- `src/stages/stage6_tree/tree_builder.py`
