# Priority status — FYP baseline complete

Project: Enhancing Cross-Domain Document Discourse Parsing Through Bottom-Up Learning

## Completed (baseline)

| ID | Item | Status |
|----|------|--------|
| **P0** | Cue-scope fix | **Done** |
| **P1** | GUM trained MiniLM LogReg + dual backend | **Done** |
| **P1b** | Heuristic vs trained on GUM gold test (5853 pairs) | **Done** |
| **P1c** | Pipeline `--relation-backend trained` (news sample) | **Done** |
| **P6** | Design note: global pairs vs adjacent merges | **Done** |
| **P7** | Experiment log + documentation | **Done** |

## Optional extras (not required for baseline claim)

| ID | Item | Status |
|----|------|--------|
| **P2** | FLAN-T5 QUD default | Code ready; default stays **template** |
| **P3** | NLI primary answerability | Code ready; **cosine** is primary |
| **P4** | Full tree UAS on gold trees | **Not done** (pair F1 only) |
| **P5** | Genre hold-out retrain metrics | Flag only; optional retrain |

## Official model accuracy (use in report)

**Task:** coarse 7-class discourse relation classification on GUM V12 gold pairs  
**Split:** document-level test (seed=42), **n = 5853 pairs**

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|-------|----------|----------|-------------|
| Heuristic baseline | 0.3147 (31.5%) | 0.2209 | 0.2400 |
| **Trained (MiniLM + LogReg)** | **0.4329 (43.3%)** | **0.3520** | **0.4567** |

Trained outperforms heuristic on all three metrics.

Source: `outputs/evaluation/p1b_backend_compare.json`

## Limitations (report honestly)
- These are **pair-level** metrics, not full discourse-tree attachment scores.
- Do not claim cross-domain generalization from three sample documents.
- Stage 6 trees use adjacent-span greedy merges (see `design_pairs_vs_merges.md`).
- News sample with trained backend was Joint-heavy (expected class bias), not a gold tree score.
