# Remaining priorities status (post-P0 / P1)

Project: Enhancing Cross-Domain Document Discourse Parsing Through Bottom-Up Learning

## Completed earlier
| ID | Item | Status |
|----|------|--------|
| P0 | Cue-scope fix (forward cues on EDU_j) | Done |
| P1 | GUM gold + trained MiniLM LogReg + dual backend | Done (model on disk) |

## This delivery

| ID | Item | Deliverable | Status |
|----|------|-------------|--------|
| **P1b** | Heuristic vs trained on GUM gold test | `scripts/compare_relation_backends.py` | Code ready; run on Windows with gold + model |
| **P1c** | Pipeline `--relation-backend` | `pipeline.py`, `run_pipeline.py`, `run_stage4.py` | Done |
| **P2** | FLAN-T5 QUD backend | Already in `qud_generator.py` (`backend=t5|auto|template`) | Implemented; default remains `template` for CPU reliability |
| **P3** | NLI answerability path | Already in `salience.py` (`use_nli=True`); CLI `--use-nli` | Optional; cosine remains primary |
| **P4** | Gold attachment-style metrics | `pair_attachment_metrics` in evaluation + pair F1 in P1b | Pair-level only; full tree UAS not claimed |
| **P5** | Genre hold-out training | `train_relation_classifier.py --holdout-genres` | Code support (retrain locally) |
| **P6** | Pairs vs merges design note | `docs/design_pairs_vs_merges.md` | Done |
| **P7** | Limitations + experiment log | this file + `experiments/experiment_log.md` | Done |

## Honest limitations
- Pair classification F1 ≠ discourse tree quality.
- Three sample .txt files do not prove cross-domain generalization.
- Template QUD is not FLAN-T5 unless `--qud-backend t5` and model loads.
- NLI is optional and may fail on Windows; cosine answerability is the reliable default.
- Stage 6 remains adjacent-span greedy.

## Windows commands

```bat
cd E:\Final_year\cross-domain-discourse-parser
git pull

:: P1b
python scripts\compare_relation_backends.py

:: P1c sample with trained relations
python scripts\run_pipeline.py data\raw\news_reuters_sample.txt --relation-backend trained --skip-qud --pair-backend minilm --min-sim 0.10

:: P2 optional (heavy)
python scripts\run_stage2.py --qud-backend t5

:: P3 optional
python scripts\run_stage3.py --use-nli

:: P5 genre hold-out example (retrain)
python scripts\train_relation_classifier.py --holdout-genres news,whow
```

Record real numbers only in `experiments/experiment_log.md`.
