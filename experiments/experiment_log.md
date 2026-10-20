# Experiment Log

Record every run here. Never invent numbers.

## Template

```
### Run ID: YYYYMMDD-HHMM-shortname
- Date:
- Phase / Stage:
- Config changes:
- Command:
- Hardware notes:
- Key metrics:
- Observations:
- Artifacts path:
- Status: success / failed / partial
```

## Entries

### Run ID: 20261019-P1-gum-minilm-logreg
- Date: 2026-10-19
- Phase: P1 trained relation classifier (training-time eval on held-out test)
- Config: GUM V12.1.0, document-level 70/15/15, seed=42, MiniLM + LogReg balanced
- Command: `python scripts\\train_relation_classifier.py`
- Key metrics (pair-level, from training script):
  - VAL ~acc 0.37, macro-F1 ~0.32
  - TEST ~acc 0.42, macro-F1 ~0.34
- Artifacts: `models\\relation_logreg.joblib`, `models\\relation_eval.json`
- Status: success

### Run ID: 20261020-P1b-full
- Date: 2026-10-20
- Phase: P1b heuristic vs trained on same GUM gold test pairs
- Command: `python scripts\\compare_relation_backends.py`
- n_pairs: 5853 (~46 docs, document-level test, seed=42)
- heuristic:  acc=0.3147  macro-F1=0.2209  weighted-F1=0.2400
- trained:    acc=0.4329  macro-F1=0.3520  weighted-F1=0.4567
- Artifact: `outputs\\evaluation\\p1b_backend_compare.json`
- Status: success
- Note: pair classification only; not tree attachment F1; not cross-domain claim

### Run ID: 20261020-P1c-news-trained
- Date: 2026-10-20
- Phase: P1c pipeline with --relation-backend trained
- Command: `python scripts\\run_pipeline.py data\\raw\\news_reuters_sample.txt --relation-backend trained --skip-qud --pair-backend minilm --min-sim 0.10`
- Result: 6 EDUs, 11 pairs, Stage4 trained labels mostly Joint (10) + Result (1); tree 5 merges height=5
- Observation: "However" pair not labeled Contrast by trained model (Joint-heavy); differs from heuristic cue behavior — expected
- Status: success
