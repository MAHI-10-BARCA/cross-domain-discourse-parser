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
- Phase / Stage: P1 trained relation classifier
- Config: GUM V12.1.0, document-level 70/15/15, seed=42, MiniLM + LogReg balanced
- Command: `python scripts\train_relation_classifier.py`
- Key metrics (from training log; pair-level only):
  - VAL ~acc 0.37 macro-F1 ~0.32
  - TEST ~acc 0.42 macro-F1 ~0.34
- Artifacts: `models\relation_logreg.joblib`, `models\relation_eval.json`, `data\gold\gum_v12_relations.jsonl`
- Status: success (training). Inference fixed with torch==2.10.0 + torchvision==0.25.0
- Does NOT claim cross-domain generalization or tree UAS

### Run ID: pending-P1b
- Command: `python scripts\compare_relation_backends.py`
- Fill accuracy / macro-F1 for heuristic vs trained after local run

### Design notes
- See `docs\design_pairs_vs_merges.md` (P6)
- See `docs\remaining_priorities_status.md`
