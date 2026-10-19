# Enhancing Cross-Domain Document Discourse Parsing Through Bottom-Up Learning

Final Year Project (BTech Information Technology)

## Overview

This project builds a discourse-aware document processing framework that:
1. Segments documents into Elementary Discourse Units (EDUs)
2. Identifies related EDU pairs
3. Generates Question-Under-Discussion (QUD) candidates
4. Scores salience and answer-fit
5. Classifies discourse relations
6. Constructs a hierarchical discourse tree via bottom-up greedy merging

## Project Structure

```
discourse_parser/
├── config/           # Configuration files
├── src/              # Source code
│   ├── data_schemas.py
│   ├── utils/
│   └── stages/       # One package per pipeline stage
├── tests/            # Unit and integration tests
├── data/             # Raw, processed, and gold data
├── models/           # Model checkpoints and caches
├── outputs/          # Intermediate results, trees, logs
├── experiments/      # Experiment logs and configs
├── docs/             # Architecture and schema docs
└── scripts/          # Utility and verification scripts
```

## Quick Start (Phase 0)

```bash
cd discourse_parser
python scripts/verify_phase0.py
pytest tests/test_phase0_schemas.py -v
```

## Pipeline Stages (Approved Architecture)

1. EDU Segmentation (spaCy)
1.5 Related EDU Pairing (MiniLM)
2. QUD Multi-Question Generation (FLAN-T5-small)
3. Neural Salience Scoring
4. Discourse Relation Classification
5. Answer-Fit Scoring
6. Bottom-Up Greedy Merging → Discourse Tree

## Hardware Notes

- CPU-only environment (no GPU)
- ~1.2 GB RAM → keep models small and batch sizes = 1
- Prefer `all-MiniLM-L6-v2` and `flan-t5-small`

## Status

- [x] Phase 0: Project setup and schemas
- [ ] Phase 1: EDU Segmentation
- [ ] Phase 1.5: Related EDU Pairing
- [ ] Phase 2: QUD Generation
- [ ] Phase 3: Salience Scoring
- [ ] Phase 4: Relation Classification
- [ ] Phase 5: Answer-Fit Scoring
- [ ] Phase 6: Bottom-Up Parser
- [ ] Phase 7–9: Evaluation, Ablations, Final Deliverables
