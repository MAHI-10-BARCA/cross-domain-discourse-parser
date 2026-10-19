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

## Quick Start

```bash
git clone https://github.com/MAHI-10-BARCA/cross-domain-discourse-parser.git
cd cross-domain-discourse-parser

# Phase 0 check
python scripts/verify_phase0.py
python -m pytest tests/test_phase0_schemas.py -v

# Phase 1 — EDU Segmentation
python scripts/run_stage1.py --backend fallback
python -m pytest tests/test_stage1_segmentation.py -v
```

Optional (for spaCy backend):
```bash
pip install spacy
python -m spacy download en_core_web_sm
python scripts/run_stage1.py --backend spacy
```

## Pipeline Stages (Approved Architecture)

1. EDU Segmentation (spaCy + fallback) ✅
1.5 Related EDU Pairing (MiniLM)
2. QUD Multi-Question Generation (FLAN-T5-small)
3. Neural Salience Scoring
4. Discourse Relation Classification
5. Answer-Fit Scoring
6. Bottom-Up Greedy Merging → Discourse Tree

## Status

- [x] Phase 0: Project setup and schemas
- [x] Phase 1: EDU Segmentation
- [ ] Phase 1.5: Related EDU Pairing
- [ ] Phase 2: QUD Generation
- [ ] Phase 3: Salience Scoring
- [ ] Phase 4: Relation Classification
- [ ] Phase 5: Answer-Fit Scoring
- [ ] Phase 6: Bottom-Up Parser
- [ ] Phase 7–9: Evaluation, Ablations, Final Deliverables
