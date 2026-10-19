# Implementation Checklist

## Phase 0 — Setup ✅
- [x] Clean directory structure
- [x] Shared data schemas
- [x] Config file
- [x] Seed & logging utilities
- [x] Architecture & schema docs
- [x] Verification script + unit tests

## Phase 1 — EDU Segmentation ✅
- [x] spaCy sentence segmentation (primary) + pure-Python fallback
- [x] Stable EDU IDs + character offsets
- [x] JSON output
- [x] Unit tests (short / medium / long docs)
- [x] Acceptance: reproducible, text preserved, stable IDs

## Phase 1.5 — Related EDU Pairing ✅
- [x] MiniLM embeddings (primary) + TF-IDF fallback
- [x] Cosine similarity
- [x] Top-K + max-distance selection
- [x] Comparison vs adjacent baseline
- [x] Acceptance: reproducible pairs, quality measured (coverage stats)

## Phase 2 — QUD Generation
- [ ] FLAN-T5-small (configurable)
- [ ] Templates + multi-question generation
- [ ] Validation & rejection logging
- [ ] Acceptance: reproducible, metadata kept, quality evaluated

## Phase 3 — Salience Scoring
- [ ] Centrality (MiniLM)
- [ ] Answerability proxy (NLI)
- [ ] Weighted combination + normalization
- [ ] Acceptance: traceable component scores

## Phase 4 — Relation Classification
- [ ] Label inventory
- [ ] Dataset + splits (no leakage)
- [ ] Baseline (MiniLM + LogReg)
- [ ] Full metrics + confusion matrix
- [ ] Acceptance: held-out evaluation, errors understood

## Phase 5 — Answer-Fit
- [ ] MiniLM cosine fit
- [ ] Ablation vs Stage 3
- [ ] Acceptance: reproducible & justified

## Phase 6 — Bottom-up Parser
- [ ] Greedy merge with priority
- [ ] Tree validation
- [ ] Visualization / textual dump
- [ ] Acceptance: structurally valid trees, traceable decisions

## Phase 7–9
- [ ] Evaluation protocol
- [ ] Cross-domain experiments
- [ ] Ablations & error analysis
- [ ] Final report & reproducibility package
