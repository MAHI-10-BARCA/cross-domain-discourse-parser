# Implementation Checklist

## Phase 0 — Setup ✅
- [x] Clean directory structure
- [x] Shared data schemas
- [x] Config file
- [x] Seed & logging utilities
- [x] Architecture & schema docs
- [x] Verification script + unit tests

## Phase 1 — EDU Segmentation
- [ ] spaCy sentence segmentation
- [ ] Stable EDU IDs + offsets
- [ ] JSON output
- [ ] Unit tests (short / medium / long docs)
- [ ] Acceptance: reproducible, text preserved, stable IDs

## Phase 1.5 — Related EDU Pairing
- [ ] MiniLM embeddings
- [ ] Cosine similarity
- [ ] Top-K + max-distance selection
- [ ] Comparison vs adjacent baseline
- [ ] Acceptance: reproducible pairs, quality measured

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
