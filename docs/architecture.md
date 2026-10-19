# Approved Architecture

## Pipeline

```
Input document
    ↓
Stage 1:   EDU segmentation          (spaCy en_core_web_sm, sentence-level)
    ↓
Stage 1.5: Related EDU pairing       (all-MiniLM-L6-v2, Top-K + distance)
    ↓
Stage 2:   QUD-based multi-question generation  (google/flan-t5-small)
    ↓
Stage 3:   Neural salience scoring   (centrality + answerability)
    ↓
Stage 4:   Discourse relation classification
    ↓
Stage 5:   Answer-fit scoring
    ↓
Stage 6:   Bottom-up greedy merging
    ↓
Final hierarchical discourse tree
    ↓
Evaluation and cross-domain analysis
```

## Stage Contracts (high-level)

| Stage | Input | Output |
|-------|-------|--------|
| 1     | raw Document | List[EDU] |
| 1.5   | List[EDU] | List[EDUPair] |
| 2     | List[EDUPair] + EDUs | List[QUDQuestion] |
| 3     | List[QUDQuestion] + context | List[SalienceScore] |
| 4     | pairs / features | List[RelationPrediction] |
| 5     | questions + EDUs | List[AnswerFitScore] |
| 6     | all previous + EDUs | DiscourseTree |

All intermediate results are saved as JSON under `outputs/intermediate/`.

## Design Principles

- Bottom-up construction (not top-down).
- Sentence-level EDUs in the first implementation.
- Explicit, documented pairing constraints (Top-K=3, max distance=5 as starting point).
- Multiple QUD candidates per pair.
- Traceable scores at every stage.
- No silent architecture changes.
