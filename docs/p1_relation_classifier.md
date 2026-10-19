# P1 — Trained Discourse Relation Classifier

## Corpus
- **GUM V12.1.0** — https://github.com/amir-zeldes/gum/releases/tag/V12.1.0
- Format: `rst/dependencies/*.rsd`
- License: see GUM LICENSE.md (CC; some genres CC-BY-NC-SA)

## Acquisition (Windows CMD)
```bat
cd E:\Final_year
curl -L -o gum-V12.1.0.tgz https://codeload.github.com/amir-zeldes/gum/tar.gz/refs/tags/V12.1.0
tar -xzf gum-V12.1.0.tgz
cd cross-domain-discourse-parser
python scripts\convert_gum_rsd.py --gum-root E:\Final_year\gum-12.1.0
python scripts\train_relation_classifier.py
```

## Mapping
See `src/stages/stage4_relation/label_mapping.py` (MAPPING_VERSION=gum_v12.1.0_map_v1).

## Backends
- `RelationClassifier(backend="heuristic")` — default, P0 preserved
- `RelationClassifier(backend="trained")` — requires `models/relation_logreg.joblib`

Missing model → FileNotFoundError (no silent fallback).

Retrain on your machine with MiniLM for the intended model; sandbox used TF-IDF fallback when sentence-transformers was unavailable.
