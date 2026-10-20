#!/usr/bin/env python3
"""
Train MiniLM + LogisticRegression discourse relation classifier on GUM gold.

Usage:
  python scripts\\train_relation_classifier.py
  python scripts\\train_relation_classifier.py --holdout-genres news
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stages.stage4_relation.label_mapping import COARSE_LABELS, MAPPING_VERSION

SEED = 42


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def document_level_split(rows, seed=SEED, train_r=0.70, val_r=0.15):
    rng = np.random.RandomState(seed)
    docs = sorted({r["doc_id"] for r in rows})
    rng.shuffle(docs)
    n = len(docs)
    n_train = max(1, int(n * train_r))
    n_val = max(1, int(n * val_r))
    if n_train + n_val >= n:
        n_val = max(1, (n - n_train) // 2)
    train_docs = set(docs[:n_train])
    val_docs = set(docs[n_train : n_train + n_val])
    test_docs = set(docs[n_train + n_val :])
    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        if r["doc_id"] in train_docs:
            splits["train"].append(r)
        elif r["doc_id"] in val_docs:
            splits["val"].append(r)
        else:
            splits["test"].append(r)
    meta = {
        "seed": seed,
        "n_docs_total": n,
        "n_docs_train": len(train_docs),
        "n_docs_val": len(val_docs),
        "n_docs_test": len(test_docs),
        "train_docs": sorted(train_docs),
        "val_docs": sorted(val_docs),
        "test_docs": sorted(test_docs),
        "n_examples": {k: len(v) for k, v in splits.items()},
        "label_dist": {k: dict(Counter(r["coarse_label"] for r in v)) for k, v in splits.items()},
    }
    return splits, meta


def embed_pairs(rows, model_name="sentence-transformers/all-MiniLM-L6-v2", vectorizer=None, fit=False):
    texts_i = [r["edu_i_text"] for r in rows]
    texts_j = [r["edu_j_text"] for r in rows]
    dist = np.array([[abs(r["edu_i_index"] - r["edu_j_index"])] for r in rows], dtype=np.float32)
    dist = dist / (dist.max() + 1e-9)
    try:
        import torch  # noqa: F401
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        emb_i = np.asarray(model.encode(texts_i, batch_size=64, show_progress_bar=True, convert_to_numpy=True), dtype=np.float32)
        emb_j = np.asarray(model.encode(texts_j, batch_size=64, show_progress_bar=True, convert_to_numpy=True), dtype=np.float32)
        diff = np.abs(emb_i - emb_j)
        prod = emb_i * emb_j
        ni = np.linalg.norm(emb_i, axis=1, keepdims=True)
        nj = np.linalg.norm(emb_j, axis=1, keepdims=True)
        cos = (emb_i * emb_j).sum(axis=1, keepdims=True) / np.maximum(ni * nj, 1e-9)
        X = np.concatenate([emb_i, emb_j, diff, prod, cos, dist], axis=1)
        return X, None, "minilm"
    except Exception as e:
        print(f"MiniLM unavailable ({e}); using TF-IDF features for supervised training.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        combined = [a + " [SEP] " + b for a, b in zip(texts_i, texts_j)]
        if fit or vectorizer is None:
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
            X_txt = vectorizer.fit_transform(combined)
        else:
            X_txt = vectorizer.transform(combined)
        X = np.hstack([X_txt.toarray().astype(np.float32), dist])
        return X, vectorizer, "tfidf"


def labels_to_y(rows, label_list=COARSE_LABELS):
    idx = {lab: i for i, lab in enumerate(label_list)}
    return np.array([idx[r["coarse_label"]] for r in rows], dtype=np.int64)


def evaluate(clf, X, y, label_list=COARSE_LABELS):
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    pred = clf.predict(X)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y, pred, target_names=label_list, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(label_list)))).tolist(),
        "n_examples": int(len(y)),
        "label_dist_true": dict(Counter(label_list[i] for i in y)),
        "label_dist_pred": dict(Counter(label_list[i] for i in pred)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "data" / "gold" / "gum_v12_relations.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "models"))
    ap.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument(
        "--holdout-genres",
        default=None,
        help="P5: comma-separated genres held out as test (e.g. news,whow)",
    )
    args = ap.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"ERROR: gold file missing: {gold_path}")
        sys.exit(1)

    rows = load_jsonl(gold_path)
    print(f"Loaded {len(rows)} gold examples from {gold_path}")

    if args.holdout_genres:
        hold = {g.strip().lower() for g in args.holdout_genres.split(",") if g.strip()}
        test_rows = [r for r in rows if str(r.get("genre", "")).lower() in hold]
        train_pool = [r for r in rows if str(r.get("genre", "")).lower() not in hold]
        if not test_rows or not train_pool:
            print(f"ERROR: holdout-genres={hold} left empty train or test.")
            sys.exit(1)
        splits_pool, meta_pool = document_level_split(train_pool, seed=SEED, train_r=0.85, val_r=0.15)
        splits = {
            "train": splits_pool["train"],
            "val": splits_pool["val"] + splits_pool["test"],
            "test": test_rows,
        }
        split_meta = {
            **meta_pool,
            "holdout_genres": sorted(hold),
            "n_examples": {k: len(v) for k, v in splits.items()},
            "n_docs_test": len({r["doc_id"] for r in test_rows}),
            "test_docs": sorted({r["doc_id"] for r in test_rows}),
            "label_dist": {k: dict(Counter(r["coarse_label"] for r in v)) for k, v in splits.items()},
        }
        print(f"P5 genre hold-out test genres={sorted(hold)} n_test={len(test_rows)}")
    else:
        splits, split_meta = document_level_split(rows, seed=SEED)

    print("Split:", split_meta["n_examples"])
    train_rows = splits["train"]
    if args.max_train:
        train_rows = train_rows[: args.max_train]

    print("Embedding train…")
    X_train, vectorizer, enc_backend = embed_pairs(train_rows, args.model_name, fit=True)
    y_train = labels_to_y(train_rows)
    print(f"Encoder backend: {enc_backend}; X_train shape={X_train.shape}")
    print("Embedding val…")
    X_val, _, _ = embed_pairs(splits["val"], args.model_name, vectorizer=vectorizer, fit=False)
    y_val = labels_to_y(splits["val"])
    print("Embedding test…")
    X_test, _, _ = embed_pairs(splits["test"], args.model_name, vectorizer=vectorizer, fit=False)
    y_test = labels_to_y(splits["test"])

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=SEED, C=1.0,
    )
    print("Fitting LogisticRegression…")
    clf.fit(X_train, y_train)

    val_report = evaluate(clf, X_val, y_val)
    test_report = evaluate(clf, X_test, y_test)
    print("VAL  accuracy={:.4f} macro_f1={:.4f} weighted_f1={:.4f}".format(
        val_report["accuracy"], val_report["macro_f1"], val_report["weighted_f1"]))
    print("TEST accuracy={:.4f} macro_f1={:.4f} weighted_f1={:.4f}".format(
        test_report["accuracy"], test_report["macro_f1"], test_report["weighted_f1"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    feature_schema = (
        ["emb_i", "emb_j", "abs_diff", "product", "cosine", "norm_distance"]
        if enc_backend == "minilm" else ["tfidf_pair", "norm_distance"]
    )
    artifact = {
        "classifier": clf,
        "vectorizer": vectorizer,
        "encoder_backend": enc_backend,
        "label_list": COARSE_LABELS,
        "encoder_name": args.model_name if enc_backend == "minilm" else "tfidf",
        "feature_schema": feature_schema,
        "mapping_version": MAPPING_VERSION,
        "corpus": "GUM-V12.1.0",
        "seed": SEED,
        "split_meta": {k: split_meta.get(k) for k in (
            "seed", "n_docs_total", "n_docs_train", "n_docs_val", "n_docs_test",
            "n_examples", "label_dist", "holdout_genres",
        ) if k in split_meta or k == "holdout_genres"},
        "hyperparams": {
            "model": "LogisticRegression", "max_iter": 1000, "solver": "lbfgs",
            "class_weight": "balanced", "C": 1.0, "random_state": SEED,
            "encoder_backend": enc_backend,
        },
        "val_metrics": {
            "accuracy": val_report["accuracy"],
            "macro_f1": val_report["macro_f1"],
            "weighted_f1": val_report["weighted_f1"],
        },
        "test_metrics": {
            "accuracy": test_report["accuracy"],
            "macro_f1": test_report["macro_f1"],
            "weighted_f1": test_report["weighted_f1"],
        },
    }
    model_path = out_dir / "relation_logreg.joblib"
    joblib.dump(artifact, model_path)
    print(f"Saved model → {model_path}")

    eval_path = out_dir / "relation_eval.json"
    eval_path.write_text(
        json.dumps({
            "val": val_report,
            "test": test_report,
            "split_meta": {
                k: split_meta[k] for k in split_meta
                if k not in ("train_docs", "val_docs", "test_docs")
            },
            "mapping_version": MAPPING_VERSION,
            "holdout_genres": split_meta.get("holdout_genres"),
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Saved eval → {eval_path}")


if __name__ == "__main__":
    main()
