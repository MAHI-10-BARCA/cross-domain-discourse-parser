#!/usr/bin/env python3
"""
Improve 7-class (coarse) relation classifier accuracy.

Same GUM gold, document-level split seed=42, same 7 coarse labels.
Improvements: StandardScaler, logreg/mlp/linear_svc/ensemble, optional adjacent-only.

Usage:
  python scripts\\train_coarse_improved.py --clf ensemble
  python scripts\\train_coarse_improved.py --clf mlp
  python scripts\\train_coarse_improved.py --clf logreg --adjacent-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stages.stage4_relation.label_mapping import COARSE_LABELS, MAPPING_VERSION

SEED = 42
LABEL_LIST = list(COARSE_LABELS)


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
    return splits


def embed_pairs(rows, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    texts_i = [r["edu_i_text"] for r in rows]
    texts_j = [r["edu_j_text"] for r in rows]
    dist = np.array([[abs(r["edu_i_index"] - r["edu_j_index"])] for r in rows], dtype=np.float32)
    dmax = float(dist.max()) if len(dist) else 1.0
    dist = dist / (dmax + 1e-9)
    model = SentenceTransformer(model_name)
    emb_i = np.asarray(model.encode(texts_i, batch_size=64, show_progress_bar=True, convert_to_numpy=True), dtype=np.float32)
    emb_j = np.asarray(model.encode(texts_j, batch_size=64, show_progress_bar=True, convert_to_numpy=True), dtype=np.float32)
    diff = np.abs(emb_i - emb_j)
    prod = emb_i * emb_j
    ni = np.linalg.norm(emb_i, axis=1, keepdims=True)
    nj = np.linalg.norm(emb_j, axis=1, keepdims=True)
    cos = (emb_i * emb_j).sum(axis=1, keepdims=True) / np.maximum(ni * nj, 1e-9)
    return np.concatenate([emb_i, emb_j, diff, prod, cos, dist], axis=1)


def to_y(rows):
    idx = {lab: i for i, lab in enumerate(LABEL_LIST)}
    return np.array([idx[r["coarse_label"]] for r in rows], dtype=np.int64)


def evaluate(clf, X, y):
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    pred = clf.predict(X)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "report": classification_report(y, pred, target_names=LABEL_LIST, zero_division=0, output_dict=True),
        "n": int(len(y)),
        "true_dist": dict(Counter(LABEL_LIST[i] for i in y)),
        "pred_dist": dict(Counter(LABEL_LIST[i] for i in pred)),
    }


def build_pipeline(clf_name: str):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import VotingClassifier

    if clf_name == "logreg":
        clf = LogisticRegression(max_iter=3000, solver="lbfgs", class_weight="balanced", C=3.0, random_state=SEED)
    elif clf_name == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(768, 384, 128), activation="relu", solver="adam", alpha=5e-4,
            batch_size=256, learning_rate_init=1e-3, max_iter=40, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=6, random_state=SEED, verbose=True,
        )
    elif clf_name == "linear_svc":
        base = LinearSVC(class_weight="balanced", C=1.0, max_iter=8000, random_state=SEED, dual=False)
        clf = CalibratedClassifierCV(base, cv=3)
    elif clf_name == "ensemble":
        lr = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", C=2.0, random_state=SEED)
        mlp = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=25, early_stopping=True, random_state=SEED, verbose=False)
        clf = VotingClassifier(estimators=[("lr", lr), ("mlp", mlp)], voting="soft")
    else:
        raise ValueError(clf_name)
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "data" / "gold" / "gum_v12_relations.jsonl"))
    ap.add_argument("--clf", default="ensemble", choices=["logreg", "mlp", "linear_svc", "ensemble"])
    ap.add_argument("--adjacent-only", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.gold))
    if args.adjacent_only:
        rows = [r for r in rows if abs(int(r["edu_i_index"]) - int(r["edu_j_index"])) == 1]
        print(f"Adjacent-only filter → {len(rows)} pairs")

    splits = document_level_split(rows)
    print("Split:", {k: len(v) for k, v in splits.items()})

    print("Embedding train/val/test…")
    X_train = embed_pairs(splits["train"], args.model_name)
    X_val = embed_pairs(splits["val"], args.model_name)
    X_test = embed_pairs(splits["test"], args.model_name)
    y_train, y_val, y_test = to_y(splits["train"]), to_y(splits["val"]), to_y(splits["test"])
    print(f"X_train {X_train.shape}")

    pipe = build_pipeline(args.clf)
    print(f"Fitting {args.clf} (with StandardScaler)…")
    pipe.fit(X_train, y_train)

    val_m = evaluate(pipe, X_val, y_val)
    test_m = evaluate(pipe, X_test, y_test)
    print(f"VAL  acc={val_m['accuracy']:.4f} macro-F1={val_m['macro_f1']:.4f} w-F1={val_m['weighted_f1']:.4f}")
    print(f"TEST acc={test_m['accuracy']:.4f} macro-F1={test_m['macro_f1']:.4f} w-F1={test_m['weighted_f1']:.4f}")

    tag = args.clf + ("_adj" if args.adjacent_only else "")
    out_dir = Path(args.out_dir or (ROOT / "models" / f"coarse_improved_{tag}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    artifact = {
        "classifier": pipe,
        "vectorizer": None,
        "encoder_backend": "minilm",
        "label_list": LABEL_LIST,
        "encoder_name": args.model_name,
        "feature_schema": ["emb_i", "emb_j", "abs_diff", "product", "cosine", "norm_distance"],
        "mapping_version": MAPPING_VERSION,
        "label_mode": "coarse",
        "clf_name": args.clf,
        "adjacent_only": args.adjacent_only,
        "uses_scaler": True,
        "corpus": "GUM-V12.1.0",
        "seed": SEED,
        "val_metrics": {"accuracy": val_m["accuracy"], "macro_f1": val_m["macro_f1"], "weighted_f1": val_m["weighted_f1"]},
        "test_metrics": {"accuracy": test_m["accuracy"], "macro_f1": test_m["macro_f1"], "weighted_f1": test_m["weighted_f1"]},
    }
    model_path = out_dir / "relation_logreg.joblib"
    joblib.dump(artifact, model_path)
    (out_dir / "relation_eval.json").write_text(json.dumps({"val": val_m, "test": test_m, "clf": args.clf, "adjacent_only": args.adjacent_only}, indent=2), encoding="utf-8")
    print(f"Saved → {model_path}")
    print(f"If TEST acc improves over 0.433: copy /Y {model_path} models\\relation_logreg.joblib")


if __name__ == "__main__":
    main()
