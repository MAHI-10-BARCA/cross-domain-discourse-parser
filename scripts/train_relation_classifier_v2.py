#!/usr/bin/env python3
"""
Improved relation classifier training (accuracy push).

- Default labels = GUM *fine* relations (no coarse 7-way mapping)
- Optional --min-count collapses rare fine labels to __RARE__
- Classifiers: logreg | linear_svc | mlp
- Document-level split seed=42, MiniLM pair features

Honesty: 85% accuracy on multi-way RST is very hard. Report real test metrics only.

Usage:
  python scripts\\train_relation_classifier_v2.py --label-mode fine --clf mlp
  python scripts\\train_relation_classifier_v2.py --label-mode fine --clf logreg --min-count 200
  python scripts\\train_relation_classifier_v2.py --label-mode coarse --clf logreg
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
        "n_examples": {k: len(v) for k, v in splits.items()},
    }
    return splits, meta


def resolve_label(row, label_mode: str, rare_map):
    if label_mode == "coarse":
        return row["coarse_label"]
    lab = row.get("fine_label") or row["coarse_label"]
    if rare_map is not None:
        return rare_map.get(lab, lab)
    return lab


def build_label_list(train_rows, label_mode: str, min_count: int):
    if label_mode == "coarse":
        return list(COARSE_LABELS), None
    counts = Counter(r["fine_label"] for r in train_rows)
    rare_map = {}
    kept = []
    for lab, c in counts.most_common():
        if c < min_count:
            rare_map[lab] = "__RARE__"
        else:
            kept.append(lab)
    if rare_map:
        kept.append("__RARE__")
    return sorted(kept), rare_map


def embed_pairs(rows, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    texts_i = [r["edu_i_text"] for r in rows]
    texts_j = [r["edu_j_text"] for r in rows]
    dist = np.array([[abs(r["edu_i_index"] - r["edu_j_index"])] for r in rows], dtype=np.float32)
    dist = dist / (dist.max() + 1e-9)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    emb_i = np.asarray(model.encode(texts_i, batch_size=64, show_progress_bar=True, convert_to_numpy=True), dtype=np.float32)
    emb_j = np.asarray(model.encode(texts_j, batch_size=64, show_progress_bar=True, convert_to_numpy=True), dtype=np.float32)
    diff = np.abs(emb_i - emb_j)
    prod = emb_i * emb_j
    ni = np.linalg.norm(emb_i, axis=1, keepdims=True)
    nj = np.linalg.norm(emb_j, axis=1, keepdims=True)
    cos = (emb_i * emb_j).sum(axis=1, keepdims=True) / np.maximum(ni * nj, 1e-9)
    return np.concatenate([emb_i, emb_j, diff, prod, cos, dist], axis=1)


def labels_to_y(rows, label_list, label_mode, rare_map):
    idx = {lab: i for i, lab in enumerate(label_list)}
    y = []
    for r in rows:
        lab = resolve_label(r, label_mode, rare_map)
        if lab not in idx:
            lab = "__RARE__" if "__RARE__" in idx else label_list[0]
        y.append(idx[lab])
    return np.array(y, dtype=np.int64)


def make_clf(name: str, seed: int = SEED):
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", C=2.0, random_state=seed)
    if name == "linear_svc":
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        base = LinearSVC(class_weight="balanced", C=1.0, max_iter=5000, random_state=seed)
        return CalibratedClassifierCV(base, cv=3)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=(512, 256), activation="relu", solver="adam", alpha=1e-4,
            batch_size=256, max_iter=30, random_state=seed, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=5, verbose=True,
        )
    raise ValueError(name)


def evaluate(clf, X, y, label_list):
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
    ap.add_argument("--out-dir", default=str(ROOT / "models" / "fine_mlp"))
    ap.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--label-mode", default="fine", choices=["fine", "coarse"])
    ap.add_argument("--clf", default="mlp", choices=["logreg", "linear_svc", "mlp"])
    ap.add_argument("--min-count", type=int, default=100)
    ap.add_argument("--max-train", type=int, default=None)
    args = ap.parse_args()

    gold_path = Path(args.gold)
    if not gold_path.is_file():
        print(f"ERROR: missing {gold_path}")
        sys.exit(1)

    rows = load_jsonl(gold_path)
    print(f"Loaded {len(rows)} examples")
    splits, split_meta = document_level_split(rows)
    print("Split sizes:", split_meta["n_examples"])

    train_rows = splits["train"]
    if args.max_train:
        train_rows = train_rows[: args.max_train]

    label_list, rare_map = build_label_list(train_rows, args.label_mode, args.min_count)
    print(f"label_mode={args.label_mode} n_labels={len(label_list)} clf={args.clf}")
    if rare_map:
        print(f"  rare fine labels collapsed: {len(rare_map)}")

    print("Embedding train…")
    X_train = embed_pairs(train_rows, args.model_name)
    y_train = labels_to_y(train_rows, label_list, args.label_mode, rare_map)
    print(f"X_train {X_train.shape}")
    print("Embedding val…")
    X_val = embed_pairs(splits["val"], args.model_name)
    y_val = labels_to_y(splits["val"], label_list, args.label_mode, rare_map)
    print("Embedding test…")
    X_test = embed_pairs(splits["test"], args.model_name)
    y_test = labels_to_y(splits["test"], label_list, args.label_mode, rare_map)

    clf = make_clf(args.clf)
    print(f"Fitting {args.clf}…")
    clf.fit(X_train, y_train)

    val_report = evaluate(clf, X_val, y_val, label_list)
    test_report = evaluate(clf, X_test, y_test, label_list)
    print("VAL  acc={:.4f} macro-F1={:.4f} w-F1={:.4f}".format(val_report["accuracy"], val_report["macro_f1"], val_report["weighted_f1"]))
    print("TEST acc={:.4f} macro-F1={:.4f} w-F1={:.4f}".format(test_report["accuracy"], test_report["macro_f1"], test_report["weighted_f1"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    artifact = {
        "classifier": clf,
        "vectorizer": None,
        "encoder_backend": "minilm",
        "label_list": label_list,
        "encoder_name": args.model_name,
        "feature_schema": ["emb_i", "emb_j", "abs_diff", "product", "cosine", "norm_distance"],
        "mapping_version": "fine_native" if args.label_mode == "fine" else MAPPING_VERSION,
        "label_mode": args.label_mode,
        "rare_map": rare_map,
        "corpus": "GUM-V12.1.0",
        "seed": SEED,
        "clf_name": args.clf,
        "min_count": args.min_count,
        "split_meta": split_meta,
        "val_metrics": {"accuracy": val_report["accuracy"], "macro_f1": val_report["macro_f1"], "weighted_f1": val_report["weighted_f1"]},
        "test_metrics": {"accuracy": test_report["accuracy"], "macro_f1": test_report["macro_f1"], "weighted_f1": test_report["weighted_f1"]},
    }
    model_path = out_dir / "relation_logreg.joblib"
    joblib.dump(artifact, model_path)
    print(f"Saved → {model_path}")
    eval_path = out_dir / "relation_eval.json"
    eval_path.write_text(json.dumps({"val": val_report, "test": test_report, "label_list": label_list, "label_mode": args.label_mode, "clf": args.clf, "min_count": args.min_count, "n_labels": len(label_list)}, indent=2), encoding="utf-8")
    print(f"Saved eval → {eval_path}")


if __name__ == "__main__":
    main()
