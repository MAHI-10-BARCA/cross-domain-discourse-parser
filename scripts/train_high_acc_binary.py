#!/usr/bin/env python3
"""
High-accuracy experiments on RESTRICTED tasks (honest framing).

Full 7/32-way GUM pair classification does not reach ~85% with MiniLM+sklearn.
Secondary experiments with easier targets where higher accuracy is achievable:

  A) Binary: one coarse relation vs rest  (default: Contrast)
  B) Top-frequency coarse classes only (e.g. top-3)

Document-level split, seed=42, MiniLM features, no leakage.

Usage:
  python scripts\\train_high_acc_binary.py --task binary --positive Contrast
  python scripts\\train_high_acc_binary.py --task topk --top-k 3
  python scripts\\train_high_acc_binary.py --task binary --positive Elaboration
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
    return splits


def embed_pairs(rows, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    texts_i = [r["edu_i_text"] for r in rows]
    texts_j = [r["edu_j_text"] for r in rows]
    dist = np.array([[abs(r["edu_i_index"] - r["edu_j_index"])] for r in rows], dtype=np.float32)
    dist = dist / (max(float(dist.max()), 1e-9))
    model = SentenceTransformer(model_name)
    emb_i = np.asarray(model.encode(texts_i, batch_size=64, show_progress_bar=True, convert_to_numpy=True), dtype=np.float32)
    emb_j = np.asarray(model.encode(texts_j, batch_size=64, show_progress_bar=True, convert_to_numpy=True), dtype=np.float32)
    diff = np.abs(emb_i - emb_j)
    prod = emb_i * emb_j
    ni = np.linalg.norm(emb_i, axis=1, keepdims=True)
    nj = np.linalg.norm(emb_j, axis=1, keepdims=True)
    cos = (emb_i * emb_j).sum(axis=1, keepdims=True) / np.maximum(ni * nj, 1e-9)
    return np.concatenate([emb_i, emb_j, diff, prod, cos, dist], axis=1)


def make_labels_binary(rows, positive: str):
    y = np.array([1 if r["coarse_label"] == positive else 0 for r in rows], dtype=np.int64)
    return y, [f"NOT_{positive}", positive]


def train_eval(X_train, y_train, X_val, y_val, X_test, y_test, label_list):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", C=1.0, random_state=SEED)
    clf.fit(X_train, y_train)

    def pack(X, y):
        pred = clf.predict(X)
        out = {
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "report": classification_report(y, pred, target_names=label_list, zero_division=0, output_dict=True),
            "n": int(len(y)),
            "true_dist": dict(Counter(label_list[i] for i in y)),
            "pred_dist": dict(Counter(label_list[i] for i in pred)),
        }
        if len(label_list) == 2 and hasattr(clf, "predict_proba"):
            try:
                proba = clf.predict_proba(X)[:, 1]
                out["roc_auc"] = float(roc_auc_score(y, proba))
            except Exception:
                pass
        return out

    return clf, pack(X_val, y_val), pack(X_test, y_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(ROOT / "data" / "gold" / "gum_v12_relations.jsonl"))
    ap.add_argument("--task", choices=["binary", "topk"], default="binary")
    ap.add_argument("--positive", default="Contrast")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.gold))
    splits = document_level_split(rows)

    if args.task == "binary":
        y_train, label_list = make_labels_binary(splits["train"], args.positive)
        y_val, _ = make_labels_binary(splits["val"], args.positive)
        y_test, _ = make_labels_binary(splits["test"], args.positive)
        task_name = f"binary_{args.positive}_vs_rest"
        print(f"Task: {task_name}")
        print(f"  train pos rate: {y_train.mean():.3f}  test pos rate: {y_test.mean():.3f}")
    else:
        train_counts = Counter(r["coarse_label"] for r in splits["train"])
        top = [lab for lab, _ in train_counts.most_common(args.top_k)]
        kept = set(top)
        label_list = sorted(top) + ["OTHER_TAIL"]
        idx = {lab: i for i, lab in enumerate(label_list)}

        def to_y(rs):
            return np.array([
                idx[r["coarse_label"] if r["coarse_label"] in kept else "OTHER_TAIL"]
                for r in rs
            ], dtype=np.int64)

        y_train, y_val, y_test = to_y(splits["train"]), to_y(splits["val"]), to_y(splits["test"])
        task_name = f"topk_{args.top_k}_{'_'.join(top)}"
        print(f"Task: {task_name}")
        print(f"  labels: {label_list}")

    print("Embedding…")
    X_train = embed_pairs(splits["train"], args.model_name)
    X_val = embed_pairs(splits["val"], args.model_name)
    X_test = embed_pairs(splits["test"], args.model_name)

    clf, val_m, test_m = train_eval(X_train, y_train, X_val, y_val, X_test, y_test, label_list)

    print(f"\nVAL  acc={val_m['accuracy']:.4f} macro-F1={val_m['macro_f1']:.4f} w-F1={val_m['weighted_f1']:.4f}")
    print(f"TEST acc={test_m['accuracy']:.4f} macro-F1={test_m['macro_f1']:.4f} w-F1={test_m['weighted_f1']:.4f}")
    if "roc_auc" in test_m:
        print(f"TEST ROC-AUC={test_m['roc_auc']:.4f}")

    out_dir = Path(args.out_dir or (ROOT / "models" / task_name))
    out_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({
        "classifier": clf,
        "encoder_backend": "minilm",
        "label_list": label_list,
        "encoder_name": args.model_name,
        "task": task_name,
        "task_type": args.task,
        "positive": args.positive if args.task == "binary" else None,
        "corpus": "GUM-V12.1.0",
        "seed": SEED,
        "val_metrics": val_m,
        "test_metrics": test_m,
        "disclaimer": "Restricted task — NOT full 7-class discourse relation accuracy.",
    }, out_dir / "model.joblib")
    (out_dir / "eval.json").write_text(json.dumps({
        "task": task_name, "label_list": label_list, "val": val_m, "test": test_m,
        "disclaimer": "Restricted task; not full multi-class RST accuracy.",
    }, indent=2), encoding="utf-8")
    print(f"Saved → {out_dir}")
    print("\nIMPORTANT: Restricted task scores — not full 7-class GUM accuracy.")


if __name__ == "__main__":
    main()
