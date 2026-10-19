#!/usr/bin/env python3
"""
Convert GUM V12.x RST dependency (.rsd) files to gold training JSONL.

Usage (Windows CMD from project root):
  python scripts\\convert_gum_rsd.py --gum-root E:\\path\\to\\gum-12.1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stages.stage4_relation.label_mapping import (
    map_label,
    mapping_table_rows,
    MAPPING_VERSION,
    COARSE_LABELS,
)


def parse_rsd(path: Path) -> dict:
    edus = {}
    edges = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        edu_id = int(parts[0])
        text = parts[1]
        parent = int(parts[6]) if parts[6] not in ("_", "") else 0
        rel = parts[7]
        edus[edu_id] = {"edu_index": edu_id - 1, "edu_id_rsd": edu_id, "text": text}
        if rel and rel not in ("ROOT", "_") and parent > 0:
            edges.append({"child_rsd": edu_id, "parent_rsd": parent, "fine_label": rel})
    return {"edus": edus, "edges": edges}


def genre_from_doc_id(doc_id: str) -> str:
    parts = doc_id.split("_")
    return parts[1] if len(parts) >= 2 else "unknown"


def convert_all(rsd_dir: Path, out_jsonl: Path, out_manifest: Path, corpus_version: str):
    rows = []
    stats = {
        "corpus_version": corpus_version,
        "mapping_version": MAPPING_VERSION,
        "n_files": 0,
        "n_docs_ok": 0,
        "n_edus": 0,
        "n_edges_raw": 0,
        "n_examples": 0,
        "n_excluded": 0,
        "fine_counts": Counter(),
        "coarse_counts": Counter(),
        "exclude_reasons": Counter(),
        "genre_counts": Counter(),
        "docs": [],
    }
    files = sorted(rsd_dir.glob("*.rsd"))
    stats["n_files"] = len(files)
    for path in files:
        doc_id = path.stem
        parsed = parse_rsd(path)
        if not parsed["edus"]:
            continue
        stats["n_docs_ok"] += 1
        stats["n_edus"] += len(parsed["edus"])
        genre = genre_from_doc_id(doc_id)
        stats["genre_counts"][genre] += 1
        n_ex = 0
        for e in parsed["edges"]:
            stats["n_edges_raw"] += 1
            fine = e["fine_label"]
            stats["fine_counts"][fine] += 1
            coarse, reason = map_label(fine)
            if coarse is None:
                stats["n_excluded"] += 1
                stats["exclude_reasons"][reason] += 1
                continue
            child = parsed["edus"][e["child_rsd"]]
            parent = parsed["edus"].get(e["parent_rsd"])
            if parent is None:
                stats["n_excluded"] += 1
                stats["exclude_reasons"]["missing_parent"] += 1
                continue
            if child["edu_index"] < parent["edu_index"]:
                i, j = child, parent
            else:
                i, j = parent, child
            example = {
                "doc_id": doc_id,
                "genre": genre,
                "edu_i_index": i["edu_index"],
                "edu_j_index": j["edu_index"],
                "edu_i_text": i["text"],
                "edu_j_text": j["text"],
                "fine_label": fine,
                "coarse_label": coarse,
                "child_rsd": e["child_rsd"],
                "parent_rsd": e["parent_rsd"],
                "mapping_reason": reason,
                "corpus_version": corpus_version,
                "mapping_version": MAPPING_VERSION,
            }
            rows.append(example)
            stats["coarse_counts"][coarse] += 1
            n_ex += 1
        stats["docs"].append(
            {"doc_id": doc_id, "genre": genre, "n_edus": len(parsed["edus"]), "n_examples": n_ex}
        )
    stats["n_examples"] = len(rows)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "corpus_version": corpus_version,
        "mapping_version": MAPPING_VERSION,
        "coarse_labels": COARSE_LABELS,
        "mapping_table": mapping_table_rows(),
        "n_files": stats["n_files"],
        "n_docs_ok": stats["n_docs_ok"],
        "n_edus": stats["n_edus"],
        "n_edges_raw": stats["n_edges_raw"],
        "n_examples": stats["n_examples"],
        "n_excluded": stats["n_excluded"],
        "fine_counts": dict(stats["fine_counts"].most_common()),
        "coarse_counts": dict(stats["coarse_counts"]),
        "exclude_reasons": dict(stats["exclude_reasons"]),
        "genre_counts": dict(stats["genre_counts"]),
        "output_jsonl": str(out_jsonl),
    }
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gum-root", default=None)
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "gold"))
    args = ap.parse_args()
    gum_root = Path(args.gum_root) if args.gum_root else None
    if gum_root is None:
        ptr = ROOT / "data" / "external" / "GUM_PATH.txt"
        if ptr.exists():
            for line in ptr.read_text().splitlines():
                if line.startswith("GUM_ROOT="):
                    gum_root = Path(line.split("=", 1)[1].strip())
        if gum_root is None:
            gum_root = ROOT / "data" / "external" / "gum-12.1.0"
    rsd_dir = gum_root / "rst" / "dependencies"
    if not rsd_dir.is_dir():
        print(f"ERROR: RSD directory not found: {rsd_dir}")
        print("Acquire GUM V12.1.0 and point --gum-root at the extracted folder.")
        sys.exit(1)
    out_dir = Path(args.out_dir)
    out_jsonl = out_dir / "gum_v12_relations.jsonl"
    out_manifest = out_dir / "gum_v12_manifest.json"
    manifest = convert_all(rsd_dir, out_jsonl, out_manifest, "GUM-V12.1.0")
    print(json.dumps({k: manifest[k] for k in manifest if k != "mapping_table"}, indent=2))
    print(f"\nWrote {manifest['n_examples']} examples \u2192 {out_jsonl}")
    print(f"Manifest \u2192 {out_manifest}")


if __name__ == "__main__":
    main()
