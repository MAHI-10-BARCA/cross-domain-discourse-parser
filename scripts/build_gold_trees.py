#!/usr/bin/env python3
"""
Build gold discourse trees from GUM .rsd dependency files.

Usage:
  python scripts\\build_gold_trees.py --gum-root E:\\Final_year\\gum-12.1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stages.stage4_relation.label_mapping import map_label, MAPPING_VERSION


def parse_rsd(path: Path) -> dict:
    edus = {}
    edges = []
    root_id = None
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
        edus[edu_id] = {"rsd_id": edu_id, "index": edu_id - 1, "text": text}
        if rel == "ROOT" or parent == 0:
            root_id = edu_id
        elif rel and rel not in ("_",) and parent > 0:
            coarse, reason = map_label(rel)
            if coarse is None:
                continue
            edges.append({
                "child_rsd": edu_id, "parent_rsd": parent,
                "child_index": edu_id - 1, "parent_index": parent - 1,
                "fine_label": rel, "coarse_label": coarse, "mapping_reason": reason,
            })
    return {
        "edus": [edus[k] for k in sorted(edus)],
        "edges": edges, "root_rsd": root_id,
        "n_edus": len(edus), "n_edges": len(edges),
    }


def genre_from_doc_id(doc_id: str) -> str:
    parts = doc_id.split("_")
    return parts[1] if len(parts) >= 2 else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gum-root", required=True)
    ap.add_argument("--out", default=str(ROOT / "data" / "gold" / "gum_v12_trees.jsonl"))
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--corpus-version", default="GUM-V12.1.0")
    args = ap.parse_args()

    gum = Path(args.gum_root)
    rsd_dir = gum / "rst" / "dependencies"
    if not rsd_dir.is_dir():
        if (gum / "dependencies").is_dir():
            rsd_dir = gum / "dependencies"
        elif list(gum.glob("*.rsd")):
            rsd_dir = gum
        else:
            print(f"ERROR: no rst/dependencies under {gum}")
            sys.exit(1)

    files = sorted(rsd_dir.glob("*.rsd"))
    if args.max_docs:
        files = files[: args.max_docs]
    print(f"Found {len(files)} .rsd files in {rsd_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_edges = 0
    fine_c, coarse_c = Counter(), Counter()
    with out_path.open("w", encoding="utf-8") as fout:
        for path in files:
            doc_id = path.stem
            parsed = parse_rsd(path)
            if parsed["n_edus"] == 0:
                continue
            for e in parsed["edges"]:
                fine_c[e["fine_label"]] += 1
                coarse_c[e["coarse_label"]] += 1
            n_edges += parsed["n_edges"]
            rec = {
                "doc_id": doc_id, "genre": genre_from_doc_id(doc_id),
                "corpus_version": args.corpus_version,
                "mapping_version": MAPPING_VERSION,
                "n_edus": parsed["n_edus"], "n_edges": parsed["n_edges"],
                "root_rsd": parsed["root_rsd"],
                "edus": parsed["edus"], "edges": parsed["edges"],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    man = {
        "corpus_version": args.corpus_version,
        "mapping_version": MAPPING_VERSION,
        "n_docs": len(files), "n_edges": n_edges,
        "fine_label_counts": dict(fine_c.most_common()),
        "coarse_label_counts": dict(coarse_c.most_common()),
        "out": str(out_path),
        "note": "Gold RST dependency trees from GUM .rsd.",
    }
    man_path = out_path.with_name("gum_v12_trees_manifest.json")
    man_path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"Wrote {len(files)} gold trees → {out_path}")
    print(f"Total gold edges: {n_edges}")
    print(f"Manifest → {man_path}")


if __name__ == "__main__":
    main()
