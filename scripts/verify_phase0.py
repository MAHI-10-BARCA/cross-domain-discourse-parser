#!/usr/bin/env python3
"""Phase 0 verification script."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main():
    print("=" * 60)
    print("Phase 0 Verification")
    print("=" * 60)

    errors = []

    # 1. Directory structure
    required_dirs = [
        "config", "src", "src/utils", "src/stages",
        "tests", "data/raw", "data/processed", "data/gold",
        "models", "outputs/intermediate", "outputs/trees", "outputs/logs",
        "experiments", "docs", "scripts"
    ]
    print("\n[1] Checking directory structure...")
    for d in required_dirs:
        p = ROOT / d
        if p.is_dir():
            print(f"  ✓ {d}")
        else:
            print(f"  ✗ MISSING: {d}")
            errors.append(f"Missing dir: {d}")

    # 2. Key files
    required_files = [
        "README.md", "requirements.txt", "config/default.yaml",
        "src/data_schemas.py", "src/utils/seed.py",
        "src/utils/logging_utils.py", "src/utils/io.py",
        "docs/architecture.md", "docs/data_schemas.md",
        "docs/implementation_checklist.md",
        "experiments/experiment_log.md",
        "tests/test_phase0_schemas.py"
    ]
    print("\n[2] Checking key files...")
    for f in required_files:
        p = ROOT / f
        if p.is_file():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ MISSING: {f}")
            errors.append(f"Missing file: {f}")

    # 3. Config load
    print("\n[3] Loading config...")
    try:
        from utils.io import load_yaml
        cfg = load_yaml(ROOT / "config" / "default.yaml")
        assert "project" in cfg and "seed" in cfg["project"]
        print(f"  ✓ Config loaded (seed={cfg['project']['seed']})")
    except Exception as e:
        print(f"  ✗ Config load failed: {e}")
        errors.append(str(e))

    # 4. Schemas
    print("\n[4] Testing schemas...")
    try:
        from data_schemas import Document, EDU, EDUPair, DiscourseTree, DiscourseNode, save_json
        doc = Document(doc_id="test_doc", text="Sentence one. Sentence two.", domain="news")
        edu0 = EDU(edu_id="test_doc_edu_000", doc_id="test_doc", text="Sentence one.", index=0)
        edu1 = EDU(edu_id="test_doc_edu_001", doc_id="test_doc", text="Sentence two.", index=1)
        pair = EDUPair(
            pair_id="p0", doc_id="test_doc",
            edu_i_id=edu0.edu_id, edu_j_id=edu1.edu_id,
            edu_i_index=0, edu_j_index=1, distance=1, similarity=0.9
        )
        root = DiscourseNode(node_id="root", span_start=0, span_end=1, relation="Elaboration",
                             children=[
                                 DiscourseNode(node_id="n0", span_start=0, span_end=0, edu_ids=[edu0.edu_id]),
                                 DiscourseNode(node_id="n1", span_start=1, span_end=1, edu_ids=[edu1.edu_id]),
                             ])
        tree = DiscourseTree(doc_id="test_doc", root=root, edu_list=[edu0, edu1])

        out_path = ROOT / "outputs" / "intermediate" / "phase0_sample.json"
        save_json({
            "document": doc.to_dict(),
            "edus": [edu0.to_dict(), edu1.to_dict()],
            "pair": pair.to_dict(),
            "tree": tree.to_dict()
        }, out_path)
        print(f"  ✓ Schemas work. Sample written to {out_path}")
    except Exception as e:
        print(f"  ✗ Schema test failed: {e}")
        errors.append(str(e))

    # 5. Seed
    print("\n[5] Testing seed utility...")
    try:
        from utils.seed import set_seed
        set_seed(42)
        print("  ✓ set_seed(42) succeeded")
    except Exception as e:
        print(f"  ✗ Seed failed: {e}")
        errors.append(str(e))

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("Phase 0 PASSED — all checks successful.")
        print("You can now proceed to Phase 1 (EDU Segmentation).")
        sys.exit(0)


if __name__ == "__main__":
    main()
