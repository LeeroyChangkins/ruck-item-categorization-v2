#!/usr/bin/env python3
"""
6_validate.py

Validate v2 categorization artifacts before DB import.

Checks:
  - All leaf_path values in matched_deduped.json exist in the taxonomy
  - All leaf_path values point to actual leaf nodes (no children)
  - No duplicate item IDs in matched_deduped.json
  - All item rows have a non-empty id and leaf_path

Can be run standalone or called as a module from 6_upload_to_db.py.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # v2/ root
TAXONOMY_PATH = ROOT / "source-files" / "categories_v1.json"
ROOT_CATEGORY_KEYS = ("materials", "tools_and_gear", "services")


# ──────────────────────────────────────────────────────────────────────────────
# Taxonomy path extraction
# ──────────────────────────────────────────────────────────────────────────────


def load_taxonomy_paths(taxonomy_path: Path | None = None) -> tuple[set[str], set[str]]:
    """
    Walk categories_v1.json and return (all_paths, leaf_paths).
    leaf_paths = nodes with no subcategories (or empty subcategories list).
    """
    path = taxonomy_path or TAXONOMY_PATH
    data = json.loads(path.read_text(encoding="utf-8"))

    all_paths: set[str] = set()
    leaf_paths: set[str] = set()

    def walk(slug_path: str, node: dict) -> None:
        all_paths.add(slug_path)
        children = [c for c in (node.get("subcategories") or []) if c.get("slug")]
        if not children:
            leaf_paths.add(slug_path)
        for child in children:
            walk(f"{slug_path}/{child['slug']}", child)

    for key in ROOT_CATEGORY_KEYS:
        root = data.get(key)
        if isinstance(root, dict):
            walk(key, root)

    return all_paths, leaf_paths


# ──────────────────────────────────────────────────────────────────────────────
# Matched items validation
# ──────────────────────────────────────────────────────────────────────────────


def validate_matched_deduped(
    matched_path: Path,
    all_paths: set[str],
    leaf_paths: set[str],
) -> list[str]:
    """
    Validate matched_deduped.json rows against the taxonomy.
    Returns a list of issue strings (empty list = clean).
    """
    data = json.loads(matched_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("matched_items") or []
    elif isinstance(data, list):
        rows = data
    else:
        return [f"matched_deduped.json has unexpected top-level type: {type(data).__name__}"]

    issues: list[str] = []
    seen_ids: set[str] = set()

    for i, row in enumerate(rows):
        item_id = row.get("id") or ""
        leaf_path = (row.get("leaf_path") or "").strip()

        if not item_id:
            issues.append(f"Row {i}: missing or empty 'id'")
            continue

        if not leaf_path:
            issues.append(f"Item {item_id}: missing or empty 'leaf_path'")
            continue

        if item_id in seen_ids:
            issues.append(f"Duplicate item ID: {item_id}")
        seen_ids.add(item_id)

        if leaf_path not in all_paths:
            issues.append(f"Item {item_id}: leaf_path not in taxonomy: {leaf_path}")
        elif leaf_path not in leaf_paths:
            issues.append(f"Item {item_id}: leaf_path is not a leaf node (has children): {leaf_path}")

    return issues


# ──────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ──────────────────────────────────────────────────────────────────────────────


def run_validation(
    matched_deduped_path: Path,
    taxonomy_path: Path | None = None,
) -> list[str]:
    """
    Run all validation checks and return a list of issue strings (empty = clean).
    """
    all_paths, leaf_paths = load_taxonomy_paths(taxonomy_path)
    issues: list[str] = []
    issues.extend(validate_matched_deduped(matched_deduped_path, all_paths, leaf_paths))
    return issues


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate v2 matched_deduped.json before DB import.")
    parser.add_argument(
        "matched_deduped",
        type=Path,
        nargs="?",
        help="Path to matched_deduped.json (defaults to latest step-4 output run)",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=None,
        help="Path to categories_v1.json (default: source-files/categories_v1.json)",
    )
    args = parser.parse_args()

    matched_path = args.matched_deduped
    if matched_path is None:
        step4_out = ROOT / "step-4" / "outputs"
        candidates = sorted(
            (p / "matched_deduped.json" for p in step4_out.iterdir() if p.is_dir()),
            key=lambda p: p.parent.name,
            reverse=True,
        )
        candidates = [p for p in candidates if p.is_file()]
        if not candidates:
            print(f"No matched_deduped.json found under {step4_out}")
            return 1
        matched_path = candidates[0]
        print(f"Using latest: {matched_path.relative_to(ROOT)}")

    issues = run_validation(matched_path, args.taxonomy)
    if issues:
        print(f"Validation failed ({len(issues)} issue(s)):")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
