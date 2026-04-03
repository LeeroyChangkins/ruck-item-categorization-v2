#!/usr/bin/env python3
"""
Step 5a — Title Template Grouping.

Normalises every item title into a structural template by replacing numeric
tokens (integers, decimals, fractions) with placeholders while preserving
surrounding text and punctuation.  Items that share the same template string
are grouped into a "cluster".  One representative title is chosen per cluster.

Clusters with fewer than --min-cluster-size items are merged into a single
_misc bucket per category so the LLM in step 5b doesn't waste tokens on
one-off titles.

Output
------
  step-5/outputs/<timestamp>/title_groups/_manifest.json
  step-5/outputs/<timestamp>/title_groups/<safe_leaf_path>.json  (one per category)

Per-category file schema:
  {
    "leaf_path":        "materials/metals_and_metal_fabrication/metal_tubing",
    "total_items":      806,
    "is_low_structure": false,      // true when >= low_structure_ratio land in misc
    "clusters": [
      {
        "template":          "{NUM} {FRAC}\" X {NUM}\" Carbon Steel Square Tube A500/A513",
        "item_count":        188,
        "representative":    "1 3/4\" X 0.120\" Carbon Steel Square Tube A500/A513",
        "representative_id": "itm_abc123",
        "item_ids":          ["itm_abc", ...]
      }
    ],
    "misc": {
      "item_count": 47,
      "item_ids":   ["itm_xyz", ...]
    }
  }

Usage
-----
  python 5a_group_title_templates.py [OPTIONS]

  --run-dir            PATH   step-4 outputs subfolder (default: latest)
  --out-dir            PATH   root output dir (default: step-5/outputs/<ts>/)
  --min-cluster-size   INT    min items to keep a named cluster (default: 3)
  --low-structure-ratio FLOAT misc fraction threshold for low_structure flag (default: 0.60)
  --dry-run                   Print summary only, do not write files
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STEP5_DIR = Path(__file__).resolve().parent
STEP4_OUTPUTS = ROOT / "step-4-dedupe-and-merge-matched-items" / "outputs"
TAXONOMY_PATH = ROOT / "source-files" / "categories_v1.json"

sys.path.insert(0, str(ROOT))
import shared_utils as _timestamp_module
from shared_utils import timestamp as _timestamp

DEFAULT_MIN_CLUSTER = 3
DEFAULT_LOW_STRUCTURE_RATIO = 0.60


# ── title normalisation ────────────────────────────────────────────────────────

_FRACTION_RE      = re.compile(r"\b\d+\s*/\s*\d+\b")
_DECIMAL_RE       = re.compile(r"\b\d+\.\d+\b")
_LEADING_DEC_RE   = re.compile(r"(?<![.\d])\.\d+\b")
_INTEGER_RE       = re.compile(r"\b\d+\b")


def normalise(title: str) -> str:
    """Replace all numeric tokens with placeholders to reveal structural pattern."""
    t = title
    t = _FRACTION_RE.sub("{FRAC}", t)
    t = _DECIMAL_RE.sub("{NUM}", t)
    t = _LEADING_DEC_RE.sub("{NUM}", t)
    t = _INTEGER_RE.sub("{NUM}", t)
    return re.sub(r"\s+", " ", t).strip()


# ── taxonomy helpers ───────────────────────────────────────────────────────────

def _collect_leaves(node: dict, parts: list[str]) -> list[tuple[str, str]]:
    subcats = node.get("subcategories", [])
    if not subcats:
        return [("/".join(parts), node.get("display_name", parts[-1]))]
    out: list[tuple[str, str]] = []
    for child in subcats:
        out.extend(_collect_leaves(child, parts + [child["slug"]]))
    return out


def load_leaves(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text())
    leaves: list[tuple[str, str]] = []
    for root_key, root_node in data.items():
        for child in root_node.get("subcategories", []):
            leaves.extend(_collect_leaves(child, [root_key, child["slug"]]))
    return leaves


# ── step-4 helpers ─────────────────────────────────────────────────────────────

def find_latest_run_dir(step4_outputs: Path) -> Path:
    dirs = [d for d in step4_outputs.iterdir() if d.is_dir()]
    if not dirs:
        sys.exit(f"No run directories found under {step4_outputs}")
    result = _timestamp_module.latest_env_path(dirs, name_attr="name")
    return result or dirs[0]


def load_matched_items(run_dir: Path) -> dict[str, list[dict]]:
    """Return leaf_path → list of item dicts."""
    candidates = list(run_dir.glob("matched_deduped*.json"))
    if not candidates:
        sys.exit(f"No matched_deduped*.json found in {run_dir}")
    data = json.loads(sorted(candidates)[-1].read_text())
    raw: list[dict] = data.get("matched_items") or data.get("items") or []
    grouped: dict[str, list[dict]] = {}
    for item in raw:
        leaf = item.get("leaf_path") or item.get("leaf_slug") or ""
        if leaf:
            grouped.setdefault(leaf, []).append(item)
    return grouped


# ── helpers ────────────────────────────────────────────────────────────────────

def safe_filename(leaf_path: str) -> str:
    return leaf_path.replace("/", "__") + ".json"


def _item_id(item: dict) -> str:
    return item.get("id") or item.get("item_id") or ""


# ── clustering ────────────────────────────────────────────────────────────────

def cluster_items(
    items: list[dict],
    min_cluster_size: int,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (clusters, misc_items).

    Clusters are sorted largest-first. Items whose template appears fewer than
    min_cluster_size times land in misc_items.
    """
    template_map: dict[str, list[dict]] = {}
    for item in items:
        tmpl = normalise(item.get("title", ""))
        template_map.setdefault(tmpl, []).append(item)

    clusters: list[dict] = []
    misc_items: list[dict] = []

    for tmpl, grp in sorted(template_map.items(), key=lambda x: -len(x[1])):
        if len(grp) < min_cluster_size:
            misc_items.extend(grp)
            continue
        # Pick representative whose title length is closest to the median
        lengths = sorted(len(it.get("title", "")) for it in grp)
        median_len = lengths[len(lengths) // 2]
        rep = min(grp, key=lambda it: abs(len(it.get("title", "")) - median_len))
        clusters.append({
            "template":          tmpl,
            "item_count":        len(grp),
            "representative":    rep.get("title", ""),
            "representative_id": _item_id(rep),
            "item_ids":          [_item_id(it) for it in grp],
        })

    return clusters, misc_items


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5a — title template grouping")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="step-4 outputs subfolder (default: latest)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Root output dir (default: step-5/outputs/<timestamp>/)")
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER,
                        help=f"Min items to keep a named cluster (default: {DEFAULT_MIN_CLUSTER})")
    parser.add_argument("--low-structure-ratio", type=float, default=DEFAULT_LOW_STRUCTURE_RATIO,
                        help=f"Misc-fraction threshold for low_structure flag (default: {DEFAULT_LOW_STRUCTURE_RATIO})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary only, do not write files")
    args = parser.parse_args()

    run_dir = args.run_dir or find_latest_run_dir(STEP4_OUTPUTS)
    print(f"Loading matched items from: {run_dir.name}")
    items_by_leaf = load_matched_items(run_dir)
    total_raw = sum(len(v) for v in items_by_leaf.values())
    print(f"  {total_raw} items across {len(items_by_leaf)} categories")

    print(f"Loading taxonomy from: {TAXONOMY_PATH.name}")
    leaves = load_leaves(TAXONOMY_PATH)
    print(f"  {len(leaves)} leaf categories")

    out_root: Path = args.out_dir or (STEP5_DIR / "outputs" / _timestamp())
    groups_dir = out_root / "title_groups"
    if not args.dry_run:
        groups_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []
    total_items_processed = 0
    low_structure_count = 0

    for leaf_path, _display_name in leaves:
        items = items_by_leaf.get(leaf_path, [])
        if not items:
            continue

        clusters, misc_items = cluster_items(items, args.min_cluster_size)
        total_count = len(items)
        misc_count = len(misc_items)
        misc_ratio = misc_count / total_count if total_count else 0.0
        is_low = misc_ratio >= args.low_structure_ratio
        if is_low:
            low_structure_count += 1

        fname = safe_filename(leaf_path)
        manifest_entries.append({
            "leaf_path":        leaf_path,
            "total_items":      total_count,
            "cluster_count":    len(clusters),
            "misc_item_count":  misc_count,
            "is_low_structure": is_low,
            "file":             fname,
        })
        total_items_processed += total_count

        if not args.dry_run:
            cat_data: dict[str, Any] = {
                "leaf_path":        leaf_path,
                "total_items":      total_count,
                "is_low_structure": is_low,
                "clusters":         clusters,
                "misc": {
                    "item_count": misc_count,
                    "item_ids":   [_item_id(it) for it in misc_items],
                },
            }
            out_file = groups_dir / fname
            out_file.write_text(
                json.dumps(cat_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    manifest: dict[str, Any] = {
        "run_dir":             str(run_dir),
        "total_categories":    len(manifest_entries),
        "total_items":         total_items_processed,
        "low_structure_count": low_structure_count,
        "categories":          manifest_entries,
    }

    if not args.dry_run:
        (groups_dir / "_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {len(manifest_entries)} category files → {groups_dir.relative_to(ROOT)}/")

    print(f"\nSummary:")
    print(f"  Categories with items : {len(manifest_entries)}")
    print(f"  Total items grouped   : {total_items_processed}")
    print(f"  Low-structure cats    : {low_structure_count}  "
          f"(>= {int(args.low_structure_ratio * 100)}% items in misc bucket)")

    if args.dry_run:
        print("\n[dry-run] No files written.")
    else:
        # Print machine-readable output_dir so the orchestrator can chain it
        print(f"\nOUTPUT_DIR={out_root}")


if __name__ == "__main__":
    main()
