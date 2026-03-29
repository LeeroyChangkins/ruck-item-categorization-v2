#!/usr/bin/env python3
"""
Step 1.5
Deduplicate and normalize matched/unmatched item lists from step 1.4 (or compatible JSON).

Matched rows: one normalized row per item id. If duplicates exist, keep the row with highest
precedence: manual_similar_title_1_6 > manual_bigram_1_3 > llm_1_4 > step-1.2 bigram.

Unmatched rows: one row per id; then drop any id that appears in the deduped matched list.

Inputs:
  - 1.4-llm_matched_*.json (matched_items)
  - 1.4-llm_unmatched_*.json (unmatched_items) — same run / timestamp as matched when using --pair-latest

Outputs (per run under step-1.5/outputs/<YYYYMMDD_HHMMSS>/):
  - matched_deduped.json
  - unmatched_deduped.json
  - matched_summary.json (master_categories t0, parent_categories t1, leaf_categories + counts)
  - unmatched_summary.json (stats + full list of unmatched item rows)

Normalized matched row shape:
  - id, title, subtitle
  - leaf_path, leaf_slug, leaf_display_name
  - method: one of t1_bigram | cascading_bigram | interactive_manual_bigram_match | llm_match

Bigram trigger details are intentionally removed from 1.5 outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]  # v2/
STEP14_OUT = ROOT / "step-3-llm-match-unmatched" / "outputs"
OUTDIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_TAXONOMY_PATH = ROOT / "source-files" / "categories_v1.json"

_SUMMARY_EMPTY = "(empty)"
_SUMMARY_UNMAPPED_MASTER = "_unmapped"

_FALLBACK_MASTER_SLUGS = ("materials", "tools_and_gear", "services")
_FALLBACK_MASTER_LABELS = {
    "materials": "Materials",
    "tools_and_gear": "Tools & Gear",
    "services": "Services",
}


def resolve_taxonomy_path(m_data: dict) -> Path:
    rel = m_data.get("taxonomy_categories_file")
    if isinstance(rel, str) and rel.strip():
        p = ROOT / "source-files" / Path(rel.strip()).name
        if p.is_file():
            return p
    return DEFAULT_TAXONOMY_PATH


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def match_row_tier(row: dict) -> int:
    """Higher wins on duplicate id. Step-1 similar-title manual is highest."""
    s = row.get("source")
    if s == "manual_similar_title_1_6":
        return 4
    if s == "manual_bigram_1_3":
        return 3
    if s == "llm_1_4":
        return 2
    return 1


def _leaf_slug_from_path(path: str) -> str:
    parts = str(path).strip("/").split("/")
    return parts[-1] if parts else ""


def normalize_matched_row(row: dict) -> dict:
    """
    Produce compact matched rows without bigram payloads.
    For step-1.2 rows, pick top-ranked category only (categories[0]).
    """
    iid = row.get("id") or ""
    title = row.get("title") or ""
    subtitle = row.get("subtitle") or ""
    source = row.get("source")

    if source == "manual_similar_title_1_6":
        leaf_path = row.get("leaf_path") or ""
        leaf_slug = row.get("leaf_slug") or _leaf_slug_from_path(leaf_path)
        return {
            "id": iid,
            "title": title,
            "subtitle": subtitle,
            "leaf_path": leaf_path,
            "leaf_slug": leaf_slug,
            "leaf_display_name": row.get("leaf_display_name") or "",
            "method": "interactive_similar_title_group",
        }

    if source == "manual_bigram_1_3":
        leaf_path = row.get("leaf_path") or ""
        leaf_slug = row.get("leaf_slug") or _leaf_slug_from_path(leaf_path)
        return {
            "id": iid,
            "title": title,
            "subtitle": subtitle,
            "leaf_path": leaf_path,
            "leaf_slug": leaf_slug,
            "leaf_display_name": row.get("leaf_display_name") or "",
            "method": "interactive_manual_bigram_match",
        }

    if source == "llm_1_4":
        leaf_path = row.get("leaf_path") or ""
        leaf_slug = row.get("leaf_slug") or _leaf_slug_from_path(leaf_path)
        return {
            "id": iid,
            "title": title,
            "subtitle": subtitle,
            "leaf_path": leaf_path,
            "leaf_slug": leaf_slug,
            "leaf_display_name": row.get("leaf_display_name") or "",
            "method": "llm_match",
        }

    # Step 1.2 shape: categories sorted by rank; keep top-ranked only.
    categories = row.get("categories") if isinstance(row.get("categories"), list) else []
    top = categories[0] if categories and isinstance(categories[0], dict) else {}
    leaf_path = str(top.get("category_slug") or "")
    leaf_slug = _leaf_slug_from_path(leaf_path)
    method = "t1_bigram" if "/" not in leaf_path else "cascading_bigram"
    return {
        "id": iid,
        "title": title,
        "subtitle": subtitle,
        "leaf_path": leaf_path,
        "leaf_slug": leaf_slug,
        "leaf_display_name": "",
        "method": method,
    }


def dedupe_matched(rows: List[Any]) -> Tuple[List[dict], int]:
    best: Dict[str, dict] = {}
    valid_in = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        iid = row.get("id")
        if not isinstance(iid, str) or not iid:
            continue
        valid_in += 1
        t = match_row_tier(row)
        if iid not in best or t > match_row_tier(best[iid]):
            best[iid] = row
    out = [normalize_matched_row(best[k]) for k in sorted(best.keys())]
    removed = valid_in - len(out)
    return out, removed


def dedupe_unmatched(rows: List[Any]) -> Tuple[List[dict], int]:
    seen: Set[str] = set()
    out: List[dict] = []
    valid_in = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        iid = row.get("id")
        if not isinstance(iid, str) or not iid:
            continue
        valid_in += 1
        if iid in seen:
            continue
        seen.add(iid)
        out.append(row)
    removed = valid_in - len(out)
    return out, removed


def strip_matched_from_unmatched(unmatched: List[dict], matched_ids: Set[str]) -> Tuple[List[dict], int]:
    out = [u for u in unmatched if u.get("id") not in matched_ids]
    removed = len(unmatched) - len(out)
    return out, removed


def _leaf_path_for_summary(row: dict) -> str:
    """Supports normalized 1.5 rows and legacy step-1.2-shaped rows (categories[0].category_slug)."""
    lp = row.get("leaf_path")
    if isinstance(lp, str) and lp.strip():
        return lp.strip()
    cats = row.get("categories")
    if isinstance(cats, list) and cats and isinstance(cats[0], dict):
        slug = cats[0].get("category_slug")
        if isinstance(slug, str) and slug:
            return slug
    return ""


def load_taxonomy_labels(categories_path: Path) -> Tuple[Dict[str, str], Dict[str, str], frozenset[str]]:
    """
    From categories_v1.json: master (t0) slugs + display names; parent (t1) = master/first-child path + display.
    Returns (master_display, parent_display, master_slugs).
    """
    master_display: Dict[str, str] = {}
    parent_display: Dict[str, str] = {}
    if not categories_path.is_file():
        return master_display, parent_display, frozenset()
    data = json.loads(categories_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return master_display, parent_display, frozenset()
    for master_slug, node in data.items():
        if not isinstance(node, dict) or not isinstance(master_slug, str):
            continue
        master_display[master_slug] = str(node.get("display_name") or master_slug)
        for sub in node.get("subcategories") or []:
            if not isinstance(sub, dict):
                continue
            pslug = sub.get("slug")
            if not isinstance(pslug, str) or not pslug:
                continue
            parent_path = f"{master_slug}/{pslug}"
            parent_display[parent_path] = str(sub.get("display_name") or pslug)
    return master_display, parent_display, frozenset(master_display.keys())


def summarize_matched_items(
    matched_out: List[dict],
    run_id: str,
    categories_path: Path | None = None,
) -> dict:
    """
    t0 master = top-level taxonomy bucket (materials | tools_and_gear | services).
    t1 parent = direct child of master (first segment under master in leaf_path).
    Leaf = full assigned path (same as row leaf_path), including master-only and parent-only paths.
    """
    tax_path = categories_path if categories_path is not None else DEFAULT_TAXONOMY_PATH
    master_labels, parent_labels, known_masters = load_taxonomy_labels(tax_path)
    if not known_masters:
        known_masters = frozenset(_FALLBACK_MASTER_SLUGS)
    for slug, label in _FALLBACK_MASTER_LABELS.items():
        master_labels.setdefault(slug, label)

    unmapped_display = "Unmapped (path does not start with a master category)"
    empty_display = "(empty path)"

    master_counts: Counter[str] = Counter()
    parent_counts: Counter[str] = Counter()
    leaf_counts: Counter[str] = Counter()

    for row in matched_out:
        lp = _leaf_path_for_summary(row) if isinstance(row, dict) else ""
        leaf_key = lp if lp else _SUMMARY_EMPTY
        leaf_counts[leaf_key] += 1
        parts = lp.split("/") if lp else []
        if not parts:
            master_counts[_SUMMARY_EMPTY] += 1
            continue
        m0 = parts[0]
        if m0 in known_masters:
            master_counts[m0] += 1
            if len(parts) >= 2:
                parent_counts[f"{m0}/{parts[1]}"] += 1
        else:
            master_counts[_SUMMARY_UNMAPPED_MASTER] += 1

    master_categories: List[dict] = []
    for slug, cnt in master_counts.most_common():
        if slug == _SUMMARY_EMPTY:
            disp = empty_display
        elif slug == _SUMMARY_UNMAPPED_MASTER:
            disp = unmapped_display
        else:
            disp = master_labels.get(slug, slug)
        master_categories.append(
            {"master_slug": slug, "display_name": disp, "count": cnt}
        )

    parent_categories: List[dict] = []
    for parent_path, cnt in parent_counts.most_common():
        segs = parent_path.split("/")
        ms = segs[0] if len(segs) > 0 else ""
        ps = segs[1] if len(segs) > 1 else ""
        parent_categories.append(
            {
                "master_slug": ms,
                "parent_slug": ps,
                "parent_path": parent_path,
                "display_name": parent_labels.get(parent_path, ps),
                "count": cnt,
            }
        )

    leaf_categories = [{"leaf_path": k, "count": v} for k, v in leaf_counts.most_common()]

    return {
        "run_id": run_id,
        "total_matched_items": len(matched_out),
        "unique_leaf_paths": len(leaf_counts),
        "taxonomy_file": str(tax_path.resolve()),
        "master_categories": master_categories,
        "parent_categories": parent_categories,
        "leaf_categories": leaf_categories,
    }


def summarize_unmatched_items(unmatched_out: List[dict], run_id: str, stats: dict) -> dict:
    """Stats plus full list of deduped unmatched rows (same shape as unmatched_deduped.json items)."""
    return {
        "run_id": run_id,
        "total_unmatched_items": len(unmatched_out),
        "stats": stats,
        "unmatched_items": unmatched_out,
    }


def find_latest_pair() -> Tuple[Path | None, Path | None]:
    """Latest 1.4 matched file and unmatched with same timestamp suffix if present."""
    matched_files = list(STEP14_OUT.glob("1.4-llm_matched_*.json"))
    if not matched_files:
        return None, None
    m_path = max(matched_files, key=lambda p: p.stat().st_mtime)
    m = re.match(r"^1\.4-llm_matched_(.+)\.json$", m_path.name)
    if not m:
        return m_path, None
    ts = m.group(1)
    u_path = STEP14_OUT / f"1.4-llm_unmatched_{ts}.json"
    if u_path.is_file():
        return m_path, u_path
    u_files = list(STEP14_OUT.glob("1.4-llm_unmatched_*.json"))
    u_fallback = max(u_files, key=lambda p: p.stat().st_mtime) if u_files else None
    return m_path, u_fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1.5: dedupe 1.4 matched/unmatched outputs.")
    parser.add_argument("--matched", metavar="PATH", help="Path to 1.4-llm_matched_*.json")
    parser.add_argument("--unmatched", metavar="PATH", help="Path to 1.4-llm_unmatched_*.json")
    parser.add_argument(
        "--pair-latest",
        action="store_true",
        help="Use newest 1.4 matched file and pair unmatched by timestamp (or latest unmatched).",
    )
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    m_path: Path | None = None
    u_path: Path | None = None

    if args.pair_latest:
        m_path, u_path = find_latest_pair()
        if not m_path:
            raise SystemExit(f"No 1.4-llm_matched_*.json under {STEP14_OUT}")
        print(f"Using matched: {m_path.relative_to(ROOT)}")
        print(f"Using unmatched: {u_path.relative_to(ROOT) if u_path else '(none)'}")
    elif args.matched or args.unmatched:
        if args.matched:
            m_path = Path(args.matched).expanduser().resolve()
        if args.unmatched:
            u_path = Path(args.unmatched).expanduser().resolve()
    else:
        m_path, u_path = find_latest_pair()
        if not m_path:
            raise SystemExit(f"No 1.4-llm_matched_*.json under {STEP14_OUT}")
        print(f"Using matched: {m_path.relative_to(ROOT)}")
        print(f"Using unmatched: {u_path.relative_to(ROOT) if u_path else '(none)'}")

    if not m_path or not m_path.exists():
        raise SystemExit("Need a valid --matched file, or place 1.4 outputs under step-1.4/outputs/.")

    m_data = json.loads(m_path.read_text(encoding="utf-8"))
    raw_matched = m_data.get("matched_items")
    if not isinstance(raw_matched, list):
        raise SystemExit("matched JSON must contain matched_items array.")

    raw_unmatched: List[Any] = []
    if u_path and u_path.exists():
        u_data = json.loads(u_path.read_text(encoding="utf-8"))
        ru = u_data.get("unmatched_items") if isinstance(u_data, dict) else None
        if isinstance(ru, list):
            raw_unmatched = ru
    elif args.unmatched:
        raise SystemExit(f"Unmatched file not found: {u_path}")

    matched_out, dup_m = dedupe_matched(raw_matched)
    matched_ids = {r["id"] for r in matched_out if isinstance(r.get("id"), str)}

    unmatched_out, dup_u = dedupe_unmatched(raw_unmatched)
    unmatched_out, stray = strip_matched_from_unmatched(unmatched_out, matched_ids)

    ts = timestamp()
    run_dir = OUTDIR / ts
    run_dir.mkdir(parents=True, exist_ok=False)

    base_meta = {
        "version": "1.5-deduped",
        "run_id": ts,
        "output_dir": str(run_dir.resolve()),
        "step_1_4_matched_input": str(m_path.resolve()),
        "step_1_4_unmatched_input": str(u_path.resolve()) if u_path else None,
        "dedupe_rule": (
            "matched: one row per id; on conflict prefer source manual_bigram_1_3 > llm_1_4 > bigram. "
            "unmatched: one row per id; remove ids present in deduped matched."
        ),
        "matched_row_shape": (
            "id,title,subtitle,leaf_path,leaf_slug,leaf_display_name,method. "
            "For step-1.2 rows, top-ranked category only."
        ),
        "stats": {
            "matched_rows_in": len(raw_matched),
            "matched_rows_out": len(matched_out),
            "matched_duplicate_rows_removed": dup_m,
            "unmatched_rows_in": len(raw_unmatched),
            "unmatched_rows_out": len(unmatched_out),
            "unmatched_duplicate_rows_removed": dup_u,
            "unmatched_removed_as_already_matched": stray,
        },
    }

    # Carry forward useful 1.4 metadata
    for k in (
        "taxonomy_categories_file",
        "unmatched_keywords_source",
        "matched_json_source",
        "manual_13_source",
        "llm_model",
        "min_confidence",
    ):
        if k in m_data and k not in base_meta:
            base_meta[k] = m_data[k]

    out_m = run_dir / "matched_deduped.json"
    out_u = run_dir / "unmatched_deduped.json"
    out_matched_summary = run_dir / "matched_summary.json"
    out_unmatched_summary = run_dir / "unmatched_summary.json"

    payload_m = {**base_meta, "matched_items": matched_out}
    payload_u = {**base_meta, "unmatched_items": unmatched_out}

    matched_summary_payload = summarize_matched_items(matched_out, ts, resolve_taxonomy_path(m_data))
    unmatched_summary_payload = summarize_unmatched_items(unmatched_out, ts, base_meta["stats"])

    out_m.write_text(json.dumps(payload_m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_u.write_text(json.dumps(payload_u, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_matched_summary.write_text(
        json.dumps(matched_summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    out_unmatched_summary.write_text(
        json.dumps(unmatched_summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\nWrote {out_m.relative_to(ROOT)}")
    print(f"Wrote {out_u.relative_to(ROOT)}")
    print(f"Wrote {out_matched_summary.relative_to(ROOT)}")
    print(f"Wrote {out_unmatched_summary.relative_to(ROOT)}")
    print(
        f"Matched: {len(raw_matched)} -> {len(matched_out)} "
        f"({dup_m} duplicate rows dropped). "
        f"Unmatched: {len(raw_unmatched)} -> {len(unmatched_out)} "
        f"({dup_u} dup rows, {stray} removed as matched)."
    )


if __name__ == "__main__":
    main()
