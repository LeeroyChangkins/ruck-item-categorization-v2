#!/usr/bin/env python3
"""
Step 1.2 — Interactive similar-title groups → taxonomy leaves

Assign leaves to groups built by step-1-similar-title-groups/1_1_build_similar_title_groups.py.

UX mirrors step-2.3 (interactive keyword match): search leaves by substring, pick by number, save after each decision.
Per group: [c] copies master_title; [s] copy + Google Images in Chrome (macOS). [o] lists catch-all
Other leaves; [x] marks unknown without picking Other.

Outputs (next to the chosen groups JSON, or --resume-from):
  1.6-manual_similar_title_<timestamp>.json
  unmatched_after_step1.json (pool for step 2)

Fields: group_assignments, unknown_groups, item_matches (source=manual_similar_title_1_6).
matched_cumulative / unmatched_and_skipped_cumulative: from full catalog (source_items_catalog) or
legacy step-1.5 pools when groups file has source_unmatched_deduped.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set
from urllib.parse import quote_plus

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taxonomy_cascade import is_catch_all_bucket_slug
from pipeline_paths import glob_step1_outputs
from shared_utils import timestamp, write_step_summary, env_suffix
from interactive_helpers import (
    copy_to_clipboard,
    open_google_images_in_chrome,
    collect_leaf_rows,
    filter_leaves,
    filter_hits_narrow,
    collect_other_bucket_leaves,
    interact_pick_other_leaf,
    interact_insert_new_category,
    yn_prompt,
)

TAXONOMY_PATH = ROOT / "source-files" / "categories_v1.json"

MAX_LEAVES_TO_SHOW = 100
PREVIEW_ITEMS_MAX = 12
MANUAL_VERSION = "1.6-manual-similar-title"

# Filled once per main() from groups file (catalog or legacy 1.5 JSONs).
_step15_dedup_cache: dict | None = None








def assign_group_to_leaf(
    *,
    gid: str,
    master_title: str,
    items_in_group: List[Any],
    chosen: Dict[str, str],
    out_path: Path,
    groups_path: Path,
    group_assignments: List[dict],
    unknown_groups: List[dict],
    item_matches: List[dict],
    done_g: Set[str],
    already_matched_ids: Set[str],
    assignment_extra: dict | None = None,
) -> int:
    """Apply one leaf to the whole group; save snapshot. Returns number of items assigned."""
    leaf_path = chosen["leaf_path"]
    leaf_slug = chosen["leaf_slug"]
    display_name = chosen.get("display_name") or ""

    ids = [
        it["id"]
        for it in items_in_group
        if isinstance(it, dict) and isinstance(it.get("id"), str)
    ]

    row: dict = {
        "group_id": gid,
        "master_title": master_title,
        "leaf_path": leaf_path,
        "leaf_slug": leaf_slug,
        "leaf_display_name": display_name,
        "item_count": len(ids),
        "item_ids": ids,
    }
    if assignment_extra:
        row.update(assignment_extra)
    group_assignments.append(row)
    done_g.add(gid)

    for it in items_in_group:
        if not isinstance(it, dict):
            continue
        iid = it.get("id")
        if not isinstance(iid, str) or not iid:
            continue
        item_matches.append(
            {
                "id": iid,
                "title": it.get("title") or "",
                "subtitle": it.get("subtitle") or "",
                "matched_via_group_id": gid,
                "leaf_path": leaf_path,
                "leaf_slug": leaf_slug,
                "leaf_display_name": display_name,
                "source": "manual_similar_title_1_6",
            }
        )
        already_matched_ids.add(iid)

    write_manual_snapshot(out_path, groups_path, group_assignments, unknown_groups, item_matches)
    return len(ids)


def item_ids_from_group_items(items_in_group: List[Any]) -> List[str]:
    out: List[str] = []
    for it in items_in_group:
        if isinstance(it, dict) and isinstance(it.get("id"), str) and it["id"]:
            out.append(it["id"])
    return out


def undo_group_decision(
    gid: str,
    group_assignments: List[dict],
    unknown_groups: List[dict],
    item_matches: List[dict],
    done_g: Set[str],
    already_matched_ids: Set[str],
    item_ids_in_group: List[str],
) -> None:
    """Remove saved decision for gid so the group can be recategorized."""
    group_assignments[:] = [x for x in group_assignments if x.get("group_id") != gid]
    unknown_groups[:] = [x for x in unknown_groups if x.get("group_id") != gid]
    item_matches[:] = [x for x in item_matches if x.get("matched_via_group_id") != gid]
    done_g.discard(gid)
    for iid in item_ids_in_group:
        already_matched_ids.discard(iid)


def find_latest_groups_file() -> Path | None:
    """Newest unmatched_similar_title_groups.json under step-1-similar-title-groups/outputs/."""
    cands = glob_step1_outputs("**/unmatched_similar_title_groups.json")
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def find_latest_manual_for_groups_source(groups_src: Path) -> Path | None:
    """
    Newest 1.6 manual JSON whose groups_source matches groups_src under step-1-similar-title-groups/outputs/ (resume).
    """
    gs = groups_src.resolve()
    best: tuple[float, Path] | None = None
    for p in glob_step1_outputs("**/1.6-manual_similar_title*.json"):
        if not p.is_file():
            continue
        try:
            prev = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        src = prev.get("groups_source")
        if not isinstance(src, str):
            continue
        try:
            if Path(src).resolve() != gs:
                continue
        except OSError:
            continue
        mtime = p.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, p)
    return best[1] if best else None


def load_resume(
    resume_p: Path,
    groups_path: Path,
    force_mismatch: bool,
) -> tuple[List[dict], List[dict], List[dict], Set[str], Set[str]]:
    prev = json.loads(resume_p.read_text(encoding="utf-8"))
    prev_src = prev.get("groups_source")
    if isinstance(prev_src, str) and Path(prev_src).resolve() != groups_path.resolve():
        if not force_mismatch:
            raise SystemExit(
                f"Resume file groups_source does not match --groups:\n"
                f"  resume: {prev_src}\n"
                f"  current: {groups_path}"
            )
        print("Warning: --force-mismatch: groups_source differs.", flush=True)

    ga = [x for x in (prev.get("group_assignments") or []) if isinstance(x, dict)]
    ug = [x for x in (prev.get("unknown_groups") or []) if isinstance(x, dict)]
    im = [x for x in (prev.get("item_matches") or []) if isinstance(x, dict)]
    done_g: Set[str] = set()
    for x in ga:
        gid = x.get("group_id")
        if isinstance(gid, str) and gid:
            done_g.add(gid)
    for x in ug:
        gid = x.get("group_id")
        if isinstance(gid, str) and gid:
            done_g.add(gid)
    matched_ids = {m["id"] for m in im if isinstance(m.get("id"), str) and m["id"]}
    return ga, ug, im, done_g, matched_ids


def init_step15_dedup_cache_from_groups(groups_path: Path) -> None:
    """Load item pool from groups file: source_items_catalog (full raw) or legacy step-1.5 unmatched_deduped."""
    global _step15_dedup_cache
    unmatched_src: str | None = None
    matched_src: str | None = None
    matched_prior: List[dict] = []
    unmatched_pool: List[dict] = []
    try:
        gd = json.loads(groups_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _step15_dedup_cache = {
            "unmatched_src": None,
            "matched_src": None,
            "matched_prior": [],
            "unmatched_pool": [],
        }
        return

    cat = gd.get("source_items_catalog")
    if isinstance(cat, str) and cat.strip():
        unmatched_src = str(Path(cat).expanduser().resolve())
        try:
            raw = json.loads(Path(unmatched_src).read_text(encoding="utf-8"))
            if isinstance(raw, list):
                unmatched_pool = [x for x in raw if isinstance(x, dict)]
            elif isinstance(raw, dict):
                ui = raw.get("unmatched_items") or raw.get("items")
                if isinstance(ui, list):
                    unmatched_pool = [x for x in ui if isinstance(x, dict)]
        except (OSError, json.JSONDecodeError):
            pass
        _step15_dedup_cache = {
            "unmatched_src": unmatched_src,
            "matched_src": None,
            "matched_prior": [],
            "unmatched_pool": unmatched_pool,
        }
        return

    u = gd.get("source_unmatched_deduped")
    if isinstance(u, str) and u.strip():
        unmatched_src = str(Path(u).expanduser().resolve())

    if unmatched_src:
        mp = Path(unmatched_src).parent / "matched_deduped.json"
        if mp.is_file():
            matched_src = str(mp.resolve())
    if matched_src:
        try:
            md = json.loads(Path(matched_src).read_text(encoding="utf-8"))
            mi = md.get("matched_items")
            if isinstance(mi, list):
                matched_prior = [x for x in mi if isinstance(x, dict)]
        except (OSError, json.JSONDecodeError):
            pass
    if unmatched_src:
        try:
            ud = json.loads(Path(unmatched_src).read_text(encoding="utf-8"))
            ui = ud.get("unmatched_items")
            if isinstance(ui, list):
                unmatched_pool = [x for x in ui if isinstance(x, dict)]
        except (OSError, json.JSONDecodeError):
            pass

    _step15_dedup_cache = {
        "unmatched_src": unmatched_src,
        "matched_src": matched_src,
        "matched_prior": matched_prior,
        "unmatched_pool": unmatched_pool,
    }

def clear_step15_dedup_cache() -> None:
    global _step15_dedup_cache
    _step15_dedup_cache = None


def build_id_to_similar_title_group_id(groups_path: Path) -> Dict[str, str]:
    """item id -> group_id for items that appear in 1.6.1 clique groups."""
    out: Dict[str, str] = {}
    try:
        data = json.loads(groups_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    for g in data.get("groups") or []:
        if not isinstance(g, dict):
            continue
        gid = g.get("group_id")
        if not isinstance(gid, str) or not gid:
            continue
        for it in g.get("items") or []:
            if not isinstance(it, dict):
                continue
            iid = it.get("id")
            if isinstance(iid, str) and iid:
                out[iid] = gid
    return out


def normalize_16_item_to_match_row(m: dict) -> dict:
    """Align 1.6 item_matches row with step-1.5 matched_items shape."""
    return {
        "id": m["id"],
        "title": m.get("title") or "",
        "subtitle": m.get("subtitle") or "",
        "leaf_path": m.get("leaf_path") or "",
        "leaf_slug": m.get("leaf_slug") or "",
        "leaf_display_name": m.get("leaf_display_name") or "",
        "method": "similar_title_group_1_6",
        "source": "manual_similar_title_1_6",
    }


def compute_cumulative_matched_and_remaining(
    groups_path: Path,
    group_assignments: List[dict],
    unknown_groups: List[dict],
    item_matches: List[dict],
) -> dict:
    """
    Full matched list = step-1.5 matched_deduped + 1.6 assignments (by id).
    Remaining = unmatched_deduped pool minus ids assigned in 1.6, with status:
      not_in_similar_title_group | not_decided_in_1_6 | unknown_1_6
    """
    if _step15_dedup_cache is not None:
        unmatched_src = _step15_dedup_cache.get("unmatched_src")
        matched_src = _step15_dedup_cache.get("matched_src")
        matched_prior = list(_step15_dedup_cache.get("matched_prior") or [])
        unmatched_pool = list(_step15_dedup_cache.get("unmatched_pool") or [])
    else:
        unmatched_src = None
        matched_src = None
        matched_prior = []
        unmatched_pool = []
        try:
            gd = json.loads(groups_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            gd = {}
        cat = gd.get("source_items_catalog")
        if isinstance(cat, str) and cat.strip():
            unmatched_src = str(Path(cat).expanduser().resolve())
            try:
                raw = json.loads(Path(unmatched_src).read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    unmatched_pool = [x for x in raw if isinstance(x, dict)]
                elif isinstance(raw, dict):
                    ui = raw.get("unmatched_items") or raw.get("items")
                    if isinstance(ui, list):
                        unmatched_pool = [x for x in ui if isinstance(x, dict)]
            except (OSError, json.JSONDecodeError):
                pass
        else:
            u = gd.get("source_unmatched_deduped")
            if isinstance(u, str) and u.strip():
                unmatched_src = str(Path(u).expanduser().resolve())

            if unmatched_src:
                mp = Path(unmatched_src).parent / "matched_deduped.json"
                if mp.is_file():
                    matched_src = str(mp.resolve())

            if matched_src:
                try:
                    md = json.loads(Path(matched_src).read_text(encoding="utf-8"))
                    mi = md.get("matched_items")
                    if isinstance(mi, list):
                        matched_prior = [x for x in mi if isinstance(x, dict)]
                except (OSError, json.JSONDecodeError):
                    pass
            if unmatched_src:
                try:
                    ud = json.loads(Path(unmatched_src).read_text(encoding="utf-8"))
                    ui = ud.get("unmatched_items")
                    if isinstance(ui, list):
                        unmatched_pool = [x for x in ui if isinstance(x, dict)]
                except (OSError, json.JSONDecodeError):
                    pass

    id_to_gid = build_id_to_similar_title_group_id(groups_path)
    unknown_gids = {x.get("group_id") for x in unknown_groups if isinstance(x.get("group_id"), str)}
    assigned_gids = {x.get("group_id") for x in group_assignments if isinstance(x.get("group_id"), str)}
    done_g = unknown_gids | assigned_gids

    ids_16 = {m["id"] for m in item_matches if isinstance(m.get("id"), str) and m["id"]}
    rows_16 = [
        normalize_16_item_to_match_row(m)
        for m in item_matches
        if isinstance(m.get("id"), str) and m["id"]
    ]

    by_id: Dict[str, dict] = {}
    for r in matched_prior:
        iid = r.get("id")
        if isinstance(iid, str) and iid:
            by_id[iid] = r
    for r in rows_16:
        by_id[r["id"]] = r
    matched_cumulative = sorted(by_id.values(), key=lambda r: r.get("id") or "")

    remaining: List[dict] = []
    for it in unmatched_pool:
        iid = it.get("id")
        if not isinstance(iid, str) or not iid:
            continue
        if iid in ids_16:
            continue
        gid = id_to_gid.get(iid)
        if gid and gid in unknown_gids:
            status = "unknown_1_6"
        elif gid and gid not in done_g:
            status = "not_decided_in_1_6"
        elif gid and gid in assigned_gids:
            status = "in_assigned_group_but_not_in_item_matches"
        else:
            status = "not_in_similar_title_group"
        row: dict = {
            "id": iid,
            "title": it.get("title") or "",
            "subtitle": it.get("subtitle") or "",
            "status": status,
        }
        if gid:
            row["similar_title_group_id"] = gid
        remaining.append(row)

    return {
        "unmatched_deduped_source": unmatched_src,
        "matched_deduped_source": matched_src,
        "matched_cumulative": matched_cumulative,
        "unmatched_and_skipped_cumulative": remaining,
        "counts": {
            "matched_rows_from_1_5_deduped": len(matched_prior),
            "matched_rows_from_1_6": len(rows_16),
            "matched_cumulative_unique_ids": len(matched_cumulative),
            "unmatched_pool_rows_from_1_5": len(unmatched_pool),
            "remaining_unmatched_or_skipped_after_1_6": len(remaining),
        },
    }


def write_manual_snapshot(
    out_path: Path,
    groups_path: Path,
    group_assignments: List[dict],
    unknown_groups: List[dict],
    item_matches: List[dict],
) -> None:
    cumulative = compute_cumulative_matched_and_remaining(
        groups_path, group_assignments, unknown_groups, item_matches
    )
    payload = {
        "version": MANUAL_VERSION,
        "taxonomy_categories_file": TAXONOMY_PATH.name,
        "groups_source": str(groups_path.resolve()),
        "assignment_rule": (
            "Each assignment applies to all items in the group. item_matches lists one row per item."
        ),
        "group_assignments": group_assignments,
        "unknown_groups": unknown_groups,
        "item_matches": item_matches,
        "matched_cumulative": cumulative["matched_cumulative"],
        "unmatched_and_skipped_cumulative": cumulative["unmatched_and_skipped_cumulative"],
        "cumulative_sources": {
            "unmatched_deduped": cumulative["unmatched_deduped_source"],
            "matched_deduped": cumulative["matched_deduped_source"],
        },
        "cumulative_counts": cumulative["counts"],
        "last_saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out_path)
    suf = env_suffix()
    ua_path = out_path.parent / f"unmatched_after_step1{suf}.json"
    ua_items = [
        {"id": r["id"], "title": r.get("title") or "", "subtitle": r.get("subtitle") or ""}
        for r in cumulative.get("unmatched_and_skipped_cumulative") or []
        if isinstance(r.get("id"), str) and r["id"]
    ]
    ua_payload = {
        "version": "unmatched-after-step1",
        "manual_session": str(out_path.resolve()),
        "groups_source": str(groups_path.resolve()),
        "item_count": len(ua_items),
        "items": ua_items,
    }
    ua_path.write_text(json.dumps(ua_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def master_title_clipboard_actions(master_title: str, which: str) -> None:
    """which: 'c' = copy only, 's' = copy + open Google Images in Chrome."""
    w = which.lower()
    if w == "c":
        if copy_to_clipboard(master_title):
            print("  Copied master_title to clipboard.")
        else:
            print("  Clipboard not available (macOS pbcopy).")
        return
    if w == "s":
        if copy_to_clipboard(master_title):
            print("  Copied master_title to clipboard.")
        else:
            print("  (Clipboard copy failed.)")
        if open_google_images_in_chrome(master_title):
            print("  Opened Google Images in Chrome.")
        else:
            print(
                "  Could not open Chrome (macOS + Google Chrome expected). "
                f"URL: https://www.google.com/search?q={quote_plus(master_title)}&tbm=isch"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive similar-title group → leaf (1.6).")
    parser.add_argument(
        "--groups",
        metavar="PATH",
        help="unmatched_similar_title_groups.json (default: newest under step-1-similar-title-groups/outputs/).",
    )
    parser.add_argument("--resume-from", metavar="PATH", help="Continue this 1.6 manual JSON.")
    parser.add_argument("--fresh-run", action="store_true", help="Start a new manual file; no auto-resume.")
    parser.add_argument("--force-mismatch", action="store_true", help="Allow resume when groups_source differs.")
    parser.add_argument(
        "--auto-random",
        action="store_true",
        help="Speed-run mode: randomly assign every group to a leaf without prompting.",
    )
    args = parser.parse_args()

    if args.resume_from and args.fresh_run:
        raise SystemExit("Use either --resume-from or --fresh-run, not both.")

    groups_path = Path(args.groups).expanduser().resolve() if args.groups else find_latest_groups_file()
    if not groups_path or not groups_path.is_file():
        raise SystemExit(
            "No unmatched_similar_title_groups.json. Run step-1-similar-title-groups/1_1_build_similar_title_groups.py first."
        )

    init_step15_dedup_cache_from_groups(groups_path)

    data = json.loads(groups_path.read_text(encoding="utf-8"))
    groups = [g for g in (data.get("groups") or []) if isinstance(g, dict)]

    with TAXONOMY_PATH.open("r", encoding="utf-8") as f:
        categories = json.load(f)
    raw_rows = collect_leaf_rows(categories)
    other_rows = collect_other_bucket_leaves(raw_rows)
    search_rows = [r for r in raw_rows if not is_catch_all_bucket_slug(r.get("leaf_slug"))]

    group_assignments: List[dict] = []
    unknown_groups: List[dict] = []
    item_matches: List[dict] = []
    done_g: Set[str] = set()
    already_matched_ids: Set[str] = set()
    out_path: Path
    review_skipped = False

    if args.resume_from:
        out_path = Path(args.resume_from).expanduser().resolve()
        group_assignments, unknown_groups, item_matches, done_g, already_matched_ids = load_resume(
            out_path, groups_path, args.force_mismatch
        )
        print(
            f"\nResuming from {out_path}: "
            f"{len(group_assignments)} assignment(s), {len(unknown_groups)} unknown, "
            f"{len(item_matches)} item row(s)."
        )
        if len(done_g) > 0:
            review_skipped = yn_prompt(
                "Also review groups marked unknown (confirm or re-categorize)?",
                default=False,
            )
    elif args.fresh_run:
        out_path = groups_path.parent / f"manual_similar_title_matches{env_suffix()}.json"
        print(f"\nNew manual file: {out_path.relative_to(ROOT)}")
    else:
        found = find_latest_manual_for_groups_source(groups_path)
        if found is not None:
            # Peek at the file to show progress before prompting
            try:
                _prev = json.loads(found.read_text(encoding="utf-8"))
                _done_count = len(_prev.get("group_assignments") or []) + len(_prev.get("unknown_groups") or [])
            except Exception:
                _done_count = 0
            _total = len(groups)
            _remaining = _total - _done_count
            print(f"\nPrevious session: {found.relative_to(ROOT)}")
            print(f"  Progress: {_done_count}/{_total} decided, {_remaining} remaining")
            if yn_prompt(
                f"Resume last session?",
                default=True,
            ):
                out_path = found
                group_assignments, unknown_groups, item_matches, done_g, already_matched_ids = load_resume(
                    found, groups_path, False
                )
                print(f"Resuming: {len(done_g)}/{_total} group(s) already decided, {_total - len(done_g)} remaining.")
                if len(done_g) > 0:
                    review_skipped = yn_prompt(
                        "Also review groups marked unknown (confirm or re-categorize)?",
                        default=False,
                    )
            else:
                out_path = groups_path.parent / f"manual_similar_title_matches{env_suffix()}.json"
                print(f"\nNew manual file: {out_path.relative_to(ROOT)}")
        else:
            out_path = groups_path.parent / f"manual_similar_title_matches{env_suffix()}.json"
            print(f"\nNew manual file: {out_path.relative_to(ROOT)}")

    # ── Auto-random speed-run mode ─────────────────────────────────────────────
    if args.auto_random:
        if not search_rows:
            raise SystemExit("No leaf categories found in taxonomy — cannot auto-assign.")
        assigned = 0
        for row in groups:
            gid = row.get("group_id")
            if not isinstance(gid, str) or not gid or gid in done_g:
                continue
            master_title = row.get("master_title") or ""
            items_in_group = row.get("items") or []
            chosen = random.choice(search_rows)
            n = assign_group_to_leaf(
                gid=gid,
                master_title=master_title,
                items_in_group=items_in_group,
                chosen=chosen,
                out_path=out_path,
                groups_path=groups_path,
                group_assignments=group_assignments,
                unknown_groups=unknown_groups,
                item_matches=item_matches,
                done_g=done_g,
                already_matched_ids=already_matched_ids,
            )
            assigned += n
        write_manual_snapshot(out_path, groups_path, group_assignments, unknown_groups, item_matches)
        print(
            f"[auto-random] Assigned {len(group_assignments)} group(s), "
            f"{assigned} item(s) → {out_path.relative_to(ROOT)}"
        )
        return

    print(
        f"\n{len(groups)} group(s), {len(search_rows)} leaves for search ({len(other_rows)} catch-all via [o] only). "
        "[Enter]=skip | [o] Other | [x] unknown | [i] insert new category | "
        "[c] copy | [p] preview items (→ [s] Google Images or type substring) | else substring searches leaves.\n"
    )

    # {"type": "assign", "chosen": <leaf dict>} | {"type": "unknown"} | {"type": "skip"}
    last_action: dict | None = None

    for ki, row in enumerate(groups, start=1):
        advance_group = False
        skip_group = False
        gid = row.get("group_id")
        if not isinstance(gid, str) or not gid:
            continue
        remaining = len(groups) - ki
        if gid in done_g:
            is_unknown = any(isinstance(x, dict) and x.get("group_id") == gid for x in unknown_groups)
            if not review_skipped or not is_unknown:
                continue
            master_title_r = row.get("master_title") or ""
            items_in_group_r = row.get("items") or []
            item_ids_r = item_ids_from_group_items(items_in_group_r)
            ga_row = next((x for x in group_assignments if x.get("group_id") == gid), None)
            print("\n" + "─" * 72 + "\n")
            print(
                f"[{ki}/{len(groups)}  {remaining} remaining] REVIEW group={gid!r}  "
                f"master_title={master_title_r!r}  items≈{row.get('item_count', '?')}"
            )
            if is_unknown:
                print("  Current: marked UNKNOWN (saved).")
            elif ga_row:
                lp = ga_row.get("leaf_path") or ""
                print(f"  Current: assigned → {lp} ({ga_row.get('item_count', '?')} item(s)).")
            else:
                print("  Current: (no assignment row; use [r] to clear and redo if needed.)")
            redo = input("  [Enter] keep | [r] recategorize: ").strip().lower()
            if redo != "r":
                continue
            undo_group_decision(
                gid,
                group_assignments,
                unknown_groups,
                item_matches,
                done_g,
                already_matched_ids,
                item_ids_r,
            )
            write_manual_snapshot(out_path, groups_path, group_assignments, unknown_groups, item_matches)
            print("  Cleared saved decision; recategorize below.")

        master_title = row.get("master_title") or ""
        ic = row.get("item_count", "?")
        items_in_group = row.get("items") or []

        print("\n" + "─" * 72 + "\n")
        print(f"[{ki}/{len(groups)}  {remaining} remaining] group={gid!r}  master_title={master_title!r}  items≈{ic}")

        first_search: str | None = None
        while first_search is None:
            if last_action is not None:
                if last_action["type"] == "assign":
                    r_hint = f" | [r] repeat (→ {last_action['chosen']['leaf_path']})"
                elif last_action["type"] == "unknown":
                    r_hint = " | [r] repeat (unknown)"
                else:
                    r_hint = " | [r] repeat (skip)"
            else:
                r_hint = ""
            pre = input(
                "  [Enter] skip | [o] Other | [x] unknown | [i] insert category | "
                f"[c] copy | [p] preview{r_hint} | or substring: "
            ).strip()
            if not pre:
                first_search = ""
                last_action = {"type": "skip"}
            elif pre.lower() == "i":
                new_leaf = interact_insert_new_category(categories, TAXONOMY_PATH)
                if new_leaf is not None:
                    # Rebuild leaf lists so the new category is searchable
                    raw_rows[:] = collect_leaf_rows(categories)
                    other_rows[:] = collect_other_bucket_leaves(raw_rows)
                    search_rows[:] = [r for r in raw_rows if not is_catch_all_bucket_slug(r.get("leaf_slug"))]
                    print(f"  New leaf available: {new_leaf['leaf_path']}")
                    if yn_prompt(f"  Assign this group to '{new_leaf['leaf_path']}'?", default=True):
                        n_assigned = assign_group_to_leaf(
                            gid=gid,
                            master_title=master_title,
                            items_in_group=items_in_group,
                            chosen=new_leaf,
                            out_path=out_path,
                            groups_path=groups_path,
                            group_assignments=group_assignments,
                            unknown_groups=unknown_groups,
                            item_matches=item_matches,
                            done_g=done_g,
                            already_matched_ids=already_matched_ids,
                            assignment_extra={"inserted_by_user": True},
                        )
                        print(f"  Assigned {n_assigned} item(s) → {new_leaf['leaf_path']} (saved)")
                        first_search = "__assigned__"
                        last_action = {"type": "assign", "chosen": new_leaf}
                continue
            elif pre.lower() == "o":
                kind, chosen_other = interact_pick_other_leaf(other_rows)
                if kind == "skip":
                    first_search = ""
                    last_action = {"type": "skip"}
                elif kind == "unknown":
                    unknown_groups.append({"group_id": gid, "master_title": master_title})
                    done_g.add(gid)
                    write_manual_snapshot(
                        out_path, groups_path, group_assignments, unknown_groups, item_matches
                    )
                    print("  Marked unknown (saved).")
                    first_search = "__unknown__"
                    last_action = {"type": "unknown"}
                elif kind == "assign":
                    assert chosen_other is not None
                    n_assigned = assign_group_to_leaf(
                        gid=gid,
                        master_title=master_title,
                        items_in_group=items_in_group,
                        chosen=chosen_other,
                        out_path=out_path,
                        groups_path=groups_path,
                        group_assignments=group_assignments,
                        unknown_groups=unknown_groups,
                        item_matches=item_matches,
                        done_g=done_g,
                        already_matched_ids=already_matched_ids,
                        assignment_extra={"picked_from_other_menu": True},
                    )
                    print(
                        f"  Assigned {n_assigned} item(s) → {chosen_other['leaf_path']} (saved) [Other bucket]"
                    )
                    first_search = "__assigned__"
                    last_action = {"type": "assign", "chosen": chosen_other}
            elif pre.lower() == "x":
                unknown_groups.append({"group_id": gid, "master_title": master_title})
                done_g.add(gid)
                write_manual_snapshot(out_path, groups_path, group_assignments, unknown_groups, item_matches)
                print("  Marked unknown (saved).")
                first_search = "__unknown__"
                last_action = {"type": "unknown"}
            elif pre.lower() == "c":
                master_title_clipboard_actions(master_title, "c")
                continue
            elif pre.lower() == "p":
                n_show = min(PREVIEW_ITEMS_MAX, len(items_in_group))
                print(f"  Items ({n_show} of {len(items_in_group)}):")
                for j, it in enumerate(items_in_group[:PREVIEW_ITEMS_MAX], start=1):
                    tid = (it.get("id") or "") if isinstance(it, dict) else ""
                    tl = (it.get("title") or "").strip() if isinstance(it, dict) else ""
                    st = (it.get("subtitle") or "").strip() if isinstance(it, dict) else ""
                    sub = f" | {st}" if st else ""
                    print(f"    {j:2}. {tid}  {tl}{sub}")
                # Inner preview loop: [s] for Google Images, substring to jump to category
                # search, or Enter to go back to the main prompt.
                while True:
                    p_sel = input(
                        "  [s] Google Images | or type category substring to search: "
                    ).strip()
                    if not p_sel:
                        break  # back to main prompt, first_search stays None
                    if p_sel.lower() == "s":
                        raw_pick = input("  Item # to open in Google Images: ").strip()
                        if raw_pick.isdigit():
                            idx = int(raw_pick) - 1
                            picked = items_in_group[idx] if 0 <= idx < len(items_in_group) else None
                            if isinstance(picked, dict):
                                tl = (picked.get("title") or "").strip()
                                st = (picked.get("subtitle") or "").strip()
                                search_str = f"{tl} | {st}" if tl and st else tl or st or master_title
                            else:
                                search_str = master_title
                            if copy_to_clipboard(search_str):
                                print(f"  Copied to clipboard: {search_str!r}")
                            if open_google_images_in_chrome(search_str):
                                print("  Opened Google Images in Chrome.")
                            else:
                                print(
                                    "  Could not open Chrome (macOS + Google Chrome expected). "
                                    f"URL: https://www.google.com/search?q={quote_plus(search_str)}&tbm=isch"
                                )
                        else:
                            print("  Enter a valid item number.")
                        continue  # stay in preview loop (search another or type substring)
                    else:
                        # Treat as category substring — exit preview and jump to leaf search
                        first_search = p_sel
                        break
                continue  # re-enters outer while if first_search still None (Enter was pressed)
            elif pre.lower() == "r":
                if last_action is None:
                    print("  No previous action to repeat.")
                    continue
                if last_action["type"] == "assign":
                    lp = last_action["chosen"]["leaf_path"]
                    print(f"  Repeat: assign → {lp}")
                elif last_action["type"] == "unknown":
                    print("  Repeat: mark unknown")
                else:
                    print("  Repeat: skip")
                conf = input("  Confirm [y/n]: ").strip().lower()
                if conf != "y":
                    continue
                if last_action["type"] == "assign":
                    chosen_r = last_action["chosen"]
                    n_assigned = assign_group_to_leaf(
                        gid=gid,
                        master_title=master_title,
                        items_in_group=items_in_group,
                        chosen=chosen_r,
                        out_path=out_path,
                        groups_path=groups_path,
                        group_assignments=group_assignments,
                        unknown_groups=unknown_groups,
                        item_matches=item_matches,
                        done_g=done_g,
                        already_matched_ids=already_matched_ids,
                        assignment_extra={"repeated_previous": True},
                    )
                    print(f"  Assigned {n_assigned} item(s) → {chosen_r['leaf_path']} (saved)")
                    first_search = "__assigned__"
                elif last_action["type"] == "unknown":
                    unknown_groups.append({"group_id": gid, "master_title": master_title})
                    done_g.add(gid)
                    write_manual_snapshot(out_path, groups_path, group_assignments, unknown_groups, item_matches)
                    print("  Marked unknown (saved).")
                    first_search = "__unknown__"
                else:
                    first_search = ""
            else:
                first_search = pre

        if first_search == "__unknown__":
            continue
        if first_search == "__assigned__":
            continue
        if not first_search:
            continue

        hits = filter_leaves(search_rows, first_search)
        while True:
            if advance_group:
                break
            if skip_group:
                break
            if not hits:
                print("  No leaf matches. New search (blank = skip group).")
                nq = ""
                while True:
                    nq = input(
                        "  Search ([o] Other | [i] insert | [c] copy | [s] Google Images) or substring; blank=skip group: "
                    ).strip()
                    if nq.lower() == "i":
                        new_leaf = interact_insert_new_category(categories, TAXONOMY_PATH)
                        if new_leaf is not None:
                            raw_rows[:] = collect_leaf_rows(categories)
                            other_rows[:] = collect_other_bucket_leaves(raw_rows)
                            search_rows[:] = [r for r in raw_rows if not is_catch_all_bucket_slug(r.get("leaf_slug"))]
                            print(f"  New leaf available: {new_leaf['leaf_path']}")
                            if yn_prompt(f"  Assign this group to '{new_leaf['leaf_path']}'?", default=True):
                                n_assigned = assign_group_to_leaf(
                                    gid=gid,
                                    master_title=master_title,
                                    items_in_group=items_in_group,
                                    chosen=new_leaf,
                                    out_path=out_path,
                                    groups_path=groups_path,
                                    group_assignments=group_assignments,
                                    unknown_groups=unknown_groups,
                                    item_matches=item_matches,
                                    done_g=done_g,
                                    already_matched_ids=already_matched_ids,
                                    assignment_extra={"inserted_by_user": True},
                                )
                                print(f"  Assigned {n_assigned} item(s) → {new_leaf['leaf_path']} (saved)")
                                last_action = {"type": "assign", "chosen": new_leaf}
                                advance_group = True
                        nq = ""
                        break
                    if nq.lower() == "o":
                        kind, chosen_other = interact_pick_other_leaf(other_rows)
                        if kind == "skip":
                            last_action = {"type": "skip"}
                            skip_group = True
                            break
                        if kind == "unknown":
                            unknown_groups.append({"group_id": gid, "master_title": master_title})
                            done_g.add(gid)
                            write_manual_snapshot(
                                out_path, groups_path, group_assignments, unknown_groups, item_matches
                            )
                            print("  Marked unknown (saved).")
                            last_action = {"type": "unknown"}
                            advance_group = True
                            break
                        assert chosen_other is not None
                        n_assigned = assign_group_to_leaf(
                            gid=gid,
                            master_title=master_title,
                            items_in_group=items_in_group,
                            chosen=chosen_other,
                            out_path=out_path,
                            groups_path=groups_path,
                            group_assignments=group_assignments,
                            unknown_groups=unknown_groups,
                            item_matches=item_matches,
                            done_g=done_g,
                            already_matched_ids=already_matched_ids,
                            assignment_extra={"picked_from_other_menu": True},
                        )
                        print(
                            f"  Assigned {n_assigned} item(s) → {chosen_other['leaf_path']} (saved) [Other]"
                        )
                        last_action = {"type": "assign", "chosen": chosen_other}
                        advance_group = True
                        break
                    if nq.lower() in ("c", "s"):
                        master_title_clipboard_actions(master_title, nq)
                        continue
                    break
                if advance_group:
                    break
                if skip_group:
                    break
                if not nq:
                    break
                hits = filter_leaves(search_rows, nq)
                continue

            if len(hits) > MAX_LEAVES_TO_SHOW:
                print(
                    f"  {len(hits)} leaf matches (showing first {MAX_LEAVES_TO_SHOW}); "
                    f"type more text to narrow this list, or n <text> for a new full search."
                )
                show = hits[:MAX_LEAVES_TO_SHOW]
            else:
                show = hits

            for i, h in enumerate(show, start=1):
                dn = h.get("display_name") or ""
                print(f"    {i:3}) {h['leaf_path']}")
                if dn:
                    print(f"        {dn}")

            sel = input(
                "  [#] assign | [o] Other | [i] insert category | [text] narrow | "
                "[n]/[n <text>] search | [c] copy | [s] Google Images | [Enter] skip: "
            ).strip()

            if not sel:
                break

            if sel.lower() == "c":
                master_title_clipboard_actions(master_title, "c")
                continue

            if sel.lower() == "s":
                n_show = min(PREVIEW_ITEMS_MAX, len(items_in_group))
                print(f"  Pick an item to search on Google Images ({n_show} of {len(items_in_group)}):")
                for j, it in enumerate(items_in_group[:PREVIEW_ITEMS_MAX], start=1):
                    tl = (it.get("title") or "").strip() if isinstance(it, dict) else ""
                    st = (it.get("subtitle") or "").strip() if isinstance(it, dict) else ""
                    sub = f" | {st}" if st else ""
                    print(f"    {j:2}. {tl}{sub}")
                raw_pick = input("  Item # (or Enter to use group title): ").strip()
                if raw_pick.isdigit():
                    idx = int(raw_pick) - 1
                    picked = items_in_group[idx] if 0 <= idx < len(items_in_group) else None
                    if isinstance(picked, dict):
                        tl = (picked.get("title") or "").strip()
                        st = (picked.get("subtitle") or "").strip()
                        search_str = f"{tl} | {st}" if tl and st else tl or st or master_title
                    else:
                        search_str = master_title
                else:
                    search_str = master_title
                if copy_to_clipboard(search_str):
                    print(f"  Copied to clipboard: {search_str!r}")
                if open_google_images_in_chrome(search_str):
                    print("  Opened Google Images in Chrome.")
                else:
                    print(
                        "  Could not open Chrome (macOS + Google Chrome expected). "
                        f"URL: https://www.google.com/search?q={quote_plus(search_str)}&tbm=isch"
                    )
                continue

            if sel.lower() == "i":
                new_leaf = interact_insert_new_category(categories, TAXONOMY_PATH)
                if new_leaf is not None:
                    raw_rows[:] = collect_leaf_rows(categories)
                    other_rows[:] = collect_other_bucket_leaves(raw_rows)
                    search_rows[:] = [r for r in raw_rows if not is_catch_all_bucket_slug(r.get("leaf_slug"))]
                    hits = filter_leaves(search_rows, first_search) if first_search else hits
                    print(f"  New leaf available: {new_leaf['leaf_path']}")
                    if yn_prompt(f"  Assign this group to '{new_leaf['leaf_path']}'?", default=True):
                        n_assigned = assign_group_to_leaf(
                            gid=gid,
                            master_title=master_title,
                            items_in_group=items_in_group,
                            chosen=new_leaf,
                            out_path=out_path,
                            groups_path=groups_path,
                            group_assignments=group_assignments,
                            unknown_groups=unknown_groups,
                            item_matches=item_matches,
                            done_g=done_g,
                            already_matched_ids=already_matched_ids,
                            assignment_extra={"inserted_by_user": True},
                        )
                        print(f"  Assigned {n_assigned} item(s) → {new_leaf['leaf_path']} (saved)")
                        last_action = {"type": "assign", "chosen": new_leaf}
                        advance_group = True
                        break
                continue

            if sel.lower() == "o":
                kind, chosen_other = interact_pick_other_leaf(other_rows)
                if kind == "skip":
                    last_action = {"type": "skip"}
                    skip_group = True
                    break
                if kind == "unknown":
                    unknown_groups.append({"group_id": gid, "master_title": master_title})
                    done_g.add(gid)
                    write_manual_snapshot(
                        out_path, groups_path, group_assignments, unknown_groups, item_matches
                    )
                    print("  Marked unknown (saved).")
                    last_action = {"type": "unknown"}
                    advance_group = True
                    break
                assert chosen_other is not None
                n_assigned = assign_group_to_leaf(
                    gid=gid,
                    master_title=master_title,
                    items_in_group=items_in_group,
                    chosen=chosen_other,
                    out_path=out_path,
                    groups_path=groups_path,
                    group_assignments=group_assignments,
                    unknown_groups=unknown_groups,
                    item_matches=item_matches,
                    done_g=done_g,
                    already_matched_ids=already_matched_ids,
                    assignment_extra={"picked_from_other_menu": True},
                )
                print(f"  Assigned {n_assigned} item(s) → {chosen_other['leaf_path']} (saved) [Other]")
                last_action = {"type": "assign", "chosen": chosen_other}
                advance_group = True
                break

            if re.fullmatch(r"\d+", sel):
                n = int(sel)
                if not (1 <= n <= len(show)):
                    print("  Out of range.")
                    continue
                chosen = show[n - 1]
                n_assigned = assign_group_to_leaf(
                    gid=gid,
                    master_title=master_title,
                    items_in_group=items_in_group,
                    chosen=chosen,
                    out_path=out_path,
                    groups_path=groups_path,
                    group_assignments=group_assignments,
                    unknown_groups=unknown_groups,
                    item_matches=item_matches,
                    done_g=done_g,
                    already_matched_ids=already_matched_ids,
                )
                print(f"  Assigned {n_assigned} item(s) → {chosen['leaf_path']} (saved)")
                last_action = {"type": "assign", "chosen": chosen}
                break

            low = sel.lower()
            if low.startswith("n ") and len(sel) > 2:
                hits = filter_leaves(search_rows, sel[2:].strip())
                continue
            if low == "n":
                nq = input("  New full taxonomy search substring: ").strip()
                if not nq:
                    continue
                hits = filter_leaves(search_rows, nq)
                continue

            narrowed = filter_hits_narrow(hits, sel)
            if not narrowed:
                print("  Narrowing removed all rows; list unchanged.")
            else:
                hits = narrowed
            continue

        if advance_group or skip_group:
            continue

    write_manual_snapshot(out_path, groups_path, group_assignments, unknown_groups, item_matches)
    write_step_summary(
        out_path.parent,
        step="step-1-similar-title-groups (1.2 manual)",
        stats={
            "group_assignments": len(group_assignments),
            "unknown_groups":    len(unknown_groups),
            "item_matches":      len(item_matches),
        },
        input_files=[str(groups_path.name)],
        output_files=[out_path.name, f"unmatched_after_step1{env_suffix()}.json"],
    )
    print(f"\nFinal: {out_path}")
    print(
        f"group_assignments: {len(group_assignments)}  unknown_groups: {len(unknown_groups)}  "
        f"item_matches: {len(item_matches)}"
    )
    cc = compute_cumulative_matched_and_remaining(
        groups_path, group_assignments, unknown_groups, item_matches
    ).get("counts") or {}
    if cc:
        print(
            f"Cumulative: matched {cc.get('matched_cumulative_unique_ids', '?')} "
            f"(step 1); remaining unmatched/skipped "
            f"{cc.get('remaining_unmatched_or_skipped_after_1_6', '?')}"
        )

    clear_step15_dedup_cache()


if __name__ == "__main__":
    main()
