#!/usr/bin/env python3
"""
Step 1.6 (interactive)
Assign taxonomy leaves to groups built by build_unmatched_similar_title_groups_1_6.py.

UX mirrors step-1.3: search leaves by substring, pick by number, save after each decision.
Per group: [c] copies master_title; [s] copy + Google Images in Chrome (macOS). [o] lists all taxonomy
Other leaves (catch-alls); [x] marks unknown without picking Other.

Outputs (new channel, not 1.3 manual):
  step-1.6/outputs/<run>/1.6-manual_similar_title_<timestamp>.json
  or --resume-from continues that file.

Fields: group_assignments, unknown_groups, item_matches (source=manual_similar_title_1_6).
Also: matched_cumulative (1.0–1.5 matched + 1.6 assigns), unmatched_and_skipped_cumulative (pool
minus 1.6 assigns, with status per row), sourced from sibling matched_deduped / unmatched_deduped
of the groups file’s source_unmatched_deduped.
"""

from __future__ import annotations

import argparse
import json
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

TAXONOMY_PATH = ROOT / "source-files" / "categories_v1.json"
STEP16_OUT = ROOT / "step-1.6" / "outputs"

MAX_LEAVES_TO_SHOW = 100
PREVIEW_ITEMS_MAX = 12
MANUAL_VERSION = "1.6-manual-similar-title"

# Filled once per main() from sibling 1.5 JSONs; avoids re-reading large matched_deduped on every save.
_step15_dedup_cache: dict | None = None


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def collect_leaf_rows(categories: dict) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    roots = ["materials", "tools_and_gear", "services"]

    def walk(node: Any, prefix: List[str]) -> None:
        if not isinstance(node, dict):
            return
        slug = node.get("slug")
        subs = node.get("subcategories") or []
        dn = (node.get("display_name") or "").strip()

        if slug is not None:
            here = prefix + [slug]
        else:
            here = prefix

        if not subs:
            if slug is not None:
                lp = "/".join(here)
                hay = f"{lp} {slug} {dn}".lower()
                rows.append(
                    {
                        "leaf_path": lp,
                        "leaf_slug": slug,
                        "display_name": dn,
                        "_hay": hay,
                    }
                )
            return
        for child in subs:
            walk(child, here)

    for root in roots:
        root_obj = categories.get(root)
        if not isinstance(root_obj, dict):
            continue
        for child in root_obj.get("subcategories", []):
            walk(child, [root])

    rows.sort(key=lambda r: r["leaf_path"])
    return rows


def filter_leaves(leaves: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
    q = query.strip().lower()
    if not q:
        return []
    return [r for r in leaves if q in r["_hay"]]


def filter_hits_narrow(hits: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
    q = query.strip().lower()
    if not q:
        return hits
    return [h for h in hits if q in h.get("_hay", "")]


def collect_other_bucket_leaves(raw_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Catch-all bucket leaves: slug `other` (per domain) plus materials top-level `miscellaneous`."""
    return [
        r
        for r in raw_rows
        if r.get("leaf_slug") == "other" or r.get("leaf_slug") == "miscellaneous"
    ]


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


def interact_pick_other_leaf(other_rows: List[Dict[str, str]]) -> tuple[str, Dict[str, str] | None]:
    """
    Prompt user to pick an Other bucket leaf.
    Returns ("assign", row) | ("unknown", None) | ("skip", None).
    Enter = skip matching entirely (no assign, no unknown; group/bigram stays unmatched for later).
    """
    if not other_rows:
        print("  (No Other leaves in taxonomy; add them in categories_v1.json.)")
        return ("skip", None)
    print(f"  Other buckets ({len(other_rows)}) — pick a catch-all leaf:")
    for i, h in enumerate(other_rows, start=1):
        dn = h.get("display_name") or ""
        print(f"    {i:3}) {h['leaf_path']}")
        if dn:
            print(f"        {dn}")
    while True:
        pick = input(
            "  [#] assign here | [Enter] skip (no category) | [x] mark unknown (saved): "
        ).strip()
        if not pick:
            return ("skip", None)
        if pick.lower() == "x":
            return ("unknown", None)
        if re.fullmatch(r"\d+", pick):
            n = int(pick)
            if 1 <= n <= len(other_rows):
                return ("assign", other_rows[n - 1])
            print("  Out of range.")
            continue
        print("  Enter a number, Enter, or x.")


def find_latest_groups_file() -> Path | None:
    cands = list(STEP16_OUT.glob("**/unmatched_similar_title_groups.json"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def find_latest_manual_for_groups_source(out_dir: Path, groups_src: Path) -> Path | None:
    gs = groups_src.resolve()
    best: tuple[float, Path] | None = None
    for p in out_dir.glob("**/1.6-manual_similar_title*.json"):
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
    """Load matched_deduped + unmatched_deduped once (siblings of source_unmatched_deduped)."""
    global _step15_dedup_cache
    unmatched_src: str | None = None
    matched_src: str | None = None
    matched_prior: List[dict] = []
    unmatched_pool: List[dict] = []
    try:
        gd = json.loads(groups_path.read_text(encoding="utf-8"))
        u = gd.get("source_unmatched_deduped")
        if isinstance(u, str) and u.strip():
            unmatched_src = str(Path(u).expanduser().resolve())
    except (OSError, json.JSONDecodeError):
        _step15_dedup_cache = {
            "unmatched_src": None,
            "matched_src": None,
            "matched_prior": [],
            "unmatched_pool": [],
        }
        return

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
            u = gd.get("source_unmatched_deduped")
            if isinstance(u, str) and u.strip():
                unmatched_src = str(Path(u).expanduser().resolve())
        except (OSError, json.JSONDecodeError):
            pass

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


def copy_to_clipboard(text: str) -> bool:
    """Best-effort clipboard (macOS pbcopy; elsewhere may fail)."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
            return True
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return False


def open_google_images_in_chrome(query: str) -> bool:
    """Open Google Images search in Google Chrome (macOS `open`). Uses `tbm=isch`."""
    url = f"https://www.google.com/search?q={quote_plus(query)}&tbm=isch"
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", "-a", "Google Chrome", url], check=False)
            return True
    except OSError:
        pass
    return False


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


def yn_prompt(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        ans = input(f"{prompt} [{hint}] ").strip().lower()
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive similar-title group → leaf (1.6).")
    parser.add_argument(
        "--groups",
        metavar="PATH",
        help="unmatched_similar_title_groups.json (default: newest under step-1.6/outputs/).",
    )
    parser.add_argument("--resume-from", metavar="PATH", help="Continue this 1.6 manual JSON.")
    parser.add_argument("--fresh-run", action="store_true", help="Start a new manual file; no auto-resume.")
    parser.add_argument("--force-mismatch", action="store_true", help="Allow resume when groups_source differs.")
    args = parser.parse_args()

    if args.resume_from and args.fresh_run:
        raise SystemExit("Use either --resume-from or --fresh-run, not both.")

    groups_path = Path(args.groups).expanduser().resolve() if args.groups else find_latest_groups_file()
    if not groups_path or not groups_path.is_file():
        raise SystemExit(
            "No unmatched_similar_title_groups.json. Run build_unmatched_similar_title_groups_1_6.py first."
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
                "Also review groups already marked assigned or unknown (confirm or change)?",
                default=False,
            )
    elif args.fresh_run:
        out_path = groups_path.parent / f"1.6-manual_similar_title_{timestamp()}.json"
        print(f"\nNew manual file: {out_path.relative_to(ROOT)}")
    else:
        found = find_latest_manual_for_groups_source(STEP16_OUT, groups_path)
        if found is not None:
            print(f"\nPrevious session: {found.relative_to(ROOT)}")
            if yn_prompt(
                f"Resume last session? ({len(groups)} group(s) in groups file.)",
                default=True,
            ):
                out_path = found
                group_assignments, unknown_groups, item_matches, done_g, already_matched_ids = load_resume(
                    found, groups_path, False
                )
                print(f"Resuming: {len(done_g)} group(s) already decided.")
                if len(done_g) > 0:
                    review_skipped = yn_prompt(
                        "Also review groups already marked assigned or unknown (confirm or change)?",
                        default=False,
                    )
            else:
                out_path = groups_path.parent / f"1.6-manual_similar_title_{timestamp()}.json"
                print(f"\nNew manual file: {out_path.relative_to(ROOT)}")
        else:
            out_path = groups_path.parent / f"1.6-manual_similar_title_{timestamp()}.json"
            print(f"\nNew manual file: {out_path.relative_to(ROOT)}")

    print(
        f"\n{len(groups)} group(s), {len(search_rows)} leaves for search ({len(other_rows)} catch-all via [o] only). "
        "[Enter]=skip | [o] Other buckets | [x] unknown | [p] preview | [c] copy | "
        "[s] Google Images | else substring searches leaves.\n"
    )

    for ki, row in enumerate(groups, start=1):
        advance_group = False
        skip_group = False
        gid = row.get("group_id")
        if not isinstance(gid, str) or not gid:
            continue
        if gid in done_g:
            if not review_skipped:
                print("-" * 72)
                print(f"[{ki}/{len(groups)}] group {gid!r} — skipped (already decided).")
                continue
            master_title_r = row.get("master_title") or ""
            items_in_group_r = row.get("items") or []
            item_ids_r = item_ids_from_group_items(items_in_group_r)
            ga_row = next((x for x in group_assignments if x.get("group_id") == gid), None)
            is_unknown = any(isinstance(x, dict) and x.get("group_id") == gid for x in unknown_groups)
            print("-" * 72)
            print(
                f"[{ki}/{len(groups)}] REVIEW group={gid!r}  "
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

        print("-" * 72)
        print(f"[{ki}/{len(groups)}] group={gid!r}  master_title={master_title!r}  items≈{ic}")

        first_search: str | None = None
        while first_search is None:
            pre = input(
                "  [Enter] skip | [o] Other | [x] unknown | [p] preview | [c] copy | [s] Google Images | "
                "or substring: "
            ).strip()
            if not pre:
                first_search = ""
            elif pre.lower() == "o":
                kind, chosen_other = interact_pick_other_leaf(other_rows)
                if kind == "skip":
                    first_search = ""
                elif kind == "unknown":
                    unknown_groups.append({"group_id": gid, "master_title": master_title})
                    done_g.add(gid)
                    write_manual_snapshot(
                        out_path, groups_path, group_assignments, unknown_groups, item_matches
                    )
                    print("  Marked unknown (saved).")
                    first_search = "__unknown__"
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
            elif pre.lower() == "x":
                unknown_groups.append({"group_id": gid, "master_title": master_title})
                done_g.add(gid)
                write_manual_snapshot(out_path, groups_path, group_assignments, unknown_groups, item_matches)
                print("  Marked unknown (saved).")
                first_search = "__unknown__"
            elif pre.lower() == "p":
                n_show = min(PREVIEW_ITEMS_MAX, len(items_in_group))
                print(f"  Sample ({n_show} of {len(items_in_group)} item(s)):")
                for j, it in enumerate(items_in_group[:PREVIEW_ITEMS_MAX], start=1):
                    tid = (it.get("id") or "") if isinstance(it, dict) else ""
                    tl = (it.get("title") or "").strip() if isinstance(it, dict) else ""
                    st = (it.get("subtitle") or "").strip() if isinstance(it, dict) else ""
                    sub = f" | {st}" if st else ""
                    print(f"    {j:2}. {tid}  {tl}{sub}")
                continue
            elif pre.lower() in ("c", "s"):
                master_title_clipboard_actions(master_title, pre)
                continue
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
                        "  Search ([o] Other | [c] copy | [s] Google Images) or substring; blank=skip group: "
                    ).strip()
                    if nq.lower() == "o":
                        kind, chosen_other = interact_pick_other_leaf(other_rows)
                        if kind == "skip":
                            skip_group = True
                            break
                        if kind == "unknown":
                            unknown_groups.append({"group_id": gid, "master_title": master_title})
                            done_g.add(gid)
                            write_manual_snapshot(
                                out_path, groups_path, group_assignments, unknown_groups, item_matches
                            )
                            print("  Marked unknown (saved).")
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
                "  [#] assign | [o] Other | [text] narrow | [n]/[n <text>] search | [c] copy | [s] Google Images | "
                "[Enter] skip: "
            ).strip()

            if not sel:
                break

            if sel.lower() in ("c", "s"):
                master_title_clipboard_actions(master_title, sel)
                continue

            if sel.lower() == "o":
                kind, chosen_other = interact_pick_other_leaf(other_rows)
                if kind == "skip":
                    skip_group = True
                    break
                if kind == "unknown":
                    unknown_groups.append({"group_id": gid, "master_title": master_title})
                    done_g.add(gid)
                    write_manual_snapshot(
                        out_path, groups_path, group_assignments, unknown_groups, item_matches
                    )
                    print("  Marked unknown (saved).")
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
            f"(1.5 + 1.6); remaining unmatched/skipped "
            f"{cc.get('remaining_unmatched_or_skipped_after_1_6', '?')}"
        )

    clear_step15_dedup_cache()


if __name__ == "__main__":
    main()
