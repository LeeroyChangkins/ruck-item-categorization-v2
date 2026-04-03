#!/usr/bin/env python3
"""
Step 2.3 — Interactive manual bigram → leaf

Build bigrams from `unmatched_word_frequencies` in a step-2.2 split `unmatched_and_keywords.json`,
then walk each bigram (>= MIN_ITEMS_PER_BIGRAM items, default 10), search taxonomy leaves, and assign a leaf to eligible unmatched items.

Bigram → item rule (same-field, like step-2.2 title/subtitle split):
  Both words appear as letter-tokens in the item title, OR both appear in the subtitle.
  (Not if one word is only in the title and the other only in the subtitle.)

Inputs:
  - source-files/categories_v1.json
  - step-2/outputs/1.2_split_*/unmatched_and_keywords.json (or --input path)

Outputs:
  - step-2/outputs/1.3-manual_bigram_matches_YYYYMMDD_HHMMSS.json (or an existing file when
    resuming). Each file stores `unmatched_keywords_source` pointing at the 2.2 split JSON.

Resume:
  - If `--fresh-run` is not set and `--resume-from` is omitted, and a matching prior manual
    file exists, you are prompted: "Resume last session? N items…, M bigrams…" (default yes).
  - `--fresh-run` always starts a new timestamped file (e.g. full pipeline from step 1.1).
  - `--resume-from PATH` continues that file with no prompt.
  - Progress is saved after each successful assignment or unknown mark (safe to interrupt).
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from urllib.parse import quote_plus

ROOT = Path(__file__).resolve().parents[1]  # v2/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taxonomy_cascade import is_catch_all_bucket_slug
from shared_utils import timestamp
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
STEP12_OUT = ROOT / "step-2-bigram-keyword-matching" / "outputs"
OUTDIR = Path(__file__).resolve().parent / "outputs"

MAX_LEAVES_TO_SHOW = 100
PREVIEW_ITEMS_MAX = 12
MIN_ITEMS_PER_BIGRAM = 10
MANUAL_VERSION = "1.4-manual-bigram"


class _Spinner:
    """Simple terminal spinner. Use as a context manager."""

    def __init__(self, msg: str = "Loading") -> None:
        self._msg = msg
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        for ch in itertools.cycle("|/-\\"):
            if self._stop.is_set():
                break
            print(f"\r  {self._msg}… {ch}", end="", flush=True)
            time.sleep(0.1)
        print("\r" + " " * (len(self._msg) + 8) + "\r", end="", flush=True)

    def __enter__(self) -> "_Spinner":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()


def tokenize_alpha_preserve(text: str) -> List[str]:
    if not text:
        return []
    s = str(text)
    out: List[str] = []
    cur: List[str] = []
    for ch in s:
        o = ord(ch)
        if (65 <= o <= 90) or (97 <= o <= 122):
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out




def assign_bigram_to_leaf(
    chosen: Dict[str, str],
    *,
    hit_items: List[dict],
    bigram_display: list,
    pk: Tuple[str, str],
    out_path: Path,
    in_path: Path,
    assignments: List[dict],
    matches: List[dict],
    unknown_bigrams: List[dict],
    already_matched_ids: Set[str],
    done_pairs: Set[Tuple[str, str]],
    extra_assignment: Dict[str, Any] | None = None,
) -> None:
    leaf_path = chosen["leaf_path"]
    leaf_slug = chosen["leaf_slug"]
    display_name = chosen.get("display_name") or ""
    ids = [it["id"] for it in hit_items]
    rec: Dict[str, Any] = {
        "bigram": list(bigram_display),
        "pair_lower_sorted": [pk[0], pk[1]],
        "leaf_path": leaf_path,
        "leaf_slug": leaf_slug,
        "leaf_display_name": display_name,
        "matched_item_count": len(hit_items),
        "item_ids": ids,
    }
    if extra_assignment:
        rec.update(extra_assignment)
    assignments.append(rec)
    done_pairs.add(pk)
    for it in hit_items:
        iid = it["id"]
        matches.append(
            {
                "id": iid,
                "title": it.get("title") or "",
                "subtitle": it.get("subtitle") or "",
                "matched_via_bigram": list(bigram_display),
                "pair_lower_sorted": [pk[0], pk[1]],
                "leaf_path": leaf_path,
                "leaf_slug": leaf_slug,
                "leaf_display_name": display_name,
                "source": "manual_bigram_1_3",
            }
        )
        already_matched_ids.add(iid)
    write_manual_snapshot(out_path, in_path, assignments, matches, unknown_bigrams)


def newest_unmatched_keywords_file() -> Path | None:
    cands = list(STEP12_OUT.glob("1.2_split_*/unmatched_and_keywords.json"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def find_latest_manual_for_source(out_dir: Path, in_path: Path) -> Path | None:
    """
    Newest 1.3 manual JSON in out_dir whose unmatched_keywords_source resolves to the same
    path as in_path (the step-2.2 split unmatched_and_keywords.json).
    """
    in_resolved = in_path.resolve()
    best: tuple[float, Path] | None = None
    for p in out_dir.glob("1.3-manual*.json"):
        if not p.is_file():
            continue
        try:
            prev = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        src = prev.get("unmatched_keywords_source")
        if not isinstance(src, str) or not src.strip():
            continue
        try:
            if Path(src).resolve() != in_resolved:
                continue
        except OSError:
            continue
        mtime = p.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, p)
    return best[1] if best else None


def load_resumed_state(
    resume_p: Path,
    in_path: Path,
    display_for_lower: Dict[str, str],
    force_mismatch: bool,
) -> tuple[List[dict], List[dict], List[dict], Set[Tuple[str, str]], Set[str]]:
    """Load bigram_assignments, item_matches, unknown_bigrams; build done_pairs and id set."""
    if not resume_p.exists():
        raise SystemExit(f"Resume file not found: {resume_p}")
    prev = json.loads(resume_p.read_text(encoding="utf-8"))
    prev_src = prev.get("unmatched_keywords_source")
    if isinstance(prev_src, str) and Path(prev_src).resolve() != in_path.resolve():
        if not force_mismatch:
            raise SystemExit(
                f"Resume file was built from a different unmatched_keywords file:\n"
                f"  resume: {prev_src}\n"
                f"  current --input: {in_path}\n"
                f"Use matching --input or pass --force-mismatch (risky)."
            )
        print("Warning: --force-mismatch: source file differs from resume metadata.", flush=True)
    legacy_kw = prev.get("keyword_assignments")
    legacy_uk = prev.get("unknown_keywords")
    if (legacy_kw or legacy_uk) and not prev.get("bigram_assignments"):
        raise SystemExit(
            "This resume file uses the old single-keyword format (keyword_assignments / "
            "unknown_keywords). Start a new output with --fresh-run, or use a manual "
            f"JSON from the bigram matcher (version {MANUAL_VERSION!r})."
        )
    pa = prev.get("bigram_assignments")
    pm = prev.get("item_matches")
    if not isinstance(pa, list):
        pa = []
    if not isinstance(pm, list):
        pm = []
    assignments = [a for a in pa if isinstance(a, dict)]
    matches = [m for m in pm if isinstance(m, dict)]
    unknown_bigrams: List[dict] = []
    uk_prev = prev.get("unknown_bigrams")
    if isinstance(uk_prev, list):
        seen_uk: Set[Tuple[str, str]] = set()
        for x in uk_prev:
            if not isinstance(x, dict):
                continue
            pls = x.get("pair_lower_sorted")
            if not isinstance(pls, list) or len(pls) != 2:
                continue
            k = pair_key_from_lower_sorted(str(pls[0]), str(pls[1]))
            if k in seen_uk:
                continue
            seen_uk.add(k)
            unknown_bigrams.append(
                {
                    "bigram": x.get("bigram")
                    if isinstance(x.get("bigram"), list)
                    else [display_for_lower.get(k[0], k[0]), display_for_lower.get(k[1], k[1])],
                    "pair_lower_sorted": [k[0], k[1]],
                }
            )
    done_pairs: Set[Tuple[str, str]] = set()
    for a in assignments:
        pls = a.get("pair_lower_sorted")
        if isinstance(pls, list) and len(pls) == 2:
            done_pairs.add(pair_key_from_lower_sorted(str(pls[0]), str(pls[1])))
    for u in unknown_bigrams:
        pls = u.get("pair_lower_sorted")
        if isinstance(pls, list) and len(pls) == 2:
            done_pairs.add(pair_key_from_lower_sorted(str(pls[0]), str(pls[1])))
    already_matched_ids = {
        m["id"] for m in matches if isinstance(m.get("id"), str) and m["id"]
    }
    return assignments, matches, unknown_bigrams, done_pairs, already_matched_ids


def pick_input_interactive() -> Path:
    latest = newest_unmatched_keywords_file()
    print("Choose unmatched_and_keywords.json (from step-2.2 split folder):")
    if latest:
        print(f"  1) Use most recent: {latest.relative_to(ROOT)}")
    else:
        print("  1) (none found under step-2/outputs/1.2_split_*)")
    print("  2) Enter a path manually")
    choice = input("> ").strip()
    if choice == "1" and latest:
        return latest
    if choice == "2":
        p = Path(input("Path: ").strip()).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"File not found: {p}")
        return p
    raise SystemExit("Invalid selection.")


def pair_key_from_lower_sorted(a: str, b: str) -> Tuple[str, str]:
    x, y = a.lower(), b.lower()
    return (x, y) if x <= y else (y, x)


def vocabulary_from_frequency_rows(kws: List[dict]) -> Tuple[List[str], Dict[str, str]]:
    """Unique words (lower) in frequency-table order; display string = first seen."""
    seen: Set[str] = set()
    order_low: List[str] = []
    display_for_lower: Dict[str, str] = {}
    for row in kws:
        if not isinstance(row, dict):
            continue
        w = row.get("word")
        if not isinstance(w, str) or not w:
            continue
        wl = w.lower()
        if wl in seen:
            continue
        seen.add(wl)
        order_low.append(wl)
        display_for_lower[wl] = w
    return order_low, display_for_lower


def build_title_subtitle_token_sets(item: dict) -> Tuple[Set[str], Set[str]]:
    title = item.get("title") or ""
    subtitle = item.get("subtitle") or ""
    title_toks = {t.lower() for t in tokenize_alpha_preserve(title)}
    sub_toks = {t.lower() for t in tokenize_alpha_preserve(subtitle)}
    return title_toks, sub_toks


def item_ids_matching_bigram_same_field(
    a_low: str,
    b_low: str,
    title_postings: Dict[str, Set[str]],
    subtitle_postings: Dict[str, Set[str]],
) -> Set[str]:
    ta = title_postings.get(a_low) or set()
    tb = title_postings.get(b_low) or set()
    sa = subtitle_postings.get(a_low) or set()
    sb = subtitle_postings.get(b_low) or set()
    return (ta & tb) | (sa & sb)


def compile_bigrams_from_unmatched(
    unmatched_items: List[dict],
    order_low: List[str],
    display_for_lower: Dict[str, str],
    min_items: int,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], List[dict]]:
    """
    Postings: word_lower -> set of item ids that contain that word as a letter-token in title
    (resp. subtitle) only.
    Returns (title_postings, subtitle_postings, bigram_rows sorted by -item_count).
    """
    title_postings: Dict[str, Set[str]] = {}
    subtitle_postings: Dict[str, Set[str]] = {}

    for it in unmatched_items:
        if not isinstance(it, dict) or not isinstance(it.get("id"), str) or not it["id"]:
            continue
        iid = it["id"]
        tt, st = build_title_subtitle_token_sets(it)
        for t in tt:
            title_postings.setdefault(t, set()).add(iid)
        for t in st:
            subtitle_postings.setdefault(t, set()).add(iid)

    vocab_set = set(order_low)
    rows: List[dict] = []
    sorted_low = sorted(vocab_set)

    for a, b in combinations(sorted_low, 2):
        ids = item_ids_matching_bigram_same_field(a, b, title_postings, subtitle_postings)
        n = len(ids)
        if n < min_items:
            continue
        pair = pair_key_from_lower_sorted(a, b)
        w1, w2 = pair[0], pair[1]
        rows.append(
            {
                "bigram": [display_for_lower[w1], display_for_lower[w2]],
                "pair_lower_sorted": [w1, w2],
                "unmatched_item_count": n,
            }
        )

    rows.sort(
        key=lambda r: (
            -int(r["unmatched_item_count"]),
            r["pair_lower_sorted"][0],
            r["pair_lower_sorted"][1],
        )
    )
    return title_postings, subtitle_postings, rows


def item_matches_bigram_same_field(
    item: dict,
    pair_lower_sorted: List[str],
) -> bool:
    if len(pair_lower_sorted) != 2:
        return False
    a, b = pair_lower_sorted[0], pair_lower_sorted[1]
    tt, st = build_title_subtitle_token_sets(item)
    return (a in tt and b in tt) or (a in st and b in st)


def count_uncategorized_items(unmatched_items: List[dict], already_matched_ids: Set[str]) -> int:
    n = 0
    for it in unmatched_items:
        if not isinstance(it, dict):
            continue
        iid = it.get("id")
        if isinstance(iid, str) and iid and iid not in already_matched_ids:
            n += 1
    return n


def count_bigrams_with_work_remaining(
    bigram_rows: List[dict],
    done_pairs: Set[Tuple[str, str]],
    unmatched_items: List[dict],
    already_matched_ids: Set[str],
) -> int:
    """Bigrams not yet decided (assign/unknown) that still have >=1 eligible item."""
    n = 0
    for row in bigram_rows:
        if not isinstance(row, dict):
            continue
        pls = row.get("pair_lower_sorted")
        if not isinstance(pls, list) or len(pls) != 2:
            continue
        pk = pair_key_from_lower_sorted(str(pls[0]), str(pls[1]))
        if pk in done_pairs:
            continue
        has_hit = False
        for it in unmatched_items:
            if not isinstance(it, dict):
                continue
            iid = it.get("id")
            if not isinstance(iid, str) or not iid:
                continue
            if iid in already_matched_ids:
                continue
            if item_matches_bigram_same_field(it, list(pk)):
                has_hit = True
                break
        if has_hit:
            n += 1
    return n


def print_session_progress(
    unmatched_items: List[dict],
    bigram_rows: List[dict],
    done_pairs: Set[Tuple[str, str]],
    already_matched_ids: Set[str],
) -> None:
    total = sum(1 for it in unmatched_items if isinstance(it, dict) and isinstance(it.get("id"), str) and it["id"])
    nu = total - len(already_matched_ids)
    nb = len(bigram_rows) - len(done_pairs)
    print(f"  Progress: {nu} item(s) still uncategorized, {nb} bigram(s) remaining.")


def write_manual_snapshot(
    out_path: Path,
    in_path: Path,
    assignments: List[dict],
    matches: List[dict],
    unknown_bigrams: List[dict],
) -> None:
    payload = {
        "version": MANUAL_VERSION,
        "taxonomy_categories_file": TAXONOMY_PATH.name,
        "unmatched_keywords_source": str(in_path.resolve()),
        "assignment_rule": (
            "Each assignment applies to unmatched items where both bigram words appear as "
            "letter-tokens in the title, OR both in the subtitle (not split across title vs "
            "subtitle). Items already in item_matches are not reassigned. Bigrams in "
            "bigram_assignments or unknown_bigrams are skipped on resume."
        ),
        "bigram_assignments": assignments,
        "item_matches": matches,
        "unknown_bigrams": unknown_bigrams,
        "last_saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="Path to unmatched_and_keywords.json",
    )
    parser.add_argument(
        "--resume-from",
        metavar="PATH",
        help="Continue this 1.3 manual JSON (overrides auto-resume).",
    )
    parser.add_argument(
        "--fresh-run",
        action="store_true",
        help="Always start a new output file; do not auto-resume from a matching prior file.",
    )
    parser.add_argument(
        "--force-mismatch",
        action="store_true",
        help="Allow --resume-from even when unmatched_keywords_source does not match --input.",
    )
    parser.add_argument(
        "--min-items",
        type=int,
        default=MIN_ITEMS_PER_BIGRAM,
        metavar="N",
        help=f"Minimum unmatched items per bigram to include (default: {MIN_ITEMS_PER_BIGRAM}).",
    )
    parser.add_argument(
        "--auto-random",
        action="store_true",
        help="Speed-run mode: randomly assign every eligible bigram to a leaf without prompting.",
    )
    args = parser.parse_args()

    if args.resume_from and args.fresh_run:
        raise SystemExit("Use either --resume-from or --fresh-run, not both.")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    if args.input:
        in_path = Path(args.input).expanduser().resolve()
    elif args.auto_random:
        # Speed-run: auto-select the most recent file without prompting
        _latest = newest_unmatched_keywords_file()
        if not _latest:
            raise SystemExit("No unmatched_and_keywords.json found — run step 2.2 first.")
        print(f"[auto-random] Using most recent input: {_latest.relative_to(ROOT)}")
        in_path = _latest
    else:
        in_path = pick_input_interactive()
    data = json.loads(in_path.read_text(encoding="utf-8"))
    unmatched_items = data.get("unmatched_items") or []
    if not isinstance(unmatched_items, list):
        raise SystemExit("Expected unmatched_items list in input JSON.")

    kws = data.get("unmatched_word_frequencies") or []
    if not isinstance(kws, list):
        raise SystemExit("Expected unmatched_word_frequencies list in input JSON.")

    order_low, display_for_lower = vocabulary_from_frequency_rows(kws)
    if len(order_low) < 2:
        raise SystemExit("Need at least two distinct unmatched words to form bigrams.")

    _, _, bigram_rows = compile_bigrams_from_unmatched(
        unmatched_items,
        order_low,
        display_for_lower,
        max(1, int(args.min_items)),
    )

    with TAXONOMY_PATH.open("r", encoding="utf-8") as f:
        categories = json.load(f)
    raw_rows = collect_leaf_rows(categories)
    other_rows = collect_other_bucket_leaves(raw_rows)
    search_rows = [r for r in raw_rows if not is_catch_all_bucket_slug(r.get("leaf_slug"))]

    assignments: List[dict] = []
    matches: List[dict] = []
    unknown_bigrams: List[dict] = []
    done_pairs: Set[Tuple[str, str]] = set()
    already_matched_ids: Set[str] = set()
    out_path: Path

    def pair_in_unknown_list(pair: Tuple[str, str], lst: List[dict]) -> bool:
        for u in lst:
            pls = u.get("pair_lower_sorted")
            if isinstance(pls, list) and len(pls) == 2:
                k = pair_key_from_lower_sorted(str(pls[0]), str(pls[1]))
                if k == pair:
                    return True
        return False

    resume_p: Path | None = None

    if args.resume_from:
        resume_p = Path(args.resume_from).expanduser().resolve()
        assignments, matches, unknown_bigrams, done_pairs, already_matched_ids = load_resumed_state(
            resume_p,
            in_path,
            display_for_lower,
            args.force_mismatch,
        )
        out_path = resume_p
        try:
            rshow = str(resume_p.relative_to(ROOT))
        except ValueError:
            rshow = str(resume_p)
        print(
            f"\nResuming (--resume-from) from {rshow}: "
            f"{len(assignments)} bigram assignment(s), "
            f"{len(unknown_bigrams)} unknown bigram(s), {len(matches)} item row(s), "
            f"{len(done_pairs)} bigram(s) to skip."
        )
    elif args.fresh_run:
        out_path = OUTDIR / f"1.3-manual_bigram_matches_{timestamp()}.json"
        try:
            oshow = str(out_path.relative_to(ROOT))
        except ValueError:
            oshow = str(out_path)
        print(f"\nStarting new manual file (--fresh-run): {oshow}")
    else:
        found = find_latest_manual_for_source(OUTDIR, in_path)
        if found is not None:
            with _Spinner("Reading previous session"):
                assignments, matches, unknown_bigrams, done_pairs, already_matched_ids = load_resumed_state(
                    found,
                    in_path,
                    display_for_lower,
                    args.force_mismatch,
                )
            total_items = sum(
                1 for it in unmatched_items
                if isinstance(it, dict) and isinstance(it.get("id"), str) and it["id"]
            )
            nu = total_items - len(already_matched_ids)
            nb = len(bigram_rows) - len(done_pairs)
            try:
                fshow = str(found.relative_to(ROOT))
            except ValueError:
                fshow = str(found)
            print(f"\nPrevious session: {fshow}")
            if yn_prompt(
                f"Resume last session? {nu} item(s) still uncategorized, "
                f"{nb} bigram(s) remaining.",
                default=True,
            ):
                resume_p = found
                out_path = resume_p
                try:
                    rshow = str(resume_p.relative_to(ROOT))
                except ValueError:
                    rshow = str(resume_p)
                print(
                    f"\nResuming from {rshow}: "
                    f"{len(assignments)} bigram assignment(s), "
                    f"{len(unknown_bigrams)} unknown bigram(s), {len(matches)} item row(s), "
                    f"{len(done_pairs)} bigram(s) already decided."
                )
            else:
                assignments, matches, unknown_bigrams = [], [], []
                done_pairs, already_matched_ids = set(), set()
                out_path = OUTDIR / f"1.3-manual_bigram_matches_{timestamp()}.json"
                try:
                    oshow = str(out_path.relative_to(ROOT))
                except ValueError:
                    oshow = str(out_path)
                print(f"\nStarting new manual file: {oshow}")
        else:
            out_path = OUTDIR / f"1.3-manual_bigram_matches_{timestamp()}.json"
            try:
                oshow = str(out_path.relative_to(ROOT))
            except ValueError:
                oshow = str(out_path)
            print(
                f"\nStarting new manual file: {oshow} "
                f"(no prior step-2.3 output for this 2.2 split JSON)."
            )

    print(
        f"\nVocabulary: {len(order_low)} word(s) → {len(bigram_rows)} bigram(s) "
        f"(>= {args.min_items} item(s), same-field title/subtitle). "
        f"{len(unmatched_items)} unmatched item(s), {len(search_rows)} leaf categories for search "
        f"({len(other_rows)} catch-all bucket(s) via [o] only)."
    )

    # ── Auto-random speed-run mode ─────────────────────────────────────────────
    if args.auto_random:
        if not search_rows:
            raise SystemExit("No leaf categories found in taxonomy — cannot auto-assign.")
        total_assigned = 0
        for row in bigram_rows:
            if not isinstance(row, dict):
                continue
            pair_lower_sorted = row.get("pair_lower_sorted")
            if not isinstance(pair_lower_sorted, list) or len(pair_lower_sorted) != 2:
                continue
            pk = pair_key_from_lower_sorted(str(pair_lower_sorted[0]), str(pair_lower_sorted[1]))
            if pk in done_pairs:
                continue
            bigram_display = row.get("bigram")
            if not isinstance(bigram_display, list) or len(bigram_display) != 2:
                bigram_display = [display_for_lower.get(pk[0], pk[0]), display_for_lower.get(pk[1], pk[1])]
            hit_items = [
                it for it in unmatched_items
                if isinstance(it, dict)
                and item_matches_bigram_same_field(it, list(pk))
                and isinstance(it.get("id"), str)
                and it["id"] not in already_matched_ids
            ]
            if len(hit_items) < args.min_items:
                done_pairs.add(pk)
                continue
            chosen = random.choice(search_rows)
            assign_bigram_to_leaf(
                chosen,
                hit_items=hit_items,
                bigram_display=list(bigram_display),
                pk=pk,
                out_path=out_path,
                in_path=in_path,
                assignments=assignments,
                matches=matches,
                unknown_bigrams=unknown_bigrams,
                already_matched_ids=already_matched_ids,
                done_pairs=done_pairs,
            )
            total_assigned += len(hit_items)
        write_manual_snapshot(out_path, in_path, assignments, matches, unknown_bigrams)
        print(
            f"[auto-random] Assigned {len(assignments)} bigram(s), "
            f"{total_assigned} item(s) → {out_path.relative_to(ROOT)}"
        )
        return

    print(
        "Per bigram: [Enter]=skip | [o]=Other | [x]=mark unknown | [i]=insert category | "
        "[p]=preview (→ [s] Google Images or type substring) | [c]=copy | else substring=search leaves.\n"
        "After leaf list: [#] pick | [o] Other | [i] insert | [s] Google Images | text=narrow | "
        "n / n <text>=new search | [Enter]=skip.\n"
    )

    # {"type": "assign", "chosen": <leaf_row>} | {"type": "unknown"} | {"type": "skip"}
    last_action: dict | None = None

    for ki, row in enumerate(bigram_rows, start=1):
        if not isinstance(row, dict):
            continue
        pair_lower_sorted = row.get("pair_lower_sorted")
        if not isinstance(pair_lower_sorted, list) or len(pair_lower_sorted) != 2:
            continue
        pk = pair_key_from_lower_sorted(str(pair_lower_sorted[0]), str(pair_lower_sorted[1]))
        if pk in done_pairs:
            # Decided in a previous session (loaded from the resume file). Silent skip.
            bg = row.get("bigram")
            label = bg if isinstance(bg, list) else list(pk)
            print("-" * 72)
            print(
                f"[{ki}/{len(bigram_rows)}] bigram={label!r}  — skipped "
                "(already assigned or marked unknown in resume file)."
            )
            continue

        bigram_display = row.get("bigram")
        if not isinstance(bigram_display, list) or len(bigram_display) != 2:
            bigram_display = [display_for_lower.get(pk[0], pk[0]), display_for_lower.get(pk[1], pk[1])]

        # items≈N is the count from the original bigram file (step 2.1) — may be higher than
        # the live count below if earlier bigrams in this session already claimed some of those items.
        ic = row.get("unmatched_item_count", "?")

        print("-" * 72)
        print(f"[{ki}/{len(bigram_rows)}] bigram={bigram_display!r}  items≈{ic}")

        # Live count: items that still match this bigram AND haven't been assigned yet this session.
        hit_items = [
            it
            for it in unmatched_items
            if isinstance(it, dict)
            and item_matches_bigram_same_field(it, list(pk))
            and isinstance(it.get("id"), str)
            and it["id"] not in already_matched_ids
        ]

        if len(hit_items) < MIN_ITEMS_PER_BIGRAM:
            reason = (
                "0 eligible items; all claimed by earlier assignments this session"
                if not hit_items
                else f"{len(hit_items)} eligible item(s); below minimum threshold of {MIN_ITEMS_PER_BIGRAM}"
            )
            print(f"  — auto-skipped ({reason})")
            done_pairs.add(pk)
            continue

        skip_bigram = False
        advance_bigram = False

        first_search: str | None = None
        while first_search is None:
            r_hint = ""
            if last_action is not None:
                if last_action["type"] == "assign":
                    r_hint = f" | [r] repeat (→ {last_action['chosen']['leaf_path']})"
                elif last_action["type"] == "unknown":
                    r_hint = " | [r] repeat (unknown)"
                else:
                    r_hint = " | [r] repeat (skip)"
            pre = input(
                "  [Enter] skip | [o] Other | [x] unknown | [i] insert category | "
                f"[p] preview | [c] copy{r_hint} | or substring to search leaves: "
            ).strip()
            if not pre:
                last_action = {"type": "skip"}
                first_search = ""
            elif pre.lower() == "c":
                bigram_text = " ".join(str(w) for w in bigram_display)
                if copy_to_clipboard(bigram_text):
                    print(f"  Copied: {bigram_text!r}")
                else:
                    print(f"  (Clipboard unavailable.) Bigram: {bigram_text!r}")
                continue
            elif pre.lower() == "r":
                if last_action is None:
                    print("  No previous action to repeat.")
                    continue
                lact = last_action
                la_type = lact["type"]
                if la_type == "assign":
                    la_lp = lact["chosen"]["leaf_path"]
                    confirm = input(f"  Repeat: assign → {la_lp}? [y/N] ").strip().lower()
                else:
                    confirm = input(f"  Repeat: {la_type}? [y/N] ").strip().lower()
                if confirm != "y":
                    continue
                if la_type == "assign":
                    assign_bigram_to_leaf(
                        lact["chosen"],
                        hit_items=hit_items,
                        bigram_display=list(bigram_display),
                        pk=pk,
                        out_path=out_path,
                        in_path=in_path,
                        assignments=assignments,
                        matches=matches,
                        unknown_bigrams=unknown_bigrams,
                        already_matched_ids=already_matched_ids,
                        done_pairs=done_pairs,
                    )
                    print(f"  Assigned {len(hit_items)} item(s) → {la_lp} (saved)")
                    print_session_progress(unmatched_items, bigram_rows, done_pairs, already_matched_ids)
                    first_search = "__assigned__"
                elif la_type == "unknown":
                    if not pair_in_unknown_list(pk, unknown_bigrams):
                        unknown_bigrams.append(
                            {"bigram": list(bigram_display), "pair_lower_sorted": [pk[0], pk[1]]}
                        )
                    done_pairs.add(pk)
                    write_manual_snapshot(out_path, in_path, assignments, matches, unknown_bigrams)
                    print(f"  Marked unknown (saved) [repeat]")
                    print_session_progress(unmatched_items, bigram_rows, done_pairs, already_matched_ids)
                    first_search = "__unknown__"
                else:
                    last_action = {"type": "skip"}
                    first_search = ""
            elif pre.lower() == "i":
                new_leaf = interact_insert_new_category(categories, TAXONOMY_PATH)
                if new_leaf is not None:
                    raw_rows[:] = collect_leaf_rows(categories)
                    other_rows[:] = collect_other_bucket_leaves(raw_rows)
                    search_rows[:] = [r for r in raw_rows if not is_catch_all_bucket_slug(r.get("leaf_slug"))]
                    print(f"  New leaf available: {new_leaf['leaf_path']}")
                    if yn_prompt(f"  Assign this bigram to '{new_leaf['leaf_path']}'?", default=True):
                        assign_bigram_to_leaf(
                            new_leaf,
                            hit_items=hit_items,
                            bigram_display=list(bigram_display),
                            pk=pk,
                            out_path=out_path,
                            in_path=in_path,
                            assignments=assignments,
                            matches=matches,
                            unknown_bigrams=unknown_bigrams,
                            already_matched_ids=already_matched_ids,
                            done_pairs=done_pairs,
                            extra_assignment={"inserted_by_user": True},
                        )
                        print(f"  Assigned {len(hit_items)} item(s) → {new_leaf['leaf_path']} (saved)")
                        print_session_progress(unmatched_items, bigram_rows, done_pairs, already_matched_ids)
                        last_action = {"type": "assign", "chosen": new_leaf}
                        first_search = "__assigned__"
                continue
            elif pre.lower() == "o":
                kind, chosen_other = interact_pick_other_leaf(other_rows)
                if kind == "skip":
                    first_search = ""
                elif kind == "unknown":
                    if not pair_in_unknown_list(pk, unknown_bigrams):
                        unknown_bigrams.append(
                            {"bigram": list(bigram_display), "pair_lower_sorted": [pk[0], pk[1]]}
                        )
                    done_pairs.add(pk)
                    write_manual_snapshot(out_path, in_path, assignments, matches, unknown_bigrams)
                    print(
                        f"  Marked unknown (saved); {len(hit_items)} item(s) still unmatched "
                        "for this bigram."
                    )
                    print_session_progress(unmatched_items, bigram_rows, done_pairs, already_matched_ids)
                    last_action = {"type": "unknown"}
                    first_search = "__unknown__"
                elif kind == "assign":
                    assert chosen_other is not None
                    assign_bigram_to_leaf(
                        chosen_other,
                        hit_items=hit_items,
                        bigram_display=list(bigram_display),
                        pk=pk,
                        out_path=out_path,
                        in_path=in_path,
                        assignments=assignments,
                        matches=matches,
                        unknown_bigrams=unknown_bigrams,
                        already_matched_ids=already_matched_ids,
                        done_pairs=done_pairs,
                        extra_assignment={"picked_from_other_menu": True},
                    )
                    print(
                        f"  Assigned {len(hit_items)} item(s) → {chosen_other['leaf_path']} "
                        "(saved) [Other bucket]"
                    )
                    print_session_progress(unmatched_items, bigram_rows, done_pairs, already_matched_ids)
                    last_action = {"type": "assign", "chosen": chosen_other}
                    first_search = "__assigned__"
            elif pre.lower() == "x":
                if not pair_in_unknown_list(pk, unknown_bigrams):
                    unknown_bigrams.append(
                        {"bigram": list(bigram_display), "pair_lower_sorted": [pk[0], pk[1]]}
                    )
                done_pairs.add(pk)
                write_manual_snapshot(out_path, in_path, assignments, matches, unknown_bigrams)
                print(f"  Marked unknown (saved); {len(hit_items)} item(s) still unmatched for this bigram.")
                print_session_progress(unmatched_items, bigram_rows, done_pairs, already_matched_ids)
                last_action = {"type": "unknown"}
                first_search = "__unknown__"
            elif pre.lower() == "p":
                if not hit_items:
                    print("  (No eligible items for this bigram.)")
                    continue
                n_show = min(PREVIEW_ITEMS_MAX, len(hit_items))
                print(f"  Items ({n_show} of {len(hit_items)}):")
                for j, it in enumerate(hit_items[:PREVIEW_ITEMS_MAX], start=1):
                    tid = it.get("id") or ""
                    tl = (it.get("title") or "").strip()
                    st = (it.get("subtitle") or "").strip()
                    sub = f" | {st}" if st else ""
                    print(f"    {j:2}. {tid}  {tl}{sub}")
                while True:
                    p_sel = input(
                        "  [s] Google Images | or type category substring to search: "
                    ).strip()
                    if not p_sel:
                        break  # back to main prompt
                    if p_sel.lower() == "s":
                        raw_pick = input("  Item # to open in Google Images: ").strip()
                        if raw_pick.isdigit():
                            idx = int(raw_pick) - 1
                            picked = hit_items[idx] if 0 <= idx < len(hit_items) else None
                            if isinstance(picked, dict):
                                tl = (picked.get("title") or "").strip()
                                st = (picked.get("subtitle") or "").strip()
                                search_str = f"{tl} | {st}" if tl and st else tl or st or ""
                            else:
                                search_str = " ".join(str(w) for w in bigram_display)
                            if copy_to_clipboard(search_str):
                                print(f"  Copied: {search_str!r}")
                            if open_google_images_in_chrome(search_str):
                                print("  Opened Google Images in Chrome.")
                            else:
                                print(f"  URL: https://www.google.com/search?q={quote_plus(search_str)}&tbm=isch")
                        else:
                            print("  Enter a valid item number.")
                        continue  # stay in preview loop
                    else:
                        first_search = p_sel
                        break  # exit preview loop → jump to leaf search
                continue  # re-enters outer while if first_search still None
            else:
                first_search = pre

        if first_search == "__unknown__":
            continue
        if first_search == "__assigned__":
            continue
        if not first_search:
            print_session_progress(unmatched_items, bigram_rows, done_pairs, already_matched_ids)
            continue

        hits = filter_leaves(search_rows, first_search)
        while True:
            if skip_bigram:
                break
            if advance_bigram:
                break
            if not hits:
                print("  No leaf matches. New search (blank = skip bigram).")
                nq = ""
                while True:
                    nq = input(
                        "  Search ([o] Other) or substring; blank=skip bigram: "
                    ).strip()
                    if nq.lower() == "o":
                        kind, chosen_other = interact_pick_other_leaf(other_rows)
                        if kind == "skip":
                            skip_bigram = True
                            break
                        if kind == "unknown":
                            if not pair_in_unknown_list(pk, unknown_bigrams):
                                unknown_bigrams.append(
                                    {
                                        "bigram": list(bigram_display),
                                        "pair_lower_sorted": [pk[0], pk[1]],
                                    }
                                )
                            done_pairs.add(pk)
                            write_manual_snapshot(
                                out_path, in_path, assignments, matches, unknown_bigrams
                            )
                            print(
                                f"  Marked unknown (saved); {len(hit_items)} item(s) still unmatched."
                            )
                            advance_bigram = True
                            break
                        assert chosen_other is not None
                        assign_bigram_to_leaf(
                            chosen_other,
                            hit_items=hit_items,
                            bigram_display=list(bigram_display),
                            pk=pk,
                            out_path=out_path,
                            in_path=in_path,
                            assignments=assignments,
                            matches=matches,
                            unknown_bigrams=unknown_bigrams,
                            already_matched_ids=already_matched_ids,
                            done_pairs=done_pairs,
                            extra_assignment={"picked_from_other_menu": True},
                        )
                        print(
                            f"  Assigned {len(hit_items)} item(s) → {chosen_other['leaf_path']} "
                            "(saved) [Other]"
                        )
                        advance_bigram = True
                        break
                    break
                if advance_bigram:
                    break
                if skip_bigram:
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
                "  [#] assign | [o] Other | [i] insert | [s] Google Images | [c] copy | [text] narrow | "
                "[n]/[n <text>] new search | [Enter] skip bigram: "
            ).strip()

            if not sel:
                break

            if sel.lower() == "i":
                new_leaf = interact_insert_new_category(categories, TAXONOMY_PATH)
                if new_leaf is not None:
                    raw_rows[:] = collect_leaf_rows(categories)
                    other_rows[:] = collect_other_bucket_leaves(raw_rows)
                    search_rows[:] = [r for r in raw_rows if not is_catch_all_bucket_slug(r.get("leaf_slug"))]
                    hits = filter_leaves(search_rows, first_search) if first_search else hits
                    print(f"  New leaf available: {new_leaf['leaf_path']}")
                    if yn_prompt(f"  Assign this bigram to '{new_leaf['leaf_path']}'?", default=True):
                        assign_bigram_to_leaf(
                            new_leaf,
                            hit_items=hit_items,
                            bigram_display=list(bigram_display),
                            pk=pk,
                            out_path=out_path,
                            in_path=in_path,
                            assignments=assignments,
                            matches=matches,
                            unknown_bigrams=unknown_bigrams,
                            already_matched_ids=already_matched_ids,
                            done_pairs=done_pairs,
                            extra_assignment={"inserted_by_user": True},
                        )
                        print(f"  Assigned {len(hit_items)} item(s) → {new_leaf['leaf_path']} (saved)")
                        print_session_progress(unmatched_items, bigram_rows, done_pairs, already_matched_ids)
                        last_action = {"type": "assign", "chosen": new_leaf}
                        advance_bigram = True
                        break
                continue

            if sel.lower() == "s":
                n_show = min(PREVIEW_ITEMS_MAX, len(hit_items))
                print(f"  Items ({n_show} of {len(hit_items)}):")
                for j, it in enumerate(hit_items[:PREVIEW_ITEMS_MAX], start=1):
                    tid = it.get("id") or ""
                    tl = (it.get("title") or "").strip()
                    st = (it.get("subtitle") or "").strip()
                    sub = f" | {st}" if st else ""
                    print(f"    {j:2}. {tid}  {tl}{sub}")
                raw_pick = input("  Item # to open in Google Images: ").strip()
                if raw_pick.isdigit():
                    idx = int(raw_pick) - 1
                    picked = hit_items[idx] if 0 <= idx < len(hit_items) else None
                    if isinstance(picked, dict):
                        tl = (picked.get("title") or "").strip()
                        st = (picked.get("subtitle") or "").strip()
                        search_str = f"{tl} | {st}" if tl and st else tl or st or ""
                    else:
                        search_str = " ".join(str(w) for w in bigram_display)
                    if copy_to_clipboard(search_str):
                        print(f"  Copied: {search_str!r}")
                    if open_google_images_in_chrome(search_str):
                        print("  Opened Google Images in Chrome.")
                    else:
                        print(f"  URL: https://www.google.com/search?q={quote_plus(search_str)}&tbm=isch")
                else:
                    print("  Enter a valid item number.")
                continue

            if sel.lower() == "c":
                bigram_text = " ".join(str(w) for w in bigram_display)
                if copy_to_clipboard(bigram_text):
                    print(f"  Copied: {bigram_text!r}")
                else:
                    print(f"  (Clipboard unavailable.) Bigram: {bigram_text!r}")
                continue

            if sel.lower() == "o":
                kind, chosen_other = interact_pick_other_leaf(other_rows)
                if kind == "skip":
                    skip_bigram = True
                    break
                if kind == "unknown":
                    if not pair_in_unknown_list(pk, unknown_bigrams):
                        unknown_bigrams.append(
                            {
                                "bigram": list(bigram_display),
                                "pair_lower_sorted": [pk[0], pk[1]],
                            }
                        )
                    done_pairs.add(pk)
                    write_manual_snapshot(out_path, in_path, assignments, matches, unknown_bigrams)
                    print(
                        f"  Marked unknown (saved); {len(hit_items)} item(s) still unmatched."
                    )
                    last_action = {"type": "unknown"}
                    advance_bigram = True
                    break
                assert chosen_other is not None
                assign_bigram_to_leaf(
                    chosen_other,
                    hit_items=hit_items,
                    bigram_display=list(bigram_display),
                    pk=pk,
                    out_path=out_path,
                    in_path=in_path,
                    assignments=assignments,
                    matches=matches,
                    unknown_bigrams=unknown_bigrams,
                    already_matched_ids=already_matched_ids,
                    done_pairs=done_pairs,
                    extra_assignment={"picked_from_other_menu": True},
                )
                print(
                    f"  Assigned {len(hit_items)} item(s) → {chosen_other['leaf_path']} (saved) [Other]"
                )
                print_session_progress(unmatched_items, bigram_rows, done_pairs, already_matched_ids)
                last_action = {"type": "assign", "chosen": chosen_other}
                advance_bigram = True
                break

            if re.fullmatch(r"\d+", sel):
                n = int(sel)
                if not (1 <= n <= len(show)):
                    print("  Out of range for the numbered list above.")
                    continue
                chosen = show[n - 1]
                assign_bigram_to_leaf(
                    chosen,
                    hit_items=hit_items,
                    bigram_display=list(bigram_display),
                    pk=pk,
                    out_path=out_path,
                    in_path=in_path,
                    assignments=assignments,
                    matches=matches,
                    unknown_bigrams=unknown_bigrams,
                    already_matched_ids=already_matched_ids,
                    done_pairs=done_pairs,
                )
                print(f"  Assigned {len(hit_items)} item(s) → {chosen['leaf_path']} (saved)")
                print_session_progress(unmatched_items, bigram_rows, done_pairs, already_matched_ids)
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

        if skip_bigram:
            print_session_progress(unmatched_items, bigram_rows, done_pairs, already_matched_ids)
            continue
        if advance_bigram:
            print_session_progress(unmatched_items, bigram_rows, done_pairs, already_matched_ids)
            continue
        print_session_progress(unmatched_items, bigram_rows, done_pairs, already_matched_ids)

    write_manual_snapshot(out_path, in_path, assignments, matches, unknown_bigrams)
    print(f"\nFinal: {out_path}")
    print(
        f"Bigram assignments: {len(assignments)}  Unknown bigrams: {len(unknown_bigrams)}  "
        f"Item rows: {len(matches)}"
    )


if __name__ == "__main__":
    main()
