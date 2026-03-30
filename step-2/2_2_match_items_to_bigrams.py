#!/usr/bin/env python3
"""
Step 2.2 — Match items against bigram mappings and output ranked category suggestions per item.

Reads:
  - ../source-files/raw-prod-items-non-deleted.json (or --items-json pool)
  - a bigram mapping JSON from step-2.1b outputs (1.1a or 1.1b), chosen interactively

Writes (timestamped) to:
  - ./outputs/1.2-bigram_sorted_items_YYYYMMDD_HHMMSS.json

Rules:
  - title is matched ONLY against mapping.title_bigrams
  - subtitle is matched ONLY against mapping.subtitle_bigrams
  - a bigram triggers if BOTH words appear in the respective string (any order, not adjacent)
  - output bigram words are preserved with original casing from the item text
  - items can have multiple categories; within each item categories are ranked by:
      matched_bigram_count desc, then max_confidence desc
  - items with no matches go to unmatched_items

Phased cascade (--cascade-mapping repeated, low depth first):
  - Each depth pass runs only on items still unmatched after prior passes.
  - Ancestor category slugs are dropped when a descendant slug is present (taxonomy path dedupe).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]  # v2/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from taxonomy_cascade import collect_slug_to_path, dedupe_category_slugs, is_catch_all_bucket_slug

ITEMS_PATH = ROOT / "source-files" / "raw-prod-items-non-deleted.json"
TAXONOMY_PATH = ROOT / "source-files" / "categories_v1.json"
def _mapping_candidate_files() -> List[Path]:
    d = ROOT / "step-2" / "outputs"
    if d.is_dir():
        return sorted(d.glob("1.1*-bigram_categories_mapping*.json"))
    return []


MAPPINGS_DIR = ROOT / "step-2" / "outputs"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

# Unmatched keyword list: drop short tokens and words that appear in too few unmatched items.
UNMATCHED_KEYWORD_MIN_CHARS = 5  # strictly more than 4 characters
UNMATCHED_KEYWORD_MIN_ITEMS = 4  # strictly more than 3 distinct unmatched items


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def tokenize_alpha_preserve(text: str) -> List[str]:
    """Letter-only tokens (A–Z runs), preserving original casing. No regex."""
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


def build_index(bigrams: List[dict]) -> Tuple[Dict[Tuple[str, str], dict], Dict[str, List[Tuple[str, str]]]]:
    """
    Build a fast lookup:
      - pair_meta: (sorted(w1,w2)) -> {w1,w2,category,confidence}
      - word_to_pairs: word -> [pair_key,...]
    """
    pair_meta: Dict[Tuple[str, str], dict] = {}
    word_to_pairs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for obj in bigrams:
        w1, w2 = obj["bigram"]
        key = tuple(sorted((w1, w2)))
        pair_meta[key] = {
            "w1": w1,
            "w2": w2,
            "category": obj["suggested_parent_category_slug"],
            "confidence": obj["confidence"],
        }

    for key in pair_meta.keys():
        a, b = key
        word_to_pairs[a].append(key)
        if b != a:
            word_to_pairs[b].append(key)

    return pair_meta, word_to_pairs


def match_side(text: str, pair_meta: Dict[Tuple[str, str], dict], word_to_pairs: Dict[str, List[Tuple[str, str]]], source: str) -> Dict[str, List[dict]]:
    tokens = tokenize_alpha_preserve(text)
    if not tokens:
        return {}

    lower_set = set()
    first_seen: Dict[str, str] = {}
    for tok in tokens:
        lo = tok.lower()
        lower_set.add(lo)
        if lo not in first_seen:
            first_seen[lo] = tok  # preserve original casing for output

    candidates = set()
    for w in lower_set:
        for pk in word_to_pairs.get(w, []):
            candidates.add(pk)

    cat_triggers: Dict[str, List[dict]] = defaultdict(list)
    for pk in candidates:
        meta = pair_meta.get(pk)
        if not meta:
            continue
        w1 = meta["w1"]
        w2 = meta["w2"]
        if w1 in lower_set and w2 in lower_set:
            cat_triggers[meta["category"]].append(
                {
                    "bigram": [first_seen[w1], first_seen[w2]],
                    "confidence": meta["confidence"],
                    "source": source,
                }
            )

    return cat_triggers


def match_item_triggers(
    item: dict,
    cross_side: bool,
    pair_meta_title: Dict[Tuple[str, str], dict],
    word_to_pairs_title: Dict[str, List[Tuple[str, str]]],
    pair_meta_sub: Dict[Tuple[str, str], dict],
    word_to_pairs_sub: Dict[str, List[Tuple[str, str]]],
) -> Dict[str, List[dict]]:
    title = item.get("title") or ""
    subtitle = item.get("subtitle") or ""
    combined_text = f"{title} {subtitle}".strip()
    cat_to_triggers: Dict[str, List[dict]] = defaultdict(list)
    if cross_side:
        for cat, triggers in match_side(combined_text, pair_meta_title, word_to_pairs_title, "cross").items():
            cat_to_triggers[cat].extend(triggers)
        for cat, triggers in match_side(combined_text, pair_meta_sub, word_to_pairs_sub, "cross").items():
            cat_to_triggers[cat].extend(triggers)
    else:
        for cat, triggers in match_side(title, pair_meta_title, word_to_pairs_title, "title").items():
            cat_to_triggers[cat].extend(triggers)
        for cat, triggers in match_side(subtitle, pair_meta_sub, word_to_pairs_sub, "subtitle").items():
            cat_to_triggers[cat].extend(triggers)
    return dict(cat_to_triggers)


def build_category_rows(cat_to_triggers: Dict[str, List[dict]], slug_to_path: Dict[str, str] | None) -> tuple[List[dict], int, float]:
    category_objs: List[dict] = []
    for cat, triggers in cat_to_triggers.items():
        if is_catch_all_bucket_slug(cat):
            continue
        seen = set()
        dedup: List[dict] = []
        for t in triggers:
            k = (t["bigram"][0].lower(), t["bigram"][1].lower(), float(t["confidence"]), t["source"])
            if k in seen:
                continue
            seen.add(k)
            dedup.append(t)

        dedup.sort(key=lambda x: (-float(x["confidence"]), x["bigram"][0], x["bigram"][1], x["source"]))
        matched_count = len(dedup)
        max_cat_conf = max(float(t["confidence"]) for t in dedup) if dedup else 0.0

        category_objs.append(
            {
                "category_slug": cat,
                "matched_bigram_count": matched_count,
                "triggering_bigrams": dedup,
                "max_confidence": max_cat_conf,
            }
        )

    category_objs.sort(key=lambda x: (-x["matched_bigram_count"], -x["max_confidence"], x["category_slug"]))

    if slug_to_path:
        slugs = [c["category_slug"] for c in category_objs]
        ordered = dedupe_category_slugs(slugs, slug_to_path)
        by_slug = {c["category_slug"]: c for c in category_objs}
        category_objs = [by_slug[s] for s in ordered if s in by_slug]

    total_triggered = sum(c["matched_bigram_count"] for c in category_objs)
    max_conf = max((c["max_confidence"] for c in category_objs), default=0.0)
    return category_objs, total_triggered, max_conf


def write_split_artifacts(
    matched_items: List[dict],
    unmatched_items: List[dict],
    split_out_dir: Path,
) -> None:
    """
    Extra artifacts for quick inspection:
      - `bigrams_combined.json`: bigram trigger stats across all matched items
      - `matched.json`: matched items only
      - `unmatched_and_keywords.json`: unmatched items + unigram frequencies from title/subtitle
    """

    split_out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Combined bigram stats from matched items
    bigram_stats: Dict[Tuple[str, str], dict] = {}

    for item in matched_items:
        item_id = item.get("id")
        for cat in item.get("categories", []) or []:
            for t in cat.get("triggering_bigrams", []) or []:
                w1, w2 = t.get("bigram") or ["", ""]
                if not w1 or not w2:
                    continue
                key = tuple(sorted((str(w1).lower(), str(w2).lower())))
                if key not in bigram_stats:
                    bigram_stats[key] = {
                        "bigram": [w1, w2],  # display (not canonical) from first seen
                        "trigger_count": 0,
                        "item_ids": set(),
                        "max_confidence": float(t.get("confidence") or 0.0),
                        "sources": set(),
                    }
                bs = bigram_stats[key]
                bs["trigger_count"] += 1
                bs["item_ids"].add(item_id)
                bs["max_confidence"] = max(bs["max_confidence"], float(t.get("confidence") or 0.0))
                bs["sources"].add(t.get("source"))

    bigrams_combined = []
    for key, bs in bigram_stats.items():
        bigrams_combined.append(
            {
                "bigram": bs["bigram"],
                "canonical_bigram_lower_sorted": [key[0], key[1]],
                "trigger_count": bs["trigger_count"],
                "matched_item_count": len(bs["item_ids"]),
                "max_confidence": bs["max_confidence"],
                "sources": sorted([s for s in bs["sources"] if s]),
            }
        )

    bigrams_combined.sort(key=lambda x: (-x["trigger_count"], -x["max_confidence"], x["canonical_bigram_lower_sorted"][0], x["canonical_bigram_lower_sorted"][1]))

    # 2) Word stats on unmatched items (title + subtitle): per-item + token counts, then filter.
    word_token_count: Dict[str, int] = defaultdict(int)
    word_item_count: Dict[str, int] = defaultdict(int)
    for it in unmatched_items:
        title = it.get("title") or ""
        subtitle = it.get("subtitle") or ""
        text = f"{title} {subtitle}".strip()
        toks = tokenize_alpha_preserve(text)
        seen_in_item: set[str] = set()
        for tok in toks:
            w = str(tok).lower()
            word_token_count[w] += 1
            seen_in_item.add(w)
        for w in seen_in_item:
            word_item_count[w] += 1

    unmatched_word_list = []
    for w, ic in word_item_count.items():
        if len(w) < UNMATCHED_KEYWORD_MIN_CHARS or ic < UNMATCHED_KEYWORD_MIN_ITEMS:
            continue
        unmatched_word_list.append(
            {
                "word": w,
                "unmatched_item_count": ic,
                "token_count": word_token_count[w],
            }
        )
    unmatched_word_list.sort(key=lambda x: (-x["unmatched_item_count"], -x["token_count"], x["word"]))

    # 3) Write files
    (split_out_dir / "bigrams_combined.json").write_text(
        json.dumps(bigrams_combined, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (split_out_dir / "matched.json").write_text(
        json.dumps(matched_items, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (split_out_dir / "unmatched_and_keywords.json").write_text(
        json.dumps(
            {
                "unmatched_items": unmatched_items,
                "unmatched_word_frequencies": unmatched_word_list,
                "unmatched_word_filters": {
                    "min_characters": UNMATCHED_KEYWORD_MIN_CHARS,
                    "note": "Words shorter than this length are excluded (i.e. keep len(word) >= min_characters).",
                    "min_distinct_unmatched_items": UNMATCHED_KEYWORD_MIN_ITEMS,
                    "note_items": "Words appearing in fewer distinct unmatched items are excluded (i.e. keep item_count >= min).",
                },
                "unmatched_word_source": (
                    "tokenize(title + ' ' + subtitle) letter-only runs, case-insensitive; "
                    f"keywords listed only if len(word) >= {UNMATCHED_KEYWORD_MIN_CHARS} and "
                    f"word appears in >= {UNMATCHED_KEYWORD_MIN_ITEMS} distinct unmatched items."
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def pick_mapping_file_interactive() -> Path:
    candidates = sorted(_mapping_candidate_files(), key=lambda p: p.stat().st_mtime)
    latest = candidates[-1] if candidates else None

    print("Choose bigram mapping file:")
    if latest:
        print(f"  1) Use most recent: {latest.relative_to(ROOT)}")
    else:
        print("  1) (No files under step-2/outputs/)")
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


def file_fingerprint(path: Path) -> str:
    """
    Lightweight fingerprint for resume correctness.
    Uses file path + size + mtime.
    """
    st = path.stat()
    raw = f"{path.as_posix()}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def checkpoint_path(fingerprint: str) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"step12_checkpoint_{fingerprint}.json"


def maybe_load_checkpoint(path: Path, expected_fingerprint: str, no_resume: bool) -> tuple[int, list[dict], list[dict]]:
    if no_resume or not path.exists():
        return 0, [], []

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("fingerprint") != expected_fingerprint:
        return 0, [], []

    start_index = int(data.get("next_index", 0))
    matched_items = data.get("matched_items", [])
    unmatched_items = data.get("unmatched_items")
    if unmatched_items is None:
        unmatched_items = data.get("unmatched_words", [])
    if not isinstance(matched_items, list) or not isinstance(unmatched_items, list):
        return 0, [], []

    return start_index, matched_items, unmatched_items


def save_checkpoint(
    path: Path,
    fingerprint: str,
    next_index: int,
    matched_items: list[dict],
    unmatched_items: list[dict],
) -> None:
    tmp = path.with_suffix(".tmp")
    payload = {
        "fingerprint": fingerprint,
        "next_index": next_index,
        "matched_items": matched_items,
        "unmatched_items": unmatched_items,
    }
    tmp.write_text(json.dumps(payload, indent=0, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def run_single_mapping(args: argparse.Namespace) -> None:
    if args.mapping:
        mapping_path = Path(args.mapping).expanduser().resolve()
        if not mapping_path.exists():
            raise SystemExit(f"Mapping file not found: {mapping_path}")
    else:
        mapping_path = pick_mapping_file_interactive()
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    title_bigrams = mapping.get("title_bigrams", [])
    subtitle_bigrams = mapping.get("subtitle_bigrams", [])

    pair_meta_title, word_to_pairs_title = build_index(title_bigrams)
    pair_meta_sub, word_to_pairs_sub = build_index(subtitle_bigrams)

    items, items_fp = _load_items_list()

    fp = (
        f"items={file_fingerprint(items_fp)}"
        f"|mapping={file_fingerprint(mapping_path)}"
        f"|cross_side={not bool(args.strict_sides)}"
    )
    ckpt = checkpoint_path(fp)
    start_index, matched_items, unmatched_items = maybe_load_checkpoint(
        ckpt, expected_fingerprint=fp, no_resume=args.no_resume
    )
    matched_items = list(matched_items)
    unmatched_items = list(unmatched_items)

    print(
        f"Resume start_index={start_index} matched_so_far={len(matched_items)} unmatched_so_far={len(unmatched_items)}"
    )

    cross_side = not args.strict_sides

    n_items = len(items)
    use_pbar = not args.no_progress and sys.stderr.isatty() and n_items > 0
    idx_iter = range(start_index, n_items)
    pbar: tqdm | None = None
    if use_pbar:
        pbar = tqdm(
            idx_iter,
            total=n_items,
            initial=start_index,
            desc="2.2 match items→bigrams",
            unit="item",
            file=sys.stderr,
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
        )
    try:
        for idx in pbar or idx_iter:
            item = items[idx]
            item_id = item.get("id")
            title = item.get("title") or ""
            subtitle = item.get("subtitle") or ""

            cat_to_triggers = match_item_triggers(
                item, cross_side, pair_meta_title, word_to_pairs_title, pair_meta_sub, word_to_pairs_sub
            )

            if not cat_to_triggers:
                unmatched_items.append({"id": item_id, "title": title, "subtitle": subtitle})
                if (idx + 1) % args.checkpoint_every == 0:
                    save_checkpoint(
                        ckpt,
                        fp,
                        next_index=idx + 1,
                        matched_items=matched_items,
                        unmatched_items=unmatched_items,
                    )
                    if pbar:
                        pbar.set_postfix_str(
                            f"ckpt@{idx + 1} matched={len(matched_items)} unmatched={len(unmatched_items)}",
                            refresh=False,
                        )
                    else:
                        print(f"Checkpoint saved at idx={idx + 1}/{n_items}")
                elif pbar:
                    pbar.set_postfix_str(
                        f"matched={len(matched_items)} unmatched={len(unmatched_items)} last=no_bigram",
                        refresh=False,
                    )
                continue

            category_objs, total_triggered, max_conf = build_category_rows(cat_to_triggers, None)

            matched_items.append(
                {
                    "id": item_id,
                    "title": title,
                    "subtitle": subtitle,
                    "total_triggered_bigrams": total_triggered,
                    "max_confidence": max_conf,
                    "categories": category_objs,
                }
            )

            if (idx + 1) % args.checkpoint_every == 0:
                save_checkpoint(ckpt, fp, next_index=idx + 1, matched_items=matched_items, unmatched_items=unmatched_items)
                if pbar:
                    pbar.set_postfix_str(
                        f"ckpt@{idx + 1} matched={len(matched_items)} unmatched={len(unmatched_items)}",
                        refresh=False,
                    )
                else:
                    print(f"Checkpoint saved at idx={idx + 1}/{n_items}")
            elif pbar:
                pbar.set_postfix_str(
                    f"matched={len(matched_items)} unmatched={len(unmatched_items)} last=ok",
                    refresh=False,
                )
    finally:
        if pbar:
            pbar.close()

    matched_items.sort(key=lambda x: (-x["total_triggered_bigrams"], -x["max_confidence"], x["id"]))

    payload = {
        "version": "1.2",
        "phased_cascade": False,
        "bigram_mapping_file": mapping_path.name,
        "items_file": str(items_fp.relative_to(ROOT)),
        "match_rules": [
            "Item matches a bigram if BOTH bigram words appear in the side text (title or subtitle), any order, not necessarily adjacent.",
            "Matching is token-based on letter-only runs; comparisons are case-insensitive.",
            "Output bigram tokens are written using the exact casing as they appear in the item.",
            "Categories are ranked per item by number of triggering bigrams, then max confidence.",
        ],
        "matched_items": matched_items,
        "unmatched_items": unmatched_items,
    }

    ts = timestamp()
    out_path = OUTPUT_DIR / f"1.2-bigram_sorted_items_{ts}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"matched_items: {len(matched_items)}")
    print(f"unmatched_items: {len(unmatched_items)}")

    split_dir = OUTPUT_DIR / f"1.2_split_{ts}"
    write_split_artifacts(matched_items, unmatched_items, split_dir)

    try:
        if ckpt.exists():
            ckpt.unlink()
    except OSError:
        pass


def run_phased_cascade(args: argparse.Namespace, cascade_paths: List[str]) -> None:
    paths = [Path(p).expanduser().resolve() for p in cascade_paths]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"Mapping file not found: {p}")

    if not args.no_resume:
        print(
            "Phased cascade: checkpoint resume is not supported — each run processes all items from scratch.",
            file=sys.stderr,
        )

    with TAXONOMY_PATH.open("r", encoding="utf-8") as f:
        categories = json.load(f)
    try:
        slug_to_path = collect_slug_to_path(categories)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    items, items_fp = _load_items_list()

    cross_side = not args.strict_sides
    print(
        f"Phased cascade: {len(paths)} mapping file(s), {len(items)} items (source={items_fp.name}), "
        f"cross-side matching={'on' if cross_side else 'off'}.",
        flush=True,
    )

    fp_parts = ["items=" + file_fingerprint(items_fp), f"cross_side={cross_side}", "cascade=1"]
    for i, p in enumerate(paths):
        fp_parts.append(f"d{i}={file_fingerprint(p)}")
    fp = "|".join(fp_parts)

    pending = set(range(len(items)))
    matched_triggers: Dict[int, Dict[str, List[dict]]] = {}

    n_maps = len(paths)
    for phase_i, mp in enumerate(paths, start=1):
        t_phase0 = time.perf_counter()
        mapping = json.loads(mp.read_text(encoding="utf-8"))
        tax_depth = mapping.get("taxonomy_depth")
        depth_label = f"taxonomy_depth={tax_depth}" if tax_depth is not None else "taxonomy_depth=?"
        title_bigrams = mapping.get("title_bigrams", [])
        subtitle_bigrams = mapping.get("subtitle_bigrams", [])
        pair_meta_title, word_to_pairs_title = build_index(title_bigrams)
        pair_meta_sub, word_to_pairs_sub = build_index(subtitle_bigrams)

        pending_list = sorted(pending)
        n_pending = len(pending_list)
        print(
            f"  Phase {phase_i}/{n_maps}: {mp.name} ({depth_label}) — "
            f"scanning {n_pending} pending item(s)…",
            flush=True,
        )

        use_pbar = not args.no_progress and sys.stderr.isatty() and n_pending > 0
        short_name = mp.name if len(mp.name) <= 44 else mp.name[:41] + "..."
        pbar = (
            tqdm(
                total=n_pending,
                desc=f"2.2 cascade [{phase_i}/{n_maps}] {short_name}",
                unit="item",
                file=sys.stderr,
                dynamic_ncols=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
            )
            if use_pbar
            else None
        )

        to_remove: List[int] = []
        try:
            for idx in pending_list:
                item = items[idx]
                cat_to_triggers = match_item_triggers(
                    item, cross_side, pair_meta_title, word_to_pairs_title, pair_meta_sub, word_to_pairs_sub
                )
                if cat_to_triggers:
                    matched_triggers[idx] = cat_to_triggers
                    to_remove.append(idx)
                if pbar:
                    pbar.update(1)
                    if pbar.n % 200 == 0 or pbar.n == n_pending:
                        pbar.set_postfix_str(
                            f"new_hits={len(to_remove)} scanned={pbar.n}/{n_pending}",
                            refresh=False,
                        )
        finally:
            if pbar:
                pbar.close()

        for idx in to_remove:
            pending.discard(idx)

        dt = time.perf_counter() - t_phase0
        print(
            f"  Phase {phase_i} done: +{len(to_remove)} matched this round, "
            f"{len(pending)} still pending ({dt:.1f}s).",
            flush=True,
        )

    print("Building category rows, deduping paths, and writing outputs…", flush=True)

    matched_items: List[dict] = []
    for idx in sorted(matched_triggers.keys()):
        item = items[idx]
        item_id = item.get("id")
        title = item.get("title") or ""
        subtitle = item.get("subtitle") or ""
        category_objs, total_triggered, max_conf = build_category_rows(matched_triggers[idx], slug_to_path)
        matched_items.append(
            {
                "id": item_id,
                "title": title,
                "subtitle": subtitle,
                "total_triggered_bigrams": total_triggered,
                "max_confidence": max_conf,
                "categories": category_objs,
            }
        )

    unmatched_items: List[dict] = []
    for idx in sorted(pending):
        it = items[idx]
        unmatched_items.append({"id": it.get("id"), "title": it.get("title") or "", "subtitle": it.get("subtitle") or ""})

    matched_items.sort(key=lambda x: (-x["total_triggered_bigrams"], -x["max_confidence"], x["id"]))

    payload = {
        "version": "1.2",
        "phased_cascade": True,
        "bigram_mapping_files": [p.name for p in paths],
        "taxonomy_categories_file": TAXONOMY_PATH.name,
        "items_file": str(items_fp.relative_to(ROOT)),
        "cascade_fingerprint": fp,
        "match_rules": [
            "Phased cascade: each mapping is applied in order; only items still unmatched are considered at the next depth.",
            "Within an item, ancestor category slugs are removed when a descendant slug is present (taxonomy path prefix rule).",
            "Item matches a bigram if BOTH words appear in the side text (or combined text in cross-side mode), any order.",
        ],
        "matched_items": matched_items,
        "unmatched_items": unmatched_items,
    }

    ts = timestamp()
    out_path = OUTPUT_DIR / f"1.2-bigram_sorted_items_{ts}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"matched_items: {len(matched_items)}")
    print(f"unmatched_items: {len(unmatched_items)}")

    split_dir = OUTPUT_DIR / f"1.2_split_{ts}"
    write_split_artifacts(matched_items, unmatched_items, split_dir)
    print(f"Split artifacts: {split_dir}", flush=True)


def _load_items_list() -> tuple[list[dict], Path]:
    """Return (items, path_used_for_fingerprint)."""
    args = _match_items_args
    if getattr(args, "items_json", None):
        ip = Path(args.items_json).expanduser().resolve()
        if not ip.is_file():
            raise SystemExit(f"Not found: {ip}")
        blob = json.loads(ip.read_text(encoding="utf-8"))
        if isinstance(blob, list):
            items = blob
        elif isinstance(blob, dict):
            items = blob.get("items") or blob.get("unmatched_items") or []
        else:
            items = []
        if not isinstance(items, list):
            raise SystemExit("items-json must contain a list or object with items/unmatched_items")
        return [x for x in items if isinstance(x, dict)], ip
    items = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise SystemExit(f"Expected array in {ITEMS_PATH}")
    return items, ITEMS_PATH


_match_items_args: argparse.Namespace | None = None


def main() -> None:
    global _match_items_args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--items-json",
        metavar="PATH",
        help="JSON list of items (e.g. unmatched_after_step1.json). Default: full raw-prod-items file.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume.")
    parser.add_argument("--checkpoint-every", type=int, default=300, help="Save progress every N items.")
    parser.add_argument("--mapping", help="Path to a 1.1*-bigram_categories_mapping*.json file (skips interactive selection).")
    parser.add_argument(
        "--cascade-mapping",
        action="append",
        default=None,
        metavar="PATH",
        help="Repeat for each depth-ordered mapping file (T0 first, then T1, ...). Enables phased cascade matching.",
    )
    parser.add_argument(
        "--strict-sides",
        action="store_true",
        help="If set, use original behavior: match title_bigrams only against title and subtitle_bigrams only against subtitle.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (phased cascade scans per mapping file).",
    )
    args = parser.parse_args()
    _match_items_args = args

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cascade = [p for p in (args.cascade_mapping or []) if p]
    if cascade and args.mapping:
        raise SystemExit("Use either --mapping or --cascade-mapping, not both.")
    if cascade:
        run_phased_cascade(args, cascade)
    else:
        run_single_mapping(args)


if __name__ == "__main__":
    main()
