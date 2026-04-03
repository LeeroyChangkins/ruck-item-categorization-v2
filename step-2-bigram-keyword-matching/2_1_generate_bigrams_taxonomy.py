#!/usr/bin/env python3
"""
Step 1.1a (taxonomy-based)
Generate bigram -> T1 parent category mappings (high confidence) using only:
  - 1.0 keyword list output (curated single words)
  - taxonomy text in categories_v1.json

Reads:
  - ../source-files/categories_v1.json
  - ../step-2-bigram-keyword-matching/outputs/1.0-title_subtitle_keyword_frequencies*.json (most recent by default)

Writes (timestamped) to:
  - ./outputs/1.1a-bigram_categories_mapping_YYYYMMDD_HHMMSS.json

Notes:
  - does NOT read or use item.category/subcategory
  - bigrams are formed ONLY from curated 1.0 word lists (title-only and subtitle-only separately)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]  # v2/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from taxonomy_cascade import (
    build_anchor_token_set,
    is_catch_all_bucket_slug,
    max_taxonomy_depth,
    nodes_at_depth,
)

TAXONOMY_PATH = ROOT / "source-files" / "categories_v1.json"
KEYWORDS_DIR = ROOT / "step-2-bigram-keyword-matching" / "outputs"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

SIMILARITY_THRESHOLD = 0.58
FIRST3_GATE = True
DEFAULT_MIN_CONF = 0.85


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def tokenize(s: str) -> List[str]:
    """Letters-only tokens from slug/display_name. No regex."""
    if not s:
        return []
    s = str(s).replace("_", " ").lower()
    out: List[str] = []
    cur: List[str] = []
    for ch in s:
        if ch.isalpha():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def build_anchor_token_sets_for_depth(categories: dict, depth: int) -> Dict[str, Set[str]]:
    """For each taxonomy anchor at depth, tokens from that node and all descendants."""
    out: Dict[str, Set[str]] = {}
    for slug, node in nodes_at_depth(categories, depth):
        if is_catch_all_bucket_slug(slug):
            continue
        out[slug] = build_anchor_token_set(node)
    return out


def word_to_parents(word: str, anchor_tokens: Dict[str, Set[str]]) -> Set[str]:
    w = (word or "").lower()
    if not w:
        return set()
    w3 = w[:3]
    parents: Set[str] = set()

    for parent, toks in anchor_tokens.items():
        if w in toks:
            parents.add(parent)
            continue

        best = 0.0
        for t in toks:
            if FIRST3_GATE and len(t) >= 3 and t[:3] != w3:
                continue
            r = similarity(w, t)
            if r > best:
                best = r
        if best >= SIMILARITY_THRESHOLD:
            parents.add(parent)

    return parents


def choose_confidence(parents_a: Set[str], parents_b: Set[str]) -> float | None:
    if len(parents_a) == 1 and len(parents_b) == 1:
        return 1.0
    if len(parents_a) == 1 or len(parents_b) == 1:
        return 0.85
    if parents_a and parents_b:
        return 0.7
    return None


def load_latest_keywords(path_override: str | None) -> dict:
    if path_override:
        p = Path(path_override).expanduser().resolve()
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    candidates = sorted(KEYWORDS_DIR.glob("1.0-title_subtitle_keyword_frequencies*.json"))
    if not candidates:
        raise SystemExit(f"No 1.0 keyword file found in {KEYWORDS_DIR}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    with latest.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["_loaded_from"] = str(latest)
    return data


def generate_bigrams(words: List[str]) -> Iterable[Tuple[str, str]]:
    # Deterministic pair generation, no co-occurrence involved.
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            yield words[i], words[j]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keywords",
        help="Path to 1.0 keyword JSON; defaults to most recent in step-2/outputs/",
    )
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONF, help="Keep only mappings >= this confidence")
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume.")
    parser.add_argument("--checkpoint-every-i", type=int, default=25, help="Checkpoint every N outer i values.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    parser.add_argument(
        "--depth-min",
        type=int,
        default=1,
        help="Taxonomy anchor depth (0=T0 roots, 1=T1 children, ...). Default 1 matches legacy 1.1a.",
    )
    parser.add_argument(
        "--depth-max",
        type=int,
        default=None,
        help="Inclusive upper depth; default = depth-min (single file).",
    )
    parser.add_argument(
        "--output-batch-tag",
        default=None,
        metavar="TAG",
        help="Timestamp suffix shared by all files in a multi-depth run (default: one new tag per invocation).",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with TAXONOMY_PATH.open("r", encoding="utf-8") as f:
        categories = json.load(f)

    keywords = load_latest_keywords(args.keywords)
    title_words = [x["word"] for x in keywords.get("title", []) if "word" in x]
    subtitle_words = [x["word"] for x in keywords.get("subtitle", []) if "word" in x]

    all_words = sorted(set(title_words) | set(subtitle_words))

    kw_path = Path(args.keywords).expanduser().resolve() if args.keywords else None
    if kw_path is None:
        candidates = sorted(KEYWORDS_DIR.glob("1.0-title_subtitle_keyword_frequencies*.json"))
        if not candidates:
            raise SystemExit(f"No 1.0 keyword file found in {KEYWORDS_DIR}")
        kw_path = max(candidates, key=lambda p: p.stat().st_mtime)

    def fp(path: Path) -> str:
        s = path.stat()
        raw = f"{path.as_posix()}|{s.st_size}|{int(s.st_mtime)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    depth_max = args.depth_max if args.depth_max is not None else args.depth_min
    if depth_max < args.depth_min:
        raise SystemExit("--depth-max must be >= --depth-min")

    tax_max = max_taxonomy_depth(categories)
    if depth_max > tax_max:
        print(
            f"Note: depth-max {depth_max} exceeds deepest slug depth seen ({tax_max}); "
            f"some passes may emit empty mappings.",
            file=sys.stderr,
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    show_progress = not args.no_progress
    batch_ts = args.output_batch_tag or timestamp()
    written: list[Path] = []

    for depth in range(args.depth_min, depth_max + 1):
        anchor_tokens = build_anchor_token_sets_for_depth(categories, depth)
        parent_sets = {w: word_to_parents(w, anchor_tokens) for w in all_words}

        fingerprint = (
            f"taxonomy={fp(TAXONOMY_PATH)}|keywords={fp(kw_path)}|min_conf={args.min_confidence}|taxonomy_depth={depth}"
        )
        ckpt_id = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
        ckpt_path = CHECKPOINT_DIR / f"step11a_checkpoint_{ckpt_id}.json"

        if not args.no_resume and ckpt_path.exists():
            ck = json.loads(ckpt_path.read_text(encoding="utf-8"))
            if ck.get("fingerprint") == fingerprint:
                title_next_i = int(ck.get("title_next_i", 0))
                subtitle_next_i = int(ck.get("subtitle_next_i", 0))
                title_bigrams = ck.get("title_bigrams", [])
                subtitle_bigrams = ck.get("subtitle_bigrams", [])
                print(
                    f"[depth={depth}] Resume: title_next_i={title_next_i} subtitle_next_i={subtitle_next_i} "
                    f"title_out_so_far={len(title_bigrams)} subtitle_out_so_far={len(subtitle_bigrams)}"
                )
            else:
                title_next_i, subtitle_next_i, title_bigrams, subtitle_bigrams = 0, 0, [], []
        else:
            title_next_i, subtitle_next_i, title_bigrams, subtitle_bigrams = 0, 0, [], []

        def map_side_with_checkpoint(
            words: List[str], side: str, next_i: int, out: List[dict], show_p: bool
        ) -> tuple[int, List[dict]]:
            n = len(words)
            use_bar = show_p and n > next_i and sys.stderr.isatty()
            pbar = (
                tqdm(
                    total=n,
                    initial=next_i,
                    desc=f"2.1b taxonomy bigrams depth={depth} {side}",
                    unit="i",
                    file=sys.stderr,
                    dynamic_ncols=True,
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
                )
                if use_bar
                else None
            )
            try:
                for i in range(next_i, n):
                    a = words[i]
                    pa = parent_sets.get(a, set())
                    if not pa:
                        if pbar:
                            pbar.update(1)
                        continue
                    for j in range(i + 1, n):
                        b = words[j]
                        pb = parent_sets.get(b, set())
                        if not pb:
                            continue
                        inter = pa & pb
                        if len(inter) != 1:
                            continue
                        best_parent = next(iter(inter))
                        conf = choose_confidence(pa, pb)
                        if conf is None or conf < args.min_confidence:
                            continue
                        out.append(
                            {
                                "bigram": [a, b],
                                "suggested_parent_category_slug": best_parent,
                                "confidence": conf,
                                "source": "taxonomy_based",
                                "side": side,
                            }
                        )

                    if (i + 1) % args.checkpoint_every_i == 0:
                        save_payload = {
                            "fingerprint": fingerprint,
                            "title_next_i": i + 1 if side == "title" else title_next_i,
                            "subtitle_next_i": i + 1 if side == "subtitle" else subtitle_next_i,
                            "title_bigrams": title_bigrams,
                            "subtitle_bigrams": subtitle_bigrams,
                        }
                        tmp = ckpt_path.with_suffix(".tmp")
                        tmp.write_text(json.dumps(save_payload, indent=0, ensure_ascii=False) + "\n", encoding="utf-8")
                        tmp.replace(ckpt_path)

                    if pbar:
                        pbar.update(1)
                        if (i + 1) % args.checkpoint_every_i == 0 or (i + 1) == n:
                            pbar.set_postfix_str(
                                f"word_i={i + 1}/{n} bigrams_kept={len(out)}",
                                refresh=False,
                            )
                        elif (i + 1) % 5 == 0:
                            pbar.set_postfix_str(f"i={i + 1}/{n} kept={len(out)}", refresh=False)
            finally:
                if pbar:
                    pbar.close()

            return n, out

        if title_next_i < len(title_words):
            next_i, title_bigrams = map_side_with_checkpoint(
                title_words, "title", title_next_i, title_bigrams, show_progress
            )
            title_next_i = next_i
        if subtitle_next_i < len(subtitle_words):
            next_i, subtitle_bigrams = map_side_with_checkpoint(
                subtitle_words, "subtitle", subtitle_next_i, subtitle_bigrams, show_progress
            )
            subtitle_next_i = next_i

        title_bigrams.sort(
            key=lambda x: (-x["confidence"], x["bigram"][0], x["bigram"][1], x["suggested_parent_category_slug"])
        )
        subtitle_bigrams.sort(
            key=lambda x: (-x["confidence"], x["bigram"][0], x["bigram"][1], x["suggested_parent_category_slug"])
        )

        payload = {
            "version": "1.1a",
            "taxonomy_depth": depth,
            "min_confidence_kept": args.min_confidence,
            "taxonomy_categories_file": TAXONOMY_PATH.name,
            "keywords_file": (keywords.get("_loaded_from") or "provided-via-args"),
            "constants": {
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "first3_gate": FIRST3_GATE,
                "confidence_tiers": {"both_unique": 1.0, "one_unique": 0.85, "fallback": 0.7},
            },
            "title_bigrams": [{k: v for k, v in x.items() if k != "side"} for x in title_bigrams],
            "subtitle_bigrams": [{k: v for k, v in x.items() if k != "side"} for x in subtitle_bigrams],
        }

        legacy_name = args.depth_min == depth_max == 1
        if legacy_name:
            out_path = OUTPUT_DIR / f"1.1a-bigram_categories_mapping_{batch_ts}.json"
        else:
            out_path = OUTPUT_DIR / f"1.1a-bigram_categories_mapping_depth{depth}_{batch_ts}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        written.append(out_path)
        print(f"Wrote {out_path}")
        print(f"  depth={depth} title_bigrams={len(payload['title_bigrams'])} subtitle_bigrams={len(payload['subtitle_bigrams'])}")

        try:
            if ckpt_path.exists():
                ckpt_path.unlink()
        except OSError:
            pass

    if len(written) > 1:
        print(f"Total mapping files written: {len(written)}")


if __name__ == "__main__":
    main()

