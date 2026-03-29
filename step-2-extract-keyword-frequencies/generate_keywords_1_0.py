#!/usr/bin/env python3
"""
Step 1.0
Generate a curated keyword frequency list from production items.

Reads:
  - ../source-files/raw-prod-items-non-deleted.json

Writes (timestamped) to:
  - ./outputs/1.0-title_subtitle_keyword_frequencies_YYYYMMDD_HHMMSS.json

Rules:
  - tokenize letters-only (A–Z runs)
  - lowercase
  - min length: 4
  - count once per item per word
  - keep words appearing in >= 3 items
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Set


ROOT = Path(__file__).resolve().parents[1]  # v2/
ITEMS_PATH = ROOT / "source-files" / "raw-prod-items-non-deleted.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

MIN_WORD_LEN = 4
MIN_ITEMS_PER_WORD = 3


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def tokenize_letters_only(text: str) -> Set[str]:
    """Return a set of lowercase letter-only tokens (A–Z runs). No regex."""
    if not text:
        return set()
    s = str(text)
    out: Set[str] = set()
    cur: list[str] = []
    for ch in s:
        o = ord(ch)
        if (65 <= o <= 90) or (97 <= o <= 122):
            cur.append(ch)
        else:
            if cur:
                w = "".join(cur).lower()
                if len(w) >= MIN_WORD_LEN:
                    out.add(w)
                cur = []
    if cur:
        w = "".join(cur).lower()
        if len(w) >= MIN_WORD_LEN:
            out.add(w)
    return out


def count_words(items: Iterable[dict], field: str) -> Counter:
    c: Counter[str] = Counter()
    for it in items:
        tokens = tokenize_letters_only(it.get(field) or "")
        c.update(tokens)  # tokens is a set => once per item
    return c


def sorted_rows(counter: Counter) -> list[dict]:
    rows = [(w, n) for w, n in counter.items() if n >= MIN_ITEMS_PER_WORD]
    rows.sort(key=lambda x: (-x[1], x[0]))
    return [{"word": w, "item_count": n} for w, n in rows]


def file_fingerprint(path: Path) -> str:
    st = path.stat()
    raw = f"{path.as_posix()}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def checkpoint_path(fingerprint: str) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return CHECKPOINT_DIR / f"step10_checkpoint_{fingerprint}.json"


def load_checkpoint(path: Path, expected_fingerprint: str, no_resume: bool) -> tuple[int, Counter, Counter]:
    if no_resume or not path.exists():
        return 0, Counter(), Counter()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("fingerprint") != expected_fingerprint:
        return 0, Counter(), Counter()
    start_index = int(data.get("next_index", 0))
    title_counts = Counter(data.get("title_counts", {}))
    subtitle_counts = Counter(data.get("subtitle_counts", {}))
    return start_index, title_counts, subtitle_counts


def save_checkpoint(path: Path, fingerprint: str, next_index: int, title_counts: Counter, subtitle_counts: Counter) -> None:
    tmp = path.with_suffix(".tmp")
    payload = {
        "fingerprint": fingerprint,
        "next_index": next_index,
        "title_counts": dict(title_counts),
        "subtitle_counts": dict(subtitle_counts),
    }
    tmp.write_text(json.dumps(payload, indent=0, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--items-json",
        metavar="PATH",
        help="Optional JSON: top-level array of items, or object with 'items' / 'unmatched_items' "
        "(e.g. unmatched_after_step1.json). When set, keyword counts use only these items.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume.")
    parser.add_argument("--checkpoint-every", type=int, default=2000, help="Save progress every N items.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.items_json:
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
        items_path_for_fp = ip
    else:
        items_path_for_fp = ITEMS_PATH
        with ITEMS_PATH.open("r", encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise SystemExit(f"Expected top-level array in {ITEMS_PATH}")

    fp = file_fingerprint(items_path_for_fp)
    ckpt = checkpoint_path(fp)
    start_index, title_counts, subtitle_counts = load_checkpoint(ckpt, fp, no_resume=args.no_resume)

    print(f"Resume start_index={start_index} title_tokens_so_far={len(title_counts)} subtitle_tokens_so_far={len(subtitle_counts)}")

    for idx in range(start_index, len(items)):
        it = items[idx]
        title_tokens = tokenize_letters_only(it.get("title") or "")
        subtitle_tokens = tokenize_letters_only(it.get("subtitle") or "")
        title_counts.update(title_tokens)
        subtitle_counts.update(subtitle_tokens)

        if (idx + 1) % args.checkpoint_every == 0:
            save_checkpoint(ckpt, fp, next_index=idx + 1, title_counts=title_counts, subtitle_counts=subtitle_counts)
            print(f"Checkpoint saved at idx={idx + 1}/{len(items)}")

    payload = {
        "version": "1.0",
        "source_file": items_path_for_fp.name,
        "total_items": len(items),
        "items_subset": bool(args.items_json),
        "rules": {
            "min_word_length": MIN_WORD_LEN,
            "letters_only_no_digits": True,
            "count_once_per_item_per_word": True,
            "min_items_per_word": MIN_ITEMS_PER_WORD,
            "sorted_by": "item_count_desc_then_word_asc",
        },
        "title": sorted_rows(title_counts),
        "subtitle": sorted_rows(subtitle_counts),
    }

    out_path = OUTPUT_DIR / f"1.0-title_subtitle_keyword_frequencies_{timestamp()}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {out_path}")
    print(f"title words kept: {len(payload['title'])}")
    print(f"subtitle words kept: {len(payload['subtitle'])}")

    # Cleanup checkpoint after success
    try:
        if ckpt.exists():
            ckpt.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    main()

