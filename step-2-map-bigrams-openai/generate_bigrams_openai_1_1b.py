#!/usr/bin/env python3
"""
Step 1.1b (LLM-based)
Generate bigram -> T1 parent category mappings by asking an OpenAI model.

Reads:
  - ../source-files/categories_v1.json
  - ../step-2-extract-keyword-frequencies/outputs/1.0-title_subtitle_keyword_frequencies*.json (most recent by default)
  - ../.env for OPENAI_API_KEY (or environment variable OPENAI_API_KEY)

Writes (timestamped) to:
  - ./outputs/1.1b-bigram_categories_mapping_YYYYMMDD_HHMMSS.json

Constraints:
  - Bigrams are formed ONLY from the curated 1.0 word lists (title-only and subtitle-only separately).
  - The model must choose only from the T1 parent category slugs.
  - We do NOT use item.category or item.subcategory for this step.

Warning:
  - Option B (all bigrams) can be very large and take a long time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]  # v2/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taxonomy_cascade import is_catch_all_bucket_slug

TAXONOMY_PATH = ROOT / "source-files" / "categories_v1.json"
KEYWORDS_DIR = ROOT / "step-2-extract-keyword-frequencies" / "outputs"
ENV_PATH = ROOT / ".env"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"

DEFAULT_MODEL = "gpt-4o-mini"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_env_dotfile(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def load_latest_keywords(path_override: str | None) -> dict:
    if path_override:
        p = Path(path_override).expanduser().resolve()
        return json.loads(p.read_text(encoding="utf-8"))
    candidates = sorted(KEYWORDS_DIR.glob("1.0-title_subtitle_keyword_frequencies*.json"))
    if not candidates:
        raise SystemExit(f"No 1.0 keyword file found in {KEYWORDS_DIR}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    d = json.loads(latest.read_text(encoding="utf-8"))
    d["_loaded_from"] = str(latest)
    return d


def build_t1_list(categories: dict) -> list[dict]:
    out: list[dict] = []
    for t0 in ("materials", "tools_and_gear", "services"):
        root = categories.get(t0)
        if not isinstance(root, dict):
            continue
        for child in root.get("subcategories", []):
            slug = child.get("slug")
            dn = child.get("display_name")
            if slug and is_catch_all_bucket_slug(slug):
                continue
            if slug:
                out.append({"slug": slug, "display_name": dn or slug, "t0": t0})
    return out


def generate_bigrams(words: list[str]) -> Iterable[Tuple[str, str]]:
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            yield words[i], words[j]


def chunked(seq: list[Tuple[str, str]], size: int) -> Iterable[list[Tuple[str, str]]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def require_openai() -> None:
    try:
        import openai  # noqa: F401
    except Exception:
        raise SystemExit("Missing dependency: openai. Install with `pip install openai` before running 1.1b.")


def call_openai_batch(model: str, t1_list: list[dict], pairs: list[Tuple[str, str]], min_conf: float) -> list[dict]:
    from openai import OpenAI

    client = OpenAI()

    allowed = [x["slug"] for x in t1_list]
    allowed_set = set(allowed)
    t1_lines = "\n".join([f"- {x['slug']}: {x['display_name']}" for x in t1_list])
    bigram_lines = "\n".join([f"- [{a}, {b}]" for a, b in pairs])

    prompt = f"""You are classifying 2-word keyword pairs into a construction materials taxonomy.

Choose exactly ONE best T1 parent category slug for each bigram, only from this allowed list:
{allowed}

T1 category descriptions:
{t1_lines}

Rules:
- Output JSON only (a JSON array).
- For each input bigram, return: {{"bigram": ["wordA","wordB"], "suggested_parent_category_slug": "...", "confidence": <number 0.7 to 1.0>}}
- Confidence reflects how strongly the bigram indicates that category.
- If the bigram is generic/ambiguous, set confidence below {min_conf}.

Bigrams:
{bigram_lines}
"""

    # Prefer Responses API; fallback to chat.completions for compatibility.
    try:
        resp = client.responses.create(model=model, input=prompt, temperature=0)
        text = resp.output_text
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = resp.choices[0].message.content

    if not isinstance(text, str):
        raise ValueError("Model response was not text.")
    text = text.strip()

    # Parse model JSON output, with a tiny code-fence salvage.
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        if "```" in text:
            mid = text.split("```", 2)[1].strip()
            data = json.loads(mid)
        else:
            raise

    if not isinstance(data, list):
        raise ValueError("Model output was not a JSON array.")

    out: list[dict] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        bg = obj.get("bigram")
        cat = obj.get("suggested_parent_category_slug")
        conf = obj.get("confidence")
        if not (isinstance(bg, list) and len(bg) == 2 and all(isinstance(x, str) for x in bg)):
            continue
        if not (isinstance(cat, str) and cat in allowed_set):
            continue
        try:
            conf_f = float(conf)
        except Exception:
            continue
        if conf_f < min_conf or conf_f > 1.0:
            continue
        out.append(
            {
                "bigram": [bg[0], bg[1]],
                "suggested_parent_category_slug": cat,
                "confidence": round(conf_f, 4),
                "source": "openai_llm",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", help="Path to 1.0 keyword JSON; defaults to most recent in step-1.0/outputs/")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume.")
    parser.add_argument("--checkpoint-every-chunk", type=int, default=1, help="Save progress every N chunks.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    load_env_dotfile(ENV_PATH)

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not found in environment or .env")

    require_openai()

    categories = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    t1_list = build_t1_list(categories)

    keywords = load_latest_keywords(args.keywords)
    title_words = [x["word"] for x in keywords.get("title", []) if "word" in x]
    subtitle_words = [x["word"] for x in keywords.get("subtitle", []) if "word" in x]

    def file_fingerprint(path: Path) -> str:
        st = path.stat()
        raw = f"{path.as_posix()}|{st.st_size}|{int(st.st_mtime)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    kw_loaded_from = keywords.get("_loaded_from")
    if not kw_loaded_from:
        raise SystemExit("Keywords did not provide _loaded_from (cannot resume safely).")
    kw_path = Path(kw_loaded_from).expanduser().resolve()

    base_fp = f"taxonomy={file_fingerprint(TAXONOMY_PATH)}|keywords={file_fingerprint(kw_path)}|model={args.model}|min_conf={args.min_confidence}|batch_size={args.batch_size}"

    # Option B: all bigrams
    title_pairs = list(generate_bigrams(title_words))
    subtitle_pairs = list(generate_bigrams(subtitle_words))

    def process_side(pairs: list[Tuple[str, str]], side: str) -> list[dict]:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = CHECKPOINT_DIR / f"step11b_checkpoint_{side}_{hashlib.sha256(base_fp.encode('utf-8')).hexdigest()[:10]}.json"

        out: list[dict] = []
        next_chunk_index = 0
        if (not args.no_resume) and ckpt_path.exists():
            ck = json.loads(ckpt_path.read_text(encoding="utf-8"))
            if ck.get("fingerprint") == base_fp:
                next_chunk_index = int(ck.get("next_chunk_index", 0))
                out = ck.get("out", [])
                if not isinstance(out, list):
                    out = []
                print(f"[{side}] Resume: next_chunk_index={next_chunk_index} kept_so_far={len(out)}")

        total_pairs = len(pairs)
        total_chunks = (total_pairs + args.batch_size - 1) // args.batch_size if total_pairs else 0

        started = time.time()
        calls = 0
        use_pbar = (
            not args.no_progress
            and total_chunks > next_chunk_index
            and sys.stderr.isatty()
        )
        pbar = (
            tqdm(
                total=total_chunks,
                initial=next_chunk_index,
                desc=f"1.1b {side} API batches",
                unit="batch",
                file=sys.stderr,
                dynamic_ncols=True,
            )
            if use_pbar
            else None
        )

        try:
            for chunk_index in range(next_chunk_index, total_chunks):
                start = chunk_index * args.batch_size
                chunk = pairs[start : start + args.batch_size]

                calls += 1
                preds = call_openai_batch(args.model, t1_list, chunk, args.min_confidence)
                out.extend(preds)

                # Update checkpoint after each chunk (or every N chunks)
                if args.checkpoint_every_chunk <= 1 or (calls % args.checkpoint_every_chunk == 0):
                    tmp = ckpt_path.with_suffix(".tmp")
                    payload = {
                        "fingerprint": base_fp,
                        "next_chunk_index": chunk_index + 1,
                        "out": out,
                    }
                    tmp.write_text(json.dumps(payload, indent=0, ensure_ascii=False) + "\n", encoding="utf-8")
                    tmp.replace(ckpt_path)

                if args.sleep_seconds:
                    time.sleep(args.sleep_seconds)

                if pbar:
                    pbar.update(1)
                    done_pairs = min(start + len(chunk), total_pairs)
                    elapsed = time.time() - started
                    rate = done_pairs / elapsed if elapsed > 0 else 0.0
                    pbar.set_postfix_str(
                        f"pairs {done_pairs}/{total_pairs} kept={len(out)} {rate:.1f} pairs/s",
                        refresh=False,
                    )
                elif calls % 10 == 0:
                    elapsed = time.time() - started
                    done_pairs = min(start + len(chunk), total_pairs)
                    rate = done_pairs / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[{side}] calls={calls} approx_pairs_done={done_pairs}/{total_pairs} "
                        f"rate~{rate:.1f} pairs/s kept={len(out)} ({chunk_index + 1}/{total_chunks})"
                    )
        finally:
            if pbar:
                pbar.close()

        # Success: delete checkpoint
        try:
            if ckpt_path.exists():
                ckpt_path.unlink()
        except Exception:
            pass

        out.sort(key=lambda x: (-x["confidence"], x["bigram"][0], x["bigram"][1], x["suggested_parent_category_slug"]))
        return out

    title_bigrams = process_side(title_pairs, "title")
    subtitle_bigrams = process_side(subtitle_pairs, "subtitle")

    payload = {
        "version": "1.1b",
        "model": args.model,
        "min_confidence_kept": args.min_confidence,
        "batch_size": args.batch_size,
        "taxonomy_categories_file": TAXONOMY_PATH.name,
        "keywords_file": (keywords.get("_loaded_from") or "provided-via-args"),
        "title_bigrams": title_bigrams,
        "subtitle_bigrams": subtitle_bigrams,
    }

    out_path = OUTPUT_DIR / f"1.1b-bigram_categories_mapping_{timestamp()}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"title_bigrams kept: {len(title_bigrams)}")
    print(f"subtitle_bigrams kept: {len(subtitle_bigrams)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise

