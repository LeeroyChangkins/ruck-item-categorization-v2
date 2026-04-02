#!/usr/bin/env python3
"""
Step 5 — LLM-based attribute generation for leaf categories.

For every leaf category in categories_v1.json, samples up to --sample items
from the step-4 matched output, then asks gpt-4o to propose product attributes
that buyers would use to filter or specify items in that category.

Batching strategy
-----------------
  • --batch-size  categories packed into a single LLM call   (default 5)
  • --workers     concurrent API requests in flight at once   (default 15)

Output
------
  step-5/outputs/proposed_attributes_YYYYMMDD_HHMMSS.json

  Schema mirrors proposed-attributes.json (for step-6 DB import):
  {
    "_meta": { ... run info ... },
    "units": {
      "<symbol>": { "symbol", "name", "description", "value_type" }
    },
    "_category_attributes": {
      "<leaf_slug_path>": [
        {
          "key":          str,   # snake_case identifier
          "label":        str,   # human-readable name
          "description":  str,   # brief definition / example values
          "value_type":   "number" | "text" | "boolean",
          "unit_required": bool,
          "unit":         str | null  # unit symbol if unit_required
        }
      ]
    }
  }

Usage
-----
  python 5_generate_attributes.py [OPTIONS]

  --run-dir      PATH   step-4 outputs subfolder (default: latest)
  --batch-size   INT    categories per LLM call   (default: 5)
  --workers      INT    parallel requests          (default: 15)
  --sample       INT    max items per category     (default: 50)
  --model        STR    OpenAI model               (default: gpt-4o)
  --dry-run             Print prompts without calling the API
  --resume       PATH   Resume from a partial output file
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]          # v2/
STEP5_DIR = Path(__file__).resolve().parent
STEP4_OUTPUTS = ROOT / "step-4" / "outputs"
OUTDIR = STEP5_DIR / "outputs"
TAXONOMY_PATH = ROOT / "source-files" / "categories_v1.json"

# Add ROOT to path for imports
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared_utils import load_dotenv_file as _load_dotenv_file, timestamp as _timestamp

DEFAULT_MODEL = "gpt-4o"
DEFAULT_BATCH_SIZE = 5
DEFAULT_WORKERS = 15
DEFAULT_SAMPLE = 50


# ── env / deps ─────────────────────────────────────────────────────────────────

def _load_env() -> None:
    _load_dotenv_file()


def _require_openai():
    try:
        import openai
        return openai
    except ImportError:
        sys.exit("openai package not installed — run: pip install openai")


# ── taxonomy helpers ───────────────────────────────────────────────────────────

def collect_leaves(node: dict, path_parts: list[str]) -> list[tuple[str, str]]:
    """Recursively collect (slug_path, display_name) for every leaf."""
    subcats = node.get("subcategories", [])
    if not subcats:
        return [("/".join(path_parts), node.get("display_name", path_parts[-1]))]
    results = []
    for child in subcats:
        results.extend(collect_leaves(child, path_parts + [child["slug"]]))
    return results


def load_leaves(taxonomy_path: Path) -> list[tuple[str, str]]:
    with open(taxonomy_path) as f:
        data = json.load(f)
    leaves: list[tuple[str, str]] = []
    for root_key, root_node in data.items():
        for child in root_node.get("subcategories", []):
            leaves.extend(collect_leaves(child, [root_key, child["slug"]]))
    return leaves


# ── step-4 data helpers ────────────────────────────────────────────────────────

def find_latest_run_dir(step4_outputs: Path) -> Path:
    dirs = sorted(
        [d for d in step4_outputs.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not dirs:
        sys.exit(f"No run directories found under {step4_outputs}")
    return dirs[0]


def load_matched_items(run_dir: Path) -> dict[str, list[dict]]:
    """Return leaf_path → list of item dicts from matched_deduped.json."""
    candidates = list(run_dir.glob("matched_deduped*.json"))
    if not candidates:
        sys.exit(f"No matched_deduped*.json found in {run_dir}")
    path = sorted(candidates)[-1]
    with open(path) as f:
        data = json.load(f)

    items: list[dict] = data.get("matched_items") or data.get("items") or []
    grouped: dict[str, list[dict]] = {}
    for item in items:
        leaf = item.get("leaf_path") or item.get("leaf_slug") or ""
        if leaf:
            grouped.setdefault(leaf, []).append(item)
    return grouped


def sample_items(items: list[dict], n: int, seed: int | None = None) -> list[str]:
    """Return up to n randomly sampled item title strings."""
    rng = random.Random(seed)
    pool = items if len(items) <= n else rng.sample(items, n)
    titles = []
    for it in pool:
        title = it.get("title", "")
        subtitle = it.get("subtitle", "")
        titles.append(f"{title} — {subtitle}".strip(" —") if subtitle else title)
    return titles


# ── prompt / LLM ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a product-data architect for a construction-materials marketplace.
Your job is to propose structured product attributes for leaf taxonomy categories.

Rules:
- Return ONLY valid JSON — no markdown fences, no extra text.
- For each category produce 3–8 attributes that buyers use to filter or specify items.
- Prefer attributes visible on the item itself (dimensions, grade, material, finish, etc.).
- Omit generic attributes that apply to every product (price, brand, SKU, weight unless
  weight is a key spec for the category).
- key     : snake_case, lowercase, no spaces.
- label   : short human-readable name (Title Case).
- description : one sentence definition; include example values where helpful.
- value_type : "number", "text", or "boolean".
- unit_required : true only for numeric attributes that need a unit (length, weight, area…).
- unit   : if unit_required is true, supply a unit object with symbol (e.g. "in", "ft",
  "lbs", "sq ft"), name, description, and value_type ("number").
  Use null if unit_required is false.

Response schema (strict):
{
  "categories": [
    {
      "path": "<leaf_slug_path>",
      "attributes": [
        {
          "key": "...",
          "label": "...",
          "description": "...",
          "value_type": "number" | "text" | "boolean",
          "unit_required": true | false,
          "unit": { "symbol": "...", "name": "...", "description": "...", "value_type": "number" } | null
        }
      ]
    }
  ]
}
"""


def _format_batch(batch: list[tuple[str, str, list[str]]]) -> str:
    """Format (path, display_name, [titles]) tuples into a user message."""
    lines = [
        "Generate attributes for the following construction-materials categories.\n"
    ]
    for i, (path, display_name, titles) in enumerate(batch, 1):
        lines.append(f"--- Category {i} ---")
        lines.append(f'path: "{path}"')
        lines.append(f'display_name: "{display_name}"')
        if titles:
            lines.append("sample items:")
            for t in titles:
                lines.append(f"  • {t}")
        else:
            lines.append("sample items: (none available)")
        lines.append("")
    return "\n".join(lines)


async def _call_llm(
    client,
    model: str,
    batch: list[tuple[str, str, list[str]]],
    semaphore: asyncio.Semaphore,
    retries: int = 3,
) -> dict | None:
    user_msg = _format_batch(batch)
    for attempt in range(retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.2,
                )
            raw = response.choices[0].message.content
            return json.loads(raw)
        except Exception as exc:
            wait = 2 ** attempt
            print(f"\n  [retry {attempt+1}/{retries}] {type(exc).__name__}: {exc} — waiting {wait}s")
            await asyncio.sleep(wait)
    print(f"\n  [FAILED] batch paths: {[p for p,_,_ in batch]}")
    return None


# ── output assembly ───────────────────────────────────────────────────────────

def _merge_result(
    result: dict,
    category_attributes: dict[str, list[dict]],
    units: dict[str, dict],
) -> None:
    """Merge one LLM response into the running output dicts (in-place)."""
    for cat in result.get("categories", []):
        path = cat.get("path", "")
        if not path:
            continue
        attrs = []
        for a in cat.get("attributes", []):
            unit_obj = a.get("unit")
            unit_symbol = None
            if unit_obj and isinstance(unit_obj, dict):
                sym = unit_obj.get("symbol", "")
                if sym and sym not in units:
                    units[sym] = {
                        "symbol": sym,
                        "name": unit_obj.get("name", sym),
                        "description": unit_obj.get("description", ""),
                        "value_type": unit_obj.get("value_type", "number"),
                    }
                unit_symbol = sym or None
            attrs.append({
                "key":           a.get("key", ""),
                "label":         a.get("label", ""),
                "description":   a.get("description", ""),
                "value_type":    a.get("value_type", "text"),
                "unit_required": bool(a.get("unit_required", False)),
                "unit":          unit_symbol,
            })
        if attrs:
            category_attributes[path] = attrs


# ── main async pipeline ───────────────────────────────────────────────────────

async def generate(
    leaves: list[tuple[str, str]],
    items_by_leaf: dict[str, list[dict]],
    model: str,
    batch_size: int,
    workers: int,
    sample_n: int,
    dry_run: bool,
    existing_paths: set[str],
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """
    Returns (category_attributes, units).
    Skips leaves already present in existing_paths (resume support).
    """
    openai = _require_openai()
    client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(workers)

    # Build work items — skip already-done leaves
    todo: list[tuple[str, str, list[str]]] = []
    skipped = 0
    for path, display_name in leaves:
        if path in existing_paths:
            skipped += 1
            continue
        titles = sample_items(
            items_by_leaf.get(path, []),
            n=sample_n,
            seed=hash(path) % (2**31),
        )
        todo.append((path, display_name, titles))

    if skipped:
        print(f"  Resuming — skipping {skipped} already-processed categories.")

    # Chunk into batches
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    total_cats = len(todo)
    total_batches = len(batches)

    print(f"  {total_cats} categories → {total_batches} batches "
          f"(batch_size={batch_size}, workers={workers})")

    if dry_run:
        print("\n=== DRY RUN — first batch prompt ===")
        if batches:
            print(_format_batch(batches[0]))
        return {}, {}

    category_attributes: dict[str, list[dict]] = {}
    units: dict[str, dict] = {}
    cats_matched = 0
    cats_failed = 0
    batch_times: list[float] = []

    try:
        from tqdm import tqdm as _tqdm
        use_tqdm = sys.stderr.isatty()
    except ImportError:
        use_tqdm = False

    pbar = None
    if use_tqdm:
        pbar = _tqdm(
            total=total_batches,
            desc="5 attr gen",
            unit="batch",
            file=sys.stderr,
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]{postfix}",
        )

    lock = asyncio.Lock()

    async def process_batch(batch):
        nonlocal cats_matched, cats_failed
        t0 = time.monotonic()
        result = await _call_llm(client, model, batch, semaphore)
        elapsed_b = time.monotonic() - t0

        async with lock:
            if result:
                _merge_result(result, category_attributes, units)
                cats_matched += len(batch)
            else:
                cats_failed += len(batch)
            batch_times.append(elapsed_b)

            if pbar:
                batches_done = len(batch_times)
                batches_remaining = total_batches - batches_done
                avg = sum(batch_times) / batches_done
                eta_secs = avg * batches_remaining
                if eta_secs >= 60:
                    eta_str = f"{int(eta_secs // 60)}m {int(eta_secs % 60)}s"
                else:
                    eta_str = f"{int(eta_secs)}s"
                pbar.set_postfix_str(
                    f"{cats_matched} done  {cats_failed} failed  ~{eta_str} left",
                    refresh=True,
                )
                pbar.update(1)
            else:
                batches_done = len(batch_times)
                status = "ok" if result else "FAILED"
                print(f"  Batch {batches_done}/{total_batches}: {len(batch)} categories [{status}]"
                      f"  total done={cats_matched} failed={cats_failed}")

    await asyncio.gather(*[process_batch(b) for b in batches])

    if pbar:
        pbar.close()

    total_elapsed = sum(batch_times)
    print(f"\n  Done in {total_elapsed:.1f}s — {cats_matched} categories attributed, {cats_failed} failed")

    return category_attributes, units


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="Step 5 — LLM attribute generation")
    parser.add_argument("--run-dir",    type=Path, default=None,
                        help="step-4 outputs subfolder (default: latest)")
    parser.add_argument("--batch-size", type=int,  default=DEFAULT_BATCH_SIZE,
                        help=f"Categories per LLM call (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--workers",    type=int,  default=DEFAULT_WORKERS,
                        help=f"Parallel API requests (default: {DEFAULT_WORKERS})")
    parser.add_argument("--sample",     type=int,  default=DEFAULT_SAMPLE,
                        help=f"Max items sampled per category (default: {DEFAULT_SAMPLE})")
    parser.add_argument("--model",      type=str,  default=DEFAULT_MODEL,
                        help=f"OpenAI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print prompts only, no API calls")
    parser.add_argument("--resume",     type=Path, default=None,
                        help="Path to partial output JSON to resume from")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY") and not args.dry_run:
        sys.exit("OPENAI_API_KEY not set — add it to .env or export it.")

    # ── load taxonomy
    print(f"Loading taxonomy from {TAXONOMY_PATH.name}…")
    leaves = load_leaves(TAXONOMY_PATH)
    print(f"  {len(leaves)} leaf categories found.")

    # ── load step-4 items
    run_dir = args.run_dir or find_latest_run_dir(STEP4_OUTPUTS)
    print(f"Loading matched items from {run_dir.name}…")
    items_by_leaf = load_matched_items(run_dir)
    covered = sum(1 for p, _ in leaves if p in items_by_leaf)
    print(f"  {covered}/{len(leaves)} leaves have matched items.")

    # ── resume support
    existing_paths: set[str] = set()
    existing_cat_attrs: dict[str, list[dict]] = {}
    existing_units: dict[str, dict] = {}

    if args.resume and args.resume.exists():
        print(f"Resuming from {args.resume.name}…")
        with open(args.resume) as f:
            prev = json.load(f)
        existing_cat_attrs = prev.get("_category_attributes", {})
        existing_units     = prev.get("units", {})
        existing_paths     = set(existing_cat_attrs.keys())
        print(f"  {len(existing_paths)} categories already done.")

    # ── run
    new_cat_attrs, new_units = asyncio.run(generate(
        leaves=leaves,
        items_by_leaf=items_by_leaf,
        model=args.model,
        batch_size=args.batch_size,
        workers=args.workers,
        sample_n=args.sample,
        dry_run=args.dry_run,
        existing_paths=existing_paths,
    ))

    if args.dry_run:
        return

    # ── merge resume + new
    merged_attrs = {**existing_cat_attrs, **new_cat_attrs}
    merged_units = {**existing_units, **new_units}

    # ── write output
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = _timestamp()
    out_path = OUTDIR / f"proposed_attributes_{ts}.json"

    output = {
        "_meta": {
            "generated_at": datetime.now().isoformat(),
            "model": args.model,
            "source_run_dir": str(run_dir),
            "leaves_total": len(leaves),
            "leaves_with_attributes": len(merged_attrs),
            "batch_size": args.batch_size,
            "workers": args.workers,
            "sample_per_category": args.sample,
        },
        "units": merged_units,
        "_category_attributes": merged_attrs,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nOutput written to: {out_path}")
    print(f"  {len(merged_attrs)} categories with attributes")
    print(f"  {len(merged_units)} unique units referenced")
    no_attr = [p for p, _ in leaves if p not in merged_attrs]
    if no_attr:
        print(f"  {len(no_attr)} leaves with no attributes generated "
              f"(no items? API errors?)")

    # ── consolidate final outputs into final-output/<timestamp>/
    final_dir = ROOT / "final-output" / ts
    final_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(out_path, final_dir / "proposed_attributes.json")
    shutil.copy2(TAXONOMY_PATH, final_dir / "categories_v1.json")

    # Find latest matched_deduped.json from step 4
    step4_matched = sorted(
        (ROOT / "step-4" / "outputs").glob("**/matched_deduped.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if step4_matched:
        shutil.copy2(step4_matched[-1], final_dir / "matched_deduped.json")
        print(f"\nFinal outputs consolidated → final-output/{ts}/")
        print(f"  proposed_attributes.json")
        print(f"  matched_deduped.json  (from {step4_matched[-1].parent.name})")
        print(f"  categories_v1.json")
    else:
        print(f"\nFinal outputs consolidated → final-output/{ts}/")
        print(f"  proposed_attributes.json")
        print(f"  categories_v1.json")
        print(f"  (matched_deduped.json not found — run step 4 first)")


if __name__ == "__main__":
    main()
