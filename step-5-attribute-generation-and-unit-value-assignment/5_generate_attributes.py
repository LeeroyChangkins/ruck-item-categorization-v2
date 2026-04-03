#!/usr/bin/env python3
"""
Step 5b — LLM Attribute Schema + Regex Pattern Generation.

Reads the title_groups/ directory produced by step 5a.  For each leaf
category it builds a prompt containing:
  • The category name / path
  • The top-N structural template clusters with one representative title each
  • A flag indicating whether the category is low-structure

The LLM (gpt-4o) is asked to return:
  1. A standard attribute schema (key, label, value_type, unit)
  2. For each attribute, an array of extraction patterns — one per template
     cluster — each containing a Python-compatible named-group regex and a
     normalisation hint.

For LOW-STRUCTURE categories the LLM skips patterns; items in those
categories that don't match any regex are handled by an LLM fallback in 5c.

Pattern smoke-test
------------------
After each LLM response, every regex is tested against the representative
title it was designed for.  Patterns that fail the smoke test are dropped and
a warning is printed.  If all patterns for an attribute fail, the attribute
is kept (schema only) and 5c will fall back to the LLM for that attribute.

Output
------
  <out_dir>/proposed_attributes.json

Schema (extends the existing format):
  {
    "_meta": { ... },
    "units": { "<symbol>": { symbol, name, description, value_type } },
    "_category_attributes": {
      "<leaf_path>": [
        {
          "key":           str,
          "label":         str,
          "description":   str,
          "value_type":    "number" | "text" | "boolean",
          "unit_required": bool,
          "unit":          str | null,
          "patterns": [        ← NEW; empty list for low-structure or no patterns
            {
              "template":         str,   // normalised template from 5a
              "regex":            str,   // Python regex with named group (?P<val>...)
              "value_normalize":  str    // "text_raw"|"text_lower"|"measurement_inches"|
                                         //  "measurement_feet"
            }
          ]
        }
      ]
    }
  }

Usage
-----
  python 5_generate_attributes.py [OPTIONS]

  --groups-dir PATH   title_groups/ dir from step 5a (default: latest under step-5/outputs/)
  --out-dir    PATH   write proposed_attributes.json here (default: same dir as groups-dir)
  --batch-size INT    categories per LLM call  (default: 3)
  --workers    INT    concurrent requests      (default: 10)
  --max-clusters INT  max clusters sent to LLM per category (default: 5)
  --model      STR    OpenAI model             (default: gpt-4o)
  --dry-run           Print prompts without calling the API
  --resume     PATH   Resume from a partial proposed_attributes.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STEP5_DIR = Path(__file__).resolve().parent
TAXONOMY_PATH = ROOT / "source-files" / "categories_v1.json"
STEP4_OUTPUTS = ROOT / "step-4-dedupe-and-merge-matched-items" / "outputs"

sys.path.insert(0, str(ROOT))
import shared_utils as _su
from shared_utils import load_dotenv_file as _load_dotenv, timestamp as _timestamp, env_suffix as _env_suffix

DEFAULT_MODEL      = "gpt-4o"
DEFAULT_BATCH_SIZE = 3
DEFAULT_WORKERS    = 10
DEFAULT_MAX_CLUSTERS = 5


# ── env / deps ─────────────────────────────────────────────────────────────────

def _require_openai():
    try:
        import openai
        return openai
    except ImportError:
        sys.exit("openai package not installed — run: pip install openai")


# ── groups-dir discovery ───────────────────────────────────────────────────────

def find_latest_groups_dir() -> Path:
    outputs = STEP5_DIR / "outputs"
    if not outputs.exists():
        sys.exit(f"No step-5/outputs/ directory found — run step 5a first.")
    # candidates are the title_groups/ dirs; filter by the parent dir's env suffix
    run_dirs = [
        d for d in outputs.iterdir()
        if d.is_dir() and any((d / "title_groups").glob("manifest*.json"))
    ]
    if not run_dirs:
        sys.exit("No title_groups/manifest*.json found — run step 5a first.")
    best = _su.latest_env_path(run_dirs, name_attr="name")
    return (best or run_dirs[0]) / "title_groups"


def load_groups(groups_dir: Path) -> dict[str, dict]:
    """Return leaf_path → category group data dict."""
    mf_cands = sorted(groups_dir.glob("manifest*.json"), key=lambda p: p.stat().st_mtime)
    if not mf_cands:
        raise FileNotFoundError(f"No manifest*.json in {groups_dir}")
    manifest = json.loads(mf_cands[-1].read_text())
    result: dict[str, dict] = {}
    for entry in manifest.get("categories", []):
        leaf_path = entry["leaf_path"]
        cat_file = groups_dir / entry["file"]
        if cat_file.exists():
            result[leaf_path] = json.loads(cat_file.read_text())
    return result


# ── taxonomy helpers ───────────────────────────────────────────────────────────

def _collect_leaves(node: dict, parts: list[str]) -> list[tuple[str, str]]:
    subcats = node.get("subcategories", [])
    if not subcats:
        return [("/".join(parts), node.get("display_name", parts[-1]))]
    out: list[tuple[str, str]] = []
    for child in subcats:
        out.extend(_collect_leaves(child, parts + [child["slug"]]))
    return out


def load_leaves(path: Path) -> dict[str, str]:
    """Return leaf_path → display_name."""
    data = json.loads(path.read_text())
    result: dict[str, str] = {}
    for root_key, root_node in data.items():
        for child in root_node.get("subcategories", []):
            for p, dn in _collect_leaves(child, [root_key, child["slug"]]):
                result[p] = dn
    return result


# ── prompt builder ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a product-data architect for a construction-materials marketplace.

For each category you receive you must return TWO things:
1. An attribute schema (3-8 attributes buyers use to filter/specify products).
2. For each attribute: a list of extraction patterns — one per structural template
   shown — that extract the attribute value from real item titles.

RULES FOR ATTRIBUTES:
- key          : snake_case, lowercase, no spaces.
- label        : short human-readable name (Title Case).
- description  : one sentence with example values.
- value_type   : "number", "text", or "boolean".
- unit_required: true only when the value needs a unit (length, weight…).
- unit         : if unit_required, include { symbol, name, description, value_type:"number" };
                 else null.

RULES FOR PATTERNS:
- Each pattern targets ONE structural template exactly as written.
- "regex"          : valid Python regex.  Use a NAMED group (?P<val>...) to capture the value.
                     The regex must match the representative title shown for that template.
- "value_normalize": MUST be one of:
    "text_raw"            – keep value exactly as captured
    "text_lower"          – lowercase and strip whitespace
    "measurement_inches"  – strip inch symbols (" / in / inches / in.) keep raw string
    "measurement_feet"    – strip foot symbols (' / ft / feet) keep raw string
- Omit patterns for attributes that cannot be reliably extracted from titles
  (e.g. colour from a title that has no colour token).
- LOW-STRUCTURE categories: the prompt will say "[low-structure]".
  For these, return ONLY the attribute schema with empty patterns arrays.
  Do NOT attempt to write patterns.

RESPONSE FORMAT — return ONLY valid JSON, no markdown fences, no extra text:
{
  "categories": [
    {
      "path": "<leaf_path>",
      "attributes": [
        {
          "key": "...",
          "label": "...",
          "description": "...",
          "value_type": "number" | "text" | "boolean",
          "unit_required": true | false,
          "unit": { "symbol": "...", "name": "...", "description": "...", "value_type": "number" } | null,
          "patterns": [
            {
              "template": "<exact normalised template string>",
              "regex": "(?P<val>...)",
              "value_normalize": "text_raw" | "text_lower" | "measurement_inches" | "measurement_feet"
            }
          ]
        }
      ]
    }
  ]
}
"""


def _format_batch(
    batch: list[tuple[str, str, dict, bool]],
    max_clusters: int,
) -> str:
    """
    batch items: (leaf_path, display_name, group_data, is_low_structure)
    """
    lines = ["Generate attributes and extraction patterns for the following categories.\n"]
    for i, (path, display_name, group_data, is_low) in enumerate(batch, 1):
        tag = "  [low-structure — schema only, NO patterns]" if is_low else ""
        lines.append(f"--- Category {i} ---{tag}")
        lines.append(f'path: "{path}"')
        lines.append(f'display_name: "{display_name}"')
        clusters = group_data.get("clusters", [])[:max_clusters]
        misc_count = (group_data.get("misc") or {}).get("item_count", 0)
        if clusters:
            lines.append("structural templates (sorted by frequency, most common first):")
            for c in clusters:
                lines.append(f"  template ({c['item_count']} items): {c['template']}")
                lines.append(f"  representative title  : {c['representative']}")
        else:
            lines.append("structural templates: (none — titles are all unique)")
        if misc_count:
            lines.append(f"  misc (unique templates): {misc_count} items — no pattern needed")
        lines.append("")
    return "\n".join(lines)


# ── regex smoke test ───────────────────────────────────────────────────────────

def _smoke_test_patterns(attr: dict, category_path: str) -> dict:
    """
    Test each pattern's regex against the representative title for its template.
    Returns the attribute with only passing patterns kept.
    """
    group_data_by_tmpl: dict[str, str] = {}
    # We'll populate this after we have the group data in context;
    # for now just validate the regex compiles and has the named group.
    good_patterns: list[dict] = []
    raw_patterns: list[dict] = attr.get("patterns") or []

    for pat in raw_patterns:
        regex_str = pat.get("regex", "")
        template  = pat.get("template", "")
        try:
            compiled = re.compile(regex_str)
        except re.error as e:
            print(f"  [smoke-test] DROPPED broken regex for {category_path}/{attr['key']}: {e}")
            continue
        if "val" not in compiled.groupindex:
            print(f"  [smoke-test] DROPPED pattern missing (?P<val>...) for "
                  f"{category_path}/{attr['key']} — template: {template[:60]}")
            continue
        good_patterns.append(pat)

    attr = dict(attr)
    attr["patterns"] = good_patterns
    return attr


def _smoke_test_against_reps(
    attr: dict,
    representative_map: dict[str, str],
    category_path: str,
) -> dict:
    """
    Test each pattern against the actual representative title for its template.
    Drop patterns that fail to produce a match.
    """
    good: list[dict] = []
    for pat in attr.get("patterns", []):
        template = pat.get("template", "")
        rep_title = representative_map.get(template, "")
        if not rep_title:
            good.append(pat)  # no representative to test against; keep
            continue
        try:
            m = re.search(pat["regex"], rep_title)
        except re.error:
            continue
        if m and m.group("val").strip():
            good.append(pat)
        else:
            print(f"  [smoke-test] DROPPED non-matching regex for "
                  f"{category_path}/{attr['key']}:\n"
                  f"    regex: {pat['regex']}\n"
                  f"    title: {rep_title}")
    attr = dict(attr)
    attr["patterns"] = good
    return attr


# ── LLM call ──────────────────────────────────────────────────────────────────

async def _call_llm(
    client,
    model: str,
    batch: list[tuple[str, str, dict, bool]],
    semaphore: asyncio.Semaphore,
    max_clusters: int,
    retries: int = 3,
) -> dict | None:
    user_msg = _format_batch(batch, max_clusters)
    for attempt in range(retries):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.2,
                )
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:
            wait = 2 ** attempt
            print(f"\n  [retry {attempt+1}/{retries}] {type(exc).__name__}: {exc} — waiting {wait}s")
            await asyncio.sleep(wait)
    print(f"\n  [FAILED] batch: {[p for p,_,_,_ in batch]}")
    return None


# ── result merger ─────────────────────────────────────────────────────────────

def _merge_result(
    result: dict,
    batch: list[tuple[str, str, dict, bool]],
    category_attributes: dict[str, list[dict]],
    units: dict[str, dict],
) -> None:
    # Build representative map for smoke tests: template → rep_title
    rep_map_by_path: dict[str, dict[str, str]] = {}
    for path, _dn, group_data, _low in batch:
        rep_map_by_path[path] = {
            c["template"]: c["representative"]
            for c in group_data.get("clusters", [])
        }

    for cat in result.get("categories", []):
        path = cat.get("path", "")
        if not path:
            continue
        rep_map = rep_map_by_path.get(path, {})
        attrs: list[dict] = []
        for a in cat.get("attributes", []):
            unit_obj = a.get("unit")
            unit_symbol: str | None = None
            if unit_obj and isinstance(unit_obj, dict):
                sym = unit_obj.get("symbol", "").strip()
                if sym and sym not in units:
                    units[sym] = {
                        "symbol":      sym,
                        "name":        unit_obj.get("name", sym),
                        "description": unit_obj.get("description", ""),
                        "value_type":  unit_obj.get("value_type", "number"),
                    }
                unit_symbol = sym or None

            attr: dict[str, Any] = {
                "key":           a.get("key", ""),
                "label":         a.get("label", ""),
                "description":   a.get("description", ""),
                "value_type":    a.get("value_type", "text"),
                "unit_required": bool(a.get("unit_required", False)),
                "unit":          unit_symbol,
                "patterns":      a.get("patterns") or [],
            }
            # Smoke-test: regex compiles + has named group
            attr = _smoke_test_patterns(attr, path)
            # Smoke-test: regex matches its representative title
            attr = _smoke_test_against_reps(attr, rep_map, path)

            attrs.append(attr)

        if attrs:
            category_attributes[path] = attrs


# ── async pipeline ─────────────────────────────────────────────────────────────

async def generate(
    leaves: dict[str, str],
    groups: dict[str, dict],
    model: str,
    batch_size: int,
    workers: int,
    max_clusters: int,
    dry_run: bool,
    existing_paths: set[str],
) -> tuple[dict[str, list[dict]], dict[str, dict]]:

    openai = _require_openai()
    client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(workers)

    todo: list[tuple[str, str, dict, bool]] = []
    skipped = 0
    for path, display_name in leaves.items():
        if path in existing_paths:
            skipped += 1
            continue
        group_data = groups.get(path, {})
        is_low = group_data.get("is_low_structure", True)
        todo.append((path, display_name, group_data, is_low))

    if skipped:
        print(f"  Resuming — skipping {skipped} already-processed categories.")

    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    total_cats   = len(todo)
    total_batches = len(batches)
    print(f"  {total_cats} categories → {total_batches} batches "
          f"(batch_size={batch_size}, workers={workers})")

    if dry_run:
        print("\n=== DRY RUN — first batch prompt ===")
        if batches:
            print(_format_batch(batches[0], max_clusters))
        return {}, {}

    category_attributes: dict[str, list[dict]] = {}
    units: dict[str, dict] = {}
    cats_done = 0
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
            total=total_batches, desc="5b attr+patterns", unit="batch",
            file=sys.stderr, dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]{postfix}",
        )

    lock = asyncio.Lock()

    async def process_batch(batch: list[tuple[str, str, dict, bool]]) -> None:
        nonlocal cats_done, cats_failed
        t0 = time.monotonic()
        result = await _call_llm(client, model, batch, semaphore, max_clusters)
        elapsed = time.monotonic() - t0

        async with lock:
            if result:
                _merge_result(result, batch, category_attributes, units)
                cats_done += len(batch)
            else:
                cats_failed += len(batch)
            batch_times.append(elapsed)

            if pbar:
                done = len(batch_times)
                remaining = total_batches - done
                avg = sum(batch_times) / done
                eta = avg * remaining
                eta_str = f"{int(eta // 60)}m {int(eta % 60)}s" if eta >= 60 else f"{int(eta)}s"
                pbar.set_postfix_str(
                    f"{cats_done} done  {cats_failed} failed  ~{eta_str} left",
                    refresh=True,
                )
                pbar.update(1)
            else:
                status = "ok" if result else "FAILED"
                print(f"  Batch {len(batch_times)}/{total_batches}: "
                      f"{len(batch)} cats [{status}]  done={cats_done} failed={cats_failed}")

    await asyncio.gather(*[process_batch(b) for b in batches])
    if pbar:
        pbar.close()

    total_elapsed = sum(batch_times)
    print(f"\n  Done in {total_elapsed:.1f}s — {cats_done} categories, {cats_failed} failed")
    return category_attributes, units


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _load_dotenv()

    parser = argparse.ArgumentParser(description="Step 5b — LLM attribute + pattern generation")
    parser.add_argument("--groups-dir",  type=Path, default=None,
                        help="title_groups/ dir from step 5a (default: latest)")
    parser.add_argument("--out-dir",     type=Path, default=None,
                        help="Write proposed_attributes.json here (default: groups-dir parent)")
    parser.add_argument("--batch-size",  type=int,  default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers",     type=int,  default=DEFAULT_WORKERS)
    parser.add_argument("--max-clusters",type=int,  default=DEFAULT_MAX_CLUSTERS,
                        help=f"Max clusters sent per category (default: {DEFAULT_MAX_CLUSTERS})")
    parser.add_argument("--model",       type=str,  default=DEFAULT_MODEL)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--resume",      type=Path, default=None,
                        help="Path to partial proposed_attributes.json to resume from")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY") and not args.dry_run:
        sys.exit("OPENAI_API_KEY not set — add it to .env or export it.")

    # ── load title groups
    groups_dir = args.groups_dir or find_latest_groups_dir()
    print(f"Loading title groups from: {groups_dir.relative_to(ROOT)}")
    groups = load_groups(groups_dir)
    print(f"  {len(groups)} categories with group data")

    out_dir: Path = args.out_dir or groups_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    _sfx = _env_suffix()
    out_path = out_dir / f"proposed_attributes{_sfx}.json"

    # ── load taxonomy display names
    leaves = load_leaves(TAXONOMY_PATH)
    print(f"Taxonomy: {len(leaves)} leaf categories")

    # ── resume support
    existing_paths: set[str] = set()
    existing_cat_attrs: dict[str, list[dict]] = {}
    existing_units: dict[str, dict] = {}

    if args.resume and args.resume.exists():
        print(f"Resuming from {args.resume.name}…")
        prev = json.loads(args.resume.read_text())
        existing_cat_attrs = prev.get("_category_attributes", {})
        existing_units     = prev.get("units", {})
        existing_paths     = set(existing_cat_attrs.keys())
        print(f"  {len(existing_paths)} categories already done.")

    # ── run
    new_attrs, new_units = asyncio.run(generate(
        leaves=leaves,
        groups=groups,
        model=args.model,
        batch_size=args.batch_size,
        workers=args.workers,
        max_clusters=args.max_clusters,
        dry_run=args.dry_run,
        existing_paths=existing_paths,
    ))

    if args.dry_run:
        return

    merged_attrs = {**existing_cat_attrs, **new_attrs}
    merged_units = {**existing_units, **new_units}

    # Count pattern coverage
    attrs_with_patterns = sum(
        1 for attrs in merged_attrs.values()
        if any(a.get("patterns") for a in attrs)
    )
    low_structure_paths = [p for p, g in groups.items() if g.get("is_low_structure")]

    output: dict[str, Any] = {
        "_meta": {
            "generated_at":          datetime.now().isoformat(),
            "model":                 args.model,
            "groups_dir":            str(groups_dir),
            "leaves_total":          len(leaves),
            "leaves_with_attributes": len(merged_attrs),
            "attrs_with_patterns":   attrs_with_patterns,
            "low_structure_cats":    len(low_structure_paths),
            "batch_size":            args.batch_size,
            "workers":               args.workers,
            "max_clusters_per_cat":  args.max_clusters,
        },
        "units":                 merged_units,
        "_category_attributes":  merged_attrs,
    }

    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nOutput: {out_path.relative_to(ROOT)}")
    print(f"  {len(merged_attrs)} categories with attributes")
    print(f"  {attrs_with_patterns} categories with at least one extraction pattern")
    print(f"  {len(low_structure_paths)} low-structure categories (schema-only, LLM fallback in 5c)")
    print(f"  {len(merged_units)} unique units")

    (out_dir / f"summary{_sfx}.json").write_text(
        json.dumps({
            "step": "5b-generate-attributes",
            "env": _sfx.lstrip("-") or "unset",
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "model": args.model,
            "output_files": [out_path.name],
            "counts": {
                "categories_with_attributes": len(merged_attrs),
                "categories_with_patterns": attrs_with_patterns,
                "low_structure_categories": len(low_structure_paths),
                "unique_units": len(merged_units),
            },
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # ── consolidate to final-output/
    final_dir = ROOT / "final-output" / out_dir.name
    final_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path, final_dir / f"proposed_attributes{_sfx}.json")
    shutil.copy2(TAXONOMY_PATH, final_dir / "categories_v1.json")
    step4_matched = sorted(
        (ROOT / "step-4-dedupe-and-merge-matched-items" / "outputs").glob("**/matched_deduped*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if step4_matched:
        shutil.copy2(step4_matched[-1], final_dir / f"matched_deduped{_sfx}.json")
    print(f"\nFinal outputs staged → final-output/{out_dir.name}/")


if __name__ == "__main__":
    main()
