#!/usr/bin/env python3
"""
Step 5c — Attribute Value Extraction.

Two-pass approach:
  Pass 1 — Regex  : for every item whose title matches a known structural
            template, apply the extraction patterns produced by step 5b.
            Successful extractions are tagged extraction_confidence="regex".

  Pass 2 — LLM fallback : items that produced zero attribute values in pass 1
            (no patterns matched, low-structure category, or patterns missing
            for some attributes) are batched and sent to the LLM.  The LLM is
            asked to extract values from the raw title only.
            Only LLM responses rated "high" or "medium" confidence are kept.
            Items rated "low"/"none" are written to unextracted_values.json
            but never to the main output.

Confidence gating
-----------------
  "regex"          — regex fired and captured a non-empty, plausible value
  "llm_high"       — LLM returned confidence "high"
  "llm_medium"     — LLM returned confidence "medium"
  "llm_low"        — LLM returned confidence "low" — EXCLUDED from main output
  "llm_none"       — LLM returned confidence "none" — EXCLUDED from main output

Output
------
  <out_dir>/item_attribute_values.json      ← rows uploaded to DB
  <out_dir>/unextracted_values.json         ← low/none confidence — review only
  <out_dir>/extraction_stats.json           ← summary counts

item_attribute_values.json row schema:
  {
    "item_id":               "itm_abc123",
    "leaf_path":             "materials/metals_and_metal_fabrication/metal_tubing",
    "attribute_key":         "outer_diameter",
    "value":                 "1 3/4",
    "unit":                  "in",
    "extraction_confidence": "regex"
  }

Usage
-----
  python 5c_extract_attribute_values.py [OPTIONS]

  --attributes PATH   proposed_attributes.json from step 5b (default: latest)
  --groups-dir PATH   title_groups/ dir from step 5a (default: sibling of attributes)
  --matched    PATH   matched_deduped.json from step 4 (default: latest)
  --out-dir    PATH   write outputs here (default: same dir as attributes file)
  --llm-batch  INT    items per LLM fallback call (default: 20)
  --workers    INT    concurrent LLM requests (default: 10)
  --model      STR    OpenAI model for fallback (default: gpt-4o)
  --no-llm            Skip LLM fallback; only regex extraction
  --dry-run           Print stats without writing files
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STEP5_DIR = Path(__file__).resolve().parent
STEP4_OUTPUTS = ROOT / "step-4-dedupe-and-merge-matched-items" / "outputs"

sys.path.insert(0, str(ROOT))
import shared_utils as _su
from shared_utils import load_dotenv_file as _load_dotenv, write_step_summary, env_suffix

DEFAULT_MODEL       = "gpt-4o"
DEFAULT_LLM_BATCH   = 20
DEFAULT_WORKERS     = 10

_ACCEPTED_CONFIDENCE = {"regex", "llm_high", "llm_medium"}
_REJECTED_CONFIDENCE = {"llm_low", "llm_none"}


# ── value normaliser ──────────────────────────────────────────────────────────

_INCH_STRIP = re.compile(r'[""]|(?<!\w)in\.?(?!\w)|inches?', re.IGNORECASE)
_FEET_STRIP = re.compile(r"[''`]|(?<!\w)ft\.?(?!\w)|feet?", re.IGNORECASE)


def normalise_value(raw: str, hint: str) -> str:
    v = raw.strip()
    if hint == "measurement_inches":
        v = _INCH_STRIP.sub("", v).strip()
    elif hint == "measurement_feet":
        v = _FEET_STRIP.sub("", v).strip()
    elif hint == "text_lower":
        v = v.lower().strip()
    # "text_raw" → unchanged
    # Reject obviously empty or suspiciously long captures
    if not v or len(v) > 120:
        return ""
    return v


# ── title normalisation (mirrors 5a) ─────────────────────────────────────────

_FRACTION_RE    = re.compile(r"\b\d+\s*/\s*\d+\b")
_DECIMAL_RE     = re.compile(r"\b\d+\.\d+\b")
_LEADING_DEC_RE = re.compile(r"(?<![.\d])\.\d+\b")
_INTEGER_RE     = re.compile(r"\b\d+\b")


def normalise_title(title: str) -> str:
    t = title
    t = _FRACTION_RE.sub("{FRAC}", t)
    t = _DECIMAL_RE.sub("{NUM}", t)
    t = _LEADING_DEC_RE.sub("{NUM}", t)
    t = _INTEGER_RE.sub("{NUM}", t)
    return re.sub(r"\s+", " ", t).strip()


# ── file finders ──────────────────────────────────────────────────────────────

def find_latest_attributes() -> Path:
    outputs = STEP5_DIR / "outputs"
    candidates = list(outputs.rglob("proposed_attributes*.json"))
    if not candidates:
        sys.exit("No proposed_attributes*.json found — run step 5b first.")
    # filter by parent dir (timestamped run folder) env suffix
    result = _su.latest_env_path(candidates, name_attr="parent")
    return result or candidates[0]


def find_latest_matched() -> Path:
    candidates = list(STEP4_OUTPUTS.rglob("matched_deduped.json"))
    if not candidates:
        sys.exit("No matched_deduped.json found — run step 4 first.")
    result = _su.latest_env_path(candidates, name_attr="parent")
    return result or candidates[0]


def find_groups_dir(attributes_path: Path) -> Path | None:
    candidate = attributes_path.parent / "title_groups"
    return candidate if candidate.is_dir() else None


# ── load helpers ──────────────────────────────────────────────────────────────

def load_attributes(path: Path) -> dict[str, list[dict]]:
    data = json.loads(path.read_text())
    return data.get("_category_attributes", {})


def load_matched_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data.get("matched_items") or data.get("items") or []


def load_template_map(groups_dir: Path) -> dict[str, dict[str, str]]:
    """
    Returns leaf_path → { normalised_template → representative_title }.
    Used to find which cluster an item belongs to.
    """
    manifest_path = groups_dir / "_manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text())
    result: dict[str, dict[str, str]] = {}
    for entry in manifest.get("categories", []):
        leaf = entry["leaf_path"]
        fname = entry["file"]
        cat_file = groups_dir / fname
        if not cat_file.exists():
            continue
        cat_data = json.loads(cat_file.read_text())
        result[leaf] = {c["template"]: c["representative"] for c in cat_data.get("clusters", [])}
    return result


# ── regex extraction pass ────────────────────────────────────────────────────

def regex_extract_item(
    item: dict,
    attributes: list[dict],
    item_template: str,
) -> list[dict]:
    """
    Returns a list of extracted value rows for this item.
    Only includes rows where the regex produced a non-empty value.
    """
    title = item.get("title", "")
    item_id = item.get("id") or item.get("item_id") or ""
    leaf_path = item.get("leaf_path") or item.get("leaf_slug") or ""
    rows: list[dict] = []

    for attr in attributes:
        patterns = attr.get("patterns") or []
        matched = False
        for pat in patterns:
            if pat.get("template", "") != item_template:
                continue
            regex_str = pat.get("regex", "")
            hint = pat.get("value_normalize", "text_raw")
            try:
                m = re.search(regex_str, title)
            except re.error:
                continue
            if not m:
                continue
            try:
                raw_val = m.group("val")
            except IndexError:
                continue
            val = normalise_value(raw_val, hint)
            if not val:
                continue
            rows.append({
                "item_id":               item_id,
                "leaf_path":             leaf_path,
                "attribute_key":         attr["key"],
                "value":                 val,
                "unit":                  attr.get("unit"),
                "extraction_confidence": "regex",
            })
            matched = True
            break  # one pattern per attribute is enough

    return rows


# ── LLM fallback ──────────────────────────────────────────────────────────────

LLM_FALLBACK_SYSTEM = """\
You are a product-data extraction assistant for a construction-materials marketplace.

Given a list of item titles and the known attribute schema for their category, extract
attribute values directly from each title.

Rules:
- Extract ONLY what you can confidently read from the title text.
- Do NOT invent or infer values that aren't visible in the title.
- For numeric measurements keep the raw string as written (e.g. "1 3/4" not 1.75).
- For each extracted value report a confidence: "high", "medium", "low", or "none".
  - "high"   : the value is clearly and unambiguously present in the title.
  - "medium" : the value is likely present but requires minor interpretation.
  - "low"    : uncertain — the value might not be correct.
  - "none"   : cannot extract this attribute from this title.
- Return ONLY valid JSON, no markdown, no extra text.

Response schema:
{
  "results": [
    {
      "item_id": "...",
      "attributes": [
        {
          "attribute_key": "...",
          "value": "...",
          "unit": "..." | null,
          "confidence": "high" | "medium" | "low" | "none"
        }
      ]
    }
  ]
}
"""


def _build_llm_fallback_prompt(
    items: list[dict],
    attributes: list[dict],
    leaf_path: str,
) -> str:
    attr_schema = []
    for a in attributes:
        unit_str = f" (unit: {a['unit']})" if a.get("unit") else ""
        attr_schema.append(f"  - {a['key']}: {a['label']}{unit_str} [{a['value_type']}]")

    lines = [
        f"Category: {leaf_path}",
        "Attribute schema:",
        *attr_schema,
        "",
        "Items to extract from:",
    ]
    for item in items:
        iid  = item.get("id") or item.get("item_id") or "?"
        title = item.get("title", "")
        lines.append(f"  item_id={iid}  title={json.dumps(title)}")
    return "\n".join(lines)


async def _llm_fallback_batch(
    client,
    model: str,
    items: list[dict],
    attributes: list[dict],
    leaf_path: str,
    semaphore: asyncio.Semaphore,
    retries: int = 3,
) -> dict | None:
    user_msg = _build_llm_fallback_prompt(items, attributes, leaf_path)
    for attempt in range(retries):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": LLM_FALLBACK_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.0,
                )
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:
            wait = 2 ** attempt
            await asyncio.sleep(wait)
    return None


async def llm_fallback_pass(
    unmatched_items: list[dict],
    attributes_by_leaf: dict[str, list[dict]],
    model: str,
    llm_batch_size: int,
    workers: int,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (accepted_rows, rejected_rows).
    accepted  = confidence in {high, medium}
    rejected  = confidence in {low, none}
    """
    if not unmatched_items:
        return [], []

    def _require_openai():
        try:
            import openai
            return openai
        except ImportError:
            sys.exit("openai not installed")

    openai_lib = _require_openai()
    client = openai_lib.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(workers)

    # Group by leaf_path for coherent prompts
    by_leaf: dict[str, list[dict]] = {}
    for item in unmatched_items:
        leaf = item.get("leaf_path") or item.get("leaf_slug") or ""
        by_leaf.setdefault(leaf, []).append(item)

    accepted: list[dict] = []
    rejected: list[dict] = []
    batch_times: list[float] = []

    tasks: list[tuple[list[dict], list[dict], str]] = []
    for leaf, items in by_leaf.items():
        attrs = attributes_by_leaf.get(leaf, [])
        if not attrs:
            continue
        for i in range(0, len(items), llm_batch_size):
            tasks.append((items[i:i + llm_batch_size], attrs, leaf))

    print(f"\n  LLM fallback: {len(unmatched_items)} items → {len(tasks)} batches")

    try:
        from tqdm import tqdm as _tqdm
        use_tqdm = sys.stderr.isatty()
    except ImportError:
        use_tqdm = False

    pbar = None
    if use_tqdm:
        pbar = _tqdm(
            total=len(tasks), desc="5c llm fallback", unit="batch",
            file=sys.stderr, dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]{postfix}",
        )

    lock = asyncio.Lock()
    acc_count = 0
    rej_count = 0

    async def process(batch_items: list[dict], attrs: list[dict], leaf: str) -> None:
        nonlocal acc_count, rej_count
        t0 = time.monotonic()
        result = await _llm_fallback_batch(client, model, batch_items, attrs, leaf, semaphore)
        elapsed = time.monotonic() - t0

        async with lock:
            batch_times.append(elapsed)
            if result:
                for r in result.get("results", []):
                    iid  = r.get("item_id", "")
                    for av in r.get("attributes", []):
                        conf_raw = av.get("confidence", "none").strip().lower()
                        conf_key = f"llm_{conf_raw}" if not conf_raw.startswith("llm_") else conf_raw
                        val = (av.get("value") or "").strip()
                        if not val or conf_raw == "none":
                            conf_key = "llm_none"
                        row: dict[str, Any] = {
                            "item_id":               iid,
                            "leaf_path":             leaf,
                            "attribute_key":         av.get("attribute_key", ""),
                            "value":                 val,
                            "unit":                  av.get("unit"),
                            "extraction_confidence": conf_key,
                        }
                        if conf_key in _ACCEPTED_CONFIDENCE:
                            accepted.append(row)
                            acc_count += 1
                        else:
                            rejected.append(row)
                            rej_count += 1

            if pbar:
                done = len(batch_times)
                rem  = len(tasks) - done
                avg  = sum(batch_times) / done
                eta  = avg * rem
                eta_str = f"{int(eta//60)}m{int(eta%60)}s" if eta >= 60 else f"{int(eta)}s"
                pbar.set_postfix_str(f"acc={acc_count} rej={rej_count} ~{eta_str}", refresh=True)
                pbar.update(1)

    await asyncio.gather(*[process(bi, at, lf) for bi, at, lf in tasks])
    if pbar:
        pbar.close()

    return accepted, rejected


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _load_dotenv()

    parser = argparse.ArgumentParser(description="Step 5c — attribute value extraction")
    parser.add_argument("--attributes", type=Path, default=None,
                        help="proposed_attributes.json from step 5b (default: latest)")
    parser.add_argument("--groups-dir", type=Path, default=None,
                        help="title_groups/ from step 5a (default: sibling of attributes)")
    parser.add_argument("--matched",    type=Path, default=None,
                        help="matched_deduped.json from step 4 (default: latest)")
    parser.add_argument("--out-dir",    type=Path, default=None,
                        help="Write outputs here (default: same dir as attributes file)")
    parser.add_argument("--llm-batch",  type=int,  default=DEFAULT_LLM_BATCH)
    parser.add_argument("--workers",    type=int,  default=DEFAULT_WORKERS)
    parser.add_argument("--model",      type=str,  default=DEFAULT_MODEL)
    parser.add_argument("--no-llm",     action="store_true",
                        help="Skip LLM fallback pass")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print stats, do not write files")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY") and not args.no_llm and not args.dry_run:
        sys.exit("OPENAI_API_KEY not set — add it to .env or use --no-llm.")

    # ── load inputs
    attr_path = args.attributes or find_latest_attributes()
    print(f"Loading attributes from: {attr_path.relative_to(ROOT)}")
    attributes_by_leaf = load_attributes(attr_path)
    print(f"  {len(attributes_by_leaf)} categories with attributes")

    groups_dir_path = args.groups_dir or find_groups_dir(attr_path)
    template_map: dict[str, dict[str, str]] = {}
    if groups_dir_path and groups_dir_path.is_dir():
        print(f"Loading title groups from: {groups_dir_path.relative_to(ROOT)}")
        template_map = load_template_map(groups_dir_path)
        print(f"  {len(template_map)} categories with template clusters")
    else:
        print("  No title_groups/ found — regex pass will be skipped, LLM-only extraction.")

    matched_path = args.matched or find_latest_matched()
    print(f"Loading matched items from: {matched_path.relative_to(ROOT)}")
    items = load_matched_items(matched_path)
    print(f"  {len(items)} matched items")

    out_dir: Path = args.out_dir or attr_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── pass 1: regex extraction
    print(f"\nPass 1 — regex extraction…")
    accepted_rows:  list[dict] = []
    unmatched_items: list[dict] = []

    regex_matched_count = 0
    no_pattern_count    = 0

    for item in items:
        leaf_path = item.get("leaf_path") or item.get("leaf_slug") or ""
        attributes = attributes_by_leaf.get(leaf_path, [])
        if not attributes:
            continue

        title    = item.get("title", "")
        tmpl     = normalise_title(title)
        cat_tmpl = template_map.get(leaf_path, {})

        # Check if this item's template is a known cluster template
        if tmpl in cat_tmpl:
            rows = regex_extract_item(item, attributes, tmpl)
            if rows:
                accepted_rows.extend(rows)
                regex_matched_count += 1
                continue

        # No regex match — flag for LLM fallback
        # Attach leaf_path for the LLM pass
        item_copy = dict(item)
        item_copy["leaf_path"] = leaf_path
        unmatched_items.append(item_copy)
        no_pattern_count += 1

    print(f"  Regex matched: {regex_matched_count} items  ({len(accepted_rows)} value rows)")
    print(f"  Sent to LLM fallback: {no_pattern_count} items")

    # ── pass 2: LLM fallback
    llm_accepted:  list[dict] = []
    llm_rejected:  list[dict] = []

    if unmatched_items and not args.no_llm and not args.dry_run:
        llm_accepted, llm_rejected = asyncio.run(llm_fallback_pass(
            unmatched_items=unmatched_items,
            attributes_by_leaf=attributes_by_leaf,
            model=args.model,
            llm_batch_size=args.llm_batch,
            workers=args.workers,
        ))
        print(f"  LLM accepted (high/medium): {len(llm_accepted)} value rows")
        print(f"  LLM rejected (low/none):    {len(llm_rejected)} value rows (written to unextracted_values.json)")
    elif args.no_llm:
        print("  LLM fallback skipped (--no-llm).")

    all_accepted = accepted_rows + llm_accepted

    # ── stats
    stats: dict[str, Any] = {
        "total_items":           len(items),
        "regex_matched_items":   regex_matched_count,
        "llm_fallback_items":    no_pattern_count,
        "total_value_rows":      len(all_accepted),
        "regex_value_rows":      len(accepted_rows),
        "llm_accepted_rows":     len(llm_accepted),
        "llm_rejected_rows":     len(llm_rejected),
        "by_confidence": {
            "regex":      len(accepted_rows),
            "llm_high":   sum(1 for r in llm_accepted if r["extraction_confidence"] == "llm_high"),
            "llm_medium": sum(1 for r in llm_accepted if r["extraction_confidence"] == "llm_medium"),
            "llm_low":    sum(1 for r in llm_rejected if r["extraction_confidence"] == "llm_low"),
            "llm_none":   sum(1 for r in llm_rejected if r["extraction_confidence"] == "llm_none"),
        },
    }

    print(f"\nExtraction summary:")
    print(f"  Total value rows (accepted)  : {stats['total_value_rows']}")
    print(f"    regex                      : {stats['by_confidence']['regex']}")
    print(f"    llm_high                   : {stats['by_confidence']['llm_high']}")
    print(f"    llm_medium                 : {stats['by_confidence']['llm_medium']}")
    print(f"  Rejected (excluded from DB)  :")
    print(f"    llm_low                    : {stats['by_confidence']['llm_low']}")
    print(f"    llm_none                   : {stats['by_confidence']['llm_none']}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # ── write outputs
    suf = env_suffix()
    values_path   = out_dir / f"item_attribute_values{suf}.json"
    rejected_path = out_dir / f"unextracted_values{suf}.json"
    stats_path    = out_dir / f"extraction_stats{suf}.json"

    values_path.write_text(
        json.dumps(all_accepted, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote: {values_path.relative_to(ROOT)}")

    if llm_rejected:
        rejected_path.write_text(
            json.dumps(llm_rejected, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote: {rejected_path.relative_to(ROOT)}  (low/none confidence — not uploaded)")

    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote: {stats_path.relative_to(ROOT)}")

    write_step_summary(
        out_dir,
        step="step-5-attribute-generation-and-unit-value-assignment (5c extraction)",
        stats=stats,
        output_files=[values_path.name]
            + ([rejected_path.name] if llm_rejected else [])
            + [stats_path.name],
    )

    # Copy to final-output/
    final_dir = ROOT / "final-output" / out_dir.name
    if final_dir.exists():
        import shutil
        shutil.copy2(values_path, final_dir / f"item_attribute_values{suf}.json")
        print(f"Copied item_attribute_values{suf}.json → final-output/{out_dir.name}/")


if __name__ == "__main__":
    main()
