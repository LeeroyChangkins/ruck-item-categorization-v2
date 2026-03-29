#!/usr/bin/env python3
"""
Step 1.4
LLM match remaining unmatched items to leaf categories (categories_v1).

Inputs:
  - source-files/categories_v1.json (authoritative leaf paths)
  - step-1.2 split: unmatched_and_keywords.json + matched.json (same folder)
  - Optional step-1.3 manual JSON: item ids matched manually are excluded from the LLM pool
  - .env for OPENAI_API_KEY

Outputs (same timestamp):
  - step-1.4/outputs/1.4-llm_matched_YYYYMMDD_HHMMSS.json
  - step-1.4/outputs/1.4-llm_unmatched_YYYYMMDD_HHMMSS.json

Behavior:
  - Unmatched pool = unmatched_items from the 1.2 split file, minus ids in 1.3 item_matches
    (if a manual file is provided or auto-discovered).
  - LLM chooses exactly one leaf_path per item; keep predictions with confidence >= min (default 0.9).
  - matched file: all items matched in step 1.2 (matched.json) + 1.3 item_matches + new LLM assignments.
  - unmatched file: items still unmatched after 1.3 exclusion and LLM (below threshold or invalid leaf).
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
from typing import Any, Dict, Iterable, List, Optional, Set


ROOT = Path(__file__).resolve().parents[1]  # v2/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taxonomy_cascade import leaf_path_is_catch_all_bucket
TAXONOMY_PATH = ROOT / "source-files" / "categories_v1.json"
STEP12_OUT = ROOT / "step-1.2" / "outputs"
STEP13_OUT = ROOT / "step-1.3" / "outputs"
STEP14_OUTDIR = Path(__file__).resolve().parent / "outputs"
ENV_PATH = ROOT / ".env"

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MIN_CONF = 0.9
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_env_dotfile(path: Path) -> None:
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


def require_openai() -> None:
    try:
        import openai  # noqa: F401
    except Exception:
        raise SystemExit("Missing dependency: openai. Install with `pip install openai` first.")


def all_leaf_paths(categories: dict) -> List[str]:
    """Leaf paths as slash-joined slug chains (materials/.../leaf_slug)."""
    leaves: List[str] = []
    roots = ["materials", "tools_and_gear", "services"]

    def walk(node: Any, prefix: List[str]) -> None:
        if not isinstance(node, dict):
            return
        slug = node.get("slug")
        subs = node.get("subcategories") or []

        if slug is not None:
            here = prefix + [slug]
        else:
            here = prefix

        if not subs:
            if slug is not None:
                leaves.append("/".join(here))
            return
        for child in subs:
            walk(child, here)

    for root in roots:
        root_obj = categories.get(root)
        if not isinstance(root_obj, dict):
            continue
        for child in root_obj.get("subcategories", []):
            walk(child, [root])

    leaves.sort()
    return [p for p in leaves if not leaf_path_is_catch_all_bucket(p)]


def newest_unmatched_split_file() -> Path | None:
    cands = list(STEP12_OUT.glob("1.2_split_*/unmatched_and_keywords.json"))
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def find_latest_manual_for_source(out_dir: Path, in_path: Path) -> Path | None:
    """Newest 1.3 manual JSON whose unmatched_keywords_source matches the split file."""
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


def pick_inputs_interactive() -> tuple[Path, Optional[Path]]:
    """Return (unmatched_and_keywords path, optional manual 1.3 path)."""
    latest = newest_unmatched_split_file()
    print("Choose step-1.2 split unmatched_and_keywords.json:")
    if latest:
        print(f"  1) Use most recent: {latest.relative_to(ROOT)}")
    else:
        print("  1) (none found under step-1.2/outputs/1.2_split_*)")
    print("  2) Enter a path manually")
    choice = input("> ").strip()
    if choice == "1" and latest:
        in_path = latest
    elif choice == "2":
        in_path = Path(input("Path to unmatched_and_keywords.json: ").strip()).expanduser().resolve()
        if not in_path.exists():
            raise SystemExit(f"File not found: {in_path}")
    else:
        raise SystemExit("Invalid selection.")

    manual: Optional[Path] = None
    found = find_latest_manual_for_source(STEP13_OUT, in_path)
    if found:
        print(f"\nFound 1.3 manual for this split: {found.relative_to(ROOT)}")
        use = input("Use it to exclude already-matched items? [Y/n] ").strip().lower()
        if use in ("", "y", "yes"):
            manual = found
    else:
        print("\nNo matching step-1.3 manual file found; all split unmatched items go to the LLM pool.")

    return in_path, manual


def chunked(items: List[dict], size: int) -> Iterable[List[dict]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def leaf_slug_from_path(leaf_path: str) -> str:
    parts = leaf_path.strip("/").split("/")
    return parts[-1] if parts else ""


def call_llm_for_batch(
    client: Any,
    model: str,
    leaf_paths: List[str],
    items_batch: List[dict],
    min_conf: float,
) -> List[dict]:
    leaf_lines = "\n".join([f"- {lp}" for lp in leaf_paths])
    item_lines = "\n".join([f"- id={it['id']} title={it['title']} subtitle={it['subtitle']}" for it in items_batch])

    prompt = f"""You are categorizing construction products into a strict taxonomy of leaf categories.

You must choose EXACTLY ONE leaf category from the provided list for each item.
If you are not confident, set confidence below {min_conf}.

Return JSON only: a JSON array of objects, one per item:
  [{{"id": "...", "leaf_path": "...", "confidence": <number 0.0 to 1.0>}}, ...]

Rules:
- leaf_path must be exactly one string from the leaf list.
- confidence >= {min_conf} means high-confidence assignment.

Leaf categories (valid options):
{leaf_lines}

Items to categorize:
{item_lines}
"""

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

    text = str(text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        if "```" in text:
            mid = text.split("```", 2)[1].strip()
            if mid.lower().startswith("json"):
                mid = mid[4:].strip()
            data = json.loads(mid)
        else:
            raise

    if not isinstance(data, list):
        raise ValueError("LLM output was not a JSON array")

    valid_leaf_set = set(leaf_paths)
    out: List[dict] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        iid = obj.get("id")
        lp = obj.get("leaf_path")
        conf = obj.get("confidence")
        if not isinstance(iid, str):
            continue
        if not (isinstance(lp, str) and lp in valid_leaf_set):
            continue
        try:
            conf_f = float(conf)
        except Exception:
            continue
        if conf_f < 0.0 or conf_f > 1.0:
            continue
        out.append({"id": iid, "leaf_path": lp, "confidence": conf_f})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1.4: LLM leaf match for remaining unmatched items.")
    parser.add_argument(
        "--input",
        metavar="PATH",
        help="Path to step-1.2 split unmatched_and_keywords.json",
    )
    parser.add_argument(
        "--manual-13",
        metavar="PATH",
        help="Step-1.3 manual JSON (item_matches excluded from LLM pool). Default: auto if unique match.",
    )
    parser.add_argument("--no-auto-manual", action="store_true", help="Do not auto-pick a 1.3 manual file.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONF)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--no-resume", action="store_true", help="Disable checkpoint resume.")
    parser.add_argument(
        "--checkpoint-every-batch",
        type=int,
        default=1,
        help="Save progress every N LLM batches.",
    )
    args = parser.parse_args()

    STEP14_OUTDIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    load_env_dotfile(ENV_PATH)

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not found in environment or .env")

    require_openai()
    from openai import OpenAI

    client = OpenAI()

    with TAXONOMY_PATH.open("r", encoding="utf-8") as f:
        categories = json.load(f)

    leaf_paths = all_leaf_paths(categories)
    if not leaf_paths:
        raise SystemExit("No leaf paths found in taxonomy.")

    if args.input:
        in_path = Path(args.input).expanduser().resolve()
        manual_13: Optional[Path] = None
        if args.manual_13:
            manual_13 = Path(args.manual_13).expanduser().resolve()
        elif not args.no_auto_manual:
            found = find_latest_manual_for_source(STEP13_OUT, in_path)
            if found:
                manual_13 = found
                print(f"Using auto-discovered 1.3 manual: {manual_13.relative_to(ROOT)}")
    else:
        in_path, manual_interactive = pick_inputs_interactive()
        manual_13 = manual_interactive

    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    matched_12_path = in_path.parent / "matched.json"
    if not matched_12_path.exists():
        raise SystemExit(
            f"Expected sibling matched.json next to the split input:\n  {matched_12_path}\n"
            "Use the same 1.2_split_* folder that contains unmatched_and_keywords.json and matched.json."
        )

    with in_path.open("r", encoding="utf-8") as f:
        split_data = json.load(f)

    unmatched_items_in = split_data.get("unmatched_items")
    if not isinstance(unmatched_items_in, list):
        raise SystemExit("Expected unmatched_items array in unmatched_and_keywords.json.")

    with matched_12_path.open("r", encoding="utf-8") as f:
        matched_12_raw = json.load(f)

    if not isinstance(matched_12_raw, list):
        raise SystemExit("Expected matched.json to be a JSON array.")

    matched_items_12: List[dict] = [x for x in matched_12_raw if isinstance(x, dict)]

    ids_13: Set[str] = set()
    item_matches_13: List[dict] = []
    if manual_13 is not None:
        if not manual_13.exists():
            raise SystemExit(f"--manual-13 not found: {manual_13}")
        mdata = json.loads(manual_13.read_text(encoding="utf-8"))
        prev_src = mdata.get("unmatched_keywords_source")
        if isinstance(prev_src, str) and Path(prev_src).resolve() != in_path.resolve():
            raise SystemExit(
                f"1.3 manual file was built from a different split than --input:\n"
                f"  manual: {prev_src}\n"
                f"  input:  {in_path}"
            )
        im = mdata.get("item_matches")
        if isinstance(im, list):
            for row in im:
                if not isinstance(row, dict):
                    continue
                iid = row.get("id")
                if isinstance(iid, str) and iid:
                    ids_13.add(iid)
                    item_matches_13.append(row)

    llm_inputs: List[dict] = []
    for it in unmatched_items_in:
        if not isinstance(it, dict):
            continue
        iid = it.get("id")
        title = it.get("title") or ""
        subtitle = it.get("subtitle") or ""
        if not isinstance(iid, str) or not iid:
            continue
        if iid in ids_13:
            continue
        llm_inputs.append({"id": iid, "title": title, "subtitle": subtitle})

    print(f"Split unmatched items: {len(unmatched_items_in)}")
    print(f"Matched in 1.2 (matched.json): {len(matched_items_12)}")
    print(f"Excluded by 1.3 manual: {len(ids_13)}")
    print(f"Items to send to LLM: {len(llm_inputs)}")
    print(f"Min confidence: {args.min_confidence}")

    def file_fingerprint(path: Path) -> str:
        st = path.stat()
        raw = f"{path.as_posix()}|{st.st_size}|{int(st.st_mtime)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def ckpt_path_for(base_fingerprint: str) -> Path:
        return CHECKPOINT_DIR / f"step14_checkpoint_{hashlib.sha256(base_fingerprint.encode('utf-8')).hexdigest()[:10]}.json"

    manual_fp = file_fingerprint(manual_13) if manual_13 else "none"
    base_fp = (
        f"taxonomy={file_fingerprint(TAXONOMY_PATH)}"
        f"|input={file_fingerprint(in_path)}"
        f"|matched12={file_fingerprint(matched_12_path)}"
        f"|manual13={manual_fp}"
        f"|model={args.model}"
        f"|min_conf={args.min_confidence}"
        f"|batch_size={args.batch_size}"
    )
    ckpt_path = ckpt_path_for(base_fp)

    kept_by_id: Dict[str, dict] = {}
    next_batch_index = 0
    if (not args.no_resume) and ckpt_path.exists():
        ck = json.loads(ckpt_path.read_text(encoding="utf-8"))
        if ck.get("fingerprint") == base_fp:
            next_batch_index = int(ck.get("next_batch_index", 0))
            kept_by_id = ck.get("kept_by_id", {}) or {}
            if not isinstance(kept_by_id, dict):
                kept_by_id = {}
            print(f"Resume: next_batch_index={next_batch_index} kept_so_far={len(kept_by_id)}")

    start = time.time()

    for batch_index, batch in enumerate(chunked(llm_inputs, args.batch_size), start=0):
        if batch_index < next_batch_index:
            continue

        bi = batch_index + 1
        print(f"Batch {bi}: size={len(batch)} ...")
        preds = call_llm_for_batch(
            client=client,
            model=args.model,
            leaf_paths=leaf_paths,
            items_batch=batch,
            min_conf=args.min_confidence,
        )

        for p in preds:
            if p["confidence"] >= args.min_confidence:
                kept_by_id[p["id"]] = p
        kept_total = len(kept_by_id)

        if args.checkpoint_every_batch <= 1 or (bi % args.checkpoint_every_batch == 0):
            tmp = ckpt_path.with_suffix(".tmp")
            payload = {
                "fingerprint": base_fp,
                "next_batch_index": batch_index + 1,
                "kept_by_id": kept_by_id,
            }
            tmp.write_text(json.dumps(payload, indent=0, ensure_ascii=False) + "\n", encoding="utf-8")
            tmp.replace(ckpt_path)
            print(f"Checkpoint saved at batch {bi}")

        if args.sleep_seconds:
            time.sleep(args.sleep_seconds)

    try:
        if ckpt_path.exists():
            ckpt_path.unlink()
    except Exception:
        pass

    elapsed = time.time() - start
    print(f"LLM high-confidence assignments kept: {len(kept_by_id)} in {elapsed:.1f}s")

    existing_ids: Set[str] = set()
    for m in matched_items_12:
        if isinstance(m.get("id"), str) and m["id"]:
            existing_ids.add(m["id"])
    for m in item_matches_13:
        if isinstance(m.get("id"), str) and m["id"]:
            existing_ids.add(m["id"])

    llm_matched_rows: List[dict] = []
    llm_added = 0
    for inp in llm_inputs:
        iid = inp["id"]
        if iid not in kept_by_id:
            continue
        if iid in existing_ids:
            continue

        pred = kept_by_id[iid]
        leaf = pred["leaf_path"]
        conf = float(pred["confidence"])
        slug = leaf_slug_from_path(leaf)

        row = {
            "id": iid,
            "title": inp["title"],
            "subtitle": inp["subtitle"],
            "leaf_path": leaf,
            "leaf_slug": slug,
            "leaf_display_name": "",
            "confidence": conf,
            "source": "llm_1_4",
        }
        llm_matched_rows.append(row)
        existing_ids.add(iid)
        llm_added += 1

    merged_matched: List[dict] = []
    merged_matched.extend(matched_items_12)
    merged_matched.extend(item_matches_13)
    merged_matched.extend(llm_matched_rows)

    remaining_unmatched: List[dict] = []
    for it in unmatched_items_in:
        if not isinstance(it, dict):
            continue
        iid = it.get("id")
        if not isinstance(iid, str) or not iid:
            continue
        if iid in ids_13:
            continue
        if iid in kept_by_id:
            continue
        remaining_unmatched.append(it)

    ts = timestamp()
    common_meta = {
        "version": "1.4",
        "llm_model": args.model,
        "min_confidence": args.min_confidence,
        "taxonomy_categories_file": TAXONOMY_PATH.name,
        "unmatched_keywords_source": str(in_path.resolve()),
        "matched_json_source": str(matched_12_path.resolve()),
        "manual_13_source": str(manual_13.resolve()) if manual_13 else None,
        "leaf_paths_count": len(leaf_paths),
        "counts": {
            "matched_1_2_bigram": len(matched_items_12),
            "matched_1_3_manual": len(item_matches_13),
            "matched_1_4_llm": llm_added,
            "merged_matched_total": len(merged_matched),
            "unmatched_after_llm": len(remaining_unmatched),
        },
        "merge_rule": (
            "matched file: matched.json (1.2) + item_matches (1.3) + LLM rows (>= min_confidence). "
            "unmatched file: split unmatched_items minus 1.3 ids minus LLM-kept ids."
        ),
    }

    out_matched = STEP14_OUTDIR / f"1.4-llm_matched_{ts}.json"
    out_unmatched = STEP14_OUTDIR / f"1.4-llm_unmatched_{ts}.json"

    payload_matched = {**common_meta, "matched_items": merged_matched}
    payload_unmatched = {**common_meta, "unmatched_items": remaining_unmatched}

    out_matched.write_text(json.dumps(payload_matched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_unmatched.write_text(json.dumps(payload_unmatched, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nWrote {out_matched.relative_to(ROOT)}")
    print(f"Wrote {out_unmatched.relative_to(ROOT)}")
    print(f"LLM new matched rows: {llm_added}")
    print(f"Remaining unmatched items: {len(remaining_unmatched)}")


if __name__ == "__main__":
    main()
