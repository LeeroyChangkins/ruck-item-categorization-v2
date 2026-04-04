#!/usr/bin/env python3
"""
Master controller for Categorization v2 pipeline.

Flow (high level):
  Step 1 — similar-title grouping (raw catalog)
    1.1  step-1-similar-title-groups/1_1_build_similar_title_groups.py
    1.2  step-1-similar-title-groups/1_2_interactive_similar_title_match.py
         → unmatched_after_step1.json for step 2
  Step 2 — keywords, bigrams, cascade match, optional manual bigram→leaf
    2.1a step-2/2_1_generate_keywords.py [--items-json …]
    2.1b step-2/2_1_generate_bigrams_taxonomy.py or 2_1_generate_bigrams_openai.py
    2.2  step-2/2_2_match_items_to_bigrams.py [--items-json …]
    2.3  step-2/2_3_interactive_keyword_match.py (optional)
  Step 3 — LLM: step-3-llm-matching/3_llm_match_unmatched.py [--step1-manual …]
  Step 4 — Dedupe: step-4/4_dedupe_and_summaries.py
  Step 5 — Attributes:
    5a  step-5/5a_group_title_templates.py   (title structural clustering)
    5b  step-5/5_generate_attributes.py      (LLM attribute schema + regex patterns)
    5c  step-5/5c_extract_attribute_values.py (regex + LLM fallback value extraction)
  Step 6 — DB upload: step-6-db-upload/6_upload_to_db.py (run separately; requires SSM tunnel)

CLI --start-step uses 1.1, 1.2, 2.1, 2.2, 2.3, 3, 4, 5, 6 (first step to run). Legacy aliases:
  1.0→2.1, 1.2→2.2, 1.3→2.3, 1.4→3, 1.5→4. (Old “start at bigrams only” → use --start-step 2.1.)

Resume: starting at 1.1 defaults to fresh --no-resume / --fresh-run where supported; later starts
default to resume with optional “start fresh” per sub-step.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import shared_utils
from shared_utils import load_dotenv_file as _load_dotenv
_load_dotenv()

from pipeline_paths import newest_under_step1

ROOT = Path(__file__).resolve().parent
STEP2_OUT = ROOT / "step-2-bigram-keyword-matching" / "outputs"
STEP3_OUT = ROOT / "step-3-llm-matching" / "outputs"
STEP4_OUT = ROOT / "step-4-dedupe-and-merge-matched-items" / "outputs"

# First sub-step in this order is run; all later sub-steps run too.
_PIPELINE_ORDER = ("1.1", "1.2", "2.1", "2.2", "2.3", "3", "4", "5", "6")

# Old --start-step values (pre folder reorg) → new ids
_LEGACY_START_ALIASES: dict[str, str] = {
    # Old 1.0 = keywords first → step 2.1 (no similar-title block). New step 1.1 is not aliased.
    "1.0": "2.1",
    "1.2": "2.2",
    "1.3": "2.3",
    "1.4": "3",
    "1.5": "4",
}


def _normalize_start_step(raw: str) -> str:
    return _LEGACY_START_ALIASES.get(raw, raw)


def _run_phase(start_step: str, phase_id: str) -> bool:
    """True if this phase should run (start is at or before this phase in the pipeline)."""
    return _PIPELINE_ORDER.index(start_step) <= _PIPELINE_ORDER.index(phase_id)


def _sort_phased_mapping_paths(paths: list[Path]) -> list[Path]:
    def depth_key(p: Path) -> tuple[int, str]:
        m = re.search(r"_depth(\d+)_", p.name)
        return (int(m.group(1)) if m else 10_000, p.name)

    return sorted(paths, key=depth_key)


def _taxonomy_max_depth_py() -> int:
    py = sys.executable or "python3"
    r = subprocess.run(
        [
            py,
            "-c",
            "import json,sys; from pathlib import Path; r=Path(sys.argv[1]); sys.path.insert(0,str(r)); "
            "from taxonomy_cascade import max_taxonomy_depth; "
            "print(max_taxonomy_depth(json.load(open(r/'source-files/categories_v1.json'))))",
            str(ROOT),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0 or not r.stdout.strip().isdigit():
        return 6
    return int(r.stdout.strip())


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def yn(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        ans = input(f"{prompt} [{hint}] ").strip().lower()
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


def choose(prompt: str, options: list[tuple[str, str]], disabled: set[str] | None = None) -> str:
    disabled = disabled or set()
    print(prompt)
    for k, label in options:
        if k in disabled:
            print(f"  {k}) \033[9m{label}\033[0m  [unavailable — see warnings above]")
        else:
            print(f"  {k}) {label}")
    while True:
        ans = input("> ").strip()
        if any(ans == k for k, _ in options if k not in disabled):
            return ans
        if any(ans == k for k, _ in options if k in disabled):
            print(f"  Step {ans} is unavailable due to missing .env variables. Pick another.")


def _check_env_vars(env_ans: str) -> tuple[list[str], set[str]]:
    """
    Returns (warnings, disabled_steps).
    warnings: human-readable lines describing what is missing.
    disabled_steps: step IDs that cannot run due to missing vars.
    """
    warnings: list[str] = []
    disabled: set[str] = set()

    if not os.environ.get("OPENAI_API_KEY"):
        warnings.append("  • OPENAI_API_KEY is not set  →  steps 3 and 5 require it")
        disabled.update({"3", "5"})

    if env_ans == "dev":
        if not (os.environ.get("DEV_DB_USER") and os.environ.get("DEV_DB_PASSWORD")):
            warnings.append("  • DEV_DB_USER / DEV_DB_PASSWORD not set  →  step 6 (dev DB upload) unavailable")
            disabled.add("6")
    else:
        if not (os.environ.get("PROD_DB_USER") and os.environ.get("PROD_DB_PASSWORD")):
            warnings.append("  • PROD_DB_USER / PROD_DB_PASSWORD not set  →  step 6 (prod DB upload) unavailable")
            disabled.add("6")

    return warnings, disabled


def newest_matching(glob_pat: str, folder: Path) -> Path | None:
    """Newest env-matching file in folder by mtime. Filenames carry the env suffix in the stem."""
    files = list(folder.glob(glob_pat))
    if not files:
        return None
    return shared_utils.latest_env_path(files, name_attr="stem")


def newest_unmatched_after_step1() -> Path | None:
    """Newest unmatched_after_step1*.json under step-1-similar-title-groups/outputs/ (resume-safe)."""
    return newest_under_step1("**/unmatched_after_step1*.json")


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    mins, sec = divmod(int(round(seconds)), 60)
    hrs, mins = divmod(mins, 60)
    if hrs >= 1:
        return f"{hrs}h {mins}m"
    if mins >= 1:
        return f"{mins}m"
    return f"{sec}s"


def _read_json_safe(p: Path) -> "dict | list | None":
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _status_row(label: str, n: "int | None", total: "int | None" = None) -> str:
    if n is None:
        return f"  {label:<32} —"
    pct = f"  ({n / total * 100:.0f}%)" if (total and total > 0) else ""
    return f"  {label:<32} {n:>8,}{pct}"


def _print_status_summary(items_file: "Path | None" = None) -> None:
    """Print a concise summary of how many items are categorized, by what method, and how many remain."""
    W = 64
    print("\n" + "═" * W)
    print("  PIPELINE STATUS")
    print("═" * W)

    # ── Step 1 ──────────────────────────────────────────────────────
    s1m = newest_under_step1("**/manual_matches*.json")
    groups_file = newest_under_step1("**/unmatched_similar_title_groups*.json")

    s1_groups_total = 0
    s1_assigned = 0
    s1_unknown_groups = 0
    s1_items_matched = 0
    s1_items_unmatched: "int | None" = None

    if groups_file:
        gd = _read_json_safe(groups_file)
        if isinstance(gd, dict):
            s1_groups_total = len(gd.get("groups") or [])
        elif isinstance(gd, list):
            s1_groups_total = len(gd)

    if s1m:
        md = _read_json_safe(s1m)
        if isinstance(md, dict):
            s1_assigned = len(md.get("group_assignments") or [])
            s1_unknown_groups = len(md.get("unknown_groups") or [])
            s1_items_matched = len(md.get("item_matches") or [])
            ua_candidates = sorted(s1m.parent.glob("unmatched_after_step1*.json"), key=lambda p: p.stat().st_mtime)
            ua = ua_candidates[-1] if ua_candidates else s1m.parent / "unmatched_after_step1.json"
            if ua.exists():
                ud = _read_json_safe(ua)
                if isinstance(ud, dict):
                    s1_items_unmatched = ud.get("item_count") or len(ud.get("items") or [])

    # Total: derive from step-1 if possible (avoids parsing the large source file)
    total: "int | None" = None
    if s1_items_matched or s1_items_unmatched:
        total = s1_items_matched + (s1_items_unmatched or 0)
    else:
        src = items_file or (ROOT / "source-files" / "prod-items-with-stores.json")
        if src.exists():
            try:
                d = json.loads(src.read_text(encoding="utf-8"))
                total = len(d) if isinstance(d, list) else None
            except Exception:
                pass

    # ── Step 2 ──────────────────────────────────────────────────────
    # 2.2 auto-match: newest 1.2_split_*/matched.json
    s22_split_dirs = sorted(
        [p for p in STEP2_OUT.glob("1.2_split_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    s22_matched_file = (s22_split_dirs[-1] / "matched.json") if s22_split_dirs else None
    s22_unmatched_file = (s22_split_dirs[-1] / "unmatched_and_keywords.json") if s22_split_dirs else None
    s22_matched = 0
    s22_unmatched: "int | None" = None
    if s22_matched_file and s22_matched_file.exists():
        d22 = _read_json_safe(s22_matched_file)
        s22_matched = len(d22) if isinstance(d22, list) else 0
    if s22_unmatched_file and s22_unmatched_file.exists():
        d22u = _read_json_safe(s22_unmatched_file)
        if isinstance(d22u, dict):
            items22u = d22u.get("unmatched_items") or d22u.get("items") or []
            s22_unmatched = len(items22u)

    # 2.3 manual bigram: newest manual_bigram_matches*.json
    s23_file = newest_matching("*/manual_bigram_matches*.json", STEP2_OUT)
    s23_matched = 0
    s23_assignments = 0
    s23_unknown = 0
    if s23_file:
        d23 = _read_json_safe(s23_file)
        if isinstance(d23, dict):
            s23_matched = len(d23.get("item_matches") or [])
            s23_assignments = len(d23.get("bigram_assignments") or [])
            s23_unknown = len(d23.get("unknown_bigrams") or [])

    s2_matched = s22_matched + s23_matched

    # ── Step 3 ──────────────────────────────────────────────────────
    s3m_file = newest_matching("*/llm_matched*.json", STEP3_OUT)
    s3u_file = newest_matching("*/llm_unmatched*.json", STEP3_OUT)
    s3_matched = 0
    s3_unmatched: "int | None" = None
    if s3m_file:
        d3 = _read_json_safe(s3m_file)
        if isinstance(d3, list):
            s3_matched = len(d3)
        elif isinstance(d3, dict):
            s3_matched = len(d3.get("matched_items") or d3.get("items") or d3.get("matched") or [])
    if s3u_file:
        d3u = _read_json_safe(s3u_file)
        if isinstance(d3u, list):
            s3_unmatched = len(d3u)
        elif isinstance(d3u, dict):
            s3_unmatched = len(d3u.get("unmatched_items") or d3u.get("items") or d3u.get("unmatched") or [])

    # ── Step 4 ──────────────────────────────────────────────────────
    s4m_file = newest_matching("**/matched_deduped.json", STEP4_OUT)
    s4u_file = newest_matching("**/unmatched_deduped.json", STEP4_OUT)
    s4_matched = 0
    s4_unmatched = 0
    if s4m_file:
        d4 = _read_json_safe(s4m_file)
        if isinstance(d4, list):
            s4_matched = len(d4)
        elif isinstance(d4, dict):
            s4_matched = len(d4.get("matched_items") or d4.get("items") or [])
    if s4u_file:
        d4u = _read_json_safe(s4u_file)
        if isinstance(d4u, list):
            s4_unmatched = len(d4u)
        elif isinstance(d4u, dict):
            s4_unmatched = len(d4u.get("unmatched_items") or d4u.get("items") or [])

    # ── Display ─────────────────────────────────────────────────────
    T = total
    print(_status_row("Total source items", T))
    print()

    if s1_groups_total or s1m:
        print("  Step 1 — similar-title groups")
        s1_remaining = max(0, s1_groups_total - s1_assigned - s1_unknown_groups)
        print(_status_row("    Groups total", s1_groups_total if s1_groups_total else None))
        if s1m:
            print(_status_row("    Groups assigned", s1_assigned))
            if s1_unknown_groups:
                print(_status_row("    Groups marked unknown", s1_unknown_groups))
            if s1_remaining:
                print(_status_row("    Groups not yet decided", s1_remaining))
            print(_status_row("    Items categorized", s1_items_matched, T))
            if s1_items_unmatched is not None:
                print(_status_row("    Items passed to step 2", s1_items_unmatched, T))
        else:
            print("    (groups built — no manual session started yet)")
    else:
        print("  Step 1 — no output yet")

    print()
    if s22_matched_file and s22_matched_file.exists():
        print("  Step 2 — bigram / keyword match")
        print(_status_row("    2.2 auto-matched (bigrams)", s22_matched, T))
        if s23_file:
            print(_status_row("    2.3 manual-matched (bigrams)", s23_matched, T))
            if s23_assignments:
                print(_status_row("      Bigrams assigned", s23_assignments))
            if s23_unknown:
                print(_status_row("      Bigrams marked unknown", s23_unknown))
        else:
            print("    2.3 manual — no session yet")
        print(_status_row("    Step 2 total", s2_matched, T))
        if s22_unmatched is not None:
            remaining_after_2 = s22_unmatched - s23_matched
            print(_status_row("    Items passed to step 3", max(0, remaining_after_2), T))
    else:
        print("  Step 2 — no output yet")

    print()
    if s3m_file:
        print("  Step 3 — LLM match")
        print(_status_row("    Items categorized", s3_matched, T))
        if s3_unmatched is not None:
            print(_status_row("    Items still unmatched", s3_unmatched, T))
    else:
        print("  Step 3 — no output yet")

    print()
    if s4m_file:
        print("  Step 4 — final deduped")
        print(_status_row("    Matched (unique)", s4_matched, T))
        print(_status_row("    Unmatched (unique)", s4_unmatched, T))
        if T and T > 0:
            print(f"\n  {'Overall categorized (step 4)':<32} {s4_matched / T * 100:.1f}%")
    else:
        print("  Step 4 — no output yet")
        best = s1_items_matched + s2_matched + s3_matched
        if best and T:
            remaining = T - best
            print()
            print(f"  Best estimate (pre-dedup)")
            print(_status_row("    Categorized across steps 1–3", best, T))
            print(_status_row("    Remaining", remaining, T))

    # ── Step 5 (attributes) ──────────────────────────────────────────
    s5_outdir = ROOT / "step-5-attribute-generation-and-unit-value-assignment" / "outputs"
    s5_file: "Path | None" = None
    if s5_outdir.exists():
        candidates = sorted(s5_outdir.glob("proposed_attributes_*.json"), key=lambda p: p.stat().st_mtime)
        if candidates:
            s5_file = candidates[-1]
    print()
    if s5_file:
        try:
            s5_data = json.loads(s5_file.read_text(encoding="utf-8"))
            n_cats = len(s5_data.get("_category_attributes") or {})
            n_units = len(s5_data.get("units") or {})
            meta = s5_data.get("_meta") or {}
            ts = meta.get("generated_at") or s5_file.stem.replace("proposed_attributes_", "")
            print(f"  Step 5 — attributes generated  ({ts})")
            print(_status_row("    Leaf categories covered", n_cats))
            print(_status_row("    Unique units defined", n_units))
        except Exception:
            print(f"  Step 5 — output present but could not parse: {s5_file.name}")
    else:
        print("  Step 5 — no attributes generated yet")

    # ── Step 6 (DB upload) ──────────────────────────────────────────
    # Step 6 is always manual; just note whether a final-output folder exists.
    final_out = ROOT / "final-output"
    print()
    if final_out.exists():
        runs = sorted([p for p in final_out.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
        if runs:
            latest_run = runs[-1]
            files = [f.name for f in latest_run.iterdir() if f.is_file()]
            print(f"  Step 5 final-output — latest: final-output/{latest_run.name}/")
            print(f"    Files: {', '.join(files)}")
        else:
            print("  Step 6 — no final-output runs yet")
    else:
        print("  Step 6 — no final-output folder yet")

    print("\n" + "═" * W + "\n")


def estimate_step_3_llm_time(
    latest_split_unmatched: Path | None,
    default_batch_size: int,
    model: str,
) -> str | None:
    if not latest_split_unmatched or not latest_split_unmatched.exists():
        return None
    try:
        data = json.loads(latest_split_unmatched.read_text(encoding="utf-8"))
    except Exception:
        return None

    unmatched = data.get("unmatched_items")
    if unmatched is None:
        unmatched = data.get("unmatched_words", [])
    if not isinstance(unmatched, list):
        return None
    n_items = len(unmatched)
    batches = math.ceil(n_items / default_batch_size) if default_batch_size > 0 else 0
    if batches <= 0:
        return None

    model_l = (model or "").lower()
    if "mini" in model_l:
        per_call_fast_s, per_call_typ_s, per_call_slow_s = 20, 30, 60
    else:
        per_call_fast_s, per_call_typ_s, per_call_slow_s = 25, 40, 90

    fast = batches * per_call_fast_s
    typ = batches * per_call_typ_s
    slow = batches * per_call_slow_s
    return (
        f"~{batches} LLM batches (@batch_size={default_batch_size}, {n_items} unmatched items). "
        f"ETA: fast {_format_duration(fast)}, typical {_format_duration(typ)}, slow {_format_duration(slow)}. "
        f"(Sleep adds extra: ~({batches-1}*sleep_seconds))"
    )


def main() -> None:
    all_start_choices = list(_PIPELINE_ORDER) + list(_LEGACY_START_ALIASES.keys())
    parser = argparse.ArgumentParser(description="Categorization v2 pipeline runner")
    parser.add_argument(
        "--start-step",
        choices=sorted(set(all_start_choices)),
        help="First sub-step to run: 1.1 … 6 (legacy 1.0–1.5 aliases accepted).",
    )
    cli_args = parser.parse_args()

    py = sys.executable or "python3"

    start_step = _normalize_start_step(cli_args.start_step) if cli_args.start_step else None
    if cli_args.start_step and start_step != cli_args.start_step:
        print(f"(Mapped --start-step {cli_args.start_step!r} → {start_step!r})")

    if start_step and start_step not in _PIPELINE_ORDER:
        raise SystemExit(f"Invalid start step after alias resolution: {start_step!r}")

    print("Categorization v2 pipeline runner\n")

    # ── Dev / Prod environment selection ────────────────────────────────────────
    _ENV_FILES = {
        "prod": ROOT / "source-files" / "prod-items-with-stores.json",
        "dev":  ROOT / "source-files" / "dev-items-with-stores.json",
    }
    while True:
        env_ans = input("Run for [prod] or [dev]? ").strip().lower()
        if env_ans in _ENV_FILES:
            break
        if env_ans in ("p", "production"):
            env_ans = "prod"
            break
        if env_ans in ("d", "development"):
            env_ans = "dev"
            break
        print(f"  Please type 'prod' or 'dev'.")
    items_file = _ENV_FILES[env_ans]
    os.environ["PIPELINE_ENV"] = env_ans   # propagated to all subprocess steps

    # ── Dev speed-run mode ───────────────────────────────────────────────────────
    speed_run = False
    if env_ans == "dev":
        while True:
            mode_ans = input("Speed run (auto-categorize all, no prompts) or normal? [speed/normal] ").strip().lower()
            if mode_ans in ("speed", "s"):
                speed_run = True
                break
            if mode_ans in ("normal", "n"):
                break
            print("  Please type 'speed' or 'normal'.")
        if speed_run:
            print("  [speed run] Will auto-assign categories and skip all confirmation prompts.")

    if not items_file.exists():
        print(f"  WARNING: items file not found: {items_file.relative_to(ROOT)}")
        print("  Step 1.1 will fail unless the file is present.")
    else:
        print(f"  Using items file: {items_file.relative_to(ROOT)}  ({items_file.stat().st_size // 1024:,} KB)")

    # ── .env variable check ──────────────────────────────────────────────────────
    env_warnings, disabled_steps = _check_env_vars(env_ans)
    if env_warnings:
        print()
        print("  ⚠️  Missing .env variables detected:")
        for w in env_warnings:
            print(w)
        print()
        raw_proceed = input("  Proceed anyway? [Y/n] ").strip().lower()
        if raw_proceed in ("n", "no"):
            print("  Exiting. Set the missing variables in your .env file and restart.")
            return
    else:
        disabled_steps = set()

    _print_status_summary(items_file=items_file)
    print("Outputs: step-1-similar-title-groups/  step-2/  step-3-llm-matching/  step-4/outputs/<run_id>/  step-5/outputs/  final-output/<ts>/")

    cascade_mapping_paths: list[Path] | None = None

    if not start_step:
        if speed_run:
            start_step = "1.1"
            print("\n[speed run] Starting from step 1.1 and running all steps automatically.")
        else:
            start_step = choose(
                "\nWhere do you want to start? (Everything after that runs in order.)",
                [
                    ("1.1", "Step 1.1 — Build similar-title groups (raw catalog)"),
                    ("1.2", "Step 1.2 — Interactive assign groups → taxonomy leaves"),
                    ("2.1", "Step 2.1 — Keyword frequencies + bigram→category mappings"),
                    ("2.2", "Step 2.2 — Match items to bigrams (cascade)"),
                    ("2.3", "Step 2.3 — Interactive manual bigram → leaf (leftovers)"),
                    ("3", "Step 3 — LLM match remaining unmatched items"),
                    ("4", "Step 4 — Dedupe + summaries"),
                    ("5", "Step 5 — Attribute generation (5a template grouping → 5b LLM schema+patterns → 5c value extraction)"),
                    ("6", "Step 6 — DB upload (requires SSM tunnel)"),
                ],
                disabled=disabled_steps,
            )

    print(f"\nStarting at {start_step!r} (runs this phase and all following phases).")

    # Fresh run defaults when beginning the full pipeline from step 1.1
    sequential_fresh = start_step == "1.1"

    run_11 = _run_phase(start_step, "1.1")
    run_12 = _run_phase(start_step, "1.2")
    run_21 = _run_phase(start_step, "2.1")
    run_22 = _run_phase(start_step, "2.2")
    run_23 = _run_phase(start_step, "2.3")
    run_3 = _run_phase(start_step, "3")
    run_4 = _run_phase(start_step, "4")
    run_5 = _run_phase(start_step, "5")
    run_6 = _run_phase(start_step, "6")

    # --- Step 1.1: build similar-title groups ---
    if not run_11:
        print("Skipping step 1.1 (build similar-title groups).")
    elif speed_run or yn(
        "Run step 1.1 — build similar-title groups from the raw production catalog?",
        default=True,
    ):
        if speed_run:
            print("\n[speed run] Running step 1.1 — build similar-title groups.")
        cmd_11 = [py, str(ROOT / "step-1-similar-title-groups" / "1_1_build_similar_title_groups.py"), "--input", str(items_file)]
        run(cmd_11)
    else:
        print("Skipping step 1.1.")

    # --- Step 1.2: interactive group → leaf ---
    if not run_12:
        print("Skipping step 1.2 (interactive similar-title → leaf).")
    elif speed_run or yn(
        "Run step 1.2 — interactive matching: similar-title groups → taxonomy leaves?",
        default=True,
    ):
        cmd_12 = [py, str(ROOT / "step-1-similar-title-groups" / "1_2_interactive_similar_title_match.py")]
        if speed_run:
            print("\n[speed run] Running step 1.2 — auto-randomly assigning groups.")
            cmd_12 += ["--auto-random", "--fresh-run"]
        elif sequential_fresh:
            cmd_12.append("--fresh-run")
        run(cmd_12)
    else:
        print("Skipping step 1.2.")

    ua = newest_unmatched_after_step1()
    if ua:
        print(f"\nUsing unmatched pool for step 2: {ua.relative_to(ROOT)}")
    else:
        print(
            "\n(No unmatched_after_step1 file — step 2.1 / 2.2 will use the full raw catalog "
            "unless you complete step 1.2 first.)"
        )

    # --- Step 2.1a: keyword frequencies ---
    if not run_21:
        print("Skipping step 2.1a (keyword frequencies).")
    elif speed_run or yn(
        "Run step 2.1a — extract keyword frequencies (optionally limited to step-1 unmatched pool)?",
        default=True,
    ):
        if speed_run:
            print("\n[speed run] Running step 2.1a — keyword frequencies.")
        cmd_kw = [py, str(ROOT / "step-2-bigram-keyword-matching" / "2_1_generate_keywords.py")]
        if ua:
            cmd_kw += ["--items-json", str(ua)]
        if sequential_fresh or speed_run:
            cmd_kw.append("--no-resume")
        elif yn("Start fresh for 2.1a (no checkpoint resume)?", default=False):
            cmd_kw.append("--no-resume")
        run(cmd_kw)
    else:
        print("Skipping step 2.1a.")

    step10_out = STEP2_OUT
    latest_10 = newest_matching("1.0-title_subtitle_keyword_frequencies*.json", step10_out)
    if latest_10:
        print(f"\nMost recent keyword frequency file: {latest_10.relative_to(ROOT)}")
    elif run_21:
        print("\nNo 1.0 keyword file found yet — run step 2.1a or pass --keywords when running bigram scripts.")

    # --- Step 2.1b: bigram mappings ---
    mapping_path: Path | None = None
    if not run_21:
        print(f"Skipping step 2.1b (bigram mappings) — starting at {start_step}.")
        mapping_path = newest_matching("1.1*-bigram_categories_mapping*.json", STEP2_OUT)
        if not mapping_path:
            print("No cached bigram mapping; step 2.2 may prompt for a file.")
    elif speed_run:
        # Speed run: use taxonomy-based mapping with defaults
        print("\n[speed run] Running step 2.1b — taxonomy-based bigram mapping (defaults).")
        cmd = [py, str(ROOT / "step-2-bigram-keyword-matching" / "2_1_generate_bigrams_taxonomy.py")]
        if latest_10:
            cmd += ["--keywords", str(latest_10)]
        cmd += ["--min-confidence", "0.85", "--no-resume"]
        run(cmd)
        cascade_mapping_paths = None
        mapping_path = newest_matching("1.1a-bigram_categories_mapping*.json", STEP2_OUT)
    else:
        step11_choice = choose(
            "\nStep 2.1b — choose bigram → category mapping method:",
            [
                ("a", "Taxonomy-based (1.1a, fast, deterministic)"),
                ("b", "OpenAI LLM-based (1.1b, slower, semantic)"),
                ("s", "Skip — use latest mapping file for step 2.2"),
            ],
        )

        if step11_choice == "a":
            cmd = [py, str(ROOT / "step-2-bigram-keyword-matching" / "2_1_generate_bigrams_taxonomy.py")]
            if latest_10 and yn("Use most recent 2.1a keyword frequency file?", default=True):
                cmd += ["--keywords", str(latest_10)]
            else:
                p = input("Path to keyword frequencies JSON: ").strip()
                if p:
                    cmd += ["--keywords", p]

            phased = yn("Generate phased depth maps (T0..TD) for cascade matching?", default=False)
            tdmax = _taxonomy_max_depth_py() if phased else 0
            if phased:
                customize_prompt = (
                    f"Customize min confidence and depth range? "
                    f"(defaults: min confidence 0.85, depth-min 0, depth-max {tdmax})"
                )
            else:
                customize_prompt = "Customize min confidence? (default 0.85)"

            min_conf = "0.85"
            dmin_s = "0"
            dmax_s = str(tdmax) if phased else "1"
            if yn(customize_prompt, default=False):
                min_conf = input("Min confidence to keep: ").strip() or "0.85"
                if phased:
                    dmin_s = input(f"depth-min (default {dmin_s}): ").strip() or dmin_s
                    dmax_s = input(f"depth-max inclusive (default {dmax_s}): ").strip() or dmax_s

            cmd += ["--min-confidence", min_conf]
            batch_tag = None
            if phased:
                batch_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                cmd += ["--depth-min", dmin_s, "--depth-max", dmax_s, "--output-batch-tag", batch_tag]
            if sequential_fresh:
                cmd.append("--no-resume")
            elif yn("Start fresh for taxonomy bigrams (no checkpoint resume)?", default=False):
                cmd.append("--no-resume")
            run(cmd)
            step11_out = STEP2_OUT
            if phased and batch_tag:
                phased_paths = list(step11_out.glob(f"1.1a-bigram_categories_mapping_depth*_{batch_tag}.json"))
                cascade_mapping_paths = _sort_phased_mapping_paths(phased_paths)
                mapping_path = cascade_mapping_paths[-1] if cascade_mapping_paths else None
            else:
                cascade_mapping_paths = None
                mapping_path = newest_matching("1.1a-bigram_categories_mapping*.json", step11_out)

        elif step11_choice == "b":
            model = input(f"OpenAI model (default {os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}): ").strip()
            if not model:
                model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            batch_size = input("Batch size (default 100): ").strip() or "100"
            min_conf = input("Min confidence to keep (default 0.85): ").strip() or "0.85"
            sleep_s = input("Sleep seconds between calls (default 0): ").strip() or "0"

            cmd = [py, str(ROOT / "step-2-bigram-keyword-matching" / "2_1_generate_bigrams_openai.py")]
            if latest_10 and yn("Use most recent 2.1a keyword frequency file?", default=True):
                cmd += ["--keywords", str(latest_10)]
            else:
                p = input("Path to keyword frequencies JSON: ").strip()
                if p:
                    cmd += ["--keywords", p]
            cmd += ["--model", model, "--batch-size", batch_size, "--min-confidence", min_conf, "--sleep-seconds", sleep_s]
            if sequential_fresh:
                cmd.append("--no-resume")
            elif yn("Start fresh for OpenAI bigrams (no checkpoint resume)?", default=False):
                cmd.append("--no-resume")
            run(cmd)
            cascade_mapping_paths = None
            mapping_path = newest_matching("1.1b-bigram_categories_mapping*.json", STEP2_OUT)

        else:
            cascade_mapping_paths = None
            step11_out = STEP2_OUT
            mapping_path = newest_matching("1.1*-bigram_categories_mapping*.json", step11_out)
            if not mapping_path:
                mapping_path = newest_matching("1.1*-bigram_categories_mapping*.json", STEP2_OUT)
            print("Using latest bigram mapping for step 2.2.")

    if mapping_path:
        print(f"\nMapping file for step 2.2: {mapping_path.relative_to(ROOT)}")
    else:
        print("\nNo single mapping file selected; step 2.2 may prompt or use cascade paths.")

    # --- Step 2.2: match items to bigrams ---
    if not run_22:
        print(f"Skipping step 2.2 (match items to bigrams) — starting at {start_step}.")
    elif speed_run or yn("\nRun step 2.2 — match items to bigrams (phased cascade if configured)?", default=True):
        if speed_run:
            print("\n[speed run] Running step 2.2 — match items to bigrams.")
        latest_map = newest_matching("1.1*-bigram_categories_mapping*.json", STEP2_OUT)
        if latest_map:
            print(f"Most recent mapping on disk: {latest_map.relative_to(ROOT)}")
        cmd = [py, str(ROOT / "step-2-bigram-keyword-matching" / "2_2_match_items_to_bigrams.py")]
        if ua:
            cmd += ["--items-json", str(ua)]
        if cascade_mapping_paths:
            for p in cascade_mapping_paths:
                cmd += ["--cascade-mapping", str(p)]
            print(f"Using phased cascade: {len(cascade_mapping_paths)} mapping files (low depth first).")
        elif latest_map:
            cmd += ["--mapping", str(latest_map)]
        if not speed_run and yn("Use strict side matching (title-only / subtitle-only bigrams)?", default=False):
            cmd += ["--strict-sides"]
        if sequential_fresh or speed_run:
            cmd.append("--no-resume")
        elif yn("Start fresh for step 2.2 (no checkpoint resume)?", default=False):
            cmd.append("--no-resume")
        run(cmd)
    else:
        print("Skipping step 2.2.")

    latest_22 = newest_matching("1.2-bigram_sorted_items*.json", STEP2_OUT)
    if latest_22:
        print(f"\nLatest step 2.2 output: {latest_22.relative_to(ROOT)}")

    latest_kw = newest_matching("**/unmatched_and_keywords.json", STEP2_OUT)
    if latest_kw:
        print(f"Latest split unmatched_and_keywords: {latest_kw.relative_to(ROOT)}")

    # --- Step 2.3: interactive keyword → leaf ---
    if not run_23:
        print(f"Skipping step 2.3 (interactive manual bigram → leaf) — starting at {start_step}.")
    elif speed_run or yn(
        "\nRun step 2.3 — interactive manual bigram → leaf (unmatched from step 2.2 split)?",
        default=(start_step == "2.3"),
    ):
        cmd_23 = [py, str(ROOT / "step-2-bigram-keyword-matching" / "2_3_interactive_keyword_match.py")]
        if speed_run:
            print("\n[speed run] Running step 2.3 — auto-randomly assigning bigrams.")
            cmd_23 += ["--auto-random", "--fresh-run"]
        elif sequential_fresh:
            cmd_23.append("--fresh-run")
        run(cmd_23)
        latest_23 = newest_matching("*/manual_bigram_matches*.json", STEP2_OUT)
        if latest_23:
            print(f"\nLatest step 2.3 manual: {latest_23.relative_to(ROOT)}")
    else:
        print("Skipping step 2.3.")

    # --- Step 3: LLM ---
    if not run_3:
        print(f"Skipping step 3 (LLM) — starting at {start_step}.")
    else:
        default_batch_size = 25
        default_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        default_min_conf = "0.9"

        if speed_run:
            print("\n[speed run] Running step 3 — LLM match with default settings.")
            s1m = newest_under_step1("**/manual_matches*.json")
            s23m = newest_matching("*/manual_bigram_matches*.json", STEP2_OUT)
            cmd = [
                py,
                str(ROOT / "step-3-llm-matching" / "3_llm_match_unmatched.py"),
                "--model", default_model,
                "--batch-size", str(default_batch_size),
                "--min-confidence", default_min_conf,
                "--no-resume",
            ]
            if latest_kw:
                cmd += ["--input", str(latest_kw)]
            if s1m:
                cmd += ["--step1-manual", str(s1m)]
            run(cmd)
        else:
            eta = estimate_step_3_llm_time(latest_kw, default_batch_size=default_batch_size, model=default_model)
            eta_str = f" {eta}" if eta else ""
            prompt_3 = f"\nRun step 3 — LLM match for remaining unmatched items?{eta_str}"
            if not yn(prompt_3, default=(start_step == "3")):
                print("Skipping step 3.")
            else:
                # Auto-discover latest manual results from step 1 and step 2.3
                s1m = newest_under_step1("**/manual_matches*.json")
                s23m = newest_matching("*/manual_bigram_matches*.json", STEP2_OUT)

                merge_sources: list[str] = []
                if s1m:
                    try:
                        s1_count = len(json.loads(s1m.read_text(encoding="utf-8")).get("item_matches") or [])
                    except Exception:
                        s1_count = "?"
                    merge_sources.append(f"  step-1 manual  {s1m.name}  ({s1_count} items)")
                if s23m:
                    try:
                        s23_count = len(json.loads(s23m.read_text(encoding="utf-8")).get("item_matches") or [])
                    except Exception:
                        s23_count = "?"
                    merge_sources.append(f"  step-2.3 manual  {s23m.name}  ({s23_count} items)")

                print("\nHow would you like to start step 3?")
                print("  [1] Quick start  — use defaults and merge all latest step 1 & 2 results (recommended)")
                print("  [2] Custom start — choose model, batch size, confidence, and merge options")
                mode_choice = input("Choice [1/2] (default 1): ").strip() or "1"

                if mode_choice == "2":
                    model = input(f"  OpenAI model (default {default_model}): ").strip() or default_model
                    batch_size = input("  Batch size (default 25): ").strip() or "25"
                    min_conf = input("  Min confidence (default 0.9): ").strip() or default_min_conf
                    do_merge = bool(merge_sources) and yn(
                        "  Merge latest step 1 & 2 results before sending to LLM?", default=True
                    )
                else:
                    model = default_model
                    batch_size = str(default_batch_size)
                    min_conf = default_min_conf
                    do_merge = bool(merge_sources)
                    if merge_sources:
                        print("\nMerging latest results into LLM pool exclusions:")
                        for line in merge_sources:
                            print(line)

                cmd = [
                    py,
                    str(ROOT / "step-3-llm-matching" / "3_llm_match_unmatched.py"),
                    "--model", model,
                    "--batch-size", batch_size,
                    "--min-confidence", min_conf,
                ]
                if latest_kw:
                    cmd += ["--input", str(latest_kw)]
                if do_merge and s1m:
                    cmd += ["--step1-manual", str(s1m)]

                if sequential_fresh:
                    cmd.append("--no-resume")
                elif yn("\nResume a previous step 3 checkpoint if one exists? (No = start fresh)", default=True):
                    pass  # resume is the default; no flag needed
                else:
                    cmd.append("--no-resume")

                run(cmd)

        latest_3m = newest_matching("*/llm_matched*.json", STEP3_OUT)
        latest_3u = newest_matching("*/llm_unmatched*.json", STEP3_OUT)
        if latest_3m:
            print(f"\nLatest step 3 matched: {latest_3m.relative_to(ROOT)}")
        if latest_3u:
            print(f"Latest step 3 unmatched: {latest_3u.relative_to(ROOT)}")

    # --- Step 4: dedupe ---
    if not run_4:
        print(f"Skipping step 4 (dedupe) — starting at {start_step}.")
    elif speed_run or yn(
        "\nRun step 4 — dedupe and finalize combined results from all steps (step 1 manual, step 2 bigrams, step 3 LLM)?",
        default=(start_step in ("1.1", "4")),
    ):
        if speed_run:
            print("\n[speed run] Running step 4 — dedupe and finalize.")
        cmd_4 = [py, str(ROOT / "step-4-dedupe-and-merge-matched-items" / "4_dedupe_and_summaries.py"), "--pair-latest"]
        run(cmd_4)
        step4_out = STEP4_OUT
        latest_4_m = newest_matching("**/matched_deduped.json", step4_out)
        latest_4_u = newest_matching("**/unmatched_deduped.json", step4_out)
        latest_4_ms = newest_matching("**/matched_summary.json", step4_out)
        latest_4_us = newest_matching("**/unmatched_summary.json", step4_out)
        if latest_4_m:
            print(f"\nLatest step 4 matched (deduped): {latest_4_m.relative_to(ROOT)}")
        if latest_4_u:
            print(f"Latest step 4 unmatched (deduped): {latest_4_u.relative_to(ROOT)}")
        if latest_4_ms:
            print(f"Latest step 4 matched summary: {latest_4_ms.relative_to(ROOT)}")
        if latest_4_us:
            print(f"Latest step 4 unmatched summary: {latest_4_us.relative_to(ROOT)}")
    else:
        print("Skipping step 4.")

    # --- Step 5: Attribute generation (3 sub-steps: 5a → 5b → 5c) ---
    if not run_5:
        print(f"Skipping step 5 (attribute generation) — starting at {start_step}.")
    elif speed_run or yn("\nRun step 5 — attribute generation (template grouping → LLM schema + patterns → value extraction)?", default=True):
        has_api_key = bool(os.environ.get("OPENAI_API_KEY"))
        if not has_api_key:
            print("  OPENAI_API_KEY not set — cannot run step 5b/5c LLM steps.")
            print("  Set it in .env and re-run from step 5.")
        else:
            if speed_run:
                print("\n[speed run] Running step 5a — title template grouping.")

            # ── 5a: title template grouping (pure Python, no LLM)
            script_5a = ROOT / "step-5-attribute-generation-and-unit-value-assignment" / "5a_group_title_templates.py"
            result_5a = subprocess.run([sys.executable, str(script_5a)], cwd=ROOT)
            if result_5a.returncode != 0:
                print("  WARNING: Step 5a (template grouping) failed — skipping 5b and 5c.")
            else:
                if speed_run:
                    print("\n[speed run] Running step 5b — LLM attribute schema + patterns.")

                # ── 5b: LLM attribute schema + regex patterns
                script_5b = ROOT / "step-5-attribute-generation-and-unit-value-assignment" / "5_generate_attributes.py"
                result_5b = subprocess.run([sys.executable, str(script_5b)], cwd=ROOT)
                if result_5b.returncode != 0:
                    print("  WARNING: Step 5b (LLM attribute generation) failed — skipping 5c.")
                else:
                    if speed_run:
                        print("\n[speed run] Running step 5c — attribute value extraction.")

                    # ── 5c: regex + LLM fallback value extraction
                    script_5c = ROOT / "step-5-attribute-generation-and-unit-value-assignment" / "5c_extract_attribute_values.py"
                    result_5c = subprocess.run([sys.executable, str(script_5c)], cwd=ROOT)
                    if result_5c.returncode != 0:
                        print("  WARNING: Step 5c (value extraction) failed — continuing.")
    else:
        print("Skipping step 5.")

    # --- Step 6: DB upload ---
    if not run_6:
        print(f"Skipping step 6 (DB upload) — starting at {start_step}.")
    else:
        step6_script = ROOT / "step-6-db-upload" / "6_upload_to_db.py"
        if speed_run:
            # In speed run mode, show summary then ask once about staging upload
            print("\n" + "=" * 72)
            print("[speed run] Pipeline complete. All steps ran automatically.")
            _print_status_summary(items_file=items_file)
            print("=" * 72)
            print(f"\nStep 6 — DB upload requires an active SSM tunnel.")
            print(f"  Script: python {step6_script.relative_to(ROOT)}")
            if yn("Upload results to staging database now?", default=False):
                subprocess.run([sys.executable, str(step6_script)], cwd=ROOT)
            else:
                print("Skipping DB upload.")
        else:
            print(f"\nStep 6 — DB upload requires an active SSM tunnel.")
            print(f"  Run separately: python {step6_script.relative_to(ROOT)}")
            if yn("Launch step 6 now?", default=False):
                subprocess.run([sys.executable, str(step6_script)], cwd=ROOT)


if __name__ == "__main__":
    main()
