#!/usr/bin/env python3
"""
Master controller for Categorization v2 pipeline.

Flow (high level):
  Step 1 — similar-title grouping (raw catalog)
    1.1  step-1/1_1_build_similar_title_groups.py
    1.2  step-1/1_2_interactive_similar_title_match.py
         → unmatched_after_step1.json for step 2
  Step 2 — keywords, bigrams, cascade match, optional manual bigram→leaf
    2.1a step-2/2_1_generate_keywords.py [--items-json …]
    2.1b step-2/2_1_generate_bigrams_taxonomy.py or 2_1_generate_bigrams_openai.py
    2.2  step-2/2_2_match_items_to_bigrams.py [--items-json …]
    2.3  step-2/2_3_interactive_keyword_match.py (optional)
  Step 3 — LLM: step-3/3_llm_match_unmatched.py [--step1-manual …]
  Step 4 — Dedupe: step-4/4_dedupe_and_summaries.py
  Step 5 — Attributes: step-5/5_generate_attributes.py
  Step 6 — DB upload: step-6/6_upload_to_db.py (run separately; requires SSM tunnel)

CLI --start-step uses 1.1, 1.2, 2.1, 2.2, 2.3, 3, 4 (first step to run). Legacy aliases:
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

from pipeline_paths import newest_under_step1

ROOT = Path(__file__).resolve().parent
STEP2_OUT = ROOT / "step-2" / "outputs"
STEP3_OUT = ROOT / "step-3" / "outputs"
STEP4_OUT = ROOT / "step-4" / "outputs"

# First sub-step in this order is run; all later sub-steps run too.
_PIPELINE_ORDER = ("1.1", "1.2", "2.1", "2.2", "2.3", "3", "4")

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


def choose(prompt: str, options: list[tuple[str, str]]) -> str:
    print(prompt)
    for k, label in options:
        print(f"  {k}) {label}")
    while True:
        ans = input("> ").strip()
        if any(ans == k for k, _ in options):
            return ans


def newest_matching(glob_pat: str, folder: Path) -> Path | None:
    files = list(folder.glob(glob_pat))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def newest_unmatched_after_step1() -> Path | None:
    """Newest unmatched_after_step1.json under step-1/outputs/ (resume-safe)."""
    return newest_under_step1("**/unmatched_after_step1.json")


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    mins, sec = divmod(int(round(seconds)), 60)
    hrs, mins = divmod(mins, 60)
    if hrs >= 1:
        return f"{hrs}h {mins}m"
    if mins >= 1:
        return f"{mins}m"
    return f"{sec}s"


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
        help="First sub-step to run: 1.1 … 4 (legacy 1.0–1.5 aliases accepted).",
    )
    cli_args = parser.parse_args()

    py = sys.executable or "python3"

    start_step = _normalize_start_step(cli_args.start_step) if cli_args.start_step else None
    if cli_args.start_step and start_step != cli_args.start_step:
        print(f"(Mapped --start-step {cli_args.start_step!r} → {start_step!r})")

    if start_step and start_step not in _PIPELINE_ORDER:
        raise SystemExit(f"Invalid start step after alias resolution: {start_step!r}")

    print("Categorization v2 pipeline runner\n")
    print("Inputs:")
    print("  - source-files/categories_v1.json")
    print("  - source-files/raw-prod-items-non-deleted.json")
    print("Outputs (each step has its own outputs/ folder; timestamped runs under step-1, step-4):")
    print("  - step-1/outputs/  step-2/outputs/  step-3/outputs/  step-4/outputs/<run_id>/")

    cascade_mapping_paths: list[Path] | None = None

    if not start_step:
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
            ],
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

    # --- Step 1.1: build similar-title groups ---
    if not run_11:
        print("Skipping step 1.1 (build similar-title groups).")
    elif yn(
        "Run step 1.1 — build similar-title groups from the raw production catalog?",
        default=True,
    ):
        run([py, str(ROOT / "step-1" / "1_1_build_similar_title_groups.py")])
    else:
        print("Skipping step 1.1.")

    # --- Step 1.2: interactive group → leaf ---
    if not run_12:
        print("Skipping step 1.2 (interactive similar-title → leaf).")
    elif yn(
        "Run step 1.2 — interactive matching: similar-title groups → taxonomy leaves?",
        default=True,
    ):
        cmd_12 = [py, str(ROOT / "step-1" / "1_2_interactive_similar_title_match.py")]
        if sequential_fresh:
            cmd_12.append("--fresh-run")
        run(cmd_12)
    else:
        print("Skipping step 1.2.")

    ua = newest_unmatched_after_step1()
    if ua:
        print(f"\nUsing unmatched pool for step 2: {ua.relative_to(ROOT)}")
    else:
        print(
            "\n(No unmatched_after_step1.json — step 2.1 / 2.2 will use the full raw catalog "
            "unless you complete step 1.2 first.)"
        )

    # --- Step 2.1a: keyword frequencies ---
    if not run_21:
        print("Skipping step 2.1a (keyword frequencies).")
    elif yn(
        "Run step 2.1a — extract keyword frequencies (optionally limited to step-1 unmatched pool)?",
        default=True,
    ):
        cmd_kw = [py, str(ROOT / "step-2" / "2_1_generate_keywords.py")]
        if ua:
            cmd_kw += ["--items-json", str(ua)]
        if sequential_fresh:
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
            cmd = [py, str(ROOT / "step-2" / "2_1_generate_bigrams_taxonomy.py")]
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

            cmd = [py, str(ROOT / "step-2" / "2_1_generate_bigrams_openai.py")]
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
    elif yn("\nRun step 2.2 — match items to bigrams (phased cascade if configured)?", default=True):
        latest_map = newest_matching("1.1*-bigram_categories_mapping*.json", STEP2_OUT)
        if latest_map:
            print(f"Most recent mapping on disk: {latest_map.relative_to(ROOT)}")
        cmd = [py, str(ROOT / "step-2" / "2_2_match_items_to_bigrams.py")]
        if ua:
            cmd += ["--items-json", str(ua)]
        if cascade_mapping_paths:
            for p in cascade_mapping_paths:
                cmd += ["--cascade-mapping", str(p)]
            print(f"Using phased cascade: {len(cascade_mapping_paths)} mapping files (low depth first).")
        elif latest_map:
            cmd += ["--mapping", str(latest_map)]
        strict_sides = yn("Use strict side matching (title-only / subtitle-only bigrams)?", default=False)
        if strict_sides:
            cmd += ["--strict-sides"]
        if sequential_fresh:
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
    elif yn(
        "\nRun step 2.3 — interactive manual bigram → leaf (unmatched from step 2.2 split)?",
        default=(start_step == "2.3"),
    ):
        cmd_23 = [py, str(ROOT / "step-2" / "2_3_interactive_keyword_match.py")]
        if sequential_fresh:
            cmd_23.append("--fresh-run")
        run(cmd_23)
        latest_23 = newest_matching("1.3-manual*.json", STEP2_OUT)
        if latest_23:
            print(f"\nLatest step 2.3 manual: {latest_23.relative_to(ROOT)}")
    else:
        print("Skipping step 2.3.")

    # --- Step 3: LLM ---
    if not run_3:
        print(f"Skipping step 3 (LLM) — starting at {start_step}.")
    else:
        step3_default = start_step == "3"
        default_batch_size = 25
        default_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        eta = estimate_step_3_llm_time(latest_kw, default_batch_size=default_batch_size, model=default_model)
        prompt_3 = "\nRun step 3 — LLM match for remaining unmatched items?"
        if eta:
            prompt_3 = f"{prompt_3} {eta}"
        if yn(prompt_3, default=step3_default):
            model = input(f"OpenAI model (default {os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}): ").strip()
            if not model:
                model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            batch_size = input("Batch size (default 25): ").strip() or "25"
            min_conf = input("Min confidence (default 0.9): ").strip() or "0.9"
            sleep_s = input("Sleep seconds between API calls (default 0): ").strip() or "0"

            cmd = [
                py,
                str(ROOT / "step-3" / "3_llm_match_unmatched.py"),
                "--model",
                model,
                "--batch-size",
                batch_size,
                "--min-confidence",
                min_conf,
                "--sleep-seconds",
                sleep_s,
            ]
            if latest_kw:
                cmd += ["--input", str(latest_kw)]
            s1m = newest_under_step1("**/1.6-manual_similar_title*.json")
            if s1m and yn(
                f"Merge step-1 similar-title assignments into the merged output?\n  {s1m.relative_to(ROOT)}",
                default=True,
            ):
                cmd += ["--step1-manual", str(s1m)]
            if sequential_fresh:
                cmd.append("--no-resume")
            elif yn("Start fresh for step 3 (no LLM checkpoint resume)?", default=False):
                cmd.append("--no-resume")
            run(cmd)

            latest_3m = newest_matching("1.4-llm_matched*.json", STEP3_OUT)
            latest_3u = newest_matching("1.4-llm_unmatched*.json", STEP3_OUT)
            if latest_3m:
                print(f"\nLatest step 3 matched: {latest_3m.relative_to(ROOT)}")
            if latest_3u:
                print(f"Latest step 3 unmatched: {latest_3u.relative_to(ROOT)}")
        else:
            print("Skipping step 3.")

    # --- Step 4: dedupe ---
    if not run_4:
        print(f"Skipping step 4 (dedupe) — starting at {start_step}.")
    elif yn(
        "\nRun step 4 — dedupe matched + unmatched from step 3 LLM outputs?",
        default=(start_step in ("1.1", "4")),
    ):
        cmd_4 = [py, str(ROOT / "step-4" / "4_dedupe_and_summaries.py"), "--pair-latest"]
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

    # --- Step 5: Attribute generation ---
    print()
    print("─" * 60)
    print("  Step 5 — LLM Attribute Generation")
    step5_script = ROOT / "step-5" / "5_generate_attributes.py"
    print(f"    python {step5_script.relative_to(ROOT)}")
    print("  (runs automatically if OPENAI_API_KEY is set)")
    print()

    if os.environ.get("OPENAI_API_KEY"):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(step5_script)],
            cwd=ROOT,
        )
        if result.returncode != 0:
            print("  WARNING: Step 5 attribute generation failed — continuing.")
    else:
        print("  OPENAI_API_KEY not set — skipping step 5.")
        print("  Run manually after setting the key.")
    print("─" * 60)

    # --- Step 6: DB upload ---
    print()
    print("─" * 60)
    print("  Step 6 — DB upload")
    print("  Run separately once the SSM tunnel is active:")
    step6_script = ROOT / "step-6" / "6_upload_to_db.py"
    print(f"    python {step6_script.relative_to(ROOT)}")
    print("─" * 60)


if __name__ == "__main__":
    main()
