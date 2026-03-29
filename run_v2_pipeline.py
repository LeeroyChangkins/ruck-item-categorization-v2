#!/usr/bin/env python3
"""
Master controller for Categorization v2 pipeline.

Runs:
  - step-1.0/generate_keywords_1_0.py
  - step-1.1/generate_bigrams_taxonomy_1_1a.py OR step-1.1/generate_bigrams_openai_1_1b.py
  - step-1.2/match_items_to_bigrams_1_2.py
  - step-1.3/interactive_keyword_match_1_3.py (optional)
  - step-1.4/llm_match_unmatched_1_4.py (optional; reads split unmatched_and_keywords + matched.json, optional 1.3 manual)
  - step-1.5/dedupe_and_cleanup_1_5.py (optional; dedupe 1.4 matched/unmatched outputs)
  - step-1.6/run_step_1_6.py (optional; menu: 1.6.1 aggregate groups or 1.6.2 interactive matching)

All step scripts write timestamped outputs into their respective step-*/outputs/ folders.
This controller mainly provides a friendly prompt flow and calls the step scripts.

Resume policy:
  - Start at 1.0 (full run): pass --no-resume / --fresh-run so each step does not continue
    from old checkpoints or manual files.
  - Start at a later step: default is to resume; optional prompts ask whether to start fresh
    for that step (1.1a/1.1b/1.2/1.3/1.4).
"""

from __future__ import annotations

import argparse
import re
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


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
    """
    options: list of (key, label)
    returns chosen key
    """
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


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    mins, sec = divmod(int(round(seconds)), 60)
    hrs, mins = divmod(mins, 60)
    if hrs >= 1:
        return f"{hrs}h {mins}m"
    if mins >= 1:
        return f"{mins}m"
    return f"{sec}s"


def estimate_step_14_time(
    latest_step12_path: Path | None,
    default_batch_size: int,
    model: str,
) -> str | None:
    if not latest_step12_path or not latest_step12_path.exists():
        return None
    try:
        data = json.loads(latest_step12_path.read_text(encoding="utf-8"))
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

    # Heuristic: 1 API call per batch. Prompt size includes full leaf list,
    # so latency is mostly per-call, not strictly per-item.
    model_l = (model or "").lower()
    if "mini" in model_l:
        per_call_fast_s = 20
        per_call_typ_s = 30
        per_call_slow_s = 60
    else:
        per_call_fast_s = 25
        per_call_typ_s = 40
        per_call_slow_s = 90

    fast = batches * per_call_fast_s
    typ = batches * per_call_typ_s
    slow = batches * per_call_slow_s
    return (
        f"~{batches} LLM batches (@batch_size={default_batch_size}, {n_items} unmatched items). "
        f"ETA: fast {_format_duration(fast)}, typical {_format_duration(typ)}, slow {_format_duration(slow)}. "
        f"(Sleep adds extra: ~({batches-1}*sleep_seconds))"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Categorization v2 pipeline runner")
    parser.add_argument(
        "--start-step",
        choices=["1.0", "1.1", "1.2", "1.3", "1.4", "1.5"],
        help="Which step to start with (steps before are skipped).",
    )
    cli_args = parser.parse_args()

    py = sys.executable or "python3"

    print("Categorization v2 pipeline runner\n")
    print("Inputs:")
    print("  - source-files/categories_v1.json")
    print("  - source-files/raw-prod-items-non-deleted.json")
    print("Outputs:")
    print("  - step-1.0/outputs/")
    print("  - step-1.1/outputs/")
    print("  - step-1.2/outputs/")
    print("  - step-1.3/outputs/")
    print("  - step-1.4/outputs/")
    print(
        "  - step-1.5/outputs/<run_id>/ (matched_deduped.json, unmatched_deduped.json, "
        "matched_summary.json, unmatched_summary.json)"
    )

    cascade_mapping_paths: list[Path] | None = None

    start_step = cli_args.start_step
    if not start_step:
        start_step = choose(
            "\nWhich step do you want to start with?",
            [
                ("1.0", "Start at step 1.0 (keywords)"),
                ("1.1", "Start at step 1.1 (bigram mappings)"),
                ("1.2", "Start at step 1.2 (item matching)"),
                ("1.3", "Start at step 1.3 (interactive keyword → leaf matching)"),
                ("1.4", "Start at step 1.4 (LLM match unmatched items)"),
                ("1.5", "Start at step 1.5 (dedupe / cleanup 1.4 outputs)"),
            ],
        )

    print(f"\nStarting from step {start_step}.")
    # Full run from 1.0: default to fresh outputs / no checkpoint resume for each step.
    # Jumping in at a later step: default to resuming prior artifacts; prompts offer "fresh" opt-in.
    sequential_from_10 = start_step == "1.0"

    run_step_10 = start_step == "1.0"
    run_step_11 = start_step in ("1.0", "1.1")
    run_step_12 = start_step in ("1.0", "1.1", "1.2")
    run_step_13 = start_step in ("1.0", "1.1", "1.2", "1.3")
    run_step_14 = start_step in ("1.0", "1.1", "1.2", "1.3", "1.4")
    run_step_15 = start_step in ("1.0", "1.1", "1.2", "1.3", "1.4", "1.5")

    # Step 1.0
    if not run_step_10:
        print("Skipping 1.0.")
    elif yn("Run step 1.0 (generate keywords)?", default=True):
        cmd_10 = [py, str(ROOT / "step-1.0" / "generate_keywords_1_0.py")]
        if sequential_from_10:
            cmd_10.append("--no-resume")
        run(cmd_10)
    else:
        print("Skipping 1.0.")

    # Pick 1.0 file (for 1.1)
    step10_out = ROOT / "step-1.0" / "outputs"
    latest_10 = newest_matching("1.0-title_subtitle_keyword_frequencies*.json", step10_out)
    if latest_10:
        print(f"\nMost recent 1.0 output: {latest_10.relative_to(ROOT)}")
    else:
        if run_step_11:
            print("\nNo 1.0 output files found. You may need to run step 1.0 first.")

    # Step 1.1
    mapping_path: Path | None = None
    if run_step_11:
        step11_choice = choose(
            "\nChoose step 1.1 mapping method:",
            [
                ("a", "1.1a taxonomy-based (fast, deterministic)"),
                ("b", "1.1b OpenAI LLM-based (slower, semantic)"),
                ("s", "skip 1.1 (use latest mapping for step 1.2)"),
            ],
        )

        if step11_choice == "a":
            cmd = [py, str(ROOT / "step-1.1" / "generate_bigrams_taxonomy_1_1a.py")]
            if latest_10 and yn("Use most recent 1.0 keywords file?", default=True):
                cmd += ["--keywords", str(latest_10)]
            else:
                p = input("Path to 1.0 keywords JSON: ").strip()
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
            if sequential_from_10:
                cmd.append("--no-resume")
            elif yn("Start fresh for 1.1a (no checkpoint resume)?", default=False):
                cmd.append("--no-resume")
            run(cmd)
            step11_out = ROOT / "step-1.1" / "outputs"
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

            cmd = [py, str(ROOT / "step-1.1" / "generate_bigrams_openai_1_1b.py")]
            if latest_10 and yn("Use most recent 1.0 keywords file?", default=True):
                cmd += ["--keywords", str(latest_10)]
            else:
                p = input("Path to 1.0 keywords JSON: ").strip()
                if p:
                    cmd += ["--keywords", p]
            cmd += ["--model", model, "--batch-size", batch_size, "--min-confidence", min_conf, "--sleep-seconds", sleep_s]
            if sequential_from_10:
                cmd.append("--no-resume")
            elif yn("Start fresh for 1.1b (no checkpoint resume)?", default=False):
                cmd.append("--no-resume")
            run(cmd)
            cascade_mapping_paths = None
            mapping_path = newest_matching("1.1b-bigram_categories_mapping*.json", ROOT / "step-1.1" / "outputs")

        else:
            cascade_mapping_paths = None
            step11_out = ROOT / "step-1.1" / "outputs"
            mapping_path = newest_matching("1.1*-bigram_categories_mapping*.json", step11_out)
            print("Using latest 1.1 mapping for step 1.2.")
    else:
        print(f"Skipping 1.1 (starting at {start_step}).")
        step11_out = ROOT / "step-1.1" / "outputs"
        mapping_path = newest_matching("1.1*-bigram_categories_mapping*.json", step11_out)
        if not mapping_path:
            print("No 1.1 mapping found. Step 1.2 may prompt for it.")

    if mapping_path:
        print(f"\nMapping file to use next: {mapping_path.relative_to(ROOT)}")
    else:
        print("\nNo mapping file detected (1.1a/1.1b). Step 1.2 can still prompt for one.")

    # Step 1.2
    if not run_step_12:
        print(f"Done (skipped 1.2) because starting at {start_step}.")
    elif yn("\nRun step 1.2 (match items to bigrams)?", default=True):
        latest_map = newest_matching("1.1*-bigram_categories_mapping*.json", ROOT / "step-1.1" / "outputs")
        if latest_map:
            print(f"Most recent mapping available: {latest_map.relative_to(ROOT)}")
        cmd = [py, str(ROOT / "step-1.2" / "match_items_to_bigrams_1_2.py")]
        if cascade_mapping_paths:
            for p in cascade_mapping_paths:
                cmd += ["--cascade-mapping", str(p)]
            print(f"Using phased cascade: {len(cascade_mapping_paths)} mapping files (T0 first).")
        elif latest_map:
            cmd += ["--mapping", str(latest_map)]
        strict_sides = yn("Use strict side matching (title-only/subtitle-only)?", default=False)
        if strict_sides:
            cmd += ["--strict-sides"]
        if sequential_from_10:
            cmd.append("--no-resume")
        elif yn("Start fresh for 1.2 (no checkpoint resume)?", default=False):
            cmd.append("--no-resume")
        run(cmd)
    else:
        print("Done (skipped 1.2).")

    # Helpful pointers
    latest_12 = newest_matching("1.2-bigram_sorted_items*.json", ROOT / "step-1.2" / "outputs")
    if latest_12:
        print(f"\nLatest 1.2 output: {latest_12.relative_to(ROOT)}")

    latest_kw = newest_matching("**/unmatched_and_keywords.json", ROOT / "step-1.2" / "outputs")
    if latest_kw:
        print(f"\nLatest unmatched_and_keywords (split): {latest_kw.relative_to(ROOT)}")

    # Step 1.3 (optional): interactive keyword → leaf
    if not run_step_13:
        print(f"Skipping 1.3 (starting at {start_step}).")
    elif yn("\nRun step 1.3 (interactive manual bigram → leaf matching)?", default=(start_step == "1.3")):
        cmd_13 = [py, str(ROOT / "step-1.3" / "interactive_keyword_match_1_3.py")]
        if sequential_from_10:
            cmd_13.append("--fresh-run")
        run(cmd_13)
        latest_13 = newest_matching("1.3-manual*.json", ROOT / "step-1.3" / "outputs")
        if latest_13:
            print(f"\nLatest 1.3 manual match output: {latest_13.relative_to(ROOT)}")
    else:
        print("Skipping 1.3.")

    # Step 1.4 (optional): LLM
    if not run_step_14:
        print(f"Skipping 1.4 (starting at {start_step}).")
    else:
        step14_default = start_step == "1.4"
        latest_default_batch_size = 25
        default_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        eta = estimate_step_14_time(latest_kw, default_batch_size=latest_default_batch_size, model=default_model)
        step14_prompt = "\nRun step 1.4 (LLM match remaining unmatched items)?"
        if eta:
            step14_prompt = f"{step14_prompt} {eta}"
        if yn(step14_prompt, default=step14_default):
            model = input(f"OpenAI model (default {os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}): ").strip()
            if not model:
                model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            batch_size = input("Batch size (default 25): ").strip() or "25"
            min_conf = input("Min confidence (default 0.9): ").strip() or "0.9"
            sleep_s = input("Sleep seconds between API calls (default 0): ").strip() or "0"

            cmd = [
                py,
                str(ROOT / "step-1.4" / "llm_match_unmatched_1_4.py"),
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
            if sequential_from_10:
                cmd.append("--no-resume")
            elif yn("Start fresh for 1.4 (no checkpoint resume)?", default=False):
                cmd.append("--no-resume")
            run(cmd)

            latest_14_m = newest_matching("1.4-llm_matched*.json", ROOT / "step-1.4" / "outputs")
            latest_14_u = newest_matching("1.4-llm_unmatched*.json", ROOT / "step-1.4" / "outputs")
            if latest_14_m:
                print(f"\nLatest 1.4 matched output: {latest_14_m.relative_to(ROOT)}")
            if latest_14_u:
                print(f"Latest 1.4 unmatched output: {latest_14_u.relative_to(ROOT)}")
        else:
            print("Skipping 1.4.")

    # Step 1.5 (optional): dedupe / cleanup 1.4 outputs
    if not run_step_15:
        print(f"Skipping 1.5 (starting at {start_step}).")
    elif yn(
        "\nRun step 1.5 (dedupe matched + unmatched from step 1.4)?",
        default=(start_step in ("1.0", "1.5")),
    ):
        cmd_15 = [py, str(ROOT / "step-1.5" / "dedupe_and_cleanup_1_5.py"), "--pair-latest"]
        run(cmd_15)
        step15_out = ROOT / "step-1.5" / "outputs"
        latest_15_m = newest_matching("**/matched_deduped.json", step15_out)
        latest_15_u = newest_matching("**/unmatched_deduped.json", step15_out)
        latest_15_ms = newest_matching("**/matched_summary.json", step15_out)
        latest_15_us = newest_matching("**/unmatched_summary.json", step15_out)
        if latest_15_m:
            print(f"\nLatest 1.5 matched output: {latest_15_m.relative_to(ROOT)}")
        if latest_15_u:
            print(f"Latest 1.5 unmatched output: {latest_15_u.relative_to(ROOT)}")
        if latest_15_ms:
            print(f"Latest 1.5 matched summary: {latest_15_ms.relative_to(ROOT)}")
        if latest_15_us:
            print(f"Latest 1.5 unmatched summary: {latest_15_us.relative_to(ROOT)}")
    else:
        print("Skipping 1.5.")


if __name__ == "__main__":
    main()

