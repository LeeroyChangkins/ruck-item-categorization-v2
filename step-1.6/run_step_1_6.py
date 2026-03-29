#!/usr/bin/env python3
"""
Single entry for step 1.6: choose aggregate (1.6.1) or interactive matching (1.6.2).

Usage:
  python3 run_step_1_6.py [extra args...]

Extra arguments are forwarded to the script you select (e.g. --min-group-size 5 for 1.6.1).

After 1.6.1 finishes successfully, you are asked whether to continue to 1.6.2 (no extra args are
passed to 1.6.2 in that flow, so build-only flags are not forwarded).

  python3 run_step_1_6.py --help   # this launcher only
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build_unmatched_similar_title_groups_1_6.py"
INTERACTIVE = HERE / "interactive_similar_title_match_1_6.py"


def yn_continue_to_1_6_2() -> bool:
    while True:
        a = input("\nContinue to 1.6.2 (interactive matching)? [Y/n] ").strip().lower()
        if not a or a in ("y", "yes"):
            return True
        if a in ("n", "no"):
            return False
        print("  Enter y or n.")


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in ("-h", "--help"):
        print(
            __doc__.strip()
            + "\n\nSee also:\n"
            + f"  {BUILD.name} --help\n"
            + f"  {INTERACTIVE.name} --help\n"
        )
        sys.exit(0)

    py = sys.executable or "python3"
    print("Step 1.6 — unmatched similar titles")
    print("  1) 1.6.1 — Build similar-title groups (aggregate from latest unmatched_deduped)")
    print("  2) 1.6.2 — Interactive matching to taxonomy leaves (resume supported)")
    while True:
        choice = input("Select [1/2]: ").strip()
        if choice == "1":
            rc = subprocess.call([py, str(BUILD)] + argv)
            if rc != 0:
                sys.exit(rc)
            if yn_continue_to_1_6_2():
                rc2 = subprocess.call([py, str(INTERACTIVE)])
                sys.exit(rc2)
            sys.exit(0)
        if choice == "2":
            rc = subprocess.call([py, str(INTERACTIVE)] + argv)
            sys.exit(rc)
        print("  Enter 1 or 2.")


if __name__ == "__main__":
    main()
