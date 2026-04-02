#!/usr/bin/env python3
"""
6_upload_to_db.py — v2 DB upload orchestrator.

Pushes step-4 categorization results (matched_deduped.json) to the marketplace
database.  This is a push-only workflow — the v2 pipeline handles all data
preparation in steps 1–4.

Flow:
  1. Select environment: dev or prod
  2. Check SSM tunnel is active
  3. Load DB credentials (from .env or interactive prompt)
  4. Browse step-4/outputs/ and select a run folder
  5. Summarize what will be uploaded
  6. Optionally check taxonomy sync and push missing categories first
  7. Run pre-upload validation (6_validate.py)
  8. Confirm dry-run vs live
  9. Push item-category relationships

Prerequisites:
  - SSM tunnel running on the expected local port (ruck-db-staging / ruck-db-prod)
  - DB_USER and DB_PASSWORD set in .env at the v2 root, or entered interactively
  - psycopg2-binary installed (pip install psycopg2-binary)

Setup guide: https://www.notion.so/AWS-Local-Configuration-2ad33ea5030a80649112edd494fc7c28
"""

from __future__ import annotations

import getpass
import importlib.util
import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

# ── Load .env if present ───────────────────────────────────────────────────────

STEP6_DIR = Path(__file__).resolve().parent
ROOT = STEP6_DIR.parent  # v2/ root
STEP4_OUTPUTS = ROOT / "step-4" / "outputs"


def _import_sibling(module_name: str, filename: str):
    """Load a sibling module from this step-6 directory by filename (handles digit-prefixed names)."""
    path = STEP6_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

sys.path.insert(0, str(ROOT))
from shared_utils import load_dotenv_file as _load_dotenv_file
_load_dotenv_file()

# ── Helpers ────────────────────────────────────────────────────────────────────


def _hr(char: str = "─", width: int = 60) -> None:
    print(char * width)


def _section(title: str) -> None:
    print()
    _hr()
    print(f"  {title}")
    _hr()


def _ask(prompt: str, options: list[str]) -> str:
    """Prompt user to pick one of a numbered list; return the chosen option string."""
    while True:
        print()
        print(prompt)
        for i, opt in enumerate(options, 1):
            print(f"  [{i}] {opt}")
        raw = input("  → ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            chosen = options[int(raw) - 1]
            print(f"  Selected: {chosen}")
            return chosen
        print("  Invalid selection — please enter a number from the list.")


def _confirm(prompt: str, default_yes: bool = False) -> bool:
    hint = " [Y/n]" if default_yes else " [y/N]"
    while True:
        raw = input(f"  {prompt}{hint}: ").strip().lower()
        if raw == "" and default_yes:
            return True
        if raw == "" and not default_yes:
            return False
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please enter y or n.")


def _prompt_credentials() -> tuple[str, str]:
    env_user = os.environ.get("DB_USER", "")
    env_pass = os.environ.get("DB_PASSWORD", "")

    if env_user and env_pass:
        print(f"  DB credentials loaded from .env (user: {env_user})")
        return env_user, env_pass

    user = input(f"  DB username [{env_user}]: ").strip() or env_user
    password = getpass.getpass("  DB password: ") or env_pass
    return user, password


def _check_tunnel(env: str) -> bool:
    """Check whether the SSM tunnel is active on the expected local port."""
    port = 15432 if env == "dev" else 15433
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            return True
    except OSError:
        return False


def _connect(env: str, user: str, password: str):
    s5 = _import_sibling("s6_db_import", "6_db_import.py")
    cfg = s5.DEV_DB_CONFIG if env == "dev" else s5.PROD_DB_CONFIG
    try:
        return s5.make_connection(cfg, user, password)
    except Exception as e:
        print(f"\n  ERROR: Could not connect to {env} database: {e}")
        print("  Make sure the SSM tunnel is running (ruck-db-staging / ruck-db-prod).")
        raise


def _list_step4_runs() -> list[Path]:
    """Return step-4 output run dirs that contain matched_deduped.json, newest first."""
    if not STEP4_OUTPUTS.exists():
        return []
    return sorted(
        (p for p in STEP4_OUTPUTS.iterdir() if p.is_dir() and (p / "matched_deduped.json").is_file()),
        key=lambda p: p.name,
        reverse=True,
    )


def _load_run_summary(run_dir: Path) -> tuple[int, str]:
    """Return (matched_item_count, run_id) for a step-4 run dir."""
    matched_path = run_dir / "matched_deduped.json"
    try:
        data = json.loads(matched_path.read_text(encoding="utf-8"))
        items = data.get("matched_items") if isinstance(data, dict) else data
        count = len(items) if isinstance(items, list) else 0
        run_id = data.get("run_id", run_dir.name) if isinstance(data, dict) else run_dir.name
        return count, run_id
    except Exception:
        return 0, run_dir.name


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Upload flow ─────────────────────────────────────────────────────────────────


def run_upload(env: str) -> None:
    _section(f"UPLOAD — {env.upper()} database")

    # ── Select step-4 run ─────────────────────────────────────────────────────
    runs = _list_step4_runs()
    if not runs:
        print(f"\n  No matched_deduped.json found under {STEP4_OUTPUTS.relative_to(ROOT)}")
        print("  Run step-4 (4_dedupe_and_summaries.py) first to produce output.")
        return

    print("\n  Available step-4 runs (matched_deduped.json):")
    options: list[str] = []
    for i, run_dir in enumerate(runs):
        count, run_id = _load_run_summary(run_dir)
        label = f"{run_dir.name}  ({count:,} items)"
        if i == 0:
            label += "  ← latest"
        options.append(label)
        print(f"  [{i + 1}] {label}")

    print()
    raw = input("  Select run [1]: ").strip()
    idx = (int(raw) - 1) if raw.isdigit() and 1 <= int(raw) <= len(runs) else 0
    selected_run = runs[idx]
    matched_path = selected_run / "matched_deduped.json"
    count, run_id = _load_run_summary(selected_run)
    print(f"  Selected: {selected_run.name}  ({count:,} matched items)")

    _do_upload(env, matched_path)


def _do_upload(
    env: str,
    matched_deduped_path: Path,
    db_user: str | None = None,
    db_password: str | None = None,
    dry_run: bool | None = None,
) -> None:
    """Core upload logic."""
    _section(f"UPLOAD — {env.upper()} database")

    # ── Credentials ───────────────────────────────────────────────────────────
    if not db_user:
        print()
        db_user, db_password = _prompt_credentials()

    # ── Confirm dry/live ──────────────────────────────────────────────────────
    if dry_run is None:
        dry_run = not _confirm("Write to the database? (No = dry-run preview)", default_yes=False)

    mode_label = "DRY RUN (no writes)" if dry_run else "LIVE RUN (writing to DB)"
    print(f"\n  Mode: {mode_label}")

    # ── Import DB module (deferred so missing psycopg2 produces a clear error) ──
    s5 = _import_sibling("s6_db_import", "6_db_import.py")
    s5.DRY_RUN = dry_run

    # ── Taxonomy sync check (live only) ───────────────────────────────────────
    if not dry_run:
        print("\n  Checking taxonomy sync…")
        conn = _connect(env, db_user, db_password)
        try:
            tax_gaps = s5.get_taxonomy_gaps(conn)
        finally:
            conn.close()

        if tax_gaps["missing_categories"]:
            missing_slugs = tax_gaps["_missing_cat_slugs"]
            print()
            print(f"  Taxonomy gaps found: {tax_gaps['missing_categories']} category slug(s) missing from DB.")
            if _confirm("Preview missing slugs (top 25)?", default_yes=False):
                for slug in missing_slugs[:25]:
                    print(f"      - {slug}")
                if len(missing_slugs) > 25:
                    print(f"      … and {len(missing_slugs) - 25} more")
            print()
            if _confirm("Sync missing categories to DB now?", default_yes=False):
                conn = _connect(env, db_user, db_password)
                try:
                    s5.push_taxonomy(conn)
                    conn.commit()
                    print("  Taxonomy synced.")
                finally:
                    conn.close()
            else:
                print("  Skipping taxonomy sync — items referencing missing categories will be skipped.")
        else:
            print("  Taxonomy is in sync.")

    # ── Pre-upload validation ─────────────────────────────────────────────────
    print("\n  Running pre-upload validation…")
    s5v = _import_sibling("s6_validate", "6_validate.py")
    issues = s5v.run_validation(matched_deduped_path)
    if issues:
        print(f"\n  Validation found {len(issues)} issue(s):")
        for issue in issues[:30]:
            print(f"    - {issue}")
        if len(issues) > 30:
            print(f"    … and {len(issues) - 30} more")
        print()
        if not _confirm("Continue with upload despite validation issues?", default_yes=False):
            print("  Upload cancelled. Fix issues and re-run.")
            return
    else:
        print("  Validation passed.")

    # ── Dry-run summary ───────────────────────────────────────────────────────
    if dry_run:
        data = json.loads(matched_deduped_path.read_text(encoding="utf-8"))
        items = data.get("matched_items") if isinstance(data, dict) else data
        count = len(items) if isinstance(items, list) else 0
        print(f"\n  [DRY RUN] Would push {count:,} item-category rows from:")
        print(f"    {matched_deduped_path.relative_to(ROOT)}")
        print("\n  Dry run complete — no changes made.")
        return

    # ── Push item-category relationships ─────────────────────────────────────
    print("\n  Pushing item-category relationships to DB…")
    conn = _connect(env, db_user, db_password)
    try:
        s5.push_item_relationships(conn, matched_deduped_path)
        conn.commit()
        print("  Item-category upload complete.")
    except Exception as e:
        conn.rollback()
        print(f"\n  ERROR during item-category upload: {e}")
        print("  Transaction rolled back.")
        raise
    finally:
        conn.close()

    # ── Push attributes + attribute values ────────────────────────────────────
    print()
    if _confirm("Push attribute schema and item attribute values to DB?", default_yes=True):
        conn = _connect(env, db_user, db_password)
        try:
            s5.push_attributes(conn)
            conn.commit()
            print("  Attribute upload complete.")
        except Exception as e:
            conn.rollback()
            print(f"\n  ERROR during attribute upload: {e}")
            print("  Transaction rolled back — item-category data was already committed.")
            raise
        finally:
            conn.close()
    else:
        print("  Skipping attribute push.")

    print("\n  Upload complete.")


# ── Main menu ──────────────────────────────────────────────────────────────────


def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      Ruck v2 Categorization — DB Upload (step 5)        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    env = _ask("Target database:", ["dev (staging)", "prod"])
    env = "dev" if env.startswith("dev") else "prod"

    # ── Tunnel check ──────────────────────────────────────────────────────────
    port = 15432 if env == "dev" else 15433
    tunnel_cmd = "ruck-db-staging" if env == "dev" else "ruck-db-prod"
    print(f"\n  Checking SSM tunnel on 127.0.0.1:{port}…", end=" ", flush=True)
    if not _check_tunnel(env):
        print("NOT FOUND")
        print()
        print(f"  ERROR: No active tunnel detected on port {port}.")
        print(f"  Run '{tunnel_cmd}' in a separate terminal and try again.")
        print()
        print("  Setup guide: https://www.notion.so/AWS-Local-Configuration-2ad33ea5030a80649112edd494fc7c28")
        return
    print("OK")

    run_upload(env)


if __name__ == "__main__":
    main()
