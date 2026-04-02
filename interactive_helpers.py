"""
interactive_helpers.py — Shared UX utilities for interactive v2 pipeline scripts.

Used by:
  step-1/1_2_interactive_similar_title_match.py
  step-2/2_3_interactive_keyword_match.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote_plus


# ── Clipboard ──────────────────────────────────────────────────────────────────

def copy_to_clipboard(text: str) -> bool:
    """Best-effort clipboard copy (macOS pbcopy; silently fails elsewhere)."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
            return True
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return False


# ── Google Images via Chrome ───────────────────────────────────────────────────

_chrome_search_window_opened: bool = False


def open_google_images_in_chrome(query: str) -> bool:
    """Open or reuse a single Chrome window for Google Images searches (macOS).

    First call: opens a new Chrome window at the search URL.
    Subsequent calls: updates the URL in that same window's active tab.
    After navigating, returns focus to Terminal / iTerm2.
    """
    global _chrome_search_window_opened
    url = f"https://www.google.com/search?q={quote_plus(query)}&tbm=isch"

    if sys.platform != "darwin":
        return False

    try:
        if not _chrome_search_window_opened:
            script = f'''
tell application "Google Chrome"
    make new window
    set URL of active tab of front window to "{url}"
    activate
end tell
'''
        else:
            script = f'''
tell application "Google Chrome"
    set URL of active tab of front window to "{url}"
    activate
end tell
'''
        subprocess.run(["osascript", "-e", script], check=False, capture_output=True)
        _chrome_search_window_opened = True

        focus_script = '''
tell application "System Events"
    set termApps to {"Terminal", "iTerm2", "iTerm"}
    repeat with appName in termApps
        if exists application process appName then
            set frontmost of application process appName to true
            exit repeat
        end if
    end repeat
end tell
'''
        subprocess.run(["osascript", "-e", focus_script], check=False, capture_output=True)
        return True
    except OSError:
        pass
    return False


# ── Taxonomy helpers ───────────────────────────────────────────────────────────

def collect_leaf_rows(categories: dict) -> List[Dict[str, str]]:
    """Return a flat list of all leaf rows from the taxonomy dict."""
    rows: List[Dict[str, str]] = []
    roots = ["materials", "tools_and_gear", "services"]

    def walk(node: object, prefix: List[str]) -> None:
        if not isinstance(node, dict):
            return
        slug = node.get("slug")
        subs = node.get("subcategories") or []
        dn = (node.get("display_name") or "").strip()

        here = prefix + [slug] if slug is not None else prefix

        if not subs:
            if slug is not None:
                lp = "/".join(here)
                hay = f"{lp} {slug} {dn}".lower()
                rows.append({"leaf_path": lp, "leaf_slug": slug, "display_name": dn, "_hay": hay})
            return
        for child in subs:
            walk(child, here)

    for root in roots:
        root_obj = categories.get(root)
        if not isinstance(root_obj, dict):
            continue
        for child in root_obj.get("subcategories", []):
            walk(child, [root])

    rows.sort(key=lambda r: r["leaf_path"])
    return rows


def filter_leaves(leaves: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
    q = query.strip().lower()
    if not q:
        return []
    return [r for r in leaves if q in r["_hay"]]


def filter_hits_narrow(hits: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
    q = query.strip().lower()
    if not q:
        return hits
    return [r for r in hits if q in r["_hay"]]


def collect_other_bucket_leaves(raw_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Catch-all bucket leaves: slug `other` (per domain) plus materials top-level `miscellaneous`."""
    return [
        r for r in raw_rows
        if r.get("leaf_slug") == "other" or r.get("leaf_slug") == "miscellaneous"
    ]


def interact_pick_other_leaf(other_rows: List[Dict[str, str]]) -> tuple[str, Dict[str, str] | None]:
    """
    Prompt user to pick an Other bucket leaf.
    Returns ("assign", row) | ("unknown", None) | ("skip", None).
    Enter = skip matching entirely (no assign, no unknown; item stays unmatched for later).
    """
    if not other_rows:
        print("  (No Other leaves in taxonomy; add them in categories_v1.json.)")
        return ("skip", None)
    print(f"  Other buckets ({len(other_rows)}) — pick a catch-all leaf:")
    for i, h in enumerate(other_rows, start=1):
        print(f"    {i:3}) {h['leaf_path']}")
    while True:
        pick = input(
            "  [#] assign here | [Enter] skip (no category) | [x] mark unknown (saved): "
        ).strip()
        if not pick:
            return ("skip", None)
        if pick.lower() == "x":
            return ("unknown", None)
        if re.fullmatch(r"\d+", pick):
            n = int(pick)
            if 1 <= n <= len(other_rows):
                return ("assign", other_rows[n - 1])
            print("  Out of range.")
            continue
        print("  Enter a number, Enter, or x.")


# ── Slug helper ────────────────────────────────────────────────────────────────

def _make_slug(name: str) -> str:
    """Slugify a display name: lowercase, non-alphanumeric runs → underscore, trim underscores."""
    s = re.sub(r"[^a-z0-9]+", "_", name.lower().strip())
    return s.strip("_")


# ── Insert new category wizard ─────────────────────────────────────────────────

def interact_insert_new_category(
    categories: dict,
    taxonomy_path: Path,
) -> Dict[str, str] | None:
    """Interactive wizard to create a new leaf category in the taxonomy.

    Navigation rules:
      - Level 0 (master: materials / tools_and_gear / services): select only, no creation.
      - Level 1 (tier-1 nodes under master): navigate only, no creation.
      - Level 2+ (tier-2 and deeper): [c] create new category here, [b] back.

    On creation:
      - Slugifies the display name and checks for uniqueness among siblings.
      - Inserts the new node into the in-memory `categories` dict (mutates caller's dict).
      - Writes categories_v1.json immediately (atomic via .tmp).

    Returns the new leaf's row dict (leaf_path, leaf_slug, display_name, _hay),
    or None if the user cancelled at any point.
    """
    ROOT_KEYS = ("materials", "tools_and_gear", "services")

    roots = [
        (k, categories[k].get("display_name") or k)
        for k in ROOT_KEYS
        if isinstance(categories.get(k), dict)
    ]
    print("\n  ── Insert new category ───────────────────────────────────────────")
    print("  Select master category:")
    for i, (k, dn) in enumerate(roots, 1):
        print(f"    [{i}]  {dn}  ({k})")
    print("    [Enter]  cancel")
    while True:
        r = input("  → ").strip()
        if not r:
            print("  Cancelled.")
            return None
        if r.isdigit() and 1 <= int(r) <= len(roots):
            root_key, _ = roots[int(r) - 1]
            break
        print("  Enter a number or Enter to cancel.")

    node_stack: list[dict] = [categories[root_key]]
    slug_stack: list[str] = [root_key]

    while True:
        current = node_stack[-1]
        current_path = "/".join(slug_stack)
        depth = len(slug_stack)
        can_create = depth >= 2

        children = [
            c for c in (current.get("subcategories") or [])
            if isinstance(c, dict) and c.get("slug")
        ]

        print(f"\n  /{current_path}/")
        if children:
            for i, child in enumerate(children, 1):
                n_sub = sum(
                    1 for c in (child.get("subcategories") or [])
                    if isinstance(c, dict) and c.get("slug")
                )
                dn = child.get("display_name") or child["slug"]
                tag = f"({n_sub} sub)" if n_sub else "[leaf]"
                print(f"    [{i:3}]  {child['slug']}  —  {dn}  {tag}")
        else:
            print("    (no subcategories yet)")

        if can_create:
            print(f"    [c]    Create new category under /{current_path}/")
        back_target = "/".join(slug_stack[:-1]) if len(slug_stack) > 1 else "root selection"
        print(f"    [b]    Back  (→ /{back_target}/)")
        print("    [Enter] Cancel")

        sel = input("  → ").strip()

        if not sel:
            print("  Cancelled.")
            return None

        if sel.lower() == "b":
            if len(slug_stack) == 1:
                return interact_insert_new_category(categories, taxonomy_path)
            node_stack.pop()
            slug_stack.pop()
            continue

        if sel.lower() == "c" and can_create:
            existing_slugs = {
                c["slug"] for c in children
                if isinstance(c, dict) and c.get("slug")
            }
            while True:
                name = input(f"\n  Display name for new category under /{current_path}/: ").strip()
                if not name:
                    print("  Cancelled.")
                    return None
                slug = _make_slug(name)
                if not slug:
                    print("  Could not generate a valid slug from that name — try again.")
                    continue
                if slug in existing_slugs:
                    print(f"  Error: slug '{slug}' already exists at this level. Use a different name.")
                    continue
                new_path = f"{current_path}/{slug}"
                print(f"\n  Will create:  {new_path}")
                print(f"  Display name: {name!r}")
                confirm = input("  Confirm? [Y/n] ").strip().lower()
                if confirm in ("n", "no"):
                    print("  Discarded — try again.")
                    continue
                break

            new_node: dict = {"slug": slug, "display_name": name, "subcategories": []}
            if not isinstance(current.get("subcategories"), list):
                current["subcategories"] = []
            current["subcategories"].append(new_node)

            tmp = taxonomy_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(categories, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            tmp.replace(taxonomy_path)
            print(f"  Saved → {taxonomy_path.name}  ({new_path})")

            hay = f"{new_path} {slug} {name}".lower()
            return {
                "leaf_path": new_path,
                "leaf_slug": slug,
                "display_name": name,
                "_hay": hay,
            }

        if sel.isdigit():
            n = int(sel)
            if 1 <= n <= len(children):
                child = children[n - 1]
                node_stack.append(child)
                slug_stack.append(child["slug"])
                continue
            print("  Out of range.")
            continue

        opts = "[c] / " if can_create else ""
        print(f"  Enter a number, {opts}[b], or Enter to cancel.")


# ── yn prompt ──────────────────────────────────────────────────────────────────

def yn_prompt(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        ans = input(f"{prompt} [{hint}] ").strip().lower()
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
