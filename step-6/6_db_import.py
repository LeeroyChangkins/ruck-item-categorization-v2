#!/usr/bin/env python3
"""
6_db_import.py

DB push and pull module for v2 marketplace categorization data.
Designed to be imported by 6_upload_to_db.py; contains no interactive prompts.

v2 differences from v1:
  - Taxonomy loaded from source-files/categories_v1.json (recursive tree, not tier1/tier2/tier3 keyed JSON)
  - Item mappings loaded from step-4/outputs/<run_id>/matched_deduped.json (not CSVs)
  - Attributes loaded from source-files/proposed-attributes.json

Push tables (in dependency order):
  1. marketplace_categories
  2. marketplace_item_categories
  3. marketplace_attribute_units
  4. marketplace_attributes

Pull tables (DB → local files):
  items, marketplace_categories, marketplace_item_categories,
  marketplace_attributes, marketplace_attribute_units, marketplace_attribute_values

No other tables are touched.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import psycopg2
from tqdm import tqdm  # noqa: F401

# ──────────────────────────────────────────────────────────────────────────────
# DB configs — user/password are passed in at call time, never stored here
# ──────────────────────────────────────────────────────────────────────────────

DEV_DB_CONFIG: dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 15432,
    "dbname": "flightcontrol",
    "sslmode": "require",
}

PROD_DB_CONFIG: dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 15433,
    "dbname": "flightcontrol",
    "sslmode": "require",
}

# ──────────────────────────────────────────────────────────────────────────────
# File paths
# ──────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent  # v2/ root
TAXONOMY_PATH = ROOT / "source-files" / "categories_v1.json"
ATTRIBUTES_PATH = ROOT / "source-files" / "proposed-attributes.json"
STEP4_OUTPUTS = ROOT / "step-4" / "outputs"

ROOT_CATEGORY_KEYS = ("materials", "tools_and_gear", "services")

# ──────────────────────────────────────────────────────────────────────────────
# DRY_RUN flag — set by 5_upload_to_db.py before calling push functions
# ──────────────────────────────────────────────────────────────────────────────

DRY_RUN: bool = False


def make_connection(db_config: dict, user: str, password: str):
    """Open and return a psycopg2 connection."""
    return psycopg2.connect(**db_config, user=user, password=password)


# ──────────────────────────────────────────────────────────────────────────────
# Taxonomy helpers
# ──────────────────────────────────────────────────────────────────────────────


def build_category_nodes(
    categories: dict,
    taxonomy_path: Path | None = None,
) -> list[tuple[str, str, str | None]]:
    """
    Flatten the full v2 taxonomy tree into (slug_path, display_name, parent_slug_path) tuples.

    Root keys (materials / tools_and_gear / services) have no 'slug' field — the dict
    key itself is the slug.  All deeper nodes carry a 'slug' field.

    Returned in breadth-first (parent-before-child) order so parent IDs are available
    when children are inserted.
    """
    if taxonomy_path is not None and taxonomy_path.is_file():
        categories = json.loads(taxonomy_path.read_text(encoding="utf-8"))

    nodes: list[tuple[str, str, str | None]] = []

    def walk(slug_path: str, node: dict, parent_path: str | None) -> None:
        display_name = node.get("display_name") or slug_path.split("/")[-1]
        nodes.append((slug_path, display_name, parent_path))
        for child in node.get("subcategories") or []:
            child_slug = child.get("slug")
            if child_slug:
                walk(f"{slug_path}/{child_slug}", child, slug_path)

    for key in ROOT_CATEGORY_KEYS:
        root = categories.get(key)
        if isinstance(root, dict):
            walk(key, root, None)

    return nodes


def load_matched_deduped(path: Path) -> list[dict]:
    """
    Load matched_deduped.json from a step-4 output run.
    Returns the list of matched item rows (each has id, leaf_path, …).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        items = data.get("matched_items")
        if isinstance(items, list):
            return items
    return []


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — marketplace_categories
# ──────────────────────────────────────────────────────────────────────────────


def import_categories(cur: Any, nodes: list[tuple[str, str, str | None]]) -> dict[str, str]:
    """
    Insert categories in parent-before-child order.
    Skips categories that already exist (matched by slug + parent_id).
    Returns slug_path → id map.
    """
    if DRY_RUN:
        path_to_id: dict[str, str] = {p: f"<dry:{p}>" for p, _, _ in nodes}
        depth_counts: dict[int, int] = {}
        for path, _, _ in nodes:
            d = path.count("/") + 1
            depth_counts[d] = depth_counts.get(d, 0) + 1
        print(f"  [categories] would ensure {len(nodes)} category nodes are present")
        for depth in sorted(depth_counts):
            print(f"    tier{depth}: {depth_counts[depth]} nodes")
        return path_to_id

    path_to_id = {}
    inserted_count = 0
    skipped_count = 0

    with tqdm(nodes, desc="  categories", unit="cat", leave=False) as bar:
        for slug_path, display_name, parent_path in nodes:
            slug = slug_path.split("/")[-1]
            parent_id = path_to_id.get(parent_path) if parent_path else None
            bar.set_postfix_str(slug_path)

            if parent_id is not None:
                cur.execute(
                    "SELECT id FROM marketplace_categories WHERE slug = %s AND parent_id = %s",
                    (slug, parent_id),
                )
            else:
                cur.execute(
                    "SELECT id FROM marketplace_categories WHERE slug = %s AND parent_id IS NULL",
                    (slug,),
                )

            existing_row = cur.fetchone()
            if existing_row:
                path_to_id[slug_path] = existing_row[0]
                skipped_count += 1
            else:
                cur.execute(
                    """
                    INSERT INTO marketplace_categories (name, slug, parent_id)
                    VALUES (%s, %s, %s)
                    RETURNING id
                    """,
                    (display_name, slug, parent_id),
                )
                path_to_id[slug_path] = cur.fetchone()[0]
                inserted_count += 1

            bar.update(1)

    tqdm.write(f"  [categories] {skipped_count} already existed, {inserted_count} inserted")
    return path_to_id


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — marketplace_item_categories
# ──────────────────────────────────────────────────────────────────────────────


def import_item_categories(
    cur: Any,
    matched_rows: list[dict],
    path_to_id: dict[str, str],
) -> None:
    """
    Insert item-to-category relationships from matched_deduped rows.
    ON CONFLICT DO NOTHING silently skips existing active pairs.
    """
    if DRY_RUN:
        missing = [r for r in matched_rows if r.get("leaf_path") not in path_to_id]
        print(f"  [item_categories] would insert up to {len(matched_rows)} rows")
        if missing:
            sample = [r.get("leaf_path") for r in missing[:5]]
            print(
                f"  WARNING: {len(missing)} rows have unknown leaf_path (would skip). Sample: {sample}",
                file=sys.stderr,
            )
        return

    inserted_count = 0
    skipped_count = 0
    missing_paths: list[str] = []

    with tqdm(matched_rows, desc="  item_categories", unit="row", leave=False) as bar:
        for row in matched_rows:
            item_id = row.get("id")
            cat_path = row.get("leaf_path") or ""
            cat_id = path_to_id.get(cat_path)

            if not cat_id:
                missing_paths.append(f"{item_id} → {cat_path}")
                bar.update(1)
                continue

            cur.execute(
                """
                INSERT INTO marketplace_item_categories (item_id, category_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
                """,
                (item_id, cat_id),
            )
            if cur.rowcount > 0:
                inserted_count += 1
            else:
                skipped_count += 1

            bar.set_postfix(inserted=inserted_count, skipped=skipped_count)
            bar.update(1)

    if missing_paths:
        tqdm.write(f"  WARNING: {len(missing_paths)} rows had unknown leaf_path (skipped):")
        for m in missing_paths[:20]:
            tqdm.write(f"    {m}")
        if len(missing_paths) > 20:
            tqdm.write(f"    … and {len(missing_paths) - 20} more")

    tqdm.write(f"  [item_categories] {skipped_count} already existed, {inserted_count} inserted")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — marketplace_attribute_units
# ──────────────────────────────────────────────────────────────────────────────


def import_units(cur: Any, units: dict[str, dict]) -> dict[str, str]:
    """
    Insert attribute units from proposed_attributes.json.
    Skips units that already exist (matched by symbol).
    Returns symbol → id map.
    """
    if DRY_RUN:
        print(f"  [units] would ensure {len(units)} units are present")
        return {sym: f"<dry:{sym}>" for sym in units}

    symbol_to_id = {}
    inserted_count = 0
    skipped_count = 0

    with tqdm(units.items(), desc="  units", unit="unit", leave=False) as bar:
        for symbol, unit_data in bar:
            bar.set_postfix_str(symbol)

            cur.execute(
                "SELECT id FROM marketplace_attribute_units WHERE symbol = %s AND deleted_at IS NULL",
                (symbol,),
            )
            existing_row = cur.fetchone()
            if existing_row:
                symbol_to_id[symbol] = existing_row[0]
                skipped_count += 1
            else:
                cur.execute(
                    """
                    INSERT INTO marketplace_attribute_units (symbol, name, description, value_type)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        symbol,
                        unit_data.get("name", symbol),
                        unit_data.get("description", ""),
                        unit_data.get("value_type", "number"),
                    ),
                )
                symbol_to_id[symbol] = cur.fetchone()[0]
                inserted_count += 1

    tqdm.write(f"  [units] {skipped_count} already existed, {inserted_count} inserted")
    return symbol_to_id


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4 — marketplace_attributes
# ──────────────────────────────────────────────────────────────────────────────


def import_attributes(
    cur: Any,
    category_attributes: dict[str, list[dict]],
    path_to_id: dict[str, str],
    symbol_to_id: dict[str, str],
) -> None:
    """
    Insert category attributes from proposed_attributes.json.
    Skips attributes that already exist (matched by category_id + key).
    """
    total_attrs = sum(len(attrs) for attrs in category_attributes.values())

    if DRY_RUN:
        missing_cats = [p for p in category_attributes if p not in path_to_id]
        print(f"  [attributes] would insert up to {total_attrs} attributes across {len(category_attributes)} categories")
        if missing_cats:
            print(f"  WARNING: {len(missing_cats)} category paths not in DB (would skip). Sample: {missing_cats[:5]}")
        return

    inserted_count = 0
    skipped_count = 0
    missing_cats: list[str] = []
    missing_units: list[str] = []

    all_attrs = [
        (cat_path, attr)
        for cat_path, attrs in category_attributes.items()
        for attr in attrs
    ]

    with tqdm(all_attrs, desc="  attributes", unit="attr", leave=False) as bar:
        for cat_path, attr in bar:
            cat_id = path_to_id.get(cat_path)
            if not cat_id:
                if cat_path not in missing_cats:
                    missing_cats.append(cat_path)
                bar.update(1)
                continue

            key = attr.get("key", "")
            label = attr.get("label", "")
            description = attr.get("description", "")
            unit_required = bool(attr.get("unit_required", False))
            unit_symbol = attr.get("unit")
            unit_id = None

            if unit_required and unit_symbol:
                unit_id = symbol_to_id.get(unit_symbol)
                if not unit_id:
                    missing_units.append(f"{cat_path}/{key} → {unit_symbol}")
                    bar.update(1)
                    continue

            bar.set_postfix_str(f"{cat_path}/{key}")

            cur.execute(
                "SELECT id FROM marketplace_attributes WHERE category_id = %s AND key = %s AND deleted_at IS NULL",
                (cat_id, key),
            )
            existing_row = cur.fetchone()
            if existing_row:
                skipped_count += 1
            else:
                cur.execute(
                    """
                    INSERT INTO marketplace_attributes (category_id, key, label, description, unit_required)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (cat_id, key, label, description, unit_required),
                )
                inserted_count += 1

            bar.update(1)

    if missing_cats:
        tqdm.write(f"  WARNING: {len(missing_cats)} category paths not found in DB (skipped):")
        for m in missing_cats[:10]:
            tqdm.write(f"    {m}")
        if len(missing_cats) > 10:
            tqdm.write(f"    … and {len(missing_cats) - 10} more")

    if missing_units:
        tqdm.write(f"  WARNING: {len(missing_units)} attributes reference unknown units (skipped):")
        for m in missing_units[:10]:
            tqdm.write(f"    {m}")
        if len(missing_units) > 10:
            tqdm.write(f"    … and {len(missing_units) - 10} more")

    tqdm.write(f"  [attributes] {skipped_count} already existed, {inserted_count} inserted")


# ──────────────────────────────────────────────────────────────────────────────
# Push orchestrators — called by 5_upload_to_db.py
# ──────────────────────────────────────────────────────────────────────────────


def push_taxonomy(conn, taxonomy_path: Path | None = None) -> None:
    """
    Push the full category tree to DB.
    taxonomy_path: path to categories_v1.json (defaults to source-files/categories_v1.json).
    """
    tax_path = taxonomy_path or TAXONOMY_PATH
    categories = json.loads(tax_path.read_text(encoding="utf-8"))
    nodes = build_category_nodes(categories)

    conn.autocommit = False
    cur = conn.cursor()
    try:
        tqdm.write("  Pushing categories…")
        import_categories(cur, nodes)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def push_item_relationships(
    conn,
    matched_deduped_path: Path,
    taxonomy_path: Path | None = None,
) -> None:
    """
    Push item-category relationships from a step-4 matched_deduped.json run.
    Rebuilds path_to_id by syncing (not inserting) categories against the DB,
    pre-filters to item IDs that exist in the items table, then inserts rows.
    """
    tax_path = taxonomy_path or TAXONOMY_PATH
    categories = json.loads(tax_path.read_text(encoding="utf-8"))
    nodes = build_category_nodes(categories)
    matched_rows = load_matched_deduped(matched_deduped_path)

    conn.autocommit = False
    cur = conn.cursor()
    try:
        tqdm.write("  Building category map from DB…")
        path_to_id = import_categories(cur, nodes)
        conn.commit()

        # Pre-filter to item IDs that exist in the items table
        all_item_ids = {r.get("id") for r in matched_rows if r.get("id")}
        if all_item_ids:
            cur.execute(
                "SELECT id FROM items WHERE id = ANY(%s) AND deleted_at IS NULL",
                (list(all_item_ids),),
            )
            existing_item_ids = {row[0] for row in cur.fetchall()}
            missing_item_ids = all_item_ids - existing_item_ids
            if missing_item_ids:
                tqdm.write(
                    f"  WARNING: {len(missing_item_ids)} item ID(s) not found in items table — skipping those rows."
                )
            matched_rows = [r for r in matched_rows if r.get("id") in existing_item_ids]

        tqdm.write("  Pushing item-category relationships…")
        import_item_categories(cur, matched_rows, path_to_id)
        conn.commit()

    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def push_attributes(
    conn,
    attributes_path: Path | None = None,
    taxonomy_path: Path | None = None,
) -> None:
    """
    Push attribute units and category attributes from proposed_attributes.json.
    Syncs categories first to build path_to_id map, then inserts units and attributes.
    
    Args:
        conn: Database connection
        attributes_path: Path to proposed_attributes.json (defaults to source-files/proposed-attributes.json)
        taxonomy_path: Path to categories_v1.json (defaults to source-files/categories_v1.json)
    """
    tax_path = taxonomy_path or TAXONOMY_PATH
    attr_path = attributes_path or ATTRIBUTES_PATH
    
    categories = json.loads(tax_path.read_text(encoding="utf-8"))
    nodes = build_category_nodes(categories)

    attr_data = json.loads(attr_path.read_text(encoding="utf-8"))
    units = attr_data.get("units", {})
    category_attributes = attr_data.get("_category_attributes", {})

    conn.autocommit = False
    cur = conn.cursor()
    try:
        tqdm.write("  Building category map from DB…")
        path_to_id = import_categories(cur, nodes)
        conn.commit()

        tqdm.write("  Pushing attribute units…")
        symbol_to_id = import_units(cur, units)
        conn.commit()

        tqdm.write("  Pushing category attributes…")
        import_attributes(cur, category_attributes, path_to_id, symbol_to_id)
        conn.commit()

    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


# ──────────────────────────────────────────────────────────────────────────────
# Pull functions — called by 5_upload_to_db.py (download / inspect path)
# ──────────────────────────────────────────────────────────────────────────────


def pull_all(conn, out_dir: Path) -> dict[str, int]:
    """
    Pull all marketplace tables + items snapshot from DB into out_dir.
    Returns dict of table_name → row_count.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cur = conn.cursor()
    counts: dict[str, int] = {}

    # items
    cur.execute("""
        SELECT DISTINCT ON (i.id)
            i.id           AS id,
            i.title        AS title,
            i.description  AS description,
            ''             AS subtitle,
            s.name         AS store_name,
            i.category,
            i.subcategory
        FROM items i
        JOIN store_items si ON si.items_id = i.id
        JOIN stores s ON s.id = si.store_id
        WHERE i.deleted_at IS NULL
        ORDER BY i.id, s.name
    """)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    _write_csv(out_dir / "items.csv", cols, rows)
    counts["items"] = len(rows)

    # marketplace_categories
    cur.execute("""
        SELECT id, parent_id, name, slug, created_at
        FROM marketplace_categories
        WHERE deleted_at IS NULL
        ORDER BY id
    """)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    _write_json(out_dir / "marketplace_categories.json", cols, rows)
    counts["marketplace_categories"] = len(rows)

    # marketplace_attribute_units
    cur.execute("""
        SELECT id, symbol, name, description, value_type
        FROM marketplace_attribute_units
        WHERE deleted_at IS NULL
        ORDER BY symbol
    """)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    _write_json(out_dir / "marketplace_attribute_units.json", cols, rows)
    counts["marketplace_attribute_units"] = len(rows)

    # marketplace_attributes
    cur.execute("""
        SELECT id, category_id, key, label, description, unit_required
        FROM marketplace_attributes
        WHERE deleted_at IS NULL
        ORDER BY category_id, key
    """)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    _write_json(out_dir / "marketplace_attributes.json", cols, rows)
    counts["marketplace_attributes"] = len(rows)

    # marketplace_item_categories
    cur.execute("""
        SELECT id, item_id, category_id, created_at
        FROM marketplace_item_categories
        WHERE deleted_at IS NULL
        ORDER BY item_id
    """)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    _write_csv(out_dir / "marketplace_item_categories.csv", cols, rows)
    counts["marketplace_item_categories"] = len(rows)

    # marketplace_attribute_values
    cur.execute("""
        SELECT id, item_id, attribute_id, attribute_unit_id, value
        FROM marketplace_attribute_values
        WHERE deleted_at IS NULL
        ORDER BY item_id
    """)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    _write_csv(out_dir / "marketplace_attribute_values.csv", cols, rows)
    counts["marketplace_attribute_values"] = len(rows)

    cur.close()
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Gap analysis helpers
# ──────────────────────────────────────────────────────────────────────────────


def get_gap_counts(conn) -> dict[str, int]:
    """
    Query DB for uncategorized and missing-attribute item counts.
    Returns {"total": N, "uncategorized": X, "missing_attrs": Y}
    """
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM items WHERE deleted_at IS NULL")
    total = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM items i
        WHERE i.deleted_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM marketplace_item_categories mic
              WHERE mic.item_id = i.id AND mic.deleted_at IS NULL
          )
    """)
    uncategorized = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(DISTINCT mic.item_id)
        FROM marketplace_item_categories mic
        WHERE mic.deleted_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM marketplace_attribute_values mav
              WHERE mav.item_id = mic.item_id AND mav.deleted_at IS NULL
          )
    """)
    missing_attrs = cur.fetchone()[0]

    cur.close()
    return {"total": total, "uncategorized": uncategorized, "missing_attrs": missing_attrs}


def get_taxonomy_gaps(conn, taxonomy_path: Path | None = None) -> dict:
    """
    Compare local categories_v1.json against DB.
    Returns counts and sample lists of missing categories.
    """
    tax_path = taxonomy_path or TAXONOMY_PATH
    categories = json.loads(tax_path.read_text(encoding="utf-8"))
    nodes = build_category_nodes(categories)

    cur = conn.cursor()

    local_slugs = {n[0].split("/")[-1] for n in nodes}
    cur.execute("SELECT slug FROM marketplace_categories WHERE deleted_at IS NULL")
    db_slugs = {row[0] for row in cur.fetchall() if row[0]}
    missing_cat_slugs = sorted(local_slugs - db_slugs)

    cur.close()
    return {
        "missing_categories": len(missing_cat_slugs),
        "_missing_cat_slugs": missing_cat_slugs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# File write helpers
# ──────────────────────────────────────────────────────────────────────────────


def _write_csv(path: Path, cols: list[str], rows: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)


def _write_json(path: Path, cols: list[str], rows: list) -> None:
    data = [dict(zip(cols, row)) for row in rows]
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
