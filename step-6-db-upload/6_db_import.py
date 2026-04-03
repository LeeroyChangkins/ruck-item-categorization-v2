#!/usr/bin/env python3
"""
6_db_import.py

DB push and pull module for v2 marketplace categorization data.
Designed to be imported by 6_upload_to_db.py; contains no interactive prompts.

v2 differences from v1:
  - Taxonomy loaded from source-files/categories_v1.json (recursive tree, not tier1/tier2/tier3 keyed JSON)
  - Item mappings loaded from step-4/outputs/<run_id>/matched_deduped.json (not CSVs)
  - Attribute upload driven by step-5 proposed_attributes.json + item_attribute_values.json

Push tables (in dependency order):
  1. marketplace_categories
  2. marketplace_item_categories
  3. marketplace_attributes  (from proposed_attributes.json)
  4. marketplace_attribute_units  (from proposed_attributes.json)
  5. marketplace_attribute_values  (from item_attribute_values.json)

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
STEP4_OUTPUTS = ROOT / "step-4-dedupe-and-merge-matched-items" / "outputs"

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


_BATCH_SIZE = 1000


def import_item_categories(
    cur: Any,
    matched_rows: list[dict],
    path_to_id: dict[str, str],
) -> None:
    """
    Insert item-to-category relationships from matched_deduped rows.
    Uses execute_values for bulk insertion; ON CONFLICT DO NOTHING skips duplicates.
    """
    from psycopg2.extras import execute_values

    missing_paths: list[str] = []
    valid_pairs: list[tuple] = []

    for row in matched_rows:
        item_id  = row.get("id")
        cat_path = row.get("leaf_path") or ""
        cat_id   = path_to_id.get(cat_path)
        if not cat_id:
            missing_paths.append(f"{item_id} → {cat_path}")
        else:
            valid_pairs.append((item_id, cat_id))

    if missing_paths:
        tqdm.write(f"  WARNING: {len(missing_paths)} rows had unknown leaf_path (skipped):")
        for m in missing_paths[:20]:
            tqdm.write(f"    {m}")
        if len(missing_paths) > 20:
            tqdm.write(f"    … and {len(missing_paths) - 20} more")

    if DRY_RUN:
        print(f"  [item_categories] would insert up to {len(valid_pairs)} rows")
        return

    inserted_count = 0
    with tqdm(total=len(valid_pairs), desc="  item_categories", unit="row", leave=False) as bar:
        for i in range(0, len(valid_pairs), _BATCH_SIZE):
            chunk = valid_pairs[i : i + _BATCH_SIZE]
            execute_values(
                cur,
                """
                INSERT INTO marketplace_item_categories (item_id, category_id)
                VALUES %s
                ON CONFLICT DO NOTHING
                """,
                chunk,
            )
            inserted_count += cur.rowcount
            bar.update(len(chunk))
            bar.set_postfix(inserted=inserted_count)

    skipped_count = len(valid_pairs) - inserted_count
    tqdm.write(f"  [item_categories] {skipped_count} already existed, {inserted_count} inserted")


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


# ──────────────────────────────────────────────────────────────────────────────
# Attribute push helpers
# ──────────────────────────────────────────────────────────────────────────────


def _find_latest_step5_file(filename: str) -> Path | None:
    """Find the most recent env-matching file under step-5/outputs/.
    filename may include a glob wildcard, e.g. 'proposed_attributes*.json'.
    """
    import shared_utils as _su
    candidates = list((ROOT / "step-5-attribute-generation-and-unit-value-assignment" / "outputs").rglob(filename))
    if not candidates:
        return None
    return _su.latest_env_path(candidates, name_attr="parent")


def import_attributes_and_units(
    cur: Any,
    proposed_attrs: dict,
) -> dict[str, dict[str, str]]:
    """
    Insert marketplace_attributes and marketplace_attribute_units from
    proposed_attributes.json (the `_category_attributes` and `units` keys).

    Returns: { leaf_path: { attribute_key: attribute_db_id } }
    """
    from psycopg2.extras import execute_values

    units_data: dict[str, dict] = proposed_attrs.get("units", {})
    category_attrs: dict[str, list[dict]] = proposed_attrs.get("_category_attributes", {})

    # ── 1. batch upsert units then fetch all IDs in one query ────────────────
    unit_symbol_to_id: dict[str, str] = {}

    if DRY_RUN:
        tqdm.write(f"  [dry-run] would upsert {len(units_data)} attribute units")
    elif units_data:
        unit_rows = [
            (sym, u.get("name", sym), u.get("description", ""), u.get("value_type", "number"))
            for sym, u in units_data.items()
        ]
        execute_values(
            cur,
            """
            INSERT INTO marketplace_attribute_units (symbol, name, description, value_type)
            VALUES %s
            ON CONFLICT (symbol) DO NOTHING
            """,
            unit_rows,
        )
        # Fetch IDs for all symbols (inserted + pre-existing)
        symbols = list(units_data.keys())
        cur.execute(
            "SELECT id, symbol FROM marketplace_attribute_units WHERE symbol = ANY(%s)",
            (symbols,),
        )
        for row in cur.fetchall():
            unit_symbol_to_id[row[1]] = str(row[0])

    # ── 2. resolve category slug_path → category DB ID ───────────────────────
    cur.execute(
        "SELECT id, slug FROM marketplace_categories WHERE deleted_at IS NULL"
    )
    slug_to_cat_id = {row[1]: str(row[0]) for row in cur.fetchall()}

    # ── 3. collect all attribute rows, resolve cat_id & unit_id in Python ────
    # Each entry: (cat_id, key, label, desc, vtype, unit_id, leaf_path)
    attr_tuples: list[tuple] = []
    leaf_path_key_to_meta: dict[tuple, dict] = {}  # (leaf_path, key) → attr meta

    for leaf_path, attrs in category_attrs.items():
        leaf_slug = leaf_path.split("/")[-1] if leaf_path else ""
        cat_id = slug_to_cat_id.get(leaf_slug)
        if not cat_id:
            for slug, cid in slug_to_cat_id.items():
                if leaf_path.endswith("/" + slug) or leaf_path == slug:
                    cat_id = cid
                    break
        if not cat_id:
            tqdm.write(f"  WARNING: category not found in DB for path {leaf_path!r} — skipping attributes")
            continue

        for attr in attrs:
            key      = attr.get("key", "")
            label    = attr.get("label", key)
            desc     = attr.get("description", "")
            vtype    = attr.get("value_type", "text")
            unit_sym = attr.get("unit")
            unit_id  = unit_symbol_to_id.get(unit_sym) if unit_sym else None
            attr_tuples.append((cat_id, key, label, desc, vtype, unit_id))
            leaf_path_key_to_meta[(leaf_path, key)] = {"cat_id": cat_id}

    if DRY_RUN:
        tqdm.write(f"  [dry-run] would insert attributes for {len(category_attrs)} categories ({len(attr_tuples)} attrs)")
        return {}

    # ── 4. batch INSERT all attributes (ON CONFLICT DO NOTHING) ──────────────
    total_inserted = 0
    with tqdm(total=len(attr_tuples), desc="  attributes", unit="attr", leave=False) as bar:
        for i in range(0, len(attr_tuples), _BATCH_SIZE):
            chunk = attr_tuples[i : i + _BATCH_SIZE]
            execute_values(
                cur,
                """
                INSERT INTO marketplace_attributes
                    (category_id, key, label, description, value_type, unit_id)
                VALUES %s
                ON CONFLICT (category_id, key) DO NOTHING
                """,
                chunk,
            )
            total_inserted += cur.rowcount
            bar.update(len(chunk))

    total_skipped = len(attr_tuples) - total_inserted
    tqdm.write(f"  [attributes] {total_inserted} inserted, {total_skipped} already existed")

    # ── 5. batch SELECT all IDs to build path_key_to_attr_id ─────────────────
    # Fetch by (category_id, key) pairs using ANY on cat_ids
    all_cat_ids = list({m["cat_id"] for m in leaf_path_key_to_meta.values()})
    cur.execute(
        """
        SELECT id, category_id, key
        FROM marketplace_attributes
        WHERE category_id = ANY(%s)
        """,
        (all_cat_ids,),
    )
    cat_key_to_attr_id: dict[tuple, str] = {
        (str(r[1]), r[2]): str(r[0]) for r in cur.fetchall()
    }

    path_key_to_attr_id: dict[str, dict[str, str]] = {}
    for (leaf_path, key), meta in leaf_path_key_to_meta.items():
        attr_id = cat_key_to_attr_id.get((meta["cat_id"], key))
        if attr_id:
            path_key_to_attr_id.setdefault(leaf_path, {})[key] = attr_id

    return path_key_to_attr_id


def import_attribute_values(
    cur: Any,
    values: list[dict],
    path_key_to_attr_id: dict[str, dict[str, str]],
    unit_symbol_to_id: dict[str, str],
) -> None:
    """
    Insert rows from item_attribute_values.json into marketplace_attribute_values.
    Uses execute_values for bulk insertion; ON CONFLICT DO NOTHING skips duplicates.
    """
    from psycopg2.extras import execute_values

    # Resolve all foreign keys in Python first; collect valid tuples
    valid_tuples: list[tuple] = []
    missing = 0

    for row in values:
        item_id   = row.get("item_id", "")
        leaf_path = row.get("leaf_path", "")
        attr_key  = row.get("attribute_key", "")
        value     = row.get("value", "")
        unit_sym  = row.get("unit")

        attr_id = (path_key_to_attr_id.get(leaf_path) or {}).get(attr_key)
        if not attr_id or not item_id or not value:
            missing += 1
            continue

        unit_id = unit_symbol_to_id.get(unit_sym) if unit_sym else None
        valid_tuples.append((item_id, attr_id, unit_id, str(value)))

    if DRY_RUN:
        tqdm.write(f"  [dry-run] would insert up to {len(valid_tuples)} attribute value rows")
        if missing:
            tqdm.write(f"  [dry-run] {missing} rows would be skipped (missing item_id, attr_id, or value)")
        return

    if missing:
        tqdm.write(f"  WARNING: {missing} rows skipped (missing item_id, attr_id, or value)")

    inserted = 0
    with tqdm(total=len(valid_tuples), desc="  attribute_values", unit="row", leave=False) as bar:
        for i in range(0, len(valid_tuples), _BATCH_SIZE):
            chunk = valid_tuples[i : i + _BATCH_SIZE]
            execute_values(
                cur,
                """
                INSERT INTO marketplace_attribute_values
                    (item_id, attribute_id, attribute_unit_id, value)
                VALUES %s
                ON CONFLICT DO NOTHING
                """,
                chunk,
            )
            inserted += cur.rowcount
            bar.update(len(chunk))
            bar.set_postfix(inserted=inserted)

    skipped = len(valid_tuples) - inserted
    tqdm.write(f"  [attribute_values] {skipped} already existed, {inserted} inserted")


def push_attributes(
    conn,
    proposed_attrs_path: Path | None = None,
    values_path: Path | None = None,
) -> None:
    """
    Push attributes (schema + units) and item attribute values to DB.

    proposed_attrs_path: proposed_attributes.json from step 5b
    values_path:         item_attribute_values.json from step 5c
    """
    attrs_file = proposed_attrs_path or _find_latest_step5_file("proposed_attributes*.json")
    if not attrs_file or not attrs_file.exists():
        tqdm.write("  No proposed_attributes.json found — skipping attribute push.")
        return

    tqdm.write(f"  Loading attributes from: {attrs_file.name}")
    proposed_attrs = json.loads(attrs_file.read_text(encoding="utf-8"))

    conn.autocommit = False
    cur = conn.cursor()
    try:
        tqdm.write("  Pushing attribute schema + units…")
        path_key_to_attr_id = import_attributes_and_units(cur, proposed_attrs)
        conn.commit()

        # Reconstruct unit_symbol_to_id for the values pass
        unit_symbol_to_id: dict[str, str] = {}
        if not DRY_RUN:
            cur.execute("SELECT id, symbol FROM marketplace_attribute_units")
            for row in cur.fetchall():
                unit_symbol_to_id[row[1]] = str(row[0])

        val_file = values_path or _find_latest_step5_file("item_attribute_values*.json")
        if val_file and val_file.exists():
            tqdm.write(f"  Loading attribute values from: {val_file.name}")
            values = json.loads(val_file.read_text(encoding="utf-8"))
            tqdm.write(f"  Pushing {len(values)} attribute value rows…")
            import_attribute_values(cur, values, path_key_to_attr_id, unit_symbol_to_id)
            conn.commit()
        else:
            tqdm.write("  No item_attribute_values.json found — skipping value push.")

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
