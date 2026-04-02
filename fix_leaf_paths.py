#!/usr/bin/env python3
"""
Fix partial leaf paths in matched_deduped.json by resolving slugs to full paths.
"""

import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from taxonomy_cascade import collect_slug_to_path

def fix_matched_deduped(file_path: Path) -> None:
    """Fix partial leaf paths in matched_deduped.json."""
    print(f"Loading {file_path}...")
    data = json.loads(file_path.read_text(encoding="utf-8"))
    
    # Load taxonomy for slug resolution
    taxonomy_path = Path("source-files/categories_v1.json")
    if not taxonomy_path.exists():
        print(f"ERROR: {taxonomy_path} not found")
        return
    
    categories = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    slug_to_path = collect_slug_to_path(categories)
    
    items = data.get("matched_items", [])
    if not items:
        print("No matched_items found")
        return
    
    fixed_count = 0
    total_count = len(items)
    
    for item in items:
        leaf_path = item.get("leaf_path", "")
        if "/" not in leaf_path:  # Partial path detected
            full_path = slug_to_path.get(leaf_path, leaf_path)
            if full_path != leaf_path:
                item["leaf_path"] = full_path
                fixed_count += 1
                # Also update leaf_slug if needed
                item["leaf_slug"] = leaf_path.split("/")[-1] if "/" in full_path else leaf_path
    
    print(f"Fixed {fixed_count} out of {total_count} items")
    
    # Write fixed file
    backup_path = file_path.with_suffix(".json.backup")
    file_path.rename(backup_path)
    file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Original backed up to {backup_path}")
    print(f"Fixed file written to {file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 fix_leaf_paths.py <matched_deduped.json>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"ERROR: {file_path} not found")
        sys.exit(1)
    
    fix_matched_deduped(file_path)
