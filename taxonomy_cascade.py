"""
Shared taxonomy helpers for phased depth-based bigram mapping (T0 → T1 → …).

Used by step-2.1b taxonomy bigrams (multi-depth) and step-2.2 phased matcher (path dedupe).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

ROOT_KEYS = ("materials", "tools_and_gear", "services")

# Leaf slugs used only for interactive catch-all assignment (steps 1.3 / 1.6 [o]), not auto matching.
CATCH_ALL_BUCKET_SLUGS = frozenset({"other", "miscellaneous"})


def is_catch_all_bucket_slug(slug: str | None) -> bool:
    return isinstance(slug, str) and slug in CATCH_ALL_BUCKET_SLUGS


def leaf_path_is_catch_all_bucket(leaf_path: str) -> bool:
    """True if the final path segment is other or miscellaneous."""
    parts = str(leaf_path).strip("/").split("/")
    return bool(parts) and is_catch_all_bucket_slug(parts[-1])


def tokenize_taxonomy_text(s: str) -> List[str]:
    """Letters-only tokens from slug/display_name. No regex."""
    if not s:
        return []
    s = str(s).replace("_", " ").lower()
    out: List[str] = []
    cur: List[str] = []
    for ch in s:
        if ch.isalpha():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def all_descendant_nodes(node: dict) -> Iterable[dict]:
    subs = node.get("subcategories") or []
    for child in subs:
        yield child
        yield from all_descendant_nodes(child)


def add_node_tokens(n: dict, toks: Set[str]) -> None:
    slug = n.get("slug")
    dn = n.get("display_name")
    if slug:
        toks.update(tokenize_taxonomy_text(slug))
    if dn:
        toks.update(tokenize_taxonomy_text(dn))


def build_anchor_token_set(node: dict) -> Set[str]:
    """Tokens from this node and all descendants (slug + display_name)."""
    toks: Set[str] = set()
    add_node_tokens(node, toks)
    for desc in all_descendant_nodes(node):
        add_node_tokens(desc, toks)
    return toks


def nodes_at_depth(categories: dict, depth: int) -> List[Tuple[str, dict]]:
    """
    Nodes at exact depth below JSON roots.
    depth 0: three roots (slug = key: materials, tools_and_gear, services).
    depth 1: direct children under each root (same as legacy T1 list).
    """
    out: List[Tuple[str, dict]] = []
    if depth < 0:
        return out
    for t0 in ROOT_KEYS:
        root = categories.get(t0)
        if not isinstance(root, dict):
            continue
        if depth == 0:
            out.append((t0, root))
            continue
        for child in root.get("subcategories") or []:
            _collect_at_depth_from_node(child, 1, depth, out)
    return out


def _collect_at_depth_from_node(node: dict, current_depth: int, target_depth: int, out: List[Tuple[str, dict]]) -> None:
    if current_depth == target_depth:
        slug = node.get("slug")
        if slug:
            out.append((slug, node))
        return
    for ch in node.get("subcategories") or []:
        _collect_at_depth_from_node(ch, current_depth + 1, target_depth, out)


def max_taxonomy_depth(categories: dict) -> int:
    """Maximum depth of any slug-bearing node (0 = only roots if no children)."""
    best = 0

    def walk(node: dict, d: int) -> None:
        nonlocal best
        if node.get("slug"):
            best = max(best, d)
        for ch in node.get("subcategories") or []:
            walk(ch, d + 1)

    for t0 in ROOT_KEYS:
        root = categories.get(t0)
        if not isinstance(root, dict):
            continue
        best = max(best, 0)
        for child in root.get("subcategories") or []:
            walk(child, 1)
    return best


def collect_slug_to_path(categories: dict) -> Dict[str, str]:
    """
    Map each category slug to one slash path from root key (e.g. materials/fencing/barbed_wire).
    Raises ValueError if the same slug appears at two different paths.
    """
    slug_to_path: Dict[str, str] = {}
    dup: Dict[str, List[str]] = defaultdict(list)

    def walk(node: dict, parts: List[str]) -> None:
        slug = node.get("slug")
        next_parts = parts + ([slug] if slug else [])
        path = "/".join(next_parts)
        if slug and not is_catch_all_bucket_slug(slug):
            if slug in slug_to_path and slug_to_path[slug] != path:
                dup[slug].extend([slug_to_path[slug], path])
            else:
                slug_to_path[slug] = path
        for ch in node.get("subcategories") or []:
            walk(ch, next_parts)

    for t0 in ROOT_KEYS:
        root = categories.get(t0)
        if not isinstance(root, dict):
            continue
        walk(root, [t0])

    if dup:
        import sys
        msg = "; ".join(f"{s}: {paths}" for s, paths in dup.items())
        print(f"WARNING: duplicate taxonomy slugs (keeping first path): {msg}", file=sys.stderr)

    # T0 JSON keys (materials, tools_and_gear, services) are valid anchor slugs at depth 0.
    for t0 in ROOT_KEYS:
        if t0 not in slug_to_path:
            slug_to_path[t0] = t0

    return slug_to_path


def path_is_strict_prefix_of(shorter: str, longer: str) -> bool:
    return longer.startswith(shorter + "/")


def filter_to_maximal_paths(paths: Set[str]) -> Set[str]:
    """Drop any path that is a strict prefix of another path in the set."""
    return {p for p in paths if not any(other != p and path_is_strict_prefix_of(p, other) for other in paths)}


def dedupe_category_slugs(category_slugs: List[str], slug_to_path: Dict[str, str]) -> List[str]:
    """
    Given category slugs for one item, drop slugs whose path is an ancestor of another kept slug.
    Returns slugs sorted by path length descending (most specific first).
    """
    paths_known = [slug_to_path[s] for s in category_slugs if s in slug_to_path]
    unknown = [s for s in category_slugs if s not in slug_to_path]
    maximal = filter_to_maximal_paths(set(paths_known))
    slug_by_path = {slug_to_path[s]: s for s in category_slugs if s in slug_to_path}
    ordered = sorted((slug_by_path[p] for p in maximal), key=lambda s: len(slug_to_path[s]), reverse=True)
    return ordered + unknown
