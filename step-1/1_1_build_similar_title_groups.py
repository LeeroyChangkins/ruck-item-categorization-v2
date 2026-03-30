#!/usr/bin/env python3
"""
Step 1.1 — Build similar-title groups (aggregate)
Cluster items whose titles are pairwise similar (default: min ratio 0.9).

- Input: source-files/raw-prod-items-non-deleted.json (default) or --input (JSON array of items).
- Blocking: only compare titles in the same bucket (first letter-token + length bucket).
- Clustering: clique — every pair in a group has similarity >= threshold. Within each block,
  repeatedly take the largest maximal clique in the remaining induced subgraph until no clique
  reaches min_group_size.
- Output: step-1/outputs/<run_id>/unmatched_similar_title_groups.json

Requires: networkx (maximal clique enumeration).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ITEMS = ROOT / "source-files" / "raw-prod-items-non-deleted.json"
OUTDIR = Path(__file__).resolve().parent / "outputs"

VERSION = "1.6-unmatched-similar-title-groups"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_title(s: str) -> str:
    t = (s or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def first_token(norm: str) -> str:
    parts = re.findall(r"[a-z0-9]+", norm)
    return parts[0] if parts else "__empty__"


def blocking_key(norm: str) -> Tuple[str, int]:
    """(first token, length bucket) — only titles in the same bucket are compared."""
    ft = first_token(norm)
    lb = len(norm) // 8 if norm else 0
    return (ft, lb)


def sim_ratio(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def master_title_for_group(titles: Sequence[str]) -> str:
    """Readable label: longest common prefix of normalized titles, else shortest original."""
    norms = [normalize_title(t) for t in titles if isinstance(t, str)]
    if not norms:
        return ""
    if len(norms) == 1:
        return titles[0] if titles else norms[0]
    m = min(len(s) for s in norms)
    i = 0
    while i < m and len({s[i] for s in norms}) == 1:
        i += 1
    lcp = norms[0][:i].strip()
    if len(lcp) >= 4:
        return lcp
    shortest = min((t for t in titles if isinstance(t, str)), key=len, default="")
    return shortest.strip() or lcp



def maximal_cliques_networkx(adj: Dict[int, Set[int]]) -> List[Set[int]]:
    import networkx as nx

    verts = list(adj.keys())
    G = nx.Graph()
    G.add_nodes_from(verts)
    for u in verts:
        for v in adj.get(u, ()):
            if u < v:
                G.add_edge(u, v)
    return [set(c) for c in nx.find_cliques(G)]


def induced_subgraph(adj: Dict[int, Set[int]], remaining: Set[int]) -> Dict[int, Set[int]]:
    return {u: (adj[u] & remaining) for u in remaining}


def iterative_clique_cover_on_bucket(
    sub_adj: Dict[int, Set[int]],
    min_group_size: int,
    *,
    status: Callable[[str], None] | None = None,
) -> List[Set[int]]:
    """
    Repeatedly take the largest maximal clique in the remaining induced subgraph until
    no clique has size >= min_group_size.
    """
    remaining: Set[int] = set(sub_adj.keys())
    groups: List[Set[int]] = []
    if status:
        status("tight groups: starting…")
    while len(remaining) >= min_group_size:
        rnd = len(groups) + 1
        if status:
            status(f"tight groups: round {rnd} · {len(remaining)} titles left…")
        sub = induced_subgraph(sub_adj, remaining)
        if status:
            status(f"tight groups: analyzing overlaps ({len(remaining)} titles — can take a bit)…")
        max_cliques = maximal_cliques_networkx(sub)
        candidates = [set(c) for c in max_cliques if len(c) >= min_group_size]
        if not candidates:
            break
        best = max(candidates, key=len)
        groups.append(best)
        remaining -= best
    if status:
        status(f"tight groups: done · {len(groups)} group(s) in this bucket")
    return groups


def run_aggregate(
    items: List[dict],
    min_sim: float,
    min_group_size: int,
    *,
    show_progress: bool = True,
) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
    """
    Returns (group_payloads, ungrouped_items, stats_extra).
    """
    # index items by position in list for stable ids
    norms: List[str] = []
    valid: List[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        iid = it.get("id")
        if not isinstance(iid, str) or not iid:
            continue
        title = it.get("title") or ""
        if not isinstance(title, str):
            title = str(title)
        norms.append(normalize_title(title))
        valid.append(it)

    n = len(valid)
    buckets: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for i in range(n):
        buckets[blocking_key(norms[i])].append(i)

    group_id_counter = 0
    out_groups: List[dict] = []
    used_global: Set[int] = set()
    items_analyzed = 0  # cumulative items in fully-processed buckets

    bucket_list = sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1]))
    use_bk_pbar = show_progress and sys.stderr.isatty() and len(bucket_list) > 0
    bk_pbar: tqdm | None = None
    if use_bk_pbar:
        bk_pbar = tqdm(
            bucket_list,
            total=len(bucket_list),
            desc="1.1 similar-title buckets",
            unit="bucket",
            file=sys.stderr,
            dynamic_ncols=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} buckets [{elapsed}<{remaining}, {rate_fmt}]{postfix}",
        )

    def _set_bucket_status(msg: str) -> None:
        """Update postfix; always prefixes current items_analyzed/n so the counter is always visible."""
        if bk_pbar:
            bk_pbar.set_postfix_str(f"items {items_analyzed}/{n}  {msg}", refresh=True)

    # Background thread: force a refresh every second so the elapsed timer keeps ticking
    # even when the main thread is blocked inside nx.find_cliques().
    _stop_refresh = threading.Event()

    def _auto_refresh() -> None:
        while not _stop_refresh.is_set():
            if bk_pbar:
                bk_pbar.refresh()
            time.sleep(1.0)

    _refresh_thread: threading.Thread | None = None
    if use_bk_pbar:
        _refresh_thread = threading.Thread(target=_auto_refresh, daemon=True)
        _refresh_thread.start()

    try:
        for _bk, idxs in bk_pbar or bucket_list:
            if len(idxs) < min_group_size:
                items_analyzed += len(idxs)
                if bk_pbar:
                    bk_pbar.update(1)
                continue
            k = len(idxs)
            total_pairs = k * (k - 1) // 2
            _set_bucket_status(f"comparing {k} titles · 0/{total_pairs} pairs")
            sub_adj: Dict[int, Set[int]] = {i: set() for i in idxs}
            pair_done = 0
            update_every = max(1, total_pairs // 80) if total_pairs else 1
            next_report = update_every
            for i, ii in enumerate(idxs):
                for j in range(i + 1, len(idxs)):
                    ia, ib = idxs[i], idxs[j]
                    if sim_ratio(norms[ia], norms[ib]) >= min_sim:
                        sub_adj[ia].add(ib)
                        sub_adj[ib].add(ia)
                    pair_done += 1
                    if bk_pbar and pair_done >= next_report:
                        pct = 100.0 * pair_done / total_pairs if total_pairs else 100.0
                        _set_bucket_status(
                            f"comparing {k} titles · {pair_done}/{total_pairs} pairs ({pct:.0f}%)"
                        )
                        next_report += update_every

            groups = iterative_clique_cover_on_bucket(
                sub_adj, min_group_size, status=_set_bucket_status if bk_pbar else None
            )
            for g in groups:
                g_items = [valid[i] for i in sorted(g)]
                titles = [str(it.get("title") or "") for it in g_items]
                gid = f"g_{group_id_counter:06d}"
                group_id_counter += 1
                out_groups.append(
                    {
                        "group_id": gid,
                        "master_title": master_title_for_group(titles),
                        "item_count": len(g_items),
                        "items": [
                            {
                                "id": it.get("id"),
                                "title": it.get("title") or "",
                                "subtitle": it.get("subtitle") or "",
                            }
                            for it in g_items
                        ],
                    }
                )
                used_global |= g

            items_analyzed += len(idxs)
            if bk_pbar:
                bk_pbar.set_postfix_str(
                    f"items {items_analyzed}/{n}  groups={len(out_groups)}  cliques={len(groups)}",
                    refresh=False,
                )
                bk_pbar.update(1)
    finally:
        if _refresh_thread:
            _stop_refresh.set()
            _refresh_thread.join(timeout=2)
        if bk_pbar:
            bk_pbar.close()

    ungrouped: List[dict] = []
    for i in range(n):
        if i not in used_global:
            ungrouped.append(valid[i])

    stats = {
        "items_in": n,
        "unmatched_items_in": n,
        "groups_emitted": len(out_groups),
        "items_in_groups": sum(g["item_count"] for g in out_groups),
        "items_ungrouped": len(ungrouped),
        "min_similarity": min_sim,
        "min_group_size": min_group_size,
        "blocking": "first_alpha_numeric_token + floor(len(normalized_title)/8)",
        "clustering": "clique (iterative largest maximal clique on remaining induced subgraph)",
    }
    return out_groups, ungrouped, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build similar-title groups from production items (step 1.1).")
    parser.add_argument(
        "--input",
        metavar="PATH",
        help=f"Path to JSON array of items (default: {DEFAULT_ITEMS.name}).",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.9,
        metavar="R",
        help="Minimum pairwise title similarity ratio [0,1] (default: 0.9).",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=5,
        metavar="N",
        help="Emit only groups with at least N items (default: 5).",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve() if args.input else DEFAULT_ITEMS
    if not in_path.is_file():
        raise SystemExit(f"Input not found: {in_path}")

    data = json.loads(in_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        raw = data
    else:
        raw = data.get("unmatched_items")
    if not isinstance(raw, list):
        raise SystemExit("Expected a JSON array of items, or an object with unmatched_items array.")

    min_sim = float(args.min_similarity)
    if not 0.0 < min_sim <= 1.0:
        raise SystemExit("--min-similarity must be in (0, 1].")
    min_gs = max(2, int(args.min_group_size))

    groups, ungrouped, stats = run_aggregate(raw, min_sim, min_gs, show_progress=not args.no_progress)

    ts = timestamp()
    run_dir = OUTDIR / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    out_path = run_dir / "unmatched_similar_title_groups.json"

    payload = {
        "version": VERSION,
        "run_id": ts,
        "output_dir": str(run_dir.resolve()),
        "source_items_catalog": str(in_path.resolve()),
        "source_unmatched_deduped": None,
        "similarity_metric": "difflib.SequenceMatcher.ratio on normalized titles (lowercase, collapsed whitespace)",
        "stats": stats,
        "groups": groups,
        "ungrouped_items": [
            {
                "id": it.get("id"),
                "title": it.get("title") or "",
                "subtitle": it.get("subtitle") or "",
            }
            for it in ungrouped
            if isinstance(it, dict)
        ],
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path.relative_to(ROOT)}")
    print(
        f"Groups: {stats['groups_emitted']}  Items in groups: {stats['items_in_groups']}  "
        f"Ungrouped: {stats['items_ungrouped']}"
    )


if __name__ == "__main__":
    main()
