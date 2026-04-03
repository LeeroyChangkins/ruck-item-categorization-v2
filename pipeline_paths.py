"""
Shared paths for pipeline step output directories (v2 layout: step-1 … step-4, each with outputs/).

Step 1 (similar-title groups) writes under step-1/outputs/ (timestamped run subfolders).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def step1_output_roots() -> list[Path]:
    return [ROOT / "step-1" / "outputs"]


def glob_step1_outputs(glob_pat: str) -> list[Path]:
    """Glob under step-1/outputs/ (e.g. '**/unmatched_similar_title_groups.json')."""
    out: list[Path] = []
    for base in step1_output_roots():
        if base.is_dir():
            out.extend(base.glob(glob_pat))
    return out


def newest_under_step1(glob_pat: str) -> Path | None:
    """Newest env-matching file under step-1/outputs/, by mtime."""
    import shared_utils as _su
    cands = glob_step1_outputs(glob_pat)
    if not cands:
        return None
    # Step-1 timestamped run dirs carry the env suffix, so check the parent dir
    return _su.latest_env_path(cands, name_attr="parent")
