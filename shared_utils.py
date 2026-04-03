"""
shared_utils.py — Common utilities shared across v2 pipeline scripts.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def env_suffix() -> str:
    """Return '-prod', '-dev', or '' based on the PIPELINE_ENV environment variable.

    The pipeline runner sets PIPELINE_ENV before launching subprocesses so every
    step can append this suffix to its timestamped output folder/file names.
    """
    val = os.environ.get("PIPELINE_ENV", "").strip().lower()
    if val in ("prod", "dev"):
        return f"-{val}"
    return ""


def timestamp() -> str:
    """Return a sortable timestamp string suffixed with the current pipeline environment.

    Examples:
      PIPELINE_ENV=prod  →  '20260401_120000-prod'
      PIPELINE_ENV=dev   →  '20260401_120000-dev'
      (unset)            →  '20260401_120000'
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S") + env_suffix()


def latest_env_path(candidates: list[Path], *, name_attr: str = "parent") -> Path | None:
    """Return the most-recently-modified path that matches the current PIPELINE_ENV suffix.

    Checks the suffix against the path component named by `name_attr`:
      - "name"   : the path's own filename / dir name  (use for directories)
      - "parent" : the immediate parent directory name  (use for files inside timestamped dirs)
      - "stem"   : the filename without extension       (use for flat files like 1.4-llm_matched_*.json)

    Falls back to the overall newest candidate if none carry the expected suffix,
    so the pipeline still works when PIPELINE_ENV is not set.
    """
    if not candidates:
        return None

    suffix = env_suffix()  # "-prod", "-dev", or ""

    def _check(p: Path) -> bool:
        if name_attr == "name":
            return p.name.endswith(suffix)
        if name_attr == "parent":
            return p.parent.name.endswith(suffix)
        if name_attr == "stem":
            return p.stem.endswith(suffix)
        return False

    if suffix:
        env_cands = [p for p in candidates if _check(p)]
        if env_cands:
            return max(env_cands, key=lambda p: p.stat().st_mtime)

    # fallback — no suffix set or nothing matched: return overall newest
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_dotenv_file(env_path: Path | None = None) -> None:
    """Load key=value pairs from a .env file into os.environ (skip keys already set).

    Tries python-dotenv first; falls back to a simple manual parser so there is
    no hard dependency on the package.  Silently does nothing if the file does not exist.
    """
    path = env_path or (Path(__file__).resolve().parent / ".env")
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore[import]
        load_dotenv(path)
        return
    except ImportError:
        pass
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
