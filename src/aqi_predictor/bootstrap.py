from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    explicit = os.getenv("AQI_PROJECT_ROOT")
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if (root / "config" / "cities.yaml").is_file():
            return root

    candidates: list[Path] = [Path.cwd(), Path(__file__).resolve()]
    try:
        candidates.append(Path(sys.argv[0]).resolve())
    except (OSError, RuntimeError):
        pass

    seen: set[Path] = set()
    for candidate in candidates:
        start = candidate if candidate.is_dir() else candidate.parent
        for parent in (start, *start.parents):
            try:
                resolved = parent.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if (
                (resolved / "pyproject.toml").is_file()
                and (resolved / "config" / "cities.yaml").is_file()
            ):
                return resolved
    raise FileNotFoundError(
        "Could not find the project root. Run commands from the repository root "
        "or set AQI_PROJECT_ROOT."
    )


PROJECT_ROOT = project_root()
