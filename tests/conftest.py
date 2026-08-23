from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("AQI_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
