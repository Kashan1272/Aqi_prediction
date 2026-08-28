from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Runtime is inside Hopsworks.
os.environ["AQI_PROJECT_ROOT"] = str(ROOT)
os.environ["RUNNING_IN_HOPSWORKS"] = "true"
os.environ["FEATURE_STORE_BACKEND"] = "hybrid"
os.environ["HOPSWORKS_FEATURE_GROUP_VERSION"] = "9"
os.environ["HOPSWORKS_ONLINE_ENABLED"] = "false"
os.environ["MODEL_N_JOBS"] = "1"


def run(*args: str) -> None:
    command = [sys.executable, *args]
    print("\nRUN:", " ".join(command), flush=True)

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    print("Starting Pearls AQI Hopsworks hourly pipeline", flush=True)

    # 1. Restore persistent state from Hopsworks.
    run(
        "scripts/hydrate_hopsworks.py",
        "--features",
        "--model",
    )

    # 2. Generate fresh Day 1-3 forecasts.
    run(
        "scripts/forecast.py",
        "--city",
        "all",
    )

    # 3. Write refreshed observations/features/predictions back.
    run(
        "scripts/sync_hopsworks.py",
        "--features",
        "--city",
        "all",
    )

    print("Hourly pipeline completed successfully", flush=True)


if __name__ == "__main__":
    main()