from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Running inside Hopsworks.
os.environ["AQI_PROJECT_ROOT"] = str(ROOT)
os.environ["RUNNING_IN_HOPSWORKS"] = "true"
os.environ["FEATURE_STORE_BACKEND"] = "hybrid"

# These cannot be supplied as Hopsworks Job env vars because HOPSWORKS_*
# names are reserved, so set them inside the Python process.
os.environ["HOPSWORKS_FEATURE_GROUP_VERSION"] = "9"
os.environ["HOPSWORKS_ONLINE_ENABLED"] = "false"

# Limit model/native parallelism to avoid unnecessary RAM spikes.
os.environ["MODEL_N_JOBS"] = os.getenv("MODEL_N_JOBS", "2")


def run(*args: str) -> None:
    command = [sys.executable, *args]

    print(
        "\nRUN:",
        " ".join(command),
        flush=True,
    )

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def run_training() -> dict:
    command = [
        sys.executable,
        "scripts/train.py",
        "--city",
        "all",
    ]

    print(
        "\nRUN:",
        " ".join(command),
        flush=True,
    )

    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    # Keep normal training output visible in Hopsworks logs.
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)

    print(result.stdout, flush=True)

    # train.py prints its final result as JSON.
    text = result.stdout.strip()

    start = text.rfind("\n{")
    if start >= 0:
        text = text[start + 1:]

    return json.loads(text)


def main() -> None:
    print(
        "Starting Pearls AQI daily training pipeline",
        flush=True,
    )

    # --------------------------------------------------------------
    # 1. Hydrate historical feature data from Hopsworks
    # --------------------------------------------------------------
    run(
        "scripts/hydrate_hopsworks.py",
        "--features",
    )

    # --------------------------------------------------------------
    # 2. Validate data contract before training
    # --------------------------------------------------------------
    run(
        "scripts/validate_data.py",
        "--city",
        "all",
        "--strict",
    )

    # --------------------------------------------------------------
    # 3. Train/evaluate models and run promotion quality gate
    # --------------------------------------------------------------
    training_result = run_training()

    version_path = Path(
        training_result["version_path"]
    )

    version = version_path.name

    promoted = bool(
        training_result.get("promoted", False)
    )

    quality_gate_passed = bool(
        training_result.get(
            "quality_gate_passed",
            False,
        )
    )

    print(
        json.dumps(
            {
                "trained_version": version,
                "promoted": promoted,
                "quality_gate_passed":
                    quality_gate_passed,
                "test_metrics":
                    training_result.get(
                        "test_metrics"
                    ),
            },
            indent=2,
        ),
        flush=True,
    )

    # --------------------------------------------------------------
    # 4. Register today's trained model as challenger
    # --------------------------------------------------------------
    run(
        "scripts/sync_hopsworks.py",
        "--candidate",
        "--version",
        version,
    )

    # --------------------------------------------------------------
    # 5. Update production registry only when newly promoted
    # --------------------------------------------------------------
    if promoted:
        print(
            "New model passed promotion gate; "
            "syncing production champion.",
            flush=True,
        )

        run(
            "scripts/sync_hopsworks.py",
            "--model",
        )

    else:
        print(
            "Candidate was not promoted. "
            "Existing production champion remains unchanged.",
            flush=True,
        )

    print(
        "Daily training pipeline completed successfully",
        flush=True,
    )


if __name__ == "__main__":
    main()