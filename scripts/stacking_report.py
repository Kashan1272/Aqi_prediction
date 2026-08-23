from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OOF stacking weights and test metrics.")
    parser.add_argument("--report", default="reports/training_report_v6.json")
    args = parser.parse_args()
    path = Path(args.report)
    if not path.is_absolute():
        path = ROOT / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = {
        "project_version": payload.get("project_version"),
        "selected_algorithms": payload.get("selected_algorithms"),
        "test_metrics": payload.get("test_metrics", {}).get("daily_mean"),
        "quality_gate": payload.get("quality_gate"),
        "promotion": payload.get("promotion"),
        "stacking": {
            key: {
                "strategy": value.get("strategy"),
                "weights": value.get("weights"),
                "oof_metrics": value.get("oof_metrics"),
                "best_single_component": value.get("best_single_component"),
                "relative_rmse_gain_vs_best": value.get("relative_rmse_gain_vs_best"),
            }
            for key, value in payload.get("candidate_evaluation", {}).items()
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
