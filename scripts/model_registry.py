from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT  # noqa: F401
from aqi_predictor.config import get_settings
from aqi_predictor.registry import LocalModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage local champion/challenger model versions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List all local model versions")

    promote = subparsers.add_parser("promote", help="Promote a specific local version")
    promote.add_argument("version")

    rollback = subparsers.add_parser("rollback", help="Restore a previous champion")
    rollback.add_argument("--steps", type=int, default=1)

    subparsers.add_parser("production", help="Show the current production pointer")

    args = parser.parse_args()
    registry = LocalModelRegistry(get_settings())

    if args.command == "list":
        print(json.dumps(registry.list_versions(), indent=2))
    elif args.command == "promote":
        path = registry.promote_version(args.version, reason="manual_registry_promotion")
        print(json.dumps({"promoted": args.version, "version_path": str(path)}, indent=2))
    elif args.command == "rollback":
        path = registry.rollback(steps=args.steps)
        print(json.dumps({"rolled_back_to": path.name, "version_path": str(path)}, indent=2))
    elif args.command == "production":
        path = registry.production_dir()
        print(json.dumps({"production_version": path.name, "version_path": str(path)}, indent=2))


if __name__ == "__main__":
    main()
