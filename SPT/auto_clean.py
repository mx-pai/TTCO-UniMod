#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
from datetime import datetime
from typing import List, Tuple


def list_run_dirs(config_dir: str) -> List[Tuple[str, float]]:
    try:
        entries = [
            (os.path.join(config_dir, name), os.path.getmtime(os.path.join(config_dir, name)))
            for name in os.listdir(config_dir)
        ]
    except FileNotFoundError:
        return []
    run_dirs = [(path, mtime) for path, mtime in entries if os.path.isdir(path)]
    return sorted(run_dirs, key=lambda item: item[1], reverse=True)


def collect_targets(root: str, keep: int, configs: List[str]) -> List[str]:
    targets = []
    config_dirs = []
    if configs:
        for cfg in configs:
            config_dirs.append(os.path.join(root, cfg))
    else:
        try:
            config_dirs = [
                os.path.join(root, name)
                for name in os.listdir(root)
                if os.path.isdir(os.path.join(root, name))
            ]
        except FileNotFoundError:
            return []

    for cfg_dir in config_dirs:
        run_dirs = list_run_dirs(cfg_dir)
        if not run_dirs:
            continue
        for path, _ in run_dirs[keep:]:
            targets.append(path)
    return targets


def human_rel_time(mtime: float) -> str:
    dt = datetime.fromtimestamp(mtime)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def print_plan(targets: List[str]) -> None:
    if not targets:
        print("Nothing to clean.")
        return
    print("Planned removals:")
    for path in targets:
        try:
            mtime = os.path.getmtime(path)
            stamp = human_rel_time(mtime)
        except FileNotFoundError:
            stamp = "unknown"
        print(f"  - {path} (last modified {stamp})")


def remove_paths(targets: List[str]) -> None:
    for path in targets:
        if not os.path.exists(path):
            continue
        print(f"Removing {path}")
        shutil.rmtree(path, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean older training runs to free disk space."
    )
    parser.add_argument(
        "--root",
        default="/root/autodl-tmp/spt_runs",
        help="Root directory containing training runs (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Limit cleanup to specific config name(s). Repeat to supply multiple.",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=3,
        help="Number of newest runs to keep for each config (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Execute deletions. Without this flag a dry-run report is shown.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.isdir(root):
        print(f"[ERROR] Root directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    targets = collect_targets(root, max(args.keep, 0), args.configs or [])
    if not args.quiet:
        print(f"Scanning runs under: {root}")
        print(f"Keeping newest {args.keep} run(s) per config.")
        print_plan(targets)

    if not targets:
        return

    if not args.force:
        print("\nDry run only. Re-run with --force to apply deletions.")
        return

    remove_paths(targets)
    print("Cleanup completed.")


if __name__ == "__main__":
    main()
