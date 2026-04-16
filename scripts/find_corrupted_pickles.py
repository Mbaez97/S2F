#!/usr/bin/env python3
"""
Find corrupted pickle files in an ortholog cache directory.

Examples:
  python3 find_corrupted_pickles.py --dir /media/marcelo_baez/HD_Disc1/.S2F/orthologs
  python3 find_corrupted_pickles.py --dir /media/marcelo_baez/HD_Disc1/.S2F/orthologs --move-bad
"""

import argparse
import os
import shutil
import sys

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect corrupted pickle files.")
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing pickle files (e.g., ortholog cache).",
    )
    parser.add_argument(
        "--move-bad",
        action="store_true",
        help="Move corrupted files to <dir>/_broken_backup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = os.path.abspath(args.dir)

    if not os.path.isdir(base_dir):
        raise SystemExit(f"Directory does not exist: {base_dir}")

    backup_dir = os.path.join(base_dir, "_broken_backup")
    if args.move_bad:
        os.makedirs(backup_dir, exist_ok=True)

    total = 0
    bad_files = []

    for name in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, name)
        if not os.path.isfile(path):
            continue
        if name == "_broken_backup":
            continue

        total += 1
        size = os.path.getsize(path)
        if size == 0:
            bad_files.append((path, "empty file"))
            continue

        try:
            pd.read_pickle(path)
        except Exception as exc:  # noqa: BLE001
            bad_files.append((path, f"{type(exc).__name__}: {exc}"))

    print(f"Scanned files: {total}")
    print(f"Corrupted files: {len(bad_files)}")

    for path, reason in bad_files:
        print(f"[BAD] {path} -> {reason}")
        if args.move_bad:
            dest = os.path.join(backup_dir, os.path.basename(path))
            shutil.move(path, dest)
            print(f"      moved to {dest}")

    if bad_files:
        sys.exit(1)


if __name__ == "__main__":
    main()
