#!/usr/bin/env python3
"""Seed the task DB from a bundle manifest (one task per camera-view frame).

Usage:
    python seed_tasks.py --bundle /tmp/va_bundle --db /tmp/va_tasks.db
    python seed_tasks.py --bundle /tmp/va_bundle --db /tmp/va_tasks.db --default_object cube_small
"""

import argparse
from pathlib import Path

import tasks_db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="bundle root (contains manifest.json)")
    ap.add_argument("--db", required=True, help="sqlite db path")
    args = ap.parse_args()

    conn = tasks_db.connect(args.db)
    n_new = tasks_db.seed_from_manifest(conn, Path(args.bundle) / "manifest.json")
    print(f"[seed] +{n_new} new task(s); totals: {tasks_db.stats(conn)}")


if __name__ == "__main__":
    main()
