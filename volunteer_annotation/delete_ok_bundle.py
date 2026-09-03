"""
Delete SAM2 embedding bundle dirs for submitted+OK experiments to free disk space.
Prompts and DB are NOT touched.

Usage:
  python delete_ok_bundle.py          # dry run — prints what would be deleted
  python delete_ok_bundle.py --delete  # actually delete
"""
import json, sqlite3, argparse, shutil
from pathlib import Path
from collections import defaultdict

DB = Path("/data/robotool/_va_bundle_v2/tasks.db")
BUNDLE_DIR = Path("/data/robotool/_va_bundle_v2")
PROMPTS_DIR = Path("/data/robotool/_va_bundle_v2_prompts")


def task_quality(row: dict) -> bool:
    """Return True if task is OK (no supplement needed)."""
    cams = json.loads(row["cameras_json"])
    prompt_dir = PROMPTS_DIR / row["task"] / row["exp"] / "tool_masks" / "prompts"
    pt_50_any = pt_last_any = False
    mo_f0_any = mo_50_any = mo_last_any = False

    for cam in cams:
        cam_name = cam["camera"]
        kf = cam.get("keyframes", [0])
        kf0 = kf[0]
        kf_last = kf[-1]
        kf50 = kf[-2] if len(kf) > 2 else kf_last
        pf = prompt_dir / f"{cam_name}.json"
        if not pf.is_file():
            return False
        try:
            data = json.loads(pf.read_text())
        except Exception:
            return False
        by_role: dict[str, set] = {}
        for o in data.get("objects", []):
            if o.get("points"):
                by_role.setdefault(o["role"], set()).add(o.get("frame_index", 0))

        pt = by_role.get("primary_tool", set())
        if kf50 in pt:
            pt_50_any = True
        if kf_last in pt:
            pt_last_any = True

        mo = by_role.get("manipulated_object", set())
        if kf0 in mo:
            mo_f0_any = True
        if kf50 in mo:
            mo_50_any = True
        if kf_last in mo:
            mo_last_any = True

    if not pt_50_any and not pt_last_any:
        return False
    if not mo_f0_any:
        return False
    if not mo_50_any and not mo_last_any:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="Actually delete (default: dry run)")
    args = parser.parse_args()

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM tasks WHERE status='submitted' ORDER BY id"
    ).fetchall()
    rows = [dict(r) for r in rows]
    print(f"Total submitted: {len(rows)}")

    to_delete = []
    to_keep = []
    missing_bundle = []

    for i, row in enumerate(rows):
        if (i + 1) % 200 == 0:
            print(f"  checked {i+1}/{len(rows)}...")
        bundle_path = BUNDLE_DIR / row["task"] / row["exp"]
        if not bundle_path.exists():
            missing_bundle.append(row["exp"])
            continue
        ok = task_quality(row)
        if ok:
            to_delete.append((row, bundle_path))
        else:
            to_keep.append((row, bundle_path))

    print(f"\nOK (can delete): {len(to_delete)}")
    print(f"Need supplement (keep): {len(to_keep)}")
    print(f"Already missing bundle: {len(missing_bundle)}")

    # Estimate size
    if to_delete:
        sample = to_delete[0][1]
        import subprocess
        result = subprocess.run(["du", "-sh", str(sample)], capture_output=True, text=True)
        sample_size = result.stdout.split()[0] if result.returncode == 0 else "?"
        print(f"Sample bundle size: {sample_size} (first exp: {sample.name})")

    if not args.delete:
        print("\n[DRY RUN] Use --delete to actually delete.")
        print("First 10 to delete:")
        for row, p in to_delete[:10]:
            print(f"  {p}")
        return

    # Actually delete
    deleted = 0
    failed = 0
    for row, bundle_path in to_delete:
        try:
            shutil.rmtree(bundle_path)
            deleted += 1
            if deleted % 100 == 0:
                print(f"  deleted {deleted}/{len(to_delete)}...")
        except Exception as e:
            print(f"  ERROR deleting {bundle_path}: {e}")
            failed += 1

    print(f"\nDeleted: {deleted}, Failed: {failed}")

    # Clean up empty task dirs
    print("Cleaning empty task dirs...")
    for task_dir in BUNDLE_DIR.iterdir():
        if task_dir.is_dir() and task_dir.name.startswith("videos_"):
            for exp_dir in list(task_dir.iterdir()):
                pass  # already deleted above
            if not any(task_dir.iterdir()):
                task_dir.rmdir()
                print(f"  removed empty: {task_dir.name}")

    print("Done.")


if __name__ == "__main__":
    main()
