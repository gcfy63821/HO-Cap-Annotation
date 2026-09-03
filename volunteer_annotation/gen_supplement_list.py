"""
Generate supplement annotation checklist (lenient rules for existing data):
- Primary tool: camera flagged only if BOTH frame0 AND ~50% (kf[3]) are missing
- Manipulated object: at least one camera must have frame0;
                      if frame0 exists but no ~50% and no end frame → require end frame
- Auxiliary tool: not required
"""
import json, sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import date

DB = Path("/data/robotool/_va_bundle_v2/tasks.db")
PROMPTS_DIR = Path("/data/robotool/_va_bundle_v2_prompts")
OUT = Path(__file__).parent / f"supplement_list_{date.today()}.md"


def task_quality(row: dict) -> dict:
    cams = json.loads(row["cameras_json"])
    prompt_dir = PROMPTS_DIR / row["task"] / row["exp"] / "tool_masks" / "prompts"
    errors = []
    pt_50_any = False
    pt_last_any = False
    mo_f0_any = False
    mo_50_any = False
    mo_last_any = False

    for cam in cams:
        cam_name = cam["camera"]
        kf = cam.get("keyframes", [0])
        kf0 = kf[0]
        kf_last = kf[-1]
        kf50 = kf[-2] if len(kf) > 2 else kf_last
        pf = prompt_dir / f"{cam_name}.json"
        if not pf.is_file():
            errors.append(f"{cam_name}:no_file")
            continue
        try:
            data = json.loads(pf.read_text())
        except Exception:
            continue
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

    # Primary tool: flag only if NO camera has 50% OR 100%
    if not pt_50_any and not pt_last_any:
        errors.append("主工具@50%和100%全视角均缺")

    # Manipulated object: frame0 required; end frame required only if 50% is also absent
    if not mo_f0_any:
        errors.append("操作对象@帧0:全视角均缺")
    if not mo_50_any and not mo_last_any:
        errors.append("操作对象@末帧:全视角均缺")

    return {"ok": len(errors) == 0, "errors": errors}


def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    # All submitted tasks
    rows = conn.execute(
        "SELECT * FROM tasks WHERE status='submitted' ORDER BY annotator_id, id"
    ).fetchall()
    rows = [dict(r) for r in rows]
    print(f"Total submitted: {len(rows)}")

    # Group by annotator
    by_ann = defaultdict(list)
    for r in rows:
        by_ann[r["annotator_id"]].append(r)

    # Stats
    total_tasks_need = 0
    total_ops = 0
    ann_stats = []

    lines = [f"# 补充标注清单 {date.today()}\n",
             "**规则**: 工具需帧0或帧~20%，操作对象需帧0，末帧不要求\n\n"]

    for ann_id in sorted(by_ann):
        tasks = by_ann[ann_id]
        need_tasks = []
        for t in tasks:
            q = task_quality(t)
            if not q["ok"]:
                need_tasks.append((t, q["errors"]))

        n_ops = sum(len(e) for _, e in need_tasks)
        total_tasks_need += len(need_tasks)
        total_ops += n_ops
        ann_stats.append((ann_id, len(tasks), len(need_tasks), n_ops))

        lines.append(f"## {ann_id}  ({len(need_tasks)}/{len(tasks)} 条需补充，共 {n_ops} 处)\n\n")
        if not need_tasks:
            lines.append("✅ 全部完整\n\n")
            continue
        for t, errs in need_tasks:
            lines.append(f"- **{t['exp']}** (id={t['id']})\n")
            for e in errs:
                lines.append(f"  - {e}\n")
        lines.append("\n")

    # Summary table at top
    summary = [f"# 补充标注清单 {date.today()}\n\n",
               "**规则**: 工具需帧0或帧~20%，操作对象需帧0，末帧不要求\n\n",
               f"**总计**: {total_tasks_need}/{len(rows)} 条需补充，共 {total_ops} 处操作\n\n",
               "| 志愿者 | 提交 | 需补充 | 操作数 |\n",
               "|--------|------|--------|--------|\n"]
    for ann_id, total, need, ops in sorted(ann_stats, key=lambda x: -x[3]):
        summary.append(f"| {ann_id} | {total} | {need} | {ops} |\n")
    summary.append("\n---\n\n")

    OUT.write_text("".join(summary) + "".join(lines[2:]), encoding="utf-8")
    print(f"Written: {OUT}")
    print(f"Tasks needing supplement: {total_tasks_need}/{len(rows)}")
    print(f"Total operations: {total_ops}")

    # Print summary table
    print(f"\n{'志愿者':<20} {'提交':>6} {'需补充':>8} {'操作数':>8}")
    print("-" * 50)
    for ann_id, total, need, ops in sorted(ann_stats, key=lambda x: -x[3]):
        print(f"{ann_id:<20} {total:>6} {need:>8} {ops:>8}")


if __name__ == "__main__":
    main()
