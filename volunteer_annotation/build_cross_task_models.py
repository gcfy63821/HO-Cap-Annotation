"""
跨任务颜色模型构建脚本

把同一工具/对象在不同任务里的人工标注汇总成一个颜色模型，
然后部署到所有需要它的任务目录。

流程：
  1. 扫描所有 human-annotated exps（按工具名前缀分组）
  2. 对每个工具名，收集跨任务颜色样本
  3. 用工具名作 keyword 保存模型到每个相关任务目录
  4. 打印缺少标注的工具列表

Usage:
  cd HO-Cap-Annotation
  conda run -n hocap-annotation python volunteer_annotation/build_cross_task_models.py \\
      --task-prefix videos_02 [--dry-run]
"""

import sys, json, re, argparse
import numpy as np
import sqlite3
from pathlib import Path
from collections import defaultdict

ROOT     = Path(__file__).resolve().parent
CLOUD    = ROOT / "cloud"
SAM2_DIR = ROOT.parents[2] / "mesh_reconstruction" / "sam2"
sys.path.insert(0, str(CLOUD))
sys.path.insert(0, str(SAM2_DIR))

BUNDLE   = Path("/data/robotool/_va_bundle_v2")
PROMPTS  = Path("/data/robotool/_va_bundle_v2_prompts")
AUTO_PROMPTS = Path("/data/robotool/_va_bundle_v2_auto_prompts")
DB_PATH  = BUNDLE / "tasks.db"
MODEL_DIR = ROOT / "color_models"

from decoder import Sam2CpuDecoder
from template_color_extract import (
    build_model_for_template, save_model, exp_keyword
)

VERBS = ['_push_','_flip_','_dig_','_scoop_','_shape_',
         '_spread_','_press_','_open_','_roll_','_flatten_','_crush_']

def parse_tool_prefix(exp: str) -> str:
    """从 exp 名提取主操作工具前缀（action 动词之前的部分）。"""
    core = re.sub(r'^\d{8}_', '', exp)
    core = re.sub(r'_\d+$', '', core)
    for v in VERBS:
        if v in core:
            return core.split(v)[0]
    return core


def collect_all_annotated(task_prefix: str) -> dict[str, list[tuple[str, str]]]:
    """返回 {tool_prefix: [(task, exp), ...]} 所有有人工标注的 exps。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT task, exp FROM tasks WHERE task LIKE ?",
        (task_prefix + "%",)
    ).fetchall()
    conn.close()

    tool_exps: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        task, exp = row["task"], row["exp"]
        prompt_dir = PROMPTS / task / exp / "tool_masks" / "prompts"
        if not (prompt_dir.is_dir() and any(prompt_dir.glob("*.json"))):
            continue
        # Only include exps that have embed files in the bundle (SAM2 can run)
        bundle_exp_dir = BUNDLE / task / exp
        if not (bundle_exp_dir.is_dir() and any(bundle_exp_dir.glob("*.embed.npz"))):
            continue
        tool = parse_tool_prefix(exp)
        tool_exps[tool].append((task, exp))
    return tool_exps


def collect_all_tasks_for_tool(task_prefix: str, tool: str) -> list[str]:
    """返回包含 tool 前缀 exps 的所有 task。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT DISTINCT task FROM tasks WHERE task LIKE ? AND exp LIKE ?",
        (task_prefix + "%", f"%{tool}%")
    ).fetchall()
    conn.close()
    # 进一步过滤：exp 的工具前缀必须精确匹配
    result = set()
    conn2 = sqlite3.connect(str(DB_PATH))
    conn2.row_factory = sqlite3.Row
    all_rows = conn2.execute(
        "SELECT task, exp FROM tasks WHERE task LIKE ?",
        (task_prefix + "%",)
    ).fetchall()
    conn2.close()
    for row in all_rows:
        if parse_tool_prefix(row["exp"]) == tool:
            result.add(row["task"])
    return sorted(result)


def build_cross_task_model(tool: str, annotated_exps: list[tuple[str, str]],
                           dec: Sam2CpuDecoder, dry_run: bool = False):
    """收集 tool 的所有跨任务颜色样本，返回合并的 role_colors。"""
    print(f"\n{'='*60}")
    print(f"Tool: {tool}  ({len(annotated_exps)} annotated exps)")

    # 合并所有任务的颜色样本
    merged: dict[str, list] = defaultdict(list)
    for task, exp in annotated_exps:
        print(f"  Collecting: {task} / {exp}")
        if dry_run:
            continue
        role_colors = build_model_for_template(task, exp, dec)
        for role, samples in role_colors.items():
            merged[role].extend(samples)

    return dict(merged)


def deploy_model(tool: str, role_colors: dict, target_tasks: list[str],
                 template_exps: list[tuple[str, str]], dry_run: bool = False):
    """把模型保存到每个 target task 的 model 目录。"""
    deployed = []
    for task in target_tasks:
        task_slug = task.replace("/", "_")
        model_dir = MODEL_DIR / task_slug
        keyword = tool  # 工具前缀即 keyword
        meta_path = model_dir / f"{keyword}.meta.json"

        if dry_run:
            print(f"  [DRY] would save → {meta_path}")
            deployed.append(task)
            continue

        if not role_colors:
            print(f"  [SKIP] no samples for {tool}")
            continue

        # Use first annotated exp as canonical template; store its task for cross-task lookup
        primary_task, primary_exp = template_exps[0]
        save_model(model_dir, keyword, role_colors, primary_exp,
                   template_task=primary_task)
        deployed.append(task)
        print(f"  → {meta_path}")

    return deployed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-prefix", default="videos_02",
                        help="数据集前缀，如 videos_02 (default)")
    parser.add_argument("--tools", nargs="+", default=None,
                        help="只处理指定工具（默认全部）")
    parser.add_argument("--max-templates", type=int, default=5,
                        help="每个工具最多用几个 template exp（默认 5）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划，不实际运行 SAM2 或写文件")
    args = parser.parse_args()

    print(f"Scanning annotated exps under task prefix: {args.task_prefix}*")
    tool_exps = collect_all_annotated(args.task_prefix)

    # 统计所有 tools（含无标注）
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    all_rows = conn.execute(
        "SELECT task, exp FROM tasks WHERE task LIKE ?",
        (args.task_prefix + "%",)
    ).fetchall()
    conn.close()
    all_tools: dict[str, int] = defaultdict(int)
    for row in all_rows:
        all_tools[parse_tool_prefix(row["exp"])] += 1

    print(f"\n{'工具':<35} {'总exps':>6} {'有标注':>6} {'状态'}")
    print("-" * 60)
    tools_to_process = []
    for tool in sorted(all_tools, key=lambda t: -all_tools[t]):
        if args.tools and tool not in args.tools:
            continue
        n = all_tools[tool]
        annotated = len(tool_exps.get(tool, []))
        status = f"✓ {annotated} exps" if annotated else "← 无标注，跳过"
        print(f"  {tool:<33} {n:>6} {annotated:>6}  {status}")
        if annotated > 0:
            tools_to_process.append(tool)

    if not tools_to_process:
        print("\n没有可处理的工具（全部缺少标注）")
        return

    print(f"\n将处理 {len(tools_to_process)} 种工具")
    if args.dry_run:
        print("(DRY RUN 模式 — 不运行 SAM2)\n")

    dec = None if args.dry_run else Sam2CpuDecoder()

    summary = []
    for tool in tools_to_process:
        annotated_exps = tool_exps[tool][:args.max_templates]  # 上限
        target_tasks = collect_all_tasks_for_tool(args.task_prefix, tool)

        role_colors = build_cross_task_model(tool, annotated_exps, dec, args.dry_run)

        deployed = deploy_model(tool, role_colors, target_tasks, annotated_exps, args.dry_run)
        n_samples = sum(len(v) for v in role_colors.values()) if role_colors else 0
        summary.append({
            "tool": tool, "templates": len(annotated_exps),
            "samples": n_samples, "deployed_to": len(deployed)
        })

    print(f"\n{'='*60}")
    print("完成汇总：")
    for s in summary:
        print(f"  {s['tool']:<35} templates={s['templates']}  "
              f"samples={s['samples']}  deployed={s['deployed_to']} tasks")

    print("\n缺少人工标注（需补标）：")
    for tool in sorted(all_tools):
        if tool not in tool_exps and all_tools[tool] > 10:
            print(f"  {tool:<35} ({all_tools[tool]} exps)")


if __name__ == "__main__":
    main()
