#!/usr/bin/env python3
"""
Browse pose-offset diagnostics produced by debug/diaognose_pose_offset.py.

Each exp's diagnostics live under
  <annotated>/<exp>/debug/pose_offset_diag/
    summary.json
    frame_<NNNNNN>/cam_<serial>.png    (one file per cam, all frames)

This tool builds a single self-contained HTML index where every diagnosed
exp gets a card. Each card has:
  - task / exp label + offset stats from summary.json
  - frame picker (← / → / ↑ / ↓ keys, click thumbnail strip, or dropdown)
  - 4x2 cam grid for the current frame; click any tile to zoom

Two modes (mirror browse_frame0_debug.py):

1) HTML index (default; static, no server):
     python scripts/browse_pose_offset_diag.py --videos_root <path>
   Writes <annotated_root>/_pose_offset_diag_index.html .

2) Live HTTP server (for cluster + SSH tunnel):
     python scripts/browse_pose_offset_diag.py --videos_root <path> --serve
   Serves <annotated_root>/ over HTTP and prints the URL.

You can also point it at a local sync of just the annotated tree:
     python scripts/browse_pose_offset_diag.py --annotated_root /path/to/local_copy
"""

import argparse
import html
import json
import os
import socket
import sys
from pathlib import Path


def annotated_root_for_videos(videos_root: Path) -> Path:
    return videos_root.parent / f"{videos_root.name}_annotated"


# ============================================================
# Discovery
# ============================================================

def gather_pose_offset_entries(annotated_root: Path, only_with_summary=True,
                                  pose_sources=("ob_in_cam", "merged")):
    """Walk <annotated_root>/<task>/<exp>/debug/pose_offset_diag{,_<src>}/
    and collect one entry per (exp, pose_source).

    Each entry has:
        task, exp, pose_source, diag_dir, summary_path,
        summary (parsed JSON or None),
        frames = [{"frame": int, "cams": {serial: rel_path_from_annotated_root}}, ...]
        merger_done, hand_done, joint_done   (status badges)
    """
    out = []
    if not annotated_root.is_dir():
        return out
    # Map pose_source -> diag subdir name (matches diagnose script convention).
    src_to_subdir = {
        s: ("pose_offset_diag" if s == "ob_in_cam" else f"pose_offset_diag_{s}")
        for s in pose_sources
    }
    for task_dir in sorted(p for p in annotated_root.iterdir() if p.is_dir()):
        for exp_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
            for pose_source, subdir in src_to_subdir.items():
                diag_dir = exp_dir / "debug" / subdir
                if not diag_dir.is_dir():
                    if only_with_summary:
                        continue
                summary_path = diag_dir / "summary.json"
                summary = None
                if summary_path.exists():
                    try:
                        summary = json.loads(summary_path.read_text())
                    except Exception as e:
                        summary = {"_parse_error": str(e)}
                if only_with_summary and summary is None:
                    continue

                # Build per-frame cam image lists. Prefer the order in summary.json
                # (so the user sees the same frames the diagnose script analyzed),
                # fall back to walking the directory.
                frames = []
                seen_frames = set()
                if isinstance(summary, dict) and "per_frame" in summary:
                    for fr_record in summary["per_frame"]:
                        fr = int(fr_record["frame"])
                        fdir = diag_dir / f"frame_{fr:06d}"
                        if not fdir.is_dir():
                            continue
                        cams = {}
                        for png in sorted(fdir.glob("cam_*.png")):
                            serial = png.stem.split("_", 1)[1]
                            cams[serial] = str(png.relative_to(annotated_root))
                        if cams:
                            frames.append({"frame": fr, "cams": cams})
                            seen_frames.add(fr)
                # Walk dir for any extra frames (or all frames if no summary).
                for fdir in sorted(diag_dir.glob("frame_*")) if diag_dir.is_dir() else []:
                    if not fdir.is_dir(): continue
                    try:
                        fr = int(fdir.name.split("_", 1)[1])
                    except ValueError:
                        continue
                    if fr in seen_frames:
                        continue
                    cams = {}
                    for png in sorted(fdir.glob("cam_*.png")):
                        serial = png.stem.split("_", 1)[1]
                        cams[serial] = str(png.relative_to(annotated_root))
                    if cams:
                        frames.append({"frame": fr, "cams": cams})

                out.append({
                    "task": task_dir.name,
                    "exp": exp_dir.name,
                    "pose_source": pose_source,
                    "diag_dir": diag_dir,
                    "summary_path": summary_path,
                    "summary": summary,
                    "frames": frames,
                    "merger_done": (exp_dir / "processed/fd_pose_solver/fd_poses_merged_fixed.npy").exists(),
                    "hand_done":   (exp_dir / "result_hand_optimized.pkl").exists(),
                    "joint_done":  (exp_dir / "processed/joint_pose_solver/poses_o.npy").exists(),
                })
    return out


# ============================================================
# HTML
# ============================================================

HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Pose offset diagnostics — {root_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         background: #1a1a1a; color: #e0e0e0; margin: 0; padding: 16px; }}
  h1 {{ font-size: 18px; margin: 0 0 12px; }}
  .stats {{ font-size: 13px; color: #aaa; margin-bottom: 12px; }}
  .filter {{ margin: 8px 0 16px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
  .filter input {{ padding: 6px 10px; width: 360px; background: #2a2a2a;
                   color: #e0e0e0; border: 1px solid #444; border-radius: 4px; }}
  .filter label {{ font-size: 12px; color: #aaa; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(720px, 1fr));
           gap: 14px; }}
  .card {{ background: #242424; border-radius: 6px; overflow: hidden;
           border: 1px solid #333; padding-bottom: 8px; }}
  .card .label {{ padding: 10px 12px; font-size: 11px; font-family: monospace;
                  border-bottom: 1px solid #333; line-height: 1.5;
                  word-break: break-all; overflow-wrap: anywhere; }}
  .card .label .task {{ color: #4af; font-weight: 600; display: block; }}
  .card .label .exp  {{ color: #fff; font-size: 12px; display: block; margin-top: 2px;
                        user-select: all; }}
  .card .label .copy_btn {{ display: inline-block; margin-left: 6px; padding: 1px 6px;
                            font-size: 10px; background: #333; color: #aaa;
                            border-radius: 3px; cursor: pointer; border: 1px solid #444; }}
  .card .label .copy_btn:hover {{ background: #555; color: #fff; }}
  .card .label .badges {{ margin-top: 6px; }}
  .badge {{ display: inline-block; font-size: 10px; padding: 1px 6px;
            border-radius: 3px; margin-right: 4px; background: #333; color: #888; }}
  .badge.ok {{ background: #1e4d2b; color: #8f8; }}
  .badge.miss {{ background: #4d1e1e; color: #f88; }}
  .offset-stats {{ margin-top: 6px; font-size: 11px; color: #cd6; }}
  .offset-stats .big {{ color: #fc6; font-weight: 600; }}
  .frame_picker {{ display: flex; align-items: center; gap: 6px;
                   padding: 6px 10px; background: #1c1c1c; border-bottom: 1px solid #333;
                   font-family: monospace; font-size: 11px; }}
  .frame_picker button {{ padding: 2px 8px; background: #333; color: #ddd;
                          border: 1px solid #555; cursor: pointer; }}
  .frame_picker .frame_label {{ color: #fc6; min-width: 130px; }}
  .frame_picker select {{ background: #111; color: #ddd; border: 1px solid #444;
                          padding: 2px; }}
  .frame_strip {{ display: flex; gap: 2px; overflow-x: auto; padding: 4px;
                   background: #161616; border-bottom: 1px solid #333; }}
  .frame_strip .ftick {{ font-size: 10px; color: #888; padding: 2px 6px;
                          background: #222; border: 1px solid #333; cursor: pointer;
                          white-space: nowrap; }}
  .frame_strip .ftick.active {{ background: #2a6; color: white; border-color: #2a6; }}
  .cam_grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 2px;
                padding: 4px; background: #111; }}
  .cam_grid .cam_tile {{ position: relative; background: #000; }}
  .cam_grid .cam_tile img {{ width: 100%; height: auto; display: block;
                              cursor: zoom-in; }}
  .cam_grid .cam_tile .serial {{ position: absolute; top: 2px; left: 4px;
                                   color: #fc6; font-size: 10px; font-family: monospace;
                                   text-shadow: 0 0 2px #000, 0 0 2px #000; }}
  .empty {{ background: #4d1e1e; color: #fcc; padding: 20px; text-align: center;
            font-size: 12px; }}
  .lightbox {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.95);
               align-items: center; justify-content: center; z-index: 100;
               cursor: zoom-out; flex-direction: column; padding: 16px; }}
  .lightbox.visible {{ display: flex; }}
  .lightbox img {{ max-width: 95vw; max-height: 90vh; }}
  .lightbox .lb_label {{ color: #ccc; font-family: monospace; font-size: 14px;
                          margin-bottom: 8px; word-break: break-all;
                          overflow-wrap: anywhere; max-width: 95vw;
                          text-align: center; user-select: all; }}
</style>
</head><body>
"""


HTML_TAIL = """
<div class="lightbox" id="lb" onclick="this.classList.remove('visible')">
  <div class="lb_label" id="lblabel"></div>
  <img id="lbimg" src="">
</div>
<script>
function copyText(btn, text) {
  navigator.clipboard.writeText(text).then(() => {
    const orig = btn.textContent;
    btn.textContent = 'copied!';
    setTimeout(() => { btn.textContent = orig; }, 900);
  }).catch(() => { alert(text); });
}

// Each card carries its full per-frame manifest in data-frames (JSON).
function frameDataOf(card) {
  if (card._frames) return card._frames;
  card._frames = JSON.parse(card.dataset.frames || '[]');
  return card._frames;
}

function setCardFrame(card, frameIdx) {
  const frames = frameDataOf(card);
  if (!frames.length) return;
  if (frameIdx < 0) frameIdx = 0;
  if (frameIdx >= frames.length) frameIdx = frames.length - 1;
  card.dataset.curIdx = frameIdx;
  const f = frames[frameIdx];
  card.querySelector('.frame_label').textContent =
      'frame ' + f.frame + '   (' + (frameIdx + 1) + '/' + frames.length + ')';
  // Update tiles
  card.querySelectorAll('.cam_tile').forEach(tile => {
    const s = tile.dataset.serial;
    const img = tile.querySelector('img');
    if (f.cams[s]) {
      img.src = f.cams[s];
      img.dataset.label = card.dataset.label + '   frame ' + f.frame + '   cam ' + s;
      tile.style.display = '';
    } else {
      tile.style.display = 'none';
    }
  });
  // Update the strip
  card.querySelectorAll('.ftick').forEach((t, i) => {
    t.classList.toggle('active', i === frameIdx);
  });
  // Update dropdown
  const sel = card.querySelector('.frame_select');
  if (sel) sel.value = String(frameIdx);
  // Scroll the active tick into view inside its strip
  const active = card.querySelectorAll('.ftick')[frameIdx];
  if (active && active.scrollIntoView) {
    active.scrollIntoView({block: 'nearest', inline: 'center'});
  }
}

function nextFrame(btn, dir) {
  const card = btn.closest('.card');
  const frames = frameDataOf(card);
  if (!frames.length) return;
  let i = parseInt(card.dataset.curIdx || '0', 10);
  i = (i + dir + frames.length) % frames.length;
  setCardFrame(card, i);
}

function jumpFrame(card, idx) {
  setCardFrame(card, parseInt(idx, 10));
}

document.addEventListener('DOMContentLoaded', () => {
  // Initialise each card to frame 0
  document.querySelectorAll('.card').forEach(c => {
    if (frameDataOf(c).length) setCardFrame(c, 0);
  });
  // Lightbox
  document.querySelectorAll('.cam_tile img').forEach(img => {
    img.addEventListener('click', e => {
      const lb = document.getElementById('lb');
      document.getElementById('lbimg').src = img.src;
      document.getElementById('lblabel').textContent = img.dataset.label || img.src;
      lb.classList.add('visible');
    });
  });
  // Keyboard arrows: switch frames on the currently-hovered card
  let hovered = null;
  document.querySelectorAll('.card').forEach(c => {
    c.addEventListener('mouseenter', () => hovered = c);
    c.addEventListener('mouseleave', () => { if (hovered === c) hovered = null; });
  });
  document.addEventListener('keydown', e => {
    if (!hovered) return;
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
      e.preventDefault();
      const i = parseInt(hovered.dataset.curIdx || '0', 10);
      setCardFrame(hovered, i - 1);
    } else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
      e.preventDefault();
      const i = parseInt(hovered.dataset.curIdx || '0', 10);
      setCardFrame(hovered, i + 1);
    }
  });
  // Filter (text + source)
  const f = document.getElementById('filter');
  const srcF = document.getElementById('srcfilter');
  function applyFilters() {
    const q = (f ? f.value : '').toLowerCase();
    const s = srcF ? srcF.value : '';
    document.querySelectorAll('.card').forEach(c => {
      const okText = c.dataset.label.toLowerCase().includes(q);
      const okSrc  = !s || c.dataset.source === s;
      c.style.display = (okText && okSrc) ? '' : 'none';
    });
  }
  if (f)    f.addEventListener('input',  applyFilters);
  if (srcF) srcF.addEventListener('change', applyFilters);
  // Sort
  const sort = document.getElementById('sortby');
  if (sort) {
    sort.addEventListener('change', () => {
      const by = sort.value;
      const grid = document.querySelector('.grid');
      const cards = Array.from(grid.querySelectorAll('.card'));
      cards.sort((a, b) => {
        const va = parseFloat(a.dataset[by] || '-1');
        const vb = parseFloat(b.dataset[by] || '-1');
        if (isNaN(va) && isNaN(vb)) return 0;
        if (isNaN(va)) return 1;
        if (isNaN(vb)) return -1;
        return vb - va;     // descending — worst offsets first
      });
      cards.forEach(c => grid.appendChild(c));
    });
  }
});
</script>
</body></html>
"""


def fmt_vec(v, precision=3):
    if v is None: return "?"
    if isinstance(v, list):
        return "[" + ", ".join(f"{float(x):+.{precision}f}" for x in v) + "]"
    return str(v)


def build_card(entry, annotated_root: Path) -> str:
    src = entry.get("pose_source", "ob_in_cam")
    label = f'{entry["task"]}/{entry["exp"]}  [{src}]'
    label_attr = html.escape(label, quote=True)

    summary = entry["summary"] or {}
    aggregate = summary.get("aggregate", {}) if isinstance(summary, dict) else {}
    mean = aggregate.get("mean_offset_cam")
    median = aggregate.get("median_offset_cam")
    std = aggregate.get("std_offset_cam")
    norm = aggregate.get("offset_norm_mean")
    n_obs = aggregate.get("n_camera_observations", "?")
    obj_id = summary.get("object_id", "?")

    parts = [f'<div class="card" data-label="{label_attr}" '
              f'data-source="{html.escape(src, quote=True)}" '
              f'data-norm="{(norm if norm is not None else -1)}" '
              f'data-frames="{html.escape(json.dumps(entry["frames"]), quote=True)}" '
              f'title="{label_attr}">']
    badges = [
        ("masks.h5",  "ok" if (annotated_root / entry["task"] / entry["exp"] / "tool_masks/masks.h5").exists() else "miss"),
        ("fd merge",  "ok" if entry["merger_done"] else "miss"),
        ("hand",      "ok" if entry["hand_done"]   else "miss"),
        ("joint",     "ok" if entry["joint_done"]  else "miss"),
    ]
    badges_html = " ".join(
        f'<span class="badge {state}">{html.escape(name)}</span>'
        for name, state in badges
    )
    stats_html = ""
    if norm is not None:
        stats_html = (
            f'<div class="offset-stats">'
            f'<span class="big">||offset||={norm:.4f} m</span>'
            f' &nbsp;·&nbsp; n_cams={n_obs}'
            f' &nbsp;·&nbsp; obj={html.escape(str(obj_id))}<br>'
            f'mean={fmt_vec(mean)}'
            f' &nbsp;·&nbsp; median={fmt_vec(median)}'
            f' &nbsp;·&nbsp; std={fmt_vec(std)}'
            f'</div>'
        )
    elif isinstance(summary, dict) and summary.get("_parse_error"):
        stats_html = f'<div class="offset-stats">summary.json parse error: {html.escape(summary["_parse_error"])}</div>'
    elif not aggregate:
        stats_html = '<div class="offset-stats">no aggregate (no valid camera observations)</div>'

    src_color = "#6cf" if src == "ob_in_cam" else "#fc6"
    parts.append(
        f'<div class="label">'
        f'<span class="task">{html.escape(entry["task"])}'
        f'  <span style="color:{src_color};font-size:10px">[{html.escape(src)}]</span>'
        f'</span>'
        f'<span class="exp">{html.escape(entry["exp"])}'
        f'<span class="copy_btn" onclick="copyText(this, &quot;{label_attr}&quot;); event.stopPropagation();">copy</span>'
        f'</span>'
        f'<div class="badges">{badges_html}</div>'
        f'{stats_html}'
        f'</div>'
    )

    frames = entry["frames"]
    if not frames:
        parts.append('<div class="empty">no frame_<NNNNNN>/cam_*.png found '
                      '— run debug/diaognose_pose_offset.py first</div>')
        parts.append('</div>')
        return "".join(parts)

    # Frame selector controls.
    frame_options = "".join(
        f'<option value="{i}">frame {f["frame"]}  ({i+1}/{len(frames)})</option>'
        for i, f in enumerate(frames)
    )
    parts.append(
        '<div class="frame_picker">'
        '<button onclick="nextFrame(this, -1)">←</button>'
        '<span class="frame_label">frame ?</span>'
        '<button onclick="nextFrame(this, +1)">→</button>'
        '<span style="color:#888">jump:</span>'
        f'<select class="frame_select" onchange="jumpFrame(this.closest(\'.card\'), this.value)">'
        f'{frame_options}</select>'
        '<span style="color:#666;margin-left:auto">arrows = prev/next on hovered card</span>'
        '</div>'
    )

    # Frame strip thumbnails (clickable ticks).
    strip = ['<div class="frame_strip">']
    for i, f in enumerate(frames):
        strip.append(
            f'<span class="ftick" onclick="setCardFrame(this.closest(\'.card\'), {i})">'
            f'{f["frame"]}</span>'
        )
    strip.append('</div>')
    parts.append("".join(strip))

    # Cam grid: collect union of serials across frames so the layout is stable.
    all_serials = sorted({s for f in frames for s in f["cams"].keys()})
    grid = ['<div class="cam_grid">']
    for s in all_serials:
        grid.append(
            f'<div class="cam_tile" data-serial="{html.escape(s)}">'
            f'<span class="serial">cam {html.escape(s)}</span>'
            f'<img src="" data-label="" alt="cam {html.escape(s)}">'
            f'</div>'
        )
    grid.append('</div>')
    parts.append("".join(grid))

    parts.append('</div>')
    return "".join(parts)


def write_index(annotated_root: Path, entries, out_path: Path):
    n_total = len(entries)
    n_with_frames = sum(1 for e in entries if e["frames"])
    n_with_offset = sum(1 for e in entries
                          if isinstance(e["summary"], dict)
                          and e["summary"].get("aggregate", {}).get("mean_offset_cam") is not None)

    parts = [HTML_HEAD.format(root_name=html.escape(annotated_root.name))]
    parts.append(f'<h1>Pose offset diagnostics <span style="color:#888">— '
                  f'{html.escape(annotated_root.name)}</span></h1>')
    by_src = {}
    for e in entries:
        by_src.setdefault(e.get("pose_source", "ob_in_cam"), 0)
        by_src[e.get("pose_source", "ob_in_cam")] += 1
    src_breakdown = "  ".join(f"{s}={n}" for s, n in sorted(by_src.items()))
    parts.append(
        f'<div class="stats">'
        f'{n_total} entries with diag dir &nbsp;·&nbsp; '
        f'{n_with_frames} have rendered frames &nbsp;·&nbsp; '
        f'{n_with_offset} have offset stats &nbsp;·&nbsp; '
        f'by source: {html.escape(src_breakdown)}'
        f'</div>'
    )
    sources_present = sorted({e.get("pose_source", "ob_in_cam") for e in entries})
    src_options = "".join(
        f'<option value="{html.escape(s)}">{html.escape(s)}</option>' for s in sources_present
    )
    parts.append(
        '<div class="filter">'
        '<input id="filter" placeholder="filter (substring of task/exp)...">'
        '<label>source:</label>'
        '<select id="srcfilter" style="background:#2a2a2a;color:#e0e0e0;border:1px solid #444;border-radius:4px;padding:5px">'
        '<option value="">(all)</option>'
        f'{src_options}'
        '</select>'
        '<label>sort by:</label>'
        '<select id="sortby" style="background:#2a2a2a;color:#e0e0e0;border:1px solid #444;border-radius:4px;padding:5px">'
        '<option value="">(input order)</option>'
        '<option value="norm">||offset|| (worst first)</option>'
        '</select>'
        '</div>'
    )
    parts.append('<div class="grid">')
    for e in entries:
        parts.append(build_card(e, annotated_root))
    parts.append('</div>')
    parts.append(HTML_TAIL)
    out_path.write_text("".join(parts))


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--videos_root",
                    help="Raw videos root (e.g. /viscam/.../videos_0102). "
                         "Sibling <videos_root>_annotated/ is browsed.")
    g.add_argument("--annotated_root",
                    help="Direct path to a videos_*_annotated tree (e.g. a "
                         "local rsync of the cluster output).")
    ap.add_argument("--out", type=str, default=None,
                    help="HTML output path. Default: <annotated_root>/_pose_offset_diag_index.html")
    ap.add_argument("--include_no_summary", action="store_true",
                    help="Also list exps that have a debug/pose_offset_diag/ directory "
                         "but no summary.json. Default OFF.")
    ap.add_argument("--pose_sources", nargs="+",
                    default=["ob_in_cam", "merged"],
                    help="Which pose source subdirs to scan. Each maps to a "
                         "diag dir name: ob_in_cam→pose_offset_diag, "
                         "<other>→pose_offset_diag_<other>. "
                         "Default: ob_in_cam merged.")
    ap.add_argument("--serve", action="store_true",
                    help="After writing the index, serve <annotated_root>/ over HTTP.")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    args = ap.parse_args()

    if args.videos_root:
        videos_root = Path(args.videos_root).resolve()
        annotated_root = annotated_root_for_videos(videos_root)
    else:
        annotated_root = Path(args.annotated_root).resolve()
        videos_root = None

    if not annotated_root.is_dir():
        print(f"Error: annotated root does not exist: {annotated_root}", file=sys.stderr)
        sys.exit(1)

    out_path = (Path(args.out).resolve() if args.out
                  else annotated_root / "_pose_offset_diag_index.html")

    entries = gather_pose_offset_entries(
        annotated_root, only_with_summary=(not args.include_no_summary),
        pose_sources=tuple(args.pose_sources),
    )
    if not entries:
        print(f"No exps with pose_offset_diag found under {annotated_root}.")
        print(f"  Run:  python debug/diaognose_pose_offset.py "
              f"--videos_root {videos_root or '<videos_root>'}")
        sys.exit(1)

    write_index(annotated_root, entries, out_path)
    print(f"[OK] wrote {out_path}")
    print(f"     {len(entries)} exps total, "
          f"{sum(1 for e in entries if e['frames'])} have rendered frames")

    if args.serve:
        os.chdir(str(annotated_root))
        rel_index = out_path.relative_to(annotated_root)
        host = args.host; port = args.port
        print()
        print(f"Serving {annotated_root} at http://{host}:{port}")
        print(f"Index : http://{host}:{port}/{rel_index}")
        try:
            local_ip = socket.gethostbyname(socket.gethostname())
            print(f"        (or http://{local_ip}:{port}/{rel_index})")
        except Exception:
            pass
        print(f"From your laptop: ssh -N -L {port}:localhost:{port} <user>@<this_host>")
        print()
        from http.server import SimpleHTTPRequestHandler
        from socketserver import ThreadingTCPServer
        ThreadingTCPServer.allow_reuse_address = True
        try:
            with ThreadingTCPServer((host, port), SimpleHTTPRequestHandler) as srv:
                srv.serve_forever()
        except KeyboardInterrupt:
            print("\n[stopped]")


if __name__ == "__main__":
    main()
