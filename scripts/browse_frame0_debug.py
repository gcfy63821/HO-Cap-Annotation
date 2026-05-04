#!/usr/bin/env python3
"""
Browse frame-0 debug montages produced by generate_frame0_debug.py.

Two modes:

1) HTML index (default; zero deps, works offline):
     python scripts/browse_frame0_debug.py --videos_root <path>
   Writes <videos_root>_annotated/_frame0_debug_index.html with a thumbnail
   grid (one cell per <task>/<exp>) + click-to-zoom. Open in any browser.

2) Live HTTP server (good when running on cluster + SSH-tunneling):
     python scripts/browse_frame0_debug.py --videos_root <path> --serve
   Serves <videos_root>_annotated/ on http://0.0.0.0:8090 and prints the
   index URL.

Each cell shows:
    <task>/<exp>            (label)
    [thumbnail PNG]         (frame0_debug.png inline)
    Status hints derived from filename presence:
      - missing PNG → red "no debug yet — run generate_frame0_debug.py"
"""

import argparse
import html
import os
import socket
import sys
from pathlib import Path


def annotated_root_for_videos(videos_root: Path) -> Path:
    return videos_root.parent / f"{videos_root.name}_annotated"


def gather_debug_pngs(annotated_root: Path, only_done=True):
    """Return list of {task, exp, png_path, exists, merger_done, ...}.

    only_done=True (default) → only include exps where Stage 5 finished
    (processed/fd_pose_solver/fd_poses_merged_fixed.npy exists). Matches the
    generator's default filter so the browser index doesn't show empty cells
    for never-tracked exps.
    """
    out = []
    if not annotated_root.is_dir():
        return out
    for task_dir in sorted(annotated_root.iterdir()):
        if not task_dir.is_dir(): continue
        for exp_dir in sorted(task_dir.iterdir()):
            if not exp_dir.is_dir(): continue
            merger_done = (exp_dir / "processed/fd_pose_solver/fd_poses_merged_fixed.npy").exists()
            if only_done and not merger_done:
                continue
            png = exp_dir / "debug" / "frame0_debug.png"
            out.append({
                "task": task_dir.name,
                "exp": exp_dir.name,
                "png_path": png,
                "rel_path": png.relative_to(annotated_root) if png.exists() else None,
                "exists": png.exists(),
                "size_kb": png.stat().st_size // 1024 if png.exists() else 0,
                "merger_done": merger_done,
                "hand_done":   (exp_dir / "result_hand_optimized.pkl").exists(),
                "joint_done":  (exp_dir / "processed/joint_pose_solver/poses_o.npy").exists(),
                "masks_h5":    (exp_dir / "tool_masks/masks.h5").exists(),
            })
    return out


HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Frame-0 debug browser — {root_name}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         background: #1a1a1a; color: #e0e0e0; margin: 0; padding: 16px; }}
  h1 {{ font-size: 18px; margin: 0 0 12px; }}
  .stats {{ font-size: 13px; color: #aaa; margin-bottom: 18px; }}
  .filter {{ margin: 8px 0 16px; }}
  .filter input {{ padding: 6px 10px; width: 360px; background: #2a2a2a;
                   color: #e0e0e0; border: 1px solid #444; border-radius: 4px; }}
  /* Wider cards (480 → fewer per row) so long exp names breathe. */
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
           gap: 14px; }}
  .card {{ background: #242424; border-radius: 6px; overflow: hidden;
           border: 1px solid #333; }}
  .card .label {{ padding: 10px 12px; font-size: 11px; font-family: monospace;
                  border-bottom: 1px solid #333; line-height: 1.5;
                  /* allow long names to break anywhere instead of overflowing */
                  word-break: break-all;
                  overflow-wrap: anywhere; }}
  .card .label .task {{ color: #4af; font-weight: 600; display: block; }}
  .card .label .exp  {{ color: #fff; font-size: 12px;
                        /* exp on its own line so it fully visible */
                        display: block; margin-top: 2px;
                        /* select-on-click for easy copy */
                        user-select: all; }}
  .card .label .copy_btn {{ display: inline-block; margin-left: 6px;
                            padding: 1px 6px; font-size: 10px;
                            background: #333; color: #aaa;
                            border-radius: 3px; cursor: pointer;
                            border: 1px solid #444; }}
  .card .label .copy_btn:hover {{ background: #555; color: #fff; }}
  .card .label .badges {{ margin-top: 6px; }}
  .badge {{ display: inline-block; font-size: 10px; padding: 1px 6px;
            border-radius: 3px; margin-right: 4px; background: #333; color: #888; }}
  .badge.ok {{ background: #1e4d2b; color: #8f8; }}
  .badge.miss {{ background: #4d1e1e; color: #f88; }}
  .card img {{ display: block; width: 100%; height: auto; cursor: zoom-in; }}
  .card .miss_png {{ background: #4d1e1e; color: #fcc; padding: 18px;
                     text-align: center; font-size: 12px; }}
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
  document.querySelectorAll('.card img').forEach(img => {
    img.addEventListener('click', e => {
      const lb = document.getElementById('lb');
      document.getElementById('lbimg').src = img.src;
      document.getElementById('lblabel').textContent =
        img.dataset.label || img.src.split('/').slice(-3, -1).join('/');
      lb.classList.add('visible');
    });
  });
  const filter = document.getElementById('filter');
  filter && filter.addEventListener('input', e => {
    const q = e.target.value.toLowerCase();
    document.querySelectorAll('.card').forEach(c => {
      c.style.display = c.dataset.label.toLowerCase().includes(q) ? '' : 'none';
    });
  });
</script>
</body></html>
"""


def write_index(annotated_root: Path, entries, out_path: Path):
    n_total = len(entries)
    n_have = sum(1 for e in entries if e["exists"])
    n_done_merge = sum(1 for e in entries if e["merger_done"])
    n_done_hand  = sum(1 for e in entries if e["hand_done"])
    parts = [HTML_HEAD.format(root_name=html.escape(annotated_root.name))]
    parts.append(f'<h1>Frame-0 debug browser <span style="color:#888">— {html.escape(annotated_root.name)}</span></h1>')
    parts.append(f'<div class="stats">'
                 f'{n_have}/{n_total} debug PNGs &nbsp;·&nbsp; '
                 f'{n_done_merge}/{n_total} fd merger done &nbsp;·&nbsp; '
                 f'{n_done_hand}/{n_total} hand pkl done'
                 f'</div>')
    parts.append('<div class="filter"><input id="filter" placeholder="filter (substring of task/exp)..."></div>')
    parts.append('<div class="grid">')

    for e in entries:
        label = f'{e["task"]}/{e["exp"]}'
        label_attr = html.escape(label, quote=True)
        badges = []
        badges.append(('masks.h5', 'ok' if e["masks_h5"]    else 'miss'))
        badges.append(('fd merge', 'ok' if e["merger_done"] else 'miss'))
        badges.append(('hand',     'ok' if e["hand_done"]   else 'miss'))
        badges.append(('joint',    'ok' if e["joint_done"]  else 'miss'))
        badges_html = ' '.join(
            f'<span class="badge {state}">{html.escape(name)}</span>'
            for name, state in badges
        )
        # title tooltip = full label (browsers show on hover, useful when the
        # exp name is super long).
        parts.append(f'<div class="card" data-label="{label_attr}" title="{label_attr}">')
        # Two lines: task on its own line, exp on its own line, both fully
        # visible (word-break: break-all in CSS handles long names).
        parts.append(
            f'<div class="label">'
            f'<span class="task">{html.escape(e["task"])}</span>'
            f'<span class="exp">{html.escape(e["exp"])}'
            f'<span class="copy_btn" onclick="copyText(this, &quot;{label_attr}&quot;); event.stopPropagation();">copy</span>'
            f'</span>'
            f'<div class="badges">{badges_html}</div>'
            f'</div>'
        )
        if e["exists"]:
            parts.append(
                f'<img src="{html.escape(str(e["rel_path"]))}" '
                f'data-label="{label_attr}" loading="lazy">'
            )
        else:
            parts.append('<div class="miss_png">no frame0_debug.png yet<br>'
                         '<span style="font-size:10px;opacity:.7">'
                         'run scripts/generate_frame0_debug.py</span></div>')
        parts.append('</div>')

    parts.append('</div>')
    parts.append(HTML_TAIL)

    out_path.write_text(''.join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos_root", required=True,
                    help="Raw videos root (e.g. /viscam/projects/robotool/data/videos_0102). "
                         "Sibling <videos_root>_annotated/ is browsed.")
    ap.add_argument("--out", type=str, default=None,
                    help="HTML output path. Default: <annotated_root>/_frame0_debug_index.html.")
    ap.add_argument("--serve", action="store_true",
                    help="Start an HTTP server rooted at <annotated_root>/ "
                         "after writing index.")
    ap.add_argument("--port", type=int, default=8090,
                    help="HTTP server port (only with --serve).")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--include_unfinished", action="store_true",
                    help="Also list exps that don't yet have "
                         "fd_poses_merged_fixed.npy. Default OFF — match "
                         "generate_frame0_debug.py and only show DONE exps.")
    args = ap.parse_args()

    videos_root = Path(args.videos_root).resolve()
    annotated_root = annotated_root_for_videos(videos_root)
    if not annotated_root.is_dir():
        print(f"Error: {annotated_root} does not exist. "
              f"Did annotation run on {videos_root}?")
        sys.exit(1)

    out_path = Path(args.out).resolve() if args.out else (annotated_root / "_frame0_debug_index.html")
    entries = gather_debug_pngs(annotated_root, only_done=(not args.include_unfinished))
    if not entries:
        which = "all" if args.include_unfinished else "DONE"
        print(f"Error: no {which} exps found under {annotated_root}")
        sys.exit(1)

    write_index(annotated_root, entries, out_path)
    print(f"[OK] wrote {out_path}")
    n_have = sum(1 for e in entries if e["exists"])
    print(f"     {n_have}/{len(entries)} exps have frame0_debug.png")
    print(f"     missing exps need:  python scripts/generate_frame0_debug.py --videos_root {videos_root}")

    if args.serve:
        os.chdir(str(annotated_root))
        rel_index = out_path.relative_to(annotated_root)
        host = args.host
        port = args.port
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
        # Use the stdlib server.
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
