#!/usr/bin/env python
"""
Headless wrapper around tools/00-3_align_cameras_global.py.

00-3 ends with three interactive o3d.visualization.draw_geometries windows that
hang on headless clusters. This wrapper monkey-patches draw_geometries to a
no-op before executing the real script, so the YAML + PLY outputs are produced
exactly as normal but no window is ever opened.

All CLI args are forwarded unchanged to 00-3.
"""
import runpy
import sys
from pathlib import Path

import open3d as o3d


def _noop(*args, **kwargs):
    name = kwargs.get("window_name", "<window>")
    print(f"[headless] skipping draw_geometries ({name})")


o3d.visualization.draw_geometries = _noop

_TARGET = Path(__file__).resolve().parent / "00-3_align_cameras_global.py"
if not _TARGET.exists():
    print(f"[ERROR] cannot find {_TARGET}", file=sys.stderr)
    sys.exit(2)

# Make the real script believe it was invoked directly.
sys.argv[0] = str(_TARGET)
runpy.run_path(str(_TARGET), run_name="__main__")
