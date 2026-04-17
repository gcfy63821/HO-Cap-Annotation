#!/usr/bin/env python
"""
Interactive slider-based manual camera alignment.

Left panel: pick the cam to edit, six sliders (tx/ty/tz/rx/ry/rz), reset/save.
Right panel: live 3D view of all cams; the selected cam is highlighted.
Rotation slider pivots around the cam's own centroid so the cloud doesn't fly off.
Output: an "_slider_aligned.yaml" next to the input extrinsic file + a merged PLY.

Usage:
  python tools/00-4_align_cameras_slider.py \\
    --cached_pc  /path/to/cached_pc \\
    --extrinsic_file /path/to/realsense_calibration.yaml \\
    [--initial_yaml /path/to/..._global_aligned.yaml]   # start from a prior alignment
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import yaml


# ---------- YAML helpers (same format as 00-0 / 00-3) ----------
def load_extrinsics(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and "extrinsics" in data:
        return [v for k, v in data["extrinsics"].items() if not k.startswith("tag_")]
    if isinstance(data, list):
        return sorted(data, key=lambda x: x.get("camera_id", 0))
    raise ValueError(f"Unsupported YAML format: {path}")


# ---------- rigid transform helpers ----------
def euler_deg_to_R(rx, ry, rz):
    """ZYX Euler (rz applied first, then ry, then rx) in degrees."""
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def manual_transform(tx, ty, tz, rx, ry, rz, pivot):
    """Rotate about pivot in world, then translate."""
    R = euler_deg_to_R(rx, ry, rz)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx, ty, tz]) + pivot - R @ pivot
    return T


# ---------- palette ----------
PALETTE = np.array([
    [0.90, 0.10, 0.10], [0.10, 0.70, 0.10], [0.10, 0.40, 0.90],
    [0.95, 0.75, 0.10], [0.70, 0.20, 0.80], [0.10, 0.75, 0.75],
    [0.95, 0.50, 0.15], [0.55, 0.55, 0.55], [0.80, 0.30, 0.50],
    [0.30, 0.60, 0.30], [0.20, 0.20, 0.70], [0.70, 0.70, 0.20],
])


class SliderAlignApp:
    def __init__(self, cached_pc_dir, extrinsic_file, initial_yaml, out_path):
        self.cached_pc_dir = Path(cached_pc_dir)
        self.extrinsic_file = Path(extrinsic_file)
        self.out_path = Path(out_path) if out_path else self.extrinsic_file.parent
        self.out_path.mkdir(parents=True, exist_ok=True)

        self.cams = load_extrinsics(self.extrinsic_file)
        self.n = len(self.cams)

        # load cached PCs (already in the world frame baked by extrinsic_file)
        self.raw_pcs = []
        for i in range(self.n):
            p = self.cached_pc_dir / f"cam{i}_uncropped.ply"
            if not p.exists():
                raise FileNotFoundError(p)
            pc = o3d.io.read_point_cloud(str(p))
            self.raw_pcs.append(pc)

        # If an initial (already-aligned) yaml is given, compute the pre-applied
        # correction per cam so the viewer starts from that state. User's manual
        # adjustment composes on top.
        self.T_init = [np.eye(4) for _ in range(self.n)]
        if initial_yaml:
            init_cams = load_extrinsics(Path(initial_yaml))
            if len(init_cams) != self.n:
                print(f"[WARN] initial_yaml has {len(init_cams)} cams, input has {self.n}; "
                      "falling back to identity inits")
            else:
                for i in range(self.n):
                    orig = np.array(self.cams[i]["transformation"]).reshape(4, 4)
                    new = np.array(init_cams[i]["transformation"]).reshape(4, 4)
                    # cached PCs are in `orig` world; to reach `new` world we apply
                    # T_init = new @ inv(orig)
                    self.T_init[i] = new @ np.linalg.inv(orig)
                # sync cams metadata to initial_yaml so save emits full correction
                self.cams = init_cams

        # slider state per cam (in degrees / meters) — starts at zero
        self.state = [dict(tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0)
                      for _ in range(self.n)]

        # per-cam pivot = centroid in post-init frame (so rotation feels natural)
        self.pivots = []
        for i in range(self.n):
            pc_t = copy.deepcopy(self.raw_pcs[i])
            pc_t.transform(self.T_init[i])
            pts = np.asarray(pc_t.points)
            self.pivots.append(pts.mean(axis=0) if len(pts) else np.zeros(3))

        self.selected = 0
        self._build_gui()
        self._add_all_clouds()
        self._fit_camera()

    # ---------- UI ----------
    def _build_gui(self):
        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "Slider Alignment", 1500, 950
        )
        em = self.window.theme.font_size

        # 3D scene
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0.08, 0.08, 0.08, 1.0])
        self.scene.scene.show_axes(True)

        # left panel
        self.panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        self.panel.add_child(gui.Label("Camera"))
        self.cam_sel = gui.Combobox()
        for i in range(self.n):
            self.cam_sel.add_item(f"cam{i}")
        self.cam_sel.set_on_selection_changed(self._on_cam_change)
        self.panel.add_child(self.cam_sel)

        self.panel.add_fixed(0.5 * em)
        self.panel.add_child(gui.Label("Translation (m)"))
        self.s_tx = self._mk_slider(-0.5, 0.5, "tx")
        self.s_ty = self._mk_slider(-0.5, 0.5, "ty")
        self.s_tz = self._mk_slider(-0.3, 0.3, "tz")

        self.panel.add_fixed(0.5 * em)
        self.panel.add_child(gui.Label("Rotation (deg, ZYX)"))
        self.s_rx = self._mk_slider(-30.0, 30.0, "rx")
        self.s_ry = self._mk_slider(-30.0, 30.0, "ry")
        self.s_rz = self._mk_slider(-180.0, 180.0, "rz")

        self.panel.add_fixed(em)
        btn_reset = gui.Button("Reset this cam")
        btn_reset.set_on_clicked(self._on_reset)
        self.panel.add_child(btn_reset)

        btn_reset_all = gui.Button("Reset all")
        btn_reset_all.set_on_clicked(self._on_reset_all)
        self.panel.add_child(btn_reset_all)

        self.hl_toggle = gui.Checkbox("Highlight selected")
        self.hl_toggle.checked = True
        self.hl_toggle.set_on_checked(lambda _: self._refresh_colors())
        self.panel.add_child(self.hl_toggle)

        self.hide_others = gui.Checkbox("Hide non-selected")
        self.hide_others.checked = False
        self.hide_others.set_on_checked(lambda _: self._refresh_visibility())
        self.panel.add_child(self.hide_others)

        self.panel.add_fixed(em)
        btn_save = gui.Button("Save  (YAML + merged PLY)")
        btn_save.set_on_clicked(self._on_save)
        self.panel.add_child(btn_save)

        self.status = gui.Label("")
        self.panel.add_child(self.status)

        self.window.add_child(self.scene)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._on_layout)

    def _mk_slider(self, lo, hi, key):
        s = gui.Slider(gui.Slider.DOUBLE)
        s.set_limits(lo, hi)
        s.double_value = 0.0
        s.set_on_value_changed(lambda v, k=key: self._on_slider(k, v))
        self.panel.add_child(gui.Label(key))
        self.panel.add_child(s)
        return s

    def _on_layout(self, ctx):
        r = self.window.content_rect
        panel_w = 22 * ctx.theme.font_size
        self.scene.frame = gui.Rect(r.x, r.y, r.width - panel_w, r.height)
        self.panel.frame = gui.Rect(r.get_right() - panel_w, r.y, panel_w, r.height)

    # ---------- scene ----------
    def _mat(self, color=None, pt_size=3.0):
        m = rendering.MaterialRecord()
        m.shader = "defaultUnlit"
        m.point_size = pt_size
        if color is not None:
            m.base_color = [color[0], color[1], color[2], 1.0]
        return m

    def _add_all_clouds(self):
        for i in range(self.n):
            name = f"cam{i}"
            # preserve original colors but also keep a uniform copy for highlight mode
            pc_col = copy.deepcopy(self.raw_pcs[i])
            if not pc_col.has_colors():
                pc_col.paint_uniform_color(PALETTE[i % len(PALETTE)].tolist())
            self.scene.scene.add_geometry(name, pc_col, self._mat())
            self.scene.scene.set_geometry_transform(name, self.T_init[i])
        self._refresh_colors()

    def _refresh_colors(self):
        # swap materials: highlighted cam gets a bright uniform color, others keep
        # real rgb if possible (use unlit shader with base color tint).
        for i in range(self.n):
            name = f"cam{i}"
            if self.hl_toggle.checked and i == self.selected:
                m = self._mat(color=[1.0, 0.2, 0.2], pt_size=5.0)
            else:
                m = self._mat(pt_size=2.5)
            self.scene.scene.modify_geometry_material(name, m)

    def _refresh_visibility(self):
        for i in range(self.n):
            name = f"cam{i}"
            visible = (not self.hide_others.checked) or (i == self.selected)
            self.scene.scene.show_geometry(name, visible)

    def _apply_cam_transform(self, i):
        st = self.state[i]
        T_manual = manual_transform(
            st["tx"], st["ty"], st["tz"],
            st["rx"], st["ry"], st["rz"],
            self.pivots[i],
        )
        self.scene.scene.set_geometry_transform(
            f"cam{i}", T_manual @ self.T_init[i]
        )

    def _fit_camera(self):
        mn = np.array([np.inf, np.inf, np.inf])
        mx = np.array([-np.inf, -np.inf, -np.inf])
        for i in range(self.n):
            if len(self.raw_pcs[i].points) == 0:
                continue
            aabb = self.raw_pcs[i].get_axis_aligned_bounding_box()
            mn = np.minimum(mn, np.asarray(aabb.min_bound))
            mx = np.maximum(mx, np.asarray(aabb.max_bound))
        if not np.all(np.isfinite(mn)):
            mn, mx = np.array([-1.0, -1.0, -0.5]), np.array([1.0, 1.0, 1.5])
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=mn.tolist(), max_bound=mx.tolist()
        )
        self.scene.setup_camera(60, bbox, bbox.get_center())

    # ---------- callbacks ----------
    def _sync_sliders_from_state(self):
        st = self.state[self.selected]
        # block callbacks while we set values
        for key, s in [("tx", self.s_tx), ("ty", self.s_ty), ("tz", self.s_tz),
                       ("rx", self.s_rx), ("ry", self.s_ry), ("rz", self.s_rz)]:
            s.double_value = st[key]

    def _on_cam_change(self, _text, _idx):
        self.selected = _idx
        self._sync_sliders_from_state()
        self._refresh_colors()
        self._refresh_visibility()
        self._set_status(f"editing cam{self.selected}")

    def _on_slider(self, key, value):
        self.state[self.selected][key] = value
        self._apply_cam_transform(self.selected)

    def _on_reset(self):
        self.state[self.selected] = dict(tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0)
        self._sync_sliders_from_state()
        self._apply_cam_transform(self.selected)
        self._set_status(f"reset cam{self.selected}")

    def _on_reset_all(self):
        for i in range(self.n):
            self.state[i] = dict(tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0)
            self._apply_cam_transform(i)
        self._sync_sliders_from_state()
        self._set_status("reset all cams")

    def _on_save(self):
        out_yaml = self.extrinsic_file.parent / f"{self.extrinsic_file.stem}_slider_aligned.yaml"
        merged = o3d.geometry.PointCloud()
        updated = []
        for i in range(self.n):
            st = self.state[i]
            T_manual = manual_transform(
                st["tx"], st["ty"], st["tz"],
                st["rx"], st["ry"], st["rz"],
                self.pivots[i],
            )
            T_total = T_manual @ self.T_init[i]
            # write: new_ext = T_total @ ext_orig   (ext_orig = extrinsic_file cam)
            # NOTE: if initial_yaml was given, self.cams already reflects initial_yaml's
            # transformation, so to stay self-consistent we recompute from the original file.
            with open(self.extrinsic_file, "r") as f:
                _orig_raw = yaml.safe_load(f)
            orig_cams = load_extrinsics(self.extrinsic_file)
            ext_orig = np.array(orig_cams[i]["transformation"]).reshape(4, 4)
            # T_init was new @ inv(orig); so T_total @ ext_orig = T_manual @ new
            new_ext = T_total @ ext_orig
            cam = self.cams[i]
            updated.append({
                "camera_id": cam.get("camera_id", i),
                "serial_number": cam["serial_number"],
                "transformation": new_ext.tolist(),
                "color_intrinsic_matrix": cam["color_intrinsic_matrix"],
                "depth_intrinsic_matrix": cam["depth_intrinsic_matrix"],
            })
            pc = copy.deepcopy(self.raw_pcs[i])
            pc.transform(T_total)
            merged += pc

        with open(out_yaml, "w") as f:
            yaml.dump(updated, f, default_flow_style=False, sort_keys=False)
        merged_ply = self.out_path / f"{self.extrinsic_file.stem}_slider_aligned.ply"
        o3d.io.write_point_cloud(str(merged_ply), merged)
        self._set_status(f"saved: {out_yaml.name} + {merged_ply.name}")
        print(f"[SAVED] {out_yaml}")
        print(f"[SAVED] {merged_ply}")

    def _set_status(self, msg):
        self.status.text = msg


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Slider-based interactive multi-camera point cloud alignment."
    )
    ap.add_argument("--cached_pc", "--cached_pc_dir", dest="cached_pc", required=True)
    ap.add_argument("--extrinsic_file", required=True)
    ap.add_argument("--initial_yaml", default=None,
                    help="start from a previously-aligned YAML (e.g. 00-3 output); "
                         "manual tweaks compose on top of it")
    ap.add_argument("--out_path", default=None,
                    help="output directory for merged PLY (default: same dir as extrinsic)")
    args = ap.parse_args()

    app = SliderAlignApp(
        cached_pc_dir=args.cached_pc,
        extrinsic_file=args.extrinsic_file,
        initial_yaml=args.initial_yaml,
        out_path=args.out_path,
    )
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
