#!/usr/bin/env python3
"""
Web-based interactive multi-camera extrinsic adjuster (Viser).

Same math as tools/00-4_align_cameras_slider.py but GUI is served over the
network so you can run it on the cluster and drive it from your laptop
browser via SSH port-forwarding:

    # cluster side (inside a Slurm job or on a login node with GPU not required):
    conda activate hocap-annotation
    python scripts/manual_align_viser.py \
        --cached_pc /viscam/.../realsense_calibrate_0101/cached_pc \
        --extrinsic_file /viscam/.../realsense_calibration_0101.yaml \
        --initial_yaml  /viscam/.../realsense_calibration_0101_global_aligned.yaml \
        --port 8080

    # laptop side:
    ssh -L 8080:<cluster-node>:8080 <user>@<login-host>
    open http://localhost:8080   # in your browser

GUI:
  - dropdown to pick which camera to edit
  - 6 sliders (tx/ty/tz in meters, rx/ry/rz in degrees)
  - per-cam visibility checkboxes; optional per-cam solid coloring
  - buttons: Reset selected / Reset all / Save
  - Save writes <stem>_slider_aligned.yaml + <stem>_slider_aligned.ply next to
    the input extrinsic file (same as 00-4).

Math (matches 00-4):
  cached PC is in the world frame baked by --extrinsic_file.
  T_init[i] brings cam i from original-world to initial_yaml-world (identity if
  no --initial_yaml given). Point cloud positions uploaded once = T_init[i] @ raw.
  User slider state produces T_manual[i] (rotate about cloud centroid, then
  translate). Each cam i is hung under a Viser frame whose pose is T_manual[i],
  so moving sliders only updates the frame — no re-upload of points.
  On save, new_ext = T_manual[i] @ T_init[i] @ ext_orig[i].
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import open3d as o3d
import viser
import yaml


PALETTE = np.array([
    [0.90, 0.10, 0.10], [0.10, 0.70, 0.10], [0.10, 0.40, 0.90],
    [0.95, 0.75, 0.10], [0.70, 0.20, 0.80], [0.10, 0.75, 0.75],
    [0.95, 0.50, 0.15], [0.55, 0.55, 0.55], [0.80, 0.30, 0.50],
    [0.30, 0.60, 0.30], [0.20, 0.20, 0.70], [0.70, 0.70, 0.20],
])


# ---------- YAML helpers (same format as 00-0/00-3/00-4) ----------
def load_extrinsics(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and "extrinsics" in data:
        return [v for k, v in data["extrinsics"].items() if not k.startswith("tag_")]
    if isinstance(data, list):
        return sorted(data, key=lambda x: x.get("camera_id", 0))
    raise ValueError(f"Unsupported YAML format: {path}")


# ---------- rigid transform helpers (match 00-4 conventions) ----------
def euler_deg_to_R(rx, ry, rz):
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


def rotmat_to_wxyz(R):
    # R: (3,3). Returns (w,x,y,z) quaternion.
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def main():
    ap = argparse.ArgumentParser(description="Viser-based interactive camera-extrinsic adjuster")
    ap.add_argument("--cached_pc", "--cached_pc_dir", dest="cached_pc", required=True,
                    help="folder with cam{i}_uncropped.ply (output of 00-0)")
    ap.add_argument("--extrinsic_file", required=True,
                    help="original calibration YAML (the one the cached PCs were built with)")
    ap.add_argument("--initial_yaml", default=None,
                    help="optional prior alignment YAML to start from (e.g. *_global_aligned.yaml)")
    ap.add_argument("--out_path", default=None,
                    help="output directory (default: extrinsic_file's directory)")
    ap.add_argument("--voxel", type=float, default=0.004,
                    help="downsampling voxel size in meters for display (0 = no downsample)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--max_trans", type=float, default=0.30,
                    help="translation slider half-range in meters")
    ap.add_argument("--max_rot", type=float, default=20.0,
                    help="rotation slider half-range in degrees")
    args = ap.parse_args()

    cached_dir = Path(args.cached_pc)
    extrinsic_file = Path(args.extrinsic_file)
    out_path = Path(args.out_path) if args.out_path else extrinsic_file.parent
    out_path.mkdir(parents=True, exist_ok=True)

    cams = load_extrinsics(extrinsic_file)
    n = len(cams)
    print(f"[INFO] {n} cams from {extrinsic_file}")

    # Load cached PCs (in original world frame).
    raw_pcs = []
    for i in range(n):
        p = cached_dir / f"cam{i}_uncropped.ply"
        if not p.exists():
            raise FileNotFoundError(p)
        pc = o3d.io.read_point_cloud(str(p))
        if args.voxel > 0:
            pc = pc.voxel_down_sample(args.voxel)
        raw_pcs.append(pc)
        print(f"  cam{i}: {len(pc.points)} pts (voxel={args.voxel*1000:.1f}mm)")

    # T_init: bring each cam from original-world to initial_yaml-world.
    T_init = [np.eye(4) for _ in range(n)]
    if args.initial_yaml:
        init_cams = load_extrinsics(Path(args.initial_yaml))
        if len(init_cams) != n:
            print(f"[WARN] initial_yaml has {len(init_cams)} cams vs {n} in extrinsic_file "
                  "— falling back to identity init")
        else:
            for i in range(n):
                orig = np.array(cams[i]["transformation"]).reshape(4, 4)
                new = np.array(init_cams[i]["transformation"]).reshape(4, 4)
                T_init[i] = new @ np.linalg.inv(orig)

    # Points baked into initial world frame. Pivots = centroid in that frame.
    init_points = []
    init_colors = []
    pivots = []
    for i in range(n):
        pc = copy.deepcopy(raw_pcs[i])
        pc.transform(T_init[i])
        pts = np.asarray(pc.points, dtype=np.float32)
        cols = np.asarray(pc.colors, dtype=np.float32) if pc.has_colors() else None
        init_points.append(pts)
        init_colors.append(cols)
        pivots.append(pts.mean(axis=0) if len(pts) else np.zeros(3, dtype=np.float32))

    # State: 6 slider values per cam.
    state = [dict(tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0) for _ in range(n)]

    # ---- Viser server ----
    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"[INFO] viser at http://{args.host}:{args.port}  "
          f"(SSH-forward this port to view from your laptop)")

    server.scene.add_frame("/world_axes", axes_length=0.1, axes_radius=0.002)

    # each cam hangs under /cam_{i} (a movable frame); the pointcloud is static under it.
    cam_frames = []
    pc_handles = []
    for i in range(n):
        color_solid = tuple((PALETTE[i % len(PALETTE)] * 255).astype(int).tolist())
        f = server.scene.add_frame(
            f"/cam_{i}", wxyz=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0),
            axes_length=0.0, axes_radius=0.0,
        )
        cam_frames.append(f)
        # true-colored if available, else solid palette color
        if init_colors[i] is not None:
            cols = (np.clip(init_colors[i], 0, 1) * 255).astype(np.uint8)
        else:
            cols = np.tile(np.array(color_solid, dtype=np.uint8), (len(init_points[i]), 1))
        h = server.scene.add_point_cloud(
            f"/cam_{i}/pc",
            points=init_points[i],
            colors=cols,
            point_size=0.003,
        )
        pc_handles.append(h)

    # ---- GUI ----
    server.gui.add_markdown("### Camera extrinsic adjuster")
    cam_names = [f"cam{i}" for i in range(n)]
    cam_dd = server.gui.add_dropdown("Edit cam", options=cam_names, initial_value=cam_names[0])

    tx = server.gui.add_slider("tx  (m)",  -args.max_trans, args.max_trans, 0.001, 0.0)
    ty = server.gui.add_slider("ty  (m)",  -args.max_trans, args.max_trans, 0.001, 0.0)
    tz = server.gui.add_slider("tz  (m)",  -args.max_trans, args.max_trans, 0.001, 0.0)
    rx = server.gui.add_slider("rx  (°)",  -args.max_rot,   args.max_rot,   0.1,   0.0)
    ry = server.gui.add_slider("ry  (°)",  -args.max_rot,   args.max_rot,   0.1,   0.0)
    rz = server.gui.add_slider("rz  (°)",  -args.max_rot,   args.max_rot,   0.1,   0.0)

    server.gui.add_markdown("---")
    btn_reset = server.gui.add_button("Reset this cam")
    btn_reset_all = server.gui.add_button("Reset ALL cams")
    btn_save = server.gui.add_button("Save YAML + merged PLY")

    server.gui.add_markdown("---")
    server.gui.add_markdown("**Point display**")
    pt_size = server.gui.add_slider("Point size", 0.0005, 0.01, 0.0005, 0.003)
    show_solid = server.gui.add_checkbox("Solid per-cam colors", initial_value=False)

    server.gui.add_markdown("---")
    server.gui.add_markdown("**Visibility**")
    vis_toggles = []
    for i in range(n):
        cb = server.gui.add_checkbox(f"cam{i}", initial_value=True)
        vis_toggles.append(cb)

    status = server.gui.add_markdown("*ready*")

    # ---- helpers ----
    def _apply_cam_transform(i):
        s = state[i]
        T = manual_transform(s["tx"], s["ty"], s["tz"], s["rx"], s["ry"], s["rz"],
                              pivots[i])
        cam_frames[i].wxyz = tuple(rotmat_to_wxyz(T[:3, :3]).tolist())
        cam_frames[i].position = tuple(T[:3, 3].tolist())

    def _sync_sliders_from_state():
        i = cam_names.index(cam_dd.value)
        s = state[i]
        # mute callbacks while we update
        tx.value = s["tx"]
        ty.value = s["ty"]
        tz.value = s["tz"]
        rx.value = s["rx"]
        ry.value = s["ry"]
        rz.value = s["rz"]

    def _set_status(msg):
        status.content = f"`{msg}`"
        print(f"[GUI] {msg}")

    # initial frame poses (all identity)
    for i in range(n):
        _apply_cam_transform(i)

    # ---- handlers ----
    @cam_dd.on_update
    def _(_evt):
        _sync_sliders_from_state()
        _set_status(f"editing {cam_dd.value}")

    def _slider_changed(key):
        i = cam_names.index(cam_dd.value)
        state[i][key] = float({"tx": tx, "ty": ty, "tz": tz,
                                "rx": rx, "ry": ry, "rz": rz}[key].value)
        _apply_cam_transform(i)

    @tx.on_update
    def _(_evt): _slider_changed("tx")
    @ty.on_update
    def _(_evt): _slider_changed("ty")
    @tz.on_update
    def _(_evt): _slider_changed("tz")
    @rx.on_update
    def _(_evt): _slider_changed("rx")
    @ry.on_update
    def _(_evt): _slider_changed("ry")
    @rz.on_update
    def _(_evt): _slider_changed("rz")

    @btn_reset.on_click
    def _(_evt):
        i = cam_names.index(cam_dd.value)
        state[i] = dict(tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0)
        _apply_cam_transform(i)
        _sync_sliders_from_state()
        _set_status(f"reset {cam_dd.value}")

    @btn_reset_all.on_click
    def _(_evt):
        for i in range(n):
            state[i] = dict(tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0)
            _apply_cam_transform(i)
        _sync_sliders_from_state()
        _set_status("reset all cams")

    @pt_size.on_update
    def _(_evt):
        for h in pc_handles:
            h.point_size = float(pt_size.value)

    @show_solid.on_update
    def _(_evt):
        for i, h in enumerate(pc_handles):
            if show_solid.value or init_colors[i] is None:
                col = (PALETTE[i % len(PALETTE)] * 255).astype(np.uint8)
                h.colors = np.tile(col, (len(init_points[i]), 1))
            else:
                h.colors = (np.clip(init_colors[i], 0, 1) * 255).astype(np.uint8)

    for idx, cb in enumerate(vis_toggles):
        def _mk(i, cb=cb):
            @cb.on_update
            def _(_evt):
                pc_handles[i].visible = bool(cb.value)
        _mk(idx)

    @btn_save.on_click
    def _(_evt):
        out_yaml = extrinsic_file.parent / f"{extrinsic_file.stem}_slider_aligned.yaml"
        out_ply = out_path / f"{extrinsic_file.stem}_slider_aligned.ply"
        # Always derive from the ORIGINAL extrinsic file, regardless of initial_yaml.
        orig_cams = load_extrinsics(extrinsic_file)
        updated = []
        merged = o3d.geometry.PointCloud()
        for i in range(n):
            s = state[i]
            T_manual = manual_transform(s["tx"], s["ty"], s["tz"],
                                         s["rx"], s["ry"], s["rz"], pivots[i])
            T_total = T_manual @ T_init[i]
            ext_orig = np.array(orig_cams[i]["transformation"]).reshape(4, 4)
            new_ext = T_total @ ext_orig
            updated.append({
                "camera_id": orig_cams[i].get("camera_id", i),
                "serial_number": orig_cams[i]["serial_number"],
                "transformation": new_ext.tolist(),
                "color_intrinsic_matrix": orig_cams[i]["color_intrinsic_matrix"],
                "depth_intrinsic_matrix": orig_cams[i]["depth_intrinsic_matrix"],
            })
            pc = copy.deepcopy(raw_pcs[i])
            pc.transform(T_total)
            merged += pc

        with open(out_yaml, "w") as f:
            yaml.dump(updated, f, default_flow_style=False, sort_keys=False)
        o3d.io.write_point_cloud(str(out_ply), merged)
        _set_status(f"SAVED  {out_yaml.name}  +  {out_ply.name}")
        print(f"[SAVED] {out_yaml}")
        print(f"[SAVED] {out_ply}")

    # block forever
    import time
    print("[INFO] Ctrl-C to stop")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[INFO] stopped")


if __name__ == "__main__":
    main()
