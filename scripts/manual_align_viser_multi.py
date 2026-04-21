#!/usr/bin/env python3
"""
Multi-session version of manual_align_viser.py.

Auto-discovers every videos_XXXX/ under --data_root that has both
  - realsense_calibrate_*/cached_pc/cam*_uncropped.ply
  - realsense_calibrate_*/realsense_calibration_*.yaml   (the original)
and presents them as a dropdown + Prev/Next buttons at the top of the GUI,
so you can align them one after another without restarting the server.

Per-session manual adjustments are preserved in memory while you switch
between sessions (until you refresh the browser or restart the process).
Click Save to flush the CURRENT session to disk as
<stem>_slider_aligned.yaml + .ply.

Math matches manual_align_viser.py exactly.

Usage:
    conda activate hocap-annotation
    python scripts/manual_align_viser_multi.py \
        --data_root /viscam/projects/robotool/data \
        --port 8080

    # then SSH-forward 8080 and open http://localhost:8080
"""
import argparse
import copy
import re
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


# ---------- YAML / math helpers (shared with manual_align_viser.py) ----------
def load_extrinsics(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and "extrinsics" in data:
        return [v for k, v in data["extrinsics"].items() if not k.startswith("tag_")]
    if isinstance(data, list):
        return sorted(data, key=lambda x: x.get("camera_id", 0))
    raise ValueError(f"Unsupported YAML format: {path}")


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
    R = euler_deg_to_R(rx, ry, rz)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx, ty, tz]) + pivot - R @ pivot
    return T


def rotmat_to_wxyz(R):
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        ])
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return np.array([
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        ])
    if m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return np.array([
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        ])
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return np.array([
        (m[1, 0] - m[0, 1]) / s,
        (m[0, 2] + m[2, 0]) / s,
        (m[1, 2] + m[2, 1]) / s,
        0.25 * s,
    ])


# ---------- session discovery ----------
def discover_sessions(data_root: Path):
    sessions = []
    for vd in sorted(data_root.iterdir()):
        if not vd.is_dir() or not vd.name.startswith("videos_") or vd.name.endswith("_annotated"):
            continue
        cal_folders = list(vd.glob("realsense_calibrate_*"))
        if not cal_folders:
            continue
        cal_folder = cal_folders[0]
        cache_dir = cal_folder / "cached_pc"
        if not cache_dir.is_dir():
            continue
        if not list(cache_dir.glob("cam*_uncropped.ply")):
            continue
        # pick original yaml (exclude any *_aligned.yaml derived variants)
        yamls = [p for p in cal_folder.glob("realsense_calibration_*.yaml")
                 if not re.search(r"_(slider|global|manual)?_?aligned\.yaml$", p.name)]
        if not yamls:
            # fallback: any calibration yaml that isn't obviously derived
            yamls = [p for p in cal_folder.glob("realsense_calibration_*.yaml")
                     if "_aligned" not in p.name]
        if not yamls:
            continue
        orig_yaml = sorted(yamls)[0]
        # preferred initial = global_aligned if present, else slider_aligned
        init = None
        for cand in [
            cal_folder / f"{orig_yaml.stem}_global_aligned.yaml",
            cal_folder / f"{orig_yaml.stem}_slider_aligned.yaml",
        ]:
            if cand.exists():
                init = cand
                break
        sessions.append({
            "name": vd.name,
            "cached_pc": cache_dir,
            "extrinsic_file": orig_yaml,
            "initial_yaml": init,
        })
    return sessions


# ---------- per-session resource loader ----------
def load_session_data(session, voxel):
    """Returns dict with raw_pcs, cams_meta, T_init, init_points, init_colors, pivots."""
    cams = load_extrinsics(session["extrinsic_file"])
    n = len(cams)
    raw_pcs = []
    for i in range(n):
        p = session["cached_pc"] / f"cam{i}_uncropped.ply"
        if not p.exists():
            print(f"[WARN] {session['name']}: missing {p.name}, padding with empty cloud")
            raw_pcs.append(o3d.geometry.PointCloud())
            continue
        pc = o3d.io.read_point_cloud(str(p))
        if voxel > 0 and len(pc.points) > 0:
            pc = pc.voxel_down_sample(voxel)
        raw_pcs.append(pc)

    T_init = [np.eye(4) for _ in range(n)]
    if session["initial_yaml"] is not None:
        init_cams = load_extrinsics(session["initial_yaml"])
        if len(init_cams) == n:
            for i in range(n):
                orig = np.array(cams[i]["transformation"]).reshape(4, 4)
                new = np.array(init_cams[i]["transformation"]).reshape(4, 4)
                T_init[i] = new @ np.linalg.inv(orig)

    init_points, init_colors, pivots = [], [], []
    for i in range(n):
        pc = copy.deepcopy(raw_pcs[i])
        pc.transform(T_init[i])
        pts = np.asarray(pc.points, dtype=np.float32)
        cols = np.asarray(pc.colors, dtype=np.float32) if pc.has_colors() else None
        init_points.append(pts)
        init_colors.append(cols)
        pivots.append(pts.mean(axis=0) if len(pts) else np.zeros(3, dtype=np.float32))

    return {
        "n": n,
        "raw_pcs": raw_pcs,
        "cams_meta": cams,
        "T_init": T_init,
        "init_points": init_points,
        "init_colors": init_colors,
        "pivots": pivots,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", required=True,
                    help="parent dir containing many videos_XXXX folders")
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--max_trans", type=float, default=0.5,
                    help="initial translation slider half-range in meters "
                         "(user can change live via GUI)")
    ap.add_argument("--max_rot", type=float, default=45.0,
                    help="initial rotation slider half-range in degrees "
                         "(user can change live via GUI)")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated videos_XXXX names to include (default: all discovered)")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    sessions = discover_sessions(data_root)
    if args.only:
        keep = set(x.strip() for x in args.only.split(","))
        sessions = [s for s in sessions if s["name"] in keep]
    if not sessions:
        raise SystemExit(f"No usable sessions under {data_root}. "
                          "Run batch_cache_pc.sh first.")

    print(f"[INFO] {len(sessions)} sessions:")
    for s in sessions:
        init_tag = f"  (init={s['initial_yaml'].name})" if s["initial_yaml"] else "  (no initial)"
        print(f"  - {s['name']}{init_tag}")

    session_names = [s["name"] for s in sessions]

    # slider state per session: list of n_cam 6dof dicts (initialised on first load)
    session_state = {}      # name -> list[dict(tx,ty,tz,rx,ry,rz)]
    session_cached = {}     # name -> load_session_data() result

    # ---- viser server ----
    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"[INFO] viser http://{args.host}:{args.port}")

    server.scene.add_frame("/world_axes", axes_length=0.1, axes_radius=0.002)

    # ---- GUI (persistent widgets) ----
    server.gui.add_markdown("### Session")
    session_dd = server.gui.add_dropdown("Session", options=session_names,
                                          initial_value=session_names[0])
    with server.gui.add_folder("Nav", expand_by_default=True):
        btn_prev_sess = server.gui.add_button("< Prev Session")
        btn_next_sess = server.gui.add_button("Next Session >")
    sess_info = server.gui.add_markdown("")

    server.gui.add_markdown("---")
    server.gui.add_markdown("### Camera edit")
    cam_dd = server.gui.add_dropdown("Edit cam", options=["cam0"], initial_value="cam0")

    # Live-adjustable slider half-ranges (GUI-driven).
    trans_scale = server.gui.add_slider("Trans half-range (m)",
                                          min=0.05, max=3.0, step=0.05,
                                          initial_value=float(args.max_trans))
    rot_scale = server.gui.add_slider("Rot half-range (°)",
                                        min=5.0, max=180.0, step=1.0,
                                        initial_value=float(args.max_rot))

    tx = server.gui.add_slider("tx  (m)",  -args.max_trans, args.max_trans, 0.001, 0.0)
    ty = server.gui.add_slider("ty  (m)",  -args.max_trans, args.max_trans, 0.001, 0.0)
    tz = server.gui.add_slider("tz  (m)",  -args.max_trans, args.max_trans, 0.001, 0.0)
    rx = server.gui.add_slider("rx  (°)",  -args.max_rot,   args.max_rot,   0.1,   0.0)
    ry = server.gui.add_slider("ry  (°)",  -args.max_rot,   args.max_rot,   0.1,   0.0)
    rz = server.gui.add_slider("rz  (°)",  -args.max_rot,   args.max_rot,   0.1,   0.0)

    def _apply_slider_ranges():
        t = float(trans_scale.value)
        r = float(rot_scale.value)
        # Clamp current values to the new range first, so viser doesn't reject
        # the min/max update because the current value falls outside.
        for h in (tx, ty, tz):
            v = float(h.value)
            if v > t: h.value = t
            elif v < -t: h.value = -t
            h.min, h.max = -t, t
        for h in (rx, ry, rz):
            v = float(h.value)
            if v > r: h.value = r
            elif v < -r: h.value = -r
            h.min, h.max = -r, r

    @trans_scale.on_update
    def _(_evt): _apply_slider_ranges()
    @rot_scale.on_update
    def _(_evt): _apply_slider_ranges()

    server.gui.add_markdown("---")
    btn_reset = server.gui.add_button("Reset this cam")
    btn_reset_all = server.gui.add_button("Reset ALL cams (this session)")
    btn_save = server.gui.add_button("Save (overwrite *_global_aligned.yaml)")

    server.gui.add_markdown("---")
    server.gui.add_markdown("**Point display**")
    pt_size = server.gui.add_slider("Point size", 0.0005, 0.01, 0.0005, 0.003)
    show_solid = server.gui.add_checkbox("Solid per-cam colors", initial_value=False)

    server.gui.add_markdown("---")
    server.gui.add_markdown("**Visibility**")
    # visibility toggles rebuilt per session
    vis_folder_mds = []   # markdown placeholders if needed
    vis_toggles = []      # active toggle handles

    status = server.gui.add_markdown("*loading first session...*")

    # ---- active session mutable state ----
    active = {
        "name": None,
        "data": None,              # output of load_session_data
        "cam_frames": [],
        "pc_handles": [],
    }

    # ---- helpers ----
    def _muted_set(handle, val):
        """Set a slider value without firing our own on_update handler."""
        # Viser callbacks still fire when we assign, so we use a flag.
        nonlocal _muting
        _muting = True
        handle.value = val
        _muting = False

    _muting = False

    def _apply_cam_transform(cam_idx):
        d = active["data"]
        s = session_state[active["name"]][cam_idx]
        T = manual_transform(s["tx"], s["ty"], s["tz"],
                              s["rx"], s["ry"], s["rz"],
                              d["pivots"][cam_idx])
        active["cam_frames"][cam_idx].wxyz = tuple(rotmat_to_wxyz(T[:3, :3]).tolist())
        active["cam_frames"][cam_idx].position = tuple(T[:3, 3].tolist())

    def _sync_sliders_from_state():
        cam_idx = int(cam_dd.value[3:])
        s = session_state[active["name"]][cam_idx]
        _muted_set(tx, s["tx"])
        _muted_set(ty, s["ty"])
        _muted_set(tz, s["tz"])
        _muted_set(rx, s["rx"])
        _muted_set(ry, s["ry"])
        _muted_set(rz, s["rz"])

    def _set_status(msg):
        status.content = f"`{msg}`"
        print(f"[GUI] {msg}")

    def _teardown_scene():
        for h in active["cam_frames"]:
            h.remove()
        for h in active["pc_handles"]:
            h.remove()
        for cb in vis_toggles:
            cb.remove()
        active["cam_frames"] = []
        active["pc_handles"] = []
        vis_toggles.clear()

    def _build_scene_for(name):
        sess = next(s for s in sessions if s["name"] == name)
        if name not in session_cached:
            _set_status(f"loading {name} …")
            session_cached[name] = load_session_data(sess, args.voxel)
        d = session_cached[name]
        n = d["n"]

        if name not in session_state:
            session_state[name] = [dict(tx=0.0, ty=0.0, tz=0.0,
                                          rx=0.0, ry=0.0, rz=0.0)
                                    for _ in range(n)]

        cam_frames, pc_handles = [], []
        for i in range(n):
            fr = server.scene.add_frame(
                f"/sess/cam_{i}",
                wxyz=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0),
                axes_length=0.0, axes_radius=0.0,
            )
            cam_frames.append(fr)
            pts = d["init_points"][i]
            cols = d["init_colors"][i]
            if cols is None or show_solid.value:
                col = (PALETTE[i % len(PALETTE)] * 255).astype(np.uint8)
                cols_u8 = np.tile(col, (len(pts), 1))
            else:
                cols_u8 = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
            h = server.scene.add_point_cloud(
                f"/sess/cam_{i}/pc",
                points=pts,
                colors=cols_u8,
                point_size=float(pt_size.value),
            )
            pc_handles.append(h)

        # rebuild cam dropdown options
        cam_names = [f"cam{i}" for i in range(n)]
        cam_dd.options = cam_names
        cam_dd.value = cam_names[0]

        # rebuild visibility toggles
        for i in range(n):
            cb = server.gui.add_checkbox(f"cam{i}", initial_value=True)

            def _mk(cam_i, cb_h=cb):
                @cb_h.on_update
                def _(_evt):
                    active["pc_handles"][cam_i].visible = bool(cb_h.value)
            _mk(i)
            vis_toggles.append(cb)

        active["name"] = name
        active["data"] = d
        active["cam_frames"] = cam_frames
        active["pc_handles"] = pc_handles

        # apply existing per-cam state (preserves edits across switches)
        for i in range(n):
            _apply_cam_transform(i)
        _sync_sliders_from_state()

        init_tag = sess["initial_yaml"].name if sess["initial_yaml"] else "none"
        idx = session_names.index(name) + 1
        sess_info.content = (f"**{idx} / {len(sessions)}** · `{name}` · "
                              f"{n} cams · initial_yaml: `{init_tag}`")
        _set_status(f"loaded {name}")

    def _switch_session(name):
        _teardown_scene()
        _build_scene_for(name)

    # ---- initial load ----
    _build_scene_for(session_names[0])

    # ---- handlers ----
    @session_dd.on_update
    def _(_evt):
        if active["name"] != session_dd.value:
            _switch_session(session_dd.value)

    @btn_prev_sess.on_click
    def _(_evt):
        idx = session_names.index(session_dd.value)
        if idx > 0:
            session_dd.value = session_names[idx - 1]
            _switch_session(session_names[idx - 1])

    @btn_next_sess.on_click
    def _(_evt):
        idx = session_names.index(session_dd.value)
        if idx < len(session_names) - 1:
            session_dd.value = session_names[idx + 1]
            _switch_session(session_names[idx + 1])

    @cam_dd.on_update
    def _(_evt):
        _sync_sliders_from_state()
        _set_status(f"editing {active['name']} / {cam_dd.value}")

    def _slider_changed(key, handle):
        if _muting:
            return
        cam_idx = int(cam_dd.value[3:])
        session_state[active["name"]][cam_idx][key] = float(handle.value)
        _apply_cam_transform(cam_idx)

    @tx.on_update
    def _(_evt): _slider_changed("tx", tx)
    @ty.on_update
    def _(_evt): _slider_changed("ty", ty)
    @tz.on_update
    def _(_evt): _slider_changed("tz", tz)
    @rx.on_update
    def _(_evt): _slider_changed("rx", rx)
    @ry.on_update
    def _(_evt): _slider_changed("ry", ry)
    @rz.on_update
    def _(_evt): _slider_changed("rz", rz)

    @btn_reset.on_click
    def _(_evt):
        cam_idx = int(cam_dd.value[3:])
        session_state[active["name"]][cam_idx] = dict(tx=0.0, ty=0.0, tz=0.0,
                                                        rx=0.0, ry=0.0, rz=0.0)
        _apply_cam_transform(cam_idx)
        _sync_sliders_from_state()
        _set_status(f"reset {active['name']} / {cam_dd.value}")

    @btn_reset_all.on_click
    def _(_evt):
        n = active["data"]["n"]
        for i in range(n):
            session_state[active["name"]][i] = dict(tx=0.0, ty=0.0, tz=0.0,
                                                      rx=0.0, ry=0.0, rz=0.0)
            _apply_cam_transform(i)
        _sync_sliders_from_state()
        _set_status(f"reset ALL cams in {active['name']}")

    @pt_size.on_update
    def _(_evt):
        for h in active["pc_handles"]:
            h.point_size = float(pt_size.value)

    @show_solid.on_update
    def _(_evt):
        d = active["data"]
        for i, h in enumerate(active["pc_handles"]):
            pts = d["init_points"][i]
            cols = d["init_colors"][i]
            if cols is None or show_solid.value:
                col = (PALETTE[i % len(PALETTE)] * 255).astype(np.uint8)
                h.colors = np.tile(col, (len(pts), 1))
            else:
                h.colors = (np.clip(cols, 0, 1) * 255).astype(np.uint8)

    @btn_save.on_click
    def _(_evt):
        name = active["name"]
        sess = next(s for s in sessions if s["name"] == name)
        d = active["data"]
        n = d["n"]
        extrinsic_file = sess["extrinsic_file"]
        # Overwrite the *_global_aligned.yaml (creating it if absent) so downstream
        # pipeline and next viser restart pick up this as the new baseline.
        out_yaml = extrinsic_file.parent / f"{extrinsic_file.stem}_global_aligned.yaml"
        out_ply = extrinsic_file.parent / f"{extrinsic_file.stem}_global_aligned.ply"

        orig_cams = load_extrinsics(extrinsic_file)
        updated = []
        merged = o3d.geometry.PointCloud()
        for i in range(n):
            s = session_state[name][i]
            T_manual = manual_transform(s["tx"], s["ty"], s["tz"],
                                          s["rx"], s["ry"], s["rz"],
                                          d["pivots"][i])
            T_total = T_manual @ d["T_init"][i]
            ext_orig = np.array(orig_cams[i]["transformation"]).reshape(4, 4)
            new_ext = T_total @ ext_orig
            updated.append({
                "camera_id": orig_cams[i].get("camera_id", i),
                "serial_number": orig_cams[i]["serial_number"],
                "transformation": new_ext.tolist(),
                "color_intrinsic_matrix": orig_cams[i]["color_intrinsic_matrix"],
                "depth_intrinsic_matrix": orig_cams[i]["depth_intrinsic_matrix"],
            })
            pc = copy.deepcopy(d["raw_pcs"][i])
            pc.transform(T_total)
            merged += pc

        with open(out_yaml, "w") as f:
            yaml.dump(updated, f, default_flow_style=False, sort_keys=False)
        o3d.io.write_point_cloud(str(out_ply), merged)
        _set_status(f"SAVED {name}: {out_yaml.name}")
        print(f"[SAVED] {out_yaml}")
        print(f"[SAVED] {out_ply}")

    import time
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[INFO] stopped")


if __name__ == "__main__":
    main()
