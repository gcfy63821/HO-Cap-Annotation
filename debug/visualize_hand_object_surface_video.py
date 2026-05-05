"""Render hand (MANO) + tool meshes as SHADED 3D SURFACES projected back
into every cam view. Sister script of debug/visualize_hand_object_video.py
(which draws wireframe). The two write to different mp4 paths so you can
edit them side-by-side / cross-fade between them.

This one uses pyrender's OpenGL offscreen renderer (z-buffered, lit, alpha
channel) so the surface looks like the 3D viser preview — but rasterised
into each camera frame and composited over the original RGB.

Output:
    <annotated>/<exp>/debug/hand_object_video/<pose_prefer>_hand_object_surface.mp4

Cluster usage: needs PYOPENGL_PLATFORM=egl (the cluster sbatch scripts
already export this). On a workstation with X11, also fine.

Two modes (mirror visualize_hand_object_video.py):
  (a) Workspace:
      python debug/visualize_hand_object_surface_video.py \\
          --videos_root /viscam/projects/robotool/data/videos_0102
  (b) Single-exp:
      python debug/visualize_hand_object_surface_video.py \\
          --annotated_sequence /viscam/.../<task>/<exp>
"""

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

# pyrender requires this BEFORE importing pyrender on headless boxes.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import h5py
import numpy as np
import trimesh
import yaml

HOCAP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HOCAP_ROOT))
sys.path.insert(0, str(HOCAP_ROOT / "debug"))

# Reuse helpers from the wireframe sibling: tool/object resolution,
# object world-pose loading, hand pkl loader, MANO bundle. Importing the
# module is safe — argparse only runs under its __main__ guard.
from visualize_hand_object_video import (    # noqa: E402
    derive_original_sequence,
    derive_annotated_for,
    find_h5_with_rgbd,
    VideoRGBSource,
    find_dataset,
    find_mesh_path,
    TOOL_NAME_MAP_JSON_DEFAULT,
    resolve_tool_from_name_map,
    load_object_world_poses,
    try_load_hand_data,
    ManoBundle,
    reconstruct_left_verts,
    reconstruct_right_verts,
    is_exp_done,
    discover_done_exps,
    parse_color,
    PALETTES,
    DEFAULT_OBJECT_COLOR,
    DEFAULT_LEFT_HAND_COLOR,
    DEFAULT_RIGHT_HAND_COLOR,
    open_writer,
    concat_frames_grid,
)

import pyrender    # noqa: E402
from pyrender.constants import RenderFlags    # noqa: E402

# OpenCV camera convention -> OpenGL camera convention (pyrender).
CV2GL = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]], dtype=np.float32)


# ============================================================
# Surface compositor
# ============================================================

class SurfaceCompositor:
    """Renders scene from a single camera and returns RGBA + depth.

    One renderer instance is reused across (cam, frame) pairs so the
    expensive GL context init only happens once. Scene is rebuilt per
    call (object/hand poses change every frame); registering meshes and
    transforms in a persistent scene vs. rebuilding is roughly a wash for
    O(8 cams × N frames) at our scale."""

    def __init__(self, W: int, H: int, znear: float = 0.01, zfar: float = 10.0):
        self.W, self.H = int(W), int(H)
        self.znear, self.zfar = float(znear), float(zfar)
        # OpenGL context creation (EGL on cluster). One renderer for the
        # whole job.
        self.renderer = pyrender.OffscreenRenderer(self.W, self.H)

    def close(self):
        try:
            self.renderer.delete()
        except Exception:
            pass

    def render(self,
                  K: np.ndarray, cam_RT_c2w: np.ndarray,
                  obj_verts_w: np.ndarray, obj_faces: np.ndarray, obj_color,
                  hand_meshes: list,    # list[(verts_w, faces, color)]
                  ambient: float = 0.45,
                  light_intensity: float = 4.0):
        """Returns (rgba uint8 (H, W, 4), depth float32 (H, W))."""
        scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0],    # transparent — we composite later
            ambient_light=[ambient, ambient, ambient],
        )
        # Camera
        cam = pyrender.IntrinsicsCamera(
            fx=float(K[0, 0]), fy=float(K[1, 1]),
            cx=float(K[0, 2]), cy=float(K[1, 2]),
            znear=self.znear, zfar=self.zfar,
        )
        cam_pose_gl = (cam_RT_c2w @ CV2GL).astype(np.float32)
        scene.add(cam, pose=cam_pose_gl)
        # A directional light parented on the camera so the lit side
        # always faces the viewer.
        scene.add(
            pyrender.DirectionalLight(color=[1.0, 1.0, 1.0],
                                        intensity=light_intensity),
            pose=cam_pose_gl,
        )
        # Object surface
        if obj_verts_w is not None and obj_faces is not None and len(obj_faces):
            obj_tm = trimesh.Trimesh(
                vertices=obj_verts_w.astype(np.float32),
                faces=obj_faces, process=False,
            )
            mat_obj = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0, roughnessFactor=0.7,
                baseColorFactor=[obj_color[0] / 255.0, obj_color[1] / 255.0,
                                  obj_color[2] / 255.0, 1.0],
                doubleSided=True, alphaMode="OPAQUE",
            )
            scene.add(pyrender.Mesh.from_trimesh(obj_tm, material=mat_obj))
        # Hand surfaces
        for verts_w, faces, color in hand_meshes:
            if verts_w is None or faces is None or len(faces) == 0:
                continue
            tm = trimesh.Trimesh(
                vertices=verts_w.astype(np.float32),
                faces=faces, process=False,
            )
            mat = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0, roughnessFactor=0.7,
                baseColorFactor=[color[0] / 255.0, color[1] / 255.0,
                                  color[2] / 255.0, 1.0],
                doubleSided=True, alphaMode="OPAQUE",
            )
            scene.add(pyrender.Mesh.from_trimesh(tm, material=mat))

        rgba, depth = self.renderer.render(
            scene, flags=RenderFlags.RGBA | RenderFlags.SKIP_CULL_FACES,
        )
        return rgba, depth


def composite_surface(bg_rgb: np.ndarray, rgba: np.ndarray,
                         mesh_alpha: float = 1.0,
                         bg_white_alpha: float = 0.0) -> np.ndarray:
    """Alpha-blend the rendered RGBA mesh layer over the background RGB.

    bg_white_alpha: if >0, fade pixels OUTSIDE the mesh silhouette toward
    white by this amount before compositing — same idea as the wireframe
    script's bg fade, gives the foreground extra pop."""
    H, W = bg_rgb.shape[:2]
    if rgba.shape[:2] != (H, W):
        rgba = cv2.resize(rgba, (W, H), interpolation=cv2.INTER_AREA)
    rgb = rgba[..., :3].astype(np.float32)
    a = rgba[..., 3:4].astype(np.float32) / 255.0    # (H,W,1) in [0,1]
    if mesh_alpha < 1.0:
        a = a * float(mesh_alpha)

    bg = bg_rgb.astype(np.float32)
    if bg_white_alpha > 0:
        # Outside the mesh silhouette (a≈0), blend bg toward white.
        outside = (1.0 - (a > 1e-6).astype(np.float32))    # 1 where no mesh
        white = np.full_like(bg, 255.0)
        bg = bg * (1.0 - outside * bg_white_alpha) + white * (outside * bg_white_alpha)

    out = a * rgb + (1.0 - a) * bg
    return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================
# Per-exp render
# ============================================================

def render_one_exp(annotated_sequence: Path,
                     start_frame: int = 0, end_frame: int = -1,
                     frame_step: int = 1,
                     pose_prefer: str = "auto",
                     fps: int = 20,
                     tile_w: int = 640, tile_h: int = 480,
                     tool_name_map_json: Path = None,
                     tool_name_map_disabled: bool = False,
                     object_id: str = None,
                     output_path: Path = None,
                     force: bool = False,
                     no_hand: bool = False,
                     mesh_alpha: float = 0.85,
                     bg_white_alpha: float = 0.0,
                     ambient: float = 0.45,
                     light_intensity: float = 4.0,
                     object_color: tuple = DEFAULT_OBJECT_COLOR,
                     left_hand_color: tuple = DEFAULT_LEFT_HAND_COLOR,
                     right_hand_color: tuple = DEFAULT_RIGHT_HAND_COLOR) -> dict:
    annotated_sequence = annotated_sequence.resolve()
    original_sequence = derive_original_sequence(annotated_sequence)
    meta_path = original_sequence / "meta.yaml"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta.yaml: {meta_path}")
    meta = yaml.safe_load(meta_path.read_text())
    calib_path = Path(meta["calibration_yaml_path"])
    models_folder = Path(meta["models_folder"])

    # ---- tool / mesh ----
    if (not tool_name_map_disabled and tool_name_map_json is None
            and TOOL_NAME_MAP_JSON_DEFAULT.is_file()):
        tool_name_map_json = TOOL_NAME_MAP_JSON_DEFAULT
    mesh_path = None; mapped_via = None
    if not object_id:
        meta_obj_id = meta["object_ids"][0]
        try:
            mesh_path = find_mesh_path(models_folder, meta_obj_id)
            object_id = meta_obj_id
            mapped_via = "meta.yaml:object_ids[0]"
        except FileNotFoundError:
            mesh_path = None
    if not object_id and tool_name_map_json is not None:
        task = annotated_sequence.parent.name
        exp = annotated_sequence.name
        name, path, key = resolve_tool_from_name_map(
            tool_name_map_json, task, exp, models_folder)
        if name is not None and path is not None:
            object_id = name; mesh_path = path
            mapped_via = f"tool_name_map_json:{key} -> {name}"
    if not object_id:
        object_id = meta["object_ids"][0]
        mapped_via = mapped_via or "meta.yaml:object_ids[0]"
    if mesh_path is None:
        mesh_path = find_mesh_path(models_folder, object_id)

    # ---- output path ----
    out_path = output_path if output_path is not None else (
        annotated_sequence / "debug" / "hand_object_video"
        / f"{pose_prefer}_hand_object_surface.mp4"
    )
    if out_path.exists() and not force:
        print(f"[skip] {out_path} exists (pass --force)")
        return {"output": str(out_path), "skipped": True}

    # ---- calib + object mesh ----
    calib = yaml.safe_load(calib_path.read_text())
    serials = [str(c["camera_id"]).zfill(2) for c in calib]
    Ks = [np.array(c["color_intrinsic_matrix"], dtype=np.float32) for c in calib]
    cam_RTs = [np.array(c["transformation"], dtype=np.float32) for c in calib]

    obj_mesh = trimesh.load(mesh_path, process=True, force="mesh")
    if not isinstance(obj_mesh, trimesh.Trimesh):
        obj_mesh = trimesh.util.concatenate(obj_mesh.dump())
    obj_verts_local = np.asarray(obj_mesh.vertices, dtype=np.float32)
    obj_faces = np.asarray(obj_mesh.faces, dtype=np.int32)

    # ---- RGB source ----
    h5_path = None; video_source = None
    try:
        h5_path = find_h5_with_rgbd(original_sequence)
    except FileNotFoundError:
        video_source = VideoRGBSource(original_sequence, len(serials))

    data_ctx = contextlib.nullcontext(None) if h5_path is None else h5py.File(h5_path, "r")
    with data_ctx as data_f:
        imgs_ds = find_dataset(data_f, "imgs") if data_f else None
        n_frames_total = (int(imgs_ds.shape[0]) if imgs_ds is not None
                            else int(video_source.frame_count))
        # source frame size — used both for compositing and for the
        # offscreen render dimensions (avoids resize artefacts on the
        # mesh silhouette).
        if imgs_ds is not None:
            src_h, src_w = int(imgs_ds.shape[2]), int(imgs_ds.shape[3])
        else:
            tmp = video_source.get_color(0, 0)
            src_h, src_w = tmp.shape[:2]

        if end_frame < 0 or end_frame >= n_frames_total:
            end_frame = n_frames_total - 1
        start_frame = max(0, min(start_frame, end_frame))
        chosen = list(range(start_frame, end_frame + 1, max(1, frame_step)))

        obj_poses_w, pose_src = load_object_world_poses(
            annotated_sequence, n_frames_total, object_idx=0, prefer=pose_prefer)

        hand_data = None; mano = None
        if not no_hand:
            hand_data = try_load_hand_data(annotated_sequence)
            if hand_data is not None:
                try:
                    mano = ManoBundle()
                except Exception as e:
                    print(f"[WARN] could not init MANO ({e}); rendering tool-only")
                    mano = None; hand_data = None
            else:
                print(f"[INFO] no result(_hand_optimized).pkl under "
                      f"{annotated_sequence} — rendering tool-only")

        # ---- pyrender renderer at source resolution ----
        compositor = SurfaceCompositor(src_w, src_h)

        n_cams = len(serials)
        grid_w = 4 * tile_w; grid_h = 2 * tile_h
        writer = open_writer(out_path, grid_w, grid_h, fps=fps)

        n_written = 0; t0 = time.time()
        try:
            for fi, frame_idx in enumerate(chosen):
                pose_w = obj_poses_w[frame_idx]
                has_obj = bool(np.all(np.isfinite(pose_w)))
                # Hand world verts (precomputed once per frame, used for all 8 cams)
                hand_world = {}
                if mano is not None and hand_data is not None:
                    lv = reconstruct_left_verts(hand_data, frame_idx, mano)
                    if lv is not None: hand_world["left"] = lv
                    rv = reconstruct_right_verts(hand_data, frame_idx, mano)
                    if rv is not None: hand_world["right"] = rv

                obj_verts_w = None
                if has_obj:
                    obj_verts_w = (pose_w[:3, :3] @ obj_verts_local.T).T + pose_w[:3, 3]

                tiles = []
                for cam_idx in range(n_cams):
                    if imgs_ds is not None:
                        rgb = np.asarray(imgs_ds[frame_idx, cam_idx], dtype=np.uint8)
                    else:
                        rgb = video_source.get_color(frame_idx, cam_idx)

                    # Build per-cam hand mesh list with colors.
                    hand_render = []
                    for side, vw in hand_world.items():
                        faces = mano.faces
                        col = left_hand_color if side == "left" else right_hand_color
                        hand_render.append((vw, faces, col))

                    rgba, _ = compositor.render(
                        K=Ks[cam_idx], cam_RT_c2w=cam_RTs[cam_idx],
                        obj_verts_w=obj_verts_w, obj_faces=obj_faces,
                        obj_color=object_color,
                        hand_meshes=hand_render,
                        ambient=ambient, light_intensity=light_intensity,
                    )
                    img = composite_surface(
                        rgb, rgba, mesh_alpha=mesh_alpha,
                        bg_white_alpha=bg_white_alpha,
                    )

                    cv2.putText(img, f"f{frame_idx:06d} cam{serials[cam_idx]}",
                                  (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                    cv2.putText(img, f"f{frame_idx:06d} cam{serials[cam_idx]}",
                                  (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    if (img.shape[1], img.shape[0]) != (tile_w, tile_h):
                        img = cv2.resize(img, (tile_w, tile_h),
                                          interpolation=cv2.INTER_AREA)
                    tiles.append(img)

                grid = concat_frames_grid(tiles, grid=(2, 4))
                writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                n_written += 1
                if (fi + 1) % 50 == 0:
                    fps_now = (fi + 1) / max(time.time() - t0, 1e-6)
                    print(f"  [{fi+1}/{len(chosen)}] frame {frame_idx}  "
                          f"{fps_now:.2f} fps")
        finally:
            writer.release()
            compositor.close()
            if video_source is not None: video_source.close()

    summary = {
        "annotated_sequence": str(annotated_sequence),
        "output": str(out_path),
        "kind": "surface",
        "tool": object_id, "tool_via": mapped_via,
        "mesh_path": str(mesh_path),
        "pose_source": pose_src,
        "with_hand": hand_data is not None and mano is not None,
        "n_frames_written": n_written,
        "frame_range": [start_frame, end_frame, frame_step],
        "tile_size": [tile_w, tile_h],
        "fps": fps,
        "mesh_alpha": mesh_alpha,
        "bg_white_alpha": bg_white_alpha,
        "ambient": ambient,
        "light_intensity": light_intensity,
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"[OK] wrote {out_path}  ({n_written} frames, "
          f"hand={'on' if summary['with_hand'] else 'off'}, "
          f"object_pose={pose_src})")
    return summary


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos_root", type=Path, default=None)
    ap.add_argument("--annotated_sequence", type=Path, default=None)
    ap.add_argument("--task_filter", nargs="+", default=None)
    ap.add_argument("--only_remasked", action="store_true",
                    help="Workspace mode: only render exps with mask_prompts.json.")
    ap.add_argument("--object_id", default=None)
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--end_frame", type=int, default=-1)
    ap.add_argument("--frame_step", type=int, default=1)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--tile_w", type=int, default=640)
    ap.add_argument("--tile_h", type=int, default=480)
    ap.add_argument("--pose_prefer", choices=("auto", "optimized", "fd"),
                    default="auto")
    ap.add_argument("--no_hand", action="store_true")
    ap.add_argument("--mesh_alpha", type=float, default=0.85,
                    help="Surface opacity over the original RGB. 1.0 = fully "
                         "opaque (covers underlying pixels), 0.0 = invisible. "
                         "0.7-0.9 reads as 'lit surface visible but RGB still "
                         "shows through'.")
    ap.add_argument("--bg_white_alpha", type=float, default=0.0,
                    help="Fade pixels OUTSIDE the mesh silhouette toward white. "
                         "Try 0.5-0.7 for a poster look that pops the foreground.")
    ap.add_argument("--ambient", type=float, default=0.45,
                    help="Ambient light level (0=harsh, 1=flat). Default 0.45.")
    ap.add_argument("--light_intensity", type=float, default=4.0,
                    help="Camera-attached directional light. Higher = more "
                         "specular contrast. Default 4.")
    ap.add_argument("--palette", choices=tuple(PALETTES.keys()), default="default")
    ap.add_argument("--object_color", type=str, default=None)
    ap.add_argument("--left_hand_color", type=str, default=None)
    ap.add_argument("--right_hand_color", type=str, default=None)
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--continue_on_error", action="store_true", default=True)
    ap.add_argument("--tool_name_map_json", type=Path, default=None)
    ap.add_argument("--no_tool_name_map_json", action="store_true")
    args = ap.parse_args()

    if args.videos_root is None and args.annotated_sequence is None:
        ap.error("provide either --videos_root or --annotated_sequence")
    if args.videos_root is not None and args.annotated_sequence is not None:
        ap.error("--videos_root and --annotated_sequence are mutually exclusive")

    pal_obj, pal_left, pal_right = PALETTES[args.palette]
    object_color     = parse_color(args.object_color, pal_obj)
    left_hand_color  = parse_color(args.left_hand_color, pal_left)
    right_hand_color = parse_color(args.right_hand_color, pal_right)
    print(f"[palette={args.palette}] tool={object_color}  "
          f"left={left_hand_color}  right={right_hand_color}")

    common = dict(
        start_frame=args.start_frame, end_frame=args.end_frame,
        frame_step=args.frame_step, pose_prefer=args.pose_prefer,
        fps=args.fps, tile_w=args.tile_w, tile_h=args.tile_h,
        no_hand=args.no_hand,
        mesh_alpha=args.mesh_alpha,
        bg_white_alpha=args.bg_white_alpha,
        ambient=args.ambient,
        light_intensity=args.light_intensity,
        object_color=object_color,
        left_hand_color=left_hand_color,
        right_hand_color=right_hand_color,
        tool_name_map_json=args.tool_name_map_json,
        tool_name_map_disabled=args.no_tool_name_map_json,
        object_id=args.object_id, force=args.force,
    )

    if args.annotated_sequence is not None:
        try:
            render_one_exp(annotated_sequence=args.annotated_sequence,
                            output_path=args.output_dir, **common)
        except Exception as e:
            print(f"[ERR] {e}", file=sys.stderr)
            traceback.print_exc(); sys.exit(1)
        return

    videos_root = args.videos_root.resolve()
    if not videos_root.is_dir():
        ap.error(f"--videos_root not a directory: {videos_root}")

    exps, n_skip = discover_done_exps(
        videos_root,
        task_filter=set(args.task_filter) if args.task_filter else None,
        only_remasked=args.only_remasked)
    if not exps:
        print(f"No DONE exps under {videos_root} (skipped {n_skip} unfinished)")
        sys.exit(0)
    print(f"[INFO] {len(exps)} done exps  (skipped {n_skip} unfinished)\n")

    n_ok = n_fail = n_skip2 = 0
    failures = []
    t0 = time.time()
    for i, (task, exp, _exp_dir, ann, reason) in enumerate(exps, 1):
        prefix = f"[{i:>4d}/{len(exps)}] {task}/{exp}"
        per_exp_out = (args.output_dir / task / exp / f"{args.pose_prefer}_hand_object_surface.mp4"
                        if args.output_dir is not None else None)
        existing = (per_exp_out if per_exp_out is not None else
                     ann / "debug" / "hand_object_video"
                     / f"{args.pose_prefer}_hand_object_surface.mp4")
        if existing.exists() and not args.force:
            print(f"{prefix}  skip  (mp4 exists; pass --force)")
            n_skip2 += 1; continue
        try:
            t1 = time.time()
            summary = render_one_exp(annotated_sequence=ann,
                                          output_path=per_exp_out, **common)
            n_ok += 1
            print(f"{prefix}  OK [{time.time()-t1:.1f}s]  "
                  f"{summary.get('n_frames_written', '?')} frames  ({reason})")
        except KeyboardInterrupt:
            print("[INTERRUPTED]"); sys.exit(130)
        except Exception as e:
            n_fail += 1; failures.append((task, exp, str(e)))
            print(f"{prefix}  FAIL  ({type(e).__name__}: {e})")
            if not args.continue_on_error: raise

    print()
    print("=" * 60)
    print(f"summary: ok={n_ok}  skip={n_skip2}  fail={n_fail}  "
          f"elapsed={time.time()-t0:.1f}s")
    if failures:
        print("\nFailures:")
        for task, exp, msg in failures[:10]:
            print(f"  {task}/{exp}: {msg}")
        if len(failures) > 10:
            print(f"  ... and {len(failures)-10} more")
    print("=" * 60)


if __name__ == "__main__":
    main()
