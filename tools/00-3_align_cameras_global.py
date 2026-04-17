#!/usr/bin/env python
"""
Global multi-camera point cloud alignment.

Pipeline:
  1. Load cached per-camera point clouds (already in initial world frame)
  2. Tight crop to the overlap region (table + workspace)
  3. For every camera pair with sufficient overlap:
        FPFH + RANSAC (coarse)  ->  point-to-plane ICP (medium)  ->  colored ICP (fine)
     + compute information matrix for pose-graph weighting
  4. Build a pose graph anchored at a reference camera and run global optimization
     (distributes residual error across the whole ring rather than accumulating).
  5. Fit a single RANSAC plane on the globally-merged cloud and rotate the
     whole world frame so that plane becomes z = 0 (table flat, centered).
  6. Compose: extrinsic_new = T_plane @ T_cam @ extrinsic_orig, save YAML + PLY.
"""

import argparse
import copy
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from tqdm import tqdm


# -------- tunables (CLI-overridable) --------
DEFAULT_X = (-0.4, 0.4)
DEFAULT_Y = (-0.4, 0.4)
DEFAULT_Z = (-0.05, 0.35)          # keep table + tools, drop ceiling/walls
DEFAULT_VOXEL_COARSE = 0.01        # 1 cm for FPFH / RANSAC
DEFAULT_VOXEL_FINE = 0.003         # 3 mm for ICP
DEFAULT_MIN_FITNESS = 0.25         # edge rejection threshold
DEFAULT_MIN_OVERLAP_POINTS = 500   # below this, skip the pair


# ---------- IO helpers ----------
def load_extrinsics(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and "extrinsics" in data:
        cams = [v for k, v in data["extrinsics"].items() if not k.startswith("tag_")]
    elif isinstance(data, list):
        cams = sorted(data, key=lambda x: x.get("camera_id", 0))
    else:
        raise ValueError(f"Unsupported YAML format: {path}")
    return cams


def crop_world(pc, x, y, z):
    p = np.asarray(pc.points)
    m = (
        (p[:, 0] > x[0]) & (p[:, 0] < x[1])
        & (p[:, 1] > y[0]) & (p[:, 1] < y[1])
        & (p[:, 2] > z[0]) & (p[:, 2] < z[1])
    )
    return pc.select_by_index(np.where(m)[0])


def preprocess(pc, voxel):
    pc_ds = pc.voxel_down_sample(voxel)
    pc_ds.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30)
    )
    pc_ds.orient_normals_towards_camera_location(np.array([0.0, 0.0, 2.0]))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pc_ds,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100),
    )
    return pc_ds, fpfh


# ---------- pairwise registration ----------
def ransac_global(src, tgt, src_f, tgt_f, voxel):
    dist = voxel * 1.5
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, tgt, src_f, tgt_f,
        mutual_filter=True,
        max_correspondence_distance=dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 500),
    )


def icp_refine(src, tgt, init, voxel_fine):
    # point-to-plane then colored ICP
    reg_p2l = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=voxel_fine * 5,
        init=init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
    )
    if src.has_colors() and tgt.has_colors():
        reg_c = o3d.pipelines.registration.registration_colored_icp(
            src, tgt,
            max_correspondence_distance=voxel_fine * 2,
            init=reg_p2l.transformation,
            estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-7, relative_rmse=1e-7, max_iteration=50
            ),
        )
        return reg_c
    return reg_p2l


def pair_register(pc_c_src, pc_c_tgt, pc_f_src, pc_f_tgt, fpfh_src, fpfh_tgt,
                  voxel_coarse, voxel_fine, use_ransac):
    """Register src -> tgt. PCs already share an approximate world frame (initial extrinsics
    applied), so default init is identity. RANSAC is opt-in for really bad initials."""
    if use_ransac:
        r = ransac_global(pc_c_src, pc_c_tgt, fpfh_src, fpfh_tgt, voxel_coarse)
        if r.fitness < 0.05:
            return None
        init = r.transformation
    else:
        init = np.eye(4)
        # coarse ICP first to absorb cm-level drift without going wild
        coarse = o3d.pipelines.registration.registration_icp(
            pc_c_src, pc_c_tgt,
            max_correspondence_distance=voxel_coarse * 3,
            init=init,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
        )
        init = coarse.transformation
    reg = icp_refine(pc_f_src, pc_f_tgt, init, voxel_fine)
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        pc_f_src, pc_f_tgt, voxel_fine * 2, reg.transformation
    )
    return reg, info


def transform_delta(T):
    """Return (translation norm in meters, rotation angle in degrees) for a 4x4 tf."""
    t = float(np.linalg.norm(T[:3, 3]))
    cos_a = (np.trace(T[:3, :3]) - 1.0) / 2.0
    a = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
    return t, a


# ---------- global plane to z=0 ----------
def plane_to_z0_transform(plane_model):
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    n = n / np.linalg.norm(n)
    # flip so it points up
    if n[2] < 0:
        n = -n
        d = -d
    target = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, target)
    s = np.linalg.norm(v)
    c_ = float(np.dot(n, target))
    if s < 1e-8:
        R = np.eye(3) if c_ > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c_) / (s * s))
    T = np.eye(4)
    T[:3, :3] = R
    # after rotating, plane becomes z = d / |n|; shift to z=0
    T[2, 3] = d
    return T


def project_to_se2(T):
    """Project SE(3) to SE(2) in the xy plane: keep yaw + xy translation, zero pitch/roll/z."""
    R = T[:3, :3]
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    cs, sn = np.cos(yaw), np.sin(yaw)
    out = np.eye(4)
    out[0, 0] = cs
    out[0, 1] = -sn
    out[1, 0] = sn
    out[1, 1] = cs
    out[0, 3] = T[0, 3]
    out[1, 3] = T[1, 3]
    out[2, 3] = 0.0
    return out


def fit_table_plane_snap(pc, voxel, dist_thresh=0.008):
    """Fit the dominant plane of `pc` and return T that rotates it onto z=0.
    Returns (T, plane_model, inlier_ratio) or (None, None, 0) on failure."""
    ds = pc.voxel_down_sample(voxel) if voxel > 0 else pc
    if len(ds.points) < 50:
        return None, None, 0.0
    try:
        plane_model, inliers = ds.segment_plane(
            distance_threshold=dist_thresh, ransac_n=3, num_iterations=3000
        )
    except Exception:
        return None, None, 0.0
    ratio = len(inliers) / max(1, len(ds.points))
    return plane_to_z0_transform(plane_model), plane_model, ratio


# ---------- main ----------
def main(cached_pc_dir, extrinsic_file, out_path, ref_idx,
         x_range, y_range, z_range, voxel_coarse, voxel_fine,
         min_fitness, min_overlap, skip_plane, do_post_refine,
         use_ransac, max_rot_deg, max_trans_m, plane_source,
         per_cam_plane_snap, plane_dist_thresh, se2_realign_iters,
         se2_bridge_dist, se2_bridge_trans_m,
         loose_cams, loose_max_rot_deg, loose_max_trans_m, loose_se2_bridge_dist):
    loose_set = set(loose_cams or [])

    def caps_for(cam_i):
        if cam_i in loose_set:
            return loose_max_rot_deg, loose_max_trans_m
        return max_rot_deg, max_trans_m

    def pair_caps(i, j):
        # if either cam in the pair is loose, the pair uses loose caps
        ri, ti = caps_for(i)
        rj, tj = caps_for(j)
        return max(ri, rj), max(ti, tj)

    cached_pc_dir = Path(cached_pc_dir)
    extrinsic_file = Path(extrinsic_file)
    out_path = Path(out_path) if out_path else extrinsic_file.parent
    out_path.mkdir(parents=True, exist_ok=True)
    posts_dir = out_path / "posts_global"
    posts_dir.mkdir(exist_ok=True)

    cams = load_extrinsics(extrinsic_file)
    n_cams = len(cams)
    print(f"[INFO] {n_cams} cameras in extrinsic file")

    # load + crop raw clouds (world frame), keep full-resolution copy for final dump
    raw_pcs, fine_pcs, coarse_pcs, coarse_fpfh, fine_fpfh = [], [], [], [], []
    for i in range(n_cams):
        f = cached_pc_dir / f"cam{i}_uncropped.ply"
        if not f.exists():
            raise FileNotFoundError(f)
        pc = o3d.io.read_point_cloud(str(f))
        pc = crop_world(pc, x_range, y_range, z_range)
        raw_pcs.append(pc)
        pc_f, fpfh_f = preprocess(pc, voxel_fine)
        pc_c, fpfh_c = preprocess(pc, voxel_coarse)
        fine_pcs.append(pc_f)
        coarse_pcs.append(pc_c)
        fine_fpfh.append(fpfh_f)
        coarse_fpfh.append(fpfh_c)
        print(f"  cam{i}: raw={len(pc.points)}  fine={len(pc_f.points)}  coarse={len(pc_c.points)}")

    # save pre-alignment merged cloud
    pre = o3d.geometry.PointCloud()
    for pc in raw_pcs:
        pre += pc
    o3d.io.write_point_cloud(str(out_path / "prealign_global.ply"), pre)

    # pairwise registration
    mode = "FPFH+RANSAC + ICP" if use_ransac else "identity-init coarse->fine ICP"
    print(f"\n[INFO] Pairwise registration ({mode})")
    print(f"[INFO] Sanity filter: reject edges with |Δrot| > {max_rot_deg}° or "
          f"|Δtrans| > {max_trans_m*100:.1f}cm")
    edges = []  # list of (i, j, T_ij, info, fitness)
    for i in tqdm(range(n_cams), desc="pairs"):
        for j in range(i + 1, n_cams):
            if len(fine_pcs[i].points) < min_overlap or len(fine_pcs[j].points) < min_overlap:
                continue
            try:
                res = pair_register(
                    coarse_pcs[i], coarse_pcs[j],
                    fine_pcs[i], fine_pcs[j],
                    coarse_fpfh[i], coarse_fpfh[j],
                    voxel_coarse, voxel_fine,
                    use_ransac=use_ransac,
                )
            except Exception as e:
                print(f"  pair ({i},{j}) failed: {e}")
                continue
            if res is None:
                continue
            reg, info = res
            dt, da = transform_delta(reg.transformation)
            print(f"  pair ({i},{j}): fitness={reg.fitness:.3f} rmse={reg.inlier_rmse:.4f} "
                  f"Δt={dt*1000:.1f}mm Δr={da:.2f}°")
            if reg.fitness < min_fitness:
                continue
            r_cap, t_cap = pair_caps(i, j)
            if dt > t_cap or da > r_cap:
                tag = " (loose-cam pair)" if (i in loose_set or j in loose_set) else ""
                print(f"    [reject] out of sanity bounds{tag} "
                      f"caps={r_cap:.1f}°/{t_cap*100:.1f}cm")
                continue
            edges.append((i, j, reg.transformation, info, reg.fitness))

    if not edges:
        raise RuntimeError("No valid pairwise edges — check crop bounds or cached PCs")

    # build pose graph
    print(f"\n[INFO] Building pose graph with {len(edges)} edges, anchor = cam{ref_idx}")
    pg = o3d.pipelines.registration.PoseGraph()
    node_pose = [np.eye(4) for _ in range(n_cams)]
    for p in node_pose:
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(p))

    # spanning tree from ref via BFS over edges, sorted by fitness (highest first)
    edges_sorted = sorted(edges, key=lambda e: -e[4])
    adj = {i: [] for i in range(n_cams)}
    for i, j, T, info, fit in edges_sorted:
        adj[i].append((j, T, info, fit))
        adj[j].append((i, np.linalg.inv(T), info, fit))

    seen = {ref_idx}
    in_tree = set()          # edge keys (min,max) chosen as "certain" odometry
    queue = [ref_idx]
    while queue:
        u = queue.pop(0)
        for v, T_uv, info, fit in adj[u]:
            if v in seen:
                continue
            # pose in graph: T_world_v = T_world_u @ T_uv (T_uv maps v->u pts)
            node_pose[v] = node_pose[u] @ T_uv
            pg.nodes[v].pose = node_pose[v]
            in_tree.add((min(u, v), max(u, v)))
            seen.add(v)
            queue.append(v)

    if len(seen) < n_cams:
        missing = set(range(n_cams)) - seen
        print(f"[WARN] cameras {missing} have no path to ref — excluded from graph")

    # add edges: tree ones uncertain=False, rest loop closure uncertain=True
    for i, j, T, info, fit in edges:
        key = (min(i, j), max(i, j))
        uncertain = key not in in_tree
        # edge convention: T transforms points from source(i) to target(j) frame
        pg.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                i, j, T, info, uncertain=uncertain
            )
        )

    # global optimization
    print("[INFO] Running global pose-graph optimization")
    opt = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=voxel_fine * 2,
        edge_prune_threshold=0.25,
        reference_node=ref_idx,
    )
    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        opt,
    )

    # per-camera correction = node pose (maps original-world -> corrected-world)
    T_cam = [np.array(pg.nodes[i].pose) for i in range(n_cams)]

    # sanity check: any cam whose optimized correction is crazy (e.g. got flipped
    # under the table by a bad loop closure) is reverted to identity
    print("\n[INFO] Post-optimization per-camera correction magnitudes")
    for i in range(n_cams):
        dt, da = transform_delta(T_cam[i])
        r_cap, t_cap = caps_for(i)
        flag = "  (loose)" if i in loose_set else ""
        if dt > t_cap or da > r_cap:
            T_cam[i] = np.eye(4)
            flag = f"  -> REVERTED to identity (>{r_cap:.0f}° / {t_cap*100:.0f}cm)"
        print(f"  cam{i}: Δt={dt*1000:.1f}mm Δr={da:.2f}°{flag}")

    # build corrected fine (downsampled) clouds for refinement / residual
    fine_corr = []
    for i in range(n_cams):
        pc = copy.deepcopy(fine_pcs[i])
        pc.transform(T_cam[i])
        fine_corr.append(pc)

    # post-hoc per-camera refinement: ICP each cam against union-of-others
    if do_post_refine:
        print("\n[INFO] Post-hoc per-camera ICP refinement (target = all other cams merged)")
        for i in range(n_cams):
            tgt = o3d.geometry.PointCloud()
            for j in range(n_cams):
                if j == i:
                    continue
                tgt += fine_corr[j]
            if len(tgt.points) < 100 or len(fine_corr[i].points) < 100:
                continue
            tgt = tgt.voxel_down_sample(voxel_fine)
            tgt.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_fine * 2.5, max_nn=30)
            )
            reg = o3d.pipelines.registration.registration_icp(
                fine_corr[i], tgt,
                max_correspondence_distance=voxel_fine * 4,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
            )
            dT = reg.transformation
            delta_t, delta_r = transform_delta(dT)
            print(f"  cam{i}: fitness={reg.fitness:.3f} rmse={reg.inlier_rmse:.4f} "
                  f"Δt={delta_t*1000:.1f}mm Δr={delta_r:.2f}°")
            if reg.fitness < 0.2:
                print(f"    [skip] fitness too low, keeping pose-graph pose")
                continue
            r_cap, t_cap = caps_for(i)
            if delta_t > t_cap or delta_r > r_cap:
                print(f"    [skip] refinement delta out of sanity bounds "
                      f"({r_cap:.0f}°/{t_cap*100:.0f}cm)")
                continue
            # fold into T_cam and re-transform fine
            T_cam[i] = dT @ T_cam[i]
            fine_corr[i] = copy.deepcopy(fine_pcs[i])
            fine_corr[i].transform(T_cam[i])

    # merge corrected clouds (full resolution)
    merged = o3d.geometry.PointCloud()
    corrected_raw = []
    for i in range(n_cams):
        pc = copy.deepcopy(raw_pcs[i])
        pc.transform(T_cam[i])
        corrected_raw.append(pc)
        merged += pc

    # ---------- plane handling ----------
    # Two modes:
    #  (a) per_cam_plane_snap = True (recommended for "everything flat on xy"):
    #      each cam independently rotates its dominant plane to z=0, then an SE(2)-only
    #      ICP pass locks yaw+xy across cams. Guarantees flatness even when some cams
    #      drifted off in pose graph.
    #  (b) per_cam_plane_snap = False: single global plane fit as before.
    T_plane_global = np.eye(4)

    if per_cam_plane_snap and not skip_plane:
        print("\n[INFO] Per-camera plane snap (each cam's table -> z=0 independently)")
        for i in range(n_cams):
            T_snap, model, ratio = fit_table_plane_snap(
                fine_corr[i], voxel_fine, plane_dist_thresh
            )
            if T_snap is None or ratio < 0.2:
                print(f"  cam{i}: plane fit failed or too sparse (ratio={ratio:.2f}) — keep as is")
                continue
            # inspect current plane normal vs world z to report severity
            a, b, c, d = model
            n = np.array([a, b, c]) / np.linalg.norm([a, b, c])
            if n[2] < 0:
                n = -n
                d = -d
            tilt = float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))
            z_offset = float(d)  # plane z-distance from origin (since points already in world)
            T_cam[i] = T_snap @ T_cam[i]
            fine_corr[i] = copy.deepcopy(fine_pcs[i])
            fine_corr[i].transform(T_cam[i])
            print(f"  cam{i}: tilt={tilt:.2f}° z_off={z_offset*1000:.1f}mm "
                  f"inliers={ratio*100:.0f}% -> snapped")

        # SE(2) realign: each cam is now flat, but yaw + xy may differ between cams.
        # Use a correspondence-distance schedule (bridge -> fine) so groups offset by
        # several cm can still find cross-group correspondences in the first pass.
        if se2_realign_iters > 0:
            # geometric schedule from se2_bridge_dist down to voxel_fine * 4
            fine_dist = voxel_fine * 4.0
            start_dist = max(se2_bridge_dist, fine_dist)
            if se2_realign_iters == 1:
                corr_schedule = [fine_dist]
            else:
                log_steps = np.linspace(
                    np.log(start_dist), np.log(fine_dist), se2_realign_iters
                )
                corr_schedule = np.exp(log_steps).tolist()

            print(f"\n[INFO] SE(2) realign ({se2_realign_iters} passes) "
                  f"corr={corr_schedule[0]*100:.1f}cm -> {corr_schedule[-1]*100:.1f}cm")
            for it, corr_dist in enumerate(corr_schedule):
                # widest-pass sanity: allow large xy bridging; later passes clamp down.
                # rotation is already locked by plane snap, so we keep a tight rot cap.
                is_bridge_pass = corr_dist > fine_dist * 1.5
                trans_cap = se2_bridge_trans_m if is_bridge_pass else max_trans_m
                rot_cap = max_rot_deg  # yaw only; per-cam plane snap prevents big tilts
                max_delta_mm = 0.0
                for i in range(n_cams):
                    tgt = o3d.geometry.PointCloud()
                    for j in range(n_cams):
                        if j != i:
                            tgt += fine_corr[j]
                    if len(tgt.points) < 100 or len(fine_corr[i].points) < 100:
                        continue
                    tgt = tgt.voxel_down_sample(voxel_fine)
                    tgt.estimate_normals(
                        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_fine * 2.5, max_nn=30)
                    )
                    # loose cams can use a wider correspondence distance so they can
                    # still bridge a big offset even during "fine" passes
                    cam_corr_dist = max(corr_dist, loose_se2_bridge_dist) if i in loose_set else corr_dist
                    cam_trans_cap = max(trans_cap, loose_max_trans_m) if i in loose_set else trans_cap
                    cam_rot_cap = max(rot_cap, loose_max_rot_deg) if i in loose_set else rot_cap
                    reg = o3d.pipelines.registration.registration_icp(
                        fine_corr[i], tgt,
                        max_correspondence_distance=cam_corr_dist,
                        init=np.eye(4),
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
                    )
                    if reg.fitness < 0.12:
                        continue
                    T_se2 = project_to_se2(reg.transformation)
                    dt, da = transform_delta(T_se2)
                    if dt > cam_trans_cap or da > cam_rot_cap:
                        continue
                    max_delta_mm = max(max_delta_mm, dt * 1000)
                    T_cam[i] = T_se2 @ T_cam[i]
                    fine_corr[i] = copy.deepcopy(fine_pcs[i])
                    fine_corr[i].transform(T_cam[i])
                tag = "bridge" if is_bridge_pass else "fine"
                print(f"  pass {it+1} [{tag}] corr={corr_dist*100:.1f}cm "
                      f"trans_cap={trans_cap*100:.1f}cm  max_cam_move={max_delta_mm:.2f}mm")
                if not is_bridge_pass and max_delta_mm < 0.3:
                    break

    elif not skip_plane:
        # legacy global plane snap
        source_pc = pre if plane_source == "prealign" else merged
        print(f"\n[INFO] Fitting global plane on {plane_source} cloud")
        source_ds = source_pc.voxel_down_sample(voxel_coarse)
        plane_model, inliers = source_ds.segment_plane(
            distance_threshold=plane_dist_thresh, ransac_n=3, num_iterations=3000
        )
        a, b, c, d = plane_model
        print(f"  plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0  "
              f"({len(inliers)}/{len(source_ds.points)} inliers)")
        T_plane_global = plane_to_z0_transform(plane_model)

    # rebuild final merged + per-camera PLYs from up-to-date T_cam
    final_merged = o3d.geometry.PointCloud()
    corrected_raw = []
    for i in range(n_cams):
        pc = copy.deepcopy(raw_pcs[i])
        pc.transform(T_plane_global @ T_cam[i])
        corrected_raw.append(pc)
        o3d.io.write_point_cloud(str(posts_dir / f"post_{i}.ply"), pc)
        final_merged += pc
    o3d.io.write_point_cloud(str(out_path / "postalign_global.ply"), final_merged)

    # write updated extrinsics
    updated = []
    for i, cam in enumerate(cams):
        ext = np.array(cam["transformation"]).reshape(4, 4)
        new_ext = T_plane_global @ T_cam[i] @ ext
        updated.append({
            "camera_id": cam.get("camera_id", i),
            "serial_number": cam["serial_number"],
            "transformation": new_ext.tolist(),
            "color_intrinsic_matrix": cam["color_intrinsic_matrix"],
            "depth_intrinsic_matrix": cam["depth_intrinsic_matrix"],
        })
    stem = extrinsic_file.stem
    out_yaml = extrinsic_file.parent / f"{stem}_global_aligned.yaml"
    with open(out_yaml, "w") as f:
        yaml.dump(updated, f, default_flow_style=False, sort_keys=False)

    # ---------- residual report ----------
    # For each cam: median nearest-neighbor distance from its points to the
    # union of all other cams' points (both in final plane-snapped frame).
    print("\n[INFO] Per-camera residual vs. union-of-others (lower = better)")
    ds_others_cache = [pc.voxel_down_sample(voxel_fine) for pc in corrected_raw]
    per_cam_residual = {}
    for i in range(n_cams):
        src_ds = ds_others_cache[i]
        tgt = o3d.geometry.PointCloud()
        for j in range(n_cams):
            if j == i:
                continue
            tgt += ds_others_cache[j]
        if len(src_ds.points) == 0 or len(tgt.points) == 0:
            per_cam_residual[i] = float("nan")
            continue
        kdt = o3d.geometry.KDTreeFlann(tgt)
        src_pts = np.asarray(src_ds.points)
        # cap sample count for speed
        if len(src_pts) > 5000:
            idx = np.random.choice(len(src_pts), 5000, replace=False)
            src_pts = src_pts[idx]
        dists = np.empty(len(src_pts))
        for k, p in enumerate(src_pts):
            _, _, d2 = kdt.search_knn_vector_3d(p, 1)
            dists[k] = np.sqrt(d2[0]) if len(d2) else np.nan
        per_cam_residual[i] = float(np.nanmedian(dists))

    sorted_cams = sorted(per_cam_residual.items(), key=lambda kv: (np.isnan(kv[1]), kv[1]))
    print(f"  {'cam':>4}  {'median NN (mm)':>16}  status")
    for i, d in sorted_cams:
        mm = d * 1000
        flag = "BAD " if d > 0.01 else ("warn" if d > 0.005 else "ok  ")
        print(f"  {i:>4}  {mm:>14.2f}    {flag}")

    print("\n[DONE]")
    print(f"  extrinsics : {out_yaml}")
    print(f"  merged PLY : {out_path / 'postalign_global.ply'}")
    print(f"  per-cam    : {posts_dir}")

    # ---------- interactive viewers ----------
    print("\n[INFO] Opening viewers — close each window to move on")
    palette = np.array([
        [0.90, 0.10, 0.10], [0.10, 0.70, 0.10], [0.10, 0.40, 0.90],
        [0.95, 0.75, 0.10], [0.70, 0.20, 0.80], [0.10, 0.75, 0.75],
        [0.95, 0.50, 0.15], [0.50, 0.50, 0.50],
    ])
    geoms = []
    for i, pc in enumerate(corrected_raw):
        vis_pc = copy.deepcopy(pc)
        vis_pc.paint_uniform_color(palette[i % len(palette)].tolist())
        geoms.append(vis_pc)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    o3d.visualization.draw_geometries(
        geoms, window_name="Per-camera colors"
    )

    # residual-colored viewer: green (good) -> red (bad), scaled by worst cam
    valid_res = [v for v in per_cam_residual.values() if not np.isnan(v)]
    vmax = max(valid_res) if valid_res else 1.0
    vmax = max(vmax, 0.005)  # avoid collapsing when everything is great
    res_geoms = []
    for i, pc in enumerate(corrected_raw):
        vis_pc = copy.deepcopy(pc)
        r = per_cam_residual.get(i, float("nan"))
        t = 0.0 if np.isnan(r) else min(1.0, r / vmax)
        color = [t, 1.0 - t, 0.1]  # green -> red
        vis_pc.paint_uniform_color(color)
        res_geoms.append(vis_pc)
    res_geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    o3d.visualization.draw_geometries(
        res_geoms,
        window_name=f"Residual heat (green=good, red=bad, scale=0..{vmax*1000:.1f}mm)"
    )

    o3d.visualization.draw_geometries(
        [final_merged], window_name="True colors"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Global multi-camera alignment (FPFH + ICP + pose graph + plane snap)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # accept both --cached_pc and --cached_pc_dir for consistency with 00-2
    ap.add_argument("--cached_pc", "--cached_pc_dir", dest="cached_pc", required=True)
    ap.add_argument("--extrinsic_file", required=True)
    ap.add_argument("--out_path", default=None)
    ap.add_argument("--ref_idx", type=int, default=6,
                    help="reference camera (its original world frame is preserved up to the plane snap)")
    ap.add_argument("--x_range", type=float, nargs=2, default=list(DEFAULT_X))
    ap.add_argument("--y_range", type=float, nargs=2, default=list(DEFAULT_Y))
    ap.add_argument("--z_range", type=float, nargs=2, default=list(DEFAULT_Z))
    ap.add_argument("--voxel_coarse", type=float, default=DEFAULT_VOXEL_COARSE)
    ap.add_argument("--voxel_fine", type=float, default=DEFAULT_VOXEL_FINE)
    ap.add_argument("--min_fitness", type=float, default=DEFAULT_MIN_FITNESS)
    ap.add_argument("--min_overlap_points", type=int, default=DEFAULT_MIN_OVERLAP_POINTS)
    ap.add_argument("--skip_plane", action="store_true",
                    help="skip final z=0 plane snap (keep reference-camera world frame)")
    ap.add_argument("--no_post_refine", action="store_true",
                    help="disable post-hoc per-camera ICP refinement against union-of-others")
    ap.add_argument("--use_ransac", action="store_true",
                    help="use FPFH+RANSAC global registration for each pair (risky on symmetric "
                         "scenes like flat tables — default is identity-init coarse->fine ICP)")
    ap.add_argument("--max_rot_deg", type=float, default=15.0,
                    help="reject any pair/node/refine correction whose rotation exceeds this")
    ap.add_argument("--max_trans_m", type=float, default=0.10,
                    help="reject any pair/node/refine correction whose translation exceeds this (meters)")
    ap.add_argument("--plane_source", choices=["prealign", "merged"], default="prealign",
                    help="[legacy global-plane mode] which cloud to fit the table plane on")
    ap.add_argument("--no_per_cam_plane_snap", action="store_true",
                    help="disable per-cam plane snap + SE(2) realign; fall back to single global plane")
    ap.add_argument("--plane_dist_thresh", type=float, default=0.008,
                    help="RANSAC distance threshold (m) for plane segmentation")
    ap.add_argument("--se2_realign_iters", type=int, default=5,
                    help="number of SE(2) realign passes after per-cam plane snap (0 = off)")
    ap.add_argument("--se2_bridge_dist", type=float, default=0.08,
                    help="first-pass SE(2) ICP correspondence distance (m); sets the "
                         "maximum inter-group xy offset that can be bridged")
    ap.add_argument("--se2_bridge_trans_m", type=float, default=0.25,
                    help="per-cam xy translation cap during bridge passes (m); later "
                         "fine passes revert to --max_trans_m")
    ap.add_argument("--loose_cams", type=int, nargs="*", default=[],
                    help="cam indices whose initial extrinsic is suspect — apply loose "
                         "sanity caps & wider SE(2) corr for these only (e.g. --loose_cams 0 4)")
    ap.add_argument("--loose_max_rot_deg", type=float, default=30.0)
    ap.add_argument("--loose_max_trans_m", type=float, default=0.30)
    ap.add_argument("--loose_se2_bridge_dist", type=float, default=0.20,
                    help="minimum SE(2) correspondence distance maintained for loose cams")
    args = ap.parse_args()

    main(
        cached_pc_dir=args.cached_pc,
        extrinsic_file=args.extrinsic_file,
        out_path=args.out_path,
        ref_idx=args.ref_idx,
        x_range=tuple(args.x_range),
        y_range=tuple(args.y_range),
        z_range=tuple(args.z_range),
        voxel_coarse=args.voxel_coarse,
        voxel_fine=args.voxel_fine,
        min_fitness=args.min_fitness,
        min_overlap=args.min_overlap_points,
        skip_plane=args.skip_plane,
        do_post_refine=not args.no_post_refine,
        use_ransac=args.use_ransac,
        max_rot_deg=args.max_rot_deg,
        max_trans_m=args.max_trans_m,
        plane_source=args.plane_source,
        per_cam_plane_snap=not args.no_per_cam_plane_snap,
        plane_dist_thresh=args.plane_dist_thresh,
        se2_realign_iters=args.se2_realign_iters,
        se2_bridge_dist=args.se2_bridge_dist,
        se2_bridge_trans_m=args.se2_bridge_trans_m,
        loose_cams=args.loose_cams,
        loose_max_rot_deg=args.loose_max_rot_deg,
        loose_max_trans_m=args.loose_max_trans_m,
        loose_se2_bridge_dist=args.loose_se2_bridge_dist,
    )
