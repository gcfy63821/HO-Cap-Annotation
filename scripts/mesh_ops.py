#!/usr/bin/env python3
"""
Mesh inspection / cleaning / decimation utility.

Single-stop tool for the common mesh-prep needs around this codebase:
  * Inspect: vertex/face counts, bbox, NaN/Inf, OOB face indices, watertight,
             face/vert ratio (suspicious if ≈1).
  * Clean  : remove degenerate faces, drop unreferenced vertices, merge
             duplicate vertices. Fixes the kind of OBJ that crashes
             nvdiffrast with "CUDA error 700: illegal memory access".
  * Decimate: trimesh quadric edge collapse to a target vertex count or
              ratio. Cheaper / more robust than SeamAwareDecimater for
              meshes used in 6D pose tracking (no UV-seam preservation).
  * Batch  : apply the same ops to every *.obj under a folder (recurses).

Examples:

    # Just look at a mesh.
    python scripts/mesh_ops.py --input /path/mesh.obj --inspect

    # Clean + write to <name>_cleaned.obj (default sidecar naming).
    python scripts/mesh_ops.py --input /path/mesh.obj --clean

    # Clean + decimate to 30 000 vertices, in-place (backs up to *.original).
    python scripts/mesh_ops.py --input /path/mesh.obj --clean --target_verts 30000 --in_place

    # Decimate to 30 % of original face count, write to /tmp/out.obj.
    python scripts/mesh_ops.py --input /path/mesh.obj --target_ratio 0.30 --output /tmp/out.obj

    # Batch: clean + cap every model to ≤ 30 k verts, in-place with backup.
    python scripts/mesh_ops.py \\
        --batch /viscam/u/chenrq/models \\
        --clean --target_verts 30000 \\
        --in_place --pattern '**/textured_mesh.obj'
"""

import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import trimesh


# ============================================================
# Inspection
# ============================================================

def inspect(mesh, label='mesh'):
    """Print a one-screen sanity summary. Returns dict of metrics."""
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.faces)
    out = {
        'verts': len(V),
        'faces': len(F),
        'face_vert_ratio': len(F) / max(1, len(V)),
        'oob_face_idx': bool(F.size and (F.max() >= len(V) or F.min() < 0)),
        'nan_verts': bool(V.size and (np.isnan(V).any() or np.isinf(V).any())),
        'extent_m': mesh.extents.tolist() if hasattr(mesh, 'extents') else None,
        'watertight': bool(getattr(mesh, 'is_watertight', False)),
    }
    try:
        deg = (mesh.area_faces < 1e-12).sum() if len(mesh.area_faces) else 0
        out['degenerate_faces'] = int(deg)
    except Exception:
        out['degenerate_faces'] = None

    print(f'  [inspect] {label}')
    print(f'    verts            : {out["verts"]:>10,}')
    print(f'    faces            : {out["faces"]:>10,}   '
          f'(F/V ratio = {out["face_vert_ratio"]:.2f}'
          + ('  ⚠️ <1.5 unusual' if out['face_vert_ratio'] < 1.5 else '')
          + ')')
    print(f'    OOB face idx     : {"⚠️  YES" if out["oob_face_idx"] else "no"}')
    print(f'    NaN/Inf verts    : {"⚠️  YES" if out["nan_verts"] else "no"}')
    print(f'    degen faces      : {out["degenerate_faces"]}')
    print(f'    extent (m)       : {out["extent_m"]}')
    print(f'    watertight       : {out["watertight"]}')
    return out


# ============================================================
# Clean
# ============================================================

def clean(mesh):
    """Standard repair: drop degen + unreferenced + merge dups + remove
    invalid attributes that confuse nvdiffrast. In-place; returns mesh.
    trimesh's API for these has shifted across versions, so try a few
    variants."""
    n_v0, n_f0 = len(mesh.vertices), len(mesh.faces)

    # Degenerate faces: try old method, then new update_faces() pattern.
    try:
        mesh.remove_degenerate_faces()                              # ≤ trimesh 3.x
    except (AttributeError, TypeError):
        try:
            keep = mesh.nondegenerate_faces(height=1e-12)           # newer prop helper
            mesh.update_faces(keep)
        except (AttributeError, TypeError):
            try:
                # Manual fallback: face area > 0
                ok = mesh.area_faces > 1e-12
                mesh.update_faces(ok)
            except Exception as e:
                print(f'    [clean] could not drop degenerate faces: {e}')

    # Unreferenced verts
    try:
        mesh.remove_unreferenced_vertices()
    except Exception as e:
        print(f'    [clean] remove_unreferenced_vertices failed: {e}')

    # Merge duplicates
    try:
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
    except TypeError:
        mesh.merge_vertices()
    except Exception as e:
        print(f'    [clean] merge_vertices failed: {e}')

    print(f'    [clean] verts {n_v0:,} → {len(mesh.vertices):,}   '
          f'faces {n_f0:,} → {len(mesh.faces):,}')
    return mesh


# ============================================================
# Decimate
# ============================================================

def decimate_to_verts(mesh, target_verts):
    """Quadric edge collapse to ~target_verts. trimesh's API takes a face
    ratio rather than an exact vertex target, so we approximate (face count
    drops roughly proportional to vertex count for typical 2-manifold meshes)."""
    n_v0, n_f0 = len(mesh.vertices), len(mesh.faces)
    if n_v0 <= target_verts:
        print(f'    [decimate] already ≤ target ({n_v0:,} ≤ {target_verts:,}), skipping')
        return mesh
    ratio = target_verts / max(1, n_v0)
    try:
        m2 = mesh.simplify_quadric_decimation(face_count=int(n_f0 * ratio))
    except TypeError:
        # Older trimesh: API was simplify_quadric_decimation(percent_or_count)
        m2 = mesh.simplify_quadric_decimation(ratio)
    print(f'    [decimate] target_verts={target_verts:,}  '
          f'(ratio≈{ratio:.3f})  →  V {n_v0:,}→{len(m2.vertices):,}  '
          f'F {n_f0:,}→{len(m2.faces):,}')
    return m2


def decimate_to_ratio(mesh, ratio):
    """Direct ratio passthrough. ratio in (0, 1]."""
    n_v0, n_f0 = len(mesh.vertices), len(mesh.faces)
    if not (0 < ratio <= 1):
        raise ValueError(f'--target_ratio must be in (0, 1], got {ratio}')
    if ratio >= 0.999:
        print(f'    [decimate] ratio={ratio} ≈ 1, skipping')
        return mesh
    try:
        m2 = mesh.simplify_quadric_decimation(face_count=int(n_f0 * ratio))
    except TypeError:
        m2 = mesh.simplify_quadric_decimation(ratio)
    print(f'    [decimate] ratio={ratio}  →  V {n_v0:,}→{len(m2.vertices):,}  '
          f'F {n_f0:,}→{len(m2.faces):,}')
    return m2


# ============================================================
# Single-mesh driver
# ============================================================

def process_single(input_path, output_path=None, in_place=False, keep_original=True,
                   do_inspect=True, do_clean=False, target_verts=None,
                   target_ratio=None):
    input_path = Path(input_path)
    if not input_path.exists():
        print(f'[ERROR] input not found: {input_path}')
        return False

    print(f'\n========== {input_path} ==========')
    t0 = time.time()
    mesh = trimesh.load(str(input_path), force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        print(f'[ERROR] {input_path} did not load as a single Trimesh '
              f'(type: {type(mesh).__name__}). Skipping.')
        return False
    print(f'  loaded in {time.time()-t0:.1f}s')

    if do_inspect:
        inspect(mesh, label='input')

    # If only inspecting, stop here.
    if not (do_clean or target_verts is not None or target_ratio is not None):
        return True

    if do_clean:
        mesh = clean(mesh)

    if target_verts is not None:
        mesh = decimate_to_verts(mesh, int(target_verts))
    elif target_ratio is not None:
        mesh = decimate_to_ratio(mesh, float(target_ratio))

    if do_inspect:
        inspect(mesh, label='output')

    # Determine output path.
    if in_place:
        if keep_original:
            backup = input_path.with_suffix(input_path.suffix + '.original')
            if not backup.exists():
                shutil.copy2(input_path, backup)
                print(f'  [backup] {backup}')
            else:
                print(f'  [backup] (already exists, not overwriting): {backup}')
        out = input_path
    elif output_path is not None:
        out = Path(output_path)
    else:
        # Default sidecar name
        suffix_parts = []
        if do_clean: suffix_parts.append('clean')
        if target_verts is not None: suffix_parts.append(f'v{int(target_verts)//1000}k')
        elif target_ratio is not None: suffix_parts.append(f'r{int(round(target_ratio*100))}')
        suffix = '_' + '_'.join(suffix_parts) if suffix_parts else '_processed'
        out = input_path.with_name(input_path.stem + suffix + input_path.suffix)

    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    print(f'  [write] {out}  ({out.stat().st_size/1024:.0f} KB)')
    print(f'  total time: {time.time()-t0:.1f}s')
    return True


# ============================================================
# Batch driver
# ============================================================

def process_batch(batch_root, pattern, **kwargs):
    batch_root = Path(batch_root)
    if not batch_root.is_dir():
        print(f'[ERROR] --batch root is not a directory: {batch_root}')
        return 1

    files = sorted(batch_root.glob(pattern))
    if not files:
        print(f'[ERROR] no files matched {pattern} under {batch_root}')
        return 1

    print(f'[batch] root    : {batch_root}')
    print(f'[batch] pattern : {pattern}')
    print(f'[batch] matched : {len(files)} file(s)')
    print()

    n_ok = 0; n_fail = 0
    for f in files:
        try:
            ok = process_single(f, **kwargs)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
        except Exception as e:
            print(f'[ERROR] {f}: {e}')
            traceback.print_exc()
            n_fail += 1

    print(f'\n[batch] done. ok={n_ok} fail={n_fail} / {len(files)}')
    return 0 if n_fail == 0 else 1


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description='Inspect / clean / decimate mesh files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--input', type=str, help='Single mesh file')
    src.add_argument('--batch', type=str, help='Directory to walk')
    ap.add_argument('--pattern', type=str, default='**/*.obj',
                    help='Glob pattern when --batch is given (default: **/*.obj)')

    # Operations
    ap.add_argument('--inspect', action='store_true', default=False,
                    help='Print stats. Default on when no other op given.')
    ap.add_argument('--clean', action='store_true',
                    help='Drop degenerate faces, unreferenced verts, merge duplicates.')
    ap.add_argument('--target_verts', type=int, default=None,
                    help='Decimate to approximately this many vertices.')
    ap.add_argument('--target_ratio', type=float, default=None,
                    help='Decimate to this face-ratio (0,1].')

    # Output control
    out_group = ap.add_mutually_exclusive_group()
    out_group.add_argument('--output', type=str, default=None,
                            help='Single-mesh output path (only with --input).')
    out_group.add_argument('--in_place', action='store_true',
                            help='Overwrite source. By default the original is '
                                 'backed up to <name>.original (disable with '
                                 '--no_backup).')
    ap.add_argument('--no_backup', action='store_true',
                    help='Skip *.original backup when --in_place. Dangerous.')

    args = ap.parse_args()

    if args.target_verts is not None and args.target_ratio is not None:
        ap.error('--target_verts and --target_ratio are mutually exclusive')

    # If user gave NO operation, default to inspect-only.
    has_any_op = args.clean or (args.target_verts is not None) or (args.target_ratio is not None) or args.inspect
    if not has_any_op:
        args.inspect = True
    # Always inspect alongside ops unless explicitly opted out (we default it on
    # so the user sees before/after; cheap and informative).
    args.inspect = True

    if args.output is not None and args.batch is not None:
        ap.error('--output cannot be used with --batch')

    common = dict(
        do_inspect=args.inspect,
        do_clean=args.clean,
        target_verts=args.target_verts,
        target_ratio=args.target_ratio,
        in_place=args.in_place,
        keep_original=(not args.no_backup),
    )

    if args.input is not None:
        ok = process_single(args.input, output_path=args.output, **common)
        sys.exit(0 if ok else 1)
    else:
        rc = process_batch(args.batch, args.pattern, **common)
        sys.exit(rc)


if __name__ == '__main__':
    main()
