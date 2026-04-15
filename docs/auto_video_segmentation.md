# Auto video segmentation (rigid hand-held tools)

`tools/auto_video_segmentation.py` produces per-camera, per-frame masks for a
single rigid hand-held tool (pestle, hammer, screwdriver, …) from the 8-view
HOCAP RGB-D H5 file, **without any manual seed annotation**.

It mirrors the manual-seed + SAM2 video propagation pipeline in
`tools/01_video_segmentation.py`, but replaces the manual click step with a
two-phase automatic seed generator:

1. **Spatial prior** — backproject each camera's depth into the world frame,
   filter points inside a rough cylinder around the tool's approximate rest
   position, take the top-5 % Z points as a SAM2 positive click (tool held
   above the workspace) and the mortar / base center as a negative click. This
   almost always works for the 2–3 cleanest cameras.
2. **DINOv2 cross-view matching** — for every camera where the spatial prior
   fails or returns a questionable mask, extract dense DINOv2 patch features
   from the *good* cameras' seed masks, average them into one reference
   vector, and find the best-matching patch in the failing view. The peak
   patch becomes the SAM2 positive click for that camera.

Once each camera has a seed mask at *some* frame (not necessarily frame 0),
the **SAM2 video predictor** propagates it forward and backward through all
frames with temporal consistency. Frame 0's mask is produced by backward
propagation — it is **not** required to be a valid seed frame.

---

## Inputs

| What | Where | Needed for |
|---|---|---|
| HOCAP data H5 | `<sequence>/hocap/data00000000.h5` with `imgs (N, 8, H, W, 3)` and `depths (N, 8, H, W)` | segmentation |
| RealSense calibration yaml | produced by `00-0_align_cameras.py` | segmentation |
| SAM2 checkpoint | `config/checkpoints/sam2/sam2.1_hiera_large.pt` (downloaded via `scripts/download_models.sh`) | segmentation |
| Object mesh (.obj) + texture (.jpg) | `<models>/<object>/textured_mesh.obj`, `material_0.jpeg` | **downstream pose estimation only** (FoundationPose / ICP), not used by this script |

The object mesh and texture are **not loaded by the segmentation script**.
They are listed here because a full tool-tracking run also needs them later,
once masks are ready.

---

## The rough spatial cylinder

Four numbers describe a vertical cylinder in the world frame where the tool
is expected to be during active manipulation. You do not need a precise
location — anywhere the tool can plausibly be lifted to is fine, because the
pipeline scans all frames to pick its own "hero" seed frame per camera
(`find_best_seed_frame`).

| Flag | Meaning | Good default for a pestle over a mortar |
|---|---|---|
| `--xy_center X Y` | Cylinder center (world-frame x, y, meters) | `-0.07 -0.01` |
| `--xy_radius R` | Cylinder radius (meters) | `0.12` |
| `--z_min`, `--z_max` | Height band above the table plane (meters) | `0.10 0.30` |

These defaults catch a pestle that is being lifted 10–30 cm above the table
directly over the mortar. For other tools, pick a center roughly over the
workspace and a z-band that covers the object only *while it is being held
aloft* — you do not want it to catch the table surface or a tool lying flat.

---

## Single-pass run

```bash
conda run -n hocap-env python -m tools.auto_video_segmentation \
  --hocap_dir   /path/to/sequence/hocap \
  --calib_yaml  /path/to/realsense_calibration.yaml \
  --output_dir  /path/to/sequence/tool_masks_sam2video \
  --xy_center   -0.07 -0.01 \
  --xy_radius   0.12 \
  --z_min       0.10 \
  --z_max       0.30 \
  --good_cameras 0 5 7
```

`--good_cameras 0 5 7` tells the script: *"only trust spatial-prior seeds
from these camera indices; for every other camera, use these three as the
DINOv2 reference"*. Pick the side/top cameras that see the tool clearly and
are least occluded by the hand — 0, 5, 7 is a good starting point for the
standard 8-camera rig, but inspect `viz_seeds/seed_masks.png` (see below)
and adjust for your setup.

### Outputs

```
<output_dir>/
├── tmp_jpegs/                 # cached per-camera JPEGs (deleted at end)
├── viz_seeds/seed_masks.png   # 8-cam grid of the seeds actually used
├── masks.h5                   # (N_frames, 8, H, W) uint8, dataset "masks"
└── viz_masks/
    ├── masks_f0.png … masks_f700.png
    └── masks_video.mp4        # full 8-cam overlay video
```

Inspect `viz_seeds/seed_masks.png` first. If every green blob sits cleanly on
the tool, let the propagation finish. If one or two cameras look wrong, kill
the run and use the resume workflow below.

---

## Iterative `--resume` workflow (recommended)

Getting all 8 cameras right in one shot is rare — especially for cameras
where the hand or another tool lands inside the spatial cylinder. The
`--resume` flag is designed so you can rebuild **only the cameras that
failed**, without re-propagating the cameras that already worked.

How resume decides what to skip (see `main()` in the script): it opens the
existing `masks.h5` in append mode and, for each camera, counts how many
frames already have non-empty masks. If ≥ 90 % of frames are non-empty, the
camera is considered done and the script moves on. Everything else gets
re-seeded and re-propagated.

Typical loop:

```bash
# Pass 1 — cam0/5/7 spatial anchor, DINOv2 for the rest
conda run -n hocap-env python -m tools.auto_video_segmentation \
  --hocap_dir <…> --calib_yaml <…> --output_dir <…> \
  --xy_center -0.07 -0.01 --xy_radius 0.12 --z_min 0.10 --z_max 0.30 \
  --good_cameras 0 5 7

# → inspect viz_seeds/seed_masks.png and viz_masks/masks_video.mp4
# → say cam3 and cam6 are wrong; everything else is good.

# Pass 2 — rebuild cam3 and cam6 only, using a different DINOv2 reference
conda run -n hocap-env python -m tools.auto_video_segmentation \
  --hocap_dir <…> --calib_yaml <…> --output_dir <…> \
  --xy_center -0.07 -0.01 --xy_radius 0.12 --z_min 0.10 --z_max 0.30 \
  --cameras       3 6 \
  --good_cameras  0 5 7 \
  --resume
```

Flags that matter for iteration:

- `--cameras C1 C2 …` — restrict this pass to the named camera indices.
- `--good_cameras C1 C2 …` — cameras whose spatial-prior seed is trusted as
  the DINOv2 reference for all *other* cameras in this pass.
- `--resume` — open `masks.h5` in append mode and skip already-complete
  cameras (≥ 90 % non-empty frames).
- `--skip_extraction` — reuse the JPEGs from `tmp_jpegs/` that were
  extracted in the previous pass. Saves 1–2 minutes per pass.
- `--test_frame_only` — stop after generating and visualizing seed masks;
  do not run video propagation. Use this to iterate on `--xy_*` / `--z_*`
  / `--good_cameras` quickly before committing to a full propagation.

Keep iterating until every camera in `viz_seeds/seed_masks.png` looks right
and `viz_masks/masks_video.mp4` tracks the tool cleanly through the whole
video. The final `masks.h5` is the accumulation of every pass.

---

## Troubleshooting

**`NO SEED` appears for some cameras in `seed_masks.png`.**
Spatial prior found < 10 points in the cylinder for that camera, and DINOv2
either had no reference (no camera was in `--good_cameras`) or returned
similarity < 0.3. Add a camera that *did* get a seed to `--good_cameras`,
or widen `--xy_radius` / lower `--z_min` so the cylinder catches the tool
at its rest position.

**Seed mask covers the mortar / workspace instead of the tool.**
The `--z_min` is too low — the top-5 % Z points inside the cylinder are
landing on the workspace rim, not on an elevated tool. Raise `--z_min` so
the cylinder only includes space where the tool has been lifted.

**DINOv2 latches onto the hand or skin.**
The model used (`dinov2_vits14`) has trouble distinguishing stone-gray tools
from skin. Rerun with a different `--good_cameras` set so the reference
feature is computed from a view where the tool is clearly separated from
the hand. If nothing works, manually seed that one camera using
`tools/01_video_segmentation.py` and run this script with `--resume` for
the remaining cameras.

**A camera's mask drifts onto another object mid-video.**
SAM2 propagation lost the object. Re-run that camera with
`--cameras <idx> --resume` and a different `--good_cameras` set so the seed
is generated from a different hero frame. You can also narrow the problem
by pointing `--cameras` at just the bad camera and watching the propagation
log for the chunk where the mask area collapses to zero.

**masks.h5 has partial coverage after a killed run.**
Just re-run with `--resume`; any camera with < 90 % non-empty frames will
be re-done, fully-propagated cameras will be skipped.

---

## From masks to poses

Once `masks.h5` is clean, downstream pose estimation (e.g. FoundationPose or
the ICP-based PCA pipeline) consumes the masks together with the object
mesh (`textured_mesh.obj`) and its texture (`material_0.jpeg`) to produce
6-DoF object poses. The segmentation script has nothing to do with those
files — they are only mentioned here so you know where they fit in the full
pipeline.
