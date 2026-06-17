#!/usr/bin/env python3
"""CPU-only SAM2 mask decoder for the cloud server.

This is the validated seam (architecture step 1): given an embedding precomputed
on a GPU box, inject it into a SAM2ImagePredictor and decode point prompts WITHOUT
running the image encoder. No GPU required. The decode-from-stored-embedding path
reproduces native SAM2 at IoU ~ 0.9999 (see internal/validate_seam.py).

The SAM2 model is loaded once (its encoder weights sit unused in RAM); each
preview just runs the tiny mask decoder, which is ms-level on CPU.
"""

import threading
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT = REPO_ROOT / "mesh_reconstruction/sam2/checkpoints/sam2.1_hiera_large.pt"
DEFAULT_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"


class Sam2CpuDecoder:
    def __init__(self, ckpt=DEFAULT_CKPT, cfg=DEFAULT_CFG, device="cpu"):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = torch.device(device)
        model = build_sam2(cfg, str(ckpt), device=self.device)
        self.predictor = SAM2ImagePredictor(model)
        self._loaded_key = None
        # The predictor holds ONE embedding at a time and is shared across
        # concurrent requests (FastAPI runs sync endpoints in a threadpool), so
        # load+predict MUST be atomic — otherwise one request's predict() can run
        # against another request's just-loaded embedding (wrong-view masks).
        self._lock = threading.Lock()

    def load_embedding(self, embed_npz):
        """Inject a precomputed embedding, bypassing the image encoder.

        Cheap to call repeatedly for the same frame (no-op if already loaded),
        so the cloud can keep the volunteer's current frame hot across previews.
        """
        if self._loaded_key == str(embed_npz):
            return
        z = np.load(embed_npz)
        to_t = lambda a: torch.from_numpy(a.astype(np.float32)).to(self.device)
        self.predictor._features = {
            "image_embed": to_t(z["image_embed"]).unsqueeze(0),
            "high_res_feats": [
                to_t(z["high_res_feat_0"]).unsqueeze(0),
                to_t(z["high_res_feat_1"]).unsqueeze(0),
            ],
        }
        h, w = int(z["orig_hw"][0]), int(z["orig_hw"][1])
        self.predictor._orig_hw = [(h, w)]
        self.predictor._is_image_set = True
        self._loaded_key = str(embed_npz)

    def infer(self, embed_npz, points, labels, box=None):
        """Atomic load-embedding + decode (safe under concurrent requests)."""
        with self._lock:
            self.load_embedding(embed_npz)
            return self.predict(points, labels, box)

    def predict(self, points, labels, box=None):
        """Decode one object. Returns (mask HxW bool, score float)."""
        pc = np.asarray(points, dtype=np.float32) if len(points) else None
        pl = np.asarray(labels, dtype=np.int32) if len(labels) else None
        bx = np.asarray(box, dtype=np.float32) if box else None
        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=pc, point_labels=pl, box=bx, multimask_output=False
            )
        return masks[0].astype(bool), float(scores[0])
