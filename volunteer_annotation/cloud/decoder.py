#!/usr/bin/env python3
"""CPU-only SAM2 mask decoder for the cloud server.

Optimisations vs the original single-decoder design:
  - LRU embedding cache (default 8 slots, ~6 MB fp32 each): repeated previews
    on the same frame skip disk I/O entirely.
  - DecoderPool: N independent decoder instances served from a Queue.  Up to N
    preview requests run truly concurrently (no lock contention between pool
    members).  Pool size is controlled by the DECODER_POOL_SIZE env var (default 2).
"""

import os
import threading
from collections import OrderedDict
from pathlib import Path
from queue import Queue

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
_env_ckpt = os.environ.get("SAM2_CHECKPOINT")
DEFAULT_CKPT = Path(_env_ckpt) if _env_ckpt else REPO_ROOT / "mesh_reconstruction/sam2/checkpoints/sam2.1_hiera_large.pt"
DEFAULT_CFG = os.environ.get("SAM2_CFG", "configs/sam2.1/sam2.1_hiera_l.yaml")
POOL_SIZE = int(os.environ.get("DECODER_POOL_SIZE", "4"))
EMBED_CACHE_SIZE = int(os.environ.get("EMBED_CACHE_SIZE", "32"))


class Sam2CpuDecoder:
    """Single decoder with an LRU embedding cache."""

    def __init__(self, ckpt=DEFAULT_CKPT, cfg=DEFAULT_CFG, device="cpu",
                 embed_cache_size=EMBED_CACHE_SIZE):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = torch.device(device)
        model = build_sam2(cfg, str(ckpt), device=self.device)
        self.predictor = SAM2ImagePredictor(model)
        # LRU cache: key (str path) → feature dict with pre-converted tensors
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = embed_cache_size
        # lock makes load+predict atomic for this decoder instance
        self._lock = threading.Lock()

    def _load_into_predictor(self, embed_npz: Path):
        key = str(embed_npz)
        if key not in self._cache:
            z = np.load(embed_npz)
            to_t = lambda a: torch.from_numpy(a.astype(np.float32)).to(self.device)
            self._cache[key] = {
                "image_embed": to_t(z["image_embed"]).unsqueeze(0),
                "high_res_feats": [
                    to_t(z["high_res_feat_0"]).unsqueeze(0),
                    to_t(z["high_res_feat_1"]).unsqueeze(0),
                ],
                "orig_hw": [(int(z["orig_hw"][0]), int(z["orig_hw"][1]))],
            }
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)  # evict LRU entry
        else:
            self._cache.move_to_end(key)  # mark as recently used

        feat = self._cache[key]
        self.predictor._features = {
            "image_embed": feat["image_embed"],
            "high_res_feats": feat["high_res_feats"],
        }
        self.predictor._orig_hw = feat["orig_hw"]
        self.predictor._is_image_set = True

    def infer(self, embed_npz, points, labels, box=None):
        """Atomic load-from-cache + decode (thread-safe for this instance)."""
        with self._lock:
            self._load_into_predictor(embed_npz)
            return self._predict(points, labels, box)

    def _predict(self, points, labels, box=None):
        pc = np.asarray(points, dtype=np.float32) if len(points) else None
        pl = np.asarray(labels, dtype=np.int32) if len(labels) else None
        bx = np.asarray(box, dtype=np.float32) if box else None
        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=pc, point_labels=pl, box=bx, multimask_output=False
            )
        return masks[0].astype(bool), float(scores[0])


class DecoderPool:
    """Pool of N Sam2CpuDecoder instances.  Up to N previews run concurrently;
    additional requests block until a decoder becomes free."""

    def __init__(self, size: int = POOL_SIZE, **decoder_kwargs):
        self._pool: Queue = Queue()
        for _ in range(size):
            self._pool.put(Sam2CpuDecoder(**decoder_kwargs))
        self.size = size

    def infer(self, embed_npz, points, labels, box=None):
        dec = self._pool.get()
        try:
            return dec.infer(embed_npz, points, labels, box)
        finally:
            self._pool.put(dec)
