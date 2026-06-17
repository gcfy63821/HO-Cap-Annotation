#!/usr/bin/env python3
"""Build the top-level manifest.json from per-exp `_manifest.json` fragments.

Each exp folder (<bundle>/<task>/<exp>/_manifest.json) holds its own camera
entries and doubles as a "this exp is done" marker — so generation is resumable
(skip exps whose fragment exists) and parallel workers never collide.

Usage:
    python merge_manifests.py --bundle /path/to/bundle
"""
import argparse
import glob
import json
from pathlib import Path

SCHEMA_VERSION = 2
FRAGMENT = "_manifest.json"   # per-exp fragment filename


def build_manifest(bundle):
    """Combine every per-exp _manifest.json into <bundle>/manifest.json.
    Bundle nests as <videos_xxxx>/<exp> OR <videos_xxxx>/<task>/<exp>, so glob
    both the 2- and 3-level depths."""
    bundle = Path(bundle)
    frags = sorted(set(glob.glob(str(bundle / "*" / "*" / FRAGMENT))) |
                   set(glob.glob(str(bundle / "*" / "*" / "*" / FRAGMENT))))
    cameras, sam2_model, embed_dtype = [], None, "float16"
    for f in frags:
        m = json.loads(Path(f).read_text())
        sam2_model = sam2_model or m.get("sam2_model")
        embed_dtype = m.get("embed_dtype", embed_dtype)
        cameras.extend(m.get("cameras", []))
    cameras.sort(key=lambda c: (c["task"], c["exp"], c["cam_index"]))
    manifest = {"schema_version": SCHEMA_VERSION, "sam2_model": sam2_model,
                "embed_dtype": embed_dtype, "cameras": cameras}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2))
    exps = {(c["task"], c["exp"]) for c in cameras}
    return len(frags), len(cameras), len(exps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    args = ap.parse_args()
    nf, nc, ne = build_manifest(args.bundle)
    print(f"[merge] {nf} exp fragment(s) -> manifest.json: {nc} cameras, {ne} exps")


if __name__ == "__main__":
    main()
