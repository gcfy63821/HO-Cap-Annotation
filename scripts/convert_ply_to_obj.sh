#!/bin/bash
# Convert SAM3D PLY point clouds to cleaned_mesh_10000.obj
#
# Usage:
#   bash scripts/convert_ply_to_obj.sh                          # process all PLYs in data/new_models
#   bash scripts/convert_ply_to_obj.sh --scale 0.15             # apply scale factor
#   bash scripts/convert_ply_to_obj.sh --input data/new_models/bottle.ply --scale 0.2

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

INPUT_DIR="data/new_models"
OUTPUT_DIR="data/models"

python preprocess/ply_to_obj.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    "$@"
