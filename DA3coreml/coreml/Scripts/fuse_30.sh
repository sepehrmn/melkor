#!/usr/bin/env bash
set -euo pipefail

# Build a feed-forward DA3 3D Gaussian Splat (PLY) from ~30 images.
#
# Usage:
#   ./Scripts/fuse_30.sh [input_dir] [sampled_dir] [output_ply] [mode]
#
# Defaults:
#   input_dir   = sample_images
#   sampled_dir = sampled_30
#   output_ply  = fused_30.ply
#   mode        = camdec   (or: raypose)
#
# Notes:
# - This uses the *pre-trained GSHead* (DA3-style). It is not the simple depth-only `to3-dgs` path.
# - The sampled folder is created via relative symlinks and is ignored by git (`sampled_*/`).
# - You can override model paths via env vars: BACKBONE, HEAD, CAMDEC, GSHEAD, MODEL_SIZE, INPUT_SIZE.

INPUT_DIR="${1:-sample_images}"
SAMPLED_DIR="${2:-sampled_30}"
OUTPUT_PLY="${3:-fused_30.ply}"
MODE="${4:-camdec}" # camdec | raypose

BACKBONE="${BACKBONE:-Models/compiled/da3_backbone_giant_official.mlmodelc}"

# Prefer a float32 head when available (more stable for confidence/rays on some scenes).
DEFAULT_HEAD="Models/compiled/dualdpt_giant_da3.mlmodelc"
DEFAULT_HEAD_CPU_ONLY=""
if [ -d "Models/compiled_official_nope_f32/dualdpt_giant_da3_official_nope_f32.mlmodelc" ]; then
  DEFAULT_HEAD="Models/compiled_official_nope_f32/dualdpt_giant_da3_official_nope_f32.mlmodelc"
  DEFAULT_HEAD_CPU_ONLY="--head-cpu-only"
fi

HEAD="${HEAD:-$DEFAULT_HEAD}"
HEAD_CPU_ONLY="${HEAD_CPU_ONLY:-$DEFAULT_HEAD_CPU_ONLY}"
CAMDEC="${CAMDEC:-Models/compiled/camdec_giant.mlmodelc}"
GSHEAD="${GSHEAD:-Models/compiled/gshead_giant.mlmodelc}"
MODEL_SIZE="${MODEL_SIZE:-giant}"
INPUT_SIZE="${INPUT_SIZE:-518}"
CONFIDENCE_ACTIVATION="${CONFIDENCE_ACTIVATION:-linear}"
GS_SUBSAMPLE="${GS_SUBSAMPLE:-4}"
GS_MIN_CONFIDENCE="${GS_MIN_CONFIDENCE:-0.0}"
GS_OFFSET_DEPTH_SCALE="${GS_OFFSET_DEPTH_SCALE:-1.0}"
GS_DISABLE_OFFSET_DEPTH="${GS_DISABLE_OFFSET_DEPTH:-0}"

if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: input_dir not found: $INPUT_DIR" >&2
  exit 1
fi

for p in "$BACKBONE" "$HEAD" "$GSHEAD"; do
  if [ ! -d "$p" ]; then
    echo "Error: model not found: $p" >&2
    exit 1
  fi
done

if [ "$MODE" = "camdec" ]; then
  if [ ! -d "$CAMDEC" ]; then
    echo "Error: camdec model not found: $CAMDEC" >&2
    exit 1
  fi
elif [ "$MODE" != "raypose" ]; then
  echo "Error: mode must be 'camdec' or 'raypose' (got: $MODE)" >&2
  exit 1
fi

CLI_BIN=".build/release/da3-coreml"
if [ ! -x "$CLI_BIN" ]; then
  echo "Building CLI (release)..." >&2
  if ! swift build -c release; then
    echo "Release build failed; retrying with --disable-sandbox..." >&2
    swift build -c release --disable-sandbox
  fi
fi

if [ ! -x "$CLI_BIN" ]; then
  CLI_BIN=".build/debug/da3-coreml"
  if [ ! -x "$CLI_BIN" ]; then
    echo "Building CLI (debug)..." >&2
    swift build -c debug --disable-sandbox
  fi
fi

echo "Sampling 30 images from: $INPUT_DIR -> $SAMPLED_DIR"
python3 Scripts/sample_images.py "$INPUT_DIR" --count 30 --output "$SAMPLED_DIR"

IMAGES=()
while IFS= read -r p; do
  IMAGES+=("$p")
done < <(
  # `sample_images.py` creates symlinks by default. Use `-L` so symlinked images are treated as files.
  find -L "$SAMPLED_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.heic' \) \
    | sort
)
if [ "${#IMAGES[@]}" -eq 0 ]; then
  echo "Error: no images found in sampled_dir: $SAMPLED_DIR" >&2
  exit 1
fi

POSE_ARGS=()
if [ "$MODE" = "raypose" ]; then
  POSE_ARGS+=(--use-ray-pose)
else
  POSE_ARGS+=(--camdec "$CAMDEC")
fi

echo "Fusing 3DGS (mode=$MODE):"
echo "- images: ${#IMAGES[@]}"
echo "- output: $OUTPUT_PLY"
echo "- confidence-activation: $CONFIDENCE_ACTIVATION"

GS_OFFSET_ARGS=()
if [ "$GS_DISABLE_OFFSET_DEPTH" = "1" ]; then
  GS_OFFSET_ARGS+=(--gs-disable-offset-depth)
else
  GS_OFFSET_ARGS+=(--gs-offset-depth-scale "$GS_OFFSET_DEPTH_SCALE")
fi

"$CLI_BIN" fuse \
  --backbone "$BACKBONE" \
  --head "$HEAD" \
  $HEAD_CPU_ONLY \
  --gshead "$GSHEAD" \
  "${POSE_ARGS[@]}" \
  --confidence-activation "$CONFIDENCE_ACTIVATION" \
  --model-size "$MODEL_SIZE" \
  --input-size "$INPUT_SIZE" \
  --gs-subsample "$GS_SUBSAMPLE" \
  --gs-min-confidence "$GS_MIN_CONFIDENCE" \
  "${GS_OFFSET_ARGS[@]}" \
  --verbose \
  --output "$OUTPUT_PLY" \
  "${IMAGES[@]}"

echo "Done. Fused PLY: $OUTPUT_PLY"
