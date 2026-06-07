#!/usr/bin/env bash
set -euo pipefail

# Smoke-test inference on the first 5 images in a folder.
#
# Usage:
#   ./Scripts/smoke_infer_5.sh [input_dir] [output_dir]
#
# Defaults:
#   input_dir  = test_images
#   output_dir = output_smoke_5
#
# Override models via env vars if needed:
#   BACKBONE=... HEAD=... MODEL_SIZE=... POSTPROCESS_BACKEND=cpu|metal VIZ_BACKEND=cpu|metal

INPUT_DIR="${1:-test_images}"
OUTPUT_DIR="${2:-output_smoke_5}"

BACKBONE="${BACKBONE:-Models/compiled/da3_backbone_giant_official.mlmodelc}"

# Prefer a float32 head when available (more stable for rays on some scenes).
DEFAULT_HEAD="Models/compiled/dualdpt_giant_da3.mlmodelc"
DEFAULT_HEAD_CPU_ONLY=""
if [ -d "Models/compiled_official_nope_f32/dualdpt_giant_da3_official_nope_f32.mlmodelc" ]; then
  DEFAULT_HEAD="Models/compiled_official_nope_f32/dualdpt_giant_da3_official_nope_f32.mlmodelc"
  DEFAULT_HEAD_CPU_ONLY="--head-cpu-only"
fi

HEAD="${HEAD:-$DEFAULT_HEAD}"
HEAD_CPU_ONLY="${HEAD_CPU_ONLY:-$DEFAULT_HEAD_CPU_ONLY}"
MODEL_SIZE="${MODEL_SIZE:-giant}"
POSTPROCESS_BACKEND="${POSTPROCESS_BACKEND:-cpu}"
VIZ_BACKEND="${VIZ_BACKEND:-cpu}"
CONFIDENCE_ACTIVATION="${CONFIDENCE_ACTIVATION:-linear}"

if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: input_dir not found: $INPUT_DIR" >&2
  exit 1
fi

if [ ! -d "$BACKBONE" ]; then
  echo "Error: backbone model not found: $BACKBONE" >&2
  exit 1
fi

if [ ! -d "$HEAD" ]; then
  echo "Error: head model not found: $HEAD" >&2
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

IMAGES=()
while IFS= read -r p; do
  IMAGES+=("$p")
done < <(
  find "$INPUT_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' -o -iname '*.heic' \) \
    | sort \
    | head -n 5
)

if [ "${#IMAGES[@]}" -eq 0 ]; then
  echo "Error: no images found in $INPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Running DA3CoreML smoke test:"
echo "- input:  $INPUT_DIR (${#IMAGES[@]} images)"
echo "- output: $OUTPUT_DIR"
echo "- backbone: $BACKBONE"
echo "- head:     $HEAD"
echo "- model-size: $MODEL_SIZE"
echo "- postprocess: $POSTPROCESS_BACKEND"
echo "- viz-backend: $VIZ_BACKEND"
echo "- confidence-activation: $CONFIDENCE_ACTIVATION"

"$CLI_BIN" infer \
  --backbone "$BACKBONE" \
  --head "$HEAD" \
  --model-size "$MODEL_SIZE" \
  $HEAD_CPU_ONLY \
  --confidence-activation "$CONFIDENCE_ACTIVATION" \
  --include-rays \
  --ray-viz \
  --ray-pose \
  --ray-pose-subsample 16 \
  --postprocess-backend "$POSTPROCESS_BACKEND" \
  --viz-backend "$VIZ_BACKEND" \
  --output "$OUTPUT_DIR" \
  -v \
  "${IMAGES[@]}"

echo "Done. Outputs in: $OUTPUT_DIR"
