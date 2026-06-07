#!/usr/bin/env bash
set -euo pipefail

# Convert and compile the *official* DA3 DualDPT head to CoreML.
#
# This is a thin wrapper around `Scripts/convert_dualdpt_official_to_coreml.py` plus `xcrun coremlc compile`.
#
# Usage:
#   ./Scripts/convert_dualdpt_official.sh --checkpoint <path> [--size giant] [--precision float16] [--conf-activation expp1|linear] [--name <basename>]
#
# Examples:
#   # 1) Official behavior (may overflow in fp16 due to exp()):
#   ./Scripts/convert_dualdpt_official.sh --checkpoint ../src/checkpoints/da3_giant.safetensors --size giant --precision float16 --conf-activation expp1
#
#   # 2) Recommended for stability on Apple GPUs: export logits and apply activation outside CoreML:
#   ./Scripts/convert_dualdpt_official.sh --checkpoint ../src/checkpoints/da3_giant.safetensors --size giant --precision float16 --conf-activation linear --name dualdpt_giant_da3_official_logits_f16
#

CHECKPOINT=""
SIZE="giant"
INPUT_SIZE="518"
PATCH_SIZE="14"
PRECISION="float16"
CONF_ACTIVATION="expp1"
NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --size) SIZE="$2"; shift 2 ;;
    --input-size) INPUT_SIZE="$2"; shift 2 ;;
    --patch-size) PATCH_SIZE="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --conf-activation) CONF_ACTIVATION="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,60p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "Error: --checkpoint is required" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Error: checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

if [[ -z "$NAME" ]]; then
  # Default naming scheme
  NAME="dualdpt_${SIZE}_official_${CONF_ACTIVATION}_${PRECISION}"
fi

OUT_MLPACKAGE="Models/converted/${NAME}.mlpackage"
OUT_COMPILED_DIR="Models/compiled_${NAME}"

mkdir -p "Models/converted"
mkdir -p "$OUT_COMPILED_DIR"

echo "Converting DualDPT (official) -> CoreML:"
echo "- checkpoint: $CHECKPOINT"
echo "- size: $SIZE"
echo "- input: ${INPUT_SIZE} patch=${PATCH_SIZE}"
echo "- precision: $PRECISION"
echo "- conf_activation: $CONF_ACTIVATION"
echo "- out: $OUT_MLPACKAGE"

python3 Scripts/convert_dualdpt_official_to_coreml.py \
  --checkpoint "$CHECKPOINT" \
  --output "$OUT_MLPACKAGE" \
  --size "$SIZE" \
  --input-size "$INPUT_SIZE" \
  --patch-size "$PATCH_SIZE" \
  --precision "$PRECISION" \
  --conf-activation "$CONF_ACTIVATION"

echo ""
echo "Compiling to .mlmodelc:"
echo "- compiled dir: $OUT_COMPILED_DIR"
xcrun coremlc compile "$OUT_MLPACKAGE" "$OUT_COMPILED_DIR"

echo ""
echo "Done."
echo "- mlpackage: $OUT_MLPACKAGE"
echo "- mlmodelc:  $OUT_COMPILED_DIR/$(basename "$OUT_MLPACKAGE" .mlpackage).mlmodelc"

