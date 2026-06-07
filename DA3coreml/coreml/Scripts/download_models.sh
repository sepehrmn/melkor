#!/bin/bash
set -euo pipefail

# Build and convert DA3 CoreML assets for multiple sizes (giant → large → base → small).
# Defaults use DINOv3-style HF ids; override with per-size env vars if needed.
# Required: set DA3_CHECKPOINT_<SIZE> (or DA3_CHECKPOINT) to a DA3 checkpoint for the head export.
# Optional: set DA3_COREML_HF_MODEL_<SIZE> or DA3_COREML_HF_MODEL to change backbone source.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/Models"

mkdir -p "$MODELS_DIR"

declare -A DEFAULT_HF
DEFAULT_HF[small]="facebook/dinov2-small"
DEFAULT_HF[base]="facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_HF[large]="facebook/dinov3-vitl14"
DEFAULT_HF[giant]="facebook/dinov3-vitg14"

sizes=(giant large base small)

function hf_model_for_size() {
    local size="$1"
    local key=${size^^}
    local env_specific="DA3_COREML_HF_MODEL_${key}"
    if [[ -n "${!env_specific:-}" ]]; then
        echo "${!env_specific}"
        return
    fi
    if [[ -n "${DA3_COREML_HF_MODEL:-}" ]]; then
        echo "${DA3_COREML_HF_MODEL}"
        return
    fi
    echo "${DEFAULT_HF[$size]}"
}

function ckpt_for_size() {
    local size="$1"
    local key=${size^^}
    local env_specific="DA3_CHECKPOINT_${key}"
    if [[ -n "${!env_specific:-}" ]]; then
        echo "${!env_specific}"
        return
    fi
    if [[ -n "${DA3_CHECKPOINT:-}" ]]; then
        echo "${DA3_CHECKPOINT}"
        return
    fi
    echo ""
}

cd "$PROJECT_DIR"

for size in "${sizes[@]}"; do
    echo "\n=== Converting size: $size ==="
    hf_model=$(hf_model_for_size "$size")
    echo "Backbone HF model: $hf_model"
    python3 "$SCRIPT_DIR/convert_dinov3_to_coreml.py" \
        --model "$hf_model" \
        --output "$MODELS_DIR/dinov3_${size}.mlpackage" \
        --precision float16

    ckpt_path=$(ckpt_for_size "$size")
    if [[ -z "$ckpt_path" ]]; then
        echo "[WARN] No checkpoint set for size $size (env DA3_CHECKPOINT_${size^^} or DA3_CHECKPOINT). Skipping head export."
        continue
    fi
    if [[ ! -f "$ckpt_path" ]]; then
        echo "[WARN] Checkpoint not found at $ckpt_path; skipping head export for $size."
        continue
    fi

    python3 "$SCRIPT_DIR/convert_dualdpt_to_coreml.py" \
        --checkpoint "$ckpt_path" \
        --size "$size" \
        --output "$MODELS_DIR/dualdpt_${size}.mlpackage" \
        --precision float16
done

echo "\n=== Building Swift package (release) ==="
CLANG_MODULE_CACHE_PATH=.clang-module-cache swift build -c release --cache-path .swiftpm-cache --scratch-path .swiftpm-scratch

echo "Done."
