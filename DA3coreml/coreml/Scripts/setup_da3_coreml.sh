#!/bin/bash
set -euo pipefail

# End-to-end helper: convert DINOv3 backbone + DA3 head to CoreML for all sizes,
# build the Swift CLI, generate mock inputs, and run a quick inference smoke test.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/Models"
BINARY="$PROJECT_DIR/.build/release/da3-coreml"

mkdir -p "$MODELS_DIR"

sizes=(giant large base small)

# Default HF ids (avoid bash 4 associative arrays for macOS default bash)
default_hf() {
    case "$1" in
        small) echo "facebook/dinov2-small" ;; # no official dinov3 small
        base)  echo "facebook/dinov3-vitb16-pretrain-lvd1689m" ;;
        large) echo "facebook/dinov3-vitl14" ;;
        giant) echo "facebook/dinov3-vitg14" ;;
        *)     echo "facebook/dinov3-vitb16-pretrain-lvd1689m" ;;
    esac
}

hf_model_for_size() {
    local size="$1"
    local key=$(echo "$size" | tr '[:lower:]' '[:upper:]')
    local env_specific="DA3_COREML_HF_MODEL_${key}"
    if [[ -n "${!env_specific:-}" ]]; then echo "${!env_specific}"; return; fi
    if [[ -n "${DA3_COREML_HF_MODEL:-}" ]]; then echo "${DA3_COREML_HF_MODEL}"; return; fi
    echo "$(default_hf "$size")"
}

ckpt_for_size() {
    local size="$1"
    local key=$(echo "$size" | tr '[:lower:]' '[:upper:]')
    local env_specific="DA3_CHECKPOINT_${key}"
    if [[ -n "${!env_specific:-}" ]]; then echo "${!env_specific}"; return; fi
    if [[ -n "${DA3_CHECKPOINT:-}" ]]; then echo "${DA3_CHECKPOINT}"; return; fi
    echo ""
}

convert_all() {
    export KMP_DUPLICATE_LIB_OK=TRUE
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

    for size in "${sizes[@]}"; do
        echo "\n=== Converting: $size ==="
        local hf_model ckpt
        hf_model=$(hf_model_for_size "$size")
        echo "Backbone HF model: $hf_model"
        python3 "$SCRIPT_DIR/convert_dinov3_to_coreml.py" \
            --model "$hf_model" \
            --output "$MODELS_DIR/dinov3_${size}.mlpackage" \
            --precision float16

        ckpt=$(ckpt_for_size "$size")
        if [[ -z "$ckpt" ]]; then
            echo "[WARN] No checkpoint for $size (set DA3_CHECKPOINT_$(echo "$size" | tr '[:lower:]' '[:upper:]') or DA3_CHECKPOINT). Skipping head export."
            continue
        fi
        if [[ ! -f "$ckpt" ]]; then
            echo "[WARN] Checkpoint not found at $ckpt; skipping head export for $size."
            continue
        fi

        python3 "$SCRIPT_DIR/convert_dualdpt_to_coreml.py" \
            --checkpoint "$ckpt" \
            --size "$size" \
            --output "$MODELS_DIR/dualdpt_${size}.mlpackage" \
            --precision float16
    done
}

build_swift() {
    echo "\n=== Building Swift package (release) ==="
    CLANG_MODULE_CACHE_PATH="$PROJECT_DIR/.clang-module-cache" \
    swift build -c release --cache-path "$PROJECT_DIR/.swiftpm-cache" --scratch-path "$PROJECT_DIR/.swiftpm-scratch"
}

make_mocks() {
    local mock_dir="$PROJECT_DIR/mock_inputs"
    mkdir -p "$mock_dir"
    python3 - <<'PY'
import os, struct
out_dir = 'mock_inputs'
os.makedirs(out_dir, exist_ok=True)

def clamp(v):
    return max(0, min(255, int(v)))

def write_bmp(path, width, height, rgb_fn):
    row_padded = (width * 3 + 3) & ~3
    image_size = row_padded * height
    file_size = 54 + image_size
    bmp = bytearray()
    bmp += b'BM'
    bmp += struct.pack('<I', file_size)
    bmp += b'\x00\x00\x00\x00'
    bmp += struct.pack('<I', 54)
    bmp += struct.pack('<I', 40)
    bmp += struct.pack('<i', width)
    bmp += struct.pack('<i', height)
    bmp += struct.pack('<H', 1)
    bmp += struct.pack('<H', 24)
    bmp += struct.pack('<I', 0)
    bmp += struct.pack('<I', image_size)
    bmp += struct.pack('<I', 2835)
    bmp += struct.pack('<I', 2835)
    bmp += struct.pack('<I', 0)
    bmp += struct.pack('<I', 0)
    for y in range(height):
        row = bytearray()
        for x in range(width):
            r, g, b = rgb_fn(x, y, width, height)
            row += bytes((clamp(b), clamp(g), clamp(r)))
        while len(row) < row_padded:
            row.append(0)
        bmp += row
    with open(path, 'wb') as f:
        f.write(bmp)

write_bmp(os.path.join(out_dir, 'mock_0.bmp'), 640, 360, lambda x,y,w,h: (255 * x / w, 128 + (y % 128), 200))
write_bmp(os.path.join(out_dir, 'mock_1.bmp'), 1024, 768, lambda x,y,w,h: (80, 255 * y / h, 255 * (x % 100) / 100))
print('Mocks in', out_dir)
PY
}

run_smoke_test() {
    local mock_dir="$PROJECT_DIR/mock_inputs"
    local out_dir="$PROJECT_DIR/output_smoke"
    mkdir -p "$out_dir"

    local chosen=""
    for size in "${sizes[@]}"; do
        if [[ -d "$MODELS_DIR/dinov3_${size}.mlpackage" && -d "$MODELS_DIR/dualdpt_${size}.mlpackage" ]]; then
            chosen="$size"
            break
        fi
    done

    if [[ -z "$chosen" ]]; then
        echo "[WARN] No matching backbone+head found; skipping smoke test."
        return
    fi

    echo "\n=== Smoke test with size: $chosen ==="
    "$BINARY" infer \
        --backbone "$MODELS_DIR/dinov3_${chosen}.mlpackage" \
        --head "$MODELS_DIR/dualdpt_${chosen}.mlpackage" \
        --output "$out_dir" \
        "$mock_dir/mock_0.bmp" "$mock_dir/mock_1.bmp"
}

convert_all
build_swift
make_mocks
run_smoke_test

echo "\nAll done."
