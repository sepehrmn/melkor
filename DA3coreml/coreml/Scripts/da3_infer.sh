#!/bin/bash
# =============================================================================
# DA3 CoreML Inference Script
# =============================================================================
# Easy-to-use wrapper for running DA3 depth inference on images.
#
# Usage:
#   ./Scripts/run_inference.sh <input_dir> [output_dir] [options]
#
# Examples:
#   # Process all images in a folder (sequential)
#   ./Scripts/run_inference.sh test_images output_depth
#
#   # Process with 4 parallel workers
#   ./Scripts/run_inference.sh test_images output_depth --parallel 4
#
#   # Include ray estimation
#   ./Scripts/run_inference.sh test_images output_depth --rays
#
#   # Sample first, then process
#   ./Scripts/run_inference.sh /path/to/images output_depth --sample 10 --parallel 4
# =============================================================================

set -e

# Default values
INPUT_DIR=""
OUTPUT_DIR="output_da3"
BACKBONE="Models/compiled/da3_backbone_giant_official.mlmodelc"
HEAD="Models/compiled/dualdpt_giant_da3.mlmodelc"
MODEL_SIZE="giant"
INCLUDE_RAYS=""
PARALLEL=1
SAMPLE_EVERY=0
VERBOSE=""
FORMAT="da3"
NO_PNG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "DA3 CoreML Inference Script"
            echo ""
            echo "Usage: $0 <input_dir> [output_dir] [options]"
            echo ""
            echo "Arguments:"
            echo "  input_dir     Directory containing input images"
            echo "  output_dir    Output directory (default: output_da3)"
            echo ""
            echo "Options:"
            echo "  --parallel N    Run N parallel workers (default: 1 = sequential)"
            echo "  --rays          Include ray estimation"
            echo "  --sample N      Sample every Nth image first"
            echo "  --backbone PATH Path to backbone model"
            echo "  --head PATH     Path to DualDPT head model"
            echo "  --model-size S  Model size: small, base, large, giant (default: giant)"
            echo "  --format F      Output format: da3, npy, raw, png (default: da3)"
            echo "  --no-png        Skip PNG visualization"
            echo "  -v, --verbose   Verbose output"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --rays)
            INCLUDE_RAYS="--include-rays"
            shift
            ;;
        --sample)
            SAMPLE_EVERY="$2"
            shift 2
            ;;
        --backbone)
            BACKBONE="$2"
            shift 2
            ;;
        --head)
            HEAD="$2"
            shift 2
            ;;
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --no-png)
            NO_PNG="--no-png"
            shift
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            if [ -z "$INPUT_DIR" ]; then
                INPUT_DIR="$1"
            elif [ "$OUTPUT_DIR" = "output_da3" ]; then
                OUTPUT_DIR="$1"
            fi
            shift
            ;;
    esac
done

# Validate input
if [ -z "$INPUT_DIR" ]; then
    echo "Error: Input directory required"
    echo "Usage: $0 <input_dir> [output_dir] [options]"
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

# Check models exist
if [ ! -d "$BACKBONE" ]; then
    echo "Error: Backbone model not found: $BACKBONE"
    echo "Convert + compile the official DA3 backbone first (see README.md Quick Start Option A)."
    exit 1
fi

if [ ! -d "$HEAD" ]; then
    echo "Error: Head model not found: $HEAD"
    echo "Run model conversion first."
    exit 1
fi

# Build CLI if needed (some sandboxed environments require disabling SwiftPM sandboxing).
CLI_BIN=".build/release/da3-coreml"
if [ ! -x "$CLI_BIN" ]; then
    echo "Building CLI (release)..."
    if ! swift build -c release; then
        echo "Release build failed; retrying with --disable-sandbox..."
        swift build -c release --disable-sandbox
    fi
fi
if [ ! -x "$CLI_BIN" ]; then
    CLI_BIN=".build/debug/da3-coreml"
    if [ ! -x "$CLI_BIN" ]; then
        echo "Building CLI (debug)..."
        swift build -c debug --disable-sandbox
    fi
fi

# Handle sampling
PROCESS_DIR="$INPUT_DIR"
if [ "$SAMPLE_EVERY" -gt 0 ]; then
    SAMPLED_DIR="${OUTPUT_DIR}_sampled"
    echo "========================================="
    echo "Sampling every ${SAMPLE_EVERY}th image"
    echo "========================================="

    python3 Scripts/sample_images.py "$INPUT_DIR" \
        --every "$SAMPLE_EVERY" \
        --output "$SAMPLED_DIR"

    PROCESS_DIR="$SAMPLED_DIR"
    echo ""
fi

# Get all images
IMAGES=($(ls "$PROCESS_DIR"/*.jpg "$PROCESS_DIR"/*.jpeg "$PROCESS_DIR"/*.png "$PROCESS_DIR"/*.JPG "$PROCESS_DIR"/*.JPEG "$PROCESS_DIR"/*.PNG 2>/dev/null | sort))
TOTAL=${#IMAGES[@]}

if [ $TOTAL -eq 0 ]; then
    echo "No images found in $PROCESS_DIR"
    exit 1
fi

echo "========================================="
echo "DA3 CoreML Inference"
echo "========================================="
echo "Input: $PROCESS_DIR ($TOTAL images)"
echo "Output: $OUTPUT_DIR"
echo "Model: $MODEL_SIZE"
echo "Workers: $PARALLEL"
[ -n "$INCLUDE_RAYS" ] && echo "Rays: enabled"
echo "========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inference
START_TIME=$(date +%s)

if [ "$PARALLEL" -eq 1 ]; then
    # Sequential processing
    echo "Processing images sequentially..."
    "$CLI_BIN" infer \
        --backbone "$BACKBONE" \
        --head "$HEAD" \
        --model-size "$MODEL_SIZE" \
        --output "$OUTPUT_DIR" \
        --format "$FORMAT" \
        $INCLUDE_RAYS \
        $NO_PNG \
        $VERBOSE \
        "${IMAGES[@]}"
else
    # Parallel processing
    echo "Launching $PARALLEL parallel workers..."

    # Calculate images per worker
    PER_WORKER=$((TOTAL / PARALLEL))
    REMAINDER=$((TOTAL % PARALLEL))

    PIDS=()
    START=0

    for ((i=0; i<PARALLEL; i++)); do
        # Calculate this worker's share
        COUNT=$PER_WORKER
        if [ $i -lt $REMAINDER ]; then
            COUNT=$((COUNT + 1))
        fi

        END=$((START + COUNT))

        # Get this worker's images
        WORKER_IMAGES=("${IMAGES[@]:$START:$COUNT}")

        if [ ${#WORKER_IMAGES[@]} -gt 0 ]; then
            echo "Worker $i: processing ${#WORKER_IMAGES[@]} images"

            # Run in background
            "$CLI_BIN" infer \
                --backbone "$BACKBONE" \
                --head "$HEAD" \
                --model-size "$MODEL_SIZE" \
                --output "$OUTPUT_DIR" \
                --format "$FORMAT" \
                $INCLUDE_RAYS \
                $NO_PNG \
                "${WORKER_IMAGES[@]}" &

            PIDS+=($!)
        fi

        START=$END
    done

    echo ""
    echo "Worker PIDs: ${PIDS[*]}"
    echo "Monitor: watch -n 5 'ls $OUTPUT_DIR/*.$FORMAT 2>/dev/null | wc -l'"
    echo ""

    # Wait for all workers
    echo "Waiting for workers to complete..."
    FAILED=0
    for pid in "${PIDS[@]}"; do
        if ! wait $pid; then
            FAILED=$((FAILED + 1))
        fi
    done

    if [ $FAILED -gt 0 ]; then
        echo "Warning: $FAILED worker(s) failed"
    fi
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Count results
COMPLETED=$(ls "$OUTPUT_DIR"/*.$FORMAT 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "========================================="
echo "Complete!"
echo "========================================="
echo "Processed: $COMPLETED / $TOTAL images"
echo "Time: ${ELAPSED}s ($(echo "scale=1; $ELAPSED / 60" | bc)m)"
if [ $COMPLETED -gt 0 ]; then
    echo "Average: $(echo "scale=2; $ELAPSED / $COMPLETED" | bc)s per image"
fi
echo "Output: $OUTPUT_DIR"
echo "========================================="
