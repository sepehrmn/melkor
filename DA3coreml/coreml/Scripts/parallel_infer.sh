#!/bin/bash
# Run DA3 inference in parallel across multiple processes
# Usage: ./Scripts/parallel_infer.sh <input_dir> <output_dir> [num_workers]

set -e

INPUT_DIR="${1:-test_images}"
OUTPUT_DIR="${2:-output_parallel}"
NUM_WORKERS="${3:-4}"

BACKBONE="Models/compiled/da3_backbone_giant_official.mlmodelc"
HEAD="Models/compiled/dualdpt_giant_da3.mlmodelc"

# Check models exist
if [ ! -d "$BACKBONE" ] || [ ! -d "$HEAD" ]; then
    echo "Error: Models not found. Make sure you've compiled them first."
    echo "See README.md Quick Start Option A for DA3 (official) backbone export."
    exit 1
fi

# Build CLI if needed
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

# Get all images
IMAGES=($(ls "$INPUT_DIR"/*.jpg "$INPUT_DIR"/*.jpeg "$INPUT_DIR"/*.png 2>/dev/null | sort))
TOTAL=${#IMAGES[@]}

if [ $TOTAL -eq 0 ]; then
    echo "No images found in $INPUT_DIR"
    exit 1
fi

echo "========================================"
echo "DA3 Parallel Inference"
echo "========================================"
echo "Input: $INPUT_DIR ($TOTAL images)"
echo "Output: $OUTPUT_DIR"
echo "Workers: $NUM_WORKERS"
echo "Images per worker: $((TOTAL / NUM_WORKERS))"
echo "========================================"

# Create output dir
mkdir -p "$OUTPUT_DIR"

# Calculate images per worker
PER_WORKER=$((TOTAL / NUM_WORKERS))
REMAINDER=$((TOTAL % NUM_WORKERS))

# Launch workers
PIDS=()
START=0

for ((i=0; i<NUM_WORKERS; i++)); do
    # Calculate this worker's share
    COUNT=$PER_WORKER
    if [ $i -lt $REMAINDER ]; then
        COUNT=$((COUNT + 1))
    fi

    END=$((START + COUNT))

    # Get this worker's images
    WORKER_IMAGES=("${IMAGES[@]:$START:$COUNT}")

    if [ ${#WORKER_IMAGES[@]} -gt 0 ]; then
        echo "Worker $i: processing ${#WORKER_IMAGES[@]} images (indices $START-$((END-1)))"

        # Run in background
        "$CLI_BIN" infer \
            --backbone "$BACKBONE" \
            --head "$HEAD" \
            --model-size giant \
            --output "$OUTPUT_DIR" \
            --include-rays \
            "${WORKER_IMAGES[@]}" &

        PIDS+=($!)
    fi

    START=$END
done

echo ""
echo "Launched ${#PIDS[@]} workers. PIDs: ${PIDS[*]}"
echo "Monitor with: watch -n 5 'ls $OUTPUT_DIR/*.da3 2>/dev/null | wc -l'"
echo ""

# Wait for all workers
echo "Waiting for all workers to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid
done

COMPLETED=$(ls "$OUTPUT_DIR"/*.da3 2>/dev/null | wc -l)
echo ""
echo "========================================"
echo "Done! Processed $COMPLETED images"
echo "Output in: $OUTPUT_DIR"
echo "========================================"
