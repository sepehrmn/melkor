#!/bin/bash
set -e

# ============================================================================
# DA3CoreML Setup Script
# ============================================================================
# Sets up the DA3CoreML environment and converts models to CoreML format.
#
# Usage:
#   ./Scripts/setup.sh [--size base] [--skip-build]
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
MODEL_SIZE="base"
SKIP_BUILD=false
SKIP_CONVERT=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1" >&2; }
log_step()    { echo -e "\n${CYAN}=== $1 ===${NC}"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-convert)
            SKIP_CONVERT=true
            shift
            ;;
        --help|-h)
            echo "DA3CoreML Setup"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --size <size>     Model size: small, base, large, giant (default: base)"
            echo "  --skip-build      Skip Swift build"
            echo "  --skip-convert    Skip model conversion"
            echo "  --help, -h        Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           DA3CoreML Setup - Depth-Anything-3 for Apple Silicon               ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

cd "$PROJECT_DIR"

# ============================================================================
# Check Prerequisites
# ============================================================================
log_step "Checking Prerequisites [1/4]"

# Check macOS
if [[ "$(uname -s)" != "Darwin" ]]; then
    log_error "DA3CoreML requires macOS"
    exit 1
fi

# Check Swift
if ! command -v swift &> /dev/null; then
    log_error "Swift not found. Install Xcode Command Line Tools:"
    log_info "  xcode-select --install"
    exit 1
fi
SWIFT_VERSION=$(swift --version 2>&1 | head -1)
log_success "Found Swift: $SWIFT_VERSION"

# Check Python
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 not found. Install with:"
    log_info "  brew install python3"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
log_success "Found Python: $PYTHON_VERSION"

# Check/install Python dependencies
log_info "Checking Python dependencies..."
MISSING_DEPS=()

python3 -c "import torch" 2>/dev/null || MISSING_DEPS+=("torch")
python3 -c "import coremltools" 2>/dev/null || MISSING_DEPS+=("coremltools")
python3 -c "import transformers" 2>/dev/null || MISSING_DEPS+=("transformers")
python3 -c "import numpy" 2>/dev/null || MISSING_DEPS+=("numpy")

if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
    log_warn "Missing Python packages: ${MISSING_DEPS[*]}"
    log_info "Installing..."
    pip3 install "${MISSING_DEPS[@]}"
fi

log_success "All Python dependencies available"

# ============================================================================
# Build Swift Package
# ============================================================================
if [[ "$SKIP_BUILD" == false ]]; then
    log_step "Building Swift Package [2/4]"
    
    log_info "Building in release mode..."
    swift build -c release
    
    log_success "Build complete"
    log_info "Binary: .build/release/da3-coreml"
else
    log_step "Skipping Swift Build [2/4]"
fi

# ============================================================================
# Convert Models to CoreML
# ============================================================================
if [[ "$SKIP_CONVERT" == false ]]; then
    log_step "Converting Models to CoreML [3/4]"
    
    mkdir -p Models
    
    # Map size to HuggingFace model (default to DINOv3, override with DA3_COREML_HF_MODEL or
    # DA3_COREML_USE_DINOV2=1 to force the older backbone)
    case $MODEL_SIZE in
        small)
            HF_MODEL_DEFAULT="facebook/dinov3-small"
            HF_MODEL_V2="facebook/dinov2-small"
            ;;
        base)
            HF_MODEL_DEFAULT="facebook/dinov3-base"
            HF_MODEL_V2="facebook/dinov2-base"
            ;;
        large)
            HF_MODEL_DEFAULT="facebook/dinov3-large"
            HF_MODEL_V2="facebook/dinov2-large"
            ;;
        giant)
            HF_MODEL_DEFAULT="facebook/dinov3-giant"
            HF_MODEL_V2="facebook/dinov2-giant"
            ;;
        *)
            log_error "Invalid model size: $MODEL_SIZE"
            exit 1
            ;;
    esac

    if [[ -n "$DA3_COREML_USE_DINOV2" ]]; then
        HF_MODEL="$HF_MODEL_V2"
    else
        HF_MODEL="$HF_MODEL_DEFAULT"
    fi

    if [[ -n "$DA3_COREML_HF_MODEL" ]]; then
        HF_MODEL="$DA3_COREML_HF_MODEL"
    fi

    log_info "Using backbone model: $HF_MODEL"
    
    log_info "Converting DINOv3 backbone (size: $MODEL_SIZE)..."
    log_info "This may take several minutes for large models..."
    
    python3 Scripts/convert_dinov3_to_coreml.py \
        --model "$HF_MODEL" \
        --output "Models/dinov3_${MODEL_SIZE}.mlpackage" \
        --precision float16
    
    log_success "DINOv3 backbone converted"
    
    log_info "Converting DualDPT head..."
    
    python3 Scripts/convert_dualdpt_to_coreml.py \
        --size "$MODEL_SIZE" \
        --output "Models/dualdpt_${MODEL_SIZE}.mlpackage" \
        --precision float16
    
    log_success "DualDPT head converted"
else
    log_step "Skipping Model Conversion [3/4]"
fi

# ============================================================================
# Verify Installation
# ============================================================================
log_step "Verifying Installation [4/4]"

# Check if models exist
if [[ -d "Models/dinov3_${MODEL_SIZE}.mlpackage" ]]; then
    log_success "DINOv3 model: Models/dinov3_${MODEL_SIZE}.mlpackage"
else
    log_warn "DINOv3 model not found - run without --skip-convert to create"
fi

if [[ -d "Models/dualdpt_${MODEL_SIZE}.mlpackage" ]]; then
    log_success "DualDPT model: Models/dualdpt_${MODEL_SIZE}.mlpackage"
else
    log_warn "DualDPT model not found - run without --skip-convert to create"
fi

# Check if binary exists
if [[ -f ".build/release/da3-coreml" ]]; then
    log_success "CLI binary: .build/release/da3-coreml"
else
    log_warn "CLI binary not found - run without --skip-build to create"
fi

# ============================================================================
# Success!
# ============================================================================
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    DA3CoreML Setup Complete! ✓                               ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
log_info "Usage:"
echo ""
echo "  # Run inference on an image"
echo "  .build/release/da3-coreml infer \\"
echo "      --backbone Models/dinov3_${MODEL_SIZE}.mlpackage \\"
echo "      --head Models/dualdpt_${MODEL_SIZE}.mlpackage \\"
echo "      --output ./output \\"
echo "      your_image.jpg"
echo ""
echo "  # Run benchmark"
echo "  .build/release/da3-coreml benchmark \\"
echo "      --backbone Models/dinov3_${MODEL_SIZE}.mlpackage \\"
echo "      --head Models/dualdpt_${MODEL_SIZE}.mlpackage"
echo ""
log_info "Documentation: README.md"
echo ""
