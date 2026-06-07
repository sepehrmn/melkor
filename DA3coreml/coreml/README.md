# DA3CoreML - Depth-Anything-3 for Apple Silicon

A pure Swift/CoreML implementation of Depth-Anything-3 (DA3) for macOS and iOS. Runs on Apple Neural Engine/GPU when supported by the exported models — **no CUDA required** (CPU fallback is used when necessary for stability or unsupported ops).

**Status:** Production-ready with DINOv2 backbone (official DA3 weights). DINOv3 backbone pipeline ready for future use.

---

## Table of Contents

1. [Features](#features)
2. [Related Work](#related-work)
3. [Backbone Options](#backbone-options)
4. [Architecture Overview](#architecture-overview)
5. [Conventions & Semantics](#conventions--semantics)
6. [Requirements](#requirements)
7. [Installation](#installation)
8. [Quick Start](#quick-start)
9. [API Reference](#api-reference)
10. [CLI Reference](#cli-reference)
11. [Memory Management](#memory-management)
12. [Model Information](#model-information)
13. [Known Issues & Fixes](#known-issues--fixes)
14. [Design Decisions](#design-decisions)
15. [File Structure](#file-structure)
16. [3D Gaussian Splatting (3DGS)](#3d-gaussian-splatting-3dgs)
17. [Contributing](#contributing)
18. [TODO](#todo)
19. [Troubleshooting](#troubleshooting)

---

## Features

- **Pure CoreML** - Runs on Apple Neural Engine + GPU (with CPU fallback when required)
- **DINOv2 Backbone** - Official DA3 weights with pre-trained DualDPT head
- **DINOv3 Support** - Pipeline ready for future DINOv3+DualDPT models
- **DualDPT Head** - Depth and ray prediction with confidence
- **Float32 Head Option** - CPU-only DualDPT heads for maximum numerical stability (especially rays)
- **Smart Memory Management** - Automatic RAM detection with configurable safety buffer
- **Adaptive Batching** - Dynamic batch sizing based on available memory
- **Memory Pressure Monitoring** - Automatic throttling under memory pressure
- **Tiled Inference** - Process images larger than available RAM
- **Metal Postprocess (Optional)** - GPU-accelerated crop/resize + tile blending (“tallying”) + stable confidence activation/visualization in float32
- **Visualization** - Built-in colormaps (spectral, turbo, viridis, plasma)
- **Float16 Precision** - Optimized for Apple Silicon

---

## Related Work

The upstream Depth-Anything-3 repository does **not** (currently) provide official CoreML / Apple Silicon exports.

Community references:
- **CoreML request / discussion**: ByteDance-Seed/Depth-Anything-3 issue #63 (CoreML / ExecuTorch)  
  https://github.com/ByteDance-Seed/Depth-Anything-3/issues/63
- **CoreML attempt (Muna)**: a community gist that compiles DA3 Metric-Large with Muna’s compiler/runtime (not a standard `.mlmodelc` drop-in).  
  https://gist.github.com/olokobayusuf/bebf8fe0bd766d51dcbb16e11b815926

If you find another working export (or an upstream release), please open an issue/PR with a link and notes on precision (fp16/fp32), outputs (depth/rays/conf), and licensing.

---

## Backbone Options

### DINOv2 (Recommended - Official DA3)

The official DA3 checkpoints use **DINOv2** as the backbone. Pre-trained weights are available on HuggingFace:

| Model | Backbone | Hidden Dim | Patch Size | HuggingFace |
|-------|----------|------------|------------|-------------|
| DA3-Small | DINOv2-S | 384 | 14 | `depth-anything/DA3-SMALL` |
| DA3-Base | DINOv2-B | 768 | 14 | `depth-anything/DA3-BASE` |
| DA3-Large | DINOv2-L | 1024 | 14 | `depth-anything/DA3-LARGE` |
| DA3-Giant | DINOv2-G | 1536 | 14 | `depth-anything/DA3-GIANT` |

**Advantages:**
- Pre-trained DualDPT head weights available
- Tested and validated by DA3 authors
- Works out of the box

### DINOv3 (Experimental - Pipeline Ready)

DINOv3 was released by Facebook Research in August 2025. The CoreML conversion pipeline is ready, but **no pre-trained DA3+DINOv3 weights exist yet**.

| Model | Hidden Dim | Patch Size | HuggingFace |
|-------|------------|------------|-------------|
| DINOv3-S | 384 | 16 | `facebook/dinov3-vits16-pretrain-lvd1689m` |
| DINOv3-B | 768 | 16 | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| DINOv3-L | 1024 | 16 | `facebook/dinov3-vitl16-pretrain-lvd1689m` |
| DINOv3-H+ | 1280 | 16 | `facebook/dinov3-vith16plus-pretrain-lvd1689m` |
| DINOv3-7B | 4096 | 16 | `facebook/dinov3-vit7b16-pretrain-lvd1689m` |

**Key Differences from DINOv2:**
- Patch size: 16 (vs 14 for DINOv2)
- Token structure: [CLS] + 4 registers + patches (skip 5 tokens vs 1)
- Input size: 512 (vs 518 for DINOv2)

**Training DualDPT for DINOv3:**
- ~47M trainable parameters in DualDPT head
- Requires depth supervision datasets (KITTI, NYUv2, MegaDepth, etc.)
- Estimated: 4-8 A100 GPUs for days/weeks
- Teacher-student training paradigm (complex setup)

**Status:** Backbone converted, DualDPT structure ready, awaiting training or official weights.

---

## Architecture Overview

### Complete Data Pipeline

```
INPUT: CGImage (any size)
        |
        v
+-----------------------------------------------------------------------+
|  1. PREPROCESSING (DINOv3CoreML.swift)                                |
|                                                                       |
|  a) Resize to 518x518 (DA3 default)                                  |
|     - Uses CGContext with noneSkipLast (NOT premultipliedLast!)      |
|     - High quality interpolation                                      |
|                                                                       |
|  b) Convert to float array [B, 3, H, W]                              |
|     - RGB channel order                                               |
|     - Pixel values: 0-255 -> 0.0-1.0                                 |
|                                                                       |
|  c) ImageNet normalization (per channel):                            |
|     mean = [0.485, 0.456, 0.406]                                     |
|     std  = [0.229, 0.224, 0.225]                                     |
|     normalized = (pixel - mean) / std                                |
|     NOTE: this is standard *input* normalization (ImageNet).         |
|                                                                       |
|  Output: MLMultiArray float32 [1, 3, 518, 518]                       |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|  2. DINOv2 BACKBONE (CoreML model)                                   |
|                                                                       |
|  Input: [B, 3, 518, 518]                                             |
|                                                                       |
|  Patch embedding: 518/14 = 37 -> 37x37 = 1369 patches                |
|                                                                       |
|  Token structure after embedding:                                     |
|    DINOv2: [CLS, patch_1, patch_2, ..., patch_1369] = 1370 tokens    |
|                                                                       |
|  Extract 4 intermediate layers (size-dependent):                     |
|    - e.g. giant: 19,27,33,39                                        |
|    - Skip CLS; DA3 cat_token=True => dim=2*hidden_dim                |
|    - Output: 4 tensors [B, 1369, 2*hidden_dim]                       |
|                                                                       |
|  Hidden dimensions by model size:                                     |
|    small: 384, base: 768, large: 1024, giant: 1536                   |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|  3. DualDPT HEAD (CoreML model)                                      |
|                                                                       |
|  Input: 4 feature tensors [B, 1369, D]                               |
|                                                                       |
|  Processing:                                                          |
|    a) Reshape: [B, 1369, D] -> [B, D, 37, 37]                        |
|    b) Project each scale to common dimension (256)                   |
|    c) Top-down fusion with 2x upsampling at each level               |
|    d) Final 4x upsample to match input resolution                    |
|                                                                       |
|  Outputs:                                                             |
|    - depth: [B, 1, H, W] - raw depth                                 |
|    - depth_confidence: [B, 1, H, W] - positive confidence weight (DA3 default: exp(x)+1) |
|    - rays: [B, 6, H, W] - ray parameters (optional)                  |
|    - ray_confidence: [B, 1, H, W] - positive confidence weight (DA3 default: exp(x)+1) |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|  4. POST-PROCESSING (DA3CoreML.swift)                                |
|                                                                       |
|  Depth activation (configurable):                                     |
|    - exp(x)     : Default, positive depth, exponential scaling       |
|    - relu(x)    : max(0, x), linear positive                         |
|    - sigmoid(x) : 1/(1+e^-x), bounded [0,1]                          |
|    - softplus(x): log(1+e^x), smooth positive                        |
|                                                                       |
|  Output: DA3CoreML.Result containing depth, confidence, rays         |
+-----------------------------------------------------------------------+
```

### Tensor Shape Transformations

| Stage | Shape | Notes |
|-------|-------|-------|
| Input image | [H, W, 4] | RGBA uint8 |
| After resize | [518, 518, 4] | RGBX (noneSkipLast) |
| After normalization | [1, 3, 518, 518] | float16, ImageNet norm |
| After patch embed | [1, 1370, D] | D=hidden_dim, +1 for CLS |
| After token skip | [1, 1369, dim_in] | Patches only; dim_in=2*hidden_dim (cat_token) |
| Layer features | [1, 1369, dim_in] | 4 tensors; layers from checkpoint (e.g. 19/27/33/39 for giant) |
| DPT reshaped | [1, dim_in, 37, 37] | sqrt(1369) = 37 |
| DPT projected | [1, 256, 37, 37] | Common feature dim |
| DPT output | [1, 1, 148, 148] | After fusion + 4x upsample |
| Final depth | [1, 1, 518, 518] | Bilinear to input size |

### Model Dimensions

| Size   | Backbone Dim | Patch | Tokens (518x518) | Depth (layers) | Memory |
|--------|--------------|-------|------------------|----------------|--------|
| Small  | 384          | 14    | 1369 (37x37)     | 12             | ~0.5GB |
| Base   | 768          | 14    | 1369 (37x37)     | 12             | ~1.5GB |
| Large  | 1024         | 14    | 1369 (37x37)     | 24             | ~4GB   |
| Giant  | 1536         | 14    | 1369 (37x37)     | 40             | ~12GB  |

---

## Conventions & Semantics

This section exists because a lot of “it looks wrong” reports are actually **convention mismatches** (normalization, viz, or camera math), not model failures.

### Input preprocessing 

You will see this in `DINOv3CoreML.swift` (and in most PyTorch DA3 codepaths):

```swift
let mean: [Float] = normalize ? [0.485, 0.456, 0.406] : [0, 0, 0]
let std: [Float]  = normalize ? [0.229, 0.224, 0.225] : [1, 1, 1]
```

That is **ImageNet normalization** for the DINO backbone. It does **not** hardcode or bias depth; it just maps input RGB into the distribution the backbone was trained on.

### Depth tensor semantics (DA3)

DA3 predicts a **ray-length** depth:
- For each pixel, the model predicts a positive scalar `depth(u,v)`.
- When we unproject to 3D, we treat `depth` as **distance along a unit camera ray direction**:
  - `dir = normalize([(u - cx)/fx, (v - cy)/fy, 1])`
  - `X_cam = dir * depth`

This is why a “depth → point cloud” step must use the **unit ray** convention (not just `z=depth`).

**Important nuance (upstream DA3 code uses both conventions):**
- The **3DGS pathway** (`GaussianAdapter`) uses **unit rays** and treats `depth` as **ray-length** (`X = origin + dir_unit * depth`).
- Some **export utilities** (e.g. GLB/COLMAP point cloud exports) use the classic pinhole `K^-1 @ [u,v,1] * depth` “z‑convention”.

DA3CoreML therefore keeps the **3DGS conversion** on the unit‑ray convention (to match DA3’s `GaussianAdapter` behavior) and exposes an explicit `--depth-convention z|ray` for streaming point cloud export where interoperability matters.

### Pixel coordinate convention (u,v)

This repo treats pixel centers as:
- `u = x + 0.5`, `v = y + 0.5` for integer pixel indices `(x,y)`.

This matches DA3’s `sample_image_grid()` ((`idx+0.5)/length`) and the common pinhole convention where intrinsics use `cx=W/2, cy=H/2`. Some upstream DA3 utilities (e.g. `utils/geometry.unproject_depth`) use `u=x, v=y` instead; the difference is a half‑pixel shift. For streaming export you can set `--pixel-center-offset 0` to match that behavior.

### Depth visualization semantics (DA3)

The `*_depth.png` files are **visualizations** only.

By default this repo matches the original DA3 `visualize_depth()` behavior:
- Convert depth → **inverse depth** (`inv = 1/depth`) for valid pixels
- Percentile normalize in inverse-depth space (2%–98%)
- Invert so **closer is warmer** (`invNorm = 1 - norm(inv)`)
- Apply the **Spectral** colormap (near = red, far = blue)

If you want “raw depth colormap” instead, use `--depth-viz-style depth`. Use `--invert-depth-viz` to flip the colors (PNG only; it does not change `.da3` depth values).

### Confidence semantics (DA3)

DA3 confidence outputs are **not probabilities** in the official setup.

Official DualDPT checkpoints use `conf_activation="expp1"`, meaning:
- `conf = exp(logit) + 1`
- `conf >= 1` and acts as a **positive weight** (used for blending, ray-pose weighting, pruning), not a calibrated probability.

For numerical stability on Apple GPUs/ANE you may want to export confidence **logits** instead and apply activation outside CoreML in float32:
- Export head with `conf_activation="linear"`
- Run with `--confidence-activation expp1` (or `softplus1`) and optional clamp.

### Ray tensor semantics (DA3)

Official DA3 ray heads output `rays: [1, 6, H, W]`:
- `rays[0..2]`: a per-pixel target/direction vector used to estimate intrinsics/rotation via homography
- `rays[3..5]`: translation channels (treated as **world-to-camera** `T`, then inverted to camera-to-world)

The ray-direction visualization (`*_rays_dir.png`) will often look like a **smooth gradient**. That is expected: it’s a camera ray field, not scene depth.

## Requirements

- macOS 14.0+ or iOS 17.0+
- Apple Silicon (M1/M2/M3/M4) recommended
- RAM requirements by model size:

| Model Size | Parameters | Minimum RAM | Recommended RAM |
|------------|-----------|-------------|-----------------|
| Small | ~22M | 4GB | 8GB |
| Base | ~98M | 8GB | 16GB |
| Large | ~335M | 16GB | 32GB |
| Giant | ~1.1B | 48GB | 96GB |

---

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(path: "/path/to/DA3CoreML")
]
```

### Build from Source

```bash
cd /path/to/DA3CoreML
swift build -c release
```

---

## Quick Start

### Option A: DINOv2 + DA3 Weights (Recommended)

This uses the official pre-trained DA3 checkpoint and exports a **matching** backbone + head pair.
Do **not** mix a generic HuggingFace `facebook/dinov2-*` backbone with a DA3-trained DualDPT head — it will run, but depth will be incorrect due to mismatched feature taps / cat_token behavior.

```bash
# Install Python dependencies
pip install torch coremltools transformers huggingface_hub safetensors numpy einops

# Download DA3-Giant checkpoint from HuggingFace (safetensors)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='depth-anything/DA3-GIANT', filename='model.safetensors', local_dir='Models/checkpoints/da3_giant')
hf_hub_download(repo_id='depth-anything/DA3-GIANT', filename='config.json', local_dir='Models/checkpoints/da3_giant')
"

# Convert DA3 backbone (official DINOv2 implementation; matches the DA3-trained head)
python Scripts/convert_da3_backbone_to_coreml.py \
    --checkpoint Models/checkpoints/da3_giant/model.safetensors \
    --size giant \
    --output Models/da3_backbone_giant_official.mlpackage

# Convert DualDPT head with pre-trained weights
python Scripts/convert_dualdpt_to_coreml.py \
    --checkpoint Models/checkpoints/da3_giant/model.safetensors \
    --size giant \
    --patch-size 14 \
    --output Models/dualdpt_giant_da3.mlpackage

# Compile models for faster loading
xcrun coremlc compile Models/da3_backbone_giant_official.mlpackage Models/compiled/
xcrun coremlc compile Models/dualdpt_giant_da3.mlpackage Models/compiled/
```

### Option B: DINOv3 Backbone (Experimental)

DINOv3 backbone is converted, but DualDPT head has random weights (no pre-trained weights available).

```bash
# Convert DINOv3-Large backbone
python Scripts/convert_dinov3_to_coreml.py \
    --model facebook/dinov3-vitl16-pretrain-lvd1689m \
    --output Models/dinov3_large.mlpackage \
    --input-size 512

# Convert DualDPT head (untrained - random weights)
python Scripts/convert_dualdpt_to_coreml.py \
    --size large \
    --dim-in 2048 \
    --patch-size 16 \
    --allow-fallback \
    --output Models/dualdpt_dinov3_large.mlpackage

# Compile
xcrun coremlc compile Models/dinov3_large.mlpackage Models/compiled/
xcrun coremlc compile Models/dualdpt_dinov3_large.mlpackage Models/compiled/
```

### Build and Run

```bash
# (Optional) clean + recompile all local CoreML packages
rm -rf Models/compiled && mkdir -p Models/compiled
for pkg in Models/*.mlpackage; do xcrun coremlc compile "$pkg" Models/compiled; done

# Build CLI
swift build -c release

# Run inference (DINOv2 - recommended)
.build/release/da3-coreml infer \
    --backbone Models/compiled/da3_backbone_giant_official.mlmodelc \
    --head Models/compiled_official_nope_f32/dualdpt_giant_da3_official_nope_f32.mlmodelc \
    --model-size giant \
    --include-rays \
    --ray-viz \
    --head-cpu-only \
    --output ./output \
    image1.jpg image2.jpg

# Run inference (DINOv3 - experimental, untrained head)
.build/release/da3-coreml infer \
    --backbone Models/compiled/dinov3_large.mlmodelc \
    --head Models/compiled/dualdpt_dinov3_large.mlmodelc \
    --model-size large \
    --input-size 512 \
    --output ./output \
    image1.jpg image2.jpg
```

### Output Files

- `image.da3` - Compressed binary with depth + confidence (and optionally rays; use `--include-rays`)
- `image_meta.json` - Metadata (dimensions, depth range, timestamp)
- `image_depth.png` - Visualization (unless --no-png)
- `image_rays_dir.png` / `image_rays_conf.png` - Ray visualization (only with `--include-rays --ray-viz`)

### File formats: why `.da3` (and what DA3 upstream uses)

The **original DA3** codebase primarily exports inference results as:
- `results.npz` (NumPy compressed; depth/conf/intrinsics/extrinsics) and
- `gs_ply/*.ply` (3D Gaussian splats).  

This repo supports multiple output formats so you can pick what fits your workflow:

- `--format da3` (default): a **single binary file** with optional zlib compression, plus a
  sidecar `*_meta.json` (metadata only).
- `--format npy`: one `.npy` per tensor (`*_depth.npy`, `*_rays.npy`, ...).
- `--format raw`: raw float32 blobs (`*_depth.raw`, ...).
- `--format png`: visualization only.

**Note on naming:** the upstream DA3 configs refer to `depth_anything_3.model.da3` (a **checkpoint / model**
container). This repo’s `.da3` is a **different** format used for **outputs**, with magic bytes `DA3C`.

**Why `.da3` instead of TOML/JSON/etc?**
- TOML/JSON/YAML are great for *metadata*, but terrible for multi‑megabyte float tensors
  (you either lose precision or end up base64‑encoding, which is slow and bloats size).
- `.da3` keeps the **bulk numeric tensors** in a compact binary format that’s fast to write/read
  from Swift (no third‑party zip libs), while keeping human-readable metadata in `*_meta.json`.

**Interoperability**
- Swift: use `DA3OutputReader` to load `.da3`.
- Python: convert `.da3` → `.npz` with:

```bash
python3 Scripts/da3_to_npz.py path/to/image.da3 --out path/to/image.npz
```

**Compression detail (Python users):**
- The Swift writer uses `NSData.compressed(using: .zlib)`, which produces a **raw DEFLATE** stream.
  If you implement your own reader, you’ll want `zlib.decompress(data, wbits=-zlib.MAX_WBITS)`.

---

## API Reference

### Basic Usage

```swift
import DA3CoreML

// Configure
var config = DA3CoreML.Config()
config.modelSize = .giant
config.memoryLimitGB = 64.0
config.safetyBufferPercent = 0.30

// Initialize
let da3 = try DA3CoreML(
    backbonePath: "Models/compiled/da3_backbone_giant_official.mlmodelc",
    headPath: "Models/compiled/dualdpt_giant_da3.mlmodelc",
    config: config
)

// Run inference
let result = try da3.predict(image: myCGImage)
print("Depth range: \(result.depthRange)")
print("Inference time: \(result.inferenceTime)s")

// Save to file
let writer = DA3OutputWriter()
try writer.save(result, to: "output/scene_001", format: .da3)
```

### Batch Processing

```swift
var config = DA3CoreML.Config()
config.modelSize = .giant
config.verboseMemory = true

let da3 = try DA3CoreML(backbonePath: backbone, headPath: head, config: config)

// DA3CoreML automatically:
// 1. Calculates optimal batch size
// 2. Monitors memory pressure
// 3. Throttles if needed
// 4. Cleans up between batches
let results = try da3.predictBatch(images: imageArray)
```

### Memory Manager

```swift
// Get memory statistics
let stats = MemoryManager.shared.getMemoryStats()
print("Total RAM: \(stats.totalGB) GB")
print("Available: \(stats.availableGB) GB")
print("Pressure: \(stats.pressure)")

// Calculate safe budget
let budgetGB = MemoryManager.shared.getSafeMemoryBudgetGB()

// Check if allocation is safe
if MemoryManager.shared.isSafeToAllocate(requiredGB: 10.0) {
    // Proceed
}

// Execute with cleanup
try MemoryManager.shared.withMemoryCleanup {
    // Memory-intensive operation
}
```

---

## CLI Reference

### Inference

```bash
da3-coreml infer [OPTIONS] <IMAGES>...

Required:
  -b, --backbone <PATH>    Path to backbone CoreML model (.mlmodelc or .mlpackage)
  -h, --head <PATH>        Path to DualDPT head CoreML model
  <IMAGES>                 Input image paths

Options:
  -o, --output <DIR>       Output directory (default: ./output)
  --model-size <SIZE>      Model size: small, base, large, giant
  --input-size <INT>       Input image size (default: 518)
  --include-rays           Include ray estimation
  --ray-viz                Save ray visualization PNGs (direction + confidence)
  --ray-pose               Estimate and print ray-pose intrinsics/extrinsics (debug; requires --include-rays)
  --ray-pose-subsample <I> Ray-pose subsample factor (debug; default: 16)
  --head-cpu-only          Force the DualDPT head to run on CPU only
  --postprocess-backend <B> Postprocess backend: cpu or metal
  --viz-backend <B>        Visualization backend for PNGs: cpu or metal
  --confidence-activation <A> Confidence activation for logits heads: linear, expp1, softplus1
  --confidence-logit-clamp-min <F> Confidence logit clamp min (activation != linear)
  --confidence-logit-clamp-max <F> Confidence logit clamp max (activation != linear)
  --no-tiling              Disable tiled inference (single pass + upscale)
  --max-tile-size <INT>    Max tile size in pixels when tiling (default: 1024)
  --tile-overlap <INT>     Tile overlap in pixels when tiling (default: 64)
  --colormap <NAME>        Colormap: spectral, turbo, viridis, plasma, magma, grayscale
  --depth-viz-style <S>    Depth viz: da3 (inverse-depth) or depth (raw)
  --invert-depth-viz       Invert depth PNG colors
  --save-raw               Save raw depth as binary
  --batch-size <INT>       Override max batch size (batch mode only)
  --memory-limit <GB>      Memory limit for batching (default: 64)
  -v, --verbose            Verbose output
  --no-png                 Skip PNG visualization
  --format <FMT>           Output format: da3, npy, raw, png
  --batch                  Enable batch mode
```

### 3DGS Export (Depth-only baseline)

```bash
da3-coreml to3-dgs [OPTIONS] <INPUTS>...

Arguments:
  <INPUTS>...              Input .da3 file(s) or directory

Options:
  -o, --output <DIR>       Output directory for PLY files (default: ./output_3dgs)
  --source-image <PATH>    Source image for colors (optional)
  --subsample <INT>        Subsample factor (default: 2)
  --min-confidence <F>     Minimum confidence threshold (default: 0.3)
  --gaussian-scale <F>     Gaussian scale (default: 0.01)
  --fov <F>                Field of view in degrees (default: 50)
  --ascii                  Output ASCII PLY (default: binary)
  -v, --verbose            Verbose output
```

This is a simple depth→point→Gaussian initializer for quick debugging. For DA3-style feed-forward splats and multi-view fusion, use `da3-coreml fuse --gshead ...` below.

### Multi-view 3DGS Fusion (DA3 GSHead)

```bash
da3-coreml fuse [OPTIONS] <IMAGES>...

Required:
  -b, --backbone <PATH>    Path to backbone CoreML model (.mlmodelc)
  -h, --head <PATH>        Path to DualDPT head CoreML model (.mlmodelc)
  --gshead <PATH>          Path to GSHead CoreML model (.mlmodelc)
  <IMAGES>...              Input image paths

Pose options (choose one):
  --camdec <PATH>          Camera decoder CoreML model (.mlmodelc)
  --use-ray-pose           Estimate pose/intrinsics from rays (DA3 use_ray_pose)

Useful knobs:
  --gs-subsample <INT>     Subsample factor (default: 4)
  --gs-min-confidence <F>  Min GS opacity/conf threshold (default: 0.0)
  --gs-disable-offset-depth
  --gs-offset-depth-scale <F>
  --confidence-activation <A>  For logits heads: linear, expp1, softplus1
  --postprocess-backend <B>    cpu or metal
  --allow-depth-only           Run depth-only fallback fusion without GSHead (lower quality; output PLY includes warning comments)
```

### Streaming (DA3-Streaming-style export)

Upstream DA3 includes `da3_streaming/` (chunking + overlap + Sim3 alignment + optional loop closure) for long videos. It runs DA3 in **multi-view mode per chunk** and exports:
- `camera_poses.txt` (per-frame 4×4 **c2w** matrices, flattened)
- `intrinsic.txt` (per-frame `fx fy cx cy`)
- `pcd/combined_pcd.ply` (binary point cloud)

This Swift/CoreML port currently exports the DA3 backbone/head as **S=1 (monocular-only)** models, so it cannot reproduce the full upstream **multi-view per-chunk** behavior. The `da3-coreml stream` command focuses on producing **compatible outputs** using the existing CoreML pipeline, and can optionally approximate upstream stitching with a **Sim3** chunk alignment using overlap frames (`--align-chunks`).

```bash
da3-coreml stream [OPTIONS] <INPUT_DIR>

Required:
  -b, --backbone <PATH>    Path to backbone CoreML model (.mlmodelc)
  -h, --head <PATH>        Path to DualDPT head CoreML model (.mlmodelc)

Pose options (choose one):
  --camdec <PATH>          Camera decoder CoreML model (.mlmodelc)
  --use-ray-pose           Estimate pose/intrinsics from rays (DA3 use_ray_pose)

Useful knobs:
  --output-dir <DIR>       Output directory (default: ./da3_stream_output)
  --chunk-size <INT>       Chunk size (default: 120)
  --overlap <INT>          Chunk overlap (default: 60)
  --align-chunks           Align chunk coordinate frames using Sim3 estimated from overlap frames (default: false)
  --sim3-scale-min <F>     Minimum allowed Sim3 scale (default: 0.3333333)
  --sim3-scale-max <F>     Maximum allowed Sim3 scale (default: 3.0)
  --pcd-sample-ratio <F>   Point cloud sample ratio (default: 0.015)
  --pcd-conf-threshold-coef <F>  threshold = mean(conf) * coef (default: 0.75)
  --[no-]subtract-confidence-one  Subtract 1.0 from conf (matches upstream `conf=exp(x)+1`)
  --depth-convention z|ray (default: z)
  --pixel-center-offset <F>  Pixel center offset added to integer pixel indices (default: 0.5). Use 0.0 to match exports that treat pixel centers as integer coordinates.
  --depth-min <F>          (default: 0)
  --depth-max <F>          (default: 15)
  --seed <INT>             Sampling seed (default: 42)
```

Outputs:
- `<out>/camera_poses.txt`: N lines of 16 floats (row-major 4×4 c2w).
- `<out>/intrinsic.txt`: N lines `fx fy cx cy`.
- `<out>/camera_poses.ply`: pose visualization point cloud (colored by chunk index).
- `<out>/pcd/<chunk>_pcd.ply` and `<out>/pcd/combined_pcd.ply`: binary PLY point cloud (x,y,z + rgb).

Notes:
- `--depth-convention z` matches upstream `da3_streaming.depth_to_point_cloud_vectorized` (`K^-1 @ [u,v,1] * z`).
- `--depth-convention ray` matches this repo’s 3DGS unprojection convention (unit ray * range).
- Pixel coordinate convention matters: this repo defaults to `u=x+0.5, v=y+0.5` (“pixel centers”). Some upstream utilities use `u=x, v=y`; use `--pixel-center-offset 0` for that behavior.
- With `--align-chunks`, the first frame becomes the world origin (`c2w = I`), and later chunks are aligned into that frame using sampled **overlap-frame depth point correspondences** (fallback: pose-only correspondences).
- Upstream `da3_streaming` uses dense overlap point maps + confidence masking for Sim3; this implementation approximates that with deterministic sampling for speed and reproducibility.
- Reruns are deterministic: `da3-coreml stream` deletes stale `camera_poses.txt`, `intrinsic.txt`, `camera_poses.ply`, and `pcd/*_pcd.ply` (including `combined_pcd.ply`) before writing new outputs.
- Point cloud colors are sampled from the **exact backbone input tensor** (ImageNet-normalized RGB, then de-normalized), so the RGB and depth grids stay pixel-aligned.
- `pcd/combined_pcd.ply` is merged in numeric chunk order (`0_pcd.ply`, `1_pcd.ply`, …).
- Full upstream parity still requires exporting a true multi-view CoreML backbone (**S>1**) and implementing loop closure.

### Benchmark

```bash
da3-coreml benchmark [OPTIONS]

Options:
  -b, --backbone <PATH>    Path to DINOv2 CoreML model
  -h, --head <PATH>        Path to DualDPT CoreML model
  --model-size <SIZE>      Model size
  --warmup <INT>           Warmup iterations (default: 3)
  --iterations <INT>       Benchmark iterations (default: 10)
```

---

## Memory Management

### The 128GB Problem

On a 128GB Mac, you **never** have 128GB available:
- macOS uses 8-15GB
- Background apps use 5-20GB
- System caches use 10-30GB
- **Actual available: typically 60-90GB**

### Safety Buffer System

```swift
// Default: 30% safety buffer
config.safetyBufferPercent = 0.30

// On 128GB system:
// User limit: 100GB
// Effective: 100GB x 0.70 = 70GB max usage
```

### Memory Pressure Levels

| Level | RAM Used | Action |
|-------|----------|--------|
| Nominal | <50% | Full speed, max batch |
| Warning | 50-70% | Reduce batch 50% |
| Critical | 70-85% | Single image only |
| Terminal | >85% | Abort operation |

### Configuration Examples

```swift
// Conservative (heavy multitasking)
config.safetyBufferPercent = 0.40  // 60% usable

// Default (recommended)
config.safetyBufferPercent = 0.30  // 70% usable

// Aggressive (dedicated machine)
config.safetyBufferPercent = 0.20  // 80% usable
```

---

## Model Information

### Backbone Comparison: DINOv2 vs DINOv3

**Depth-Anything-3 uses DINOv2** (released 2023). DINOv3 was released by Facebook in August 2025, after DA3's development.

| Aspect | DINOv2 | DINOv3 |
|--------|--------|--------|
| Release Date | 2023 | August 2025 |
| Used by DA3 | **Yes** | No (pipeline ready) |
| Patch Size | 14 | 16 |
| Input Size | 518 (37×37 patches) | 512 (32×32 patches) |
| Token Structure | [CLS] + patches | [CLS] + 4 registers + patches |
| Tokens to Skip | 1 | 5 |
| Pre-trained Head | **Available** | Not available |

### Available Models

**DINOv2 + DA3 (Working):**
| Model | Parameters | Memory | Status |
|-------|-----------|--------|--------|
| DA3-Small | ~22M | ~2GB | Available |
| DA3-Base | ~98M | ~4GB | Available |
| DA3-Large | ~335M | ~8GB | Available |
| DA3-Giant | ~1.1B | ~24GB | Best quality (large RAM) |

**DINOv3 (Experimental):**
| Model | Parameters | Memory | Status |
|-------|-----------|--------|--------|
| DINOv3-L + DualDPT | ~350M | ~8GB | Backbone only, head untrained |
| DINOv3-H+ + DualDPT | ~850M | ~20GB | Backbone only, head untrained |

### Critical Constants

```swift
// ImageNet normalization (both DINOv2 and DINOv3)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

// These are standard ImageNet preprocessing constants for the backbone.
// They do NOT affect how depth is computed other than feeding the model the
// input distribution it was trained on.

// DINOv2 (DA3 official)
inputSize = 518
patchSize = 14
skipTokens = 1  // CLS only

// DINOv3 (experimental)
inputSize = 512
patchSize = 16
skipTokens = 5  // CLS + 4 registers
```

### Performance

Performance depends heavily on model size, head precision (fp16 vs fp32), tiling, and CoreML compute placement (ANE/GPU/CPU).
For repeatable numbers on your machine, use the built-in benchmark:

```bash
swift build -c release
.build/release/da3-coreml benchmark --backbone <PATH> --head <PATH> --model-size giant
```

---

### Precision & Hardware Notes (DA3 PyTorch vs CoreML)

This project is trying to match the **behavior** of the original DA3 implementation while running on Apple Silicon.
The biggest practical differences are **mixed-precision defaults** (bf16/fp16 on CUDA vs fp16 on CoreML GPU/ANE)
and which parts DA3 intentionally forces to run in full precision.

#### What precision does the original DA3 use?

In the official DA3 Python API, inference runs under `torch.autocast`:
- Autocast dtype: **bfloat16 if supported, else float16** (see `src/depth_anything_3/api.py`, `autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16`)

But the DA3 model also explicitly disables autocast for several sensitive parts (runs them in float32):
- Camera token encoding from extrinsics/intrinsics
- Depth head post-processing + pose estimation (`use_ray_pose`) + camera decoding (`camdec`) + GS head branch
  (see `src/depth_anything_3/model/da3.py`, multiple `with torch.autocast(..., enabled=False):` blocks)

**Why this matters:** even if the backbone is bf16/fp16, DA3 keeps numerically sensitive operations (including `exp()` in confidence activations and geometric steps) out of mixed precision.

#### Why can CoreML float16 produce NaN/Inf for rays/confidence?

Apple GPU/ANE execution is typically **float16** for performance. Float16 has a very limited dynamic range:
- Max finite fp16 is **65504**
- `exp(x)` overflows to `Inf` around `x ≈ ln(65504) ≈ 11.09`

DA3’s default confidence activation is `expp1 = exp(x) + 1`. If the confidence logit drifts above ~11 in fp16,
the CoreML graph can produce `Inf` (and downstream computations may become `NaN`).

#### Recommended stability fix: export confidence logits (pre-activation)

If you control the model export, the most robust approach is:
1. Export the DualDPT head with **confidence logits** (`conf_activation="linear"`). This removes `exp()` from the CoreML graph.
2. Apply a stable activation **outside CoreML** in float32:
   - `expp1`: `exp(clamp(x)) + 1` (matches DA3 semantics)
   - `softplus1`: `softplus(clamp(x)) + 1` (always finite in float32; sometimes more stable for extreme logits)

This repo supports that workflow:
- Conversion: `Scripts/convert_dualdpt_official_to_coreml.py --conf-activation linear ...`
- Swift: set `DA3CoreML.Config.confidenceActivation = .expp1` (or `.softplus1`) and optionally tighten the clamp range.

Why this works: the overflow happened because **fp16 `exp()`** was inside the CoreML graph. Moving the activation outside the
graph keeps that math in **float32**, which avoids `Inf/NaN` in practice.

On Nvidia H100/H200-class GPUs, DA3 often runs autocast as **bf16**, which has the **same exponent range as float32**.
That makes `exp()` vastly less likely to overflow, even when using mixed precision.

#### Can the float32 head run on Apple GPU/ANE?

In practice, a float32 MLProgram head may **fail to compile** for GPU/ANE or fall back to CPU. On this repo’s
current models, attempting to run the official float32 DualDPT head with GPU/ANE enabled can fail with a CoreML
execution-plan build error (you’ll see an error like “Failed to build the model execution plan … error code: -6”).

That’s why the CLI exposes:
- `--head-cpu-only` (forces `.cpuOnly` for the head)

#### Is “ray pose” expensive? Should it run on GPU?

The expensive part of DA3 is the **neural inference** (backbone + heads). The ray-based pose estimator (`--use-ray-pose`)
is a **geometric fit** over the ray grid and is usually not the bottleneck:
- Ray grid is low-res (often **296×296** for the official 518px head)
- This implementation subsamples by default (`DA3RayPoseEstimator.Config.subsample = 4`)
  → ~`(296/4)^2 ≈ 5.5k` rays
- RANSAC iterations default to 100; with ~5–10k samples this is typically milliseconds on CPU
  (and implemented with `Accelerate` + SIMD in `Sources/DA3CoreML/DA3RayPoseEstimator.swift`)

**Extrinsics convention (important):**
- DA3’s `get_extrinsic_from_camray()` returns a **world-to-camera (w2c)** matrix.
- DA3 then applies `affine_inverse()` to convert it to **camera-to-world (c2w)** before using it downstream
  (see `depth_anything_3/model/da3.py` `_process_ray_pose_estimation`).
- This repo follows the same convention and exposes **c2w** from the Swift ray-pose estimator.

Running ray-pose on GPU/ANE is not something CoreML “just does” because it’s not a neural model. You *could* rewrite it
as a Metal compute pipeline (or MPSGraph) but:
- It’s a lot of work for little end-to-end speedup (vs model inference time)
- Debuggability is worse than the CPU `Accelerate` version

If your goal is an “all-CoreML” path, prefer **CamDec** (`--camdec ...`) for camera parameters (it’s a neural model and runs in CoreML).

#### Practical recommendations (Speed vs Stability)

- **Best stability for rays / ray-pose:** use a float32 DualDPT head and force CPU:
  - `--head Models/compiled_official_nope_f32/...mlmodelc --head-cpu-only`
- **Best speed (ANE/GPU):** use float16 heads on `.all`, but expect that ray confidence can be less stable on some scenes:
  - Prefer `--camdec ...` over `--use-ray-pose` when fusing.
- **If GSHead fusion scale explodes:** try `--gs-disable-offset-depth` or `--gs-offset-depth-scale 0.01`.

#### What runs on Apple GPU/ANE vs CPU in this repo?

At a high level:

| Component | What it does | Implementation here | Runs on |
|---|---|---|---|
| Backbone (DINOv2/DINOv3) | ViT feature extraction | CoreML (`DINOv3CoreML`) | ANE/GPU/CPU (`.all`) |
| DualDPT head | Depth + rays + confidence | CoreML (`DualDPTCoreML`) | GPU/ANE for float16; **CPU-only recommended** for float32 |
| CamDec | Camera intrinsics/extrinsics from features | CoreML (`CamDecCoreML`) | ANE/GPU/CPU (`.all`) |
| Ray-pose (`use_ray_pose`) | RANSAC + QL solve from ray field | Swift + `Accelerate` (`DA3RayPoseEstimator`) | CPU |
| Postprocess (resize + tiling blend) | Crop/resize + “tallying” | Swift/Metal (`DA3CoreML.postprocessPrediction`, `DA3MetalPostProcessor`) | CPU or GPU (Metal) |
| GSHead | Feed-forward 3DGS parameters | CoreML (`GSHeadCoreML`) | ANE/GPU/CPU (`.all`) |
| DA3→PLY conversion | Writes splats to disk | Swift (`DA3GSHeadTo3DGS`, `DA3PLYWriter`) | CPU |

So “all CoreML” typically means “all **neural** parts are CoreML”. The remaining pieces are geometry/IO and are CPU-side.

#### GPU Postprocess (Metal) in this repo

By default, postprocessing runs on CPU (portable, simplest). For large images and/or tiled inference, CPU postprocess can
become a significant fraction of end-to-end time. This repo provides an optional **Metal** backend for:
- Crop + bilinear resize (depth / confidence / rays)
- Tiled blending (“tallying”) + normalization
- Confidence logits activation (`expp1` / `softplus1`) in float32
- Depth visualization PNGs (Spectral/Turbo/Grayscale) when `--viz-backend metal` is enabled

Key properties:
- Runs in **float32** (better numerical headroom than fp16) and uses unified memory on Apple Silicon.
- Does **not** use ANE (these operations aren’t neural inference).
- First use may pay a one-time shader compilation cost; subsequent calls reuse pipelines.

Enable it:
- Swift: `DA3CoreML.Config.postprocessBackend = .metal`
- CLI: see `--postprocess-backend` (and keep `--head-cpu-only` available for head stability when needed)

#### MLX vs MPSGraph vs Metal (what to use where)

This repo intentionally mixes tools based on what they’re best at:
- **CoreML**: runs the large neural graphs efficiently on Apple hardware (ANE/GPU/CPU scheduling).
- **Metal compute**: best for custom, data-parallel postprocess (tallying/blend kernels, CHW resizes) with predictable memory.
- **MPSGraph**: great for “tensor math” graphs in float32 (stable `exp`/`softplus`/clamp). In this repo, confidence
  logits→weights activation is applied outside CoreML in float32 (CPU by default, or Metal when `--postprocess-backend metal`
  is enabled). MPSGraph would also be a reasonable implementation choice if you prefer a graph API over custom Metal kernels.
- **MLX**: excellent for research/training loops and lightweight inference, but it does not integrate with CoreML’s ANE
  execution, and swapping the whole DA3 model over to MLX would be a separate project.

#### Does DA3 use int8 / FP8 on H100/H200/B100?

Not in the reference codepath. The official DA3 repo runs inference in floating point (PyTorch AMP):
- Backbone under autocast (bf16 on supporting GPUs, else fp16)
- Several numerically sensitive parts run in float32 (autocast disabled in those blocks)

If you want int8/FP8 quantized inference, that’s a separate optimization project and is not implemented here.

## Known Issues & Fixes

### MLMultiArray Stride Handling (FIXED)

**Issue:** CoreML MLMultiArray outputs may have non-contiguous memory layouts with padding between rows.

**Root Cause:** The depth output had strides `[281792, 281792, 544, 1]` instead of contiguous `[268324, 268324, 518, 1]`. Direct pointer access read garbage bytes from padding.

**Symptom:** Depth maps showed spurious near-zero values (0.001-0.01).

**Fix Applied:** Changed `readFloatArray()` to use MLMultiArray subscript operator:

```swift
// WRONG - ignores strides, reads garbage bytes
let ptr = UnsafeMutablePointer<Float16>(OpaquePointer(array.dataPointer))
return (0..<count).map { Float(ptr[$0]) }

// CORRECT - respects strides
return (0..<count).map { array[$0].floatValue }
```

### Tiled Inference Edge Coverage (FIXED)

**Status:** Fixed - last tile in each row/column now placed at image boundary

**Previous Issue:** Edge pixels not covered by any tile got zero depth values due to
the `min(tx * stride, image.width - tileSize)` formula causing intermediate tiles to
jump backward, creating gaps.

**Fix Applied:** Changed tile placement logic:
- Regular tiles: placed at stride intervals (`tx * stride`)
- Last tile in row: placed at `max(0, image.width - tileSize)` (right edge)
- Last tile in column: placed at `max(0, image.height - tileSize)` (bottom edge)

This ensures complete coverage for all image sizes without gaps.

---

### `.da3` Reader Crash (Unaligned Loads) (FIXED)

**Issue:** `da3-coreml to3-dgs` could crash with `Fatal error: load from misaligned raw pointer` when reading `.da3` files.

**Root Cause:** `DA3OutputReader` was using `Data.withUnsafeBytes { $0.load(fromByteOffset:as:) }`, which requires aligned memory. `Data` byte buffers are not guaranteed to be aligned for arbitrary loads.

**Fix Applied:** Switched `.da3` parsing to explicit little-endian byte copies (unaligned-safe), and bulk-copied float blobs into `[Float]` without per-element `load()` calls.

Implementation: `Sources/DA3CoreML/DA3OutputWriter.swift` (`DA3OutputReader`).

---

### Streaming Output Accumulation (FIXED)

**Issue:** Re-running `da3-coreml stream` into the same `--output-dir` could accidentally merge stale `pcd/*_pcd.ply` chunk files from previous runs, producing an incorrect `pcd/combined_pcd.ply`.

**Fix Applied:** `da3-coreml stream` now deletes:
- `<out>/camera_poses.txt`, `<out>/intrinsic.txt`, `<out>/camera_poses.ply`
- `<out>/pcd/*_pcd.ply` and `<out>/pcd/combined_pcd.ply`

before writing new outputs.

### Point Cloud Merge Order (FIXED)

**Issue:** Lexicographic merges can produce non-deterministic ordering (`10_pcd.ply` before `2_pcd.ply`).

**Fix Applied:** `DA3PointCloudPLYWriter.mergeBinaryPointCloudPLYFiles(...)` now sorts inputs by numeric prefix when possible, with lexicographic fallback.

### Tiled Inference (Tiling + “Tallying” / Weighted Blending)

DA3CoreML enables tiling automatically when the input image is larger than `maxTileSize` (CLI `--max-tile-size`, default 1024).
The goal is to keep peak memory bounded while still producing a full-resolution output.

**Terminology**
- `tileSize` = `maxTileSize`
- `overlap` = `tileOverlap` (CLI `--tile-overlap`, default 64)
- `stride` = `tileSize - overlap`

**Tile grid**
- `numTilesX = ceil(imageWidth / stride)`
- `numTilesY = ceil(imageHeight / stride)`
- Regular tiles start at `(tx * stride, ty * stride)`
- The last tile in a row/column is clamped to the image edge:
  - `tileX = max(0, imageWidth - tileSize)`
  - `tileY = max(0, imageHeight - tileSize)`

This “last tile snaps to the edge” rule is what prevents uncovered gaps at the right/bottom boundaries.

**Per-tile pipeline**
1. Crop a `CGImage` tile.
2. Run backbone + head on that tile.
3. Crop away any padding introduced by preprocessing (using `PreprocessInfo.padLeft/padTop`).
4. Resize predictions back to the tile’s `(tileW, tileH)` in original pixel space.
5. Blend (“tally”) the tile prediction into full-size output buffers.

**Tallying / weighted blending**

For depth + confidence, we maintain three full-resolution buffers:
- `accDepth[y,x]`  (sum of `depth * weight`)
- `accConf[y,x]`   (sum of `conf * weight`)
- `accW[y,x]`      (sum of `weight`)

For each pixel in a tile we compute a weight ramp based on distance to tile edges:
- `minDist = min(distLeft, distRight, distTop, distBottom)`
- `weight = clamp(minDist / overlap, 0..1)` (or `1` when `overlap == 0`)

At **image boundaries** (global left/right/top/bottom edges), we force full weight to avoid dim borders:
- If the tile touches the image boundary, we treat the boundary-side distance as “large”.

After processing all tiles:
- `depth = accDepth / accW` (where `accW > 0`)
- `conf  = accConf  / accW` (where `accW > 0`)

Implementation references:
- Tile loop: `Sources/DA3CoreML/DA3CoreML.swift` (`predictTiled`)
- Depth/conf blending: `Sources/DA3CoreML/DA3CoreML.swift` (`blendTile`)
- Final normalization: `Sources/DA3CoreML/DA3CoreML.swift` (`normalizeByWeights`)

**Rays + tiling**

Depth benefits from tiling; the **ray field does not**.

The DA3 ray tensor encodes **global camera geometry** (intrinsics/rotation/translation). If you run the ray head on
independent tiles and stitch/blend the results, each tile’s ray field can be inconsistent (different local “camera”),
which makes `--ray-pose` degenerate or unstable.

DA3CoreML therefore does the following when tiling is enabled and rays are requested:
- Run **tiled** inference for depth + depth_confidence.
- Run a separate **global** pass at model resolution (e.g. 518×518) to get the **native head ray grid** (`headRays`) and
  `headRayConfidence` for ray-pose, then crop/resize those rays back to the original image size for output.

Note: official DualDPT heads often output rays at an **aux resolution** (e.g. 296×296 for 518 inputs). Cropping back to the
unpadded valid region must therefore **scale** `PreprocessInfo.padLeft/padTop` into ray-grid coordinates; otherwise rays can
look “random” or blocky and ray-pose degenerates.

## Design Decisions

### 1. Pure Swift/CoreML (No CUDA)

**Rationale:**
- Apple Silicon native runtime (macOS/iOS)
- CoreML leverages ANE + GPU
- Unified memory benefits large models

**Trade-offs:**
- Works on any Apple Silicon Mac
- Efficient unified memory
- Slower than high-end desktop Nvidia GPUs for the largest models

### 2. DINOv2 as Default Backbone

**Rationale:**
- DA3 was trained with DINOv2
- Weights readily available on HuggingFace
- Well-tested and stable

### 3. Float16 Precision

**Rationale:**
- Apple Silicon optimized for float16
- 50% memory reduction
- Minimal quality loss for depth

**Caveat (rays/confidence):**
- DA3’s ray/confidence branches contain `exp()`-based activations (`expp1 = exp(x)+1`) and geometric post-processing.
  In the reference PyTorch code, several of these steps are forced to run outside autocast (float32).
- On CoreML GPU/ANE with float16 compute, those same activations can overflow (producing `Inf/NaN`) on some scenes.
  If you need stable rays, prefer a float32 head with `--head-cpu-only`, or fuse with `--camdec` instead of ray-pose.

### 4. 30% Default Safety Buffer

**Rationale:**
- Prevents OOM crashes
- Accounts for system overhead
- Conservative but reliable

### 5. Tiled Inference

**Rationale:**
- Allows processing images larger than memory
- Configurable tile size and overlap
- Smooth blending at boundaries

---

## File Structure

```
DA3CoreML/
├── Package.swift                 # Swift Package manifest
├── README.md                     # This file (centralized docs)
├── Sources/
│   ├── DA3CoreML/
│   │   ├── DA3CoreML.swift       # Main API
│   │   ├── DINOv3CoreML.swift    # Backbone wrapper
│   │   ├── DualDPTCoreML.swift   # Head wrapper
│   │   ├── CamDecCoreML.swift    # Camera decoder wrapper
│   │   ├── DA3PointCloudPLYWriter.swift # Binary point cloud PLY writer/merger
│   │   ├── MemoryManager.swift   # Memory management
│   │   ├── DA3DepthTo3DGS.swift  # Depth to 3DGS conversion
│   │   └── DA3Error.swift        # Error types
│   └── DA3CLI/
│       └── main.swift            # CLI tool
├── Scripts/
│   ├── setup.sh                  # Setup script
│   ├── convert_dinov3_to_coreml.py
│   ├── convert_dualdpt_to_coreml.py
│   ├── convert_camdec_to_coreml.py
│   ├── convert_camenc_to_coreml.py
│   └── convert_da3_backbone_to_coreml.py
├── Models/                       # CoreML models (after conversion)
│   └── compiled/                 # Compiled .mlmodelc files
├── test_images_realworld/        # Real-world test images (1879 images)
├── test_images_50/               # Synthetic test images
└── Tests/
    └── DA3CoreMLTests/
```

---

## Full Pipeline: Images to Gaussian Splats (DA3-Giant)

This section provides complete scripts for processing a folder of images through DA3-Giant to produce Gaussian splat outputs suitable for 3DGS rendering.

### Prerequisites

```bash
# Ensure models are converted and compiled
cd /path/to/DA3CoreML

# Verify models exist
ls -la Models/compiled/da3_backbone_giant_official.mlmodelc
ls -la Models/compiled/dualdpt_giant_da3.mlmodelc
```

### Step 1: Process All Images (Sequential - Memory Safe)

**For 2000+ images, use sequential processing (NOT batch mode):**

```bash
# Create output directory structure
mkdir -p output_gaussian_splat/{depth,confidence,rays,meta,visualizations}

# Process all images sequentially (memory-safe for 2K+ images)
# ~40 seconds per image on M4 Max with Giant model
.build/release/da3-coreml infer \
  --backbone Models/compiled/da3_backbone_giant_official.mlmodelc \
  --head Models/compiled/dualdpt_giant_da3.mlmodelc \
  --model-size giant \
  --output output_gaussian_splat \
  --include-rays \
  --format da3 \
  -v \
  /path/to/your/images/*.jpg

# For  edth images folder:
.build/release/da3-coreml infer \
  --backbone Models/compiled/da3_backbone_giant_official.mlmodelc \
  --head Models/compiled/dualdpt_giant_da3.mlmodelc \
  --model-size giant \
  --output output__gaussian \
  --include-rays \
  --format da3 \
  -v \
  "/path/to/sample_images/"*.jpg
```

**Estimated Time for 2000 images:**
- DA3-Giant: ~40s/image = **22 hours total**
- DA3-Large: ~15s/image = **8 hours total** (smaller model, less accurate)

### Sampling Images (Reduce Dataset Size)

If 2000 images is too many, sample every Nth image:

```bash
# Take every 10th image (1879 -> 188 images, 2.1 hours instead of 21 hours)
python Scripts/sample_images.py "/path/to/images" --every 10 --output sampled_images/

# Take every 3rd image (1879 -> 626 images)
python Scripts/sample_images.py "/path/to/images" --every 3 --output sampled_images/ --copy

# Take every 20th image (1879 -> 94 images, quick test)
python Scripts/sample_images.py "/path/to/images" --every 20 --output sampled_images/

# Take exactly 100 evenly spaced images
python Scripts/sample_images.py "/path/to/images" --count 100 --output sampled_images/

# Dry run - just see what would be selected
python Scripts/sample_images.py "/path/to/images" --every 10 --dry-run
```

**Sampling recommendations:**
| Sampling | Images (from 1879) | Time (Giant) | Use Case |
|----------|-------------------|--------------|----------|
| Every 3rd | 626 | 7 hours | Full quality |
| Every 10th | 188 | 2 hours | Good balance |
| Every 20th | 94 | 1 hour | Quick test |
| Count 50 | 50 | 30 min | Fast preview |

### Step 2: Monitor Progress

```bash
# Watch progress in another terminal
watch -n 60 'ls -1 output__gaussian/*.da3 | wc -l'

# Or check specific output files
ls -la output__gaussian/ | tail -20
```

### Step 3: Output Structure

After processing, each image produces:

```
output_gaussian_splat/
├── image001.da3           # Binary: depth + rays + confidence (compressed)
├── image001_meta.json     # Metadata: depth range, dimensions, timing
├── image001_depth.png     # Visualization (turbo colormap)
├── image002.da3
├── image002_meta.json
├── image002_depth.png
...
```

### Step 4: Convert to Gaussian Splats

```bash
# Convert DA3 outputs to PLY format for 3DGS
.build/release/da3-coreml to3-dgs \
  --input output__gaussian \
  --output gaussian_splats \
  --format ply

# Or use SPZ (compressed) format
.build/release/da3-coreml to3-dgs \
  --input output__gaussian \
  --output gaussian_splats \
  --format spz
```

**Note:** `to3-dgs` is a **depth→point-cloud→Gaussian init** pipeline. It’s useful for previews, but the splats
can look like “rectangles” because they are not the DA3 feed-forward Gaussians.

### Step 4b (Recommended): Feed-forward DA3 Gaussian Splats (GS Head)

DA3 includes a pre-trained **Gaussian Splatting head** (`gs_head`) and **camera decoder** (`cam_dec`) in the
official checkpoints. This produces proper 3DGS parameters (scale/rotation/opacity/SH color) directly.

```bash
# Convert + compile cam_dec and gs_head from the same DA3 checkpoint
python Scripts/convert_camdec_to_coreml.py \
  --checkpoint Models/checkpoints/da3_giant/model.safetensors \
  --size giant \
  --precision float16 \
  --output Models/camdec_giant.mlpackage

python Scripts/convert_gshead_to_coreml.py \
  --checkpoint Models/checkpoints/da3_giant/model.safetensors \
  --size giant \
  --precision float16 \
  --patch-size 14 \
  --input-size 518 \
  --output Models/gshead_giant.mlpackage

xcrun coremlc compile Models/camdec_giant.mlpackage Models/compiled/
xcrun coremlc compile Models/gshead_giant.mlpackage Models/compiled/

**CamDec note (upstream vs this CoreML pipeline):**
- Upstream DA3 feeds `cam_dec` with the backbone **camera token** (`feats[-1][1]`), not the patch grid.
- The default CoreML backbones in this repo export only **patch tokens** (`features_layer11`), so `CamDecCoreML.decodePose` runs CamDec on the patch grid and **mean-reduces** pose parameters as an approximation.
- This can noticeably change predicted intrinsics/pose (example on `sample_images/20251115_072731.jpg`: camera-token `fx≈384` vs patch-token `fx≈713`, ΔR≈8.5°). Reproduce with:
  - `Scripts/check_camdec_token_mode_pytorch.py`

# Fuse multiple images into a single feed-forward 3DGS PLY
.build/release/da3-coreml fuse \
  --backbone Models/compiled/da3_backbone_giant_official.mlmodelc \
  --head Models/compiled_official_nope_f32/dualdpt_giant_da3_official_nope_f32.mlmodelc \
  --gshead Models/compiled/gshead_giant.mlmodelc \
  --use-ray-pose \
  --head-cpu-only \
  --model-size giant \
  --gs-subsample 4 \
  --gs-min-confidence 0.0 \
  --output fused_scene_gshead.ply \
  "/path/to/your/images/"*.jpg

# If the fused scene scale is enormous, try:
#   --gs-disable-offset-depth
# or:
#   --gs-offset-depth-scale 0.01
#
# If ray-pose translations are near-zero (poor multi-view fusion), try CamDec instead:
#   --camdec Models/compiled/camdec_giant.mlmodelc
# Note: In this repo, CamDec is currently a *patch-token* approximation (see note above).
```

### Step 5: Verify Output Quality

```bash
# Check depth statistics across all images
for f in output__gaussian/*_meta.json; do
  echo "=== $(basename $f) ==="
  jq '{depthMin, depthMax, inferenceTime}' "$f"
done | head -50

# Check for any failed images (missing .da3 files)
ls /path/to/images/*.jpg | while read img; do
  base=$(basename "$img" .jpg)
  if [ ! -f "output__gaussian/${base}.da3" ]; then
    echo "MISSING: $base"
  fi
done
```

### Model Size Recommendations

| Use Case | Model | Time/Image | RAM | Gaussian Quality |
|----------|-------|------------|-----|-----------------|
| Quick preview | DA3-Large | ~15s | 8GB | Good |
| Production | DA3-Giant | ~40s | 24GB | **Best** |
| Low memory | DA3-Base | ~8s | 4GB | Moderate |

**Important:** Smaller models (Small, Base) may not produce sufficient depth accuracy for high-quality Gaussian splatting. **Use DA3-Giant or DA3-Large for production splatting.**

### Memory Management for Long Runs

The CLI automatically:
- Processes images one at a time (no bulk loading)
- Monitors memory pressure between images
- Cleans up intermediate tensors
- Skips failed images and continues

For 128GB M4 Max:
- DA3-Giant uses ~24GB peak
- Safe to run with other apps open
- No special configuration needed

### Parallel Processing (Advanced)

To speed up processing, run multiple instances on different image subsets:

```bash
# Split images into 4 groups and process in parallel terminals
# Terminal 1:
.build/release/da3-coreml infer ... /path/to/images/image_000*.jpg

# Terminal 2:
.build/release/da3-coreml infer ... /path/to/images/image_001*.jpg

# Terminal 3:
.build/release/da3-coreml infer ... /path/to/images/image_002*.jpg

# Terminal 4:
.build/release/da3-coreml infer ... /path/to/images/image_003*.jpg
```

**Warning:** With DA3-Giant (~24GB each), only run 2-3 parallel instances on 128GB Mac.

---

## 3D Gaussian Splatting (3DGS)

### DA3 -> 3DGS Pipeline

```
DA3 Depth Output
      |
      v
+-------------------+
| Depth Unprojection|  <- Needs camera intrinsics
| depth -> 3D points|
+-------------------+
      |
      v
+-------------------+
| Point Cloud       |
| (x, y, z, color)  |
+-------------------+
      |
      v
+-------------------+
| Gaussian Init     |  <- Convert points to splats
| - Position: xyz   |
| - Color: SH DC    |
| - Scale: default  |
| - Rotation: I     |
+-------------------+
      |
      v
+-------------------+
| 3D Gaussian Set   |  <- in-memory splats
+-------------------+
      |
  +---+---+
  v       v
+-----+ +-----+
| PLY | | SPZ |
+-----+ +-----+
```

### Included 3DGS Utilities

- `DA3GaussianCloud` / `DA3GaussianSplat` - In-memory 3DGS data structures
- `PLY Writer/Reader` - Standard 3DGS format
- `SPZ Encoder/Decoder` - Compressed 3DGS format
- `GLB Reader` - Convert mesh to 3DGS

---

## Current Issues & Debugging Notes (For Future Development)

### Issue: Blocky/Rectangular Depth Maps

**Status:** Partially expected (patch grid), but **tile-grid rectangles are a bug**

**Analysis (Dec 2025):**

Some “patch grid” texture is common in ViT+DPT style models because:

1. **Native resolution is only 37×37 patches** for 518×518 input (patch_size=14)
2. The DualDPT head upsamples through fusion layers:
   - ConvTranspose2d x4 (layer 0): 37→148
   - ConvTranspose2d x2 (layer 1): 37→74
   - Identity (layer 2): 37→37
   - Conv2d /2 (layer 3): 37→19
3. Final bilinear interpolation to original resolution (e.g., 4000×3000)

**Why it looks blocky:**
- Each 14×14 pixel region in the input shares ONE backbone feature token
- ConvTranspose2d causes visible grid artifacts at upsample boundaries (known issue with deconvolution)
- This is inherent to ALL ViT+DPT architectures (Depth-Anything, MiDaS, etc.)

**But:** if you see **large rectangles aligned to tile boundaries** (e.g. 1024×1024 blocks) or hard seams, that indicates a tiling/postprocess bug (crop window mismatch, weights normalization bug, stride misread). In that case try:
- `--no-tiling` to isolate tiling
- `--postprocess-backend metal` (float32 crop/resize + tallying)
- `--include-rays --ray-viz` to check rays also blend smoothly across seams

**Partial fix available:** `Scripts/reconvert_dualdpt.sh` replaces ConvTranspose2d with BilinearUpsampleConv
```bash
./Scripts/reconvert_dualdpt.sh Models/checkpoints/da3_giant/model.safetensors
```

**Alternative mitigations:**
- Post-processing with bilateral/guided filter
- Using higher input resolution (increases patch count)
- Depth refinement networks

### Issue: Depth Map Color/Mapping

**Status:** Clarified / matches original DA3 by default

The **PNG** is just a visualization. By default this repo now matches the original DA3 `visualize_depth()` convention:
- Convert depth → **inverse-depth** (`1/depth`) for valid pixels
- Percentile-normalize (2–98%)
- Invert so **closer = warmer** (`--depth-viz-style da3`)

If you want raw depth visualization (farther = warmer), use `--depth-viz-style depth`. `--invert-depth-viz` flips the final colormap either way (PNG only; it does not change `.da3` depth values).

### Issue: Rays are NaN or Look Random

**Status:** FIXED / workaround documented

If ray outputs are **NaN** or the ray visualizations look like random noise:

- If the issue is **NaN/Inf** in `ray_confidence` (common with fp16 `exp()` overflow):
  - Export the head with **confidence logits** (`--conf-activation linear` in `Scripts/convert_dualdpt_official_to_coreml.py`)
  - Apply activation outside CoreML in float32 (`DA3CoreML.Config.confidenceActivation = .expp1` or `.softplus1`)
    - For large batches/tiling, keep this on GPU with `--postprocess-backend metal` (activates in float32 via Metal kernels).
- If the issue is inside the ray branch itself (**NaN/Inf in `rays`**):
  - Use a **float32 DualDPT head** and force the head to run on **CPU-only** (`--head-cpu-only`)
- Generate ray sanity-check PNGs:
  - `--include-rays --ray-viz`

This matches DA3 conventions and avoids float16 numerical issues in the ray head. Also ensure you’re on a recent version of
this repo: aux-resolution ray cropping is now scaled correctly, and tiled inference uses a **global** ray pass (rather than
stitching per-tile rays) so `--ray-pose` stays meaningful on large images.

**What should `*_rays_dir.png` look like?**
- Often a **smooth color gradient**. That’s normal: channels 0–2 encode a *camera ray field* (used to estimate intrinsics/rotation),
  not per-pixel scene geometry. It won’t “trace object edges” the way depth does.
- Red flags: flat single-color outputs, obvious checkerboard/tiling artifacts, or NaN/Inf (often from fp16 `exp()` overflow upstream).

### Issue: GSHead Fusion Looks Like Giant Planes / Wrong Scale

If your fused feed-forward 3DGS looks like huge “rectangles” or the scene scale is enormous, the most common culprit is the GSHead `offset_depth` channel dominating the base depth (e.g. `offset_depth` in the hundreds/thousands while base depth is ~0–5).

**This repo has been triple-checked end-to-end** (image → backbone → DualDPT depth → GSHead params → unprojection → PLY):
- The unprojection math matches DA3’s `GaussianAdapter` (`depth` is a ray distance along a **unit** camera ray; `offset_xy` perturbs pixel centers; `offset_depth` is added to depth).
- The failure mode is reproducible with the included models: `offset_depth` can be orders of magnitude larger than the base depth, which explodes world-space Z unless you scale/disable it.

**How to confirm quickly:**
- Run `da3-coreml fuse ... --verbose`
- Look for:
  - `✓ depth range (model): ...`
  - `✓ gs offset_depth range: ...`
  - If `offset_depth_max >> depth_max`, the CLI prints a warning and a suggested scale to try.

The suggested value is computed as ~`depth_max / offset_depth_max`. Example (single image from `sample_images/`): `depth_max≈4.08`, `offset_depth_max≈708.5` → suggested `--gs-offset-depth-scale≈0.0058`, which brings the fused Z range back to ~`0..4`.

Debug knobs:
- Disable `offset_depth` entirely:
  - `--gs-disable-offset-depth`
- Or keep it but scale it down:
  - `--gs-offset-depth-scale 0.01`
  - In practice, `0.001`–`0.01` is a good starting search range. On `sample_images` (30-view sample), `0.003` keeps Z in a sane range.

**Checked vs the original DA3 Python implementation (PyTorch):**
- This is **not CoreML-only**: using upstream `GSDPT` (`depth_anything_3/model/gsdpt.py`, `pos_embed=True`) + the official `da3-giant` checkpoint, `offset_depth` outliers are present as well.
- Example (single image, square 518×518 preprocess to match this repo’s CoreML CLI): on `sample_images/20251115_072731.jpg`, upstream PyTorch produces `offset_depth max ≈ 716` (mean ≈ 86.8) while base depth is `~0.18..4.1`.
- Conclusion: if you want a *nice-looking feed-forward PLY*, you generally need to **scale/disable** `offset_depth` (or add additional pruning based on `depth + offset_depth`), regardless of whether you run Python/CUDA or CoreML.
- Online check: as of 2026-01-01, no upstream public issues/PRs in `ByteDance-Seed/Depth-Anything-3` mention `offset_depth` by name, but multiple issues report “garbled / poor” 3DGS results and pose/extrinsics alignment pitfalls (e.g. #46, #100, #136). Upstream also suggests trying updated checkpoints for few-view fusion (e.g. `DA3NESTED-GIANT-LARGE-1.1` mentioned in #136).

**Important note (confidence doesn’t save you):**
- In upstream PyTorch, `offset_depth` outliers are often **high-confidence** (e.g. `gs_conf≈1.0` even for the top 0.1% largest `offset_depth` pixels; Pearson correlation ~0.84 on the sample image above).
- That means increasing `--gs-min-confidence` usually **does not remove** the exploding-depth Gaussians. You still need to scale/disable/prune `offset_depth`.

**GSHead re-export parity (pos-embed):**
- `Scripts/convert_gshead_to_coreml.py` now supports `--pos-embed/--no-pos-embed` so you can export a GSHead that matches DA3’s default (`pos_embed=True`) behavior.
- This improves parity with upstream, but it **does not eliminate** the `offset_depth` outliers shown above (they exist in the upstream model too).

**Reproduce the upstream PyTorch stats locally (recommended sanity check):**
```bash
python3.11 -m venv .venv
.venv/bin/pip install torch safetensors numpy pillow einops addict omegaconf
.venv/bin/python Scripts/check_gshead_offset_depth_pytorch.py \
  --checkpoint Models/checkpoints/da3_giant/model.safetensors \
  --image sample_images/20251115_072731.jpg
```

**Re-export GSHead with pos-embed enabled (CoreML):**
```bash
.venv/bin/pip install coremltools
.venv/bin/python Scripts/convert_gshead_to_coreml.py \
  --checkpoint Models/checkpoints/da3_giant/model.safetensors \
  --output Models/gshead_giant_posembed.mlpackage \
  --pos-embed
xcrun coremlc compile Models/gshead_giant_posembed.mlpackage Models/compiled
```
Note: the current CoreML export precomputes the pos-embed tensors for the fixed input size, which increases the `.mlpackage` size (for 518×518 it’s roughly +70MB).

**Traceability:** the fused PLY header includes comments like:
- `comment gs_offset_depth: true|false`
- `comment gs_offset_depth_scale: <value>`

**Script shortcut:** `Scripts/fuse__30.sh` accepts:
- `GS_OFFSET_DEPTH_SCALE=<float>` (default `1.0`)
- `GS_DISABLE_OFFSET_DEPTH=1` (disables it)

**If you want to “delete and recompile everything” (compiled CoreML only):**
```bash
rm -rf Models/compiled && mkdir -p Models/compiled
for pkg in Models/*.mlpackage; do xcrun coremlc compile "$pkg" Models/compiled; done
```
This rebuilds `.mlmodelc` artifacts from the on-disk `.mlpackage` models (it does not re-export weights from PyTorch).

Also note:
- If you are using `--use-ray-pose`, ensure your code treats the returned matrix as **c2w** (DA3 returns w2c then inverts).
- `--use-ray-pose` estimates intrinsics/rotation well, but on **single-image** runs the predicted translation can be near-zero. If you need more camera motion for fusion, try `--camdec Models/compiled/camdec_giant.mlmodelc` instead of `--use-ray-pose`.

### Architecture Deep-Dive (For Debugging)

**Original DA3 Data Flow (from `src/depth_anything_3/`):**

```
Input: [B, S, 3, H, W]  (S=num_views, typically S=1 for single image)
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ DinoV2 Backbone (model/dinov2/dinov2.py)                           │
│                                                                     │
│ • get_intermediate_layers() returns:                               │
│   ((features, camera_tokens), aux_outputs)                         │
│                                                                     │
│ • features = list of 4 tuples [(tensor, cam_token), ...]           │
│   where tensor shape = (B, S, N, C)                                │
│   N = num_patches = 1369 (for 518x518, patch=14: 37×37)           │
│   C = embed_dim * 2 (cat_token=True: 1536*2=3072 for giant)       │
│                                                                     │
│ • Token structure: [CLS→cam_token, (register_tokens...), patch_1..patch_1369] │
│   `get_intermediate_layers()` strips special tokens → heads use `patch_start_idx=0` │
└─────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ DualDPT Head (model/dualdpt.py)                                    │
│                                                                     │
│ Input: feats = List[Tuple[Tensor, CamToken]]                       │
│        Each tensor: (B, S, N, C) where N=1369, C=3072              │
│                                                                     │
│ Line 181: B, S, N, C = feats[0][0].shape                          │
│ Line 182: feats = [feat[0].reshape(B*S, N, C) for feat in feats]  │
│                                                                     │
│ _forward_impl():                                                   │
│   Line 216: ph, pw = H // patch_size, W // patch_size  # 37, 37   │
│   Line 221: x = x.permute(0, 2, 1).reshape(B, C, ph, pw)          │
│             # (B*S, 1369, 3072) → (B*S, 3072, 37, 37)              │
│                                                                     │
│ Fusion chain:                                                      │
│   • refinenet4: (37,37) → (19,19) via /2 conv                     │
│   • refinenet3: (19,19) → (37,37) via 2x upsample                 │
│   • refinenet2: (37,37) → (74,74) via 2x upsample                 │
│   • refinenet1: (74,74) → (148,148) via 2x upsample               │
│                                                                     │
│ Line 236-237: Bilinear upsample to (H, W) = (518, 518)            │
│                                                                     │
│ Output: depth (B, S, 1, H, W), rays (B, S, 6, H, W)               │
└─────────────────────────────────────────────────────────────────────┘
```

**CoreML Conversion Issues Found:**

1. **`convert_da3_backbone_to_coreml.py` local_x tracking (FIXED):**
   The script now correctly implements DA3's alternating attention pattern:
   ```python
   # Before alt_start: all blocks are local attention
   if self.alt_start == -1 or i < self.alt_start:
       local_x = x
   # From alt_start onwards: only even blocks are local attention
   elif i % 2 == 0:
       local_x = x
   # Odd blocks (i % 2 == 1) are global attention - don't update local_x
   ```
   The `alt_start` values per model size: small/base=4, large=8, giant=13.

2. **`convert_dualdpt_to_coreml.py` line 185-186:**
   ```python
   # Uses ConvTranspose2d which causes checkerboard artifacts
   nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
   nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
   ```
   Fix: Replace with `BilinearUpsampleConv` (already defined in file, now being used)

### Scripts Reference

| Script | Purpose |
|--------|---------|
| `Scripts/setup.sh` | Create a Python venv and install conversion deps (`torch`, `coremltools`, ...) |
| `Scripts/convert_da3_backbone_to_coreml.py` | Convert the **official DA3 DINOv2** backbone (recommended) |
| `Scripts/convert_dualdpt_official_to_coreml.py` | Convert the **official DA3 DualDPT** head (with optional logits export) |
| `Scripts/convert_dualdpt_official.sh` | Wrapper: convert + `coremlc compile` (recommended) |
| `Scripts/validate_dualdpt_official_coreml.py` | PyTorch vs CoreML head validation (sanity check) |
| `Scripts/check_gshead_offset_depth_pytorch.py` | Upstream PyTorch GSHead `offset_depth`/confidence stats (square 518×518 preprocess to match this repo’s CoreML CLI) |
| `Scripts/check_camdec_token_mode_pytorch.py` | Upstream PyTorch CamDec check: camera-token vs patch-token feeding (quantifies how much poses/intrinsics can differ) |
| `Scripts/reconvert_dualdpt.sh` | Re-export head with bilinear upsample to reduce checkerboard artifacts |
| `Scripts/smoke_infer_5.sh` | Run inference on 5 images + save depth/rays + ray-pose debug |
| `Scripts/fuse__30.sh` | Sample 30 images and fuse a **feed-forward GSHead** PLY (3DGS). Supports `GS_OFFSET_DEPTH_SCALE` / `GS_DISABLE_OFFSET_DEPTH`. |
| `Scripts/da3_to_npz.py` | Convert `.da3` outputs to `results.npz`-style archives (Python interop) |
| `Scripts/sample_images.py` | Sample images evenly spaced (creates symlinks by default) |
| `Scripts/run_inference.sh` | Unified parallel/sequential inference wrapper |
| `Scripts/test_da3_pytorch.py` | Run the upstream PyTorch DA3 pipeline for comparison |
| `Scripts/parallel_infer.sh` | Legacy parallel processing script |
| `Scripts/clean_outputs.sh` | Remove generated `output_*` folders and other local artifacts |

### Testing Original DA3 PyTorch

To verify PyTorch DA3 produces correct output (for comparison):

```bash
python3 Scripts/test_da3_pytorch.py \
  --checkpoint Models/checkpoints/da3_giant/model.safetensors \
  --image test_images/test_01_small.jpg \
  --output output_pytorch_test
```

**Note:** The test script requires the DA3 source tree available on disk (see the script header for the expected path).

### Key Constants Reference

```python
# DINOv2 Giant (DA3 default)
embed_dim = 1536
depth = 40 layers
num_heads = 24
patch_size = 14
input_size = 518  # → 37×37 = 1369 patches
output_layers = [19, 27, 33, 39]  # Features extracted from these blocks
cat_token = True  # Output dim = 3072 (doubled)

# DualDPT Giant
dim_in = 3072  # matches cat_token output
features = 256  # fusion channel dimension
out_channels = [256, 512, 1024, 1024]  # per-stage channels
output_dim = 2  # depth + confidence logit
ray_out_dim = 7  # ray params + confidence logit
```

---

## Contributing

This project is intended to be publishable and friendly to external contributions.

### Development Setup

**Swift / CoreML**
- macOS 14+ (or iOS 17+ for device deployment)
- Xcode Command Line Tools (`xcode-select --install`)
- SwiftPM workflow:
  - Build: `swift build -c release`
  - Test: `swift test`

If your environment blocks SwiftPM’s sandboxing (rare on normal macOS setups), you can try:
- `swift build -c release --disable-sandbox`
- `swift test --disable-sandbox`
- If module-cache writes are blocked, also set: `CLANG_MODULE_CACHE_PATH=.clang-module-cache`

**Python (only needed for model conversion)**

Conversion scripts live in `Scripts/` and depend on:
- `torch`
- `transformers`
- `coremltools`
- `safetensors`
- `numpy`

Suggested setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers coremltools safetensors numpy
```

### What to Contribute

Good first contributions:
- Documentation fixes and clarifications (especially around conventions/precision)
- Ray stability improvements (without changing DA3 semantics)
- More robust image loading + preprocessing test cases
- Faster CPU-side geometry (ray-pose) using `Accelerate`

Bigger projects:
- CoreML export improvements (e.g. more ANE/GPU-friendly heads)
- GPU postprocess improvements (Metal/MPSGraph) for visualization + depth→point ops
- End-to-end quality tests comparing CoreML outputs vs DA3 PyTorch outputs

### PR Guidelines

- Keep PRs focused (one logical change).
- Add tests when changing math, conventions, or array indexing/strides.
- Don’t commit large artifacts:
  - `.mlmodelc`, `.mlpackage`, checkpoints, and output folders (use local builds instead)
- Update `README.md` when user-facing behavior/flags change.

### Reporting Bugs

When filing an issue, please include:
- macOS version + Apple Silicon model
- Which backbone/head models you used (and their precision)
- Exact CLI command line
- Whether `--head-cpu-only`, `--no-tiling`, `--include-rays`, `--use-ray-pose`, `--camdec` were enabled
- Any printed ranges (min/max depth, offset_depth range, etc.)

### Publishing / Open-Source Checklist

Before making this repository public:
- Add a `LICENSE` file for the Swift/Python code (and ensure it’s compatible with your intended distribution).
- Do **not** publish large artifacts unless you intend to:
  - model checkpoints
  - converted `.mlpackage` / compiled `.mlmodelc`
  - large image datasets (`sample_images`, etc.)
  - `output_*` folders
- Note that model weights/checkpoints remain under their upstream licenses (e.g. DA3 / DINOv2 / DINOv3). Always follow those terms.

## TODO

### High Priority
- [x] Convert DINOv3 backbone to CoreML (pipeline ready)
- [x] Convert DualDPT head structure for DINOv3
- [x] **Download and convert DA3-Giant with pre-trained weights**
- [x] Diagnose blocky depth map issue (documented above - expected behavior)
- [x] Create BilinearUpsampleConv fix for reduced artifacts
- [x] Fix `local_x` tracking in backbone conversion (implemented DA3 alt_start logic)
- [x] Fix tiled inference edge coverage (last tile now placed at edge boundary)

### Medium Priority
- [x] Add camera decoder integration for multi-view fusion (CamDecCoreML + CLI `fuse`)
- [x] ~~Implement depth->point cloud unprojection with camera intrinsics~~ (DA3DepthTo3DGS.swift)
- [x] Add `da3-coreml stream` (DA3-Streaming-style outputs: camera_poses.txt + intrinsic.txt + pcd/combined_pcd.ply)
- [x] Add chunk Sim3 alignment for `da3-coreml stream` (`--align-chunks`)
- [ ] Full upstream `da3_streaming` parity: multi-view CoreML (**S>1**) + loop closure
- [x] Fix ray tiling seams (use global ray pass for tiled inference; scale crop for aux ray grids)
- [x] Metal postprocess backend (GPU crop/resize + tile blending + normalization in float32)
- [x] Confidence-logits workflow (export `conf_activation=linear`, activate in Swift float32)
- [ ] Add COLMAP/GLOMAP camera intrinsics file reader (parsing transforms.json, cameras.bin)
- [ ] Train DualDPT head for DINOv3 backbone (requires GPU cluster)
- [x] Integrate GSDPT/GSHead for direct Gaussian splat output (GSHeadCoreML + DA3GSHeadTo3DGS)
- [ ] Add post-processing filters (bilateral/guided) for depth smoothing

### Low Priority
- [ ] Add progress callbacks for batch processing (reference uses tqdm with progress 0.0-1.0)
- [x] ~~Support for different input sizes~~ (CLI `--input-size`, requires model reconversion per size)
- [x] Metal depth visualization kernels (Spectral/Turbo/Grayscale) for large images (`--viz-backend metal`)

### Completed
- [x] DINOv2-Giant backbone extracted from DA3 checkpoint
- [x] DA3-Giant DualDPT head with trained weights (156 weight tensors)
- [x] Correct output layer indices [19, 27, 33, 39] for Giant model
- [x] cat_token=True support (3072 dim output for Giant)
- [x] DINOv3-Large backbone converted to CoreML
- [x] DINOv3-H+ backbone converted to CoreML
- [x] DualDPT head structure converted (untrained)
- [x] Swift ModelSize enum updated for DINOv3 (huge variant, patch16)
- [x] Coremltools upgraded to 9.0
- [x] Documentation consolidated into README.md
- [x] Full pipeline documentation for Gaussian splatting
- [x] Created unified `run_inference.sh` script
- [x] Created `convert_dinov2_backbone.py` for pure DINOv2 HuggingFace conversion
- [x] Documented depth map blocky artifact root cause
- [x] Fixed `local_x` tracking with proper alt_start logic (blocks < alt_start: all local; blocks >= alt_start: even=local, odd=global)
- [x] Fixed tiled inference edge coverage (last tile in each row/column placed at boundary)
- [x] Fixed tiled inference edge weight bug (image boundary pixels now get full weight instead of zero)
- [x] Full depth→point cloud→3DGS pipeline (DA3DepthTo3DGS.swift, DA3GaussianSplat.swift)
- [x] Multi-view world-space Gaussian fusion with camera extrinsics
- [x] PLY export for 3D Gaussian splats

### Technical Debt / Identified Flaws
- [ ] **CLI**: Default HuggingFace model for DINOv3 conversion incorrectly uses `dinov2` (Sources/DA3CLI/main.swift:1228).
- [ ] **CLI**: Scripts directory lookup is brittle/relative (Sources/DA3CLI/main.swift:1220).
- [ ] **Performance**: Depth activation/visualization uses slow linear `MLMultiArray` indexing (`.floatValue`) instead of direct pointer access (Sources/DA3CoreML/DA3CoreML.swift).
- [ ] **Performance**: Metal post-processor compiles source at runtime; should use precompiled `.metallib` for iOS suitability (Sources/DA3CoreML/DA3MetalPostProcessor.swift).
- [ ] **File Format**: `.da3` header lacks explicit ray channel count/shape, forcing readers to assume 6 channels (Sources/DA3CoreML/DA3OutputWriter.swift).

---

## Troubleshooting

### All images produce identical depth / “rainbow gradient”

This usually means the preprocessing step is seeing a **blank (all-zero) image buffer**.
On macOS this can happen with some real-world JPEGs that ImageIO loads but CoreGraphics fails to render.

Fixes:
- Install `ffmpeg` (the CLI auto-falls-back to it when CoreGraphics renders blank): `brew install ffmpeg`
- Or convert the inputs to PNG and rerun.

### "Out of memory" Error

```swift
var config = DA3CoreML.Config()
config.modelSize = .base           // Use smaller model
config.safetyBufferPercent = 0.40  // Increase safety buffer
config.maxBatchSize = 1            // Single image at a time
config.enableTiling = true         // Enable tiling for large images
config.maxTileSize = 512           // Smaller tiles
```

### "Memory pressure is terminal"

System is critically low on memory:
1. Close other applications
2. Use a smaller model size
3. Increase safety buffer
4. Process fewer images at once

### Slow Performance

```swift
config.useGPU = true  // Ensure GPU is enabled (default)
```

### Model Loading Fails

1. Check paths are correct
2. Verify .mlmodelc (compiled) or .mlpackage format
3. Ensure models were converted successfully

### Garbage Depth Output

1. Check token extraction (should skip 1 for DINOv2)
2. Verify ImageNet normalization values
3. Ensure input size matches model (518x518)

---

## License

### Code (this repository)

MIT License (see `LICENSE` once you add it).

### Third‑Party Models / Weights (Important)

This repository is a **re-implementation + conversion pipeline**. It does **not** grant you any rights to redistribute third‑party weights.

- **Depth-Anything-3 (DA3)** code is Apache-2.0 upstream, but **checkpoints / model weights may use different licenses** per model card.
- **DINOv2/DINOv3** backbones are published by Meta, and their weights also have their own licenses/terms.
- If you publish **converted CoreML** models, treat them as **derivative artifacts**: keep attribution and follow the **original weight license**.

Practical checklist before you upload anything:
1. Find the upstream model card (e.g. on Hugging Face) and read its `license` field.
2. If the license is **non-commercial** (e.g. CC BY‑NC), you may still be allowed to redistribute, but **only under those terms**.
3. If the license is unclear or restrictive, **do not upload** converted weights — publish only the code/scripts and require users to download weights themselves.

This section is informational, not legal advice.

## Citation

If you use this project, please cite the upstream work **and** this repo’s conversion/Apple implementation.

```bibtex
@software{da3coreml,
  title  = {DA3CoreML: Depth-Anything-3 for Apple Silicon (CoreML)},
  author = {<YOUR NAME OR ORG>},
  year   = {2025},
  url    = {<YOUR REPO URL>}
}
```

Upstream citations:
- Depth-Anything-3 (use the citation provided by the DA3 authors / model card / paper).
- DINOv2 / DINOv3 (use Meta’s citation guidance).

## Publishing to Hugging Face (Credit + Compliance)

You can publish CoreML exports on the Hugging Face Hub (and keep them private until you’re ready).

Recommended workflow:
1. **Pick a source checkpoint** (e.g. `depth-anything/DA3-BASE`) and verify its **weight license** (this controls what you can redistribute).
2. Convert/export CoreML with this repo’s scripts (prefer distributing `.mlpackage/`, which users can compile locally).
3. Create a Hugging Face *model* repo (name example: `<your-handle>/DA3-BASE-CoreML`).
4. In the model card (`README.md` in the HF repo), include:
   - “Converted by <you> using DA3CoreML” + link to your GitHub repo (this is how you get credit).
   - Link back to the upstream model + paper/repo.
   - The correct `license:` metadata (must match the upstream weight license).
   - Exact input normalization + output semantics (depth vs inverse-depth, ray directions, etc.).
5. Upload artifacts (large files use LFS; `.mlpackage` is a directory).
6. Only after validation, flip the HF repo to **public**.

Tip: if you want the Swift code to be open to contributions earlier, open-source the code repo first and publish weights later (or never publish weights if the license is restrictive).

## Acknowledgments

- [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) - Upstream DA3 model/research
- [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) - Original implementation
- [DINOv2](https://github.com/facebookresearch/dinov2) - Vision transformer backbone
- [Apple CoreML](https://developer.apple.com/documentation/coreml) - ML framework
