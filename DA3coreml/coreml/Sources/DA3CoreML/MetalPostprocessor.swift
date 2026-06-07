import Foundation
import CoreGraphics
import CoreML
import Metal
import simd

/// Metal-accelerated postprocessing for DA3CoreML.
///
/// This is intentionally scoped to predictable, data-parallel operations:
/// - crop + bilinear resize (depth / confidence / rays)
/// - tiled blend ("tallying") and normalization
/// - stable confidence activation (logits -> weights) in float32
/// - depth visualization and depth unprojection (optional helpers)
///
/// Neural inference remains in CoreML (ANE/GPU/CPU). These ops are not a good fit for ANE and
/// are typically faster (and more numerically stable) when executed on GPU in float32.
@available(macOS 14.0, iOS 17.0, *)
final class DA3MetalPostProcessor {

    // MARK: - Metal

    private static let sharedLock = NSLock()
    private static var sharedInstance: DA3MetalPostProcessor?

    /// Shared Metal postprocessor instance.
    ///
    /// This avoids recompiling the inlined `.metal` source on every call-site (which is expensive).
    static func shared() -> DA3MetalPostProcessor? {
        sharedLock.lock()
        defer { sharedLock.unlock() }
        if let inst = sharedInstance { return inst }
        guard let inst = try? DA3MetalPostProcessor() else { return nil }
        sharedInstance = inst
        return inst
    }

    private let device: MTLDevice
    private let queue: MTLCommandQueue

    private let resizeCropF16: MTLComputePipelineState
    private let resizeCropF32: MTLComputePipelineState
    private let blendTileDepthConf: MTLComputePipelineState
    private let blendTileRays: MTLComputePipelineState
    private let normalize1CHW: MTLComputePipelineState
    private let normalizeCHW: MTLComputePipelineState
    private let activateConfidenceF16: MTLComputePipelineState
    private let activateConfidenceF32: MTLComputePipelineState
    private let unprojectDepthRayF16: MTLComputePipelineState
    private let unprojectDepthRayF32: MTLComputePipelineState
    private let unprojectGSWorldF16: MTLComputePipelineState
    private let unprojectGSWorldF32: MTLComputePipelineState
    private let visualizeDepthRGBAF16: MTLComputePipelineState
    private let visualizeDepthRGBAF32: MTLComputePipelineState

    // MARK: - Init

    init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw DA3Error.inferenceError("Metal not available (no MTLDevice)")
        }
        self.device = dev
        guard let q = dev.makeCommandQueue() else {
            throw DA3Error.inferenceError("Metal not available (failed to create command queue)")
        }
        self.queue = q

        let library = try dev.makeLibrary(source: Self.metalSource, options: nil)

        func makePSO(_ name: String) throws -> MTLComputePipelineState {
            guard let fn = library.makeFunction(name: name) else {
                throw DA3Error.inferenceError("Missing Metal function '\(name)'")
            }
            return try dev.makeComputePipelineState(function: fn)
        }

        self.resizeCropF16 = try makePSO("resizeCropCHW_f16")
        self.resizeCropF32 = try makePSO("resizeCropCHW_f32")
        self.blendTileDepthConf = try makePSO("blendTileDepthConf")
        self.blendTileRays = try makePSO("blendTileRays")
        self.normalize1CHW = try makePSO("normalize1CHW")
        self.normalizeCHW = try makePSO("normalizeCHW")
        self.activateConfidenceF16 = try makePSO("activateConfidence1CHW_f16")
        self.activateConfidenceF32 = try makePSO("activateConfidence1CHW_f32")
        self.unprojectDepthRayF16 = try makePSO("unprojectDepthRay_f16")
        self.unprojectDepthRayF32 = try makePSO("unprojectDepthRay_f32")
        self.unprojectGSWorldF16 = try makePSO("unprojectGSDepthToWorldXYZ_f16")
        self.unprojectGSWorldF32 = try makePSO("unprojectGSDepthToWorldXYZ_f32")
        self.visualizeDepthRGBAF16 = try makePSO("visualizeDepthRGBA_f16")
        self.visualizeDepthRGBAF32 = try makePSO("visualizeDepthRGBA_f32")
    }

    // MARK: - Public Ops

    struct CropRect {
        var startX: Int
        var startY: Int
        var width: Int
        var height: Int
    }

    /// Crop a CHW/BCHW tensor (B must be 1 if present) and bilinear-resize to the target size.
    ///
    /// - Returns: float32 tensor of shape `[C, outH, outW]` (or `[1, outH, outW]` for single channel).
    func resizeCropCHW(
        input: MLMultiArray,
        channels: Int,
        inWidth: Int,
        inHeight: Int,
        crop: CropRect,
        outWidth: Int,
        outHeight: Int
    ) throws -> MLMultiArray {
        // CoreML can return non-contiguous MLMultiArrays. Metal kernels in this file assume
        // row-major contiguous CHW/BCHW (B must be 1 if present). If the input is not contiguous,
        // materialize a float32 contiguous copy first.
        var inputForGPU: MLMultiArray = input
        if let reader = try? MLMultiArrayFloatReader(input), !reader.isContiguousRowMajor() {
            let data = reader.readAll()
            let contig = try MLMultiArray(shape: input.shape, dataType: .float32)
            let ptr = UnsafeMutablePointer<Float>(OpaquePointer(contig.dataPointer))
            for i in 0..<data.count { ptr[i] = data[i] }
            inputForGPU = contig
        }

        let out = try MLMultiArray(
            shape: [NSNumber(value: channels), NSNumber(value: outHeight), NSNumber(value: outWidth)],
            dataType: .float32
        )

        let inBuf = try makeBuffer(for: inputForGPU)
        let outBuf = try makeBuffer(for: out)

        var params = ResizeCropParams(
            inW: UInt32(inWidth),
            inH: UInt32(inHeight),
            outW: UInt32(outWidth),
            outH: UInt32(outHeight),
            channels: UInt32(channels),
            startX: UInt32(crop.startX),
            startY: UInt32(crop.startY),
            cropW: UInt32(crop.width),
            cropH: UInt32(crop.height)
        )

        let pso: MTLComputePipelineState
        switch inputForGPU.dataType {
        case .float16: pso = resizeCropF16
        case .float32: pso = resizeCropF32
        default:
            throw DA3Error.invalidInput("Metal resizeCrop only supports float16/float32 inputs (got \(inputForGPU.dataType))")
        }

        let cmd = try makeCommandBuffer()
        guard let enc = cmd.makeComputeCommandEncoder() else {
            throw DA3Error.inferenceError("Metal: failed to create compute encoder")
        }
        enc.label = "da3.resizeCropCHW"
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&params, length: MemoryLayout<ResizeCropParams>.stride, index: 2)

        let grid = MTLSize(width: outWidth, height: outHeight, depth: channels)
        let tg = Self.threadsPerThreadgroup2D(pso: pso)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        return out
    }

    /// Accumulate a single tile into global depth/confidence/weights buffers using DA3-style edge ramps.
    func blendTileDepthConf(
        tileDepth: MLMultiArray,
        tileConf: MLMultiArray,
        outDepth: MLMultiArray,
        outConf: MLMultiArray,
        weights: MLMultiArray,
        atX: Int,
        atY: Int,
        tileW: Int,
        tileH: Int,
        overlap: Int,
        outW: Int,
        outH: Int
    ) throws {
        let tileDepthBuf = try makeBuffer(for: tileDepth)
        let tileConfBuf = try makeBuffer(for: tileConf)
        let outDepthBuf = try makeBuffer(for: outDepth)
        let outConfBuf = try makeBuffer(for: outConf)
        let weightsBuf = try makeBuffer(for: weights)

        var params = BlendParams(
            outW: UInt32(outW),
            outH: UInt32(outH),
            tileW: UInt32(tileW),
            tileH: UInt32(tileH),
            atX: UInt32(atX),
            atY: UInt32(atY),
            overlap: UInt32(overlap)
        )

        let cmd = try makeCommandBuffer()
        guard let enc = cmd.makeComputeCommandEncoder() else {
            throw DA3Error.inferenceError("Metal: failed to create compute encoder")
        }
        enc.label = "da3.blendTileDepthConf"
        enc.setComputePipelineState(blendTileDepthConf)
        enc.setBuffer(tileDepthBuf, offset: 0, index: 0)
        enc.setBuffer(tileConfBuf, offset: 0, index: 1)
        enc.setBuffer(outDepthBuf, offset: 0, index: 2)
        enc.setBuffer(outConfBuf, offset: 0, index: 3)
        enc.setBuffer(weightsBuf, offset: 0, index: 4)
        enc.setBytes(&params, length: MemoryLayout<BlendParams>.stride, index: 5)

        let grid = MTLSize(width: tileW, height: tileH, depth: 1)
        let tg = Self.threadsPerThreadgroup2D(pso: blendTileDepthConf)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    /// Accumulate a ray tile into global rays/rayConf buffers (weights are provided externally).
    ///
    /// This accumulates with the same spatial ramp weights as depth tiling so that ray fields stay
    /// smooth across tile seams. Normalization is done later via `normalizeCHW/normalize1CHW`.
    func blendTileRays(
        tileRays: MLMultiArray,
        tileConf: MLMultiArray,
        outRays: MLMultiArray,
        outConf: MLMultiArray,
        atX: Int,
        atY: Int,
        tileW: Int,
        tileH: Int,
        overlap: Int,
        outW: Int,
        outH: Int,
        channels: Int
    ) throws {
        let tileRaysBuf = try makeBuffer(for: tileRays)
        let tileConfBuf = try makeBuffer(for: tileConf)
        let outRaysBuf = try makeBuffer(for: outRays)
        let outConfBuf = try makeBuffer(for: outConf)

        var params = BlendRaysParams(
            outW: UInt32(outW),
            outH: UInt32(outH),
            tileW: UInt32(tileW),
            tileH: UInt32(tileH),
            atX: UInt32(atX),
            atY: UInt32(atY),
            overlap: UInt32(overlap),
            channels: UInt32(channels)
        )

        let cmd = try makeCommandBuffer()
        guard let enc = cmd.makeComputeCommandEncoder() else {
            throw DA3Error.inferenceError("Metal: failed to create compute encoder")
        }
        enc.label = "da3.blendTileRays"
        enc.setComputePipelineState(blendTileRays)
        enc.setBuffer(tileRaysBuf, offset: 0, index: 0)
        enc.setBuffer(tileConfBuf, offset: 0, index: 1)
        enc.setBuffer(outRaysBuf, offset: 0, index: 2)
        enc.setBuffer(outConfBuf, offset: 0, index: 3)
        enc.setBytes(&params, length: MemoryLayout<BlendRaysParams>.stride, index: 4)

        let grid = MTLSize(width: tileW, height: tileH, depth: 1)
        let tg = Self.threadsPerThreadgroup2D(pso: blendTileRays)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    func normalize1CHW(values: MLMultiArray, weights: MLMultiArray, width: Int, height: Int) throws {
        let vBuf = try makeBuffer(for: values)
        let wBuf = try makeBuffer(for: weights)
        var params = NormParams(width: UInt32(width), height: UInt32(height))

        let cmd = try makeCommandBuffer()
        guard let enc = cmd.makeComputeCommandEncoder() else {
            throw DA3Error.inferenceError("Metal: failed to create compute encoder")
        }
        enc.label = "da3.normalize1CHW"
        enc.setComputePipelineState(normalize1CHW)
        enc.setBuffer(vBuf, offset: 0, index: 0)
        enc.setBuffer(wBuf, offset: 0, index: 1)
        enc.setBytes(&params, length: MemoryLayout<NormParams>.stride, index: 2)

        let grid = MTLSize(width: width, height: height, depth: 1)
        let tg = Self.threadsPerThreadgroup2D(pso: normalize1CHW)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    func normalizeCHW(values: MLMultiArray, weights: MLMultiArray, channels: Int, width: Int, height: Int) throws {
        let vBuf = try makeBuffer(for: values)
        let wBuf = try makeBuffer(for: weights)
        var params = NormCHWParams(width: UInt32(width), height: UInt32(height), channels: UInt32(channels))

        let cmd = try makeCommandBuffer()
        guard let enc = cmd.makeComputeCommandEncoder() else {
            throw DA3Error.inferenceError("Metal: failed to create compute encoder")
        }
        enc.label = "da3.normalizeCHW"
        enc.setComputePipelineState(normalizeCHW)
        enc.setBuffer(vBuf, offset: 0, index: 0)
        enc.setBuffer(wBuf, offset: 0, index: 1)
        enc.setBytes(&params, length: MemoryLayout<NormCHWParams>.stride, index: 2)

        let grid = MTLSize(width: width, height: height, depth: channels)
        let tg = Self.threadsPerThreadgroup2D(pso: normalizeCHW)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    /// Apply confidence activation to logits in float32 on GPU.
    ///
    /// This is useful when you export the head with `conf_activation="linear"` and want to apply
    /// `expp1`/`softplus1` outside CoreML without falling back to CPU loops.
    func activateConfidence1CHW(
        logits: MLMultiArray,
        width: Int,
        height: Int,
        activation: DA3CoreML.ConfidenceActivation,
        clampMin: Float,
        clampMax: Float
    ) throws -> MLMultiArray {
        if activation == .linear {
            return logits
        }

        // Ensure contiguous row-major; if not, materialize float32.
        var inputForGPU: MLMultiArray = logits
        if let reader = try? MLMultiArrayFloatReader(logits), !reader.isContiguousRowMajor() {
            let data = reader.readAll()
            let contig = try MLMultiArray(shape: logits.shape, dataType: .float32)
            let ptr = UnsafeMutablePointer<Float>(OpaquePointer(contig.dataPointer))
            for i in 0..<data.count { ptr[i] = data[i] }
            inputForGPU = contig
        }

        let out = try MLMultiArray(shape: logits.shape, dataType: .float32)
        let inBuf = try makeBuffer(for: inputForGPU)
        let outBuf = try makeBuffer(for: out)

        var params = ActivateParams(
            width: UInt32(width),
            height: UInt32(height),
            clampMin: clampMin,
            clampMax: clampMax,
            mode: activation == .expp1 ? 1 : 2
        )

        let pso: MTLComputePipelineState
        switch inputForGPU.dataType {
        case .float16: pso = activateConfidenceF16
        case .float32: pso = activateConfidenceF32
        default:
            throw DA3Error.invalidInput("Metal activateConfidence only supports float16/float32 inputs (got \(inputForGPU.dataType))")
        }

        let cmd = try makeCommandBuffer()
        guard let enc = cmd.makeComputeCommandEncoder() else {
            throw DA3Error.inferenceError("Metal: failed to create compute encoder")
        }
        enc.label = "da3.activateConfidence1CHW"
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&params, length: MemoryLayout<ActivateParams>.stride, index: 2)

        let grid = MTLSize(width: width, height: height, depth: 1)
        let tg = Self.threadsPerThreadgroup2D(pso: pso)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        return out
    }

    /// Unproject a depth map (1×H×W) into camera-space points (3×H×W), using DA3 semantics:
    /// depth is interpreted as distance along a **unit** camera ray direction.
    func unprojectDepthRayLengthToCameraXYZ(
        depth: MLMultiArray,
        width: Int,
        height: Int,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float
    ) throws -> MLMultiArray {
        // Ensure contiguous row-major; if not, materialize float32.
        var inputForGPU: MLMultiArray = depth
        if let reader = try? MLMultiArrayFloatReader(depth), !reader.isContiguousRowMajor() {
            let data = reader.readAll()
            let contig = try MLMultiArray(shape: depth.shape, dataType: .float32)
            let ptr = UnsafeMutablePointer<Float>(OpaquePointer(contig.dataPointer))
            for i in 0..<data.count { ptr[i] = data[i] }
            inputForGPU = contig
        }

        let out = try MLMultiArray(
            shape: [3, NSNumber(value: height), NSNumber(value: width)],
            dataType: .float32
        )

        let inBuf = try makeBuffer(for: inputForGPU)
        let outBuf = try makeBuffer(for: out)

        var params = UnprojectParams(
            width: UInt32(width),
            height: UInt32(height),
            fx: fx,
            fy: fy,
            cx: cx,
            cy: cy
        )

        let pso: MTLComputePipelineState
        switch inputForGPU.dataType {
        case .float16: pso = unprojectDepthRayF16
        case .float32: pso = unprojectDepthRayF32
        default:
            throw DA3Error.invalidInput("Metal unprojectDepth only supports float16/float32 inputs (got \(inputForGPU.dataType))")
        }

        let cmd = try makeCommandBuffer()
        guard let enc = cmd.makeComputeCommandEncoder() else {
            throw DA3Error.inferenceError("Metal: failed to create compute encoder")
        }
        enc.label = "da3.unprojectDepthRay"
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&params, length: MemoryLayout<UnprojectParams>.stride, index: 2)

        let grid = MTLSize(width: width, height: height, depth: 1)
        let tg = Self.threadsPerThreadgroup2D(pso: pso)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        return out
    }

    /// Unproject DA3 GSHead inputs to world-space XYZ on GPU.
    ///
    /// This matches the DA3 `GaussianAdapter` convention:
    /// - `depth` is a ray distance along a **unit** ray direction
    /// - `gsParams` supplies `offset_xy` (channels 0..1) and optionally `offset_depth` (channel 36)
    /// - world point = `c2w * (dir_unit * (depth + offset_depth * scale))`
    ///
    /// This is a convenience kernel for feed-forward 3DGS pipelines; it intentionally does *not*
    /// apply confidence/border pruning (that stays on CPU for flexibility).
    func unprojectGSDepthToWorldXYZ(
        depth: MLMultiArray,
        gsParams: MLMultiArray,
        width: Int,
        height: Int,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        applyOffsetXY: Bool,
        applyOffsetDepth: Bool,
        offsetDepthScale: Float,
        c2w: simd_float4x4
    ) throws -> MLMultiArray {
        // Ensure contiguous row-major; if not, materialize float32.
        func makeContigFloat32(_ arr: MLMultiArray) throws -> MLMultiArray {
            if let reader = try? MLMultiArrayFloatReader(arr), reader.isContiguousRowMajor(), arr.dataType == .float32 {
                return arr
            }
            let data = (try? MLMultiArrayFloatReader(arr).readAll()) ?? (0..<arr.count).map { arr[$0].floatValue }
            let contig = try MLMultiArray(shape: arr.shape, dataType: .float32)
            let ptr = UnsafeMutablePointer<Float>(OpaquePointer(contig.dataPointer))
            for i in 0..<data.count { ptr[i] = data[i] }
            return contig
        }

        var depthForGPU = depth
        var gsForGPU = gsParams

        if let r = try? MLMultiArrayFloatReader(depth), !r.isContiguousRowMajor() {
            depthForGPU = try makeContigFloat32(depth)
        }
        if let r = try? MLMultiArrayFloatReader(gsParams), !r.isContiguousRowMajor() {
            gsForGPU = try makeContigFloat32(gsParams)
        }

        // Keep dtypes aligned (kernels are specialized for f16/f32 pairs).
        if depthForGPU.dataType != gsForGPU.dataType {
            depthForGPU = try makeContigFloat32(depthForGPU)
            gsForGPU = try makeContigFloat32(gsForGPU)
        }

        let out = try MLMultiArray(
            shape: [3, NSNumber(value: height), NSNumber(value: width)],
            dataType: .float32
        )

        let depthBuf = try makeBuffer(for: depthForGPU)
        let gsBuf = try makeBuffer(for: gsForGPU)
        let outBuf = try makeBuffer(for: out)

        var flags: UInt32 = 0
        if applyOffsetXY { flags |= 1 }
        if applyOffsetDepth { flags |= 2 }

        var params = UnprojectGSParams(
            width: UInt32(width),
            height: UInt32(height),
            fx: fx,
            fy: fy,
            cx: cx,
            cy: cy,
            offsetDepthScale: offsetDepthScale,
            flags: flags,
            c2w: c2w
        )

        let pso: MTLComputePipelineState
        switch depthForGPU.dataType {
        case .float16: pso = unprojectGSWorldF16
        case .float32: pso = unprojectGSWorldF32
        default:
            throw DA3Error.invalidInput("Metal unprojectGS only supports float16/float32 inputs (got \(depthForGPU.dataType))")
        }

        let cmd = try makeCommandBuffer()
        guard let enc = cmd.makeComputeCommandEncoder() else {
            throw DA3Error.inferenceError("Metal: failed to create compute encoder")
        }
        enc.label = "da3.unprojectGSWorld"
        enc.setComputePipelineState(pso)
        enc.setBuffer(depthBuf, offset: 0, index: 0)
        enc.setBuffer(gsBuf, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBytes(&params, length: MemoryLayout<UnprojectGSParams>.stride, index: 3)

        let grid = MTLSize(width: width, height: height, depth: 1)
        let tg = Self.threadsPerThreadgroup2D(pso: pso)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        return out
    }

    /// Visualize depth to an RGBA CGImage on GPU.
    ///
    /// Important: percentile min/max (for DA3-style inverse-depth) is computed on CPU by the caller.
    func visualizeDepthToCGImage(
        depth: MLMultiArray,
        width: Int,
        height: Int,
        depthMin: Float,
        depthMax: Float,
        style: DepthVisualizationStyle,
        colormap: Colormap,
        invert: Bool
    ) throws -> CGImage {
        // Ensure contiguous row-major; if not, materialize float32.
        var inputForGPU: MLMultiArray = depth
        if let reader = try? MLMultiArrayFloatReader(depth), !reader.isContiguousRowMajor() {
            let data = reader.readAll()
            let contig = try MLMultiArray(shape: depth.shape, dataType: .float32)
            let ptr = UnsafeMutablePointer<Float>(OpaquePointer(contig.dataPointer))
            for i in 0..<data.count { ptr[i] = data[i] }
            inputForGPU = contig
        }

        guard let outBuf = device.makeBuffer(length: width * height * 4, options: .storageModeShared) else {
            throw DA3Error.inferenceError("Metal: failed to allocate RGBA output buffer")
        }

        let inBuf = try makeBuffer(for: inputForGPU)

        var params = VizParams(
            width: UInt32(width),
            height: UInt32(height),
            depthMin: depthMin,
            depthMax: depthMax,
            invert: invert ? 1 : 0,
            style: style == .da3 ? 1 : 0,
            colormap: {
                switch colormap {
                case .spectral: return 0
                case .turbo: return 1
                case .grayscale: return 2
                default: return 0
                }
            }(),
            _pad0: 0
        )

        let pso: MTLComputePipelineState
        switch inputForGPU.dataType {
        case .float16: pso = visualizeDepthRGBAF16
        case .float32: pso = visualizeDepthRGBAF32
        default:
            throw DA3Error.invalidInput("Metal visualizeDepth only supports float16/float32 inputs (got \(inputForGPU.dataType))")
        }

        let cmd = try makeCommandBuffer()
        guard let enc = cmd.makeComputeCommandEncoder() else {
            throw DA3Error.inferenceError("Metal: failed to create compute encoder")
        }
        enc.label = "da3.visualizeDepthRGBA"
        enc.setComputePipelineState(pso)
        enc.setBuffer(inBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBytes(&params, length: MemoryLayout<VizParams>.stride, index: 2)

        let grid = MTLSize(width: width, height: height, depth: 1)
        let tg = Self.threadsPerThreadgroup2D(pso: pso)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        // Copy bytes into a Data-backed CGImage.
        let rgba = Data(bytes: outBuf.contents(), count: width * height * 4)
        guard let provider = CGDataProvider(data: rgba as CFData) else {
            throw DA3Error.imageProcessingFailed("Failed to create CGDataProvider for depth visualization")
        }
        let cs = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        guard let img = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: cs,
            bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        ) else {
            throw DA3Error.imageProcessingFailed("Failed to create CGImage for depth visualization")
        }
        return img
    }

    // MARK: - Helpers

    private func makeCommandBuffer() throws -> MTLCommandBuffer {
        guard let cmd = queue.makeCommandBuffer() else {
            throw DA3Error.inferenceError("Metal: failed to create command buffer")
        }
        return cmd
    }

    private func makeBuffer(for array: MLMultiArray) throws -> MTLBuffer {
        let elementSize: Int
        switch array.dataType {
        case .float16: elementSize = 2
        case .float32: elementSize = 4
        default:
            throw DA3Error.invalidInput("Metal buffer only supports float16/float32 arrays (got \(array.dataType))")
        }
        let length = array.count * elementSize
        guard let buf = device.makeBuffer(bytesNoCopy: array.dataPointer, length: length, options: .storageModeShared, deallocator: nil) else {
            throw DA3Error.inferenceError("Metal: failed to create MTLBuffer (len=\(length))")
        }
        return buf
    }

    private static func threadsPerThreadgroup2D(pso: MTLComputePipelineState) -> MTLSize {
        // Prefer 16x16 when possible; clamp to the device limit.
        let w = min(16, pso.threadExecutionWidth)
        // Max threads per tg is typically 1024; pick a conservative square-ish size.
        let h = max(1, min(16, pso.maxTotalThreadsPerThreadgroup / max(1, w)))
        return MTLSize(width: w, height: h, depth: 1)
    }

    // MARK: - Metal Shaders

    // Keep the shader self-contained to make SwiftPM + iOS packaging simple.
    private static let metalSource: String = """
    #include <metal_stdlib>
    using namespace metal;

    struct ResizeCropParams {
        uint inW;
        uint inH;
        uint outW;
        uint outH;
        uint channels;
        uint startX;
        uint startY;
        uint cropW;
        uint cropH;
    };

    inline float sampleCHW_f16(const device half* input, uint c, uint x, uint y, const constant ResizeCropParams& p) {
        return float(input[c * p.inH * p.inW + y * p.inW + x]);
    }

    inline float sampleCHW_f32(const device float* input, uint c, uint x, uint y, const constant ResizeCropParams& p) {
        return input[c * p.inH * p.inW + y * p.inW + x];
    }

    kernel void resizeCropCHW_f16(
        const device half* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant ResizeCropParams& p [[buffer(2)]],
        uint3 gid [[thread_position_in_grid]]
    ) {
        uint x = gid.x;
        uint y = gid.y;
        uint c = gid.z;
        if (x >= p.outW || y >= p.outH || c >= p.channels) { return; }

        float scaleX = float(p.cropW) / float(p.outW);
        float scaleY = float(p.cropH) / float(p.outH);

        float gx = (float(x) + 0.5f) * scaleX - 0.5f;
        float gy = (float(y) + 0.5f) * scaleY - 0.5f;

        int x0 = int(floor(gx));
        int y0 = int(floor(gy));
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        x0 = clamp(x0, 0, int(p.cropW) - 1);
        x1 = clamp(x1, 0, int(p.cropW) - 1);
        y0 = clamp(y0, 0, int(p.cropH) - 1);
        y1 = clamp(y1, 0, int(p.cropH) - 1);

        float wx = clamp(gx - float(x0), 0.0f, 1.0f);
        float wy = clamp(gy - float(y0), 0.0f, 1.0f);

        uint ix0 = p.startX + uint(x0);
        uint ix1 = p.startX + uint(x1);
        uint iy0 = p.startY + uint(y0);
        uint iy1 = p.startY + uint(y1);

        float v00 = sampleCHW_f16(input, c, ix0, iy0, p);
        float v01 = sampleCHW_f16(input, c, ix1, iy0, p);
        float v10 = sampleCHW_f16(input, c, ix0, iy1, p);
        float v11 = sampleCHW_f16(input, c, ix1, iy1, p);

        float top = mix(v00, v01, wx);
        float bot = mix(v10, v11, wx);
        float v = mix(top, bot, wy);

        output[c * p.outH * p.outW + y * p.outW + x] = v;
    }

    kernel void resizeCropCHW_f32(
        const device float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant ResizeCropParams& p [[buffer(2)]],
        uint3 gid [[thread_position_in_grid]]
    ) {
        uint x = gid.x;
        uint y = gid.y;
        uint c = gid.z;
        if (x >= p.outW || y >= p.outH || c >= p.channels) { return; }

        float scaleX = float(p.cropW) / float(p.outW);
        float scaleY = float(p.cropH) / float(p.outH);

        float gx = (float(x) + 0.5f) * scaleX - 0.5f;
        float gy = (float(y) + 0.5f) * scaleY - 0.5f;

        int x0 = int(floor(gx));
        int y0 = int(floor(gy));
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        x0 = clamp(x0, 0, int(p.cropW) - 1);
        x1 = clamp(x1, 0, int(p.cropW) - 1);
        y0 = clamp(y0, 0, int(p.cropH) - 1);
        y1 = clamp(y1, 0, int(p.cropH) - 1);

        float wx = clamp(gx - float(x0), 0.0f, 1.0f);
        float wy = clamp(gy - float(y0), 0.0f, 1.0f);

        uint ix0 = p.startX + uint(x0);
        uint ix1 = p.startX + uint(x1);
        uint iy0 = p.startY + uint(y0);
        uint iy1 = p.startY + uint(y1);

        float v00 = sampleCHW_f32(input, c, ix0, iy0, p);
        float v01 = sampleCHW_f32(input, c, ix1, iy0, p);
        float v10 = sampleCHW_f32(input, c, ix0, iy1, p);
        float v11 = sampleCHW_f32(input, c, ix1, iy1, p);

        float top = mix(v00, v01, wx);
        float bot = mix(v10, v11, wx);
        float v = mix(top, bot, wy);

        output[c * p.outH * p.outW + y * p.outW + x] = v;
    }

    struct BlendParams {
        uint outW;
        uint outH;
        uint tileW;
        uint tileH;
        uint atX;
        uint atY;
        uint overlap;
    };

    inline float rampWeight(uint x, uint y, const constant BlendParams& p) {
        bool atLeftEdge = (p.atX == 0);
        bool atRightEdge = (p.atX + p.tileW >= p.outW);
        bool atTopEdge = (p.atY == 0);
        bool atBottomEdge = (p.atY + p.tileH >= p.outH);

        float distLeft = float(x);
        float distRight = float(int(p.tileW) - 1 - int(x));
        float distTop = float(y);
        float distBottom = float(int(p.tileH) - 1 - int(y));

        float big = float(p.overlap) + 1.0f;
        if (atLeftEdge) distLeft = big;
        if (atRightEdge) distRight = big;
        if (atTopEdge) distTop = big;
        if (atBottomEdge) distBottom = big;

        float minDist = min(min(distLeft, distRight), min(distTop, distBottom));
        if (p.overlap == 0) return 1.0f;
        return min(1.0f, minDist / float(p.overlap));
    }

    kernel void blendTileDepthConf(
        const device float* tileDepth [[buffer(0)]],
        const device float* tileConf [[buffer(1)]],
        device float* outDepth [[buffer(2)]],
        device float* outConf [[buffer(3)]],
        device float* outWts [[buffer(4)]],
        constant BlendParams& p [[buffer(5)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint x = gid.x;
        uint y = gid.y;
        if (x >= p.tileW || y >= p.tileH) { return; }

        uint outX = p.atX + x;
        uint outY = p.atY + y;
        if (outX >= p.outW || outY >= p.outH) { return; }

        float w = rampWeight(x, y, p);
        uint tileIdx = y * p.tileW + x;
        uint outIdx = outY * p.outW + outX;

        outDepth[outIdx] += tileDepth[tileIdx] * w;
        outConf[outIdx] += tileConf[tileIdx] * w;
        outWts[outIdx] += w;
    }

    struct BlendRaysParams {
        uint outW;
        uint outH;
        uint tileW;
        uint tileH;
        uint atX;
        uint atY;
        uint overlap;
        uint channels;
    };

    kernel void blendTileRays(
        const device float* tileRays [[buffer(0)]],
        const device float* tileConf [[buffer(1)]],
        device float* outRays [[buffer(2)]],
        device float* outConf [[buffer(3)]],
        constant BlendRaysParams& p [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint x = gid.x;
        uint y = gid.y;
        if (x >= p.tileW || y >= p.tileH) { return; }

        uint outX = p.atX + x;
        uint outY = p.atY + y;
        if (outX >= p.outW || outY >= p.outH) { return; }

        BlendParams bp;
        bp.outW = p.outW;
        bp.outH = p.outH;
        bp.tileW = p.tileW;
        bp.tileH = p.tileH;
        bp.atX = p.atX;
        bp.atY = p.atY;
        bp.overlap = p.overlap;
        float w = rampWeight(x, y, bp);

        uint tileIdx2D = y * p.tileW + x;
        uint outIdx2D = outY * p.outW + outX;

        // Rays: CHW
        uint tileHW = p.tileH * p.tileW;
        uint outHW = p.outH * p.outW;
        for (uint c = 0; c < p.channels; c++) {
            uint tileIdx = c * tileHW + tileIdx2D;
            uint outIdx = c * outHW + outIdx2D;
            outRays[outIdx] += tileRays[tileIdx] * w;
        }
        outConf[outIdx2D] += tileConf[tileIdx2D] * w;
    }

    struct NormParams {
        uint width;
        uint height;
    };

    kernel void normalize1CHW(
        device float* values [[buffer(0)]],
        const device float* weights [[buffer(1)]],
        constant NormParams& p [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint x = gid.x;
        uint y = gid.y;
        if (x >= p.width || y >= p.height) { return; }
        uint idx = y * p.width + x;
        float w = weights[idx];
        values[idx] = (w > 0.0f) ? (values[idx] / w) : 0.0f;
    }

    struct NormCHWParams {
        uint width;
        uint height;
        uint channels;
    };

    kernel void normalizeCHW(
        device float* values [[buffer(0)]],
        const device float* weights [[buffer(1)]],
        constant NormCHWParams& p [[buffer(2)]],
        uint3 gid [[thread_position_in_grid]]
    ) {
        uint x = gid.x;
        uint y = gid.y;
        uint c = gid.z;
        if (x >= p.width || y >= p.height || c >= p.channels) { return; }

        uint idx2D = y * p.width + x;
        float w = weights[idx2D];
        uint idx = c * p.height * p.width + idx2D;
        values[idx] = (w > 0.0f) ? (values[idx] / w) : 0.0f;
    }

    struct ActivateParams {
        uint width;
        uint height;
        float clampMin;
        float clampMax;
        uint mode; // 1 = expp1, 2 = softplus1
    };

    inline float softplus_stable(float x) {
        // softplus(x) = max(x,0) + log(1 + exp(-abs(x)))
        float ax = fabs(x);
        return max(x, 0.0f) + log(1.0f + exp(-ax));
    }

    kernel void activateConfidence1CHW_f16(
        const device half* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant ActivateParams& p [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint x = gid.x;
        uint y = gid.y;
        if (x >= p.width || y >= p.height) { return; }
        uint idx = y * p.width + x;
        float v = float(input[idx]);
        if (!isfinite(v)) { output[idx] = 0.0f; return; }
        float lo = min(p.clampMin, p.clampMax);
        float hi = max(p.clampMin, p.clampMax);
        float z = clamp(v, lo, hi);
        if (p.mode == 1) {
            output[idx] = exp(z) + 1.0f;
        } else {
            output[idx] = softplus_stable(z) + 1.0f;
        }
    }

    kernel void activateConfidence1CHW_f32(
        const device float* input [[buffer(0)]],
        device float* output [[buffer(1)]],
        constant ActivateParams& p [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint x = gid.x;
        uint y = gid.y;
        if (x >= p.width || y >= p.height) { return; }
        uint idx = y * p.width + x;
        float v = input[idx];
        if (!isfinite(v)) { output[idx] = 0.0f; return; }
        float lo = min(p.clampMin, p.clampMax);
        float hi = max(p.clampMin, p.clampMax);
        float z = clamp(v, lo, hi);
        if (p.mode == 1) {
            output[idx] = exp(z) + 1.0f;
        } else {
            output[idx] = softplus_stable(z) + 1.0f;
        }
    }

    struct UnprojectParams {
        uint width;
        uint height;
        float fx;
        float fy;
        float cx;
        float cy;
    };

    kernel void unprojectDepthRay_f16(
        const device half* depth [[buffer(0)]],
        device float* outXYZ [[buffer(1)]],
        constant UnprojectParams& p [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint x = gid.x;
        uint y = gid.y;
        if (x >= p.width || y >= p.height) { return; }
        uint idx2D = y * p.width + x;
        float d = float(depth[idx2D]);
        if (!isfinite(d) || d <= 0.0f) {
            outXYZ[idx2D] = 0.0f;
            outXYZ[p.width * p.height + idx2D] = 0.0f;
            outXYZ[2 * p.width * p.height + idx2D] = 0.0f;
            return;
        }
        float u = (float(x) + 0.5f - p.cx) / max(1e-6f, p.fx);
        float v = (float(y) + 0.5f - p.cy) / max(1e-6f, p.fy);
        float3 dir = float3(u, v, 1.0f);
        float len = length(dir);
        dir = (len > 0.0f) ? (dir / len) : float3(0.0f, 0.0f, 1.0f);
        float3 pt = dir * d;
        uint hw = p.width * p.height;
        outXYZ[0 * hw + idx2D] = pt.x;
        outXYZ[1 * hw + idx2D] = pt.y;
        outXYZ[2 * hw + idx2D] = pt.z;
    }

    kernel void unprojectDepthRay_f32(
        const device float* depth [[buffer(0)]],
        device float* outXYZ [[buffer(1)]],
        constant UnprojectParams& p [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint x = gid.x;
        uint y = gid.y;
        if (x >= p.width || y >= p.height) { return; }
        uint idx2D = y * p.width + x;
        float d = depth[idx2D];
        if (!isfinite(d) || d <= 0.0f) {
            outXYZ[idx2D] = 0.0f;
            outXYZ[p.width * p.height + idx2D] = 0.0f;
            outXYZ[2 * p.width * p.height + idx2D] = 0.0f;
            return;
        }
        float u = (float(x) + 0.5f - p.cx) / max(1e-6f, p.fx);
        float v = (float(y) + 0.5f - p.cy) / max(1e-6f, p.fy);
        float3 dir = float3(u, v, 1.0f);
        float len = length(dir);
        dir = (len > 0.0f) ? (dir / len) : float3(0.0f, 0.0f, 1.0f);
        float3 pt = dir * d;
        uint hw = p.width * p.height;
        outXYZ[0 * hw + idx2D] = pt.x;
        outXYZ[1 * hw + idx2D] = pt.y;
        outXYZ[2 * hw + idx2D] = pt.z;
    }

    struct UnprojectGSParams {
        uint width;
        uint height;
        float fx;
        float fy;
        float cx;
        float cy;
        float offsetDepthScale;
        uint flags; // bit0=offset_xy, bit1=offset_depth
        float4x4 c2w;
    };

    kernel void unprojectGSDepthToWorldXYZ_f16(
        const device half* depth [[buffer(0)]],
        const device half* gsParams [[buffer(1)]],
        device float* outXYZ [[buffer(2)]],
        constant UnprojectGSParams& p [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint x = gid.x;
        uint y = gid.y;
        if (x >= p.width || y >= p.height) { return; }
        uint idx2D = y * p.width + x;
        uint hw = p.width * p.height;

        float d0 = float(depth[idx2D]);
        if (!isfinite(d0) || d0 <= 0.0f) {
            outXYZ[0 * hw + idx2D] = 0.0f;
            outXYZ[1 * hw + idx2D] = 0.0f;
            outXYZ[2 * hw + idx2D] = 0.0f;
            return;
        }

        float offX = 0.0f;
        float offY = 0.0f;
        if ((p.flags & 1u) != 0u) {
            offX = float(gsParams[0u * hw + idx2D]);
            offY = float(gsParams[1u * hw + idx2D]);
        }

        float offD = 0.0f;
        if ((p.flags & 2u) != 0u) {
            offD = float(gsParams[36u * hw + idx2D]);
        }

        float rayDepth = d0 + offD * p.offsetDepthScale;
        if (!isfinite(rayDepth) || rayDepth <= 0.0f) {
            outXYZ[0 * hw + idx2D] = 0.0f;
            outXYZ[1 * hw + idx2D] = 0.0f;
            outXYZ[2 * hw + idx2D] = 0.0f;
            return;
        }

        float u = (float(x) + 0.5f + offX - p.cx) / max(1e-6f, p.fx);
        float v = (float(y) + 0.5f + offY - p.cy) / max(1e-6f, p.fy);
        float3 dir = float3(u, v, 1.0f);
        float len = length(dir);
        dir = (len > 0.0f) ? (dir / len) : float3(0.0f, 0.0f, 1.0f);
        float3 camPt = dir * rayDepth;

        float4 world4 = p.c2w * float4(camPt, 1.0f);
        outXYZ[0 * hw + idx2D] = world4.x;
        outXYZ[1 * hw + idx2D] = world4.y;
        outXYZ[2 * hw + idx2D] = world4.z;
    }

    kernel void unprojectGSDepthToWorldXYZ_f32(
        const device float* depth [[buffer(0)]],
        const device float* gsParams [[buffer(1)]],
        device float* outXYZ [[buffer(2)]],
        constant UnprojectGSParams& p [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint x = gid.x;
        uint y = gid.y;
        if (x >= p.width || y >= p.height) { return; }
        uint idx2D = y * p.width + x;
        uint hw = p.width * p.height;

        float d0 = depth[idx2D];
        if (!isfinite(d0) || d0 <= 0.0f) {
            outXYZ[0 * hw + idx2D] = 0.0f;
            outXYZ[1 * hw + idx2D] = 0.0f;
            outXYZ[2 * hw + idx2D] = 0.0f;
            return;
        }

        float offX = 0.0f;
        float offY = 0.0f;
        if ((p.flags & 1u) != 0u) {
            offX = gsParams[0u * hw + idx2D];
            offY = gsParams[1u * hw + idx2D];
        }

        float offD = 0.0f;
        if ((p.flags & 2u) != 0u) {
            offD = gsParams[36u * hw + idx2D];
        }

        float rayDepth = d0 + offD * p.offsetDepthScale;
        if (!isfinite(rayDepth) || rayDepth <= 0.0f) {
            outXYZ[0 * hw + idx2D] = 0.0f;
            outXYZ[1 * hw + idx2D] = 0.0f;
            outXYZ[2 * hw + idx2D] = 0.0f;
            return;
        }

        float u = (float(x) + 0.5f + offX - p.cx) / max(1e-6f, p.fx);
        float v = (float(y) + 0.5f + offY - p.cy) / max(1e-6f, p.fy);
        float3 dir = float3(u, v, 1.0f);
        float len = length(dir);
        dir = (len > 0.0f) ? (dir / len) : float3(0.0f, 0.0f, 1.0f);
        float3 camPt = dir * rayDepth;

        float4 world4 = p.c2w * float4(camPt, 1.0f);
        outXYZ[0 * hw + idx2D] = world4.x;
        outXYZ[1 * hw + idx2D] = world4.y;
        outXYZ[2 * hw + idx2D] = world4.z;
    }

    struct VizParams {
        uint width;
        uint height;
        float depthMin;
        float depthMax;
        uint invert;
        uint style;    // 0 = raw depth, 1 = DA3 inverse-depth
        uint colormap; // 0 = spectral, 1 = turbo, 2 = grayscale
        uint _pad0;
    };

    inline float3 spectralColor(float t) {
        // 11 control points (ColorBrewer Spectral). Low=red, high=blue.
        const float3 stops[11] = {
            float3(158.0/255.0,   1.0/255.0,  66.0/255.0),
            float3(213.0/255.0,  62.0/255.0,  79.0/255.0),
            float3(244.0/255.0, 109.0/255.0,  67.0/255.0),
            float3(253.0/255.0, 174.0/255.0,  97.0/255.0),
            float3(254.0/255.0, 224.0/255.0, 139.0/255.0),
            float3(255.0/255.0, 255.0/255.0, 191.0/255.0),
            float3(230.0/255.0, 245.0/255.0, 152.0/255.0),
            float3(171.0/255.0, 221.0/255.0, 164.0/255.0),
            float3(102.0/255.0, 194.0/255.0, 165.0/255.0),
            float3( 50.0/255.0, 136.0/255.0, 189.0/255.0),
            float3( 94.0/255.0,  79.0/255.0, 162.0/255.0),
        };
        float s = clamp(t, 0.0f, 1.0f) * 10.0f;
        int i0 = clamp(int(floor(s)), 0, 9);
        int i1 = i0 + 1;
        float f = s - float(i0);
        return mix(stops[i0], stops[i1], f);
    }

    inline float3 turboColor(float t) {
        float x = clamp(t, 0.0f, 1.0f);
        float r = clamp(0.13572138 + x * (4.6153926 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943)))), 0.0f, 1.0f);
        float g = clamp(0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604)))), 0.0f, 1.0f);
        float b = clamp(0.1066733 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973)))), 0.0f, 1.0f);
        return float3(r, g, b);
    }

    inline uchar4 packRGBA(float3 rgb) {
        uchar r = (uchar)clamp(rgb.x * 255.0f, 0.0f, 255.0f);
        uchar g = (uchar)clamp(rgb.y * 255.0f, 0.0f, 255.0f);
        uchar b = (uchar)clamp(rgb.z * 255.0f, 0.0f, 255.0f);
        return uchar4(r, g, b, 255);
    }

    inline float depthToNorm(float d, const constant VizParams& p) {
        float range = p.depthMax - p.depthMin;
        if (range == 0.0f) range = 1.0f;
        if (p.style == 1) {
            float inv = (d > 0.0f) ? (1.0f / d) : 0.0f;
            float t = clamp((inv - p.depthMin) / range, 0.0f, 1.0f);
            // DA3 inverts after normalization.
            return 1.0f - t;
        }
        return clamp((d - p.depthMin) / range, 0.0f, 1.0f);
    }

    inline float applyInvert(float t, const constant VizParams& p) {
        return (p.invert != 0) ? (1.0f - t) : t;
    }

    kernel void visualizeDepthRGBA_f16(
        const device half* depth [[buffer(0)]],
        device uchar4* outRGBA [[buffer(1)]],
        constant VizParams& p [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint x = gid.x;
        uint y = gid.y;
        if (x >= p.width || y >= p.height) { return; }
        uint idx = y * p.width + x;
        float d = float(depth[idx]);
        if (!isfinite(d)) d = 0.0f;
        float t = applyInvert(depthToNorm(d, p), p);
        float3 rgb;
        if (p.colormap == 1) rgb = turboColor(t);
        else if (p.colormap == 2) rgb = float3(t, t, t);
        else rgb = spectralColor(t);
        outRGBA[idx] = packRGBA(rgb);
    }

    kernel void visualizeDepthRGBA_f32(
        const device float* depth [[buffer(0)]],
        device uchar4* outRGBA [[buffer(1)]],
        constant VizParams& p [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint x = gid.x;
        uint y = gid.y;
        if (x >= p.width || y >= p.height) { return; }
        uint idx = y * p.width + x;
        float d = depth[idx];
        if (!isfinite(d)) d = 0.0f;
        float t = applyInvert(depthToNorm(d, p), p);
        float3 rgb;
        if (p.colormap == 1) rgb = turboColor(t);
        else if (p.colormap == 2) rgb = float3(t, t, t);
        else rgb = spectralColor(t);
        outRGBA[idx] = packRGBA(rgb);
    }
    """
}

// MARK: - Metal param structs (must match shader layout)

@available(macOS 14.0, iOS 17.0, *)
private struct ResizeCropParams {
    var inW: UInt32
    var inH: UInt32
    var outW: UInt32
    var outH: UInt32
    var channels: UInt32
    var startX: UInt32
    var startY: UInt32
    var cropW: UInt32
    var cropH: UInt32
}

@available(macOS 14.0, iOS 17.0, *)
private struct BlendParams {
    var outW: UInt32
    var outH: UInt32
    var tileW: UInt32
    var tileH: UInt32
    var atX: UInt32
    var atY: UInt32
    var overlap: UInt32
}

@available(macOS 14.0, iOS 17.0, *)
private struct BlendRaysParams {
    var outW: UInt32
    var outH: UInt32
    var tileW: UInt32
    var tileH: UInt32
    var atX: UInt32
    var atY: UInt32
    var overlap: UInt32
    var channels: UInt32
}

@available(macOS 14.0, iOS 17.0, *)
private struct NormParams {
    var width: UInt32
    var height: UInt32
}

@available(macOS 14.0, iOS 17.0, *)
private struct NormCHWParams {
    var width: UInt32
    var height: UInt32
    var channels: UInt32
}

@available(macOS 14.0, iOS 17.0, *)
private struct ActivateParams {
    var width: UInt32
    var height: UInt32
    var clampMin: Float
    var clampMax: Float
    var mode: UInt32
}

@available(macOS 14.0, iOS 17.0, *)
private struct UnprojectParams {
    var width: UInt32
    var height: UInt32
    var fx: Float
    var fy: Float
    var cx: Float
    var cy: Float
}

@available(macOS 14.0, iOS 17.0, *)
private struct UnprojectGSParams {
    var width: UInt32
    var height: UInt32
    var fx: Float
    var fy: Float
    var cx: Float
    var cy: Float
    var offsetDepthScale: Float
    var flags: UInt32
    var c2w: simd_float4x4
}

@available(macOS 14.0, iOS 17.0, *)
private struct VizParams {
    var width: UInt32
    var height: UInt32
    var depthMin: Float
    var depthMax: Float
    var invert: UInt32
    var style: UInt32
    var colormap: UInt32
    var _pad0: UInt32
}
