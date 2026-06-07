import Foundation
import CoreML
import simd

/// Feed-forward DA3 Gaussian splat generation using the pre-trained GS head.
///
/// This converts:
/// - multi-view camera pose (optional, for world-space fusion)
/// - DualDPT depth (for 3D position)
/// - GS head parameters (for offsets, scale, rotation, color SH, opacity)
/// into a `DA3GaussianCloud` that can be written as a standard 3DGS PLY.
@available(macOS 14.0, iOS 17.0, *)
public final class DA3GSHeadTo3DGS {

    // MARK: - Config

    public struct Config {
        /// Subsample factor in pixel space (1 = all pixels, 2 = every other pixel, ...)
        public var subsample: Int = 4
        /// Minimum GS confidence/opacity threshold (channel 37).
        ///
        /// Note: Depending on the export, channel 37 may be a probability (0..1) or a logit.
        /// This converter detects the encoding once per conversion (based on global min/max):
        /// - If all values are in [0, 1], treats them as probabilities.
        /// - Otherwise, treats them as logits and applies `sigmoid` for thresholding.
        public var minConfidence: Float = 0.0

        /// Whether to apply the GS head `offset_depth` channel (36).
        /// When false, uses the base DualDPT depth only.
        public var applyOffsetDepth: Bool = true
        /// Scale factor applied to `offset_depth` before adding to depth.
        ///
        /// DA3's reference implementation uses a scale of 1.0; this knob exists to help debug
        /// mismatched exports or unit conventions.
        public var offsetDepthScale: Float = 1.0
        /// Minimum allowed depth (after applying offset_depth)
        public var minDepth: Float = 1e-3
        /// Maximum allowed depth (after applying offset_depth)
        public var maxDepth: Float = 1e6
        
        /// Enable DA3-style border pruning (removes noisy edge Gaussians)
        public var enableBorderPruning: Bool = true
        /// Border trim percentage (DA3 default: 8/256 ≈ 0.03125)
        public var borderTrimPercent: Float = 0.03125
        
        /// Enable DA3-style depth percentile pruning (removes far outliers)
        public var enableDepthPercentilePruning: Bool = true
        /// Keep only pixels with base depth <= this percentile (DA3 export default: 0.9)
        public var depthPercentileThreshold: Float = 0.9
        
        /// GaussianAdapter scale clamp (DA3 defaults)
        public var gaussianScaleMin: Float = 1e-5
        public var gaussianScaleMax: Float = 30.0
        /// GaussianAdapter scale multiplier constant (DA3 default: 0.1)
        public var gaussianScaleMultiplier: Float = 0.1

        /// Use Metal to unproject depth(+offsets) to world-space XYZ.
        ///
        /// This moves the most expensive, data-parallel portion of the conversion
        /// (ray construction + normalization + optional c2w transform) onto GPU in float32.
        /// Pruning/thresholding still happens on CPU for flexibility.
        ///
        /// If Metal is unavailable, the converter automatically falls back to CPU.
        public var useMetalUnprojection: Bool = false

        public init() {}
    }

    public var config: Config

    public init(config: Config = Config()) {
        self.config = config
    }

    // MARK: - Conversion

    public func convert(
        gsParams: GSHeadCoreML.GSParams,
        depth: MLMultiArray,
        intrinsics: DA3DepthTo3DGS.CameraIntrinsics,
        extrinsics: DA3DepthTo3DGS.CameraExtrinsics? = nil
    ) throws -> DA3GaussianCloud {
        let gs = try MLMultiArrayFloatReader(gsParams.raw)
        let depthR = try MLMultiArrayFloatReader(depth)

        let (gsH, gsW) = (gsParams.height, gsParams.width)

        // Depth shape sanity: accept [1, 1, H, W] or [1, H, W] or [H, W].
        let depthShape = depth.shape.map { $0.intValue }
        let depthH = depthShape.count >= 2 ? depthShape[depthShape.count - 2] : 0
        let depthW = depthShape.count >= 1 ? depthShape[depthShape.count - 1] : 0
        guard depthH == gsH, depthW == gsW else {
            throw DA3Error.invalidShape("Depth shape \(depthShape) does not match gs_params (\(gsH)x\(gsW))")
        }

        func depthAt(y: Int, x: Int) -> Float {
            switch depthShape.count {
            case 4:
                // [B, C, H, W]
                return depthR.read(0, 0, y, x)
            case 3:
                // [C, H, W] or [B, H, W]
                if depthShape[0] == 1 {
                    return depthR.read(0, y, x)
                } else {
                    // Treat as [C,H,W] and read channel 0
                    return depthR.read(0, y, x)
                }
            case 2:
                return depthR.read(y, x)
            default:
                // Fallback to linear
                return depthR.readLinear(y * gsW + x)
            }
        }

        // DA3 export uses border trim (8/256 of H/W) and depth percentile pruning (0.9).
        let trimH = config.enableBorderPruning ? Int(config.borderTrimPercent * Float(gsH)) : 0
        let trimW = config.enableBorderPruning ? Int(config.borderTrimPercent * Float(gsW)) : 0
        let minY = trimH
        let maxY = gsH - trimH
        let minX = trimW
        let maxX = gsW - trimW
        
        // Compute depth percentile threshold based on the *base* depth map (before offset_depth),
        // matching DA3's save_gaussian_ply(ctx_depth=pred_depth, prune_by_depth_percent=...).
        var depthPercentileValue: Float = .greatestFiniteMagnitude
        if config.enableDepthPercentilePruning && config.depthPercentileThreshold < 1.0 {
            var baseDepths: [Float] = []
            baseDepths.reserveCapacity(gsH * gsW)
            for y in 0..<gsH {
                for x in 0..<gsW {
                    baseDepths.append(depthAt(y: y, x: x))
                }
            }
            depthPercentileValue = computePercentile(baseDepths, percentile: config.depthPercentileThreshold)
        }

        // Determine whether the confidence channel is probability or logit, once per conversion.
        // Per-element heuristics (e.g. "if 0..1 then prob") are incorrect because logits commonly
        // fall inside [0,1] as well.
        var confMin: Float = .greatestFiniteMagnitude
        var confMax: Float = -.greatestFiniteMagnitude
        for y in 0..<gsH {
            for x in 0..<gsW {
                let v = gs.read(0, 37, y, x)
                guard v.isFinite else { continue }
                if v < confMin { confMin = v }
                if v > confMax { confMax = v }
            }
        }
        let confidenceIsProbability: Bool = (confMin != .greatestFiniteMagnitude) && (confMin >= 0) && (confMax <= 1)
        
        let cloud = DA3GaussianCloud()
        let effectiveW = max(0, maxX - minX)
        let effectiveH = max(0, maxY - minY)
        cloud.reserve((effectiveW / max(1, config.subsample)) * (effectiveH / max(1, config.subsample)))

        let fx = intrinsics.fx
        let fy = intrinsics.fy
        let cx = intrinsics.cx
        let cy = intrinsics.cy

        // Optional GPU unprojection for world-space XYZ (still applies pruning/thresholding on CPU).
        var worldXYZReader: MLMultiArrayFloatReader? = nil
        if config.useMetalUnprojection,
           let mp = DA3MetalPostProcessor.shared() {
            do {
                let c2w = extrinsics?.c2w ?? matrix_identity_float4x4
                let xyz = try mp.unprojectGSDepthToWorldXYZ(
                    depth: depth,
                    gsParams: gsParams.raw,
                    width: gsW,
                    height: gsH,
                    fx: fx,
                    fy: fy,
                    cx: cx,
                    cy: cy,
                    applyOffsetXY: true,
                    applyOffsetDepth: config.applyOffsetDepth,
                    offsetDepthScale: config.offsetDepthScale,
                    c2w: c2w
                )
                worldXYZReader = try MLMultiArrayFloatReader(xyz)
            } catch {
                worldXYZReader = nil
            }
        }
        
        // DA3 GaussianAdapter scale multiplier: 0.1 * (1/fx + 1/fy) (fx/fy in pixels).
        let scaleMultiplier = config.gaussianScaleMultiplier * ((1.0 / fx) + (1.0 / fy))
        let scaleMin = config.gaussianScaleMin
        let scaleMax = config.gaussianScaleMax

        for y in stride(from: minY, to: maxY, by: max(1, config.subsample)) {
            for x in stride(from: minX, to: maxX, by: max(1, config.subsample)) {
                // Channel 37 is the GS head confidence.
                // In DA3's Python export pipeline, the saved PLY expects `opacity` in logit space
                // (via `inverse_sigmoid`).
                let confRaw = gs.read(0, 37, y, x)
                guard confRaw.isFinite else { continue }
                let confProb: Float = confidenceIsProbability ? confRaw : sigmoid(confRaw)
                guard confProb.isFinite, confProb >= config.minConfidence else { continue }
                let opacityLogit: Float = confidenceIsProbability ? inverseSigmoid(confProb) : confRaw
                guard opacityLogit.isFinite else { continue }

                let baseDepth = depthAt(y: y, x: x)
                guard baseDepth.isFinite, baseDepth > 0 else { continue }
                guard baseDepth <= depthPercentileValue else { continue }
                
                let offsetDepth = config.applyOffsetDepth ? gs.read(0, 36, y, x) : 0
                let rayDepth = baseDepth + offsetDepth * config.offsetDepthScale
                guard rayDepth.isFinite, rayDepth >= config.minDepth, rayDepth <= config.maxDepth else { continue }

                let offX = gs.read(0, 0, y, x)
                let offY = gs.read(0, 1, y, x)
                
                // DA3 uses normalized (0..1) pixel centers plus offset_xy*pixel_size.
                // In pixel units this is (x+0.5+offX, y+0.5+offY).
                let u = Float(x) + 0.5 + offX
                let v = Float(y) + 0.5 + offY

                let worldPoint: simd_float3
                if let xyzR = worldXYZReader {
                    // World-space XYZ already includes offset_xy/offset_depth and the c2w transform.
                    worldPoint = simd_float3(
                        xyzR.read(0, y, x),
                        xyzR.read(1, y, x),
                        xyzR.read(2, y, x)
                    )
                    guard worldPoint.x.isFinite, worldPoint.y.isFinite, worldPoint.z.isFinite else { continue }
                } else {
                    // CPU fallback: build a unit ray and multiply by ray depth (DA3 convention).
                    let dirRaw = simd_float3((u - cx) / fx, (v - cy) / fy, 1.0)
                    let dirLen = simd_length(dirRaw)
                    guard dirLen.isFinite, dirLen > 0 else { continue }
                    let camPoint = (dirRaw / dirLen) * rayDepth

                    if let extr = extrinsics {
                        worldPoint = extr.transformPoint(camPoint)
                    } else {
                        worldPoint = camPoint
                    }
                }

                // Rotation: DA3 assumes quaternion order **xyzw** (historical quirk), normalized,
                // then rotated into world space via c2w.
                let qx = gs.read(0, 5, y, x)
                let qy = gs.read(0, 6, y, x)
                let qz = gs.read(0, 7, y, x)
                let qw = gs.read(0, 8, y, x)
                var camQuat = simd_quatf(ix: qx, iy: qy, iz: qz, r: qw)
                camQuat = camQuat.normalized
                if camQuat.real < 0 { camQuat = simd_quatf(ix: -camQuat.imag.x, iy: -camQuat.imag.y, iz: -camQuat.imag.z, r: -camQuat.real) }

                let worldQuat: simd_quatf = {
                    if let extr = extrinsics {
                        return extr.transformQuaternion(camQuat)
                    }
                    return camQuat
                }()

                // Scales: DA3 GaussianAdapter maps raw scales via sigmoid into [scaleMin, scaleMax],
                // then multiplies by rayDepth and a resolution/intrinsics-dependent multiplier.
                let rawS0 = gs.read(0, 2, y, x)
                let rawS1 = gs.read(0, 3, y, x)
                let rawS2 = gs.read(0, 4, y, x)
                let s0 = scaleMin + (scaleMax - scaleMin) * sigmoid(rawS0)
                let s1 = scaleMin + (scaleMax - scaleMin) * sigmoid(rawS1)
                let s2 = scaleMin + (scaleMax - scaleMin) * sigmoid(rawS2)
                
                let gsScale0 = s0 * rayDepth * scaleMultiplier
                let gsScale1 = s1 * rayDepth * scaleMultiplier
                let gsScale2 = s2 * rayDepth * scaleMultiplier

                // SH DC coefficients (assume [R(9), G(9), B(9)] ordering)
                let shDC0 = gs.read(0, 9, y, x)
                let shDC1 = gs.read(0, 18, y, x)
                let shDC2 = gs.read(0, 27, y, x)

                var splat = DA3GaussianSplat()
                splat.x = worldPoint.x
                splat.y = worldPoint.y
                splat.z = worldPoint.z
                splat.shDC0 = shDC0
                splat.shDC1 = shDC1
                splat.shDC2 = shDC2
                splat.opacityLogit = opacityLogit
                splat.scaleLog0 = log(max(1e-9, gsScale0))
                splat.scaleLog1 = log(max(1e-9, gsScale1))
                splat.scaleLog2 = log(max(1e-9, gsScale2))
                splat.rotW = worldQuat.real
                splat.rotX = worldQuat.imag.x
                splat.rotY = worldQuat.imag.y
                splat.rotZ = worldQuat.imag.z
                splat.normalizeRotation()

                cloud.add(splat)
            }
        }

        return cloud
    }
}

// Uses `MLMultiArrayFloatReader` from `MLMultiArrayFloatReader.swift`.

@available(macOS 14.0, iOS 17.0, *)
private func sigmoid(_ x: Float) -> Float {
    1.0 / (1.0 + exp(-x))
}

@available(macOS 14.0, iOS 17.0, *)
private func inverseSigmoid(_ p: Float, eps: Float = 1e-6) -> Float {
    let x = min(1.0 - eps, max(eps, p))
    return log(x / (1.0 - x))
}

@available(macOS 14.0, iOS 17.0, *)
private func computePercentile(_ values: [Float], percentile: Float) -> Float {
    guard !values.isEmpty else { return 0 }
    guard percentile > 0 && percentile < 1 else {
        return percentile <= 0 ? (values.min() ?? 0) : (values.max() ?? 0)
    }
    
    let validValues = values.filter { $0.isFinite && $0 > 0 }
    guard !validValues.isEmpty else { return 0 }
    
    let sorted = validValues.sorted()
    let n = sorted.count
    
    let index = percentile * Float(n - 1)
    let lower = Int(index)
    let upper = min(lower + 1, n - 1)
    let frac = index - Float(lower)
    return sorted[lower] * (1 - frac) + sorted[upper] * frac
}
