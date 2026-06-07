import Foundation
import CoreGraphics
import simd

/// Converts DA3 depth maps to 3D Gaussian splats.
///
/// This is DA3CoreML's own 3DGS implementation (self-contained; no external renderer required).
/// Pipeline: Depth Map → Point Cloud → Gaussian Splats → PLY file
///
/// Supports two modes:
/// 1. **View-aligned mode**: Simple pinhole unprojection (default)
/// 2. **World-space mode**: Full camera-to-world transform using extrinsics (DA3-compatible)
@available(macOS 14.0, iOS 17.0, *)
public final class DA3DepthTo3DGS {
    
    // MARK: - Types
    
    /// Camera intrinsics for depth unprojection
    public struct CameraIntrinsics {
        /// Focal length X (pixels)
        public var fx: Float
        /// Focal length Y (pixels)
        public var fy: Float
        /// Principal point X (pixels)
        public var cx: Float
        /// Principal point Y (pixels)
        public var cy: Float
        /// Image width
        public var width: Int
        /// Image height
        public var height: Int
        
        public init(fx: Float, fy: Float, cx: Float, cy: Float, width: Int, height: Int) {
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy
            self.width = width
            self.height = height
        }
        
        /// Estimate intrinsics from image dimensions (assumes ~50° FOV)
        public static func estimate(width: Int, height: Int, fovDegrees: Float = 50) -> CameraIntrinsics {
            let fovRad = fovDegrees * .pi / 180
            let fx = Float(width) / (2 * tan(fovRad / 2))
            let fy = fx  // Square pixels
            return CameraIntrinsics(
                fx: fx, fy: fy,
                cx: Float(width) / 2,
                cy: Float(height) / 2,
                width: width, height: height
            )
        }
        
        /// Convert to 3x3 matrix form
        public var matrix: simd_float3x3 {
            simd_float3x3(rows: [
                simd_float3(fx, 0, cx),
                simd_float3(0, fy, cy),
                simd_float3(0, 0, 1)
            ])
        }
        
        /// Inverse matrix for unprojection
        public var inverseMatrix: simd_float3x3 {
            matrix.inverse
        }
    }
    
    /// Camera extrinsics (camera-to-world transform) for world-space 3DGS
    ///
    /// This is a 4x4 homogeneous transformation matrix that converts
    /// points from camera coordinates to world coordinates.
    ///
    /// Convention: **OpenCV-style camera coordinates** (X right, Y down, Z forward).
    /// This matches DA3's `utils/geometry.py` unprojection.
    public struct CameraExtrinsics {
        /// 4x4 camera-to-world transformation matrix
        public var c2w: simd_float4x4
        
        public init(c2w: simd_float4x4) {
            self.c2w = c2w
        }
        
        /// Create from rotation matrix and translation vector
        public init(rotation: simd_float3x3, translation: simd_float3) {
            self.c2w = simd_float4x4(
                simd_float4(rotation.columns.0, 0),
                simd_float4(rotation.columns.1, 0),
                simd_float4(rotation.columns.2, 0),
                simd_float4(translation, 1)
            )
        }
        
        /// Create from quaternion (WXYZ) and translation
        public init(quaternion: simd_quatf, translation: simd_float3) {
            let rotation = simd_float3x3(quaternion)
            self.init(rotation: rotation, translation: translation)
        }
        
        /// Identity transform (camera at origin, looking down -Z)
        public static var identity: CameraExtrinsics {
            CameraExtrinsics(c2w: matrix_identity_float4x4)
        }
        
        /// Create from flat array [r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2, 0, 0, 0, 1]
        public init(flatArray: [Float]) {
            precondition(flatArray.count >= 12, "Need at least 12 values for 3x4 matrix")
            let r00 = flatArray[0], r01 = flatArray[1], r02 = flatArray[2], t0 = flatArray[3]
            let r10 = flatArray[4], r11 = flatArray[5], r12 = flatArray[6], t1 = flatArray[7]
            let r20 = flatArray[8], r21 = flatArray[9], r22 = flatArray[10], t2 = flatArray[11]
            self.c2w = simd_float4x4(columns: (
                simd_float4(r00, r10, r20, 0),
                simd_float4(r01, r11, r21, 0),
                simd_float4(r02, r12, r22, 0),
                simd_float4(t0, t1, t2, 1)
            ))
        }
        
        /// Rotation component as 3x3 matrix
        public var rotation: simd_float3x3 {
            simd_float3x3(
                simd_float3(c2w.columns.0.x, c2w.columns.0.y, c2w.columns.0.z),
                simd_float3(c2w.columns.1.x, c2w.columns.1.y, c2w.columns.1.z),
                simd_float3(c2w.columns.2.x, c2w.columns.2.y, c2w.columns.2.z)
            )
        }
        
        /// Translation component
        public var translation: simd_float3 {
            simd_float3(c2w.columns.3.x, c2w.columns.3.y, c2w.columns.3.z)
        }
        
        /// Camera position in world space (same as translation)
        public var position: simd_float3 { translation }
        
        /// Transform a point from camera space to world space
        public func transformPoint(_ point: simd_float3) -> simd_float3 {
            let p4 = simd_float4(point, 1)
            let w4 = c2w * p4
            return simd_float3(w4.x, w4.y, w4.z)
        }
        
        /// Transform a direction from camera space to world space (no translation)
        public func transformDirection(_ direction: simd_float3) -> simd_float3 {
            rotation * direction
        }
        
        /// Transform a quaternion from camera space to world space
        /// Input: quaternion in WXYZ format representing rotation in camera space
        /// Output: quaternion in WXYZ format representing rotation in world space
        public func transformQuaternion(_ camQuat: simd_quatf) -> simd_quatf {
            let rotationQuat = simd_quatf(rotation)
            return rotationQuat * camQuat
        }
    }
    
    /// Configuration for 3DGS conversion
    public struct Config {
        /// Subsample factor (1 = all pixels, 2 = every 2nd pixel, etc.)
        public var subsample: Int = 1
        /// Minimum depth threshold (filter noise)
        public var minDepth: Float = 0.01
        /// Maximum depth threshold
        public var maxDepth: Float = 100.0
        /// Minimum confidence threshold.
        ///
        /// Note: Official DA3 checkpoints use `conf_activation="expp1"` (exp(x)+1), so values are
        /// positive weights (>= 1) rather than probabilities.
        public var minConfidence: Float = 0.3
        /// Default Gaussian scale (relative to point spacing)
        public var gaussianScale: Float = 0.01
        /// Default opacity for Gaussians
        public var defaultOpacity: Float = 0.9
        /// Normalize depth to [0, maxNormalizedDepth]
        public var normalizeDepth: Bool = true
        /// Max depth after normalization
        public var maxNormalizedDepth: Float = 10.0
        
        // MARK: - DA3-Quality Features
        
        /// Enable border pruning to remove noisy edge Gaussians (DA3 default: true)
        public var enableBorderPruning: Bool = true
        /// Border trim percentage (DA3 default: 8/256 ≈ 0.03125 = 3.125%)
        public var borderTrimPercent: Float = 0.03125
        
        /// Enable depth percentile filtering to prune far outliers (DA3 default: true)
        public var enableDepthPercentilePruning: Bool = true
        /// Depth percentile threshold - keep only depths <= this percentile (DA3 default: 0.9 = 90th percentile)
        public var depthPercentileThreshold: Float = 0.9
        
        /// Enable shift & scale normalization for consistent viewing (DA3 default: false)
        /// When enabled, centers cloud at median position and scales by 95th percentile
        public var shiftAndScale: Bool = false
        
        /// View sampling interval for multi-view inputs (1 = all views, 2 = every 2nd view, etc.)
        public var viewInterval: Int = 1
        
        public init() {}
        
        /// DA3-compatible preset with high quality settings
        public static var da3Quality: Config {
            var config = Config()
            config.enableBorderPruning = true
            config.borderTrimPercent = 0.03125  // 8/256
            config.enableDepthPercentilePruning = true
            config.depthPercentileThreshold = 0.9
            config.shiftAndScale = false
            config.minConfidence = 0.3
            config.gaussianScale = 0.01
            config.defaultOpacity = 0.9
            return config
        }
    }
    
    // MARK: - Properties
    
    public var config: Config
    
    // MARK: - Initialization
    
    public init(config: Config = Config()) {
        self.config = config
    }
    
    // MARK: - Conversion from DA3 Result
    
    /// Convert DA3 result directly to Gaussian cloud
    public func convert(
        result: DA3CoreML.Result,
        sourceImage: CGImage?,
        intrinsics: CameraIntrinsics? = nil
    ) throws -> DA3GaussianCloud {
        let width = result.originalSize.width
        let height = result.originalSize.height
        
        // Use provided intrinsics or estimate
        let camera = intrinsics ?? CameraIntrinsics.estimate(width: width, height: height)
        
        // Extract depth values
        var depthValues = [Float](repeating: 0, count: result.depth.count)
        for i in 0..<result.depth.count {
            depthValues[i] = result.depth[i].floatValue
        }
        
        // Extract confidence if available
        var confidenceValues: [Float]?
        confidenceValues = [Float](repeating: 1.0, count: result.depthConfidence.count)
        for i in 0..<result.depthConfidence.count {
            confidenceValues![i] = result.depthConfidence[i].floatValue
        }
        
        // Extract colors from source image if available
        var colors: [(r: Float, g: Float, b: Float)]?
        if let image = sourceImage {
            colors = try extractColors(from: image, width: width, height: height)
        }
        
        return convert(
            depth: depthValues,
            confidence: confidenceValues,
            colors: colors,
            width: width,
            height: height,
            intrinsics: camera
        )
    }
    
    /// Convert DA3 result to world-space Gaussian cloud using camera extrinsics
    ///
    /// This method produces world-space Gaussians compatible with DA3's Python output.
    /// The extrinsics define the camera-to-world transform.
    ///
    /// - Parameters:
    ///   - result: DA3 inference result
    ///   - sourceImage: Optional source image for colors
    ///   - intrinsics: Camera intrinsics (if nil, estimated from image size)
    ///   - extrinsics: Camera extrinsics (camera-to-world transform)
    /// - Returns: World-space Gaussian cloud
    public func convertWorldSpace(
        result: DA3CoreML.Result,
        sourceImage: CGImage?,
        intrinsics: CameraIntrinsics?,
        extrinsics: CameraExtrinsics
    ) throws -> DA3GaussianCloud {
        let width = result.originalSize.width
        let height = result.originalSize.height
        
        let camera = intrinsics ?? CameraIntrinsics.estimate(width: width, height: height)
        
        var depthValues = [Float](repeating: 0, count: result.depth.count)
        for i in 0..<result.depth.count {
            depthValues[i] = result.depth[i].floatValue
        }
        
        var confidenceValues: [Float]?
        confidenceValues = [Float](repeating: 1.0, count: result.depthConfidence.count)
        for i in 0..<result.depthConfidence.count {
            confidenceValues![i] = result.depthConfidence[i].floatValue
        }
        
        var colors: [(r: Float, g: Float, b: Float)]?
        if let image = sourceImage {
            colors = try extractColors(from: image, width: width, height: height)
        }
        
        return convertWorldSpace(
            depth: depthValues,
            confidence: confidenceValues,
            colors: colors,
            width: width,
            height: height,
            intrinsics: camera,
            extrinsics: extrinsics
        )
    }
    
    // MARK: - Conversion from Loaded Data
    
    /// Convert loaded DA3 data to Gaussian cloud
    public func convert(
        data: DA3OutputReader.LoadedData,
        sourceImage: CGImage? = nil
    ) throws -> DA3GaussianCloud {
        let intrinsics = CameraIntrinsics.estimate(width: data.width, height: data.height)
        
        var colors: [(r: Float, g: Float, b: Float)]?
        if let image = sourceImage {
            colors = try extractColors(from: image, width: data.width, height: data.height)
        }
        
        return convert(
            depth: data.depth,
            confidence: data.depthConfidence,
            colors: colors,
            width: data.width,
            height: data.height,
            intrinsics: intrinsics
        )
    }
    
    // MARK: - Core Conversion
    
    /// Convert depth array to Gaussian cloud
    public func convert(
        depth: [Float],
        confidence: [Float]?,
        colors: [(r: Float, g: Float, b: Float)]?,
        width: Int,
        height: Int,
        intrinsics: CameraIntrinsics
    ) -> DA3GaussianCloud {
        let cloud = DA3GaussianCloud()
        
        // Normalize depth if requested
        var normalizedDepth = depth
        if config.normalizeDepth {
            let minD = depth.min() ?? 0
            let maxD = depth.max() ?? 1
            let range = maxD - minD
            if range > 0 {
                normalizedDepth = depth.map { (($0 - minD) / range) * config.maxNormalizedDepth }
            }
        }
        
        // DA3-Quality: Compute depth percentile threshold
        var depthPercentileValue: Float = .greatestFiniteMagnitude
        if config.enableDepthPercentilePruning && config.depthPercentileThreshold < 1.0 {
            depthPercentileValue = computePercentile(normalizedDepth, percentile: config.depthPercentileThreshold)
        }
        
        // DA3-Quality: Compute border trim bounds
        let trimH = config.enableBorderPruning ? Int(config.borderTrimPercent * Float(height)) : 0
        let trimW = config.enableBorderPruning ? Int(config.borderTrimPercent * Float(width)) : 0
        let minY = trimH
        let maxY = height - trimH
        let minX = trimW
        let maxX = width - trimW
        
        // Calculate expected point count for reservation (accounting for border trim)
        let effectiveWidth = max(0, maxX - minX)
        let effectiveHeight = max(0, maxY - minY)
        let expectedCount = (effectiveWidth / config.subsample) * (effectiveHeight / config.subsample)
        cloud.reserve(expectedCount)
        
        // Unproject each pixel to 3D
        for y in stride(from: minY, to: maxY, by: config.subsample) {
            for x in stride(from: minX, to: maxX, by: config.subsample) {
                let idx = y * width + x
                guard idx < normalizedDepth.count else { continue }
                
                let d = normalizedDepth[idx]
                
                // Filter by depth threshold
                guard d >= config.minDepth && d <= config.maxDepth else { continue }
                
                // DA3-Quality: Filter by depth percentile
                guard d <= depthPercentileValue else { continue }
                
                // Filter by confidence
                if let conf = confidence, idx < conf.count {
                    guard conf[idx] >= config.minConfidence else { continue }
                }
                
                // DA3 depth semantics:
                // Depth is a ray distance along a **unit** camera ray direction (not Z-depth).
                let u = (Float(x) + 0.5 - intrinsics.cx) / max(1e-6, intrinsics.fx)
                let v = (Float(y) + 0.5 - intrinsics.cy) / max(1e-6, intrinsics.fy)
                var dir = simd_float3(u, v, 1.0)
                let dirLen = simd_length(dir)
                guard dirLen.isFinite, dirLen > 0 else { continue }
                dir /= dirLen
                let camPoint = dir * d
                
                // Get color
                let color: (r: Float, g: Float, b: Float)
                if let colors = colors, idx < colors.count {
                    color = colors[idx]
                } else {
                    // Default gray
                    color = (0.5, 0.5, 0.5)
                }
                
                // Create Gaussian splat
                let splat = DA3GaussianSplat(
                    position: (camPoint.x, camPoint.y, camPoint.z),
                    rgb: color,
                    opacity: config.defaultOpacity,
                    scale: config.gaussianScale
                )
                
                cloud.add(splat)
            }
        }
        
        // DA3-Quality: Apply shift & scale normalization if enabled
        if config.shiftAndScale {
            cloud.shiftAndScaleNormalize()
        }
        
        return cloud
    }
    
    // MARK: - World-Space Conversion (DA3-Compatible)
    
    /// Convert depth array to world-space Gaussian cloud using camera extrinsics
    ///
    /// This produces world-space Gaussians matching DA3's Python implementation.
    /// Points are unprojected to camera space, then transformed to world space.
    /// Gaussian orientations are also rotated to world space.
    public func convertWorldSpace(
        depth: [Float],
        confidence: [Float]?,
        colors: [(r: Float, g: Float, b: Float)]?,
        width: Int,
        height: Int,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics
    ) -> DA3GaussianCloud {
        let cloud = DA3GaussianCloud()
        
        // Normalize depth if requested
        var normalizedDepth = depth
        if config.normalizeDepth {
            let minD = depth.min() ?? 0
            let maxD = depth.max() ?? 1
            let range = maxD - minD
            if range > 0 {
                normalizedDepth = depth.map { (($0 - minD) / range) * config.maxNormalizedDepth }
            }
        }
        
        // DA3-Quality: Compute depth percentile threshold
        var depthPercentileValue: Float = .greatestFiniteMagnitude
        if config.enableDepthPercentilePruning && config.depthPercentileThreshold < 1.0 {
            depthPercentileValue = computePercentile(normalizedDepth, percentile: config.depthPercentileThreshold)
        }
        
        // DA3-Quality: Compute border trim bounds
        let trimH = config.enableBorderPruning ? Int(config.borderTrimPercent * Float(height)) : 0
        let trimW = config.enableBorderPruning ? Int(config.borderTrimPercent * Float(width)) : 0
        let minY = trimH
        let maxY = height - trimH
        let minX = trimW
        let maxX = width - trimW
        
        let effectiveWidth = max(0, maxX - minX)
        let effectiveHeight = max(0, maxY - minY)
        let expectedCount = (effectiveWidth / config.subsample) * (effectiveHeight / config.subsample)
        cloud.reserve(expectedCount)
        
        // Compute scale multiplier based on intrinsics (matches DA3 Python)
        let scaleMultiplier = computeScaleMultiplier(intrinsics: intrinsics)
        
        // Unproject each pixel to world space
        for y in stride(from: minY, to: maxY, by: config.subsample) {
            for x in stride(from: minX, to: maxX, by: config.subsample) {
                let idx = y * width + x
                guard idx < normalizedDepth.count else { continue }
                
                let d = normalizedDepth[idx]
                
                guard d >= config.minDepth && d <= config.maxDepth else { continue }
                guard d <= depthPercentileValue else { continue }
                
                if let conf = confidence, idx < conf.count {
                    guard conf[idx] >= config.minConfidence else { continue }
                }
                
                // DA3 depth semantics:
                // Depth is a ray distance along a **unit** camera ray direction (not Z-depth).
                let u = (Float(x) + 0.5 - intrinsics.cx) / max(1e-6, intrinsics.fx)
                let v = (Float(y) + 0.5 - intrinsics.cy) / max(1e-6, intrinsics.fy)
                var dir = simd_float3(u, v, 1.0)
                let dirLen = simd_length(dir)
                guard dirLen.isFinite, dirLen > 0 else { continue }
                dir /= dirLen
                let camPoint = dir * d
                
                // Transform to world space
                let worldPoint = extrinsics.transformPoint(camPoint)
                
                // Get color
                let color: (r: Float, g: Float, b: Float)
                if let colors = colors, idx < colors.count {
                    color = colors[idx]
                } else {
                    color = (0.5, 0.5, 0.5)
                }
                
                // Compute world-space Gaussian orientation
                // In camera space, Gaussians are aligned to view direction (identity rotation)
                // Transform to world space using camera rotation
                let camQuat = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1) // Identity
                let worldQuat = extrinsics.transformQuaternion(camQuat)
                
                // Compute depth-dependent scale (matches DA3 Python: scale * depth * multiplier)
                let gaussianScaleWorld = config.gaussianScale * d * scaleMultiplier
                
                // Create world-space Gaussian splat
                var splat = DA3GaussianSplat(
                    position: (worldPoint.x, worldPoint.y, worldPoint.z),
                    rgb: color,
                    opacity: config.defaultOpacity,
                    scale: gaussianScaleWorld
                )
                
                // Set world-space rotation (WXYZ)
                splat.rotW = worldQuat.real
                splat.rotX = worldQuat.imag.x
                splat.rotY = worldQuat.imag.y
                splat.rotZ = worldQuat.imag.z
                splat.normalizeRotation()
                
                cloud.add(splat)
            }
        }
        
        if config.shiftAndScale {
            cloud.shiftAndScaleNormalize()
        }
        
        return cloud
    }
    
    // MARK: - Multi-View Fusion
    
    /// Input for multi-view world-space conversion
    public struct ViewInput {
        public let depth: [Float]
        public let confidence: [Float]?
        public let colors: [(r: Float, g: Float, b: Float)]?
        public let width: Int
        public let height: Int
        public let intrinsics: CameraIntrinsics
        public let extrinsics: CameraExtrinsics
        
        public init(
            depth: [Float],
            confidence: [Float]?,
            colors: [(r: Float, g: Float, b: Float)]?,
            width: Int,
            height: Int,
            intrinsics: CameraIntrinsics,
            extrinsics: CameraExtrinsics
        ) {
            self.depth = depth
            self.confidence = confidence
            self.colors = colors
            self.width = width
            self.height = height
            self.intrinsics = intrinsics
            self.extrinsics = extrinsics
        }
    }
    
    /// Convert multiple views to a single world-space Gaussian cloud
    ///
    /// This fuses Gaussians from multiple camera views into a unified world-space
    /// representation, matching DA3's multi-view 3DGS output.
    ///
    /// - Parameter views: Array of view inputs, each with depth, colors, and camera params
    /// - Returns: Fused world-space Gaussian cloud
    public func convertMultiViewWorldSpace(views: [ViewInput]) -> DA3GaussianCloud {
        let fusedCloud = DA3GaussianCloud()
        
        // Temporarily disable shiftAndScale for per-view conversion
        // We'll apply it once globally after fusion
        let originalShiftAndScale = config.shiftAndScale
        config.shiftAndScale = false
        
        // Process views at configured interval
        for (viewIdx, view) in views.enumerated() {
            guard viewIdx % config.viewInterval == 0 else { continue }
            
            let viewCloud = convertWorldSpace(
                depth: view.depth,
                confidence: view.confidence,
                colors: view.colors,
                width: view.width,
                height: view.height,
                intrinsics: view.intrinsics,
                extrinsics: view.extrinsics
            )
            
            fusedCloud.add(contentsOf: viewCloud.allSplats)
        }
        
        // Restore original config
        config.shiftAndScale = originalShiftAndScale
        
        // Apply global normalization after fusion (if enabled)
        if config.shiftAndScale {
            fusedCloud.shiftAndScaleNormalize()
        }
        
        return fusedCloud
    }
    
    /// Convert multiple DA3 results with camera poses to world-space Gaussian cloud
    public func convertMultiViewWorldSpace(
        results: [(result: DA3CoreML.Result, image: CGImage?, extrinsics: CameraExtrinsics)],
        intrinsics: CameraIntrinsics?
    ) throws -> DA3GaussianCloud {
        var views: [ViewInput] = []
        
        for (result, image, extrinsics) in results {
            let width = result.originalSize.width
            let height = result.originalSize.height
            let camera = intrinsics ?? CameraIntrinsics.estimate(width: width, height: height)
            
            var depthValues = [Float](repeating: 0, count: result.depth.count)
            for i in 0..<result.depth.count {
                depthValues[i] = result.depth[i].floatValue
            }
            
            var confidenceValues: [Float] = [Float](repeating: 1.0, count: result.depthConfidence.count)
            for i in 0..<result.depthConfidence.count {
                confidenceValues[i] = result.depthConfidence[i].floatValue
            }
            
            var colors: [(r: Float, g: Float, b: Float)]?
            if let img = image {
                colors = try extractColors(from: img, width: width, height: height)
            }
            
            views.append(ViewInput(
                depth: depthValues,
                confidence: confidenceValues,
                colors: colors,
                width: width,
                height: height,
                intrinsics: camera,
                extrinsics: extrinsics
            ))
        }
        
        return convertMultiViewWorldSpace(views: views)
    }
    
    // MARK: - Scale Computation (DA3-Compatible)
    
    /// Compute scale multiplier based on intrinsics
    /// Matches DA3 Python's get_scale_multiplier:
    /// xy_multipliers = multiplier * einsum(inv(K[:2,:2]), pixel_size, "i j, j -> i")
    /// return xy_multipliers.sum()
    private func computeScaleMultiplier(intrinsics: CameraIntrinsics) -> Float {
        // Matches DA3 Python's GaussianAdapter.get_scale_multiplier:
        //   intr_normed = K; intr_normed[0,:] /= W; intr_normed[1,:] /= H
        //   pixel_size = (1/W, 1/H)
        //   xy_multipliers = 0.1 * inv(intr_normed[:2,:2]) @ pixel_size
        // which simplifies (for pinhole intrinsics in pixel units) to:
        //   0.1 * (1/fx + 1/fy)
        let multiplier: Float = 0.1
        return multiplier * ((1.0 / intrinsics.fx) + (1.0 / intrinsics.fy))
    }
    
    // MARK: - Percentile Computation
    
    /// Compute the value at a given percentile (0.0 to 1.0)
    /// Uses linear interpolation for accuracy
    private func computePercentile(_ values: [Float], percentile: Float) -> Float {
        guard !values.isEmpty else { return 0 }
        guard percentile > 0 && percentile < 1 else {
            return percentile <= 0 ? (values.min() ?? 0) : (values.max() ?? 0)
        }
        
        // Filter out invalid values and sort
        let validValues = values.filter { $0.isFinite && $0 > 0 }
        guard !validValues.isEmpty else { return 0 }
        
        let sorted = validValues.sorted()
        let n = sorted.count
        
        // Calculate index with linear interpolation
        let index = percentile * Float(n - 1)
        let lowerIndex = Int(index)
        let upperIndex = min(lowerIndex + 1, n - 1)
        let fraction = index - Float(lowerIndex)
        
        // Linear interpolation between adjacent values
        return sorted[lowerIndex] * (1 - fraction) + sorted[upperIndex] * fraction
    }
    
    // MARK: - Color Extraction
    
    private func extractColors(from image: CGImage, width: Int, height: Int) throws -> [(r: Float, g: Float, b: Float)] {
        // Resize image to match depth dimensions
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw DA3Error.imageProcessingFailed("Failed to create context for color extraction")
        }
        
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let data = context.data else {
            throw DA3Error.imageProcessingFailed("Failed to get pixel data")
        }
        
        let ptr = data.assumingMemoryBound(to: UInt8.self)
        var colors = [(r: Float, g: Float, b: Float)]()
        colors.reserveCapacity(width * height)
        
        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * 4
                let r = Float(ptr[offset]) / 255.0
                let g = Float(ptr[offset + 1]) / 255.0
                let b = Float(ptr[offset + 2]) / 255.0
                colors.append((r, g, b))
            }
        }
        
        return colors
    }
}

// MARK: - PLY Writer for DA3

/// Writes DA3 Gaussian clouds to PLY format.
@available(macOS 14.0, iOS 17.0, *)
public final class DA3PLYWriter {
    
    public enum Format {
        case binary
        case ascii
    }
    
    public init() {}
    
    /// Write Gaussian cloud to PLY file
    public func write(_ cloud: DA3GaussianCloud, to path: String, format: Format = .binary, comments: [String] = []) throws {
        let url = URL(fileURLWithPath: path)
        var data = Data()
        
        let extraComments = comments
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        var headerLines: [String] = []
        headerLines.append("ply")
        headerLines.append("format \(format == .binary ? "binary_little_endian" : "ascii") 1.0")
        headerLines.append("comment Generated by DA3CoreML")
        headerLines.append(contentsOf: extraComments.map { "comment \($0)" })
        headerLines.append("element vertex \(cloud.count)")
        headerLines.append("property float x")
        headerLines.append("property float y")
        headerLines.append("property float z")
        headerLines.append("property float nx")
        headerLines.append("property float ny")
        headerLines.append("property float nz")
        headerLines.append("property float f_dc_0")
        headerLines.append("property float f_dc_1")
        headerLines.append("property float f_dc_2")
        headerLines.append("property float opacity")
        headerLines.append("property float scale_0")
        headerLines.append("property float scale_1")
        headerLines.append("property float scale_2")
        headerLines.append("property float rot_0")
        headerLines.append("property float rot_1")
        headerLines.append("property float rot_2")
        headerLines.append("property float rot_3")
        headerLines.append("end_header")

        let header = headerLines.joined(separator: "\n") + "\n"
        
        data.append(header.data(using: .ascii)!)
        
        // Write splats
        if format == .binary {
            for splat in cloud.allSplats {
                // Position
                appendFloat(&data, splat.x)
                appendFloat(&data, splat.y)
                appendFloat(&data, splat.z)
                // Normals (dummy)
                appendFloat(&data, 0)
                appendFloat(&data, 0)
                appendFloat(&data, 1)
                // Color (SH DC)
                appendFloat(&data, splat.shDC0)
                appendFloat(&data, splat.shDC1)
                appendFloat(&data, splat.shDC2)
                // Opacity
                appendFloat(&data, splat.opacityLogit)
                // Scale
                appendFloat(&data, splat.scaleLog0)
                appendFloat(&data, splat.scaleLog1)
                appendFloat(&data, splat.scaleLog2)
                // Rotation
                appendFloat(&data, splat.rotW)
                appendFloat(&data, splat.rotX)
                appendFloat(&data, splat.rotY)
                appendFloat(&data, splat.rotZ)
            }
        } else {
            for splat in cloud.allSplats {
                let line = String(format: "%.6f %.6f %.6f 0 0 1 %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                    splat.x, splat.y, splat.z,
                    splat.shDC0, splat.shDC1, splat.shDC2,
                    splat.opacityLogit,
                    splat.scaleLog0, splat.scaleLog1, splat.scaleLog2,
                    splat.rotW, splat.rotX, splat.rotY, splat.rotZ
                )
                data.append(line.data(using: .ascii)!)
            }
        }
        
        try data.write(to: url)
    }
    
    private func appendFloat(_ data: inout Data, _ value: Float) {
        var v = value
        withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
    }
}
