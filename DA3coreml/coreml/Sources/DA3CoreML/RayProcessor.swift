import Foundation
import CoreML
import Accelerate

/// Depth-Anything-3 CoreML implementation.
///
/// This is the main entry point for depth and ray estimation using CoreML.
/// It combines a DINOv3 vision transformer backbone with a DualDPT head.
///
/// Features:
/// - Memory-efficient batched inference for large images
/// - Automatic chunking for 128GB+ RAM constraints
/// - Float16 precision for Apple Silicon optimization
/// - Supports both depth-only and depth+ray modes
///
/// Usage:
/// ```swift
/// let da3 = try DA3CoreML(
///     backbonePath: "dinov3.mlpackage",
///     headPath: "dualdpt.mlpackage"
/// )
/// let result = try da3.predict(image: myImage)
/// print("Depth range: \(result.depthRange)")
/// ```
@available(macOS 14.0, iOS 17.0, *)
public final class DA3CoreML {
    
    // MARK: - Types

    /// Post-processing backend for operations that are not part of the CoreML neural graphs.
    ///
    /// CoreML/ANE is great for neural inference; large postprocess steps (resize, tile blending)
    /// can be moved to Metal for speed and to keep numerically sensitive math in float32.
    public enum PostprocessBackend: String {
        case cpu
        case metal
    }

    /// Confidence activation applied *outside* CoreML when the head exports confidence logits.
    ///
    /// Official DA3 heads typically embed `conf_activation="expp1"` inside the model graph
    /// and therefore should use `.linear` here to avoid double-activation.
    public enum ConfidenceActivation: String {
        /// No-op: the CoreML head already produced the final confidence values.
        case linear
        /// DA3 default: `exp(x) + 1` (positive weights).
        case expp1
        /// Stable positive alternative: `softplus(x) + 1` (always finite in float32).
        case softplus1
    }

    /// Model size variants
    ///
    /// Supports both DINOv2 and DINOv3 backbones:
    /// - DINOv2: small, base, large, giant (patch_size=14)
    /// - DINOv3: large, huge (patch_size=16)
    public enum ModelSize: String, CaseIterable {
        case small = "small"   // DINOv2 ~22M params
        case base = "base"     // DINOv2 ~98M params
        case large = "large"   // DINOv2/v3 ~335M params (1024 hidden)
        case giant = "giant"   // DINOv2 ~1.1B params (1536 hidden, patch14)
        case huge = "huge"     // DINOv3 ViT-H+ (1280 hidden, patch16)

        /// Hidden dimension for each size (backbone embed_dim)
        public var hiddenDim: Int {
            switch self {
            case .small: return 384
            case .base: return 768
            case .large: return 1024
            case .giant: return 1536
            case .huge: return 1280
            }
        }

        /// Feature dimension output by backbone (2x hiddenDim due to cat_token=True)
        /// This is the actual dimension used by DualDPT and GS heads
        public var featureDim: Int {
            return hiddenDim * 2
        }

        /// Patch size (14 for DINOv2, 16 for DINOv3)
        public var patchSize: Int {
            switch self {
            case .small, .base, .giant: return 14  // DINOv2
            case .large, .huge: return 16          // DINOv3
            }
        }

        /// Estimated memory usage in GB
        public var estimatedMemoryGB: Double {
            switch self {
            case .small: return 0.5
            case .base: return 1.5
            case .large: return 4.0
            case .giant: return 12.0
            case .huge: return 10.0
            }
        }
    }
    
    /// Progress update for batch processing
    public struct ProgressUpdate {
        /// Current item index (0-based)
        public let current: Int
        /// Total number of items
        public let total: Int
        /// Progress fraction (0.0 to 1.0)
        public var progress: Float { Float(current + 1) / Float(total) }
        /// Current stage description
        public let stage: String
        /// Optional message
        public let message: String?

        public init(current: Int, total: Int, stage: String, message: String? = nil) {
            self.current = current
            self.total = total
            self.stage = stage
            self.message = message
        }
    }

    /// Progress callback closure type
    public typealias ProgressCallback = (ProgressUpdate) -> Void

    /// Configuration for DA3
    public struct Config {
        /// Model size variant
        public var modelSize: ModelSize = .base
        /// Input image size (default: 518 for DA3)
        public var inputSize: Int = 518
        /// Override patch size; when nil, uses modelSize default or backbone metadata
        public var patchSize: Int? = nil
        /// Override register token count (for DINOv3); optional
        public var registerTokens: Int? = nil
        /// Maximum batch size for memory efficiency
        public var maxBatchSize: Int = 4
        /// Enable tiled inference for very large images
        public var enableTiling: Bool = true
        /// Maximum tile size when tiling
        public var maxTileSize: Int = 1024
        /// Tile overlap in pixels
        public var tileOverlap: Int = 64
        /// Memory limit in GB (for automatic batch sizing)
        /// Note: This is the USER-SPECIFIED limit, actual usage will be lower due to safety buffer
        public var memoryLimitGB: Double = 64.0
        /// Safety buffer percentage (0.0-1.0) - reserves this much RAM for system
        /// Default: 0.30 (30%) - on a 128GB system, max 89.6GB will be used
        public var safetyBufferPercent: Double = 0.30
        /// Use GPU (ANE + GPU) if available - uses .all compute units for optimal ANE/GPU/CPU scheduling
        public var useGPU: Bool = true
        /// Prefer Neural Engine over GPU when possible
        /// When true, uses .cpuAndNeuralEngine; when false with useGPU=true, uses .all
        public var preferNeuralEngine: Bool = false
        /// Override compute selection for the DualDPT head only.
        ///
        /// This is useful for float32 head models which may only be reliable on CPU.
        /// When nil, the head uses `useGPU` / `preferNeuralEngine`.
        public var headUseGPU: Bool? = nil
        /// Prefer Neural Engine for the head only (when `headUseGPU` is true).
        /// When nil, uses `preferNeuralEngine`.
        public var headPreferNeuralEngine: Bool? = nil
        /// Output depth activation (exp, relu, sigmoid, softplus)
        /// Depth activation applied in Swift after the head output. Set to `.linear` when the
        /// exported CoreML head already includes its activation (DA3 checkpoints do). Use `.exp`
        /// only if you export a linear head and want to mirror the PyTorch post-processing.
        public var depthActivation: DepthActivation = .linear

        /// Confidence activation applied in Swift when the head exports *logits* (pre-activation).
        /// For the official DA3 exports (which already include `expp1`), keep this as `.linear`.
        public var confidenceActivation: ConfidenceActivation = .linear

        /// Clamp range applied to confidence logits before exp/softplus (float32 safety belt).
        /// This is only used when `confidenceActivation != .linear`.
        public var confidenceLogitClampMin: Float = -30.0
        public var confidenceLogitClampMax: Float = 30.0

        /// Backend for post-processing (resize, tiling blend). Default is CPU for maximum portability.
        public var postprocessBackend: PostprocessBackend = .cpu
        /// Enable verbose memory logging
        public var verboseMemory: Bool = false
        /// Progress callback for batch processing (called on main thread)
        public var progressCallback: ProgressCallback? = nil

        public init() {}

        /// Compute units based on configuration
        public var computeUnits: MLComputeUnits {
            if !useGPU {
                return .cpuOnly
            } else if preferNeuralEngine {
                return .cpuAndNeuralEngine
            } else {
                return .all  // Let CoreML decide optimal mix of CPU/GPU/ANE
            }
        }
        
        /// Effective memory limit after applying safety buffer
        public var effectiveMemoryLimitGB: Double {
            memoryLimitGB * (1.0 - safetyBufferPercent)
        }
        
        /// Calculate optimal batch size based on memory limit with safety buffer
        public var optimalBatchSize: Int {
            let memPerImage = modelSize.estimatedMemoryGB * 2.0 // 2x safety margin for intermediate tensors
            let safeBudget = effectiveMemoryLimitGB
            return max(1, Int(safeBudget / memPerImage))
        }
    }
    
    /// Depth activation function
    public enum DepthActivation: String {
        case linear = "linear"
        case exp = "exp"
        case relu = "relu"
        case sigmoid = "sigmoid"
        case softplus = "softplus"
    }
    
    /// Result from depth-ray prediction
    public struct Result {
        /// Depth map - shape: (H, W) in meters or normalized
        public let depth: MLMultiArray
        /// Depth confidence map - shape: (H, W)
        ///
        /// Note: Official DA3 DualDPT checkpoints use `conf_activation="expp1"` (exp(x)+1),
        /// so confidence values are **positive weights** (>= 1), not probabilities.
        public let depthConfidence: MLMultiArray
        /// Ray field - shape: (C, H, W) where C is typically 6 for official DA3 checkpoints.
        ///
        /// DA3's `ray` tensor encodes:
        /// - channels 0..2: a direction/target vector (used for pose/intrinsics estimation)
        /// - channels 3..5: a translation vector (used as a weighted average)
        public let rays: MLMultiArray?
        /// Ray confidence - shape: (H, W)
        ///
        /// Note: Official DA3 DualDPT checkpoints use `conf_activation="expp1"` (exp(x)+1),
        /// so confidence values are **positive weights** (>= 1), not probabilities.
        public let rayConfidence: MLMultiArray?
        /// Original image size
        public let originalSize: (width: Int, height: Int)
        /// Inference time in seconds
        public let inferenceTime: TimeInterval
        
        /// Minimum depth value
        public var minDepth: Float {
            var minVal: Float = .greatestFiniteMagnitude
            for i in 0..<depth.count {
                let val = depth[i].floatValue
                if val < minVal { minVal = val }
            }
            return minVal
        }
        
        /// Maximum depth value
        public var maxDepth: Float {
            var maxVal: Float = -.greatestFiniteMagnitude
            for i in 0..<depth.count {
                let val = depth[i].floatValue
                if val > maxVal { maxVal = val }
            }
            return maxVal
        }
        
        /// Depth range as tuple
        public var depthRange: (min: Float, max: Float) {
            (minDepth, maxDepth)
        }
    }

    /// Detailed prediction including intermediate (model-space) tensors.
    ///
    /// This exists to support “DA3-convention” debugging and downstream modules (ray-pose, GSHead)
    /// without re-running the backbone/head. The `result` is always present; the `head*` tensors
    /// are only populated for the **non-tiled** inference path.
    public struct DetailedResult {
        /// Final postprocessed result (typically resized back to the original image size).
        public let result: Result

        /// Depth at model resolution (after `depthActivation` if configured).
        public let headDepth: MLMultiArray?
        /// Depth confidence at model resolution (after `confidenceActivation` if configured).
        public let headDepthConfidence: MLMultiArray?
        /// Ray field at the native head resolution (typically 296×296 for 518 input with patch=14).
        public let headRays: MLMultiArray?
        /// Ray confidence at the native head resolution (after `confidenceActivation` if configured).
        public let headRayConfidence: MLMultiArray?

        /// Preprocess metadata describing the resize/pad applied before inference.
        public let preprocessInfo: DINOv3CoreML.PreprocessInfo?

        public init(
            result: Result,
            headDepth: MLMultiArray?,
            headDepthConfidence: MLMultiArray?,
            headRays: MLMultiArray?,
            headRayConfidence: MLMultiArray?,
            preprocessInfo: DINOv3CoreML.PreprocessInfo?
        ) {
            self.result = result
            self.headDepth = headDepth
            self.headDepthConfidence = headDepthConfidence
            self.headRays = headRays
            self.headRayConfidence = headRayConfidence
            self.preprocessInfo = preprocessInfo
        }
    }
    
    // MARK: - Properties
    
    let backbone: DINOv3CoreML
    let head: DualDPTCoreML
    public let config: Config
    private let memoryManager: MemoryManager
    private let metalPostProcessor: DA3MetalPostProcessor?
    
    // MARK: - Initialization
    
    /// Initialize with separate backbone and head model paths
    public init(backbonePath: String, headPath: String, config: Config = Config()) throws {
        var effectiveConfig = config
        
        // Initialize memory manager with safety buffer
        var memConfig = MemoryManager.Config()
        memConfig.safetyBufferPercent = effectiveConfig.safetyBufferPercent
        memConfig.maxUsagePercent = 0.60  // Use max 60% of available at once
        memConfig.minimumFreeGB = 4.0     // Always keep 4GB free
        self.memoryManager = MemoryManager(config: memConfig)

        // Metal is not always available (e.g. CI / headless environments). When requested but
        // unavailable, fall back to CPU postprocess instead of failing model load.
        if effectiveConfig.postprocessBackend == .metal {
            if let mp = DA3MetalPostProcessor.shared() {
                self.metalPostProcessor = mp
            } else {
                self.metalPostProcessor = nil
                effectiveConfig.postprocessBackend = .cpu
                if effectiveConfig.verboseMemory {
                    print("⚠️ Metal postprocess requested but Metal is unavailable; falling back to CPU postprocess.")
                }
            }
        } else {
            self.metalPostProcessor = nil
        }

        self.config = effectiveConfig
        
        // Check if we have enough memory before loading models
        let requiredGB = effectiveConfig.modelSize.estimatedMemoryGB * 1.5
        if !memoryManager.isSafeToAllocate(requiredGB: requiredGB) {
            let stats = memoryManager.getMemoryStats()
            throw DA3Error.outOfMemory(
                "Insufficient memory to load \(effectiveConfig.modelSize.rawValue) model. " +
                "Required: \(String(format: "%.1f", requiredGB))GB, " +
                "Available: \(String(format: "%.1f", stats.availableGB))GB"
            )
        }
        
        if effectiveConfig.verboseMemory {
            memoryManager.logMemoryStats()
        }
        
        var backboneConfig = DINOv3CoreML.Config()
        // Try to pick up patch size / register tokens from model metadata (preferred for DINOv3)
        var resolvedPatchSize = effectiveConfig.patchSize ?? effectiveConfig.modelSize.patchSize
        var resolvedRegisterTokens = effectiveConfig.registerTokens
        if let meta = DINOv3CoreML.BackboneMetadata.load(fromPath: backbonePath) {
            if effectiveConfig.patchSize == nil, let ps = meta.patchSize { resolvedPatchSize = ps }
            if effectiveConfig.registerTokens == nil, let rt = meta.registerTokens { resolvedRegisterTokens = rt }
        }

        backboneConfig.inputSize = effectiveConfig.inputSize
        backboneConfig.patchSize = resolvedPatchSize
        backboneConfig.hiddenDim = effectiveConfig.modelSize.hiddenDim
        backboneConfig.maxBatchSize = effectiveConfig.optimalBatchSize
        backboneConfig.useGPU = effectiveConfig.useGPU
        backboneConfig.preferNeuralEngine = effectiveConfig.preferNeuralEngine
        if let rt = resolvedRegisterTokens { backboneConfig.numRegisterTokens = rt }

        var headConfig = DualDPTCoreML.Config()
        headConfig.dimIn = effectiveConfig.modelSize.featureDim  // Use featureDim (2x hiddenDim due to cat_token)
        headConfig.patchSize = resolvedPatchSize
        headConfig.useGPU = effectiveConfig.headUseGPU ?? effectiveConfig.useGPU
        headConfig.preferNeuralEngine = effectiveConfig.headPreferNeuralEngine ?? effectiveConfig.preferNeuralEngine

        self.backbone = try DINOv3CoreML(modelPath: backbonePath, config: backboneConfig)
        self.head = try DualDPTCoreML(modelPath: headPath, config: headConfig)
    }
    
    /// Initialize with model URLs
    public init(backboneURL: URL, headURL: URL, config: Config = Config()) throws {
        var effectiveConfig = config
        
        // Initialize memory manager
        var memConfig = MemoryManager.Config()
        memConfig.safetyBufferPercent = effectiveConfig.safetyBufferPercent
        self.memoryManager = MemoryManager(config: memConfig)

        if effectiveConfig.postprocessBackend == .metal {
            if let mp = try? DA3MetalPostProcessor() {
                self.metalPostProcessor = mp
            } else {
                self.metalPostProcessor = nil
                effectiveConfig.postprocessBackend = .cpu
                if effectiveConfig.verboseMemory {
                    print("⚠️ Metal postprocess requested but Metal is unavailable; falling back to CPU postprocess.")
                }
            }
        } else {
            self.metalPostProcessor = nil
        }

        self.config = effectiveConfig
        
        // Check memory before loading
        let requiredGB = effectiveConfig.modelSize.estimatedMemoryGB * 1.5
        if !memoryManager.isSafeToAllocate(requiredGB: requiredGB) {
            let stats = memoryManager.getMemoryStats()
            throw DA3Error.outOfMemory(
                "Insufficient memory to load \(effectiveConfig.modelSize.rawValue) model. " +
                "Required: \(String(format: "%.1f", requiredGB))GB, " +
                "Available: \(String(format: "%.1f", stats.availableGB))GB"
            )
        }
        
        var backboneConfig = DINOv3CoreML.Config()
        // Resolve patch size/register tokens from metadata
        var resolvedPatchSize = effectiveConfig.patchSize ?? effectiveConfig.modelSize.patchSize
        var resolvedRegisterTokens = effectiveConfig.registerTokens
        if let meta = DINOv3CoreML.BackboneMetadata.load(fromURL: backboneURL) {
            if effectiveConfig.patchSize == nil, let ps = meta.patchSize { resolvedPatchSize = ps }
            if effectiveConfig.registerTokens == nil, let rt = meta.registerTokens { resolvedRegisterTokens = rt }
        }

        backboneConfig.inputSize = effectiveConfig.inputSize
        backboneConfig.patchSize = resolvedPatchSize
        backboneConfig.hiddenDim = effectiveConfig.modelSize.hiddenDim
        backboneConfig.maxBatchSize = effectiveConfig.optimalBatchSize
        backboneConfig.useGPU = effectiveConfig.useGPU
        backboneConfig.preferNeuralEngine = effectiveConfig.preferNeuralEngine
        if let rt = resolvedRegisterTokens { backboneConfig.numRegisterTokens = rt }

        var headConfig = DualDPTCoreML.Config()
        headConfig.dimIn = effectiveConfig.modelSize.featureDim  // Use featureDim (2x hiddenDim due to cat_token)
        headConfig.patchSize = resolvedPatchSize
        headConfig.useGPU = effectiveConfig.headUseGPU ?? effectiveConfig.useGPU
        headConfig.preferNeuralEngine = effectiveConfig.headPreferNeuralEngine ?? effectiveConfig.preferNeuralEngine

        self.backbone = try DINOv3CoreML(modelURL: backboneURL, config: backboneConfig)
        self.head = try DualDPTCoreML(modelURL: headURL, config: headConfig)
    }

    /// Extract backbone features (and preprocess info) for external modules like camera decoder.
    public func extractBackboneFeatures(image: CGImage) throws -> (DINOv3CoreML.Features, DINOv3CoreML.PreprocessInfo?) {
        let feats = try backbone.extractFeatures(from: image, normalize: true)
        return (feats, backbone.lastPreprocessInfo)
    }
    
    // MARK: - Prediction
    
    /// Predict depth and rays from a CGImage
    ///
    /// - Parameters:
    ///   - image: Input CGImage
    ///   - includeRays: Whether to compute ray directions (default: true)
    /// - Returns: Depth and ray prediction result
    public func predict(image: CGImage, includeRays: Bool = true) throws -> Result {
        return try predictDetailed(image: image, includeRays: includeRays).result
    }

    /// Predict depth and rays, optionally returning intermediate tensors at model/head resolution.
    ///
    /// - Important: When tiled inference is used, intermediate tensors are generally tile-local.
    ///   If `includeRays == true`, this implementation runs an additional **global** pass at model
    ///   resolution to produce `headRays` / `headRayConfidence` for DA3-style ray-pose estimation.
    public func predictDetailed(image: CGImage, includeRays: Bool = true) throws -> DetailedResult {
        // Check memory pressure before inference
        try memoryManager.checkMemoryPressure()
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let originalSize = (width: image.width, height: image.height)
        
        // Check if we need tiling for large images
        if config.enableTiling && (image.width > config.maxTileSize || image.height > config.maxTileSize) {
            // Depth benefits from tiling; ray fields do not (they are globally constrained and
            // tile-local crops can yield inconsistent rays/intrinsics). When rays are requested,
            // compute them once from a global pass at model resolution, then upscale/crop.
            let headRaysInfo: (headRays: MLMultiArray, headRayConf: MLMultiArray, preprocessInfo: DINOv3CoreML.PreprocessInfo?)?
            if includeRays {
                headRaysInfo = try predictHeadRaysForRayPose(image: image)
            } else {
                headRaysInfo = nil
            }

            let tiledDepthOnly = try predictTiled(image: image, includeRays: false, startTime: startTime)

            if includeRays, let info = headRaysInfo {
                // Resize/crop the head ray grid to the original image size (matches non-tiled behavior).
                let (raysOut, rayConfOut) = try postprocessRays(
                    rays: info.headRays,
                    rayConf: info.headRayConf,
                    targetWidth: originalSize.width,
                    targetHeight: originalSize.height,
                    preprocessInfo: info.preprocessInfo
                )

                let inferenceTime = CFAbsoluteTimeGetCurrent() - startTime
                let result = Result(
                    depth: tiledDepthOnly.depth,
                    depthConfidence: tiledDepthOnly.depthConfidence,
                    rays: raysOut,
                    rayConfidence: rayConfOut,
                    originalSize: originalSize,
                    inferenceTime: inferenceTime
                )
                return DetailedResult(
                    result: result,
                    headDepth: nil,
                    headDepthConfidence: nil,
                    headRays: info.headRays,
                    headRayConfidence: info.headRayConf,
                    preprocessInfo: info.preprocessInfo
                )
            }

            let inferenceTime = CFAbsoluteTimeGetCurrent() - startTime
            let result = Result(
                depth: tiledDepthOnly.depth,
                depthConfidence: tiledDepthOnly.depthConfidence,
                rays: nil,
                rayConfidence: nil,
                originalSize: originalSize,
                inferenceTime: inferenceTime
            )
            return DetailedResult(
                result: result,
                headDepth: nil,
                headDepthConfidence: nil,
                headRays: nil,
                headRayConfidence: nil,
                preprocessInfo: nil
            )
        }
        
        // Run inference with memory cleanup
        return try memoryManager.withMemoryCleanup {
            // Extract backbone features
            let features = try backbone.extractFeatures(from: image, normalize: true)
            let preprocessInfo = backbone.lastPreprocessInfo
            
            // Run DualDPT head
            let prediction = try head.predict(from: features)

            // Defensive validation: some float16 DualDPT exports can emit NaNs/Infs in the ray branch
            // while depth remains finite. Fail fast (only if the caller requested rays).
            if includeRays {
                try validateFinite(prediction.rays, name: "rays")
            }

            // Apply depth activation
            let activatedDepth = try applyDepthActivation(prediction.depth)
            let activatedDepthConf = try applyConfidenceActivation(prediction.depthConfidence, name: "depth_confidence")
            let activatedRayConf: MLMultiArray?
            if includeRays {
                activatedRayConf = try applyConfidenceActivation(prediction.rayConfidence, name: "ray_confidence")
            } else {
                activatedRayConf = nil
            }

            // Resize/crop back to the original image size
            let post = try postprocessPrediction(
                depth: activatedDepth,
                depthConf: activatedDepthConf,
                rays: includeRays ? prediction.rays : nil,
                rayConf: activatedRayConf,
                targetWidth: originalSize.width,
                targetHeight: originalSize.height,
                preprocessInfo: preprocessInfo
            )

            let inferenceTime = CFAbsoluteTimeGetCurrent() - startTime
            let result = Result(
                depth: post.depth,
                depthConfidence: post.depthConf,
                rays: includeRays ? post.rays : nil,
                rayConfidence: includeRays ? post.rayConf : nil,
                originalSize: originalSize,
                inferenceTime: inferenceTime
            )

            // Return intermediate tensors at model/head resolution for debugging and DA3-style
            // downstream modules (e.g., ray pose estimation).
            return DetailedResult(
                result: result,
                headDepth: activatedDepth,
                headDepthConfidence: activatedDepthConf,
                headRays: includeRays ? prediction.rays : nil,
                headRayConfidence: activatedRayConf,
                preprocessInfo: preprocessInfo
            )
        }
    }

    // MARK: - Rays (Global pass for tiled inference)

    /// Compute the **native head ray grid** (and confidence) for the full image.
    ///
    /// This is used to support DA3-style ray-pose estimation when tiled inference is enabled.
    /// Running the ray head per-tile and blending the results is not reliable because the ray
    /// field encodes **global** camera geometry (intrinsics/rotation/translation), which can be
    /// inconsistent across independent tile crops.
    private func predictHeadRaysForRayPose(
        image: CGImage
    ) throws -> (headRays: MLMultiArray, headRayConf: MLMultiArray, preprocessInfo: DINOv3CoreML.PreprocessInfo?) {
        return try memoryManager.withMemoryCleanup {
            let features = try backbone.extractFeatures(from: image, normalize: true)
            let preprocessInfo = backbone.lastPreprocessInfo

            let prediction = try head.predict(from: features)

            // Fail fast on invalid rays; depth-only inference should be used when rays are unstable.
            try validateFinite(prediction.rays, name: "rays")

            // Confidence may be logits or weights depending on export; apply configured activation
            // (in float32) and sanitize non-finite values to 0.
            let activatedRayConf = try applyConfidenceActivation(prediction.rayConfidence, name: "ray_confidence")
            return (prediction.rays, activatedRayConf, preprocessInfo)
        }
    }

    private func postprocessRays(
        rays: MLMultiArray,
        rayConf: MLMultiArray,
        targetWidth: Int,
        targetHeight: Int,
        preprocessInfo: DINOv3CoreML.PreprocessInfo?
    ) throws -> (rays: MLMultiArray, rayConf: MLMultiArray) {
        if config.postprocessBackend == .metal, let mp = metalPostProcessor {
            let rayDims = rayArrayDims(rays)
            let rayRect = scaledCropRect(width: rayDims.width, height: rayDims.height, preprocessInfo: preprocessInfo)
            let raysOut = try mp.resizeCropCHW(
                input: rays,
                channels: rayDims.channels,
                inWidth: rayDims.width,
                inHeight: rayDims.height,
                crop: .init(startX: rayRect.startX, startY: rayRect.startY, width: rayRect.cropW, height: rayRect.cropH),
                outWidth: targetWidth,
                outHeight: targetHeight
            )

            let confDims = depthArrayDims(rayConf)
            let confRect = scaledCropRect(width: confDims.width, height: confDims.height, preprocessInfo: preprocessInfo)
            let rayConfOut = try mp.resizeCropCHW(
                input: rayConf,
                channels: 1,
                inWidth: confDims.width,
                inHeight: confDims.height,
                crop: .init(startX: confRect.startX, startY: confRect.startY, width: confRect.cropW, height: confRect.cropH),
                outWidth: targetWidth,
                outHeight: targetHeight
            )
            return (raysOut, rayConfOut)
        }

        let rayDims = rayArrayDims(rays)
        let rayData = readFloatArray(rays)

        let rayConfDims = depthArrayDims(rayConf)
        let rayConfData = readFloatArray(rayConf)

        let (croppedRays, rayCropW, rayCropH) = cropToValidRegion(
            data: rayData,
            channels: rayDims.channels,
            width: rayDims.width,
            height: rayDims.height,
            preprocessInfo: preprocessInfo
        )
        let (croppedRayConf, rayConfCropW, rayConfCropH) = cropToValidRegion(
            data: rayConfData,
            channels: 1,
            width: rayConfDims.width,
            height: rayConfDims.height,
            preprocessInfo: preprocessInfo
        )

        let resizedRays = resizeBilinear(
            data: croppedRays,
            channels: rayDims.channels,
            inWidth: rayCropW,
            inHeight: rayCropH,
            outWidth: targetWidth,
            outHeight: targetHeight
        )
        let resizedRayConf = resizeBilinear(
            data: croppedRayConf,
            channels: 1,
            inWidth: rayConfCropW,
            inHeight: rayConfCropH,
            outWidth: targetWidth,
            outHeight: targetHeight
        )

        let raysOut = try writeFloatArray(
            resizedRays,
            shape: [NSNumber(value: rayDims.channels), NSNumber(value: targetHeight), NSNumber(value: targetWidth)],
            dataType: rays.dataType
        )
        let rayConfOut = try writeFloatArray(
            resizedRayConf,
            shape: [1, NSNumber(value: targetHeight), NSNumber(value: targetWidth)],
            dataType: rayConf.dataType
        )
        return (raysOut, rayConfOut)
    }
    
    /// Predict depth for multiple images with batching
    ///
    /// Uses adaptive batch sizing based on current memory pressure.
    /// Reports progress via config.progressCallback if set.
    ///
    /// - Parameters:
    ///   - images: Array of CGImages
    ///   - includeRays: Whether to compute ray directions
    /// - Returns: Array of results, one per image
    public func predictBatch(images: [CGImage], includeRays: Bool = true) throws -> [Result] {
        // Calculate adaptive batch size based on current memory state
        let adaptiveBatchSize = memoryManager.calculateOptimalBatchSize(
            modelMemoryGB: config.modelSize.estimatedMemoryGB,
            overhead: 2.0  // Account for intermediate tensors
        )
        let batchSize = min(images.count, min(config.maxBatchSize, adaptiveBatchSize))

        if config.verboseMemory {
            print("📊 Batch size: \(batchSize) (adaptive: \(adaptiveBatchSize), max: \(config.maxBatchSize))")
            memoryManager.logMemoryStats()
        }

        // Report initial progress
        config.progressCallback?(ProgressUpdate(
            current: 0,
            total: images.count,
            stage: "starting",
            message: "Starting batch inference with \(batchSize) batch size"
        ))

        var results: [Result] = []
        var processedCount = 0

        for batchStart in stride(from: 0, to: images.count, by: batchSize) {
            // Check memory pressure before each batch
            try memoryManager.checkMemoryPressure()

            let batchEnd = min(batchStart + batchSize, images.count)
            let batch = Array(images[batchStart..<batchEnd])

            // Process batch with memory cleanup between batches
            try memoryManager.withMemoryCleanup {
                for (_, image) in batch.enumerated() {
                    let result = try predict(image: image, includeRays: includeRays)
                    results.append(result)
                    processedCount += 1

                    // Report progress after each image
                    config.progressCallback?(ProgressUpdate(
                        current: processedCount - 1,
                        total: images.count,
                        stage: "inference",
                        message: "Processed image \(processedCount)/\(images.count)"
                    ))
                }
            }

            if config.verboseMemory && batchEnd < images.count {
                print("📊 Processed \(batchEnd)/\(images.count) images")
                memoryManager.logMemoryStats()
            }
        }

        // Report completion
        config.progressCallback?(ProgressUpdate(
            current: images.count - 1,
            total: images.count,
            stage: "completed",
            message: "Batch inference complete"
        ))

        return results
    }
    
    // MARK: - Tiled Inference
    
    /// Predict depth using tiled inference for very large images
    private func predictTiled(image: CGImage, includeRays: Bool, startTime: CFAbsoluteTime) throws -> Result {
        let originalSize = (width: image.width, height: image.height)
        let tileSize = config.maxTileSize
        let overlap = config.tileOverlap
        let stride = tileSize - overlap

        let useMetalPost = (config.postprocessBackend == .metal && metalPostProcessor != nil)
        
        // Calculate number of tiles
        let numTilesX = max(1, (image.width + stride - 1) / stride)
        let numTilesY = max(1, (image.height + stride - 1) / stride)
        
        // Create output buffers and zero-initialize them
        let depthShape: [NSNumber] = [1, NSNumber(value: image.height), NSNumber(value: image.width)]
        let depthType: MLMultiArrayDataType = useMetalPost ? .float32 : .float16
        let depth = try MLMultiArray(shape: depthShape, dataType: depthType)
        // Confidence can overflow in float16 when using `exp()`-based activations. When the
        // caller opts into logits-based activation, keep confidence/weights in float32.
        let confType: MLMultiArrayDataType = useMetalPost ? .float32 : ((config.confidenceActivation == .linear) ? .float16 : .float32)
        let depthConf = try MLMultiArray(shape: depthShape, dataType: confType)

        // Zero-initialize depth and depthConf using direct pointer access
        do {
            switch depthType {
            case .float16:
                let depthPtr = UnsafeMutablePointer<Float16>(OpaquePointer(depth.dataPointer))
                for i in 0..<depth.count { depthPtr[i] = 0 }
            case .float32:
                let depthPtr = UnsafeMutablePointer<Float>(OpaquePointer(depth.dataPointer))
                for i in 0..<depth.count { depthPtr[i] = 0 }
            default:
                break
            }
            switch confType {
            case .float16:
                let confPtr = UnsafeMutablePointer<Float16>(OpaquePointer(depthConf.dataPointer))
                for i in 0..<depthConf.count { confPtr[i] = 0 }
            case .float32:
                let confPtr = UnsafeMutablePointer<Float>(OpaquePointer(depthConf.dataPointer))
                for i in 0..<depthConf.count { confPtr[i] = 0 }
            default:
                break
            }
        }

        var rays: MLMultiArray?
        var rayConf: MLMultiArray?
        var rayChannels: Int = 0

        // Weight buffer for blending overlapping regions
        let weights = try MLMultiArray(shape: depthShape, dataType: confType)
        // Zero-initialize weights using direct pointer access
        do {
            switch confType {
            case .float16:
                let weightsPtr = UnsafeMutablePointer<Float16>(OpaquePointer(weights.dataPointer))
                for i in 0..<weights.count { weightsPtr[i] = 0 }
            case .float32:
                let weightsPtr = UnsafeMutablePointer<Float>(OpaquePointer(weights.dataPointer))
                for i in 0..<weights.count { weightsPtr[i] = 0 }
            default:
                break
            }
        }
        
        // Process tiles
        // Fixed tile placement: regular tiles at stride intervals, last tile at edge
        // This ensures complete coverage without gaps (the old min() approach could create gaps
        // when intermediate tiles jumped backward)
        let totalTiles = numTilesX * numTilesY
        var processedTiles = 0

        for ty in 0..<numTilesY {
            for tx in 0..<numTilesX {
                // Check memory pressure before each tile
                try memoryManager.checkMemoryPressure()

                // Report tile progress
                config.progressCallback?(ProgressUpdate(
                    current: processedTiles,
                    total: totalTiles,
                    stage: "tiling",
                    message: "Processing tile \(processedTiles + 1)/\(totalTiles)"
                ))

                // Calculate tile position:
                // - Last tile in row/column: place at edge to ensure full coverage
                // - Other tiles: place at regular stride intervals
                let tileX: Int
                let tileY: Int

                if tx == numTilesX - 1 {
                    // Last tile in row: place at right edge
                    tileX = max(0, image.width - tileSize)
                } else {
                    tileX = tx * stride
                }

                if ty == numTilesY - 1 {
                    // Last tile in column: place at bottom edge
                    tileY = max(0, image.height - tileSize)
                } else {
                    tileY = ty * stride
                }

                let tileW = min(tileSize, image.width - tileX)
                let tileH = min(tileSize, image.height - tileY)
                
                // Extract tile
                guard let tile = image.cropping(to: CGRect(x: tileX, y: tileY, width: tileW, height: tileH)) else {
                    continue
                }
                
                // Process tile with memory cleanup
                let (tileDepth, tileDepthConf, tileRays, tileRayConf, tilePreprocessInfo) = try memoryManager.withMemoryCleanup {
                    let tileFeatures = try backbone.extractFeatures(from: tile, normalize: true)
                    let tilePreprocess = backbone.lastPreprocessInfo
                    let tilePrediction = try head.predict(from: tileFeatures)
                    if includeRays {
                        try validateFinite(tilePrediction.rays, name: "rays")
                    }
                    let tileDepth = try applyDepthActivation(tilePrediction.depth)
                    let tileDepthConf = try applyConfidenceActivation(tilePrediction.depthConfidence, name: "depth_confidence")
                    let tileRayConf: MLMultiArray?
                    if includeRays {
                        tileRayConf = try applyConfidenceActivation(tilePrediction.rayConfidence, name: "ray_confidence")
                    } else {
                        tileRayConf = nil
                    }
                    return (tileDepth, tileDepthConf, tilePrediction.rays, tileRayConf, tilePreprocess)
                }

                let post = try postprocessPrediction(
                    depth: tileDepth,
                    depthConf: tileDepthConf,
                    rays: includeRays ? tileRays : nil,
                    rayConf: tileRayConf,
                    targetWidth: tileW,
                    targetHeight: tileH,
                    preprocessInfo: tilePreprocessInfo
                )

                // Blend tile into output with distance-based weights
                if useMetalPost, let mp = metalPostProcessor {
                    try mp.blendTileDepthConf(
                        tileDepth: post.depth,
                        tileConf: post.depthConf,
                        outDepth: depth,
                        outConf: depthConf,
                        weights: weights,
                        atX: tileX,
                        atY: tileY,
                        tileW: tileW,
                        tileH: tileH,
                        overlap: overlap,
                        outW: image.width,
                        outH: image.height
                    )
                } else {
                    try blendTile(
                        tileDepth: post.depth,
                        tileConf: post.depthConf,
                        into: depth,
                        confInto: depthConf,
                        weights: weights,
                        atX: tileX,
                        atY: tileY,
                        tileW: tileW,
                        tileH: tileH,
                        overlap: overlap
                    )
                }
                
                // Blend rays if requested
                if includeRays {
                    if let postRays = post.rays, let postRayConf = post.rayConf {
                        // Allocate ray buffers lazily based on model output channels
                        if rays == nil || rayConf == nil {
                            let dims = rayArrayDims(postRays)
                            rayChannels = dims.channels
                            let rayShape: [NSNumber] = [NSNumber(value: rayChannels), NSNumber(value: image.height), NSNumber(value: image.width)]
                            let rayType: MLMultiArrayDataType = useMetalPost ? .float32 : .float16
                            rays = try MLMultiArray(shape: rayShape, dataType: rayType)
                            rayConf = try MLMultiArray(shape: depthShape, dataType: confType)

                            // Defensive zero-init (the ray blender may not touch every pixel if the
                            // caller changes tiling params or if crops are unusual).
                            if let raysOut = rays {
                                switch rayType {
                                case .float16:
                                    let ptr = UnsafeMutablePointer<Float16>(OpaquePointer(raysOut.dataPointer))
                                    for i in 0..<raysOut.count { ptr[i] = 0 }
                                case .float32:
                                    let ptr = UnsafeMutablePointer<Float>(OpaquePointer(raysOut.dataPointer))
                                    for i in 0..<raysOut.count { ptr[i] = 0 }
                                default:
                                    break
                                }
                            }
                            if let rayConfOut = rayConf {
                                switch confType {
                                case .float16:
                                    let ptr = UnsafeMutablePointer<Float16>(OpaquePointer(rayConfOut.dataPointer))
                                    for i in 0..<rayConfOut.count { ptr[i] = 0 }
                                case .float32:
                                    let ptr = UnsafeMutablePointer<Float>(OpaquePointer(rayConfOut.dataPointer))
                                    for i in 0..<rayConfOut.count { ptr[i] = 0 }
                                default:
                                    break
                                }
                            }
                        }

                        if let raysOut = rays, let rayConfOut = rayConf {
                            if useMetalPost, let mp = metalPostProcessor {
                                try mp.blendTileRays(
                                    tileRays: postRays,
                                    tileConf: postRayConf,
                                    outRays: raysOut,
                                    outConf: rayConfOut,
                                    atX: tileX,
                                    atY: tileY,
                                    tileW: tileW,
                                    tileH: tileH,
                                    overlap: overlap,
                                    outW: image.width,
                                    outH: image.height,
                                    channels: rayChannels
                                )
                            } else {
                                try blendTileRays(
                                    tileRays: postRays,
                                    tileConf: postRayConf,
                                    into: raysOut,
                                    confInto: rayConfOut,
                                    atX: tileX,
                                    atY: tileY,
                                    tileW: tileW,
                                    tileH: tileH,
                                    overlap: overlap
                                )
                            }
                        }
                    }
                }
                
                processedTiles += 1

                if config.verboseMemory {
                    print("📊 Processed tile (\(tx+1), \(ty+1)) of (\(numTilesX), \(numTilesY))")
                }
            }
        }

        // Report tiling complete
        config.progressCallback?(ProgressUpdate(
            current: totalTiles - 1,
            total: totalTiles,
            stage: "blending",
            message: "Blending tiles and normalizing"
        ))

        // Normalize by weights
        if useMetalPost, let mp = metalPostProcessor {
            try mp.normalize1CHW(values: depth, weights: weights, width: image.width, height: image.height)
            try mp.normalize1CHW(values: depthConf, weights: weights, width: image.width, height: image.height)
            if includeRays, let rays, let rayConf {
                try mp.normalizeCHW(values: rays, weights: weights, channels: rayChannels, width: image.width, height: image.height)
                try mp.normalize1CHW(values: rayConf, weights: weights, width: image.width, height: image.height)
            }
        } else {
            try normalizeByWeights(depth, weights: weights)
            try normalizeByWeights(depthConf, weights: weights)
            if includeRays, let rays, let rayConf {
                try normalizeRaysByWeights(rays: rays, rayConf: rayConf, weights: weights)
            }
        }

        let inferenceTime = CFAbsoluteTimeGetCurrent() - startTime
        
        return Result(
            depth: depth,
            depthConfidence: depthConf,
            rays: rays,
            rayConfidence: rayConf,
            originalSize: originalSize,
            inferenceTime: inferenceTime
        )
    }
    
    // MARK: - Depth Activation
    
    /// Apply activation function to raw depth output
    private func applyDepthActivation(_ rawDepth: MLMultiArray) throws -> MLMultiArray {
        // Common case: the CoreML head already applied activation (DA3 checkpoints). Avoid
        // double-applying by returning the raw tensor when configured as linear.
        if config.depthActivation == .linear {
            return rawDepth
        }

        let output = try MLMultiArray(shape: rawDepth.shape, dataType: rawDepth.dataType)

        for i in 0..<rawDepth.count {
            let val = rawDepth[i].floatValue
            let activated: Float
            
            switch config.depthActivation {
            case .exp:
                // Prevent float overflow: exp(88) ~ 1.65e38 (near Float.max).
                // Depth heads usually do not require such extreme logits; clamping preserves stability.
                let x = min(max(val, -80.0), 80.0)
                activated = exp(x)
            case .relu:
                activated = max(0, val)
            case .sigmoid:
                // Stable sigmoid:
                // - for large positive val, exp(-val) underflows to 0 (fine)
                // - for large negative val, exp(-val) can overflow; use alternative form
                if val >= 0 {
                    activated = 1.0 / (1.0 + exp(-val))
                } else {
                    let e = exp(val)
                    activated = e / (1.0 + e)
                }
            case .softplus:
                // Stable softplus: max(x,0) + log1p(exp(-abs(x)))
                let ax = abs(val)
                activated = max(val, 0) + log1p(exp(-ax))
            case .linear:
                activated = val
            }
            
            output[i] = NSNumber(value: activated)
        }
        
        return output
    }

    // MARK: - Confidence Activation (Logits -> Positive Weights)

    /// Apply a numerically-stable confidence activation in **float32**.
    ///
    /// Why this exists:
    /// - Official DA3 heads typically embed `conf_activation="expp1"` inside the CoreML graph.
    /// - When exported to float16, the internal `exp()` can overflow and produce NaN/Inf in
    ///   `*_confidence`, especially for the ray branch.
    /// - A robust workaround is to export the head with `conf_activation="linear"` (logits),
    ///   then apply `expp1`/`softplus1` here (float32) with optional logit clamping.
    public static func activateConfidence(
        _ rawConfidence: MLMultiArray,
        activation: ConfidenceActivation,
        clampMin: Float = -30.0,
        clampMax: Float = 30.0
    ) throws -> MLMultiArray {
        if activation == .linear {
            return rawConfidence
        }

        // IMPORTANT: CoreML outputs can have non-standard strides; use a stride-aware reader.
        let logits = (try? MLMultiArrayFloatReader(rawConfidence))?.readAll()
            ?? (0..<rawConfidence.count).map { rawConfidence[$0].floatValue }

        var activated = [Float](repeating: 0, count: logits.count)

        let lo = min(clampMin, clampMax)
        let hi = max(clampMin, clampMax)

        @inline(__always)
        func softplusStable(_ x: Float) -> Float {
            // softplus(x) = max(x,0) + log1p(exp(-abs(x)))  (stable for large |x|)
            let ax = abs(x)
            return max(x, 0) + log1p(exp(-ax))
        }

        for i in 0..<logits.count {
            let x0 = logits[i]
            guard x0.isFinite else {
                activated[i] = 0
                continue
            }
            let x = min(max(x0, lo), hi)

            switch activation {
            case .linear:
                activated[i] = x
            case .expp1:
                activated[i] = exp(x) + 1.0
            case .softplus1:
                activated[i] = softplusStable(x) + 1.0
            }
        }

        // Keep confidence in float32 to avoid re-introducing float16 overflow.
        let shape = rawConfidence.shape
        return try {
            let arr = try MLMultiArray(shape: shape, dataType: .float32)
            let ptr = UnsafeMutablePointer<Float>(OpaquePointer(arr.dataPointer))
            for i in 0..<activated.count { ptr[i] = activated[i] }
            return arr
        }()
    }

    private func applyConfidenceActivation(_ rawConfidence: MLMultiArray, name: String = "confidence") throws -> MLMultiArray {
        // If the head already produced final confidence values, we still need to guard against fp16
        // `exp()` overflow inside the CoreML graph (producing Inf/NaN). Instead of failing the entire
        // inference, sanitize non-finite values to 0 (interpreted as “no weight”) and continue.
        if config.confidenceActivation == .linear {
            if containsNonFinite(rawConfidence, samples: 8192) {
                if config.verboseMemory {
                    print("⚠️ \(name) contains NaN/Inf; replacing non-finite values with 0 (to avoid propagating invalid weights).")
                }
                let shape = rawConfidence.shape
                let out = try MLMultiArray(shape: shape, dataType: .float32)
                let ptr = UnsafeMutablePointer<Float>(OpaquePointer(out.dataPointer))
                let data = (try? MLMultiArrayFloatReader(rawConfidence).readAll())
                    ?? (0..<rawConfidence.count).map { rawConfidence[$0].floatValue }
                for i in 0..<data.count {
                    let v = data[i]
                    ptr[i] = v.isFinite ? v : 0
                }
                return out
            }
            return rawConfidence
        }

        // If Metal postprocess is enabled, keep this activation on GPU in float32 so we don't
        // bounce large tensors through CPU just for a pointwise exp/softplus.
        if config.postprocessBackend == .metal,
           let mp = metalPostProcessor,
           config.confidenceActivation != .linear {
            let dims = depthArrayDims(rawConfidence)
            return try mp.activateConfidence1CHW(
                logits: rawConfidence,
                width: dims.width,
                height: dims.height,
                activation: config.confidenceActivation,
                clampMin: config.confidenceLogitClampMin,
                clampMax: config.confidenceLogitClampMax
            )
        }

        return try DA3CoreML.activateConfidence(
            rawConfidence,
            activation: config.confidenceActivation,
            clampMin: config.confidenceLogitClampMin,
            clampMax: config.confidenceLogitClampMax
        )
    }

    // MARK: - Output Validation

    private func validateFinite(_ array: MLMultiArray?, name: String) throws {
        guard let array else { return }
        if containsNonFinite(array, samples: 8192) {
            throw DA3Error.inferenceError(
                "\(name) contains NaN/Inf. This usually indicates float16 numerical instability in the DualDPT confidence/ray heads (often `exp()` overflow). " +
                "Workarounds: (1) use a float32 head model and/or force the head to CPU-only, or (2) export confidence logits (`conf_activation=linear`) and set `confidenceActivation` in Swift (float32)."
            )
        }
    }

    /// Returns true if the tensor contains NaN or +/-Inf (checked on a stride-aware sample).
    private func containsNonFinite(_ array: MLMultiArray, samples: Int) -> Bool {
        guard let reader = try? MLMultiArrayFloatReader(array) else { return true }
        let count = max(0, array.count)
        if count == 0 { return false }

        // Sample evenly across the logical row-major tensor.
        let targetSamples = max(64, samples)
        let step = max(1, count / targetSamples)
        var i = 0
        while i < count {
            let v = rowMajorValue(reader: reader, linearIndex: i)
            if !v.isFinite { return true }
            i += step
        }
        // Also check the last element to reduce edge-case misses when count % step != 0.
        let vLast = rowMajorValue(reader: reader, linearIndex: count - 1)
        return !vLast.isFinite
    }

    private func rowMajorValue(reader: MLMultiArrayFloatReader, linearIndex: Int) -> Float {
        let shape = reader.shape
        let strides = reader.strides
        var t = linearIndex
        var offset = 0
        for dim in shape.indices.reversed() {
            let size = max(1, shape[dim])
            let idx = t % size
            t /= size
            offset += idx * strides[dim]
        }
        return reader.readLinear(offset)
    }
    
    // MARK: - Tile Blending Helpers
    
    private func blendTile(
        tileDepth: MLMultiArray,
        tileConf: MLMultiArray,
        into output: MLMultiArray,
        confInto confOutput: MLMultiArray,
        weights: MLMultiArray,
        atX: Int,
        atY: Int,
        tileW: Int,
        tileH: Int,
        overlap: Int
    ) throws {
        let outputW = output.shape[2].intValue
        let outputH = output.shape[1].intValue

        // Tile depth/conf shape is [1, tileH, tileW], so we need to account for the leading 1
        let tileShape = tileDepth.shape
        let tileActualW = tileShape.count >= 3 ? tileShape[2].intValue : tileW
        let tileActualH = tileShape.count >= 2 ? tileShape[tileShape.count - 2].intValue : tileH

        // Determine if this tile is at the image boundary
        // Image boundary edges should get full weight (no blending needed there)
        let atLeftEdge = (atX == 0)
        let atRightEdge = (atX + tileW >= outputW)
        let atTopEdge = (atY == 0)
        let atBottomEdge = (atY + tileH >= outputH)

        for y in 0..<min(tileH, tileActualH) {
            for x in 0..<min(tileW, tileActualW) {
                let outX = atX + x
                let outY = atY + y

                // Calculate blend weight based on distance from tile edges
                // BUT only apply ramping on edges that have overlapping tiles
                // Image boundary edges should get full weight
                var distLeft = Float(x)
                var distRight = Float(tileW - 1 - x)
                var distTop = Float(y)
                var distBottom = Float(tileH - 1 - y)

                // At image boundaries, set distance to large value (no weight ramping)
                if atLeftEdge { distLeft = Float(overlap + 1) }
                if atRightEdge { distRight = Float(overlap + 1) }
                if atTopEdge { distTop = Float(overlap + 1) }
                if atBottomEdge { distBottom = Float(overlap + 1) }

                let minDist = min(min(distLeft, distRight), min(distTop, distBottom))
                // Prevent division by zero when overlap is 0
                let weight = overlap > 0 ? min(1.0, minDist / Float(overlap)) : 1.0

                let outIdx = outY * outputW + outX
                // For shape [1, H, W], index is y * W + x (the leading 1 doesn't change linear indexing)
                let tileIdx = y * tileActualW + x

                let depthVal = tileDepth[tileIdx].floatValue
                let confVal = tileConf[tileIdx].floatValue
                let currentWeight = weights[outIdx].floatValue

                // Weighted accumulation
                output[outIdx] = NSNumber(value: output[outIdx].floatValue + depthVal * weight)
                confOutput[outIdx] = NSNumber(value: confOutput[outIdx].floatValue + confVal * weight)
                weights[outIdx] = NSNumber(value: currentWeight + weight)
            }
        }
    }
    
    private func blendTileRays(
        tileRays: MLMultiArray,
        tileConf: MLMultiArray,
        into output: MLMultiArray,
        confInto confOutput: MLMultiArray,
        atX: Int,
        atY: Int,
        tileW: Int,
        tileH: Int,
        overlap: Int
    ) throws {
        let outputDims = rayArrayDims(output)
        let tileDims = rayArrayDims(tileRays)
        let outputW = outputDims.width
        let outputH = outputDims.height
        let channelCount = min(tileDims.channels, outputDims.channels)

        let atLeftEdge = (atX == 0)
        let atRightEdge = (atX + tileW >= outputW)
        let atTopEdge = (atY == 0)
        let atBottomEdge = (atY + tileH >= outputH)
        
        for y in 0..<tileH {
            for x in 0..<tileW {
                let outX = atX + x
                let outY = atY + y
                
                guard outX < outputW && outY < outputH else { continue }

                // Spatial ramp weight (same as `blendTile`).
                var distLeft = Float(x)
                var distRight = Float(tileW - 1 - x)
                var distTop = Float(y)
                var distBottom = Float(tileH - 1 - y)

                let big = Float(overlap + 1)
                if atLeftEdge { distLeft = big }
                if atRightEdge { distRight = big }
                if atTopEdge { distTop = big }
                if atBottomEdge { distBottom = big }

                let minDist = min(min(distLeft, distRight), min(distTop, distBottom))
                let w = overlap > 0 ? min(1.0, minDist / Float(overlap)) : 1.0
                
                for c in 0..<channelCount {
                    let tileIdx = c * tileH * tileW + y * tileW + x
                    let outIdx = c * outputH * outputW + outY * outputW + outX
                    output[outIdx] = NSNumber(value: output[outIdx].floatValue + tileRays[tileIdx].floatValue * w)
                }
                
                let confTileIdx = y * tileW + x
                let confOutIdx = outY * outputW + outX
                confOutput[confOutIdx] = NSNumber(value: confOutput[confOutIdx].floatValue + tileConf[confTileIdx].floatValue * w)
            }
        }
    }
    
    private func normalizeByWeights(_ array: MLMultiArray, weights: MLMultiArray) throws {
        var zeroWeightCount = 0
        for i in 0..<array.count {
            let w = weights[i].floatValue
            if w > 0 {
                array[i] = NSNumber(value: array[i].floatValue / w)
            } else {
                // If weight is 0, this pixel was never covered by any tile
                // Set to 0 (or could interpolate from neighbors in production)
                array[i] = NSNumber(value: Float(0.0))
                zeroWeightCount += 1
            }
        }
        
        // Warn if significant number of pixels had zero coverage
        if zeroWeightCount > 0 && config.verboseMemory {
            let percent = Double(zeroWeightCount) / Double(array.count) * 100
            print("⚠️ Warning: \(zeroWeightCount) pixels (\(String(format: "%.1f", percent))%) had zero tile coverage")
        }
    }

    private func normalizeRaysByWeights(
        rays: MLMultiArray,
        rayConf: MLMultiArray,
        weights: MLMultiArray
    ) throws {
        let dims = rayArrayDims(rays)
        let C = dims.channels
        let H = dims.height
        let W = dims.width
        guard weights.count == H * W else {
            throw DA3Error.invalidShape("weights shape does not match ray grid (\(weights.count) vs \(H * W))")
        }

        for y in 0..<H {
            for x in 0..<W {
                let idx2D = y * W + x
                let w = weights[idx2D].floatValue
                if w > 0 {
                    for c in 0..<C {
                        let idx = c * H * W + idx2D
                        rays[idx] = NSNumber(value: rays[idx].floatValue / w)
                    }
                    rayConf[idx2D] = NSNumber(value: rayConf[idx2D].floatValue / w)
                } else {
                    for c in 0..<C {
                        let idx = c * H * W + idx2D
                        rays[idx] = 0
                    }
                    rayConf[idx2D] = 0
                }
            }
        }
    }

    // MARK: - Postprocessing to original resolution

    private func postprocessPrediction(
        depth: MLMultiArray,
        depthConf: MLMultiArray,
        rays: MLMultiArray?,
        rayConf: MLMultiArray?,
        targetWidth: Int,
        targetHeight: Int,
        preprocessInfo: DINOv3CoreML.PreprocessInfo?
    ) throws -> (depth: MLMultiArray, depthConf: MLMultiArray, rays: MLMultiArray?, rayConf: MLMultiArray?) {
        if config.postprocessBackend == .metal, let mp = metalPostProcessor {
            let depthDims = depthArrayDims(depth)
            let depthRect = scaledCropRect(width: depthDims.width, height: depthDims.height, preprocessInfo: preprocessInfo)
            let depthOut = try mp.resizeCropCHW(
                input: depth,
                channels: depthDims.channels,
                inWidth: depthDims.width,
                inHeight: depthDims.height,
                crop: .init(startX: depthRect.startX, startY: depthRect.startY, width: depthRect.cropW, height: depthRect.cropH),
                outWidth: targetWidth,
                outHeight: targetHeight
            )

            let confDims = depthArrayDims(depthConf)
            let confRect = scaledCropRect(width: confDims.width, height: confDims.height, preprocessInfo: preprocessInfo)
            let depthConfOut = try mp.resizeCropCHW(
                input: depthConf,
                channels: confDims.channels,
                inWidth: confDims.width,
                inHeight: confDims.height,
                crop: .init(startX: confRect.startX, startY: confRect.startY, width: confRect.cropW, height: confRect.cropH),
                outWidth: targetWidth,
                outHeight: targetHeight
            )

            var raysOut: MLMultiArray? = nil
            var rayConfOut: MLMultiArray? = nil

            if let rays, let rayConf {
                let rayDims = rayArrayDims(rays)
                let rayRect = scaledCropRect(width: rayDims.width, height: rayDims.height, preprocessInfo: preprocessInfo)
                raysOut = try mp.resizeCropCHW(
                    input: rays,
                    channels: rayDims.channels,
                    inWidth: rayDims.width,
                    inHeight: rayDims.height,
                    crop: .init(startX: rayRect.startX, startY: rayRect.startY, width: rayRect.cropW, height: rayRect.cropH),
                    outWidth: targetWidth,
                    outHeight: targetHeight
                )

                let rayConfDims = depthArrayDims(rayConf)
                let rayConfRect = scaledCropRect(width: rayConfDims.width, height: rayConfDims.height, preprocessInfo: preprocessInfo)
                rayConfOut = try mp.resizeCropCHW(
                    input: rayConf,
                    channels: rayConfDims.channels,
                    inWidth: rayConfDims.width,
                    inHeight: rayConfDims.height,
                    crop: .init(startX: rayConfRect.startX, startY: rayConfRect.startY, width: rayConfRect.cropW, height: rayConfRect.cropH),
                    outWidth: targetWidth,
                    outHeight: targetHeight
                )
            }

            return (depthOut, depthConfOut, raysOut, rayConfOut)
        }

        let depthDims = depthArrayDims(depth)
        let depthData = readFloatArray(depth)
        let depthConfData = readFloatArray(depthConf)

        let (croppedDepth, cropW, cropH) = cropToValidRegion(
            data: depthData,
            channels: depthDims.channels,
            width: depthDims.width,
            height: depthDims.height,
            preprocessInfo: preprocessInfo
        )

        let (croppedDepthConf, confCropW, confCropH) = cropToValidRegion(
            data: depthConfData,
            channels: 1,
            width: depthDims.width,
            height: depthDims.height,
            preprocessInfo: preprocessInfo
        )

        let resizedDepth = resizeBilinear(
            data: croppedDepth,
            channels: depthDims.channels,
            inWidth: cropW,
            inHeight: cropH,
            outWidth: targetWidth,
            outHeight: targetHeight
        )
        let resizedDepthConf = resizeBilinear(
            data: croppedDepthConf,
            channels: 1,
            inWidth: confCropW,
            inHeight: confCropH,
            outWidth: targetWidth,
            outHeight: targetHeight
        )

        let depthOut = try writeFloatArray(
            resizedDepth,
            shape: [1, NSNumber(value: targetHeight), NSNumber(value: targetWidth)],
            dataType: depth.dataType
        )
        let depthConfOut = try writeFloatArray(
            resizedDepthConf,
            shape: [1, NSNumber(value: targetHeight), NSNumber(value: targetWidth)],
            dataType: depthConf.dataType
        )

        var raysOut: MLMultiArray? = nil
        var rayConfOut: MLMultiArray? = nil

        if let rays = rays, let rayConf = rayConf {
            let rayDims = rayArrayDims(rays)
            let rayData = readFloatArray(rays)
            let rayConfData = readFloatArray(rayConf)

            let (croppedRays, rayCropW, rayCropH) = cropToValidRegion(
                data: rayData,
                channels: rayDims.channels,
                width: rayDims.width,
                height: rayDims.height,
                preprocessInfo: preprocessInfo
            )
            let (croppedRayConf, rayConfCropW, rayConfCropH) = cropToValidRegion(
                data: rayConfData,
                channels: 1,
                width: rayDims.width,
                height: rayDims.height,
                preprocessInfo: preprocessInfo
            )

            let resizedRays = resizeBilinear(
                data: croppedRays,
                channels: rayDims.channels,
                inWidth: rayCropW,
                inHeight: rayCropH,
                outWidth: targetWidth,
                outHeight: targetHeight
            )
            let resizedRayConf = resizeBilinear(
                data: croppedRayConf,
                channels: 1,
                inWidth: rayConfCropW,
                inHeight: rayConfCropH,
                outWidth: targetWidth,
                outHeight: targetHeight
            )

            raysOut = try writeFloatArray(
                resizedRays,
                shape: [NSNumber(value: rayDims.channels), NSNumber(value: targetHeight), NSNumber(value: targetWidth)],
                dataType: rays.dataType
            )
            rayConfOut = try writeFloatArray(
                resizedRayConf,
                shape: [1, NSNumber(value: targetHeight), NSNumber(value: targetWidth)],
                dataType: rayConf.dataType
            )
        }

        return (depthOut, depthConfOut, raysOut, rayConfOut)
    }

    private func depthArrayDims(_ array: MLMultiArray) -> (channels: Int, height: Int, width: Int) {
        let shape = array.shape
        let width = shape.last?.intValue ?? 1
        let height = shape.count >= 2 ? shape[shape.count - 2].intValue : 1
        let channels = shape.count >= 3 ? shape[shape.count - 3].intValue : 1
        return (channels, height, width)
    }

    private func scaledCropRect(
        width: Int,
        height: Int,
        preprocessInfo: DINOv3CoreML.PreprocessInfo?
    ) -> (startX: Int, startY: Int, cropW: Int, cropH: Int) {
        guard let info = preprocessInfo else {
            return (0, 0, width, height)
        }

        // `PreprocessInfo` is defined at the backbone input resolution (typically 518×518).
        // Depth is usually produced at that same resolution, but rays can be produced at a
        // smaller aux resolution (e.g. 296×296). Scale the crop window accordingly.
        let scaleX = Double(width) / Double(max(1, info.inputWidth))
        let scaleY = Double(height) / Double(max(1, info.inputHeight))

        let startX = max(0, min(width, Int(round(Double(info.padLeft) * scaleX))))
        let startY = max(0, min(height, Int(round(Double(info.padTop) * scaleY))))
        let endX = max(0, min(width, Int(round(Double(info.padLeft + info.scaledWidth) * scaleX))))
        let endY = max(0, min(height, Int(round(Double(info.padTop + info.scaledHeight) * scaleY))))

        let cropW = max(0, endX - startX)
        let cropH = max(0, endY - startY)
        if cropW == 0 || cropH == 0 {
            return (0, 0, width, height)
        }

        return (startX, startY, cropW, cropH)
    }

    private func cropToValidRegion(
        data: [Float],
        channels: Int,
        width: Int,
        height: Int,
        preprocessInfo: DINOv3CoreML.PreprocessInfo?
    ) -> (data: [Float], width: Int, height: Int) {
        guard let info = preprocessInfo else {
            return (data, width, height)
        }
        let rect = scaledCropRect(width: width, height: height, preprocessInfo: info)

        var out = [Float](repeating: 0, count: channels * rect.cropW * rect.cropH)
        for c in 0..<channels {
            for y in 0..<rect.cropH {
                for x in 0..<rect.cropW {
                    let srcIdx = c * height * width + (rect.startY + y) * width + (rect.startX + x)
                    let dstIdx = c * rect.cropW * rect.cropH + y * rect.cropW + x
                    out[dstIdx] = data[srcIdx]
                }
            }
        }
        return (out, rect.cropW, rect.cropH)
    }

    private func resizeBilinear(
        data: [Float],
        channels: Int,
        inWidth: Int,
        inHeight: Int,
        outWidth: Int,
        outHeight: Int
    ) -> [Float] {
        if inWidth == outWidth && inHeight == outHeight {
            return data
        }
        var out = [Float](repeating: 0, count: channels * outWidth * outHeight)
        let scaleX = Float(inWidth) / Float(outWidth)
        let scaleY = Float(inHeight) / Float(outHeight)

        for c in 0..<channels {
            for y in 0..<outHeight {
                let gy = (Float(y) + 0.5) * scaleY - 0.5
                let y0 = Int(floor(gy)).clamped(to: 0..<(inHeight))
                let y1 = min(y0 + 1, inHeight - 1)
                // Clamp wy to [0, 1] to prevent extrapolation
                let wy = max(0, min(1, gy - Float(y0)))

                for x in 0..<outWidth {
                    let gx = (Float(x) + 0.5) * scaleX - 0.5
                    let x0 = Int(floor(gx)).clamped(to: 0..<(inWidth))
                    let x1 = min(x0 + 1, inWidth - 1)
                    // Clamp wx to [0, 1] to prevent extrapolation
                    let wx = max(0, min(1, gx - Float(x0)))

                    let cBase = c * inHeight * inWidth
                    let v00 = data[cBase + y0 * inWidth + x0]
                    let v01 = data[cBase + y0 * inWidth + x1]
                    let v10 = data[cBase + y1 * inWidth + x0]
                    let v11 = data[cBase + y1 * inWidth + x1]

                    let top = v00 * (1 - wx) + v01 * wx
                    let bottom = v10 * (1 - wx) + v11 * wx
                    let value = top * (1 - wy) + bottom * wy

                    let outIdx = c * outHeight * outWidth + y * outWidth + x
                    out[outIdx] = value
                }
            }
        }
        return out
    }

    private func readFloatArray(_ array: MLMultiArray) -> [Float] {
        // IMPORTANT: CoreML can return MLMultiArrays with non-standard strides (padding / non-contiguous
        // layouts). Linear indexing (0..<count) can therefore read incorrect values. Use a stride-aware
        // reader when possible and fall back to CoreML's indexing otherwise.
        if let reader = try? MLMultiArrayFloatReader(array) {
            return reader.readAll()
        }
        let count = array.count
        return (0..<count).map { i in array[i].floatValue }
    }

    private func writeFloatArray(
        _ data: [Float],
        shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape, dataType: dataType)
        switch dataType {
        case .float16:
            let ptr = UnsafeMutablePointer<Float16>(OpaquePointer(arr.dataPointer))
            for i in 0..<data.count { ptr[i] = Float16(data[i]) }
        case .float32:
            let ptr = UnsafeMutablePointer<Float>(OpaquePointer(arr.dataPointer))
            for i in 0..<data.count { ptr[i] = data[i] }
        default:
            for i in 0..<data.count { arr[i] = NSNumber(value: data[i]) }
        }
        return arr
    }

    private func rayArrayDims(_ array: MLMultiArray) -> (channels: Int, height: Int, width: Int) {
        let shape = array.shape
        switch shape.count {
        case 4:
            return (shape[1].intValue, shape[2].intValue, shape[3].intValue)
        case 3:
            return (shape[0].intValue, shape[1].intValue, shape[2].intValue)
        default:
            let c = shape.count > 0 ? shape[0].intValue : 0
            let h = shape.count > 1 ? shape[1].intValue : 1
            let w = shape.count > 2 ? shape[2].intValue : 1
            return (c, h, w)
        }
    }

}

private extension Int {
    func clamped(to range: Range<Int>) -> Int {
        return Swift.max(range.lowerBound, Swift.min(self, range.upperBound - 1))
    }
}

// MARK: - Convenience Extensions

@available(macOS 14.0, iOS 17.0, *)
extension DA3CoreML.Result {
    /// Convert depth to a CGImage for visualization
    public func depthAsCGImage(
        colormap: Colormap = .turbo,
        invert: Bool = false,
        style: DepthVisualizationStyle = .depth
    ) throws -> CGImage {
        let width = depth.shape[depth.shape.count - 1].intValue
        let height = depth.shape[depth.shape.count - 2].intValue
        
        var pixels = [UInt8](repeating: 0, count: width * height * 4)

        let (minD, maxD): (Float, Float) = {
            switch style {
            case .depth:
                return depthRange
            case .da3:
                // Match DA3 `visualize_depth()`:
                //   inv = 1/depth for valid pixels, compute 2/98 percentiles on inv,
                //   normalize, then invert so closer = warmer colors.
                let percentile: Float = 2.0
                let maxSamples = 1_000_000
                let count = depth.count
                let step = Swift.max(1, count / maxSamples)

                var samples = [Float]()
                samples.reserveCapacity(Swift.min(maxSamples, count))

                var i = 0
                while i < count {
                    let d = depth[i].floatValue
                    if d > 0 {
                        samples.append(1.0 / d)
                    }
                    i += step
                }

                guard samples.count > 10 else {
                    return (0, 0)
                }

                samples.sort()

                let n = samples.count
                let p = percentile / 100.0
                let loIndex = Int((Float(n - 1) * p).rounded(.down)).clamped(to: 0..<n)
                let hiIndex = Int((Float(n - 1) * (1.0 - p)).rounded(.down)).clamped(to: 0..<n)
                var lo = samples[loIndex]
                var hi = samples[hiIndex]
                if lo == hi {
                    lo -= 1e-6
                    hi += 1e-6
                }
                return (lo, hi)
            }
        }()

        let range = maxD - minD
        
        for y in 0..<height {
            for x in 0..<width {
                let idx = y * width + x
                let depthVal = depth[idx].floatValue
                let base: Float = {
                    switch style {
                    case .depth:
                        return range > 0 ? (depthVal - minD) / range : 0.5
                    case .da3:
                        let inv = depthVal > 0 ? (1.0 / depthVal) : 0.0
                        let t = range > 0 ? ((inv - minD) / range) : 0.5
                        // DA3 inverts after normalization.
                        return 1.0 - max(0, min(1, t))
                    }
                }()
                let normalized = invert ? (1.0 - base) : base
                
                let (r, g, b) = colormap.color(for: normalized)
                let pixelIdx = idx * 4
                pixels[pixelIdx] = UInt8(r * 255)
                pixels[pixelIdx + 1] = UInt8(g * 255)
                pixels[pixelIdx + 2] = UInt8(b * 255)
                pixels[pixelIdx + 3] = 255
            }
        }
        
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let cgImage = CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: width * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              ) else {
            throw DA3Error.imageProcessingFailed("Failed to create depth visualization image")
        }
        
        return cgImage
    }
}

/// Colormaps for depth visualization
public enum Colormap {
    case turbo
    case viridis
    case plasma
    case magma
    case grayscale
    case spectral
    
    /// Get RGB color for normalized value [0, 1]
    func color(for value: Float) -> (r: Float, g: Float, b: Float) {
        let v = max(0, min(1, value))
        
        switch self {
        case .turbo:
            return turboColormap(v)
        case .viridis:
            return viridisColormap(v)
        case .plasma:
            return plasmaColormap(v)
        case .magma:
            return magmaColormap(v)
        case .grayscale:
            return (v, v, v)
        case .spectral:
            return spectralColormap(v)
        }
    }
    
    private func turboColormap(_ t: Float) -> (r: Float, g: Float, b: Float) {
        // Simplified turbo colormap
        let r = max(0, min(1, 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))))))
        let g = max(0, min(1, 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))))))
        let b = max(0, min(1, 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))))))
        return (r, g, b)
    }
    
    private func viridisColormap(_ t: Float) -> (r: Float, g: Float, b: Float) {
        let r = max(0, min(1, 0.267004 + t * (0.282327 + t * (-1.117651 + t * (2.168057 + t * (-1.599767))))))
        let g = max(0, min(1, 0.004874 + t * (1.316386 + t * (-0.441546 + t * (0.095545)))))
        let b = max(0, min(1, 0.329415 + t * (0.770914 + t * (-2.324501 + t * (2.244595 + t * (-0.678886))))))
        return (r, g, b)
    }
    
    private func plasmaColormap(_ t: Float) -> (r: Float, g: Float, b: Float) {
        let r = max(0, min(1, 0.050383 + t * (2.028879 + t * (-2.110679 + t * 1.037227))))
        let g = max(0, min(1, 0.029803 + t * (0.261242 + t * (1.448178 + t * (-0.755424)))))
        let b = max(0, min(1, 0.527975 + t * (-0.317663 + t * (0.921337 + t * (-1.067911 + t * 0.436665)))))
        return (r, g, b)
    }
    
    private func magmaColormap(_ t: Float) -> (r: Float, g: Float, b: Float) {
        let r = max(0, min(1, 0.001462 + t * (0.817394 + t * (1.503403 + t * (-1.347021)))))
        let g = max(0, min(1, 0.000466 + t * (0.107776 + t * (0.871498 + t * 0.020573))))
        let b = max(0, min(1, 0.013866 + t * (1.015398 + t * (-0.672433 + t * 0.643573))))
        return (r, g, b)
    }

    private func spectralColormap(_ t: Float) -> (r: Float, g: Float, b: Float) {
        // ColorBrewer Spectral (11) control points (Matplotlib's "Spectral" family).
        // Low values are red, high values are blue.
        let stops: [(Float, Float, Float)] = [
            (158 / 255.0, 1 / 255.0, 66 / 255.0),
            (213 / 255.0, 62 / 255.0, 79 / 255.0),
            (244 / 255.0, 109 / 255.0, 67 / 255.0),
            (253 / 255.0, 174 / 255.0, 97 / 255.0),
            (254 / 255.0, 224 / 255.0, 139 / 255.0),
            (255 / 255.0, 255 / 255.0, 191 / 255.0),
            (230 / 255.0, 245 / 255.0, 152 / 255.0),
            (171 / 255.0, 221 / 255.0, 164 / 255.0),
            (102 / 255.0, 194 / 255.0, 165 / 255.0),
            (50 / 255.0, 136 / 255.0, 189 / 255.0),
            (94 / 255.0, 79 / 255.0, 162 / 255.0),
        ]

        let scaled = max(0, min(1, t)) * Float(stops.count - 1)
        let i0 = Int(scaled).clamped(to: 0..<stops.count)
        let i1 = (i0 + 1).clamped(to: 0..<stops.count)
        let f = scaled - Float(i0)

        let (r0, g0, b0) = stops[i0]
        let (r1, g1, b1) = stops[i1]
        return (r0 + (r1 - r0) * f, g0 + (g1 - g0) * f, b0 + (b1 - b0) * f)
    }
}

/// Depth visualization mapping conventions.
///
/// - `depth`: visualize raw depth (larger = farther) normalized by min/max.
/// - `da3`: match Depth-Anything-3 `visualize_depth()` (inverse-depth percentile scaling, closer = warmer).
public enum DepthVisualizationStyle: String {
    case depth
    case da3
}
