import Foundation
import CoreML
import CoreGraphics

/// DINOv3 Vision Transformer wrapper for CoreML inference.
///
/// This class wraps a DINOv3 CoreML model and provides:
/// - Multi-scale feature extraction (layers 5, 7, 9, 11)
/// - Deterministic preprocessing (sRGB decode + ImageNet normalization)
/// - Float32 I/O (matches the compiled backbones in this repo)
///
/// Usage:
/// ```swift
/// let dino = try DINOv3CoreML(modelPath: "dinov3.mlpackage")
/// let features = try dino.extractFeatures(from: image)
/// ```
@available(macOS 14.0, iOS 17.0, *)
public final class DINOv3CoreML {

    private static let sRGBColorSpace: CGColorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    
    // MARK: - Types
    
    /// Multi-scale features extracted from DINOv3
    public struct Features {
        /// Features from layer 5 - shape: (B, numPatches, hiddenDim)
        public let layer5: MLMultiArray
        /// Features from layer 7 - shape: (B, numPatches, hiddenDim)
        public let layer7: MLMultiArray
        /// Features from layer 9 - shape: (B, numPatches, hiddenDim)
        public let layer9: MLMultiArray
        /// Features from layer 11 - shape: (B, numPatches, hiddenDim)
        public let layer11: MLMultiArray
        
        /// All features as array for iteration
        public var all: [MLMultiArray] {
            [layer5, layer7, layer9, layer11]
        }
    }
    
    /// Configuration for DINOv3 model
    public struct Config {
        /// Input image size (default: 518 for DA3)
        public var inputSize: Int = 518
        /// Patch size (default: 14 for DINOv2, 16 for DINOv3)
        public var patchSize: Int = 14
        /// Hidden dimension
        public var hiddenDim: Int = 768
        /// Number of register tokens (DINOv3 has 4)
        public var numRegisterTokens: Int = 4
        /// Maximum batch size for memory efficiency
        public var maxBatchSize: Int = 4
        /// Use GPU if available
        public var useGPU: Bool = true
        /// Prefer Neural Engine over GPU when possible
        /// When true, uses .cpuAndNeuralEngine; when false with useGPU=true, uses .all
        public var preferNeuralEngine: Bool = false

        /// Number of patches for given input size
        public var numPatches: Int {
            let gridSize = inputSize / patchSize
            return gridSize * gridSize
        }

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

        public init() {}
    }

    /// Lightweight metadata read from the mlmodel to auto-configure patch size and registers
    public struct BackboneMetadata {
        public let patchSize: Int?
        public let registerTokens: Int?

        public static func load(fromPath path: String) -> BackboneMetadata? {
            return load(fromURL: URL(fileURLWithPath: path))
        }

        public static func load(fromURL url: URL) -> BackboneMetadata? {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuOnly  // metadata read only; keep light
            guard let model = try? MLModel(contentsOf: url, configuration: cfg) else { return nil }
            guard let meta = model.modelDescription.metadata[.creatorDefinedKey] as? [String: String] else { return nil }
            let patchSize = meta["patch_size"].flatMap { Int($0) }
            let registerTokens = meta["register_tokens"].flatMap { Int($0) }
            return BackboneMetadata(patchSize: patchSize, registerTokens: registerTokens)
        }
    }

    /// Metadata describing how an image was preprocessed before inference.
    public struct PreprocessInfo {
        public let scaledWidth: Int
        public let scaledHeight: Int
        public let padLeft: Int
        public let padTop: Int
        public let inputWidth: Int
        public let inputHeight: Int
        public let scale: Double
    }
    
    // MARK: - Properties
    
    private let model: MLModel
    public let config: Config
    /// Metadata for the most recent call to `extractFeatures(from image:)`.
    public private(set) var lastPreprocessInfo: PreprocessInfo?
    
    // MARK: - Initialization
    
    /// Initialize with a CoreML model path
    public init(modelPath: String, config: Config = Config()) throws {
        self.config = config

        let url = URL(fileURLWithPath: modelPath)
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = config.computeUnits

        self.model = try MLModel(contentsOf: url, configuration: modelConfig)
    }

    /// Initialize with a compiled CoreML model URL
    public init(modelURL: URL, config: Config = Config()) throws {
        self.config = config

        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = config.computeUnits

        self.model = try MLModel(contentsOf: modelURL, configuration: modelConfig)
    }
    
    // MARK: - Feature Extraction
    
    /// Extract multi-scale features from an image
    ///
    /// - Parameter pixelValues: Input tensor of shape (B, 3, H, W) normalized to [-1, 1] or [0, 1]
    /// - Returns: Multi-scale features from layers 5, 7, 9, 11
    public func extractFeatures(from pixelValues: MLMultiArray) throws -> Features {
        // The compiled backbones in this repo are exported with `pixel_values` as Float32.
        // Passing Float16 can lead to silent misinterpretation on some CoreML versions, so
        // always upcast to Float32 at the boundary.
        let pv32 = try MLMultiArrayCast.toFloat32(pixelValues)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": MLFeatureValue(multiArray: pv32)
        ])
        
        let output = try model.prediction(from: input)
        
        guard let layer5 = output.featureValue(for: "features_layer5")?.multiArrayValue,
              let layer7 = output.featureValue(for: "features_layer7")?.multiArrayValue,
              let layer9 = output.featureValue(for: "features_layer9")?.multiArrayValue,
              let layer11 = output.featureValue(for: "features_layer11")?.multiArrayValue else {
            throw DA3Error.modelOutputMissing("Missing feature outputs from DINOv3 model")
        }

        // The exported backbones emit Float16 features, but downstream heads (DualDPT/GSHead/CamDec)
        // are exported to take Float32. Cast here once so all consumers receive the expected type.
        let l5 = try MLMultiArrayCast.toFloat32(layer5)
        let l7 = try MLMultiArrayCast.toFloat32(layer7)
        let l9 = try MLMultiArrayCast.toFloat32(layer9)
        let l11 = try MLMultiArrayCast.toFloat32(layer11)

        return Features(layer5: l5, layer7: l7, layer9: l9, layer11: l11)
    }
    
    /// Extract features from a CGImage
    ///
    /// - Parameters:
    ///   - image: Input CGImage
    ///   - normalize: Whether to normalize to ImageNet stats
    /// - Returns: Multi-scale features
    public func extractFeatures(from image: CGImage, normalize: Bool = true) throws -> Features {
        let (pixelValues, info) = try preprocessImage(image, normalize: normalize)
        self.lastPreprocessInfo = info
        return try extractFeatures(from: pixelValues)
    }

    /// Preprocess an image into the model input tensor (CHW float32) and return the associated info.
    public func preprocess(image: CGImage, normalize: Bool = true) throws -> (pixelValues: MLMultiArray, info: PreprocessInfo) {
        let (pixelValues, info) = try preprocessImage(image, normalize: normalize)
        self.lastPreprocessInfo = info
        return (pixelValues, info)
    }

    /// Convenience helper to run preprocessing once and return both pixelValues and extracted features.
    public func extractFeaturesAndPixels(from image: CGImage, normalize: Bool = true) throws -> (features: Features, pixelValues: MLMultiArray, info: PreprocessInfo) {
        let (pixelValues, info) = try preprocessImage(image, normalize: normalize)
        self.lastPreprocessInfo = info
        let features = try extractFeatures(from: pixelValues)
        return (features, pixelValues, info)
    }
    
    // MARK: - Image Preprocessing
    
    /// Preprocess a single image for DINOv3
    private func preprocessImage(_ image: CGImage, normalize: Bool) throws -> (MLMultiArray, PreprocessInfo) {
        let inputW = config.inputSize
        let inputH = config.inputSize
        let origW = image.width
        let origH = image.height

        // IMPORTANT: The exported CoreML backbone/head are traced for a fixed square input (e.g. 518×518).
        // For best correspondence with the trained checkpoint, do a direct resize to the fixed input size.
        // Letterboxing (aspect-ratio preserving + padding) changes the content distribution and tends to
        // produce "blobby"/misaligned depth for wide/tall images.
        let scale = min(Double(inputW) / Double(origW), Double(inputH) / Double(origH))
        let scaledW = inputW
        let scaledH = inputH
        let padLeft = 0
        let padTop = 0

        // Create output array: (1, 3, H, W)
        let shape: [NSNumber] = [1, 3, NSNumber(value: inputH), NSNumber(value: inputW)]
        // Backbone input expects Float32.
        let pixelValues = try MLMultiArray(shape: shape, dataType: .float32)

        // Resize + decode to a well-defined RGBA8 buffer (sRGB) and convert to CHW float array.
        // This avoids relying on `CGImage.dataProvider` layouts (which can be inconsistent for
        // some wide-gamut JPEGs) and prevents the “all images produce identical depth” failure mode.
        let rgba = try renderRGBA8(image, width: inputW, height: inputH)
        try rgbaBytesToFloatArray(
            rgba,
            width: inputW,
            height: inputH,
            output: pixelValues,
            batchIndex: 0,
            normalize: normalize,
            offsetX: padLeft,
            offsetY: padTop
        )

        let info = PreprocessInfo(
            scaledWidth: scaledW,
            scaledHeight: scaledH,
            padLeft: padLeft,
            padTop: padTop,
            inputWidth: inputW,
            inputHeight: inputH,
            scale: scale
        )

        return (pixelValues, info)
    }

    /// Render an image to a tightly-packed RGBA8 buffer in sRGB space.
    private func renderRGBA8(_ image: CGImage, width: Int, height: Int) throws -> [UInt8] {
        var buffer = [UInt8](repeating: 0, count: width * height * 4)
        let rect = CGRect(x: 0, y: 0, width: width, height: height)

        // Prefer a CGBitmapContext draw: CoreImage rendering can fail silently for some
        // source formats and yield an all-zero buffer, which then makes the model output
        // identical for every input (the “blank input” failure mode).
        let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue
        try buffer.withUnsafeMutableBytes { rawBuf in
            guard let base = rawBuf.baseAddress else {
                throw DA3Error.imageProcessingFailed("Failed to allocate RGBA buffer")
            }
            guard let ctx = CGContext(
                data: base,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: width * 4,
                space: Self.sRGBColorSpace,
                bitmapInfo: bitmapInfo
            ) else {
                throw DA3Error.imageProcessingFailed("Failed to create bitmap context")
            }

            ctx.interpolationQuality = .high
            ctx.draw(image, in: rect)
        }

        return buffer
    }

    /// Convert RGBA8 bytes (row-major) into CHW float tensor with optional ImageNet normalization.
    private func rgbaBytesToFloatArray(
        _ rgba: [UInt8],
        width: Int,
        height: Int,
        output: MLMultiArray,
        batchIndex: Int,
        normalize: Bool,
        offsetX: Int = 0,
        offsetY: Int = 0
    ) throws {
        // ImageNet normalization stats
        let mean: [Float] = normalize ? [0.485, 0.456, 0.406] : [0, 0, 0]
        let std: [Float] = normalize ? [0.229, 0.224, 0.225] : [1, 1, 1]

        let bytesPerRow = width * 4

        // Fast path: write directly into the MLMultiArray backing store using strides.
        // This avoids per-element NSNumber boxing (which is extremely slow) while still
        // being correct for non-standard strides.
        let shape = output.shape.map { $0.intValue }
        let strides = output.strides.map { $0.intValue }
        let canUseStrides = (shape.count == 4 && strides.count == 4 && shape[1] >= 3)

        try rgba.withUnsafeBytes { rawBuf in
            guard let bytes = rawBuf.bindMemory(to: UInt8.self).baseAddress else {
                throw DA3Error.imageProcessingFailed("Failed to access RGBA buffer")
            }

            if canUseStrides {
                let sb = strides[0]
                let sc = strides[1]
                let sy = strides[2]
                let sx = strides[3]

                switch output.dataType {
                case .float32:
                    let dst = UnsafeMutablePointer<Float>(OpaquePointer(output.dataPointer))
                    for c in 0..<3 {
                        let baseBC = batchIndex * sb + c * sc
                        for y in 0..<height {
                            let rowBase = baseBC + (y + offsetY) * sy + offsetX * sx
                            let srcRow = y * bytesPerRow
                            for x in 0..<width {
                                let pixelOffset = srcRow + x * 4
                                let pixelValue = Float(bytes[pixelOffset + c]) / 255.0
                                let normalizedValue = (pixelValue - mean[c]) / std[c]
                                dst[rowBase + x * sx] = normalizedValue
                            }
                        }
                    }
                    return
                case .float16:
                    let dst = UnsafeMutablePointer<Float16>(OpaquePointer(output.dataPointer))
                    for c in 0..<3 {
                        let baseBC = batchIndex * sb + c * sc
                        for y in 0..<height {
                            let rowBase = baseBC + (y + offsetY) * sy + offsetX * sx
                            let srcRow = y * bytesPerRow
                            for x in 0..<width {
                                let pixelOffset = srcRow + x * 4
                                let pixelValue = Float(bytes[pixelOffset + c]) / 255.0
                                let normalizedValue = (pixelValue - mean[c]) / std[c]
                                dst[rowBase + x * sx] = Float16(normalizedValue)
                            }
                        }
                    }
                    return
                default:
                    break
                }
            }

            // Fallback: safe but slow path using multi-index subscripting.
            for c in 0..<3 {
                for y in 0..<height {
                    for x in 0..<width {
                        let pixelOffset = y * bytesPerRow + x * 4
                        let pixelValue = Float(bytes[pixelOffset + c]) / 255.0
                        let normalizedValue = (pixelValue - mean[c]) / std[c]

                        let index = [
                            NSNumber(value: batchIndex),
                            NSNumber(value: c),
                            NSNumber(value: y + offsetY),
                            NSNumber(value: x + offsetX),
                        ]
                        output[index] = NSNumber(value: normalizedValue)
                    }
                }
            }
        }
    }
}
