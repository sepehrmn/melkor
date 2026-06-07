import Foundation
import CoreML

/// GS (Gaussian Splatting) head wrapper for CoreML inference.
///
/// This class wraps the GS head CoreML model that outputs 38-channel Gaussian
/// splatting parameters from backbone features + RGB image.
///
/// The 38 output channels are:
/// - 2: offset_xy (pixel offset for Gaussian center)
/// - 3: scales (Gaussian scales in 3D)
/// - 4: quaternion (rotation as normalized quaternion; **official DA3 uses xyzw order** due to a historical quirk)
/// - 27: SH coefficients (3 colors * 9 SH basis for degree 2)
/// - 1: offset_depth (depth offset from predicted depth)
/// - 1: confidence/opacity
///
/// Usage:
/// ```swift
/// let gsHead = try GSHeadCoreML(modelPath: "gshead_giant.mlmodelc")
/// let gsParams = try gsHead.predict(from: features, image: pixelValues)
/// ```
@available(macOS 14.0, iOS 17.0, *)
public final class GSHeadCoreML {

    // MARK: - Types

    /// Gaussian splatting parameters for a single image
    public struct GSParams {
        /// Raw GS parameters - shape: (1, 38, H, W)
        public let raw: MLMultiArray

        /// Output height
        public let height: Int
        /// Output width
        public let width: Int

        /// Parameter channel indices
        public enum Channel {
            public static let offsetXY = 0..<2
            public static let scales = 2..<5
            /// Quaternion channels in **xyzw** order (x,y,z,w) in the official DA3 export.
            public static let quaternion = 5..<9
            public static let shCoeffs = 9..<36  // 27 channels for SH degree 2
            public static let offsetDepth = 36
            public static let confidence = 37
        }
    }

    /// Configuration for GS head
    public struct Config {
        /// Input hidden dimension (3072 for giant with cat_token)
        public var dimIn: Int = 3072
        /// Patch size
        public var patchSize: Int = 14
        /// Output channels (38 for full GS params)
        public var gsOutDim: Int = 38
        /// Use GPU if available
        public var useGPU: Bool = true

        public init() {}
    }

    // MARK: - Properties

    private let model: MLModel
    public let config: Config

    // MARK: - Initialization

    /// Initialize with a CoreML model path
    public init(modelPath: String, config: Config = Config()) throws {
        self.config = config

        let url = URL(fileURLWithPath: modelPath)
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = config.useGPU ? .all : .cpuOnly

        self.model = try MLModel(contentsOf: url, configuration: modelConfig)
    }

    /// Initialize with a CoreML model URL
    public init(modelURL: URL, config: Config = Config()) throws {
        self.config = config

        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = config.useGPU ? .all : .cpuOnly

        self.model = try MLModel(contentsOf: modelURL, configuration: modelConfig)
    }

    // MARK: - Prediction

    /// Run GS head on backbone features and RGB image
    ///
    /// - Parameters:
    ///   - features: Multi-scale features from DINOv3 backbone
    ///   - image: RGB image tensor - shape: (1, 3, H, W)
    /// - Returns: Gaussian splatting parameters
    public func predict(from features: DINOv3CoreML.Features, image: MLMultiArray) throws -> GSParams {
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "features_layer5": MLFeatureValue(multiArray: features.layer5),
            "features_layer7": MLFeatureValue(multiArray: features.layer7),
            "features_layer9": MLFeatureValue(multiArray: features.layer9),
            "features_layer11": MLFeatureValue(multiArray: features.layer11),
            "image": MLFeatureValue(multiArray: image),
        ])

        let output = try model.prediction(from: input)

        guard let gsParams = output.featureValue(for: "gs_params")?.multiArrayValue else {
            throw DA3Error.modelOutputMissing("Missing gs_params output from GS head model")
        }

        // Extract dimensions from output shape
        let shape = gsParams.shape
        let height = shape.count >= 3 ? shape[shape.count - 2].intValue : 518
        let width = shape.count >= 4 ? shape[shape.count - 1].intValue : 518

        return GSParams(raw: gsParams, height: height, width: width)
    }

    /// Extract Gaussian parameters at a specific pixel
    public func extractGaussianAt(params: GSParams, x: Int, y: Int) -> GaussianPoint {
        let raw = params.raw
        let w = params.width
        let h = params.height

        func getValue(_ channel: Int) -> Float {
            let idx = channel * h * w + y * w + x
            return raw[idx].floatValue
        }

        return GaussianPoint(
            offsetX: getValue(0),
            offsetY: getValue(1),
            scaleX: getValue(2),
            scaleY: getValue(3),
            scaleZ: getValue(4),
            // Official DA3 convention: quaternion is stored as xyzw (real part last).
            quatX: getValue(5),
            quatY: getValue(6),
            quatZ: getValue(7),
            quatW: getValue(8),
            sh: (0..<27).map { getValue(9 + $0) },
            offsetDepth: getValue(36),
            confidence: getValue(37)
        )
    }
}

/// A single Gaussian point with all parameters
public struct GaussianPoint {
    /// Pixel offset X
    public let offsetX: Float
    /// Pixel offset Y
    public let offsetY: Float
    /// Scale X
    public let scaleX: Float
    /// Scale Y
    public let scaleY: Float
    /// Scale Z
    public let scaleZ: Float
    /// Quaternion X
    public let quatX: Float
    /// Quaternion Y
    public let quatY: Float
    /// Quaternion Z
    public let quatZ: Float
    /// Quaternion W (scalar; **real part last** in DA3 xyzw convention)
    public let quatW: Float
    /// Spherical harmonics coefficients (27 values for degree 2)
    public let sh: [Float]
    /// Depth offset
    public let offsetDepth: Float
    /// Confidence/opacity
    public let confidence: Float

    /// Normalized quaternion
    public var normalizedQuaternion: (w: Float, x: Float, y: Float, z: Float) {
        let len = sqrt(quatW*quatW + quatX*quatX + quatY*quatY + quatZ*quatZ)
        guard len > 0 else { return (1, 0, 0, 0) }
        return (quatW/len, quatX/len, quatY/len, quatZ/len)
    }

    /// RGB color from DC component of spherical harmonics
    public var baseColor: (r: Float, g: Float, b: Float) {
        // DC coefficient (l=0, m=0) is at index 0, 9, 18 for R, G, B
        // SH coefficient to color: (sh + 0.5) clamped to [0, 1]
        let r = max(0, min(1, sh[0] + 0.5))
        let g = max(0, min(1, sh[9] + 0.5))
        let b = max(0, min(1, sh[18] + 0.5))
        return (r, g, b)
    }
}
