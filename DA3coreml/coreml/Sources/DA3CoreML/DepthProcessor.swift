import Foundation
import CoreML

/// DualDPT (Dual Dense Prediction Transformer) head for depth and ray prediction.
///
/// This implements the DualDPT architecture from Depth-Anything-3:
/// - Takes multi-scale features from DINOv3 backbone (layers 5, 7, 9, 11)
/// - Processes through two separate fusion pathways (depth and ray)
/// - Outputs depth map with confidence and ray directions with confidence
///
/// The architecture follows a top-down feature pyramid with:
/// - Per-stage projections from backbone hidden dim to internal features
/// - Spatial alignment to common resolution
/// - Separate refinement chains for depth and ray
@available(macOS 14.0, iOS 17.0, *)
public final class DualDPTCoreML {
    
    // MARK: - Types
    
    /// Output from DualDPT head
    public struct Prediction {
        /// Depth prediction - shape: (B, 1, H, W)
        public let depth: MLMultiArray
        /// Depth confidence - shape: (B, 1, H, W)
        public let depthConfidence: MLMultiArray
        /// Ray directions - shape: (B, 3, H, W) or (B, 6, H, W)
        public let rays: MLMultiArray
        /// Ray confidence - shape: (B, 1, H, W)
        public let rayConfidence: MLMultiArray
    }
    
    /// Configuration for DualDPT head
    public struct Config {
        /// Input hidden dimension from backbone
        public var dimIn: Int = 768
        /// Patch size from backbone
        public var patchSize: Int = 14
        /// Internal feature dimension
        public var features: Int = 256
        /// Depth output channels (1 for depth + 1 for confidence)
        public var depthOutputDim: Int = 2
        /// Ray output channels (6 for ray + 1 for confidence)
        public var rayOutputDim: Int = 7
        /// Activation for depth (exp, relu, sigmoid, softplus)
        public var depthActivation: String = "exp"
        /// Activation for ray (none/linear)
        public var rayActivation: String = "linear"
        /// Use GPU if available
        public var useGPU: Bool = true
        /// Prefer Neural Engine over GPU when possible
        /// When true, uses .cpuAndNeuralEngine; when false with useGPU=true, uses .all
        public var preferNeuralEngine: Bool = false

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
    
    // MARK: - Properties
    
    private let model: MLModel
    public let config: Config
    
    // MARK: - Initialization
    
    /// Initialize with a CoreML model path
    public init(modelPath: String, config: Config = Config()) throws {
        self.config = config

        let url = URL(fileURLWithPath: modelPath)
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = config.computeUnits

        self.model = try MLModel(contentsOf: url, configuration: modelConfig)
    }

    /// Initialize with a CoreML model URL
    public init(modelURL: URL, config: Config = Config()) throws {
        self.config = config

        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = config.computeUnits

        self.model = try MLModel(contentsOf: modelURL, configuration: modelConfig)
    }
    
    // MARK: - Prediction
    
    /// Run DualDPT head on backbone features
    ///
    /// - Parameter features: Multi-scale features from DINOv3 backbone
    /// - Returns: Depth and ray predictions with confidence
    public func predict(from features: DINOv3CoreML.Features) throws -> Prediction {
        // DualDPT CoreML models in this repo are exported with Float32 feature inputs.
        let f5 = try MLMultiArrayCast.toFloat32(features.layer5)
        let f7 = try MLMultiArrayCast.toFloat32(features.layer7)
        let f9 = try MLMultiArrayCast.toFloat32(features.layer9)
        let f11 = try MLMultiArrayCast.toFloat32(features.layer11)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "features_layer5": MLFeatureValue(multiArray: f5),
            "features_layer7": MLFeatureValue(multiArray: f7),
            "features_layer9": MLFeatureValue(multiArray: f9),
            "features_layer11": MLFeatureValue(multiArray: f11),
        ])
        
        let output = try model.prediction(from: input)
        
        guard let depth = output.featureValue(for: "depth")?.multiArrayValue,
              let depthConf = output.featureValue(for: "depth_confidence")?.multiArrayValue,
              let rays = output.featureValue(for: "rays")?.multiArrayValue,
              let rayConf = output.featureValue(for: "ray_confidence")?.multiArrayValue else {
            throw DA3Error.modelOutputMissing("Missing outputs from DualDPT model")
        }
        
        return Prediction(
            depth: depth,
            depthConfidence: depthConf,
            rays: rays,
            rayConfidence: rayConf
        )
    }
}
