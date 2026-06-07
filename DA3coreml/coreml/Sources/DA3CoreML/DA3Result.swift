import Foundation
import CoreML
import simd

/// Camera decoder wrapper (DA3 cam_dec) for pose estimation.
@available(macOS 14.0, iOS 17.0, *)
public final class CamDecCoreML {

    public struct Config {
        public var dimIn: Int = 768
        public var numTokens: Int = 1024  // 32x32 (patch 16, 518 input)
        public var useGPU: Bool = true
        public init() {}
    }

    private let model: MLModel
    public let config: Config

    public init(modelPath: String, config: Config = Config()) throws {
        self.config = config
        let url = URL(fileURLWithPath: modelPath)
        let mc = MLModelConfiguration()
        mc.computeUnits = config.useGPU ? .all : .cpuOnly
        self.model = try MLModel(contentsOf: url, configuration: mc)
    }

    /// Predict pose encoding (B, N, 9) from layer11 features (B, N, D)
    public func predictPose(from features: MLMultiArray) throws -> MLMultiArray {
        let f32 = try MLMultiArrayCast.toFloat32(features)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "features": MLFeatureValue(multiArray: f32)
        ])
        let out = try model.prediction(from: input)
        guard let pose = out.featureValue(for: "pose_enc")?.multiArrayValue else {
            throw DA3Error.modelOutputMissing("pose_enc")
        }
        return pose
    }

    /// Reduce pose enc over tokens -> single pose, then convert to extrinsics/intrinsics.
    public func decodePose(
        poseEnc: MLMultiArray,
        imageWidth: Int,
        imageHeight: Int
    ) -> (c2w: simd_float4x4, intrinsics: simd_float3x3) {
        // poseEnc shape: [1, N, 9] or [N, 9]
        // NOTE: CoreML can return float16 + non-contiguous strides. Always read stride-aware.
        let reader = try? MLMultiArrayFloatReader(poseEnc)
        let shape = reader?.shape ?? poseEnc.shape.map { $0.intValue }
        let hasBatch = shape.count == 3
        let N = hasBatch ? shape[1] : (shape.count == 2 ? shape[0] : 0)

        guard N > 0 else {
            // Fallback: identity pose + ~50° FOV intrinsics.
            let H = Float(imageHeight)
            let W = Float(imageWidth)
            let fov: Float = 50.0 * .pi / 180.0
            let fy = (H / 2.0) / max(1e-6, tan(fov / 2.0))
            let fx = (W / 2.0) / max(1e-6, tan(fov / 2.0))
            let cx = W / 2.0
            let cy = H / 2.0
            let K = simd_float3x3(rows: [
                simd_float3(fx, 0, cx),
                simd_float3(0, fy, cy),
                simd_float3(0, 0, 1)
            ])
            return (matrix_identity_float4x4, K)
        }

        var t = simd_float3(0, 0, 0)
        var q = simd_float4(0, 0, 0, 1) // xyzw
        var fovh: Float = 0
        var fovw: Float = 0

        if let r = reader, N > 0 {
            for i in 0..<N {
                if hasBatch {
                    t += simd_float3(r.read(0, i, 0), r.read(0, i, 1), r.read(0, i, 2))
                    q += simd_float4(r.read(0, i, 3), r.read(0, i, 4), r.read(0, i, 5), r.read(0, i, 6))
                    fovh += r.read(0, i, 7)
                    fovw += r.read(0, i, 8)
                } else {
                    t += simd_float3(r.read(i, 0), r.read(i, 1), r.read(i, 2))
                    q += simd_float4(r.read(i, 3), r.read(i, 4), r.read(i, 5), r.read(i, 6))
                    fovh += r.read(i, 7)
                    fovw += r.read(i, 8)
                }
            }
        }

        let invN = 1.0 / Float(N)
        t *= invN; q *= invN; fovh *= invN; fovw *= invN
        // Normalize quaternion (xyzw)
        let qnorm = simd_length(q)
        let qn = qnorm > 0 ? q / qnorm : simd_float4(0,0,0,1)
        let R = quatToMat(qn)

        // c2w matrix
        var c2w = simd_float4x4(1)
        c2w.columns.0 = simd_float4(R.columns.0, 0)
        c2w.columns.1 = simd_float4(R.columns.1, 0)
        c2w.columns.2 = simd_float4(R.columns.2, 0)
        c2w.columns.3 = simd_float4(t, 1)

        // intrinsics from fov
        let H = Float(imageHeight)
        let W = Float(imageWidth)
        let fy = (H / 2.0) / max(1e-6, tan(fovh / 2.0))
        let fx = (W / 2.0) / max(1e-6, tan(fovw / 2.0))
        let cx = W / 2.0
        let cy = H / 2.0
        let K = simd_float3x3(rows: [
            simd_float3(fx, 0, cx),
            simd_float3(0, fy, cy),
            simd_float3(0, 0, 1)
        ])

        return (c2w, K)
    }

    private func quatToMat(_ q: simd_float4) -> simd_float3x3 {
        // q = xyzw
        let x = q.x, y = q.y, z = q.z, w = q.w
        let xx = x*x, yy = y*y, zz = z*z
        let xy = x*y, xz = x*z, yz = y*z
        let wx = w*x, wy = w*y, wz = w*z
        return simd_float3x3(rows: [
            simd_float3(1 - 2*(yy+zz), 2*(xy-wz),     2*(xz+wy)),
            simd_float3(2*(xy+wz),     1-2*(xx+zz),   2*(yz-wx)),
            simd_float3(2*(xz-wy),     2*(yz+wx),     1-2*(xx+yy))
        ])
    }
}
