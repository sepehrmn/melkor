import Foundation
import Accelerate
import simd

/// Similarity transform in 3D: `p' = s * (R * p) + t`.
@available(macOS 14.0, iOS 17.0, *)
public struct DA3Sim3: Sendable {
    public var scale: Float
    public var rotation: simd_float3x3
    public var translation: simd_float3

    public init(scale: Float, rotation: simd_float3x3, translation: simd_float3) {
        self.scale = scale
        self.rotation = rotation
        self.translation = translation
    }

    public static let identity = DA3Sim3(
        scale: 1,
        rotation: matrix_identity_float3x3,
        translation: .zero
    )

    public struct EstimateConfig: Sendable {
        /// Minimum number of point correspondences required.
        public var minPointCount: Int = 3
        /// Whether to estimate an isotropic scale factor.
        public var estimateScale: Bool = true
        /// Clamp range for the estimated scale factor (useful as a safety belt in streaming alignment).
        public var scaleClamp: ClosedRange<Float> = (1.0 / 3.0)...3.0

        public init() {}
    }

    @inlinable
    public func transformPoint(_ p: simd_float3) -> simd_float3 {
        (rotation * p) * scale + translation
    }

    /// Applies this similarity transform to a camera-to-world pose:
    /// - rotation: left-multiplied by `R`
    /// - translation: transformed by `sR` + `t`
    @inlinable
    public func transformPoseC2W(_ c2w: simd_float4x4) -> simd_float4x4 {
        let r = simd_float3x3(
            simd_float3(c2w.columns.0.x, c2w.columns.0.y, c2w.columns.0.z),
            simd_float3(c2w.columns.1.x, c2w.columns.1.y, c2w.columns.1.z),
            simd_float3(c2w.columns.2.x, c2w.columns.2.y, c2w.columns.2.z)
        )
        let t = simd_float3(c2w.columns.3.x, c2w.columns.3.y, c2w.columns.3.z)

        let r2 = rotation * r
        let t2 = (rotation * t) * scale + translation

        return simd_float4x4(
            simd_float4(r2.columns.0, 0),
            simd_float4(r2.columns.1, 0),
            simd_float4(r2.columns.2, 0),
            simd_float4(t2, 1)
        )
    }

    /// Estimate a similarity transform mapping `source[i]` to `target[i]`.
    ///
    /// Uses Horn's absolute orientation (quaternion eigenvector method) for rotation and
    /// a least-squares scalar for scale.
    public static func estimate(
        from source: [simd_float3],
        to target: [simd_float3],
        config: EstimateConfig = EstimateConfig()
    ) -> DA3Sim3? {
        guard source.count == target.count else { return nil }

        var ps: [simd_double3] = []
        var qs: [simd_double3] = []
        ps.reserveCapacity(source.count)
        qs.reserveCapacity(target.count)

        for (p, q) in zip(source, target) {
            if !p.x.isFinite || !p.y.isFinite || !p.z.isFinite { continue }
            if !q.x.isFinite || !q.y.isFinite || !q.z.isFinite { continue }
            ps.append(simd_double3(Double(p.x), Double(p.y), Double(p.z)))
            qs.append(simd_double3(Double(q.x), Double(q.y), Double(q.z)))
        }

        let n = ps.count
        guard n >= max(1, config.minPointCount) else { return nil }

        var pMean = simd_double3.zero
        var qMean = simd_double3.zero
        for i in 0..<n {
            pMean += ps[i]
            qMean += qs[i]
        }
        let invN = 1.0 / Double(n)
        pMean *= invN
        qMean *= invN

        // Cross-covariance M = Σ p' q'^T (source -> target).
        var M = simd_double3x3(rows: [.zero, .zero, .zero])
        var denom: Double = 0
        var pc = [simd_double3](repeating: .zero, count: n)
        var qc = [simd_double3](repeating: .zero, count: n)
        for i in 0..<n {
            let p = ps[i] - pMean
            let q = qs[i] - qMean
            pc[i] = p
            qc[i] = q

            M.columns.0 += p * q.x
            M.columns.1 += p * q.y
            M.columns.2 += p * q.z

            denom += simd_length_squared(p)
        }

        // Degenerate configuration: no spatial variance in the source point set.
        // In this case rotation/scale are underdetermined, so return a stable translation-only transform.
        if !denom.isFinite || denom <= 1e-12 {
            let t = qMean - pMean
            return DA3Sim3(
                scale: 1,
                rotation: matrix_identity_float3x3,
                translation: simd_float3(Float(t.x), Float(t.y), Float(t.z))
            )
        }

        // Horn's 4x4 symmetric matrix N from M (Sxx..Szz).
        let Sxx = M.columns.0.x, Syx = M.columns.0.y, Szx = M.columns.0.z
        let Sxy = M.columns.1.x, Syy = M.columns.1.y, Szy = M.columns.1.z
        let Sxz = M.columns.2.x, Syz = M.columns.2.y, Szz = M.columns.2.z

        // If there is effectively no cross-covariance, the rotation is also underdetermined.
        let covEnergy =
            Sxx * Sxx + Sxy * Sxy + Sxz * Sxz +
            Syx * Syx + Syy * Syy + Syz * Syz +
            Szx * Szx + Szy * Szy + Szz * Szz
        if !covEnergy.isFinite || covEnergy <= 1e-12 {
            let t = qMean - pMean
            return DA3Sim3(
                scale: 1,
                rotation: matrix_identity_float3x3,
                translation: simd_float3(Float(t.x), Float(t.y), Float(t.z))
            )
        }

        let trace = Sxx + Syy + Szz

        let n00 = trace
        let n01 = Syz - Szy
        let n02 = Szx - Sxz
        let n03 = Sxy - Syx

        let n11 = Sxx - Syy - Szz
        let n12 = Sxy + Syx
        let n13 = Szx + Sxz

        let n22 = -Sxx + Syy - Szz
        let n23 = Syz + Szy

        let n33 = -Sxx - Syy + Szz

        var N = [Double](repeating: 0, count: 16)
        func set(_ r: Int, _ c: Int, _ v: Double) {
            N[r + c * 4] = v
        }

        set(0, 0, n00)
        set(0, 1, n01); set(1, 0, n01)
        set(0, 2, n02); set(2, 0, n02)
        set(0, 3, n03); set(3, 0, n03)

        set(1, 1, n11)
        set(1, 2, n12); set(2, 1, n12)
        set(1, 3, n13); set(3, 1, n13)

        set(2, 2, n22)
        set(2, 3, n23); set(3, 2, n23)

        set(3, 3, n33)

        guard let eig = largestEigenvectorSymmetric(N, n: 4) else { return nil }
        // Eigenvector is [w, x, y, z] for Horn's formulation.
        var q = simd_quatd(ix: eig[1], iy: eig[2], iz: eig[3], r: eig[0]).normalized
        if q.real < 0 { q = simd_quatd(vector: -q.vector) }

        let rEst = simd_double3x3(q)

        // Estimate scale (least squares) with the rotation fixed.
        var scale: Double = 1.0
        if config.estimateScale, denom.isFinite, denom > 1e-12 {
            var num: Double = 0
            for i in 0..<n {
                num += simd_dot(qc[i], rEst * pc[i])
            }
            if num.isFinite {
                scale = num / denom
            }
        }
        if !scale.isFinite || scale <= 0 {
            scale = 1.0
        }
        let clampedScale = Double(min(config.scaleClamp.upperBound, max(config.scaleClamp.lowerBound, Float(scale))))

        let tEst = qMean - clampedScale * (rEst * pMean)

        let rFloat = simd_float3x3(
            simd_float3(Float(rEst.columns.0.x), Float(rEst.columns.0.y), Float(rEst.columns.0.z)),
            simd_float3(Float(rEst.columns.1.x), Float(rEst.columns.1.y), Float(rEst.columns.1.z)),
            simd_float3(Float(rEst.columns.2.x), Float(rEst.columns.2.y), Float(rEst.columns.2.z))
        )
        let tFloat = simd_float3(Float(tEst.x), Float(tEst.y), Float(tEst.z))

        return DA3Sim3(scale: Float(clampedScale), rotation: rFloat, translation: tFloat)
    }

    private static func largestEigenvectorSymmetric(_ A: [Double], n: Int) -> [Double]? {
        var a = A
        var w = [Double](repeating: 0, count: n)
        var n_ = __LAPACK_int(n)
        var lda = n_
        var jobz: Int8 = 86 // 'V'
        var uplo: Int8 = 85 // 'U'
        var info: __LAPACK_int = 0

        // Workspace query.
        var lwork: __LAPACK_int = -1
        var workQuery: Double = 0
        dsyev_(&jobz, &uplo, &n_, &a, &lda, &w, &workQuery, &lwork, &info)
        if info != 0 { return nil }

        lwork = __LAPACK_int(max(1, Int(workQuery)))
        var work = [Double](repeating: 0, count: Int(lwork))
        dsyev_(&jobz, &uplo, &n_, &a, &lda, &w, &work, &lwork, &info)
        if info != 0 { return nil }

        // Largest eigenvalue is w[n-1], eigenvector is last column.
        let start = (n - 1) * n
        let end = n * n
        return Array(a[start..<end])
    }
}
