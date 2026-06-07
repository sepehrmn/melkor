import Foundation
import CoreML
import Accelerate
import simd

/// DA3-style camera pose/intrinsics estimation from predicted camera rays.
///
/// This is a Swift port of `depth_anything_3/utils/ray_utils.py`:
/// - `camray_to_caminfo`
/// - `get_extrinsic_from_camray`
///
/// Notes / practical differences from the PyTorch reference:
/// - The official implementation operates on the ray grid resolution returned by the DualDPT head.
///   Some CoreML exports use an auxiliary ray grid (smaller than the backbone input) for speed.
///   For performance, this implementation
///   can **subsample** that grid while preserving the original coordinate mapping.
/// - DA3’s `get_extrinsic_from_camray()` forms an extrinsic matrix from the estimated (R, T) and
///   then inverts it inside the model forward pass for consistency with the camera-decoder pathway.
///   This implementation returns **camera-to-world (c2w)** for downstream geometry.
@available(macOS 14.0, iOS 17.0, *)
public enum DA3RayPoseEstimator {

    // MARK: - Types

    public struct Config {
        /// Subsample factor on the ray grid (1 = use all rays, 2 = every other ray, ...).
        /// Set to 1 to match DA3 exactly (slowest).
        public var subsample: Int = 4
        /// RANSAC iterations.
        public var ransacIterations: Int = 100
        /// Candidate sampling ratio (top-k by confidence).
        public var sampleRatio: Double = 0.3
        /// Number of points used to estimate a candidate homography in each RANSAC iter.
        public var numSampleForRansac: Int = 8
        /// Reprojection threshold in normalized plane coordinates.
        public var reprojThreshold: Double = 0.2
        /// Z threshold for validity.
        public var zThreshold: Double = 1e-4
        /// Limit inlier points used for final refit (matches DA3's `max_inlier_num` default behavior).
        public var maxInlierCount: Int = 8000
        /// Seed for deterministic sampling.
        public var seed: UInt64 = 0xDA3DA3DA3

        public init() {}
    }

    public struct Pose {
        /// Camera-to-world transform.
        public let c2w: simd_float4x4
        /// Pinhole intrinsics in pixel units.
        public let intrinsics: simd_float3x3
    }

    // MARK: - Public API

    /// Estimate camera pose (c2w) and intrinsics from a predicted ray field.
    ///
    /// - Parameters:
    ///   - rays: Ray tensor from DualDPT. Expected shapes:
    ///     - `[1, 6, H, W]` or `[6, H, W]` (channel-first).
    ///     The first 3 channels are a direction/target vector, last 3 channels are a translation vector.
    ///   - rayConfidence: Confidence tensor. Expected shapes:
    ///     - `[1, 1, H, W]`, `[1, H, W]`, or `[H, W]`.
    ///   - imageWidth: Image width in pixels (e.g. 518).
    ///   - imageHeight: Image height in pixels (e.g. 518).
    ///   - config: Estimator configuration.
        public static func estimatePose(
        rays: MLMultiArray,
        rayConfidence: MLMultiArray,
        imageWidth: Int,
        imageHeight: Int,
        config: Config = Config()
    ) throws -> Pose {
        let raysR = try MLMultiArrayFloatReader(rays)
        let confR = try MLMultiArrayFloatReader(rayConfidence)

        // Fail fast on NaN/Inf in the *ray vectors*. When the DualDPT ray branch is numerically unstable
        // (commonly in float16 CoreML exports), downstream pose estimation silently degenerates (often to
        // the identity pose), which then produces "rectangle" 3DGS fusions.
        //
        // Note: we intentionally do NOT fail-fast on non-finite confidence values here. Confidence is used
        // as a weight, and the per-sample loop below already clamps non-finite weights to 0. This makes the
        // estimator more robust against `exp()` overflow in `ray_confidence` while still rejecting invalid rays.
        if containsNonFinite(reader: raysR, totalCount: rays.count, samples: 4096) {
            throw DA3Error.inferenceError(
                "Ray pose estimation failed: rays contain NaN/Inf. " +
                "Use the float32 DualDPT head and/or force the head to CPU-only."
            )
        }

        let raysShape = raysR.shape
        let (channels, h, w): (Int, Int, Int) = {
            switch raysShape.count {
            case 4: return (raysShape[1], raysShape[2], raysShape[3]) // [B,C,H,W]
            case 3: return (raysShape[0], raysShape[1], raysShape[2]) // [C,H,W]
            default: return (0, 0, 0)
            }
        }()
        guard channels >= 6, h > 0, w > 0 else {
            throw DA3Error.invalidShape("rays shape \(raysShape) must be [*,6,H,W]")
        }

        // DA3's ray pose operates directly on the ray grid resolution:
        // `camray.shape[-3]` / `camray.shape[-2]` are treated as (num_patches_y, num_patches_x)
        // in `unproject_depth(ixt_normalized=True, ...)`.
        //
        // The canonical source points are therefore:
        //   src = (x-1, y-1) where x = (2*px+1)/w, y = (2*py+1)/h.
        // In other words:
        //   srcX = -1 + (2*x + 1)/w
        //   srcY = -1 + (2*y + 1)/h

        let subsample = max(1, config.subsample)
        let sampleH = (h + subsample - 1) / subsample
        let sampleW = (w + subsample - 1) / subsample
        let N = sampleH * sampleW
        var srcX = [Double](repeating: 0, count: N)
        var srcY = [Double](repeating: 0, count: N)
        var dstX = [Double](repeating: 0, count: N)
        var dstY = [Double](repeating: 0, count: N)
        var wts = [Double](repeating: 0, count: N)
        var tX = [Double](repeating: 0, count: N)
        var tY = [Double](repeating: 0, count: N)
        var tZ = [Double](repeating: 0, count: N)

        func ray(_ ch: Int, _ y: Int, _ x: Int) -> Double {
            switch raysShape.count {
            case 4: return Double(raysR.read(0, ch, y, x))
            case 3: return Double(raysR.read(ch, y, x))
            default: return 0
            }
        }

        let confShape = confR.shape
        func conf(_ y: Int, _ x: Int) -> Double {
            switch confShape.count {
            case 4: return Double(confR.read(0, 0, y, x))
            case 3:
                // Treat as [C,H,W] and read channel 0.
                return Double(confR.read(0, y, x))
            case 2:
                return Double(confR.read(y, x))
            default:
                return 1.0
            }
        }

        var k = 0
        for y in stride(from: 0, to: h, by: subsample) {
            let sy = -1.0 + (2.0 * Double(y) + 1.0) / Double(h)
            for x in stride(from: 0, to: w, by: subsample) {
                let sx = -1.0 + (2.0 * Double(x) + 1.0) / Double(w)

                let rx = ray(0, y, x)
                let ry = ray(1, y, x)
                let rz = ray(2, y, x)
                let wt = conf(y, x)

                srcX[k] = sx
                srcY[k] = sy

                // Normalize target by Z (matches DA3).
                if rz.isFinite, abs(rz) > config.zThreshold {
                    dstX[k] = rx / rz
                    dstY[k] = ry / rz
                    wts[k] = wt.isFinite ? max(0, wt) : 0
                } else {
                    dstX[k] = 0
                    dstY[k] = 0
                    wts[k] = 0
                }

                tX[k] = ray(3, y, x)
                tY[k] = ray(4, y, x)
                tZ[k] = ray(5, y, x)

                k += 1
            }
        }

        if k != N {
            // Should not happen, but avoid mismatched array assumptions if stride math changes.
            srcX.removeLast(N - k)
            srcY.removeLast(N - k)
            dstX.removeLast(N - k)
            dstY.removeLast(N - k)
            wts.removeLast(N - k)
            tX.removeLast(N - k)
            tY.removeLast(N - k)
            tZ.removeLast(N - k)
        }

        // If there are too few valid points, fail fast. Returning identity here silently produces
        // "rectangle" fusions (every view shares the same pose) and masks upstream issues.
        let nonZero = wts.reduce(0) { $1 > 0 ? $0 + 1 : $0 }
        guard nonZero >= 16 else {
            throw DA3Error.inferenceError(
                "Ray pose estimation failed: too few valid rays (nonZero=\(nonZero)). " +
                "This can happen if ray_confidence is near-zero everywhere or rays have invalid Z. " +
                "Try using the float32 DualDPT head, forcing the head to CPU, or using CamDec instead."
            )
        }

        let H = estimateHomographyRANSAC(
            srcX: srcX,
            srcY: srcY,
            dstX: dstX,
            dstY: dstY,
            weights: wts,
            iterations: config.ransacIterations,
            sampleRatio: config.sampleRatio,
            sampleSize: config.numSampleForRansac,
            reprojThreshold: config.reprojThreshold,
            maxInlierCount: config.maxInlierCount,
            seed: config.seed
        )

        // QL decomposition (DA3) -> rotation + intrinsics in normalized 2×2 image space.
        let (R, L) = qlDecompositionDA3(H)

        // Convert intrinsics to pixel units (matches DA3):
        // pred_focal_lengths = 1/f; pred_pp = pp + 1
        let fxNorm = safeInv(L[0][0])
        let fyNorm = safeInv(L[1][1])
        let ppXNorm = L[2][0] + 1.0
        let ppYNorm = L[2][1] + 1.0

        let fxPix = Float(fxNorm / 2.0 * Double(imageWidth))
        let fyPix = Float(fyNorm / 2.0 * Double(imageHeight))
        let cxPix = Float(ppXNorm * Double(imageWidth) * 0.5)
        let cyPix = Float(ppYNorm * Double(imageHeight) * 0.5)

        guard fxPix.isFinite, fyPix.isFinite, cxPix.isFinite, cyPix.isFinite,
              fxPix > 1e-3, fyPix > 1e-3 else {
            throw DA3Error.inferenceError(
                "Ray pose estimation produced invalid intrinsics (fx=\(fxPix), fy=\(fyPix), cx=\(cxPix), cy=\(cyPix)). " +
                "This usually indicates unstable ray predictions or degenerate RANSAC."
            )
        }

        let K = simd_float3x3(rows: [
            simd_float3(fxPix, 0, cxPix),
            simd_float3(0, fyPix, cyPix),
            simd_float3(0, 0, 1),
        ])

        // Translation from per-pixel (weighted) average of ray translation channels.
        var sumW: Double = 0
        var tx: Double = 0
        var ty: Double = 0
        var tz: Double = 0
        for i in 0..<N {
            let wi = wts[i]
            if wi <= 0 { continue }
            sumW += wi
            tx += wi * tX[i]
            ty += wi * tY[i]
            tz += wi * tZ[i]
        }
        if sumW <= 0 { sumW = 1 }
        let t = simd_float3(Float(tx / sumW), Float(ty / sumW), Float(tz / sumW))
        guard t.x.isFinite, t.y.isFinite, t.z.isFinite else {
            throw DA3Error.inferenceError("Ray pose estimation produced non-finite translation (t=\(t)).")
        }

        let r0 = simd_float3(Float(R[0][0]), Float(R[0][1]), Float(R[0][2]))
        let r1 = simd_float3(Float(R[1][0]), Float(R[1][1]), Float(R[1][2]))
        let r2 = simd_float3(Float(R[2][0]), Float(R[2][1]), Float(R[2][2]))
        let rot = simd_float3x3(rows: [r0, r1, r2])
        guard rot.columns.0.x.isFinite else {
            throw DA3Error.inferenceError("Ray pose estimation produced non-finite rotation matrix.")
        }

        // DA3 convention: the estimated (R, T) corresponds to a **world-to-camera** transform.
        // DA3 then inverts it to get c2w:
        //   c2w.R = Rᵀ
        //   c2w.t = -Rᵀ T
        let w2cRot = rot
        let w2cT = t

        let c2wRot = w2cRot.transpose
        let c2wT = -(c2wRot * w2cT)

        let c2w = simd_float4x4(
            simd_float4(c2wRot.columns.0, 0),
            simd_float4(c2wRot.columns.1, 0),
            simd_float4(c2wRot.columns.2, 0),
            simd_float4(c2wT, 1)
        )

        return Pose(c2w: c2w, intrinsics: K)
    }

    // MARK: - Validation

    private static func containsNonFinite(reader: MLMultiArrayFloatReader, totalCount: Int, samples: Int) -> Bool {
        let count = max(0, totalCount)
        if count == 0 { return false }
        let shape = reader.shape
        let strides = reader.strides

        func rowMajorValue(_ linear: Int) -> Float {
            var t = linear
            var offset = 0
            for dim in shape.indices.reversed() {
                let size = max(1, shape[dim])
                let idx = t % size
                t /= size
                offset += idx * strides[dim]
            }
            return reader.readLinear(offset)
        }

        let target = max(64, samples)
        let step = max(1, count / target)
        var i = 0
        while i < count {
            let v = rowMajorValue(i)
            if !v.isFinite { return true }
            i += step
        }
        let vLast = rowMajorValue(count - 1)
        return !vLast.isFinite
    }

    // MARK: - Homography (DA3 RANSAC)

    private static func estimateHomographyRANSAC(
        srcX: [Double],
        srcY: [Double],
        dstX: [Double],
        dstY: [Double],
        weights: [Double],
        iterations: Int,
        sampleRatio: Double,
        sampleSize: Int,
        reprojThreshold: Double,
        maxInlierCount: Int,
        seed: UInt64
    ) -> [[Double]] {
        let N = srcX.count
        precondition(dstX.count == N && dstY.count == N && weights.count == N)

        // Candidate points: top-k by confidence.
        let nSample = max(sampleSize, Int(Double(N) * max(0, min(1, sampleRatio))))
        let sorted = (0..<N).sorted { weights[$0] > weights[$1] }
        let candidates = Array(sorted.prefix(min(nSample, sorted.count)))

        var rng = SplitMix64(seed: seed)

        var bestScore: Double = -1
        var bestMask = [Bool](repeating: false, count: N)

        // Temporary storage to avoid reallocations.
        var sampleIdx = [Int](repeating: 0, count: max(1, sampleSize))

        for _ in 0..<max(1, iterations) {
            // Random unique sample from candidates (partial Fisher-Yates).
            var tmp = candidates
            let k = min(sampleSize, tmp.count)
            for i in 0..<k {
                let j = i + Int(rng.nextUInt64() % UInt64(max(1, tmp.count - i)))
                tmp.swapAt(i, j)
                sampleIdx[i] = tmp[i]
            }

            let H = computeHomographyDLT(
                srcX: srcX, srcY: srcY, dstX: dstX, dstY: dstY, weights: weights, indices: Array(sampleIdx.prefix(k))
            )
            if H.isEmpty { continue }

            var score: Double = 0
            var mask = [Bool](repeating: false, count: N)
            for i in 0..<N {
                let wi = weights[i]
                if wi <= 0 { continue }
                let e = reprojError(H, srcX[i], srcY[i], dstX[i], dstY[i])
                if e.isFinite, e < reprojThreshold {
                    mask[i] = true
                    score += wi
                }
            }

            if score > bestScore {
                bestScore = score
                bestMask = mask
            }
        }

        // Refit using inliers.
        var inliers: [Int] = []
        inliers.reserveCapacity(N)
        for i in 0..<N where bestMask[i] {
            inliers.append(i)
        }
        if inliers.isEmpty {
            // Fall back to a weighted fit on all points.
            inliers = (0..<N).filter { weights[$0] > 0 }
        }

        // DA3: cap inlier count by sampling.
        if inliers.count > maxInlierCount {
            inliers.sort { weights[$0] > weights[$1] }
            let keepLen = max(Int(Double(inliers.count) * 0.95), maxInlierCount)
            let trimmed = Array(inliers.prefix(min(keepLen, inliers.count)))
            var tmp = trimmed
            // Shuffle and keep maxInlierCount.
            for i in 0..<maxInlierCount {
                let j = i + Int(rng.nextUInt64() % UInt64(max(1, tmp.count - i)))
                tmp.swapAt(i, j)
            }
            inliers = Array(tmp.prefix(maxInlierCount))
        }

        var H = computeHomographyDLT(srcX: srcX, srcY: srcY, dstX: dstX, dstY: dstY, weights: weights, indices: inliers)
        if H.isEmpty {
            H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        }

        // Ensure positive determinant (DA3).
        if det3(H) < 0 {
            for r in 0..<3 { for c in 0..<3 { H[r][c] = -H[r][c] } }
        }

        return H
    }

    /// Weighted DLT homography via smallest eigenvector of (AᵀA).
    private static func computeHomographyDLT(
        srcX: [Double],
        srcY: [Double],
        dstX: [Double],
        dstY: [Double],
        weights: [Double],
        indices: [Int]
    ) -> [[Double]] {
        guard indices.count >= 4 else { return [] }

        // Accumulate M = A^T A (9x9) in column-major (Fortran) layout for LAPACK.
        var M = [Double](repeating: 0, count: 9 * 9)

        for idx in indices {
            let w = weights[idx]
            if w <= 0 { continue }
            let ww = sqrt(w)

            let x = srcX[idx]
            let y = srcY[idx]
            let u = dstX[idx]
            let v = dstY[idx]

            // Two rows, weighted by sqrt(w).
            var r1 = [Double](repeating: 0, count: 9)
            var r2 = [Double](repeating: 0, count: 9)

            r1[0] = -x * ww
            r1[1] = -y * ww
            r1[2] = -1.0 * ww
            r1[6] = x * u * ww
            r1[7] = y * u * ww
            r1[8] = u * ww

            r2[3] = -x * ww
            r2[4] = -y * ww
            r2[5] = -1.0 * ww
            r2[6] = x * v * ww
            r2[7] = y * v * ww
            r2[8] = v * ww

            accumulateSymmetricOuter(row: r1, into: &M, n: 9)
            accumulateSymmetricOuter(row: r2, into: &M, n: 9)
        }

        // Mirror upper -> lower to fully define the symmetric matrix.
        for i in 0..<9 {
            for j in 0..<i {
                M[i + j * 9] = M[j + i * 9]
            }
        }

        guard let vec = smallestEigenvectorSymmetric(M, n: 9) else { return [] }
        let h33 = vec[8]
        let s = abs(h33) > 1e-12 ? (1.0 / h33) : 1.0
        let v = vec.map { $0 * s }
        return [
            [v[0], v[1], v[2]],
            [v[3], v[4], v[5]],
            [v[6], v[7], v[8]],
        ]
    }

    private static func reprojError(_ H: [[Double]], _ x: Double, _ y: Double, _ u: Double, _ v: Double) -> Double {
        let d = H[2][0] * x + H[2][1] * y + H[2][2]
        if !d.isFinite || abs(d) < 1e-12 { return .infinity }
        let up = (H[0][0] * x + H[0][1] * y + H[0][2]) / d
        let vp = (H[1][0] * x + H[1][1] * y + H[1][2]) / d
        let du = up - u
        let dv = vp - v
        return sqrt(du * du + dv * dv)
    }

    private static func det3(_ A: [[Double]]) -> Double {
        let a = A[0][0], b = A[0][1], c = A[0][2]
        let d = A[1][0], e = A[1][1], f = A[1][2]
        let g = A[2][0], h = A[2][1], i = A[2][2]
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    }

    // MARK: - QL decomposition (DA3)

    /// Returns (Q, L) where A ≈ QL, matching DA3's `ql_decomposition` for extracting camera parameters.
    private static func qlDecompositionDA3(_ A: [[Double]]) -> ([[Double]], [[Double]]) {
        // P swaps axis 0 and 2.
        func swapCols(_ M: [[Double]]) -> [[Double]] {
            [
                [M[0][2], M[0][1], M[0][0]],
                [M[1][2], M[1][1], M[1][0]],
                [M[2][2], M[2][1], M[2][0]],
            ]
        }
        func swapRows(_ M: [[Double]]) -> [[Double]] {
            [M[2], M[1], M[0]]
        }

        let Atilde = swapCols(A) // A * P
        let (Qtilde, Rtilde) = qrDecomposition3x3(Atilde)
        var Q = swapCols(Qtilde) // Qtilde * P
        var L = swapRows(swapCols(Rtilde)) // P * Rtilde * P

        // Sign corrections so diag(L) is positive (DA3).
        for i in 0..<3 {
            let d = L[i][i]
            let s = d < 0 ? -1.0 : 1.0
            if s != 1.0 {
                // Q[:, i] *= s (column)
                for r in 0..<3 { Q[r][i] *= s }
                // L[i, :] *= s (row)
                for c in 0..<3 { L[i][c] *= s }
            }
        }

        // Normalize so L[2][2] == 1 (homography scale).
        let denom = (abs(L[2][2]) > 1e-12) ? L[2][2] : 1.0
        for r in 0..<3 { for c in 0..<3 { L[r][c] /= denom } }

        return (Q, L)
    }

    /// QR decomposition for a 3×3 matrix using classical Gram-Schmidt on columns.
    /// Returns (Q, R) where A = Q R, Q orthonormal, R upper-triangular.
    private static func qrDecomposition3x3(_ A: [[Double]]) -> ([[Double]], [[Double]]) {
        func col(_ c: Int) -> [Double] { [A[0][c], A[1][c], A[2][c]] }
        func dot(_ a: [Double], _ b: [Double]) -> Double { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
        func norm(_ a: [Double]) -> Double { sqrt(max(1e-24, dot(a, a))) }
        func scale(_ a: [Double], _ s: Double) -> [Double] { [a[0] * s, a[1] * s, a[2] * s] }
        func sub(_ a: [Double], _ b: [Double]) -> [Double] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
        func add(_ a: [Double], _ b: [Double]) -> [Double] { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }

        let a0 = col(0)
        let a1 = col(1)
        let a2 = col(2)

        let r00 = norm(a0)
        let q0 = scale(a0, 1.0 / r00)

        let r01 = dot(q0, a1)
        let u1 = sub(a1, scale(q0, r01))
        let r11 = norm(u1)
        let q1 = scale(u1, 1.0 / r11)

        let r02 = dot(q0, a2)
        let r12 = dot(q1, a2)
        let u2 = sub(sub(a2, scale(q0, r02)), scale(q1, r12))
        let r22 = norm(u2)
        let q2 = scale(u2, 1.0 / r22)

        // Q from columns.
        let Q = [
            [q0[0], q1[0], q2[0]],
            [q0[1], q1[1], q2[1]],
            [q0[2], q1[2], q2[2]],
        ]

        let R = [
            [r00, r01, r02],
            [0, r11, r12],
            [0, 0, r22],
        ]

        // Re-orthogonalize lightly (optional) could go here if needed.
        _ = add // silence unused warning if compiler inlines aggressively
        return (Q, R)
    }

    // MARK: - Linear algebra helpers

    /// Accumulate `rowᵀ row` into the upper-triangle of a symmetric matrix stored column-major.
    private static func accumulateSymmetricOuter(row: [Double], into M: inout [Double], n: Int) {
        precondition(row.count == n)
        for i in 0..<n {
            let ri = row[i]
            for j in i..<n {
                M[i + j * n] += ri * row[j]
            }
        }
    }

    /// Compute smallest eigenvector of a symmetric matrix using LAPACK `dsyev`.
    /// - Parameter A: column-major symmetric matrix (n×n).
    private static func smallestEigenvectorSymmetric(_ A: [Double], n: Int) -> [Double]? {
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

        // Eigenvectors are stored in columns; smallest eigenvalue is w[0], vector is first column.
        return Array(a[0..<n])
    }

    private static func safeInv(_ x: Double) -> Double {
        if !x.isFinite || abs(x) < 1e-12 { return 0 }
        return 1.0 / x
    }
}

// MARK: - Small deterministic RNG

/// SplitMix64 RNG (fast, deterministic) used for RANSAC sampling.
private struct SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed == 0 ? 0x9E3779B97F4A7C15 : seed
    }

    mutating func nextUInt64() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
