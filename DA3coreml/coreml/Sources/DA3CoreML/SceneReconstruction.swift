import Foundation
import simd

/// A single 3D Gaussian splat for DA3CoreML.
///
/// Self-contained 3DGS data model (independent of any external renderer).
/// It represents a 3D Gaussian primitive with position, color, scale, rotation, and opacity.
@available(macOS 14.0, iOS 17.0, *)
public struct DA3GaussianSplat {
    // MARK: - Position (x, y, z)
    public var x: Float
    public var y: Float
    public var z: Float
    
    // MARK: - Color (Spherical Harmonics DC coefficients)
    /// Red channel SH DC coefficient
    public var shDC0: Float
    /// Green channel SH DC coefficient
    public var shDC1: Float
    /// Blue channel SH DC coefficient
    public var shDC2: Float
    
    // MARK: - Opacity (logit space)
    /// Opacity in logit space. Use `opacity` property for [0,1] value.
    public var opacityLogit: Float
    
    // MARK: - Scale (log space)
    /// Scale X in log space. Use `scaleX` property for actual scale.
    public var scaleLog0: Float
    /// Scale Y in log space
    public var scaleLog1: Float
    /// Scale Z in log space
    public var scaleLog2: Float
    
    // MARK: - Rotation (quaternion: w, x, y, z)
    public var rotW: Float
    public var rotX: Float
    public var rotY: Float
    public var rotZ: Float
    
    // MARK: - Computed Properties
    
    /// Opacity in [0, 1] range
    public var opacity: Float {
        get { 1.0 / (1.0 + exp(-opacityLogit)) }
        set {
            // Guard against log(0) -> +/-Inf when callers pass 0 or 1.
            let p = max(0.001, min(0.999, newValue))
            opacityLogit = log(p / (1.0 - p))
        }
    }
    
    /// Actual scale values
    public var scaleX: Float { exp(scaleLog0) }
    public var scaleY: Float { exp(scaleLog1) }
    public var scaleZ: Float { exp(scaleLog2) }
    
    /// RGB color in [0, 1] range
    public var rgb: (r: Float, g: Float, b: Float) {
        let c0: Float = 0.28209479177387814  // SH C0 constant
        return (
            r: shDC0 * c0 + 0.5,
            g: shDC1 * c0 + 0.5,
            b: shDC2 * c0 + 0.5
        )
    }
    
    // MARK: - Initialization
    
    public init() {
        x = 0; y = 0; z = 0
        shDC0 = 0; shDC1 = 0; shDC2 = 0
        opacityLogit = 0  // 0.5 opacity
        scaleLog0 = -3; scaleLog1 = -3; scaleLog2 = -3  // Small scale
        rotW = 1; rotX = 0; rotY = 0; rotZ = 0  // Identity rotation
    }
    
    public init(
        position: (x: Float, y: Float, z: Float),
        rgb: (r: Float, g: Float, b: Float),
        opacity: Float = 0.9,
        scale: Float = 0.01
    ) {
        self.x = position.x
        self.y = position.y
        self.z = position.z
        
        // Convert RGB to SH DC
        let c0: Float = 0.28209479177387814
        self.shDC0 = (rgb.r - 0.5) / c0
        self.shDC1 = (rgb.g - 0.5) / c0
        self.shDC2 = (rgb.b - 0.5) / c0
        
        // Opacity in logit space
        let clampedOpacity = max(0.001, min(0.999, opacity))
        self.opacityLogit = log(clampedOpacity / (1.0 - clampedOpacity))
        
        // Scale in log space
        let logScale = log(max(1e-7, scale))
        self.scaleLog0 = logScale
        self.scaleLog1 = logScale
        self.scaleLog2 = logScale
        
        // Identity rotation
        self.rotW = 1; self.rotX = 0; self.rotY = 0; self.rotZ = 0
    }
    
    /// Normalize the rotation quaternion
    public mutating func normalizeRotation() {
        let len = sqrt(rotW*rotW + rotX*rotX + rotY*rotY + rotZ*rotZ)
        if len > 0 {
            let invLen = 1.0 / len
            rotW *= invLen
            rotX *= invLen
            rotY *= invLen
            rotZ *= invLen
        } else {
            rotW = 1; rotX = 0; rotY = 0; rotZ = 0
        }
        // Canonical form: w positive
        if rotW < 0 {
            rotW = -rotW; rotX = -rotX; rotY = -rotY; rotZ = -rotZ
        }
    }
    
    /// Position as SIMD vector
    public var positionSIMD: simd_float3 {
        get { simd_float3(x, y, z) }
        set { x = newValue.x; y = newValue.y; z = newValue.z }
    }
    
    /// Rotation as SIMD quaternion (WXYZ format)
    public var rotationSIMD: simd_quatf {
        get { simd_quatf(ix: rotX, iy: rotY, iz: rotZ, r: rotW) }
        set {
            rotW = newValue.real
            rotX = newValue.imag.x
            rotY = newValue.imag.y
            rotZ = newValue.imag.z
        }
    }
    
    /// Scale as SIMD vector (actual scale, not log)
    public var scaleSIMD: simd_float3 {
        get { simd_float3(scaleX, scaleY, scaleZ) }
        set {
            scaleLog0 = log(max(1e-7, newValue.x))
            scaleLog1 = log(max(1e-7, newValue.y))
            scaleLog2 = log(max(1e-7, newValue.z))
        }
    }
}

/// Collection of Gaussian splats for DA3CoreML.
@available(macOS 14.0, iOS 17.0, *)
public final class DA3GaussianCloud {
    
    // MARK: - Properties
    
    private var splats: [DA3GaussianSplat] = []
    
    /// Spherical harmonics degree (0 = DC only, up to 3)
    public var shDegree: Int = 0
    
    // MARK: - Initialization
    
    public init() {}
    
    public init(capacity: Int) {
        splats.reserveCapacity(capacity)
    }
    
    // MARK: - Accessors
    
    public var count: Int { splats.count }
    public var isEmpty: Bool { splats.isEmpty }
    
    public subscript(index: Int) -> DA3GaussianSplat {
        get { splats[index] }
        set { splats[index] = newValue }
    }
    
    public var allSplats: [DA3GaussianSplat] { splats }
    
    // MARK: - Modification
    
    public func add(_ splat: DA3GaussianSplat) {
        splats.append(splat)
    }
    
    public func add(contentsOf newSplats: [DA3GaussianSplat]) {
        splats.append(contentsOf: newSplats)
    }
    
    public func reserve(_ capacity: Int) {
        splats.reserveCapacity(capacity)
    }
    
    public func clear() {
        splats.removeAll()
    }
    
    // MARK: - Bounding Box
    
    public func computeBoundingBox() -> (min: (x: Float, y: Float, z: Float), max: (x: Float, y: Float, z: Float)) {
        guard !splats.isEmpty else {
            return (min: (0, 0, 0), max: (0, 0, 0))
        }
        
        var minX: Float = .greatestFiniteMagnitude
        var minY: Float = .greatestFiniteMagnitude
        var minZ: Float = .greatestFiniteMagnitude
        var maxX: Float = -.greatestFiniteMagnitude
        var maxY: Float = -.greatestFiniteMagnitude
        var maxZ: Float = -.greatestFiniteMagnitude
        
        for splat in splats {
            minX = min(minX, splat.x)
            minY = min(minY, splat.y)
            minZ = min(minZ, splat.z)
            maxX = max(maxX, splat.x)
            maxY = max(maxY, splat.y)
            maxZ = max(maxZ, splat.z)
        }
        
        return (min: (minX, minY, minZ), max: (maxX, maxY, maxZ))
    }
    
    /// Center the cloud at origin
    public func centerAtOrigin() {
        let bbox = computeBoundingBox()
        let cx = (bbox.min.x + bbox.max.x) / 2
        let cy = (bbox.min.y + bbox.max.y) / 2
        let cz = (bbox.min.z + bbox.max.z) / 2
        
        for i in 0..<splats.count {
            splats[i].x -= cx
            splats[i].y -= cy
            splats[i].z -= cz
        }
    }
    
    /// Scale the cloud uniformly
    public func scale(by factor: Float) {
        for i in 0..<splats.count {
            splats[i].x *= factor
            splats[i].y *= factor
            splats[i].z *= factor
            splats[i].scaleLog0 += log(factor)
            splats[i].scaleLog1 += log(factor)
            splats[i].scaleLog2 += log(factor)
        }
    }
    
    // MARK: - DA3-Quality Normalization
    
    /// DA3-style shift and scale normalization.
    /// Shifts the scene so that the median Gaussian is at the origin,
    /// then rescales so that most Gaussians (95th percentile) are within [-1, 1].
    /// This matches the DA3 Python implementation in gsply_helpers.py.
    public func shiftAndScaleNormalize() {
        guard !splats.isEmpty else { return }
        
        // Extract positions
        let xs = splats.map { $0.x }
        let ys = splats.map { $0.y }
        let zs = splats.map { $0.z }
        
        // Compute median for each axis
        let medX = computeMedian(xs)
        let medY = computeMedian(ys)
        let medZ = computeMedian(zs)
        
        // Shift to median
        for i in 0..<splats.count {
            splats[i].x -= medX
            splats[i].y -= medY
            splats[i].z -= medZ
        }
        
        // Compute 95th percentile of absolute values for each axis
        let absXs = splats.map { abs($0.x) }
        let absYs = splats.map { abs($0.y) }
        let absZs = splats.map { abs($0.z) }
        
        let q95X = computePercentile(absXs, percentile: 0.95)
        let q95Y = computePercentile(absYs, percentile: 0.95)
        let q95Z = computePercentile(absZs, percentile: 0.95)
        
        // Scale factor is the max of 95th percentiles across all axes
        let scaleFactor = max(max(q95X, q95Y), q95Z)
        
        // Avoid division by zero
        guard scaleFactor > 1e-7 else { return }
        
        let invScale = 1.0 / scaleFactor
        
        // Apply scale to positions and Gaussian scales
        let logScaleFactor = log(scaleFactor)
        for i in 0..<splats.count {
            splats[i].x *= invScale
            splats[i].y *= invScale
            splats[i].z *= invScale
            // Adjust Gaussian scale (in log space: log(s/f) = log(s) - log(f))
            splats[i].scaleLog0 -= logScaleFactor
            splats[i].scaleLog1 -= logScaleFactor
            splats[i].scaleLog2 -= logScaleFactor
        }
    }
    
    /// Compute the median of an array of floats
    private func computeMedian(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let n = sorted.count
        if n % 2 == 0 {
            return (sorted[n/2 - 1] + sorted[n/2]) / 2
        } else {
            return sorted[n/2]
        }
    }
    
    /// Compute the value at a given percentile (0.0 to 1.0)
    private func computePercentile(_ values: [Float], percentile: Float) -> Float {
        let validValues = values.filter { $0.isFinite }
        guard !validValues.isEmpty else { return 0 }
        let sorted = validValues.sorted()
        let n = sorted.count
        let index = percentile * Float(n - 1)
        let lowerIndex = Int(index)
        let upperIndex = min(lowerIndex + 1, n - 1)
        let fraction = index - Float(lowerIndex)
        return sorted[lowerIndex] * (1 - fraction) + sorted[upperIndex] * fraction
    }
    
    // MARK: - World-Space Transforms
    
    /// Transform all Gaussians using a 4x4 transformation matrix
    /// Useful for aligning coordinate systems or applying global transforms
    public func transform(by matrix: simd_float4x4) {
        let rotation = simd_float3x3(
            simd_float3(matrix.columns.0.x, matrix.columns.0.y, matrix.columns.0.z),
            simd_float3(matrix.columns.1.x, matrix.columns.1.y, matrix.columns.1.z),
            simd_float3(matrix.columns.2.x, matrix.columns.2.y, matrix.columns.2.z)
        )
        let rotationQuat = simd_quatf(rotation)
        
        // Compute uniform scale from matrix (average of column norms)
        let scaleX = simd_length(simd_float3(matrix.columns.0.x, matrix.columns.0.y, matrix.columns.0.z))
        let scaleY = simd_length(simd_float3(matrix.columns.1.x, matrix.columns.1.y, matrix.columns.1.z))
        let scaleZ = simd_length(simd_float3(matrix.columns.2.x, matrix.columns.2.y, matrix.columns.2.z))
        let uniformScale = (scaleX + scaleY + scaleZ) / 3.0
        let logScale = log(uniformScale)
        
        for i in 0..<splats.count {
            // Transform position
            let pos = simd_float4(splats[i].x, splats[i].y, splats[i].z, 1)
            let newPos = matrix * pos
            splats[i].x = newPos.x
            splats[i].y = newPos.y
            splats[i].z = newPos.z
            
            // Transform rotation
            let oldQuat = splats[i].rotationSIMD
            let newQuat = rotationQuat * oldQuat
            splats[i].rotationSIMD = newQuat
            splats[i].normalizeRotation()
            
            // Scale Gaussian sizes
            splats[i].scaleLog0 += logScale
            splats[i].scaleLog1 += logScale
            splats[i].scaleLog2 += logScale
        }
    }
}
