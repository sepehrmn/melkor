import Foundation
import CoreGraphics
import CoreML
import ImageIO

/// Output writer for DA3CoreML results.
///
/// Saves depth maps, ray maps, and confidence maps to disk in formats that can be
/// loaded later by other tools (3DGS, Python scripts, etc.) without keeping the
/// DA3 model in memory.
///
/// Supported formats:
/// - `.da3` - Custom binary format with metadata (recommended)
/// - `.npy` - NumPy-compatible format for Python interop
/// - `.raw` - Raw float32 binary
/// - `.png` - Visualization (depth colormap)
///
/// Usage:
/// ```swift
/// let writer = DA3OutputWriter()
/// try writer.save(result, to: "output/scene_001", format: .da3)
///
/// // Later, in a separate process:
/// let reader = DA3OutputReader()
/// let data = try reader.load(from: "output/scene_001.da3")
/// // Use data.depth, data.rays with 3DGS pipeline
/// ```
@available(macOS 14.0, iOS 17.0, *)
public final class DA3OutputWriter {
    
    // MARK: - Types
    
    /// Output format for depth/ray data
    public enum OutputFormat: String {
        case da3 = "da3"      // Custom format with metadata
        case npy = "npy"      // NumPy compatible
        case raw = "raw"      // Raw float32 binary
        case png = "png"      // Visualization only
    }
    
    /// Configuration for output writing
    public struct Config {
        public enum VisualizationBackend: String {
            case cpu
            case metal
        }

        /// Include ray data (increases file size)
        public var includeRays: Bool = true
        /// Include confidence maps
        public var includeConfidence: Bool = true
        /// Colormap for PNG visualization
        public var colormap: Colormap = .spectral
        /// Depth mapping convention for PNG visualization
        public var depthVisualizationStyle: DepthVisualizationStyle = .da3
        /// Invert depth visualization (so closer appears "hotter"/brighter)
        public var invertDepthVisualization: Bool = false
        /// Backend used to generate visualization PNGs (CPU is the most portable).
        public var visualizationBackend: VisualizationBackend = .cpu
        /// Compress data (for .da3 format)
        public var compress: Bool = true
        
        public init() {}
    }
    
    // MARK: - Properties
    
    public var config: Config
    
    // MARK: - Initialization
    
    public init(config: Config = Config()) {
        self.config = config
    }
    
    // MARK: - Save Methods
    
    /// Save DA3 result to file(s)
    ///
    /// - Parameters:
    ///   - result: DA3 inference result
    ///   - path: Output path (without extension)
    ///   - format: Output format
    ///   - imageInfo: Optional source image information
    public func save(
        _ result: DA3CoreML.Result,
        to path: String,
        format: OutputFormat = .da3,
        imageInfo: ImageInfo? = nil
    ) throws {
        switch format {
        case .da3:
            try saveDA3Format(result, to: path, imageInfo: imageInfo)
        case .npy:
            try saveNpyFormat(result, to: path)
        case .raw:
            try saveRawFormat(result, to: path)
        case .png:
            try savePngFormat(result, to: path)
        }
    }
    
    /// Save multiple results with automatic naming
    public func saveBatch(
        _ results: [(result: DA3CoreML.Result, imagePath: String)],
        to directory: String,
        format: OutputFormat = .da3
    ) throws {
        try FileManager.default.createDirectory(
            atPath: directory,
            withIntermediateDirectories: true
        )
        
        for (index, item) in results.enumerated() {
            let baseName = (item.imagePath as NSString).deletingPathExtension
            let fileName = (baseName as NSString).lastPathComponent
            let outputPath = "\(directory)/\(fileName)"
            
            let imageInfo = ImageInfo(
                sourcePath: item.imagePath,
                index: index,
                totalCount: results.count
            )
            
            try save(item.result, to: outputPath, format: format, imageInfo: imageInfo)
        }
    }
    
    // MARK: - DA3 Format (Custom with metadata)
    
    private func saveDA3Format(
        _ result: DA3CoreML.Result,
        to path: String,
        imageInfo: ImageInfo?
    ) throws {
        let fullPath = path.hasSuffix(".da3") ? path : "\(path).da3"
        
        var data = Data()
        
        // Magic number: "DA3C" (DA3 CoreML)
        data.append(contentsOf: [0x44, 0x41, 0x33, 0x43])
        
        // Version (uint16)
        var version: UInt16 = 1
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        
        // Flags (uint16): bit 0 = has rays, bit 1 = has confidence, bit 2 = compressed
        var flags: UInt16 = 0
        if config.includeRays && result.rays != nil { flags |= 1 }
        if config.includeConfidence { flags |= 2 }
        if config.compress { flags |= 4 }
        withUnsafeBytes(of: &flags) { data.append(contentsOf: $0) }
        
        // Dimensions
        let width = result.originalSize.width
        let height = result.originalSize.height
        var w = UInt32(width)
        var h = UInt32(height)
        withUnsafeBytes(of: &w) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &h) { data.append(contentsOf: $0) }
        
        // Depth range
        var minD = result.minDepth
        var maxD = result.maxDepth
        withUnsafeBytes(of: &minD) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &maxD) { data.append(contentsOf: $0) }
        
        // Inference time
        var inferenceTime = Float(result.inferenceTime)
        withUnsafeBytes(of: &inferenceTime) { data.append(contentsOf: $0) }
        
        // Timestamp
        var timestamp = UInt64(Date().timeIntervalSince1970 * 1000)
        withUnsafeBytes(of: &timestamp) { data.append(contentsOf: $0) }
        
        // Reserved (32 bytes for future use)
        data.append(contentsOf: [UInt8](repeating: 0, count: 32))
        
        // Depth data
        let depthData = try arrayToFloat32Data(result.depth)
        var depthSize = UInt32(depthData.count)
        withUnsafeBytes(of: &depthSize) { data.append(contentsOf: $0) }
        
        if config.compress {
            let compressed = try compressData(depthData)
            var compressedSize = UInt32(compressed.count)
            withUnsafeBytes(of: &compressedSize) { data.append(contentsOf: $0) }
            data.append(compressed)
        } else {
            var compressedSize = depthSize
            withUnsafeBytes(of: &compressedSize) { data.append(contentsOf: $0) }
            data.append(depthData)
        }
        
        // Depth confidence (if enabled)
        if config.includeConfidence {
            let confData = try arrayToFloat32Data(result.depthConfidence)
            var confSize = UInt32(confData.count)
            withUnsafeBytes(of: &confSize) { data.append(contentsOf: $0) }
            
            if config.compress {
                let compressed = try compressData(confData)
                var compressedSize = UInt32(compressed.count)
                withUnsafeBytes(of: &compressedSize) { data.append(contentsOf: $0) }
                data.append(compressed)
            } else {
                var compressedSize = confSize
                withUnsafeBytes(of: &compressedSize) { data.append(contentsOf: $0) }
                data.append(confData)
            }
        }
        
        // Ray data (if enabled and available)
        if config.includeRays, let rays = result.rays {
            let rayData = try arrayToFloat32Data(rays)
            var raySize = UInt32(rayData.count)
            withUnsafeBytes(of: &raySize) { data.append(contentsOf: $0) }
            
            if config.compress {
                let compressed = try compressData(rayData)
                var compressedSize = UInt32(compressed.count)
                withUnsafeBytes(of: &compressedSize) { data.append(contentsOf: $0) }
                data.append(compressed)
            } else {
                var compressedSize = raySize
                withUnsafeBytes(of: &compressedSize) { data.append(contentsOf: $0) }
                data.append(rayData)
            }
            
            // Ray confidence
            if config.includeConfidence, let rayConf = result.rayConfidence {
                let rayConfData = try arrayToFloat32Data(rayConf)
                var rayConfSize = UInt32(rayConfData.count)
                withUnsafeBytes(of: &rayConfSize) { data.append(contentsOf: $0) }
                
                if config.compress {
                    let compressed = try compressData(rayConfData)
                    var compressedSize = UInt32(compressed.count)
                    withUnsafeBytes(of: &compressedSize) { data.append(contentsOf: $0) }
                    data.append(compressed)
                } else {
                    var compressedSize = rayConfSize
                    withUnsafeBytes(of: &compressedSize) { data.append(contentsOf: $0) }
                    data.append(rayConfData)
                }
            }
        }
        
        try data.write(to: URL(fileURLWithPath: fullPath))
        
        // Write metadata JSON alongside
        if let info = imageInfo {
            let metadata = DA3Metadata(
                version: "1.0",
                format: "da3",
                width: width,
                height: height,
                depthMin: result.minDepth,
                depthMax: result.maxDepth,
                inferenceTime: result.inferenceTime,
                timestamp: Date(),
                sourceImage: info.sourcePath,
                hasRays: config.includeRays && result.rays != nil,
                hasConfidence: config.includeConfidence
            )
            
            let jsonPath = fullPath.replacingOccurrences(of: ".da3", with: "_meta.json")
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let jsonData = try encoder.encode(metadata)
            try jsonData.write(to: URL(fileURLWithPath: jsonPath))
        }
    }
    
    // MARK: - NPY Format (NumPy compatible)
    
    private func saveNpyFormat(_ result: DA3CoreML.Result, to path: String) throws {
        // Save depth as .npy
        let depthPath = "\(path)_depth.npy"
        try saveAsNpy(result.depth, to: depthPath)
        
        // Save confidence
        if config.includeConfidence {
            let confPath = "\(path)_depth_conf.npy"
            try saveAsNpy(result.depthConfidence, to: confPath)
        }
        
        // Save rays
        if config.includeRays, let rays = result.rays {
            let rayPath = "\(path)_rays.npy"
            try saveAsNpy(rays, to: rayPath)
            
            if config.includeConfidence, let rayConf = result.rayConfidence {
                let rayConfPath = "\(path)_rays_conf.npy"
                try saveAsNpy(rayConf, to: rayConfPath)
            }
        }
    }
    
    private func saveAsNpy(_ array: MLMultiArray, to path: String) throws {
        // NPY format header
        var data = Data()
        
        // Magic number
        data.append(contentsOf: [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]) // "\x93NUMPY"
        
        // Version 1.0
        data.append(contentsOf: [0x01, 0x00])
        
        // Build header string
        let shape = array.shape.map { $0.intValue }
        let shapeStr = "(\(shape.map { String($0) }.joined(separator: ", "))\(shape.count == 1 ? "," : ""))"
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': \(shapeStr), }"
        
        // Pad header to 64-byte alignment
        let headerLen = header.count
        let padding = 64 - ((10 + headerLen) % 64)
        let paddedHeader = header + String(repeating: " ", count: padding - 1) + "\n"
        
        // Header length (little endian uint16)
        var headerLenLE = UInt16(paddedHeader.count)
        withUnsafeBytes(of: &headerLenLE) { data.append(contentsOf: $0) }
        
        // Header
        data.append(paddedHeader.data(using: .ascii)!)
        
        // Data (float32)
        let floatData = try arrayToFloat32Data(array)
        data.append(floatData)
        
        try data.write(to: URL(fileURLWithPath: path))
    }
    
    // MARK: - Raw Format
    
    private func saveRawFormat(_ result: DA3CoreML.Result, to path: String) throws {
        let depthPath = "\(path)_depth.raw"
        let depthData = try arrayToFloat32Data(result.depth)
        try depthData.write(to: URL(fileURLWithPath: depthPath))
        
        // Write shape info
        let shape = result.depth.shape.map { $0.intValue }
        let infoPath = "\(path)_depth.shape"
        let shapeStr = shape.map { String($0) }.joined(separator: ",")
        try shapeStr.write(toFile: infoPath, atomically: true, encoding: .utf8)
        
        if config.includeRays, let rays = result.rays {
            let rayPath = "\(path)_rays.raw"
            let rayData = try arrayToFloat32Data(rays)
            try rayData.write(to: URL(fileURLWithPath: rayPath))
            
            let rayShape = rays.shape.map { $0.intValue }
            let rayInfoPath = "\(path)_rays.shape"
            let rayShapeStr = rayShape.map { String($0) }.joined(separator: ",")
            try rayShapeStr.write(toFile: rayInfoPath, atomically: true, encoding: .utf8)
        }
    }
    
    // MARK: - PNG Format (Visualization)
    
    private func savePngFormat(_ result: DA3CoreML.Result, to path: String) throws {
        let pngPath = path.hasSuffix(".png") ? path : "\(path)_depth.png"
        let image: CGImage = try {
            // Metal visualization is optional and only supports a subset of colormaps to keep the
            // shader small. Fall back to the CPU implementation for others.
            let canUseMetalColormap: Bool = {
                switch config.colormap {
                case .spectral, .turbo, .grayscale: return true
                default: return false
                }
            }()

            if config.visualizationBackend == .metal, canUseMetalColormap, let mp = DA3MetalPostProcessor.shared() {
                let width = result.originalSize.width
                let height = result.originalSize.height

                let (minD, maxD): (Float, Float) = {
                    switch config.depthVisualizationStyle {
                    case .depth:
                        return result.depthRange
                    case .da3:
                        // Match DA3 visualize_depth(): compute 2/98 percentiles over inverse depth.
                        let percentile: Float = 2.0
                        let maxSamples = 1_000_000
                        let count = result.depth.count
                        let step = Swift.max(1, count / maxSamples)

                        var samples = [Float]()
                        samples.reserveCapacity(Swift.min(maxSamples, count))

                        var i = 0
                        while i < count {
                            let d = result.depth[i].floatValue
                            if d > 0 {
                                samples.append(1.0 / d)
                            }
                            i += step
                        }

                        guard samples.count > 10 else { return (0, 0) }
                        samples.sort()

                        let n = samples.count
                        let p = percentile / 100.0
                        func clampIndex(_ idx: Int) -> Int { max(0, min(idx, n - 1)) }
                        let loIndex = clampIndex(Int((Float(n - 1) * p).rounded(.down)))
                        let hiIndex = clampIndex(Int((Float(n - 1) * (1.0 - p)).rounded(.down)))
                        var lo = samples[loIndex]
                        var hi = samples[hiIndex]
                        if lo == hi {
                            lo -= 1e-6
                            hi += 1e-6
                        }
                        return (lo, hi)
                    }
                }()

                return try mp.visualizeDepthToCGImage(
                    depth: result.depth,
                    width: width,
                    height: height,
                    depthMin: minD,
                    depthMax: maxD,
                    style: config.depthVisualizationStyle,
                    colormap: config.colormap,
                    invert: config.invertDepthVisualization
                )
            }

            return try result.depthAsCGImage(
                colormap: config.colormap,
                invert: config.invertDepthVisualization,
                style: config.depthVisualizationStyle
            )
        }()
        
        let url = URL(fileURLWithPath: pngPath)
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL,
            "public.png" as CFString,
            1,
            nil
        ) else {
            throw DA3Error.imageProcessingFailed("Failed to create PNG destination")
        }
        
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else {
            throw DA3Error.imageProcessingFailed("Failed to write PNG")
        }
    }
    
    // MARK: - Helpers
    
    private func arrayToFloat32Data(_ array: MLMultiArray) throws -> Data {
        // IMPORTANT: CoreML can return MLMultiArrays with non-standard strides (padding / non-contiguous
        // layouts). Linear indexing via `array[i]` can therefore read incorrect values (often NaNs),
        // especially for multi-channel outputs like rays. Always materialize via a stride-aware reader.
        let values: [Float]
        if let reader = try? MLMultiArrayFloatReader(array) {
            values = reader.readAll()
        } else {
            // Fallback (slow, but correct for arbitrary dtypes)
            values = array.shape.isEmpty ? [] : (0..<array.count).map { idx in array[idx].floatValue }
        }

        // Pack as little-endian float32 (native endian on Apple Silicon is little-endian).
        return values.withUnsafeBytes { Data($0) }
    }
    
    private func compressData(_ data: Data) throws -> Data {
        // Use zlib compression
        return try (data as NSData).compressed(using: .zlib) as Data
    }
}

// MARK: - Supporting Types

/// Metadata for DA3 output files
public struct DA3Metadata: Codable {
    public let version: String
    public let format: String
    public let width: Int
    public let height: Int
    public let depthMin: Float
    public let depthMax: Float
    public let inferenceTime: TimeInterval
    public let timestamp: Date
    public let sourceImage: String?
    public let hasRays: Bool
    public let hasConfidence: Bool
}

/// Source image information
public struct ImageInfo {
    public let sourcePath: String
    public let index: Int
    public let totalCount: Int
    
    public init(sourcePath: String, index: Int = 0, totalCount: Int = 1) {
        self.sourcePath = sourcePath
        self.index = index
        self.totalCount = totalCount
    }
}

// MARK: - DA3OutputReader

/// Reader for DA3 output files
@available(macOS 14.0, iOS 17.0, *)
public final class DA3OutputReader {
    
    /// Loaded DA3 data
    public struct LoadedData {
        public let depth: [Float]
        public let depthConfidence: [Float]?
        public let rays: [Float]?
        public let rayConfidence: [Float]?
        public let width: Int
        public let height: Int
        public let depthMin: Float
        public let depthMax: Float
        public let metadata: DA3Metadata?
    }
    
    public init() {}
    
    /// Load DA3 data from file
    public func load(from path: String) throws -> LoadedData {
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        
        // Check magic number
        guard data.count >= 4,
              data[0] == 0x44, data[1] == 0x41,
              data[2] == 0x33, data[3] == 0x43 else {
            throw DA3Error.invalidInput("Not a valid DA3 file")
        }
        
        var offset = 4
        
        // Version
        let version: UInt16 = try readUInt16LE(from: data, offset: &offset)
        
        guard version == 1 else {
            throw DA3Error.invalidInput("Unsupported DA3 version: \(version)")
        }
        
        // Flags
        let flags: UInt16 = try readUInt16LE(from: data, offset: &offset)
        
        let hasRays = (flags & 1) != 0
        let hasConfidence = (flags & 2) != 0
        let isCompressed = (flags & 4) != 0
        
        // Dimensions
        let width: UInt32 = try readUInt32LE(from: data, offset: &offset)
        
        let height: UInt32 = try readUInt32LE(from: data, offset: &offset)
        
        // Depth range
        let depthMin: Float = try readFloat32LE(from: data, offset: &offset)
        
        let depthMax: Float = try readFloat32LE(from: data, offset: &offset)
        
        // Skip inference time, timestamp, reserved
        offset += 4 + 8 + 32
        
        // Read depth data
        let depth = try readFloatArray(from: data, offset: &offset, compressed: isCompressed)
        
        // Read confidence if present
        var depthConfidence: [Float]?
        if hasConfidence {
            depthConfidence = try readFloatArray(from: data, offset: &offset, compressed: isCompressed)
        }
        
        // Read rays if present
        var rays: [Float]?
        var rayConfidence: [Float]?
        if hasRays {
            rays = try readFloatArray(from: data, offset: &offset, compressed: isCompressed)
            
            if hasConfidence {
                rayConfidence = try readFloatArray(from: data, offset: &offset, compressed: isCompressed)
            }
        }
        
        // Try to load metadata
        let metaPath = path.replacingOccurrences(of: ".da3", with: "_meta.json")
        var metadata: DA3Metadata?
        if FileManager.default.fileExists(atPath: metaPath) {
            let metaData = try Data(contentsOf: URL(fileURLWithPath: metaPath))
            metadata = try JSONDecoder().decode(DA3Metadata.self, from: metaData)
        }
        
        return LoadedData(
            depth: depth,
            depthConfidence: depthConfidence,
            rays: rays,
            rayConfidence: rayConfidence,
            width: Int(width),
            height: Int(height),
            depthMin: depthMin,
            depthMax: depthMax,
            metadata: metadata
        )
    }
    
    private func readFloatArray(from data: Data, offset: inout Int, compressed: Bool) throws -> [Float] {
        let originalSize: UInt32 = try readUInt32LE(from: data, offset: &offset)
        
        let storedSize: UInt32 = try readUInt32LE(from: data, offset: &offset)
        
        guard storedSize <= Int.max else {
            throw DA3Error.invalidInput("Invalid blob size: \(storedSize)")
        }
        guard offset >= 0, offset + Int(storedSize) <= data.count else {
            throw DA3Error.invalidInput("Unexpected EOF while reading DA3 blob (storedSize=\(storedSize))")
        }
        let storedData = data.subdata(in: offset..<(offset + Int(storedSize)))
        offset += Int(storedSize)
        
        let floatData: Data
        if compressed {
            floatData = try (storedData as NSData).decompressed(using: .zlib) as Data
        } else {
            floatData = storedData
        }
        
        guard originalSize % 4 == 0 else {
            throw DA3Error.invalidInput("Invalid float blob size (not multiple of 4): \(originalSize)")
        }
        guard floatData.count == Int(originalSize) else {
            throw DA3Error.invalidInput("Blob size mismatch (expected \(originalSize) bytes, got \(floatData.count))")
        }

        let count = Int(originalSize) / 4
        var result = [Float](repeating: 0, count: count)
        _ = result.withUnsafeMutableBytes { dst in
            floatData.copyBytes(to: dst)
        }
        
        return result
    }

    // MARK: - Safe (unaligned) binary readers

    private func readUInt16LE(from data: Data, offset: inout Int) throws -> UInt16 {
        var v: UInt16 = 0
        try readBytes(into: &v, from: data, offset: &offset)
        return UInt16(littleEndian: v)
    }

    private func readUInt32LE(from data: Data, offset: inout Int) throws -> UInt32 {
        var v: UInt32 = 0
        try readBytes(into: &v, from: data, offset: &offset)
        return UInt32(littleEndian: v)
    }

    private func readFloat32LE(from data: Data, offset: inout Int) throws -> Float {
        var bits: UInt32 = 0
        try readBytes(into: &bits, from: data, offset: &offset)
        return Float(bitPattern: UInt32(littleEndian: bits))
    }

    private func readBytes<T>(into value: inout T, from data: Data, offset: inout Int) throws {
        let size = MemoryLayout<T>.size
        guard offset >= 0, offset + size <= data.count else {
            throw DA3Error.invalidInput("Unexpected EOF while reading DA3 header (need \(size) bytes)")
        }
        _ = Swift.withUnsafeMutableBytes(of: &value) { dst in
            data.copyBytes(to: dst, from: offset..<(offset + size))
        }
        offset += size
    }
}
