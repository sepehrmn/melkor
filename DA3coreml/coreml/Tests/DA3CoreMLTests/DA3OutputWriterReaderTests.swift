import XCTest
import CoreML
@testable import DA3CoreML

@available(macOS 14.0, iOS 17.0, *)
final class DA3OutputWriterReaderTests: XCTestCase {
    func testDA3RoundTrip_compressed_withRaysAndConfidence() throws {
        let width = 7
        let height = 5

        let depth = try MLMultiArray(shape: [1, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
        let depthConf = try MLMultiArray(shape: [1, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
        let rays = try MLMultiArray(shape: [6, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
        let rayConf = try MLMultiArray(shape: [1, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)

        let depthPtr = UnsafeMutablePointer<Float>(OpaquePointer(depth.dataPointer))
        let depthConfPtr = UnsafeMutablePointer<Float>(OpaquePointer(depthConf.dataPointer))
        let raysPtr = UnsafeMutablePointer<Float>(OpaquePointer(rays.dataPointer))
        let rayConfPtr = UnsafeMutablePointer<Float>(OpaquePointer(rayConf.dataPointer))

        for i in 0..<depth.count {
            depthPtr[i] = 1.0 + Float(i) * 0.01
            depthConfPtr[i] = 1.0 + Float(i) * 0.001
            rayConfPtr[i] = 1.0 + Float(i) * 0.0001
        }
        for i in 0..<rays.count {
            raysPtr[i] = Float(i) * 0.1 - 10.0
        }

        let result = DA3CoreML.Result(
            depth: depth,
            depthConfidence: depthConf,
            rays: rays,
            rayConfidence: rayConf,
            originalSize: (width: width, height: height),
            inferenceTime: 0.123
        )

        var cfg = DA3OutputWriter.Config()
        cfg.includeRays = true
        cfg.includeConfidence = true
        cfg.compress = true
        let writer = DA3OutputWriter(config: cfg)

        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let tmpDir = root.appendingPathComponent(".build/da3_test_tmp/\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        let outBase = tmpDir.appendingPathComponent("sample").path

        try writer.save(result, to: outBase, format: .da3, imageInfo: nil)

        let reader = DA3OutputReader()
        let loaded = try reader.load(from: outBase + ".da3")

        XCTAssertEqual(loaded.width, width)
        XCTAssertEqual(loaded.height, height)
        XCTAssertEqual(loaded.depth.count, width * height)
        XCTAssertEqual(loaded.depthConfidence?.count, width * height)
        XCTAssertEqual(loaded.rays?.count, 6 * width * height)
        XCTAssertEqual(loaded.rayConfidence?.count, width * height)

        for i in 0..<(width * height) {
            XCTAssertEqual(loaded.depth[i], depthPtr[i], accuracy: 0)
            XCTAssertEqual(loaded.depthConfidence![i], depthConfPtr[i], accuracy: 0)
            XCTAssertEqual(loaded.rayConfidence![i], rayConfPtr[i], accuracy: 0)
        }
        for i in 0..<(6 * width * height) {
            XCTAssertEqual(loaded.rays![i], raysPtr[i], accuracy: 0)
        }
    }

    func testDA3RoundTrip_uncompressed_withoutRaysOrConfidence() throws {
        let width = 9
        let height = 4

        let depth = try MLMultiArray(shape: [1, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)
        let depthConf = try MLMultiArray(shape: [1, NSNumber(value: height), NSNumber(value: width)], dataType: .float32)

        let depthPtr = UnsafeMutablePointer<Float>(OpaquePointer(depth.dataPointer))
        let depthConfPtr = UnsafeMutablePointer<Float>(OpaquePointer(depthConf.dataPointer))
        for i in 0..<depth.count {
            depthPtr[i] = Float(i)
            depthConfPtr[i] = 1.0
        }

        let result = DA3CoreML.Result(
            depth: depth,
            depthConfidence: depthConf,
            rays: nil,
            rayConfidence: nil,
            originalSize: (width: width, height: height),
            inferenceTime: 0.0
        )

        var cfg = DA3OutputWriter.Config()
        cfg.includeRays = false
        cfg.includeConfidence = false
        cfg.compress = false
        let writer = DA3OutputWriter(config: cfg)

        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let tmpDir = root.appendingPathComponent(".build/da3_test_tmp/\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        let outBase = tmpDir.appendingPathComponent("sample").path

        try writer.save(result, to: outBase, format: .da3, imageInfo: nil)

        let reader = DA3OutputReader()
        let loaded = try reader.load(from: outBase + ".da3")

        XCTAssertEqual(loaded.width, width)
        XCTAssertEqual(loaded.height, height)
        XCTAssertEqual(loaded.depth.count, width * height)
        XCTAssertNil(loaded.depthConfidence)
        XCTAssertNil(loaded.rays)
        XCTAssertNil(loaded.rayConfidence)

        for i in 0..<(width * height) {
            XCTAssertEqual(loaded.depth[i], depthPtr[i], accuracy: 0)
        }
    }
}

