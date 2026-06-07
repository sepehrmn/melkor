import Foundation
import XCTest
@testable import DA3CoreML

@available(macOS 14.0, iOS 17.0, *)
final class DA3PointCloudPLYWriterTests: XCTestCase {
    func testWriteBinaryPointCloud_hasExpectedHeaderAndLayout() throws {
        let writer = DA3PointCloudPLYWriter()

        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // DA3CoreMLTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // Package root
        let tmpDir = root.appendingPathComponent(".build/da3_test_tmp/\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)

        let path = tmpDir.appendingPathComponent("0_pcd.ply").path

        var vertexData = Data()
        // Vertex 0: (1,2,3) RGB (255,0,0)
        vertexData.appendFloat32LE(1)
        vertexData.appendFloat32LE(2)
        vertexData.appendFloat32LE(3)
        vertexData.append(UInt8(255))
        vertexData.append(UInt8(0))
        vertexData.append(UInt8(0))
        // Vertex 1: (4,5,6) RGB (0,255,0)
        vertexData.appendFloat32LE(4)
        vertexData.appendFloat32LE(5)
        vertexData.appendFloat32LE(6)
        vertexData.append(UInt8(0))
        vertexData.append(UInt8(255))
        vertexData.append(UInt8(0))

        try writer.writeBinaryPointCloud(vertexCount: 2, vertexData: vertexData, to: path)

        let fileData = try Data(contentsOf: URL(fileURLWithPath: path))
        let marker = Data("end_header\n".utf8)
        guard let headerRange = fileData.range(of: marker) else {
            return XCTFail("Missing end_header marker")
        }
        let bodyStart = headerRange.upperBound
        let header = String(data: fileData.subdata(in: 0..<bodyStart), encoding: .ascii)
        XCTAssertNotNil(header)
        XCTAssertTrue(header!.contains("format binary_little_endian 1.0"))
        XCTAssertTrue(header!.contains("element vertex 2"))
        XCTAssertTrue(header!.contains("property float x"))
        XCTAssertTrue(header!.contains("property uchar blue"))

        let body = fileData.subdata(in: bodyStart..<fileData.count)
        XCTAssertEqual(body.count, 2 * 15, "Expected 15 bytes per vertex (3 float32 + 3 uchar)")

        XCTAssertEqual(body.readFloat32LE(at: 0), 1, accuracy: 1e-6)
        XCTAssertEqual(body.readFloat32LE(at: 4), 2, accuracy: 1e-6)
        XCTAssertEqual(body.readFloat32LE(at: 8), 3, accuracy: 1e-6)
        XCTAssertEqual(body.readUInt8(at: 12), 255)
        XCTAssertEqual(body.readUInt8(at: 13), 0)
        XCTAssertEqual(body.readUInt8(at: 14), 0)

        let v1 = 15
        XCTAssertEqual(body.readFloat32LE(at: v1 + 0), 4, accuracy: 1e-6)
        XCTAssertEqual(body.readFloat32LE(at: v1 + 4), 5, accuracy: 1e-6)
        XCTAssertEqual(body.readFloat32LE(at: v1 + 8), 6, accuracy: 1e-6)
        XCTAssertEqual(body.readUInt8(at: v1 + 12), 0)
        XCTAssertEqual(body.readUInt8(at: v1 + 13), 255)
        XCTAssertEqual(body.readUInt8(at: v1 + 14), 0)
    }

    func testMergeBinaryPointCloudPLYFiles_sumsVertexCountsAndConcatenatesBodies() throws {
        let writer = DA3PointCloudPLYWriter()

        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let tmpDir = root.appendingPathComponent(".build/da3_test_tmp/\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)

        let f0 = tmpDir.appendingPathComponent("0_pcd.ply").path
        let f1 = tmpDir.appendingPathComponent("1_pcd.ply").path
        let combined = tmpDir.appendingPathComponent("combined_pcd.ply").path

        // 0_pcd: 1 vertex
        var d0 = Data()
        d0.appendFloat32LE(1)
        d0.appendFloat32LE(2)
        d0.appendFloat32LE(3)
        d0.append(UInt8(10)); d0.append(UInt8(20)); d0.append(UInt8(30))
        try writer.writeBinaryPointCloud(vertexCount: 1, vertexData: d0, to: f0)

        // 1_pcd: 2 vertices
        var d1 = Data()
        d1.appendFloat32LE(4)
        d1.appendFloat32LE(5)
        d1.appendFloat32LE(6)
        d1.append(UInt8(40)); d1.append(UInt8(50)); d1.append(UInt8(60))
        d1.appendFloat32LE(7)
        d1.appendFloat32LE(8)
        d1.appendFloat32LE(9)
        d1.append(UInt8(70)); d1.append(UInt8(80)); d1.append(UInt8(90))
        try writer.writeBinaryPointCloud(vertexCount: 2, vertexData: d1, to: f1)

        // Create a pre-existing combined file to ensure it doesn't get counted as an input.
        try writer.writeBinaryPointCloud(vertexCount: 99, vertexData: Data(repeating: 0, count: 99 * 15), to: combined)

        try writer.mergeBinaryPointCloudPLYFiles(inputDir: tmpDir.path, outputPath: combined)

        let out = try Data(contentsOf: URL(fileURLWithPath: combined))
        let marker = Data("end_header\n".utf8)
        guard let headerRange = out.range(of: marker) else {
            return XCTFail("Missing end_header marker in merged PLY")
        }
        let bodyStart = headerRange.upperBound
        let header = String(data: out.subdata(in: 0..<bodyStart), encoding: .ascii)
        XCTAssertNotNil(header)
        XCTAssertTrue(header!.contains("element vertex 3"), "Expected 1+2 vertices from inputs (excluding existing combined)")

        let body = out.subdata(in: bodyStart..<out.count)
        XCTAssertEqual(body.count, 3 * 15)

        // First vertex should match 0_pcd (lexicographically first).
        XCTAssertEqual(body.readFloat32LE(at: 0), 1, accuracy: 1e-6)
        XCTAssertEqual(body.readUInt8(at: 12), 10)

        // Second vertex should match first of 1_pcd.
        let v1 = 15
        XCTAssertEqual(body.readFloat32LE(at: v1 + 0), 4, accuracy: 1e-6)
        XCTAssertEqual(body.readUInt8(at: v1 + 12), 40)

        // Third vertex should match second of 1_pcd.
        let v2 = 30
        XCTAssertEqual(body.readFloat32LE(at: v2 + 0), 7, accuracy: 1e-6)
        XCTAssertEqual(body.readUInt8(at: v2 + 12), 70)
    }
}

@available(macOS 14.0, iOS 17.0, *)
private extension Data {
    mutating func append(_ value: UInt8) {
        var v = value
        Swift.withUnsafeBytes(of: &v) { append(contentsOf: $0) }
    }

    mutating func appendFloat32LE(_ value: Float) {
        var bits = value.bitPattern.littleEndian
        Swift.withUnsafeBytes(of: &bits) { append(contentsOf: $0) }
    }

    func readUInt8(at offset: Int) -> UInt8 {
        return self[self.startIndex.advanced(by: offset)]
    }

    func readFloat32LE(at offset: Int) -> Float {
        var bits: UInt32 = 0
        _ = Swift.withUnsafeMutableBytes(of: &bits) { dst in
            self.copyBytes(to: dst, from: offset..<(offset + 4))
        }
        return Float(bitPattern: UInt32(littleEndian: bits))
    }
}
