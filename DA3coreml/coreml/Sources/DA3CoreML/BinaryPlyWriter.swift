import Foundation

/// Binary PLY writer/merger for RGB point clouds (x,y,z + uchar r,g,b).
@available(macOS 14.0, iOS 17.0, *)
public final class DA3PointCloudPLYWriter {
    public init() {}

    public func writeBinaryPointCloud(vertexCount: Int, vertexData: Data, to path: String) throws {
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        var data = Data()
        data.append(contentsOf: "ply\n".utf8)
        data.append(contentsOf: "format binary_little_endian 1.0\n".utf8)
        data.append(contentsOf: "element vertex \(vertexCount)\n".utf8)
        data.append(contentsOf: "property float x\n".utf8)
        data.append(contentsOf: "property float y\n".utf8)
        data.append(contentsOf: "property float z\n".utf8)
        data.append(contentsOf: "property uchar red\n".utf8)
        data.append(contentsOf: "property uchar green\n".utf8)
        data.append(contentsOf: "property uchar blue\n".utf8)
        data.append(contentsOf: "end_header\n".utf8)
        data.append(vertexData)

        try data.write(to: url, options: .atomic)
    }

    public func mergeBinaryPointCloudPLYFiles(
        inputDir: String,
        outputPath: String,
        fileSuffix: String = "_pcd.ply",
        excludeFileName: String = "combined_pcd.ply"
    ) throws {
        let fm = FileManager.default
        let inputURL = URL(fileURLWithPath: inputDir)
        let outURL = URL(fileURLWithPath: outputPath)

        let fileNames = try fm.contentsOfDirectory(atPath: inputURL.path)
            .filter { $0.hasSuffix(fileSuffix) }
            .filter { $0 != excludeFileName }
            .sorted { a, b in
                let aPrefix = String(a.dropLast(fileSuffix.count))
                let bPrefix = String(b.dropLast(fileSuffix.count))
                let ai = Int(aPrefix)
                let bi = Int(bPrefix)
                switch (ai, bi) {
                case let (x?, y?):
                    return x < y
                case (_?, nil):
                    return true
                case (nil, _?):
                    return false
                default:
                    return a < b
                }
            }

        guard !fileNames.isEmpty else {
            // If there are no inputs, still write an empty point cloud.
            try writeBinaryPointCloud(vertexCount: 0, vertexData: Data(), to: outputPath)
            return
        }

        // First pass: sum vertex counts.
        var totalVertices = 0
        for name in fileNames {
            let url = inputURL.appendingPathComponent(name)
            totalVertices += try readVertexCount(from: url)
        }

        try fm.createDirectory(
            at: outURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if fm.fileExists(atPath: outURL.path) {
            try fm.removeItem(at: outURL)
        }
        fm.createFile(atPath: outURL.path, contents: nil)

        let outHandle = try FileHandle(forWritingTo: outURL)
        defer { try? outHandle.close() }

        // Write merged header.
        let header =
            "ply\n" +
            "format binary_little_endian 1.0\n" +
            "element vertex \(totalVertices)\n" +
            "property float x\n" +
            "property float y\n" +
            "property float z\n" +
            "property uchar red\n" +
            "property uchar green\n" +
            "property uchar blue\n" +
            "end_header\n"
        try outHandle.write(contentsOf: Data(header.utf8))

        // Second pass: stream bodies.
        for name in fileNames {
            let url = inputURL.appendingPathComponent(name)
            let inHandle = try FileHandle(forReadingFrom: url)
            defer { try? inHandle.close() }

            let (_, remainder) = try readHeaderAndRemainder(from: inHandle)
            if !remainder.isEmpty {
                try outHandle.write(contentsOf: remainder)
            }

            while true {
                let chunk = try inHandle.read(upToCount: 1 << 20) ?? Data()
                if chunk.isEmpty { break }
                try outHandle.write(contentsOf: chunk)
            }
        }
    }

    private func readVertexCount(from url: URL) throws -> Int {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        let (headerData, _) = try readHeaderAndRemainder(from: handle)
        guard let headerStr = String(data: headerData, encoding: .ascii) else {
            throw DA3Error.invalidInput("Invalid PLY header encoding: \(url.path)")
        }

        for line in headerStr.split(separator: "\n") {
            if line.hasPrefix("element vertex ") {
                let parts = line.split(separator: " ")
                if let last = parts.last, let n = Int(last) {
                    return n
                }
            }
        }
        throw DA3Error.invalidInput("Missing 'element vertex' in PLY header: \(url.path)")
    }

    private func readHeaderAndRemainder(from handle: FileHandle) throws -> (header: Data, remainder: Data) {
        let marker = Data("end_header\n".utf8)
        var buf = Data()
        let maxHeaderBytes = 1 << 20

        while buf.count < maxHeaderBytes {
            let chunk = try handle.read(upToCount: 4096) ?? Data()
            if chunk.isEmpty {
                break
            }
            buf.append(chunk)

            if let range = buf.range(of: marker) {
                let headerEnd = range.upperBound
                let header = buf.subdata(in: 0..<headerEnd)
                let remainder = buf.subdata(in: headerEnd..<buf.count)
                return (header, remainder)
            }
        }

        throw DA3Error.invalidInput("PLY header too large or missing end_header")
    }
}
