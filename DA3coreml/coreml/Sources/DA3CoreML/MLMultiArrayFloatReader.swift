import Foundation
import CoreML

/// Stride-aware float reader for `MLMultiArray`.
///
/// CoreML can return `MLMultiArray`s with non-standard strides (padding / non-contiguous layouts).
/// Reading via linear indices (0..<count) can therefore produce incorrect results. This helper
/// provides safe float access and utilities to materialize a contiguous `[Float]` buffer in
/// row-major order.
@available(macOS 14.0, iOS 17.0, *)
public struct MLMultiArrayFloatReader {
    private let array: MLMultiArray
    public let shape: [Int]
    public let strides: [Int]
    private let dataType: MLMultiArrayDataType

    private let ptr16: UnsafePointer<Float16>?
    private let ptr32: UnsafePointer<Float>?

    public init(_ array: MLMultiArray) throws {
        self.array = array
        self.shape = array.shape.map { $0.intValue }
        self.strides = array.strides.map { $0.intValue }
        self.dataType = array.dataType

        switch array.dataType {
        case .float16:
            self.ptr16 = UnsafePointer<Float16>(OpaquePointer(array.dataPointer))
            self.ptr32 = nil
        case .float32:
            self.ptr16 = nil
            self.ptr32 = UnsafePointer<Float>(OpaquePointer(array.dataPointer))
        default:
            throw DA3Error.invalidInput("Unsupported MLMultiArray dtype for float reader: \(array.dataType)")
        }
    }

    public func readLinear(_ index: Int) -> Float {
        switch dataType {
        case .float16:
            return Float(ptr16![index])
        case .float32:
            return ptr32![index]
        default:
            return array[index].floatValue
        }
    }

    public func read(_ i0: Int, _ i1: Int, _ i2: Int, _ i3: Int) -> Float {
        let idx = i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + i3 * strides[3]
        return readLinear(idx)
    }

    public func read(_ i0: Int, _ i1: Int, _ i2: Int) -> Float {
        let idx = i0 * strides[0] + i1 * strides[1] + i2 * strides[2]
        return readLinear(idx)
    }

    public func read(_ i0: Int, _ i1: Int) -> Float {
        let idx = i0 * strides[0] + i1 * strides[1]
        return readLinear(idx)
    }

    /// Returns true if the array is stored contiguously in standard row-major order.
    public func isContiguousRowMajor() -> Bool {
        guard shape.count == strides.count, !shape.isEmpty else { return false }
        var expected = 1
        for i in shape.indices.reversed() {
            if strides[i] != expected { return false }
            expected *= max(1, shape[i])
        }
        return true
    }

    /// Materialize the array into a contiguous `[Float]` in row-major order.
    public func readAll() -> [Float] {
        var out = [Float](repeating: 0, count: array.count)

        if isContiguousRowMajor() {
            for i in 0..<array.count {
                out[i] = readLinear(i)
            }
            return out
        }

        var k = 0
        switch shape.count {
        case 1:
            for i0 in 0..<shape[0] {
                out[k] = readLinear(i0 * strides[0])
                k += 1
            }
        case 2:
            for i0 in 0..<shape[0] {
                for i1 in 0..<shape[1] {
                    out[k] = read(i0, i1)
                    k += 1
                }
            }
        case 3:
            for i0 in 0..<shape[0] {
                for i1 in 0..<shape[1] {
                    for i2 in 0..<shape[2] {
                        out[k] = read(i0, i1, i2)
                        k += 1
                    }
                }
            }
        case 4:
            for i0 in 0..<shape[0] {
                for i1 in 0..<shape[1] {
                    for i2 in 0..<shape[2] {
                        for i3 in 0..<shape[3] {
                            out[k] = read(i0, i1, i2, i3)
                            k += 1
                        }
                    }
                }
            }
        default:
            // Generic fallback: use CoreML's multi-index subscript.
            // This is slower, but it is correct for arbitrary ranks.
            var indices = [NSNumber](repeating: 0, count: shape.count)
            for linear in 0..<array.count {
                var t = linear
                for dim in shape.indices.reversed() {
                    let size = max(1, shape[dim])
                    indices[dim] = NSNumber(value: t % size)
                    t /= size
                }
                out[linear] = array[indices].floatValue
            }
        }

        return out
    }
}
