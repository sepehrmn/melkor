import Foundation
import CoreML

@available(macOS 14.0, iOS 17.0, *)
internal enum MLMultiArrayCast {
    /// Returns `array` if it is already float32, otherwise returns a float32 copy.
    static func toFloat32(_ array: MLMultiArray) throws -> MLMultiArray {
        if array.dataType == .float32 { return array }
        let reader = try MLMultiArrayFloatReader(array)
        let out = try MLMultiArray(shape: array.shape, dataType: .float32)
        let outPtr = UnsafeMutablePointer<Float>(OpaquePointer(out.dataPointer))
        let values = reader.readAll()
        for i in 0..<values.count {
            outPtr[i] = values[i]
        }
        return out
    }
}

