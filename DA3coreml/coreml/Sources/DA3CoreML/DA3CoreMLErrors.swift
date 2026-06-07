import Foundation

/// Errors that can occur during DA3CoreML operations
public enum DA3Error: LocalizedError {
    case modelNotFound(String)
    case modelLoadFailed(String)
    case modelOutputMissing(String)
    case imageProcessingFailed(String)
    case invalidShape(String)
    case invalidInput(String)
    case inferenceError(String)
    case outOfMemory(String)
    case conversionFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let message):
            return "Model not found: \(message)"
        case .modelLoadFailed(let message):
            return "Failed to load model: \(message)"
        case .modelOutputMissing(let message):
            return "Model output missing: \(message)"
        case .imageProcessingFailed(let message):
            return "Image processing failed: \(message)"
        case .invalidShape(let message):
            return "Invalid tensor shape: \(message)"
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        case .inferenceError(let message):
            return "Inference error: \(message)"
        case .outOfMemory(let message):
            return "Out of memory: \(message)"
        case .conversionFailed(let message):
            return "Conversion failed: \(message)"
        }
    }
}
