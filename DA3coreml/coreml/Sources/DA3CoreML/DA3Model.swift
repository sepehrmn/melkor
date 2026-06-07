import Foundation
import CoreML

/// Memory management utilities for DA3CoreML.
///
/// Provides intelligent memory management for large model inference:
/// - Dynamic RAM detection with configurable safety buffer
/// - Adaptive batch sizing based on available memory
/// - Memory pressure monitoring and automatic throttling
/// - Automatic cleanup between inference batches
///
/// Best Practices for 128GB RAM Systems:
/// - Never assume full RAM is available (OS, apps use 20-40%)
/// - Use 30% safety buffer by default
/// - Monitor memory pressure and throttle accordingly
/// - Release intermediate tensors aggressively
@available(macOS 14.0, iOS 17.0, *)
public final class MemoryManager {
    
    // MARK: - Singleton
    
    /// Shared memory manager instance
    public static let shared = MemoryManager()
    
    // MARK: - Types
    
    /// Memory pressure levels
    public enum MemoryPressure: String, CaseIterable {
        case nominal    // < 50% used - full speed
        case warning    // 50-70% used - reduce batch size
        case critical   // 70-85% used - single image only
        case terminal   // > 85% used - abort operations
        
        /// Recommended batch multiplier for this pressure level
        public var batchMultiplier: Double {
            switch self {
            case .nominal: return 1.0
            case .warning: return 0.5
            case .critical: return 0.25
            case .terminal: return 0.0
            }
        }
    }
    
    /// Memory statistics
    public struct MemoryStats {
        /// Total physical RAM in bytes
        public let totalRAM: UInt64
        /// Currently available RAM in bytes
        public let availableRAM: UInt64
        /// RAM used by this process in bytes
        public let processRAM: UInt64
        /// Current memory pressure level
        public let pressure: MemoryPressure
        /// Percentage of RAM available
        public let availablePercent: Double
        
        /// Total RAM in GB
        public var totalGB: Double { Double(totalRAM) / 1_073_741_824 }
        /// Available RAM in GB
        public var availableGB: Double { Double(availableRAM) / 1_073_741_824 }
        /// Process RAM in GB
        public var processGB: Double { Double(processRAM) / 1_073_741_824 }
        
        /// Human-readable description
        public var description: String {
            String(format: "RAM: %.1f/%.1f GB available (%.0f%%), pressure: %@",
                   availableGB, totalGB, availablePercent * 100, pressure.rawValue)
        }
    }
    
    /// Configuration for memory management
    public struct Config {
        /// Safety buffer as percentage of total RAM (0.0-1.0)
        /// Default: 0.30 (30%) - never use more than 70% of total RAM
        public var safetyBufferPercent: Double = 0.30
        
        /// Minimum RAM to keep free in GB
        /// Default: 4GB for system stability
        public var minimumFreeGB: Double = 4.0
        
        /// Maximum percentage of available RAM to use per operation
        /// Default: 0.60 (60% of available)
        public var maxUsagePercent: Double = 0.60
        
        /// Enable automatic memory cleanup between batches
        public var autoCleanup: Bool = true
        
        /// Enable memory pressure monitoring
        public var monitorPressure: Bool = true
        
        /// Abort if memory pressure reaches terminal
        public var abortOnTerminalPressure: Bool = true
        
        public init() {}
    }
    
    // MARK: - Properties
    
    private var _config: Config
    private var lastStats: MemoryStats?
    private let lock = NSLock()
    
    /// Thread-safe access to configuration
    public var config: Config {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _config
        }
        set {
            lock.lock()
            defer { lock.unlock() }
            _config = newValue
        }
    }
    
    // MARK: - Initialization
    
    private init() {
        self._config = Config()
    }
    
    /// Create a new MemoryManager instance with custom configuration
    /// Note: For most use cases, use `MemoryManager.shared` singleton instead
    public init(config: Config) {
        self._config = config
    }
    
    // MARK: - Memory Statistics
    
    /// Get current memory statistics
    public func getMemoryStats() -> MemoryStats {
        let totalRAM = getTotalRAM()
        let availableRAM = getAvailableRAM()
        let processRAM = getProcessRAM()
        
        let usedPercent = 1.0 - (Double(availableRAM) / Double(totalRAM))
        let pressure = calculatePressure(usedPercent: usedPercent)
        let availablePercent = Double(availableRAM) / Double(totalRAM)
        
        let stats = MemoryStats(
            totalRAM: totalRAM,
            availableRAM: availableRAM,
            processRAM: processRAM,
            pressure: pressure,
            availablePercent: availablePercent
        )
        
        lock.lock()
        lastStats = stats
        lock.unlock()
        
        return stats
    }
    
    /// Get total physical RAM
    private func getTotalRAM() -> UInt64 {
        var size: size_t = MemoryLayout<UInt64>.size
        var totalRAM: UInt64 = 0
        sysctlbyname("hw.memsize", &totalRAM, &size, nil, 0)
        return totalRAM
    }
    
    /// Get available RAM (approximation using vm_statistics)
    private func getAvailableRAM() -> UInt64 {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        
        guard result == KERN_SUCCESS else {
            // Fallback: estimate 50% available
            return getTotalRAM() / 2
        }
        
        let pageSize = UInt64(vm_page_size)
        let freePages = UInt64(stats.free_count)
        let inactivePages = UInt64(stats.inactive_count)
        let purgablePages = UInt64(stats.purgeable_count)
        
        // Available = free + inactive + purgeable (can be reclaimed)
        return (freePages + inactivePages + purgablePages) * pageSize
    }
    
    /// Get RAM used by current process
    private func getProcessRAM() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        guard result == KERN_SUCCESS else {
            return 0
        }
        
        return info.resident_size
    }
    
    /// Calculate memory pressure level
    private func calculatePressure(usedPercent: Double) -> MemoryPressure {
        switch usedPercent {
        case ..<0.50: return .nominal
        case 0.50..<0.70: return .warning
        case 0.70..<0.85: return .critical
        default: return .terminal
        }
    }
    
    // MARK: - Memory Budget Calculation
    
    /// Calculate safe memory budget for an operation
    ///
    /// - Returns: Maximum memory in bytes that can be safely used
    public func getSafeMemoryBudget() -> UInt64 {
        let stats = getMemoryStats()
        let totalRAM = Double(stats.totalRAM)
        let availableRAM = Double(stats.availableRAM)
        
        // Calculate budget with safety buffer
        let maxFromTotal = totalRAM * (1.0 - config.safetyBufferPercent)
        let maxFromAvailable = availableRAM * config.maxUsagePercent
        let minimumFreeBytes = config.minimumFreeGB * 1_073_741_824
        
        // Use the most conservative estimate
        var budget = min(maxFromTotal, maxFromAvailable)
        budget = min(budget, availableRAM - minimumFreeBytes)
        budget = max(0, budget)
        
        return UInt64(budget)
    }
    
    /// Calculate safe memory budget in GB
    public func getSafeMemoryBudgetGB() -> Double {
        return Double(getSafeMemoryBudget()) / 1_073_741_824
    }
    
    /// Calculate optimal batch size for given model and image size
    ///
    /// - Parameters:
    ///   - modelMemoryGB: Estimated memory per image for the model
    ///   - overhead: Additional overhead factor (default: 1.5 for intermediate tensors)
    /// - Returns: Recommended batch size
    public func calculateOptimalBatchSize(modelMemoryGB: Double, overhead: Double = 1.5) -> Int {
        let stats = getMemoryStats()
        let budgetGB = getSafeMemoryBudgetGB()
        
        // Account for pressure
        let pressureMultiplier = stats.pressure.batchMultiplier
        let effectiveBudgetGB = budgetGB * pressureMultiplier
        
        // Calculate batch size
        let memoryPerImage = modelMemoryGB * overhead
        let batchSize = Int(effectiveBudgetGB / memoryPerImage)
        
        // Ensure at least 1
        return max(1, batchSize)
    }
    
    /// Check if an operation with given memory requirement is safe
    ///
    /// - Parameter requiredGB: Memory required in GB
    /// - Returns: Whether the operation is safe to proceed
    public func isSafeToAllocate(requiredGB: Double) -> Bool {
        let budgetGB = getSafeMemoryBudgetGB()
        return requiredGB <= budgetGB
    }
    
    /// Check memory pressure and throw if critical
    public func checkMemoryPressure() throws {
        let stats = getMemoryStats()
        
        if config.abortOnTerminalPressure && stats.pressure == .terminal {
            throw DA3Error.outOfMemory(
                "Memory pressure is terminal (\(String(format: "%.1f", (1 - stats.availablePercent) * 100))% used). " +
                "Free up memory or reduce batch size."
            )
        }
    }
    
    // MARK: - Memory Cleanup
    
    /// Force memory cleanup
    ///
    /// Hints to the system to release cached memory.
    /// Call this between batches for large model inference.
    /// Note: Actual memory release depends on system state.
    public func cleanup() {
        #if os(macOS)
        // Request memory pressure relief from malloc zones
        // This is a hint to the system, not guaranteed
        malloc_zone_pressure_relief(nil, 0)
        #endif
    }
    
    /// Execute a closure with automatic memory cleanup
    ///
    /// Wraps the operation in an autoreleasepool and triggers cleanup after.
    /// This ensures temporary objects are released promptly.
    ///
    /// - Parameter body: Closure to execute
    /// - Returns: Result of the closure
    public func withMemoryCleanup<T>(_ body: () throws -> T) rethrows -> T {
        let result = try autoreleasepool {
            try body()
        }
        // Cleanup after autoreleasepool has drained
        if config.autoCleanup {
            cleanup()
        }
        return result
    }
    
    /// Execute a closure with memory pressure check
    ///
    /// - Parameter body: Closure to execute
    /// - Returns: Result of the closure
    /// - Throws: DA3Error.outOfMemory if memory pressure is too high
    public func withMemoryCheck<T>(_ body: () throws -> T) throws -> T {
        try checkMemoryPressure()
        return try withMemoryCleanup(body)
    }
    
    // MARK: - Logging
    
    /// Print current memory statistics
    public func logMemoryStats() {
        let stats = getMemoryStats()
        print("📊 Memory: \(stats.description)")
    }
}

// MARK: - Memory-Aware Model Configuration

@available(macOS 14.0, iOS 17.0, *)
extension DA3CoreML.Config {
    
    /// Create a memory-aware configuration
    ///
    /// Automatically adjusts settings based on available system memory.
    ///
    /// - Parameters:
    ///   - modelSize: Desired model size
    ///   - safetyBuffer: Safety buffer percentage (default: 0.30)
    /// - Returns: Optimized configuration
    public static func memoryAware(
        modelSize: DA3CoreML.ModelSize = .base,
        safetyBuffer: Double = 0.30
    ) -> DA3CoreML.Config {
        var memConfig = MemoryManager.Config()
        memConfig.safetyBufferPercent = safetyBuffer
        
        let memManager = MemoryManager(config: memConfig)
        let budgetGB = memManager.getSafeMemoryBudgetGB()
        let stats = memManager.getMemoryStats()
        
        var config = DA3CoreML.Config()
        config.modelSize = modelSize
        
        // Calculate memory limit (leave 30% buffer from budget)
        config.memoryLimitGB = budgetGB * 0.70
        
        // Adjust settings based on memory pressure
        switch stats.pressure {
        case .nominal:
            config.maxBatchSize = memManager.calculateOptimalBatchSize(
                modelMemoryGB: modelSize.estimatedMemoryGB
            )
            config.maxTileSize = 1024
            
        case .warning:
            config.maxBatchSize = max(1, memManager.calculateOptimalBatchSize(
                modelMemoryGB: modelSize.estimatedMemoryGB
            ) / 2)
            config.maxTileSize = 768
            
        case .critical:
            config.maxBatchSize = 1
            config.maxTileSize = 512
            config.enableTiling = true
            
        case .terminal:
            // Use smallest possible settings
            config.maxBatchSize = 1
            config.maxTileSize = 256
            config.enableTiling = true
        }
        
        return config
    }
}

// MARK: - Tensor Memory Utilities

@available(macOS 14.0, iOS 17.0, *)
public extension MLMultiArray {
    
    /// Estimated memory size in bytes
    var estimatedMemorySize: Int {
        let elementSize: Int
        switch dataType {
        case .float16: elementSize = 2
        case .float32: elementSize = 4
        case .float64, .double: elementSize = 8
        case .int32: elementSize = 4
        default: elementSize = 4  // Conservative fallback for unknown types
        }
        return count * elementSize
    }
    
    /// Estimated memory size in MB
    var estimatedMemoryMB: Double {
        Double(estimatedMemorySize) / 1_048_576
    }
    
    /// Estimated memory size in GB
    var estimatedMemoryGB: Double {
        Double(estimatedMemorySize) / 1_073_741_824
    }
}
