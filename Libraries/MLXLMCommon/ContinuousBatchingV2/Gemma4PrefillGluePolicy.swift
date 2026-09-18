// Copyright © 2026 Eigen Labs.
import Foundation

/// Latched per model, never a process-global mutable execution policy.
/// Source-only candidate: all fusion remains OFF without exact opt-in.
public struct Gemma4PrefillGluePolicy: Sendable {
    public let enabled: Bool
    public let vectorized: Bool
    public let chained: Bool
    public let branchPrefix: Bool
    public let scatter: Bool
    public let expertTail: Bool
    public let geglu: Bool
    public let routeCounting: Bool
    public let routeBitset: Bool
    public let routeParallelScan: Bool

    public init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        enabled = environment["DARKBLOOM_GEMMA4_PREFILL_GLUE"] == "1"
        vectorized = environment["DARKBLOOM_GEMMA4_PREFILL_GLUE_VEC4"] == "1"
        chained = environment["DARKBLOOM_GEMMA4_PREFILL_GLUE_CHAIN"] != "0"
        branchPrefix = environment["DARKBLOOM_GEMMA4_PREFILL_BRANCH_PREFIX"] == "1"
        scatter = environment["DARKBLOOM_GEMMA4_PREFILL_PRENORM_GATHER"] == "1"
        expertTail = environment["DARKBLOOM_GEMMA4_PREFILL_EXPERT_TAIL_FUSION"] == "1"
        geglu = environment["DARKBLOOM_GEMMA4_PROMPT_GLUE"] == "1"
        routeCounting = environment["DARKBLOOM_ROUTE_CSORT_PREFILL"] == "1"
        routeBitset = environment["DARKBLOOM_ROUTE_CSORT_PREFILL_BITSET"] != "0"
        routeParallelScan = environment["DARKBLOOM_GEMMA4_PROMPT_GLUE2"] == "1"
    }

    /// Shape alone must never admit an MTP verification rectangle.
    public func context(scheduledPrefill: Bool, eligibleModel: Bool) -> Context? {
        guard enabled, scheduledPrefill, eligibleModel else { return nil }
        return Context(vectorized: vectorized, chained: chained, branchPrefix: branchPrefix,
                       scatter: scatter, expertTail: expertTail, geglu: geglu,
                       routeCounting: routeCounting, routeBitset: routeBitset,
                       routeParallelScan: routeParallelScan)
    }

    public struct Context: Sendable {
        public let vectorized: Bool
        public let chained: Bool
        public let branchPrefix: Bool
        public let scatter: Bool
        public let expertTail: Bool
        public let geglu: Bool
        public let routeCounting: Bool
        public let routeBitset: Bool
        public let routeParallelScan: Bool
        fileprivate init(vectorized: Bool, chained: Bool, branchPrefix: Bool, scatter: Bool,
                         expertTail: Bool, geglu: Bool, routeCounting: Bool,
                         routeBitset: Bool, routeParallelScan: Bool) {
            self.vectorized = vectorized
            self.chained = chained
            self.branchPrefix = branchPrefix
            self.scatter = scatter
            self.expertTail = expertTail
            self.geglu = geglu
            self.routeCounting = routeCounting
            self.routeBitset = routeBitset
            self.routeParallelScan = routeParallelScan
        }

        public func rows(shape: [Int], inputBF16: Bool, weightShape: [Int],
                         weightBF16: Bool, eps: Float) -> Int? {
            guard shape.count == 3, shape[0] > 0, shape[1] >= 2,
                  shape[2] == 2816, inputBF16, weightBF16,
                  weightShape == [2816], eps == Float(1e-6) else { return nil }
            let (rows, overflow) = shape[0].multipliedReportingOverflow(by: shape[1])
            // MLXFast's grid bridge uses Int32. Reject overflow before lowering.
            guard !overflow, rows <= Int(Int32.max) else { return nil }
            return rows
        }

        public func scatterAssignments(rows: Int, indexShape: [Int], indicesUInt32: Bool) -> Int? {
            guard scatter else { return nil }
            return expertAssignments(rows: rows, indexShape: indexShape, indicesUInt32: indicesUInt32)
        }

        public func expertAssignments(rows: Int, indexShape: [Int], indicesUInt32: Bool) -> Int? {
            guard scatter || (expertTail && chained), rows > 0,
                indexShape == [rows, 8], indicesUInt32 else { return nil }
            let (assignments, overflow) = rows.multipliedReportingOverflow(by: 8)
            guard !overflow, assignments >= 64, assignments <= Int(Int32.max) else { return nil }
            return assignments
        }

        public struct GeGLUPlan: Sendable {
            public let rows: Int
            public let columns: Int
            public let pitch: Int
            public let upOffset: Int
            public let threads: Int
            public let outputShape: [Int]
        }

        public func gegluPlan(shape: [Int], inputBF16: Bool, compiledBaseline: Bool,
                              fusedHidden: Int? = nil) -> GeGLUPlan? {
            guard geglu, compiledBaseline, inputBF16, shape.count >= 2,
                shape.allSatisfy({ $0 > 0 && $0 <= Int(Int32.max) }) else { return nil }
            let columns = fusedHidden ?? shape[shape.count - 1]
            guard columns == 2112 || columns == 704,
                shape.last == (fusedHidden == nil ? columns : 2 * columns) else { return nil }
            var rows = 1
            for extent in shape.dropLast() {
                let product = rows.multipliedReportingOverflow(by: extent)
                guard !product.overflow else { return nil }
                rows = product.partialValue
            }
            let threads = rows.multipliedReportingOverflow(by: columns / 4)
            guard rows >= 1024, rows <= Int(Int32.max), !threads.overflow,
                threads.partialValue <= Int(Int32.max) else { return nil }
            var outputShape = shape
            outputShape[outputShape.count - 1] = columns
            return GeGLUPlan(rows: rows, columns: columns, pitch: shape[shape.count - 1],
                upOffset: fusedHidden == nil ? 0 : columns, threads: threads.partialValue,
                outputShape: outputShape)
        }

        public struct RoutePlan: Sendable {
            public let rows: Int
            public let assignments: Int
            public let blocks: Int
            public let parallelScan: Bool
            public let bitset: Bool
        }

        public func routePlan(scoreShape: [Int]) -> RoutePlan? {
            guard routeCounting, scoreShape.count == 3, scoreShape[0] > 0,
                scoreShape[1] >= 2, scoreShape[2] == 128 else { return nil }
            let rows = scoreShape[0].multipliedReportingOverflow(by: scoreShape[1])
            guard !rows.overflow else { return nil }
            let count = rows.partialValue.multipliedReportingOverflow(by: 8)
            guard !count.overflow, count.partialValue > 64, count.partialValue <= (1 << 28) else { return nil }
            let blocks = (count.partialValue + 255) / 256
            return RoutePlan(rows: rows.partialValue, assignments: count.partialValue, blocks: blocks,
                parallelScan: routeParallelScan && rows.partialValue >= 1024 && blocks >= 8 && blocks.isMultiple(of: 8),
                bitset: routeBitset && count.partialValue >= 4096)
        }
    }

    /// Nonblocking backing metadata is required for widened pointer accesses.
    /// Row-contiguous views alone do not establish an eight-byte alignment.
    public static func permitsVectorLoad(available: Bool, rowContiguous: Bool,
                                         byteOffset: UInt, allocatedBytes: UInt) -> Bool {
        available && rowContiguous && allocatedBytes > 0 && byteOffset.isMultiple(of: 8)
    }
}
