import Foundation
import MLX

public enum CBv2PagedGatheredAttentionAdmissionMode: Sendable, Equatable {
    case poolLifetime
    case stepOwned(CBv2PagedAttentionWorkProfile)
}

/// Explicit correctness-first page-backed attention limits. This selects
/// native page gather + ordinary MLX SDPA, NOT a fused asymmetric page kernel.
/// Bounds are checked, never silently substituted for model context length.
public struct CBv2PagedGatheredAttentionLimits: Sendable, Equatable {
    public let maximumBatchSize: Int
    public let maximumQueryTokens: Int
    public let maximumContextTokens: Int
    public let maximumInFlightGraphs: Int
    public let maximumScratchBytes: Int
    public let admissionMode: CBv2PagedGatheredAttentionAdmissionMode

    public init(
        maximumBatchSize: Int, maximumQueryTokens: Int, maximumContextTokens: Int,
        maximumInFlightGraphs: Int, maximumScratchBytes: Int,
        admissionMode: CBv2PagedGatheredAttentionAdmissionMode = .poolLifetime
    ) throws {
        // EngineV2 may build one chained successor before finalizing its
        // predecessor. One graph is therefore not a valid serving envelope.
        guard maximumBatchSize > 0, maximumBatchSize <= Int(Int32.max), maximumQueryTokens > 0,
            maximumContextTokens >= maximumQueryTokens,
            maximumContextTokens <= Int(Int32.max),
            (2 ... 8).contains(maximumInFlightGraphs), maximumScratchBytes > 0
        else {
            throw CBv2KVError.backendIneligible(
                reason: "invalid explicit asymmetric gathered-attention limits")
        }
        self.maximumBatchSize = maximumBatchSize
        self.maximumQueryTokens = maximumQueryTokens
        self.maximumContextTokens = maximumContextTokens
        self.maximumInFlightGraphs = maximumInFlightGraphs
        self.maximumScratchBytes = maximumScratchBytes
        self.admissionMode = admissionMode
    }

    /// Conservative reservation, not measured live usage. Sum all potentially
    /// coexisting layer graphs and one retained borrower-prefill generation.
    /// Each buffer gets its allocator bound before multiplicity is applied.
    public func scratchUpperBound(layerKinds: [CBv2LayerKind], pageSize: Int) throws -> Int {
        guard pageSize > 0 else { throw CBv2CompleteCheckpointError.invalidManifest }
        var bytes = 64 << 10  // bounded Swift control/table envelope
        func addBuffer(_ factors: [Int], itemBytes: Int = 4, copies: Int = 1) throws {
            var logical = itemBytes
            for value in factors {
                guard value >= 0, let next = CBv2KVGeometry.multiply(logical, value) else {
                    throw CBv2KVError.backendIneligible(
                        reason: "asymmetric attention scratch projection overflow")
                }
                logical = next
            }
            let allocation = try Memory.allocationFootprintUpperBound(byteCount: max(1, logical))
            guard let total = CBv2KVGeometry.multiply(allocation, copies),
                let next = CBv2KVGeometry.add(bytes, total)
            else {
                throw CBv2KVError.backendIneligible(
                    reason: "asymmetric attention scratch projection overflow")
            }
            bytes = next
        }
        for kind in layerKinds where kind.headDim != kind.valueHeadDim {
            guard kind.kvGeometry != nil, kind.queryHeads > 0,
                kind.queryHeads % kind.kvHeads == 0
            else {
                throw CBv2KVError.backendIneligible(
                    reason: "invalid asymmetric attention scratch geometry")
            }
            let visible: Int
            if case .slidingWindow(let window) = kind.attention {
                guard let exposure = CBv2KVGeometry.add(window, maximumQueryTokens), window > 0
                else {
                    throw CBv2CompleteCheckpointError.invalidManifest
                }
                visible = min(maximumContextTokens, exposure)
            } else {
                visible = maximumContextTokens
            }
            guard let padding = CBv2KVGeometry.multiply(2, pageSize),
                let padded = CBv2KVGeometry.add(visible, padding),
                let sinkWidth = CBv2KVGeometry.add(visible, 1)
            else {
                throw CBv2CompleteCheckpointError.invalidManifest
            }
            let b = maximumBatchSize
            let q = maximumQueryTokens
            let h = kind.queryHeads
            // Independent K/V gather, transpose/contiguous conversion, concat,
            // dtype/GQA conversion and graph temporaries. Price at FP32.
            for width in [kind.headDim, kind.valueHeadDim] {
                try addBuffer([b, h, padded, width], copies: 8)
                try addBuffer([b, h, q, width], copies: 4)
            }
            // Full unblocked score/softmax/mask envelope: query blocking may
            // reduce a peak but is not relied upon for admission correctness.
            try addBuffer([b, h, q, sinkWidth], copies: 8)
            try addBuffer([b, h, q, sinkWidth], itemBytes: 1, copies: 4)
            try addBuffer([b, padded, 32], copies: 2)  // padded transfer records/positions
            let segments = padded / pageSize + 2
            guard let rows = CBv2KVGeometry.multiply(b, segments),
                let witnesses = CBv2KVGeometry.multiply(rows, 4)
            else {
                throw CBv2CompleteCheckpointError.invalidManifest
            }
            try addBuffer([1], copies: witnesses)  // write/read/completion witnesses
        }
        guard let envelope = CBv2KVGeometry.multiply(bytes, maximumInFlightGraphs + 1),
            envelope <= maximumScratchBytes
        else {
            throw CBv2KVError.backendIneligible(
                reason:
                    "explicit asymmetric gathered-attention scratch ceiling is too small; context/query limits are not reduced automatically"
            )
        }
        return envelope
    }
}

extension PagedKVPool {
    var hasAsymmetricLayers: Bool { layerKinds.contains { $0.headDim != $0.valueHeadDim } }

    /// Acquire once before publishing a serving row. C remains held through
    /// pool/cache lifetime, including graph/copy coexistence. No M credit.
    func prepareGatheredAttention(maximumSequenceLength: Int) throws {
        guard hasAsymmetricLayers else { return }
        guard segmentGrant != nil, let limits = config.gatheredAttention,
            let admission = memoryAdmission, maximumSequenceLength > 0,
            maximumSequenceLength <= limits.maximumContextTokens
        else {
            throw CBv2KVError.backendIneligible(
                reason:
                    "asymmetric paging requires a bound segmented Admission owner and explicit sufficient gathered-attention limits"
            )
        }
        if usesStepOwnedAttention {
            guard admission.hasProcessMemoryOwner, attentionWorkEngineRefusal == nil,
                attentionWorkEnginePrepared
            else {
                throw CBv2KVError.backendIneligible(
                    reason: attentionWorkEngineRefusal
                        ?? "step-owned paging requires its real process owner and sealed engine envelope"
                )
            }
            return
        }
        if gatheredAttentionReservation == nil {
            gatheredAttentionReservation = try admission.reserveTransient(
                bytes: gatheredAttentionScratchBound)
        }
    }

    var gatheredAttentionIsPrepared: Bool {
        usesStepOwnedAttention
            ? attentionWorkEnginePrepared && attentionWorkEngineRefusal == nil
            : gatheredAttentionReservation != nil
    }
}
