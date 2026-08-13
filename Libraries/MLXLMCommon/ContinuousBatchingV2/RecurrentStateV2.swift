// RecurrentStateV2.swift
//
// Request-owned recurrent state for hybrid CBv2 target models. Recurrent
// state is deliberately separate from attention KV: it has fixed per-request
// residency, model-specific tensor shapes, and transactional step lifecycle.

import Foundation
import MLX

/// Explicit model/backend feature gates. Attention-only models inherit the
/// historical all-enabled defaults; first-generation recurrent adapters opt
/// out of paths whose state semantics have not been proven.
public struct CBv2ModelCapabilities: Sendable, Equatable {
    public var supportsPrefixReuse: Bool
    public var supportsPagedKV: Bool
    public var supportsCompiledDecode: Bool
    public var supportsPackedPrefill: Bool
    public var supportsMTP: Bool

    public init(
        supportsPrefixReuse: Bool = true,
        supportsPagedKV: Bool = true,
        supportsCompiledDecode: Bool = true,
        supportsPackedPrefill: Bool = true,
        supportsMTP: Bool = true
    ) {
        self.supportsPrefixReuse = supportsPrefixReuse
        self.supportsPagedKV = supportsPagedKV
        self.supportsCompiledDecode = supportsCompiledDecode
        self.supportsPackedPrefill = supportsPackedPrefill
        self.supportsMTP = supportsMTP
    }

    public static let attentionOnly = CBv2ModelCapabilities()
    public static let initialRecurrentTarget = CBv2ModelCapabilities(
        supportsPrefixReuse: false,
        supportsPagedKV: false,
        supportsCompiledDecode: false,
        supportsPackedPrefill: false,
        supportsMTP: false)
}

public protocol CBv2ModelCapabilityProviding {
    var cbv2Capabilities: CBv2ModelCapabilities { get }
}

/// Tensor shapes for one recurrent model layer. Shapes include the singleton
/// request row so `elements` is also the exact per-request element count.
public struct CBv2RecurrentLayerStateSpec: Sendable, Equatable {
    public let modelLayerIndex: Int
    public let convShape: [Int]
    public let convDType: DType
    public let ssmShape: [Int]
    public let ssmDType: DType

    public init(
        modelLayerIndex: Int,
        convShape: [Int], convDType: DType,
        ssmShape: [Int], ssmDType: DType = .float32
    ) {
        self.modelLayerIndex = modelLayerIndex
        self.convShape = convShape
        self.convDType = convDType
        self.ssmShape = ssmShape
        self.ssmDType = ssmDType
    }
}

public enum CBv2RecurrentStateError: Error, Equatable {
    case invalidSpec(String)
    case byteCountOverflow
    case lifecycleViolation(String)
}

/// Complete fixed-residency description for one request.
public struct CBv2RecurrentStateSpec: Sendable, Equatable {
    /// One committed generation plus the two pending generations retained by
    /// chained decode and serial MTP verification before finalization.
    public static let maximumLiveGenerations = 3

    public let layers: [CBv2RecurrentLayerStateSpec]

    public init(layers: [CBv2RecurrentLayerStateSpec]) {
        self.layers = layers
    }

    /// Exact fixed bytes for one live request. Every multiplication and sum
    /// is checked; malformed or hostile configurations fail instead of
    /// wrapping into an artificially small admission charge.
    public func fixedBytesPerRequest() throws -> Int {
        var total = 0
        var seen = Set<Int>()
        for layer in layers {
            guard seen.insert(layer.modelLayerIndex).inserted else {
                throw CBv2RecurrentStateError.invalidSpec(
                    "duplicate recurrent model layer \(layer.modelLayerIndex)")
            }
            let conv = try Self.byteCount(shape: layer.convShape, dtype: layer.convDType)
            let ssm = try Self.byteCount(shape: layer.ssmShape, dtype: layer.ssmDType)
            let (layerBytes, layerOverflow) = conv.addingReportingOverflow(ssm)
            let (newTotal, totalOverflow) = total.addingReportingOverflow(layerBytes)
            guard !layerOverflow, !totalOverflow else {
                throw CBv2RecurrentStateError.byteCountOverflow
            }
            total = newTotal
        }
        return total
    }

    public func peakBytesPerRequest() throws -> Int {
        let (bytes, overflow) = try fixedBytesPerRequest().multipliedReportingOverflow(
            by: Self.maximumLiveGenerations)
        guard !overflow else { throw CBv2RecurrentStateError.byteCountOverflow }
        return bytes
    }

    public var modelLayerIndices: [Int] { layers.map(\.modelLayerIndex) }

    private static func byteCount(shape: [Int], dtype: DType) throws -> Int {
        guard !shape.isEmpty, shape.allSatisfy({ $0 >= 0 }) else {
            throw CBv2RecurrentStateError.invalidSpec("recurrent shapes must be nonempty and nonnegative")
        }
        var elements = 1
        for dimension in shape {
            let (next, overflow) = elements.multipliedReportingOverflow(by: dimension)
            guard !overflow else { throw CBv2RecurrentStateError.byteCountOverflow }
            elements = next
        }
        let (bytes, overflow) = elements.multipliedReportingOverflow(by: dtype.size)
        guard !overflow else { throw CBv2RecurrentStateError.byteCountOverflow }
        return bytes
    }
}

public struct CBv2RecurrentLayerState {
    public let conv: MLXArray?
    public let ssm: MLXArray?

    public init(conv: MLXArray?, ssm: MLXArray?) {
        self.conv = conv
        self.ssm = ssm
    }
}

/// One request's recurrent state. Mutated only by the engine thread.
public final class CBv2RecurrentRequestState {
    public let spec: CBv2RecurrentStateSpec
    public let byteCount: Int
    public var materializedByteCount: Int {
        // A captured verify window holds one full conv/SSM copy PER captured
        // position (its leading `[S, ...]` axis); a plain generation holds
        // exactly one.
        var generations = committed.isEmpty ? 0 : 1
        for generation in pending {
            let (next, overflow) = generations.addingReportingOverflow(
                generation.capturedPositions ?? 1)
            if overflow { return Int.max }
            generations = next
        }
        let (bytes, overflow) = byteCount.multipliedReportingOverflow(by: generations)
        return overflow ? Int.max : bytes
    }

    private struct Generation {
        let id: UInt64
        let layers: [Int: CBv2RecurrentLayerState]
        /// Non-nil marks a CAPTURED verify-window generation: every layer's
        /// conv/ssm array carries a leading per-position axis of this length
        /// (`[S, ...]` instead of the spec's `[1, ...]`). Such a generation
        /// can only leave `pending` through `commit(keepPositions:)` (which
        /// selects one position) or `rollback()`.
        let capturedPositions: Int?
    }

    private var committed: [Int: CBv2RecurrentLayerState] = [:]
    private var pending: [Generation] = []
    private var nextGeneration: UInt64 = 0
    private var bindingOpen = false
    public private(set) var isReleased = false

    public init(spec: CBv2RecurrentStateSpec) throws {
        self.spec = spec
        self.byteCount = try spec.fixedBytesPerRequest()
    }

    /// Bind the latest visible generation for a target forward. A chained
    /// decode therefore reads the previous still-pending generation without
    /// prematurely committing it.
    public func bind() throws -> CBv2RecurrentStateEvaluation {
        guard !isReleased else {
            throw CBv2RecurrentStateError.lifecycleViolation("bind after release")
        }
        guard !bindingOpen else {
            throw CBv2RecurrentStateError.lifecycleViolation("overlapping recurrent bindings")
        }
        bindingOpen = true
        let last = pending.last
        if let captured = last?.capturedPositions {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "bind over an unresolved captured verify window (\(captured) positions)")
        }
        let input = last?.layers ?? committed
        let evaluation = CBv2RecurrentStateEvaluation(
            owner: self, generation: nextGeneration, input: input,
            requiredLayers: Set(spec.modelLayerIndices))
        nextGeneration &+= 1
        return evaluation
    }

    fileprivate func evaluate(
        generation: UInt64, layers: [Int: CBv2RecurrentLayerState],
        capturedPositions: Int?
    ) throws {
        guard bindingOpen, !isReleased else {
            throw CBv2RecurrentStateError.lifecycleViolation("evaluate without an active binding")
        }
        pending.append(
            Generation(
                id: generation, layers: layers, capturedPositions: capturedPositions))
        bindingOpen = false
    }

    fileprivate func abandonBinding() {
        bindingOpen = false
    }

    fileprivate func commit(generation: UInt64, keepPositions: Int? = nil) throws {
        guard !isReleased, let first = pending.first, first.id == generation else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "recurrent commits must follow evaluation order")
        }
        if let captured = first.capturedPositions {
            guard let keep = keepPositions, (1 ... captured).contains(keep) else {
                throw CBv2RecurrentStateError.lifecycleViolation(
                    "captured commit requires keepPositions in 1...\(captured) "
                        + "(got \(String(describing: keepPositions)))")
            }
            // Device-side select: committing position `keep` keeps the state
            // AFTER consuming `keep` window tokens. The slice restores the
            // spec's leading singleton row axis, so downstream forwards read
            // the same shapes a plain generation would have staged.
            committed = first.layers.mapValues { layer in
                CBv2RecurrentLayerState(
                    conv: layer.conv.map { $0[(keep - 1) ..< keep] },
                    ssm: layer.ssm.map { $0[(keep - 1) ..< keep] })
            }
        } else {
            guard keepPositions == nil else {
                throw CBv2RecurrentStateError.lifecycleViolation(
                    "keepPositions is only valid for captured verify windows")
            }
            committed = first.layers
        }
        pending.removeFirst()
    }

    fileprivate func rollback(generation: UInt64) throws {
        guard !isReleased, let last = pending.last, last.id == generation else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "recurrent rollback must target the newest pending generation")
        }
        pending.removeLast()
    }

    /// Test/integration inspection of the latest visible state.
    public func state(modelLayerIndex: Int) -> CBv2RecurrentLayerState? {
        pending.last?.layers[modelLayerIndex] ?? committed[modelLayerIndex]
    }

    public func release() throws {
        guard !bindingOpen, pending.isEmpty else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "release with recurrent work still in flight")
        }
        committed.removeAll(keepingCapacity: false)
        isReleased = true
    }
}

/// A single forward's recurrent transaction. The model reads its row-local
/// input and stages one output for every declared recurrent layer. The engine
/// then calls `evaluate()` before submitting the arrays to MLX and later
/// chooses exactly one of `commit()` or `rollback()` at finalization.
public final class CBv2RecurrentStateEvaluation {
    private unowned let owner: CBv2RecurrentRequestState
    private let generation: UInt64
    private let input: [Int: CBv2RecurrentLayerState]
    private let requiredLayers: Set<Int>
    private var staged: [Int: CBv2RecurrentLayerState] = [:]
    private var stagedCapturedPositions: Int?
    private var evaluated = false
    private var finalized = false

    /// True when this transaction staged per-position captured stacks
    /// (MTP capture-verify) and must be finalized via
    /// `commit(keepPositions:)` or `rollback()`.
    public var isCaptured: Bool { stagedCapturedPositions != nil }

    fileprivate init(
        owner: CBv2RecurrentRequestState, generation: UInt64,
        input: [Int: CBv2RecurrentLayerState], requiredLayers: Set<Int>
    ) {
        self.owner = owner
        self.generation = generation
        self.input = input
        self.requiredLayers = requiredLayers
    }

    deinit {
        if !evaluated { owner.abandonBinding() }
    }

    public func inputState(modelLayerIndex: Int) -> CBv2RecurrentLayerState? {
        input[modelLayerIndex]
    }

    public func stage(modelLayerIndex: Int, conv: MLXArray, ssm: MLXArray) throws {
        guard stagedCapturedPositions == nil else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "cannot mix captured and plain recurrent stages")
        }
        try stageRaw(modelLayerIndex: modelLayerIndex, conv: conv, ssm: ssm)
    }

    private func stageRaw(modelLayerIndex: Int, conv: MLXArray, ssm: MLXArray) throws {
        guard !evaluated, requiredLayers.contains(modelLayerIndex) else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "stage for undeclared or already-evaluated recurrent layer \(modelLayerIndex)")
        }
        guard staged[modelLayerIndex] == nil else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "duplicate recurrent stage for layer \(modelLayerIndex)")
        }
        staged[modelLayerIndex] = CBv2RecurrentLayerState(conv: conv, ssm: ssm)
    }

    /// Stage one layer's CAPTURED per-position states for an MTP verify
    /// window: `conv`/`ssm` carry a leading position axis of length
    /// `positions` (`[S, ...]`) where index `s` is the state after consuming
    /// window token `s`. Index `positions - 1` must equal the state a plain
    /// `stage` would have produced for the whole window. Mixing captured and
    /// plain stages in one transaction is a lifecycle violation.
    public func stageCaptured(
        modelLayerIndex: Int, conv: MLXArray, ssm: MLXArray, positions: Int
    ) throws {
        guard positions >= 1 else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "captured stage requires at least one position")
        }
        if let existing = stagedCapturedPositions, existing != positions {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "captured position count changed mid-transaction "
                    + "(\(existing) -> \(positions))")
        }
        if stagedCapturedPositions == nil, !staged.isEmpty {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "cannot mix captured and plain recurrent stages")
        }
        guard conv.dim(0) == positions, ssm.dim(0) == positions else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "captured stacks must carry the position axis first "
                    + "(conv \(conv.shape), ssm \(ssm.shape), positions \(positions))")
        }
        try stageRaw(modelLayerIndex: modelLayerIndex, conv: conv, ssm: ssm)
        stagedCapturedPositions = positions
    }

    @discardableResult
    public func evaluate() throws -> [MLXArray] {
        guard !evaluated, Set(staged.keys) == requiredLayers else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "recurrent evaluation did not stage every declared layer")
        }
        try owner.evaluate(
            generation: generation, layers: staged,
            capturedPositions: stagedCapturedPositions)
        evaluated = true
        return staged.keys.sorted().flatMap { index in
            [staged[index]!.conv, staged[index]!.ssm].compactMap { $0 }
        }
    }

    public func commit() throws {
        guard evaluated, !finalized, stagedCapturedPositions == nil else {
            throw CBv2RecurrentStateError.lifecycleViolation("invalid recurrent commit")
        }
        try owner.commit(generation: generation)
        finalized = true
    }

    /// Finalize a captured verify window by committing the state after
    /// `keepPositions` consumed window tokens (1-based; the accepted prefix
    /// length including the seed column).
    public func commit(keepPositions: Int) throws {
        guard evaluated, !finalized, stagedCapturedPositions != nil else {
            throw CBv2RecurrentStateError.lifecycleViolation(
                "captured commit on a non-captured recurrent transaction")
        }
        try owner.commit(generation: generation, keepPositions: keepPositions)
        finalized = true
    }

    public func rollback() throws {
        guard evaluated, !finalized else {
            throw CBv2RecurrentStateError.lifecycleViolation("invalid recurrent rollback")
        }
        try owner.rollback(generation: generation)
        finalized = true
    }
}

/// Model-level target seam used by `CBv2SteppableLanguageModelAdapter`.
public protocol CBv2RecurrentLanguageModelForwardable: CBv2ModelCapabilityProviding {
    var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec { get }
    func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray
}

public protocol CBv2PositionedRecurrentLanguageModelForwardable:
    CBv2RecurrentLanguageModelForwardable
{
    func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray
}

/// Engine-facing optional refinement. The optional spec keeps the generic
/// language-model adapter source-compatible for attention-only models.
public protocol CBv2RecurrentSteppableModel: CBv2SteppableModel, CBv2ModelCapabilityProviding {
    var recurrentStateSpec: CBv2RecurrentStateSpec? { get }
    func forward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray
}

/// Optional recurrent forward with explicit request-owned model positions.
public protocol CBv2PositionedRecurrentSteppableModel: CBv2RecurrentSteppableModel {
    func forward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray
}
