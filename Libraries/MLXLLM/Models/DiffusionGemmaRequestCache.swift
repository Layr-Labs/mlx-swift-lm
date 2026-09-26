// Native request-owned committed encoder storage.
import Foundation
import MLX
import MLXLMCommon

/// Per-request committed encoder state. Denoising reads snapshots and NEVER
/// writes provisional canvas K/V here. Uses native full/windowed sequence stores.
public final class DiffusionGemmaRequestCache {
    public let configuration: DiffusionGemmaTextConfiguration
    public let maximumSequenceLength: Int
    /// Experimental page-backed state uses native gathers plus the unchanged
    /// SDPA graph. It is NOT the numerically different scalar paged kernel.
    public var usesPagedStorage: Bool { pagedBackend != nil }
    private let pagedBackend: PagedKVBackend?
    private var failed = false
    private(set) var rows: [any CBv2SequenceKV]
    private var importedPrefixOwner: DiffusionGemmaPrefixCheckpoint?
    private(set) var owner: UUID?
    private(set) var committedTokenBatches = [MLXArray]()
    private(set) var committedTokenCount = 0
    private(set) var windowOrder = DiffusionGemmaWindowOrder()
    public var position: Int { rows.first?.absoluteOffset ?? 0 }
    public var retainedBytes: Int { rows.reduce(0) { $0 + $1.byteCount } }

    func retainImportedPrefix(_ checkpoint: DiffusionGemmaPrefixCheckpoint) {
        if checkpoint.isImported { importedPrefixOwner = checkpoint }
    }

    deinit {
        // Restored rows/graphs may alias the import's charged buffers. Retire
        // every native row before releasing that external destination permit.
        if let pagedBackend { pagedBackend.release(rows.map { Optional($0) }) }
        rows.removeAll(keepingCapacity: false)
        committedTokenBatches.removeAll(keepingCapacity: false)
        importedPrefixOwner = nil
    }

    public init(
        configuration: DiffusionGemmaTextConfiguration, expectedPromptLength: Int,
        maximumSequenceLength: Int? = nil, pagedBackend: PagedKVBackend? = nil
    ) throws {
        let maximum = maximumSequenceLength ?? configuration.maxPositionEmbeddings
        guard maximum > 0, maximum <= configuration.maxPositionEmbeddings,
            expectedPromptLength >= 0, expectedPromptLength <= maximum
        else {
            throw DiffusionGemmaModelError.invalidInput("prompt capacity")
        }
        self.configuration = configuration
        self.maximumSequenceLength = maximum
        self.pagedBackend = pagedBackend
        if let pagedBackend {
            let expected = configuration.diffusionPagedLayerKinds
            guard pagedBackend.layerKinds == expected else {
                throw DiffusionGemmaModelError.invalidInput("paged cache geometry")
            }
            let state = try pagedBackend.makeSequenceState(
                layerKinds: expected,
                promptLength: expectedPromptLength, maxLength: maximum)
            guard state.count == expected.count, state.allSatisfy({ $0 != nil }) else {
                pagedBackend.release(state)
                throw DiffusionGemmaModelError.invalidInput("incomplete paged cache")
            }
            rows = state.compactMap { $0 }
            return
        }
        rows = configuration.layerTypes.map { kind in
            if kind == "sliding_attention" {
                return CBv2WindowedSequenceKV(
                    window: configuration.slidingWindow,
                    kvHeads: configuration.keyValueHeads, headDim: configuration.headDimension)
            }
            return CBv2FullSequenceKV(
                promptLength: expectedPromptLength,
                maxLength: maximum,
                kvHeads: configuration.globalKeyValueHeads ?? configuration.keyValueHeads,
                headDim: configuration.globalHeadDimension)
        }
    }

    public func snapshots() -> [(keys: MLXArray, values: MLXArray, offset: Int)] {
        rows.filter { $0.retainedCount > 0 }.map { $0.snapshot() }
    }

    /// Explicit graph roots for the owner's evaluation/retirement fence.
    public func stateArrays() -> [MLXArray] {
        snapshots().flatMap { [$0.keys, $0.values] }
    }

    func recordCommittedTokens(
        _ tokens: MLXArray, restoredWindowOrder: DiffusionGemmaWindowOrder? = nil
    ) {
        // Keep a separate array context, not the caller's mutable wrapper.
        committedTokenBatches.append(tokens[0..., 0...])
        windowOrder =
            restoredWindowOrder
            ?? windowOrder.appending(
                tokens.dim(1), priorPosition: committedTokenCount,
                window: configuration.slidingWindow)
        committedTokenCount += tokens.dim(1)
    }

    func validate(
        for config: DiffusionGemmaTextConfiguration, owner requestedOwner: UUID, appendCount: Int,
        kvDType: DType
    ) throws {
        guard !failed, configuration == config, rows.count == config.layerCount,
            position == committedTokenCount,
            rows.allSatisfy({ $0.absoluteOffset == position }), appendCount >= 0,
            appendCount <= maximumSequenceLength, position <= maximumSequenceLength - appendCount
        else { throw DiffusionGemmaModelError.invalidInput("cache identity/position/capacity") }
        if let pagedBackend, appendCount > pagedBackend.pool.config.maxPrefillChunk {
            throw DiffusionGemmaModelError.invalidInput(
                "paged cache chunk exceeds configured bound")
        }
        if let pagedBackend, !pagedBackend.pool.layerDTypes.allSatisfy({ $0 == kvDType }) {
            throw DiffusionGemmaModelError.invalidInput("paged cache native dtype")
        }
        if let owner, owner != requestedOwner {
            throw DiffusionGemmaModelError.invalidInput("cache model owner")
        }
        owner = requestedOwner
    }

    /// Page reads refer to mutable shared storage. An explicit native step
    /// finishes before another step can write/recycle it, including direct SDK
    /// calls that are not run by a scheduler. This path is experimental and is
    /// never advertised as a fused paged-attention speedup.
    func finishPagedStep(_ outputs: [MLXArray]) throws {
        guard usesPagedStorage else { return }
        try MLX.withError { errors in
            eval(outputs.isEmpty ? stateArrays() : outputs)
            StreamOrDevice.default.stream.synchronize()
            try errors.check()
        }
    }

    func performStep<Result>(_ body: () throws -> Result) throws -> Result {
        guard let pagedBackend else { return try body() }
        guard !failed else {
            throw DiffusionGemmaModelError.invalidInput("retired failed paged request")
        }
        return try pagedBackend.performNativeBlockStep(
            body,
            onFailure: {
                failed = true
                pagedBackend.release(rows.map { Optional($0) })
                rows.removeAll(keepingCapacity: false)
                committedTokenBatches.removeAll(keepingCapacity: false)
                importedPrefixOwner = nil
            })
    }
}
