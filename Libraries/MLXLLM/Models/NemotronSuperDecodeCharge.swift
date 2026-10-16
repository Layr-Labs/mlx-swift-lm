import MLX
import MLXLMCommon

/// Admitted bytes of one request in batched Super text decode, derived from
/// the loaded model's shapes and its ModelOpt graph workspace quotes. This is
/// configuration arithmetic, not a measured peak. It adds to the provider's
/// activation floor and backend allowance and never replaces them.
public struct NemotronSuperDecodeCharge: Sendable, Equatable {
    public enum Failure: Error, Equatable { case unsupportedExecution, invalidBound, overflow }

    /// Explicit arrays that one request row adds to a batched decode step.
    public struct RowWorkspace: Sendable, Equatable {
        /// Every ModelOpt call of the heaviest layer for one token, all live.
        public let modelOptLayerBytes: Int
        /// The output head's graph call for one row.
        public let outputHeadBytes: Int
        /// Output-dtype logits row and two float32 sampler copies.
        public let logitsBytes: Int
        /// The heaviest layer's other intermediates and the residual stream,
        /// priced as float32 and doubled for backend contiguous copies.
        public let trunkBytes: Int
        /// Split-K partial sums and attention partials of one row.
        public let backendScratchBytes: Int
        public let totalBytes: Int
    }

    /// Convolution and SSM state of every Mamba layer for one generation.
    public let recurrentBytesPerGeneration: Int
    /// Generations the engine keeps live per request: committed plus pending.
    public let recurrentGenerations: Int
    /// Keys and values of every attention layer for one token.
    public let attentionKVBytesPerToken: Int
    public let rowWorkspace: RowWorkspace

    /// Per-row allowance for backend scratch. MLX split-K products allocate at
    /// most 16Ki float32 partial sums per row (64 KiB) for each live product.
    public static let backendScratchBytesPerRow = 1 << 20

    /// Configuration arithmetic for the recurrent part; the engine's own
    /// `CBv2RecurrentStateSpec` gives the same value for a loaded model.
    public static func recurrentBytesPerGeneration(
        configuration: NemotronHConfiguration, convolutionDType: DType
    ) throws -> Int {
        try configuration.cbv2RecurrentStateSpec(activationDType: convolutionDType)
            .fixedBytesPerRequest()
    }

    public static func attentionKVBytesPerToken(
        configuration: NemotronHConfiguration, elementBytes: Int
    ) throws -> Int {
        try attentionKVBytesPerToken(
            kinds: configuration.cbv2LayerKinds, elementBytes: elementBytes)
    }

    static func attentionKVBytesPerToken(kinds: [CBv2LayerKind], elementBytes: Int) throws -> Int {
        guard elementBytes > 0 else { throw Failure.invalidBound }
        return try sum(
            kinds.map { try product($0.kvHeads, $0.headDim + $0.valueHeadDim, elementBytes) })
    }

    /// Non-ModelOpt intermediates of the heaviest layer and the residual
    /// stream for one token, in elements: Mamba projection, padded
    /// convolution, SiLU output, SSM output and gated norm; attention Q/K/V
    /// and merged heads; router scores, latent and routed rows, shared expert.
    static func trunkElements(configuration c: NemotronHConfiguration) throws -> Int {
        let inner = try product(c.mambaNumHeads, c.mambaHeadDim)
        let convolution = try sum([inner, try product(2, c.nGroups, c.ssmStateSize)])
        let mamba = try sum([
            inner, convolution, c.mambaNumHeads, try product(c.convKernel, convolution),
            try product(2, convolution), try product(4, inner),
        ])
        let headDim = c.headDim ?? c.hiddenSize / max(1, c.numAttentionHeads)
        let attention = try product(
            try sum([try product(2, c.numAttentionHeads), try product(2, c.numKeyValueHeads)]),
            headDim)
        let latent = c.moeLatentSize ?? c.hiddenSize
        let moe = try sum([
            try product(4, c.nRoutedExperts), latent, try product(c.numExpertsPerTok, latent),
            try product(2, c.moeSharedExpertIntermediateSize), c.hiddenSize,
        ])
        return try sum([try product(3, c.hiddenSize), max(mamba, attention, moe)])
    }

    /// Recurrent generations, attention KV at the sequence bound and one row
    /// of decode workspace.
    public func requestBytes(maximumTokens: Int) throws -> Int {
        guard maximumTokens > 0 else { throw Failure.invalidBound }
        return try Self.sum([
            try Self.product(recurrentGenerations, recurrentBytesPerGeneration),
            try Self.product(attentionKVBytesPerToken, maximumTokens),
            rowWorkspace.totalBytes,
        ])
    }

    /// A request's committed recurrent rows are views of its step's batched
    /// state buffers. After rows leave a batch the survivors keep those
    /// buffers alive until they commit again, so every live generation can
    /// hold up to `maximumRows - 1` departed rows beyond the per-request
    /// charges. Zero for serial decode.
    public func batchAliasBytes(maximumRows: Int) throws -> Int {
        guard maximumRows > 0 else { throw Failure.invalidBound }
        return try Self.product(
            maximumRows - 1, recurrentGenerations, recurrentBytesPerGeneration)
    }

    static func product(_ values: Int...) throws -> Int {
        var result = 1
        for value in values {
            guard value >= 0 else { throw Failure.invalidBound }
            let (next, overflow) = result.multipliedReportingOverflow(by: value)
            guard !overflow else { throw Failure.overflow }
            result = next
        }
        return result
    }

    static func sum(_ values: [Int]) throws -> Int {
        try values.reduce(0) { total, value in
            guard value >= 0 else { throw Failure.invalidBound }
            let (next, overflow) = total.addingReportingOverflow(value)
            guard !overflow else { throw Failure.overflow }
            return next
        }
    }
}

extension NemotronHModel {
    /// The batched decode charge of this loaded model. Requires the graph
    /// ModelOpt path on every installed module and a ModelOpt output head:
    /// the eager reference has no per-row workspace quote.
    public func superDecodeCharge(kvElementBytes: Int) throws -> NemotronSuperDecodeCharge {
        typealias Charge = NemotronSuperDecodeCharge
        guard let layerBytes = modelOptGraphPrefillBytesPerToken,
            let head = lmHead as? ModelOptCalibratedLinear, head.execution == .graph
        else { throw Charge.Failure.unsupportedExecution }
        let headBytes = try head.graphWorkspaceBytes(rows: 1)
        let logits = try Charge.product(vocabularySize, head.weightDType.size + 8)
        let trunk = try Charge.product(2, 4, try Charge.trunkElements(configuration: configuration))
        let scratch = Charge.backendScratchBytesPerRow
        let workspace = Charge.RowWorkspace(
            modelOptLayerBytes: layerBytes, outputHeadBytes: headBytes, logitsBytes: logits,
            trunkBytes: trunk, backendScratchBytes: scratch,
            totalBytes: try Charge.sum([layerBytes, headBytes, logits, trunk, scratch]))
        return Charge(
            recurrentBytesPerGeneration: try cbv2RecurrentStateSpec.fixedBytesPerRequest(),
            recurrentGenerations: CBv2RecurrentStateSpec.maximumLiveGenerations,
            attentionKVBytesPerToken: try Charge.attentionKVBytesPerToken(
                kinds: cbv2LayerKinds, elementBytes: kvElementBytes),
            rowWorkspace: workspace)
    }
}
