// Copyright 2026 the HuggingFace Team. All rights reserved.
// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0
// Native assembly of the pinned Transformers GLM5Next equations, f0705f991a71f05dad9569336b181119f5546f93.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

public final class GLM5NextRMSNorm: Module, UnaryLayer {
    public var weight: MLXArray
    let epsilon: Float
    /// Set once at target construction by `GLM5NextExecutionPolicy.normalization`.
    /// MLX's fused kernel rounds at the same point (`w * T(x * rsqrt)`) but sums
    /// the squares in its own order, so it is not bit-identical.
    var usesFusedKernel = false
    init(weight: MLXArray, epsilon: Float) {
        self.weight = weight
        self.epsilon = epsilon
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if usesFusedKernel { return MLXFast.rmsNorm(x, weight: weight, eps: epsilon) }
        let value = x.asType(.float32)
        return weight
            * (value * rsqrt(value.square().mean(axis: -1, keepDims: true) + epsilon)).asType(
                x.dtype)
    }
}

final class GLM5NextWeight: Module {
    var weight: MLXArray
    init(_ weight: MLXArray) { self.weight = weight }
}

/// Original affine rows are dequantized only after token lookup.
final class GLM5NextPackedEmbedding: Embedding, Quantized {
    let groupSize: Int
    let bits: Int
    let mode = QuantizationMode.affine
    var scales: MLXArray
    var biases: MLXArray
    init(
        weight: MLXArray, scales: MLXArray, biases: MLXArray,
        packing: GLM5NextWeightInventory.Packing
    ) {
        groupSize = packing.groupSize
        bits = packing.bits
        self.scales = scales
        self.biases = biases
        super.init(weight: weight)
        freeze()
    }
    public override var shape: (Int, Int) { (weight.dim(0), weight.dim(1) * 32 / bits) }
    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let indices = x.flattened()
        return dequantized(
            weight[indices], scales: scales[indices], biases: biases[indices],
            groupSize: groupSize, bits: bits
        ).reshaped(shape + [-1])
    }
    public override func asLinear(_ x: MLXArray) -> MLXArray {
        quantizedMM(x, weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
    }

    /// Token steps up to this size may take `streams` (decode, MTP verification).
    static let maximumStreamTokens = 8

    /// `broadcast(self(tokens).expandedDimensions(axis: 2), to: [B, T, copies, D])`,
    /// materialized in one launch: each value is MLX's `affine_dequantize`
    /// arithmetic (`T scale * uint8 q + T bias`) on the token's packed row,
    /// read in place instead of gathered, stored to every copy. Same bits as
    /// the gathers, dequantize and broadcast, without the copy the first
    /// collapse makes of a broadcast. nil off the default GPU stream or for
    /// layouts it does not read.
    func streams(_ tokens: MLXArray, copies: Int) -> MLXArray? {
        let (vocabulary, hidden) = shape
        guard tokens.ndim == 2, tokens.size > 0, copies > 0,
            [.int32, .uint32, .int64].contains(tokens.dtype), [4, 8].contains(bits),
            [32, 64, 128].contains(groupSize), hidden % groupSize == 0,
            weight.dtype == .uint32, weight.shape == [vocabulary, hidden * bits / 32],
            scales.dtype.isFloatingPoint, biases.dtype == scales.dtype,
            scales.shape == [vocabulary, hidden / groupSize], biases.shape == scales.shape,
            tokens.size * copies * hidden <= Int(Int32.max), glm5NextOnDefaultGPUStream()
        else { return nil }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            let rows = tokens.size
            return Self.streamsKernel(
                [tokens, weight, scales, biases],
                template: [
                    ("T", scales.dtype), ("Hidden", hidden), ("Vocabulary", vocabulary),
                    ("GroupSize", groupSize), ("Bits", bits), ("Copies", copies),
                ],
                grid: ((hidden + 255) / 256 * 256, rows, 1), threadGroup: (256, 1, 1),
                outputShapes: [tokens.shape + [copies, hidden]], outputDTypes: [scales.dtype])[0]
        #else
            return nil
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        /// One thread per (token, column). MLX's `take` wraps a negative
        /// index once; its `affine_dequantize` reads byte `column / (8 / Bits)`
        /// of the row, low bits first, and computes `scale * d + bias` with T
        /// scale and offset and a uint8 `d`.
        private static let streamsKernel = MLXFast.metalKernel(
            name: "glm5_next_packed_embedding_streams",
            inputNames: ["tokens", "weight", "scales", "biases"], outputNames: ["streams"],
            source: """
                const uint column = thread_position_in_grid.x;
                const uint row = thread_position_in_grid.y;
                if (column >= uint(Hidden)) { return; }
                int64_t token = static_cast<int64_t>(tokens[row]);
                if (token < 0) { token += Vocabulary; }
                constexpr uint pack_factor = 8 / Bits;
                const device uint8_t* w =
                    reinterpret_cast<const device uint8_t*>(weight) + size_t(token) * (Hidden * Bits / 8);
                const uint val = w[column / pack_factor];
                uint8_t d;
                if (Bits == 4) {
                    d = (val >> (Bits * (column % pack_factor))) & 0x0f;
                } else {
                    d = val;
                }
                const size_t group = size_t(token) * (Hidden / GroupSize) + column / GroupSize;
                T scale = scales[group];
                T bias = biases[group];
                const T value = scale * d + bias;
                for (uint copy = 0; copy < uint(Copies); ++copy) {
                    streams[(size_t(row) * Copies + copy) * Hidden + column] = value;
                }
                """,
            ensureRowContiguous: true)
    #endif
}

extension GLM5NextTensorLoader {
    func norm(_ prefix: String, epsilon: Double) throws -> GLM5NextRMSNorm {
        GLM5NextRMSNorm(weight: try array(prefix + ".weight"), epsilon: Float(epsilon))
    }
    func embedding(_ prefix: String) throws -> Embedding {
        let weight = try array(prefix + ".weight")
        return weight.dtype == .uint32
            ? GLM5NextPackedEmbedding(
                weight: weight, scales: try array(prefix + ".scales"),
                biases: try array(prefix + ".biases"), packing: try packing(prefix))
            : Embedding(weight: weight)
    }
    func mlp(
        _ c: GLM5NextTextConfiguration, prefix: String, intermediate: Int,
        projections: GLM5NextExecutionPolicy.Projections = .separate
    ) throws -> GLM5NextMLP {
        guard !inventory.storage.isPacked || releaseSourceTensors != nil else {
            throw GLM5NextLoadError.invalid("Q4 MLP loading requires source-cache eviction")
        }
        var gate = try linear(prefix + ".gate_proj")
        var up = try linear(prefix + ".up_proj")
        var fused: GLM5NextFusedProjection?
        if projections == .fusedInputs,
            let result = try GLM5NextFusedProjection.make(
                [gate, up], materialize: materializeFusedProjections)
        {
            (gate, up, fused) = (result.members[0], result.members[1], result.fused)
        }
        let mlp = try GLM5NextMLP(
            hiddenSize: c.hiddenSize, intermediateSize: intermediate,
            gateProj: gate, upProj: up,
            downProj: linear(prefix + ".down_proj"), swigluLimit: Float(c.swigluLimit))
        mlp.fusedGateUp = fused
        return mlp
    }
    func moe(
        _ c: GLM5NextTextConfiguration, prefix: String, materialize: ((MLXArray) throws -> Void)?,
        projections: GLM5NextExecutionPolicy.Projections = .separate
    ) throws -> GLM5NextMoE {
        guard !inventory.storage.isPacked || (materialize != nil && releaseSourceTensors != nil)
        else {
            throw GLM5NextLoadError.invalid(
                "Q4 expert loading requires materialization and source-cache eviction")
        }
        var banks: [String: SwitchLinear] = [:]
        for name in ["gate_proj", "up_proj", "down_proj"] {
            let output = name == "down_proj" ? c.hiddenSize : c.moeIntermediateSize
            let input = name == "down_proj" ? c.moeIntermediateSize : c.hiddenSize
            let members = (0 ..< c.nRoutedExperts).map { prefix + ".experts.\($0).\(name)" }
            // Resolved before any expert read; QuantizedSwitchLinear has one policy.
            let packing = try bankPacking(members)
            func bank(_ suffix: String) throws -> MLXArray {
                let names = members.map { $0 + "." + suffix }
                // Release the resolver's original aliases only after the stack
                // evaluation fence; release also runs on a failed load.
                defer { releaseSourceTensors?(names) }
                let value = try stacked(
                    names.map { try array($0, releaseSource: false) })
                // Required for full Q4 loading: release the individual source
                // arrays after each one-bank stack, not after whole-model eval.
                if let materialize { try materialize(value) }
                return value
            }
            let weight = try bank("weight")
            if weight.dtype == .uint32 {
                guard materialize != nil else {
                    throw GLM5NextLoadError.invalid(
                        "Q4 loading requires bounded expert-stack materialization")
                }
                guard let packing else {
                    throw GLM5NextLoadError.invalid("undeclared packing \(members[0])")
                }
                if packing.mode == .nvfp4 {
                    let globalScales = try bank("weight_scale_2").reshaped(c.nRoutedExperts)
                    try glm5NextRequirePositiveFinite(globalScales)
                    banks[name] = try QuantizedSwitchLinear(
                        nvfp4InputDims: input, outputDims: output, numExperts: c.nRoutedExperts,
                        weight: weight, scales: bank("scales"), globalScales: globalScales)
                    continue
                }
                banks[name] = try QuantizedSwitchLinear(
                    inputDims: input, outputDims: output, numExperts: c.nRoutedExperts,
                    weight: weight, scales: bank("scales"), biases: bank("biases"),
                    groupSize: packing.groupSize, bits: packing.bits)
            } else {
                banks[name] = SwitchLinear(
                    inputDims: input, outputDims: output, numExperts: c.nRoutedExperts,
                    weight: weight)
            }
        }
        guard let gate = banks["gate_proj"], let up = banks["up_proj"],
            let down = banks["down_proj"]
        else { throw GLM5NextLoadError.invalid("missing expert projection") }
        let experts = try GLM5NextExperts(
            hiddenSize: c.hiddenSize, intermediateSize: c.moeIntermediateSize,
            numExperts: c.nRoutedExperts, gateProj: gate, upProj: up, downProj: down,
            swigluLimit: Float(c.swigluLimit))
        let router = try GLM5NextRouter(
            hiddenSize: c.hiddenSize, numExperts: c.nRoutedExperts,
            topK: c.numExpertsPerTok, weight: array(prefix + ".gate.weight"),
            correctionBias: array(prefix + ".gate.e_score_correction_bias"),
            nGroup: c.nGroup, topkGroup: c.topkGroup, normTopkProb: c.normTopkProb,
            routedScalingFactor: Float(c.routedScalingFactor))
        return GLM5NextMoE(
            gate: router, experts: experts,
            sharedExperts: try mlp(
                c, prefix: prefix + ".shared_experts",
                intermediate: c.moeIntermediateSize * c.nSharedExperts,
                projections: projections))
    }
}

final class GLM5NextIndexNorm: Module {
    var weight: MLXArray
    var bias: MLXArray
    /// Set at target construction by `GLM5NextExecutionPolicy.normalization`:
    /// one launch per call on the default GPU stream (`GLM5NextIndexNormKernel`).
    var usesFusedKernel = false
    init(weight: MLXArray, bias: MLXArray) {
        self.weight = weight
        self.bias = bias
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if usesFusedKernel, let fused = GLM5NextIndexNormKernel.apply(x, weight: weight, bias: bias)
        {
            return fused
        }
        let k = x.asType(.float32)
        let centered = k - k.mean(axis: -1, keepDims: true)
        let normalized = centered * rsqrt(centered.square().mean(axis: -1, keepDims: true) + 1e-6)
        return (normalized * weight.asType(.float32) + bias.asType(.float32)).asType(x.dtype)
    }
}

final class GLM5NextIndexer: Module {
    @ModuleInfo(key: "wq_b") var query: Linear
    @ModuleInfo(key: "wk") var key: Linear
    @ModuleInfo(key: "weights_proj") var headWeights: Linear
    @ModuleInfo(key: "k_norm") var keyNorm: GLM5NextIndexNorm
    @ParameterInfo(key: "index_kpool_compress_ape") var positionBias: MLXArray
    @ParameterInfo(key: "index_kpool_compress_gate") var poolGate: MLXArray
    let configuration: GLM5NextTextConfiguration
    /// `key`/`headWeights` are the already loaded wk/weights_proj modules.
    init(
        _ c: GLM5NextTextConfiguration, prefix: String, loader: GLM5NextTensorLoader,
        key: Linear, headWeights: Linear
    ) throws {
        configuration = c
        _query.wrappedValue = try loader.linear(prefix + ".wq_b")
        _key.wrappedValue = key
        _headWeights.wrappedValue = headWeights
        _keyNorm.wrappedValue = GLM5NextIndexNorm(
            weight: try loader.array(prefix + ".k_norm.weight"),
            bias: try loader.array(prefix + ".k_norm.bias"))
        _positionBias.wrappedValue = try loader.array(prefix + ".index_kpool_compress_ape")
        _poolGate.wrappedValue = try loader.array(prefix + ".index_kpool_compress_gate")
    }
    func project(_ x: MLXArray, residual: MLXArray) -> (
        queries: MLXArray, keys: MLXArray, gates: MLXArray, weights: MLXArray
    ) {
        project(x, residual: residual, rawKeys: key(x), weights: headWeights(x))
    }

    /// `rawKeys` and `weights` are wk(x) and weights_proj(x).
    func project(_ x: MLXArray, residual: MLXArray, rawKeys: MLXArray, weights: MLXArray) -> (
        queries: MLXArray, keys: MLXArray, gates: MLXArray, weights: MLXArray
    ) {
        project(x, rawQueries: query(residual), rawKeys: rawKeys, weights: weights)
    }

    /// `project(_:residual:rawKeys:weights:)` with wq_b routed by `prefill`
    /// (`GLM5NextExecutionPolicy.PrefillProjections`; `.mlx` is the same graph).
    func project(
        _ x: MLXArray, residual: MLXArray, rawKeys: MLXArray, weights: MLXArray,
        prefill: GLM5NextExecutionPolicy.PrefillProjections
    ) throws -> (queries: MLXArray, keys: MLXArray, gates: MLXArray, weights: MLXArray) {
        project(
            x, rawQueries: try prefill.project(query, residual), rawKeys: rawKeys,
            weights: weights)
    }

    private func project(_ x: MLXArray, rawQueries: MLXArray, rawKeys: MLXArray, weights: MLXArray)
        -> (queries: MLXArray, keys: MLXArray, gates: MLXArray, weights: MLXArray)
    {
        (
            rawQueries.reshaped(
                x.dim(0), x.dim(1), configuration.indexNHeads, configuration.indexHeadDim),
            keyNorm(rawKeys), matmul(x, poolGate.T), weights
        )
    }
}

public final class GLM5NextDSAAttention: Module {
    /// Retain this value until the output evaluation fence. Its private views
    /// keep the actual admitted read buffers charged, even if state is closed.
    public struct StagedOutput {
        public let values: MLXArray
        public var evaluationTargets: [MLXArray] { [values] }
        private let readLeases: [AnyObject]
        fileprivate init(values: MLXArray, readLeases: [AnyObject]) {
            self.values = values
            self.readLeases = readLeases
        }
        public func withEvaluationLease(_ fence: ([MLXArray]) throws -> Void) rethrows {
            try withExtendedLifetime(readLeases) { try fence(evaluationTargets) }
        }
    }
    @ModuleInfo(key: "q_a_proj") var qA: Linear
    @ModuleInfo(key: "q_a_layernorm") var qNorm: GLM5NextRMSNorm
    @ModuleInfo(key: "q_b_proj") var qB: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvA: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvNorm: GLM5NextRMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvB: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    let indexer: GLM5NextIndexer
    let configuration: GLM5NextTextConfiguration
    public let modelLayerIndex: Int
    /// Fixed at load for every path of this layer: its target's policy.
    public let executionPolicy: GLM5NextExecutionPolicy
    /// Present exactly for absorbed sparse attention: views of kv_b, not
    /// Module parameters, so the checkpoint's parameter tree is unchanged.
    let absorbed: GLM5NextAbsorbedProjections?
    /// FP32 latent bound of one batched absorbed block; a test seam only.
    let absorbedBlockBytes: Int
    /// q_a, kv_a, indexer wk and weights_proj as one matmul, for
    /// `Projections.fusedInputs`.
    let fusedInputs: GLM5NextFusedProjection?
    /// Prefill steps issue the sparse rows' absorbed query product (the
    /// classic SIMD-group quantized matmul) right after the first block of
    /// index scores (a tensor-unit kernel), from a contiguous copy of the
    /// transposed queries made earlier, so no barrier separates the two and
    /// they run side by side; the selection after the scores waits for both.
    /// Ordering edges only (`depends`): the same kernels on the same values,
    /// so every bit is kept. False keeps the graph's own order.
    var overlapsAbsorbedQueries = true
    public convenience init(
        configuration: GLM5NextTextConfiguration, sourcePrefix: String,
        inventory: GLM5NextWeightInventory, tensor: @escaping (String) throws -> MLXArray,
        releaseSourceTensors: (([String]) -> Void)? = nil,
        executionPolicy: GLM5NextExecutionPolicy = .reference,
        materializeFusedProjections: (([MLXArray]) throws -> Void)? = nil
    ) throws {
        guard configuration == inventory.configuration.textConfig else {
            throw GLM5NextLoadError.invalid("DSA configuration/inventory mismatch")
        }
        guard !inventory.storage.isPacked || releaseSourceTensors != nil else {
            throw GLM5NextLoadError.invalid("Q4 DSA loading requires source-cache eviction")
        }
        try self.init(
            configuration, prefix: sourcePrefix,
            loader: GLM5NextTensorLoader(
                inventory, tensor, releaseSourceTensors: releaseSourceTensors,
                materializeFusedProjections: materializeFusedProjections),
            executionPolicy: executionPolicy)
    }
    init(
        _ c: GLM5NextTextConfiguration, prefix: String, loader: GLM5NextTensorLoader,
        executionPolicy: GLM5NextExecutionPolicy,
        absorbedBlockBytes: Int = GLM5NextDSAAttention.absorbedLatentBlockBytes
    ) throws {
        let components = prefix.split(separator: ".")
        guard components.count == 5, components[0] == "model", components[1] == "language_model",
            components[2] == "layers", components[4] == "self_attn",
            let index = Int(components[3]), index >= 0
        else {
            throw GLM5NextLoadError.invalid("DSA requires an exact source layer prefix")
        }
        modelLayerIndex = index
        configuration = c
        // The MTP layer builds its attention here too, so the norms follow the
        // policy without relying on the target's module walk.
        let qNorm = try loader.norm(prefix + ".q_a_layernorm", epsilon: c.rmsNormEps)
        let kvNorm = try loader.norm(prefix + ".kv_a_layernorm", epsilon: c.rmsNormEps)
        qNorm.usesFusedKernel = executionPolicy.normalization == .fusedKernel
        kvNorm.usesFusedKernel = qNorm.usesFusedKernel
        var inputs = try [
            "q_a_proj", "kv_a_proj_with_mqa", "indexer.wk", "indexer.weights_proj",
        ].map { try loader.linear(prefix + "." + $0) }
        var fused: GLM5NextFusedProjection?
        if executionPolicy.projections == .fusedInputs,
            let result = try GLM5NextFusedProjection.make(
                inputs, materialize: loader.materializeFusedProjections)
        {
            (inputs, fused) = (result.members, result.fused)
        }
        fusedInputs = fused
        _qA.wrappedValue = inputs[0]
        _qNorm.wrappedValue = qNorm
        _qB.wrappedValue = try loader.linear(prefix + ".q_b_proj")
        _kvA.wrappedValue = inputs[1]
        _kvNorm.wrappedValue = kvNorm
        let kvB = try loader.linear(prefix + ".kv_b_proj")
        _kvB.wrappedValue = kvB
        _output.wrappedValue = try loader.linear(prefix + ".o_proj")
        indexer = try GLM5NextIndexer(
            c, prefix: prefix + ".indexer", loader: loader, key: inputs[2],
            headWeights: inputs[3])
        indexer.keyNorm.usesFusedKernel = executionPolicy.normalization == .fusedKernel
        self.executionPolicy = executionPolicy
        absorbed =
            executionPolicy.sparseAttention != .reference
            ? try GLM5NextAbsorbedProjections(kvB: kvB, configuration: c) : nil
        guard executionPolicy.sparseDecode == .reference || absorbed != nil else {
            throw GLM5NextLoadError.invalid("native sparse decode requires absorbed DSA")
        }
        guard executionPolicy.sparsePrefill == .reference || absorbed != nil else {
            throw GLM5NextLoadError.invalid("native sparse prefill requires absorbed DSA")
        }
        self.absorbedBlockBytes = absorbedBlockBytes
    }
    /// Graph-only compatibility cache, not an evaluated publication boundary.
    /// Admitted request owners use stageForward and retain its evaluation lease.
    public func forward(_ hidden: MLXArray, cache: GLM5NextDSACache, validity: MLXArray) throws
        -> MLXArray
    {
        try forwardRetainingSelection(hidden, cache: cache, validity: validity).values
    }

    /// What a step's final query attended, reused by index-shared MTP proposal
    /// steps (`index_share_for_mtp_iteration`): the dense causal prefix of
    /// `keyCount` keys, or its sparse selection row.
    enum SharedSelection {
        case dense(keyCount: Int)
        case sparse(GLM5NextSparseAttention.Selection)
    }

    /// A recursive proposal step attends with the step-0 final query's
    /// selection over step-0 history instead of running the indexer. It neither
    /// reads nor writes later proposal rows.
    func attendSharedSelection(
        _ hidden: MLXArray, latents: MLXArray, keyValidity: MLXArray, selection: SharedSelection
    ) throws -> MLXArray {
        let c = configuration
        guard hidden.ndim == 3, hidden.dim(0) == 1, hidden.dim(1) == 1,
            hidden.dim(2) == c.hiddenSize, latents.ndim == 3, latents.dim(0) == 1,
            keyValidity.shape == [1, latents.dim(1)]
        else { throw GLM5NextLoadError.invalid("shared-selection DSA geometry") }
        let (q, _, _) = try projectInputs(hidden)
        switch selection {
        case .dense(let keyCount):
            guard let absorbed, keyCount > 0, keyCount <= latents.dim(1) else {
                throw GLM5NextLoadError.invalid("shared dense DSA selection")
            }
            // The step-0 final query sat at position keyCount - 1 and saw
            // every earlier key.
            return try denseHistoryAttention(
                q: q, history: latents[0..., ..<keyCount],
                keyValidity: keyValidity[0..., ..<keyCount],
                queryValidity: MLXArray.ones([1, 1], dtype: .bool),
                queryStart: keyCount - 1, projections: absorbed)
        case .sparse(let selection):
            guard selection.historyLength <= latents.dim(1), selection.indices.dim(1) == 1 else {
                throw GLM5NextLoadError.invalid("shared sparse DSA selection")
            }
            let history = latents[0..., ..<selection.historyLength]
            if let absorbed, usesNativeSparseAttention(queries: 1) {
                return try nativeSparseAttention(
                    q: q, selection: selection, history: history, projections: absorbed)
            }
            return try attend(
                q: q, selection: selection,
                gather: { ids in
                    takeAlong(
                        history,
                        clip(ids, min: 0, max: history.dim(1) - 1).expandedDimensions(axis: -1),
                        axis: 1)
                },
                absorbedSource: .batched { Self.gatherBlock(history, ids: $0) })
        }
    }

    func forwardRetainingSelection(
        _ hidden: MLXArray, cache: GLM5NextDSACache, validity: MLXArray
    ) throws -> (values: MLXArray, selection: SharedSelection) {
        let c = configuration
        guard hidden.ndim == 3, hidden.dim(2) == c.hiddenSize,
            validity.shape == Array(hidden.shape.prefix(2)), validity.dtype == .bool
        else { throw GLM5NextLoadError.invalid("DSA input geometry") }
        let (q, newLatents, projected) = try projectInputs(hidden)
        let working = cache.copy() as! GLM5NextDSACache
        let start = working.offset
        try working.append(
            latents: newLatents, keys: projected.keys, gates: projected.gates,
            validity: validity)
        guard let latents = working.latents, let keys = working.indexKeys,
            let gates = working.poolGates, let keyValidity = working.validity
        else {
            throw GLM5NextLoadError.invalid("DSA history is incomplete")
        }
        let count = hidden.dim(1)
        let dense = denseQueryCount(start: start, count: count)
        var parts: [MLXArray] = []
        var finalSelection = SharedSelection.dense(keyCount: start + count)
        if let absorbed, dense > 0 {
            let keyCount = min(working.offset, c.indexTopK)
            parts.append(
                try denseHistoryAttention(
                    q: q[0..., ..<dense], history: latents[0..., ..<keyCount],
                    keyValidity: keyValidity[0..., ..<keyCount],
                    queryValidity: validity[0..., ..<dense], queryStart: start,
                    projections: absorbed))
        }
        if dense < count {
            let rest = dense ..< count
            // Absorbed queries beside the index scores (`overlapsAbsorbedQueries`).
            var indexQueries = projected.queries[0..., rest]
            var queryLatents: MLXArray?
            var afterScores: ((MLXArray) -> MLXArray)?
            if let absorbed, overlapsAbsorbedQueries,
                usesNativeSparseAttention(queries: rest.count)
            {
                // The copy precedes the scores, so the product reads nothing
                // written after the scores' barrier.
                let transposed = q[0..., rest].transposed(0, 2, 1, 3)
                    .asType(absorbed.inputDType).contiguous()
                indexQueries = depends(input: indexQueries, dependencies: [transposed])
                afterScores = { scores in
                    let latents = absorbed.absorbTransposedQueries(
                        depends(input: transposed, dependencies: [scores]))
                    queryLatents = latents
                    return depends(input: scores, dependencies: [latents])
                }
            }
            let selection = try selectSparse(
                indexQueries: indexQueries, indexKeys: keys, poolGates: gates,
                headWeights: projected.weights[0..., rest], keyValidity: keyValidity,
                queryValidity: validity[0..., rest], queryStart: start + dense,
                afterScores: afterScores)
            finalSelection = .sparse(selection.finalQuery)
            if let absorbed, usesNativeSparseAttention(queries: rest.count) {
                parts.append(
                    try nativeSparseAttention(
                        q: q[0..., rest], selection: selection, history: latents,
                        projections: absorbed, queryLatents: queryLatents))
            } else {
                parts.append(
                    try attend(
                        q: q[0..., rest], selection: selection,
                        gather: { ids in
                            takeAlong(
                                latents,
                                clip(ids, min: 0, max: working.offset - 1).expandedDimensions(
                                    axis: -1),
                                axis: 1)
                        },
                        absorbedSource: .batched { Self.gatherBlock(latents, ids: $0) }))
            }
        }
        cache.adopt(working)
        return (parts.count == 1 ? parts[0] : concatenated(parts, axis: 1), finalSelection)
    }

    /// Under `SparseDecode.nativeKernels`, whether a step's `queries` sparse
    /// queries use the native selection and, for contiguous history, attention.
    private func usesNativeSparseDecode(queries: Int) -> Bool {
        executionPolicy.sparseDecode == .nativeKernels && absorbed != nil
            && queries <= GLM5NextSparseDecodeKernel.maximumQueries
    }

    /// Whether a step's `queries` sparse queries over a contiguous history
    /// attend in place through the indices (`nativeSparseAttention`): native
    /// decode steps, or longer steps under `SparsePrefill.nativeKernel`.
    private func usesNativeSparseAttention(queries: Int) -> Bool {
        usesNativeSparseDecode(queries: queries)
            || (executionPolicy.sparsePrefill == .nativeKernel && absorbed != nil
                && queries > GLM5NextSparseDecodeKernel.maximumQueries)
    }

    /// Selection of a step's sparse queries. `poolingType` is the dtype the
    /// stored keys and gates are pooled in when it differs from theirs.
    private func selectSparse(
        indexQueries: MLXArray, indexKeys: MLXArray, poolGates: MLXArray, headWeights: MLXArray,
        keyValidity: MLXArray, queryValidity: MLXArray, queryStart: Int,
        poolingType: (keys: DType, gates: DType)? = nil,
        afterScores: ((MLXArray) -> MLXArray)? = nil
    ) throws -> GLM5NextSparseAttention.Selection {
        if usesNativeSparseDecode(queries: indexQueries.dim(1)) {
            // Gates enter pooling as FP32 logits; reading stored FP32 values
            // that widened `poolingType.gates` exactly equals casting first.
            return try GLM5NextSparseAttention.selectDecodeTokens(
                indexQueries: indexQueries, indexKeys: indexKeys, poolGates: poolGates,
                poolPositionBias: indexer.positionBias, headWeights: headWeights,
                keyValidity: keyValidity, queryValidity: queryValidity, queryStart: queryStart,
                tokenBudget: configuration.indexTopK,
                strategy: executionPolicy.sparseSelection.selection,
                poolingType: poolingType?.keys)
        }
        return try GLM5NextSparseAttention.selectTokens(
            indexQueries: indexQueries,
            indexKeys: poolingType.map { indexKeys.asType($0.keys) } ?? indexKeys,
            poolGates: poolingType.map { poolGates.asType($0.gates) } ?? poolGates,
            poolPositionBias: indexer.positionBias, headWeights: headWeights,
            keyValidity: keyValidity, queryValidity: queryValidity, queryStart: queryStart,
            tokenBudget: configuration.indexTopK, execution: executionPolicy.sparseSelection,
            nativePrefillScores: executionPolicy.sparsePrefill == .nativeKernel
                && indexQueries.dim(1) > GLM5NextSparseDecodeKernel.maximumQueries,
            afterScores: afterScores)
    }

    /// Under `.absorbedDenseHistory`, the leading queries of a step at raw
    /// positions `start...` whose selection budget covers every pool they can
    /// see (position < index_topk), so their selected set is the dense causal
    /// one and only the first index_topk keys matter.
    private func denseQueryCount(start: Int, count: Int) -> Int {
        guard executionPolicy.sparseAttention == .absorbedDenseHistory, absorbed != nil else {
            return 0
        }
        return max(0, min(count, configuration.indexTopK - start))
    }

    /// Injection hook for the real one-row, AdmissionV2-backed native owner.
    /// The caller begins/commits/rolls back the transaction and joins its KDA
    /// state at the same acceptance fence. This method NEVER commits that owner,
    /// creates a budget, or materializes the complete latent history.
    public func stageForward(
        _ hidden: MLXArray, state: GLM5NextPagedSparseState,
        transaction: GLM5NextPagedSparseState.Transaction,
        binding: GLM5NextSparseStateBinding, validity: MLXArray
    ) throws -> StagedOutput {
        let c = configuration
        guard hidden.ndim == 3, hidden.dim(0) == 1, hidden.dim(1) > 0,
            hidden.dim(2) == c.hiddenSize, validity.shape == [1, hidden.dim(1)],
            validity.dtype == .bool, state.binding == binding,
            binding.modelLayerIndex == modelLayerIndex
        else {
            throw GLM5NextLoadError.invalid("paged DSA requires the bound single request row")
        }
        let (q, newLatents, projected) = try projectInputs(hidden)
        let start = state.committedTokenCount + state.stagedTokenCount
        try state.stage(
            transaction, at: start, latents: newLatents,
            indexKeys: projected.keys, poolGates: projected.gates, validity: validity)
        let index = try state.indexView(transaction)
        var leases: [AnyObject] = [index]
        // Each query's read reserves exactly what it gathers; both paths read
        // once per query and retain every read's lease until the output fence.
        func gather(_ ids: MLXArray) throws -> MLXArray {
            let selected = try state.gatherLatents(ids.squeezed(axis: 0), transaction: transaction)
            leases.append(selected)
            return selected.latents
        }
        let count = hidden.dim(1)
        let dense = denseQueryCount(start: start, count: count)
        var parts: [MLXArray] = []
        if let absorbed, dense > 0 {
            // One read of the first keyCount latents, no larger than one
            // query's selected read.
            let keyCount = min(index.validity.dim(1), c.indexTopK)
            parts.append(
                try denseHistoryAttention(
                    q: q[0..., ..<dense],
                    history: try gather(
                        MLXArray.arange(keyCount, dtype: .int32).reshaped(1, keyCount)),
                    keyValidity: index.validity[0..., ..<keyCount],
                    queryValidity: validity[0..., ..<dense], queryStart: start,
                    projections: absorbed))
        }
        if dense < count {
            let rest = dense ..< count
            let selection = try selectSparse(
                indexQueries: projected.queries[0..., rest], indexKeys: index.indexKeys,
                poolGates: index.poolGates, headWeights: projected.weights[0..., rest],
                keyValidity: index.validity, queryValidity: validity[0..., rest],
                queryStart: start + dense)
            parts.append(
                try attend(
                    q: q[0..., rest], selection: selection, gather: gather,
                    absorbedSource: .perQuery(gather)))
        }
        let result = parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
        return StagedOutput(values: result, readLeases: leases)
    }

    /// Native contiguous CBv2 storage is the DSA payload itself: latent K,
    /// and V = [raw index keys, raw pool gates, exact 0/1 validity].
    func forwardCBv2(
        _ hidden: MLXArray, cache: any CBv2AttendingLayerCache, validity: MLXArray,
        targetIdentity: ObjectIdentifier
    ) throws -> MLXArray {
        let c = configuration
        let starts = cache.rows.map(\.absoluteOffset)
        let (queries, latents, projected) = try projectInputs(hidden)
        let values = concatenated(
            [
                projected.keys, projected.gates,
                validity.asType(projected.keys.dtype).expandedDimensions(axis: -1),
            ], axis: -1)
        let nativeKeys = latents.asType(.float32).expandedDimensions(axis: 1)
        let nativeValues = values.asType(.float32).expandedDimensions(axis: 1)
        let paged: GLM5NextPagedPayload?
        let history: [(keys: MLXArray, values: MLXArray)]
        if let cache = cache as? PagedLayerCache {
            paged = try cache.beginGLM5NextPayload(
                keys: nativeKeys, values: nativeValues, targetIdentity: targetIdentity)
            history = []
        } else if let cache = cache as? CBv2LayerCache {
            paged = nil
            history = cache.updateKVAndAdvanceOffsets(keys: nativeKeys, values: nativeValues)
        } else {
            throw GLM5NextLoadError.invalid("unbound GLM payload cache")
        }
        var outputs: [MLXArray] = []
        for row in 0 ..< hidden.dim(0) {
            // Canonical FP32 storage losslessly extends native activation values.
            // Restore each projection's dtype before its trained pooling/matmul.
            let metadata = paged?.values(row: row) ?? history[row].values.squeezed(axis: 1)
            let count = hidden.dim(1)
            guard metadata.dim(1) == starts[row] + count else {
                throw GLM5NextLoadError.invalid("native GLM row history does not end at the step")
            }
            let dense = denseQueryCount(start: starts[row], count: count)
            let rowKeyValidity = metadata[0..., 0..., 2 * c.indexHeadDim] .> 0
            // A paged prefill-sized row reads its latent history once and then
            // runs exactly the contiguous arithmetic over it; smaller paged
            // steps keep one priced read per query.
            let latentHistory: MLXArray?
            if let paged {
                let sparse = count - dense
                if paged.readsLatentHistory(row: row, sparseQueries: sparse) {
                    latentHistory = try paged.latentHistory(
                        row: row, sparseQueries: sparse, dtype: latents.dtype)
                } else {
                    latentHistory = nil
                }
            } else {
                latentHistory = history[row].keys.squeezed(axis: 1)
            }
            var parts: [MLXArray] = []
            if let absorbed, dense > 0 {
                let keyCount = min(metadata.dim(1), c.indexTopK)
                let prefix: MLXArray
                if let latentHistory {
                    prefix = latentHistory[0..., ..<keyCount]
                } else {
                    // A paged row reads its first keyCount latents as one priced
                    // selection-width read (-1 pads the unused columns).
                    let width = c.indexTopK + 3
                    prefix = try paged!.selectedKeys(
                        row: row,
                        indices: concatenated(
                            [
                                MLXArray.arange(keyCount, dtype: .int32),
                                MLXArray.zeros([width - keyCount], dtype: .int32) - 1,
                            ]
                        ).reshaped(1, width)
                    )[0..., ..<keyCount]
                }
                parts.append(
                    try denseHistoryAttention(
                        q: queries[row ..< row + 1, ..<dense],
                        history: prefix,
                        keyValidity: rowKeyValidity[0..., ..<keyCount],
                        queryValidity: validity[row ..< row + 1, ..<dense],
                        queryStart: starts[row], projections: absorbed))
            }
            if dense < count {
                let rest = dense ..< count
                let selection = try selectSparse(
                    indexQueries: projected.queries[row ..< row + 1, rest],
                    indexKeys: metadata[0..., 0..., ..<c.indexHeadDim],
                    poolGates: metadata[0..., 0..., c.indexHeadDim ..< (2 * c.indexHeadDim)],
                    headWeights: projected.weights[row ..< row + 1, rest],
                    keyValidity: rowKeyValidity,
                    queryValidity: validity[row ..< row + 1, rest], queryStart: starts[row] + dense,
                    poolingType: (projected.keys.dtype, projected.gates.dtype))
                if let absorbed, let latentHistory, usesNativeSparseAttention(queries: rest.count) {
                    // FP32 storage widened the activation dtype losslessly.
                    parts.append(
                        try nativeSparseAttention(
                            q: queries[row ..< row + 1, rest], selection: selection,
                            history: latentHistory, projections: absorbed))
                } else if let latentHistory {
                    parts.append(
                        try attend(
                            q: queries[row ..< row + 1, rest], selection: selection,
                            gather: { ids in
                                let keys = latentHistory.asType(latents.dtype)
                                return takeAlong(
                                    keys,
                                    clip(ids, min: 0, max: keys.dim(1) - 1).expandedDimensions(
                                        axis: -1),
                                    axis: 1)
                            },
                            // Absorbed attention widens each gathered block to FP32;
                            // both history dtypes hold the same activation values.
                            absorbedSource: .batched { ids in
                                Self.gatherBlock(latentHistory, ids: ids)
                            }))
                } else if let paged {
                    // Each paged read is one priced query.
                    parts.append(
                        try attend(
                            q: queries[row ..< row + 1, rest], selection: selection,
                            gather: { ids in
                                try paged.selectedKeys(row: row, indices: ids).asType(
                                    latents.dtype)
                            },
                            absorbedSource: .perQuery { ids in
                                try paged.selectedKeys(
                                    row: row, indices: ids,
                                    staticPadding: selection.staticPaddingColumns
                                ).asType(.float32)
                            }))
                }
            }
            outputs.append(parts.count == 1 ? parts[0] : concatenated(parts, axis: 1))
        }
        let result = concatenated(outputs, axis: 0)
        paged?.retainResult(result)
        return result
    }

    /// Dispatches to the load-time arithmetic. `gather` reads one query's full
    /// selection row, `[B,W] -> [B,W,C]`, for the expanded reference;
    /// `absorbedSource` is built only for the absorbed path and reads only its
    /// non-padding columns.
    /// Queries `[B,T,H,Dq]`, normalized latents and the indexer projections.
    /// The input, q_b and indexer wq_b projections run through the load-time
    /// `prefillProjections` member (`.mlx` builds exactly the previous graph).
    private func projectInputs(_ hidden: MLXArray) throws -> (
        queries: MLXArray, latents: MLXArray,
        index: (queries: MLXArray, keys: MLXArray, gates: MLXArray, weights: MLXArray)
    ) {
        let c = configuration
        let prefill = executionPolicy.prefillProjections
        let parts =
            try fusedInputs?.project(hidden, prefill: prefill).members
            ?? [qA, kvA, indexer.key, indexer.headWeights].map { try prefill.project($0, hidden) }
        let residual = qNorm(parts[0])
        let queries = try prefill.project(qB, residual).reshaped(
            hidden.dim(0), hidden.dim(1), c.attentionHeads, c.qkNopeHeadDim)
        let index = try indexer.project(
            hidden, residual: residual, rawKeys: parts[2], weights: parts[3], prefill: prefill)
        return (queries, kvNorm(parts[1]), index)
    }

    private func attend(
        q: MLXArray, selection: GLM5NextSparseAttention.Selection,
        gather: (MLXArray) throws -> MLXArray,
        absorbedSource: @autoclosure () -> AbsorbedLatentSource
    ) throws -> MLXArray {
        guard let absorbed else {
            return try expandedAttention(q: q, selection: selection, gather: gather)
        }
        return try absorbedAttention(
            q: q, selection: selection, projections: absorbed, source: absorbedSource())
    }

    /// One batched read of a contiguous `[B,N,C]` history for `[B,Tq,L]` ids.
    private static func gatherBlock(_ latents: MLXArray, ids: MLXArray) -> MLXArray {
        takeAlong(
            latents, clip(ids, min: 0, max: latents.dim(1) - 1).reshaped(ids.dim(0), -1, 1),
            axis: 1
        ).reshaped(ids.shape + [latents.dim(2)])
    }

    private func expandedAttention(
        q: MLXArray, selection: GLM5NextSparseAttention.Selection,
        gather: (MLXArray) throws -> MLXArray
    ) throws -> MLXArray {
        let c = configuration
        var rows: [MLXArray] = []
        for t in 0 ..< q.dim(1) {
            let ids = selection.indices[0..., t]
            let valid = ids .>= 0
            let selected = try gather(ids)
            // Apply the checkpoint's original projection ONLY to selected latents.
            // This preserves Q4 kv_b without full dequantization/requantization.
            let expanded = kvB(selected).reshaped(
                q.dim(0), ids.dim(1), c.attentionHeads, c.qkNopeHeadDim + c.vHeadDim
            )
            .transposed(0, 2, 1, 3)
            let k = expanded[.ellipsis, ..<c.qkNopeHeadDim]
            let v = expanded[.ellipsis, c.qkNopeHeadDim...]
            let query = q[0..., t, 0..., 0...].expandedDimensions(axis: 2)
            let score =
                (query.asType(.float32) * k.asType(.float32)).sum(axis: -1)
                * (1 / sqrt(Float(c.qkNopeHeadDim)))
            let mask = valid.expandedDimensions(axis: 1)
            let masked = MLX.where(mask, score, -Float.infinity)
            let maximum = MLX.where(
                mask.any(axis: -1, keepDims: true), masked.max(axis: -1, keepDims: true), 0)
            let exps = MLX.where(mask, exp(masked - maximum), 0)
            let total = exps.sum(axis: -1, keepDims: true)
            let probability = (exps / MLX.where(total .> 0, total, 1)).asType(q.dtype)
            let value = matmul(probability.expandedDimensions(axis: 2), v).squeezed(axis: 2)
            rows.append(output(value.reshaped(q.dim(0), 1, c.attentionHeads * c.vHeadDim)))
        }
        return concatenated(rows, axis: 1)
    }
}

final class GLM5NextKDASublayer: Module {
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "f_a_proj") var fA: Linear
    @ModuleInfo(key: "f_b_proj") var fB: Linear
    @ModuleInfo(key: "g_a_proj") var gA: Linear
    @ModuleInfo(key: "g_b_proj") var gB: Linear
    @ModuleInfo(key: "b_proj") var beta: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "q_conv1d") var qConv: GLM5NextWeight
    @ModuleInfo(key: "k_conv1d") var kConv: GLM5NextWeight
    @ModuleInfo(key: "v_conv1d") var vConv: GLM5NextWeight
    @ModuleInfo(key: "o_norm") var outputNorm: GLM5NextWeight
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    let configuration: GLM5NextTextConfiguration
    /// Fixed at load for every path of this layer: its target's policy.
    let linearAttention: GLM5NextExecutionPolicy.LinearAttention
    let decodeStep: GLM5NextExecutionPolicy.LinearDecodeStep
    /// q, k, v, f_a, g_a and b as one matmul, for `Projections.fusedInputs`.
    let fusedInputs: GLM5NextFusedProjection?
    let prefillProjections: GLM5NextExecutionPolicy.PrefillProjections
    init(
        _ c: GLM5NextTextConfiguration, prefix: String, loader: GLM5NextTensorLoader,
        linearAttention: GLM5NextExecutionPolicy.LinearAttention,
        decodeStep: GLM5NextExecutionPolicy.LinearDecodeStep = .composed,
        projections: GLM5NextExecutionPolicy.Projections = .separate,
        prefillProjections: GLM5NextExecutionPolicy.PrefillProjections = .mlx
    ) throws {
        // The configuration already guarantees a negative gate lower bound.
        guard decodeStep == .composed || linearAttention != .reference else {
            throw GLM5NextLoadError.invalid("fused KDA decode requires native linear attention")
        }
        if linearAttention == .nativeRecurrenceChunkedPrefill {
            guard
                GLM5NextLinearChunkedRecurrence.supports(
                    chunkSize: GLM5NextExecutionPolicy.linearAttentionChunkSize,
                    gateLowerBound: Float(c.linearAttention.gateLowerBound))
            else {
                throw GLM5NextLoadError.invalid("KDA gate lower bound does not admit chunking")
            }
        }
        configuration = c
        self.linearAttention = linearAttention
        self.decodeStep = decodeStep
        self.prefillProjections = prefillProjections
        var inputs = try ["q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj"].map {
            try loader.linear(prefix + "." + $0)
        }
        var fused: GLM5NextFusedProjection?
        if projections == .fusedInputs,
            let result = try GLM5NextFusedProjection.make(
                inputs, materialize: loader.materializeFusedProjections)
        {
            (inputs, fused) = (result.members, result.fused)
        }
        fusedInputs = fused
        _q.wrappedValue = inputs[0]
        _k.wrappedValue = inputs[1]
        _v.wrappedValue = inputs[2]
        _fA.wrappedValue = inputs[3]
        _fB.wrappedValue = try loader.linear(prefix + ".f_b_proj")
        _gA.wrappedValue = inputs[4]
        _gB.wrappedValue = try loader.linear(prefix + ".g_b_proj")
        _beta.wrappedValue = inputs[5]
        _output.wrappedValue = try loader.linear(prefix + ".o_proj")
        _qConv.wrappedValue = GLM5NextWeight(try loader.array(prefix + ".q_conv1d.weight"))
        _kConv.wrappedValue = GLM5NextWeight(try loader.array(prefix + ".k_conv1d.weight"))
        _vConv.wrappedValue = GLM5NextWeight(try loader.array(prefix + ".v_conv1d.weight"))
        _outputNorm.wrappedValue = GLM5NextWeight(try loader.array(prefix + ".o_norm.weight"))
        _aLog.wrappedValue = try loader.array(prefix + ".A_log")
        _dtBias.wrappedValue = try loader.array(prefix + ".dt_bias")
    }
    /// `captureWindow`: also keep every token's state in the cache, so it can
    /// be truncated to any token of this step (an MTP verification window).
    func forward(
        _ x: MLXArray, cache: GLM5NextKDACache, validity: MLXArray,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy = .reference,
        captureWindow: Bool = false
    ) throws -> MLXArray {
        let result = try forwardCapturing(
            x, state: cache.current, validity: validity, capture: captureWindow,
            recurrenceExecution: recurrenceExecution)
        cache.advance(result.state, count: x.dim(1), window: result.captured)
        return result.values
    }

    func forward(
        _ x: MLXArray, state initialState: GLM5NextLinearAttention.State?, validity: MLXArray,
        canonicalConvolution: Bool = false,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy = .reference,
        wholeStepQualification: GLM5NextLinearAttention.ExecutionPolicy? = nil
    ) throws
        -> (values: MLXArray, state: GLM5NextLinearAttention.State)
    {
        let result = try forwardCapturing(
            x, state: initialState, validity: validity,
            canonicalConvolution: canonicalConvolution, capture: false,
            recurrenceExecution: recurrenceExecution,
            wholeStepQualification: wholeStepQualification)
        return (result.values, result.state)
    }

    /// Per-position states of an MTP verification window: `[B,T,H,D,D]` FP32
    /// recurrent states and `[B,T,3HD,K]` convolution windows, entry t being
    /// the state after token t (unchanged across invalid tokens). Each comes
    /// from the same recurrence that produced the window's final state.
    struct CapturedStates {
        let convolution: MLXArray
        let recurrent: MLXArray
    }

    /// `forward`, optionally capturing every window position's state. The
    /// chunked prefill recurrence never chunks a window of at most
    /// `linearAttentionChunkSize - 1` tokens; it refuses capture otherwise.
    func forwardCapturing(
        _ x: MLXArray, state initialState: GLM5NextLinearAttention.State?, validity: MLXArray,
        canonicalConvolution: Bool = false, capture: Bool,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy = .reference,
        wholeStepQualification: GLM5NextLinearAttention.ExecutionPolicy? = nil
    ) throws
        -> (
            values: MLXArray, state: GLM5NextLinearAttention.State, captured: CapturedStates?
        )
    {
        let wholeStep: GLM5NextLinearAttention.ExecutionPolicy?
        if let wholeStepQualification {
            // Like `recurrenceExecution`, a seam of the reference policy only.
            guard linearAttention == .reference, recurrenceExecution == .reference else {
                throw GLM5NextLoadError.invalid("KDA execution is fixed by the load-time policy")
            }
            wholeStep = wholeStepQualification
        } else {
            wholeStep = try wholeStepExecution(length: x.dim(1), override: recurrenceExecution)
        }
        let c = configuration
        let linear = c.linearAttention
        let shape = [x.dim(0), x.dim(1), linear.numHeads, linear.headDim]
        // The fused step ignores an invalid token's projections (unchanged
        // window and state, zero output), so it skips masking the input.
        let fusedStep: Bool
        switch decodeStep {
        case .composed: fusedStep = false
        case .fusedKernel: fusedStep = x.dim(1) <= GLM5NextLinearDecodeKernel.maximumTokens
        case .fusedEveryStep:
            fusedStep = !capture || x.dim(1) <= GLM5NextLinearDecodeKernel.maximumTokens
        }
        let fused =
            fusedStep && wholeStep != nil
            && GLM5NextLinearDecodeKernel.supports(
                tokens: x.dim(1), dimension: configuration.linearAttention.headDim)
        let masked = fused ? x : MLX.where(validity.expandedDimensions(axis: -1), x, 0)
        let fusedRows = try fusedInputs?.project(masked, prefill: prefillProjections)
        let projected =
            try fusedRows?.members
            ?? [q, k, v, fA, gA, beta].map { try prefillProjections.project($0, masked) }
        let query = projected[0].reshaped(shape)
        let key = projected[1].reshaped(shape)
        let value = projected[2].reshaped(shape)
        let b = projected[5]
        // f_b/g_b and the initial state are built only where they are read:
        // the fused step applies the low-rank gate rows in-kernel and takes the
        // retained state as it is.
        func projectedGates() -> (forget: MLXArray, output: MLXArray) {
            (fB(projected[3]).reshaped(shape), gB(projected[4]).reshaped(shape))
        }
        if fused {
            let gates =
                GLM5NextLinearDecodeKernel.Gates.inKernel(
                    forgetInput: projected[3], outputInput: projected[4], forget: fB, output: gB)
                ?? {
                    let (forget, output) = projectedGates()
                    return .projected(forget: forget, output: output)
                }()
            let result = try GLM5NextLinearDecodeKernel.apply(
                .init(
                    query: query, key: key, value: value, gates: gates, betaLogits: b,
                    queryConvolution: qConv.weight,
                    keyConvolution: kConv.weight, valueConvolution: vConv.weight, aLog: aLog,
                    dtBias: dtBias, outputNorm: outputNorm.weight,
                    // FP32 canonical history is passed through unconverted;
                    // a missing state is zero in-kernel.
                    state: initialState, validity: validity,
                    projectionRows: projectionRows(fusedRows?.output)),
                normEpsilon: Float(c.rmsNormEps), gateLowerBound: Float(linear.gateLowerBound),
                capture: capture)
            let values = result.output.values
            return (
                try prefillProjections.project(
                    output, values.reshaped(x.dim(0), x.dim(1), linear.numHeads * linear.headDim)),
                result.output.state,
                capture
                    ? CapturedStates(
                        convolution: result.capturedConvolution!,
                        recurrent: result.capturedRecurrent!) : nil
            )
        }
        let (forget, gate) = projectedGates()
        var state: GLM5NextLinearAttention.State =
            initialState.map {
                // Native CBv2 stores raw convolution history in FP32; these
                // values came from query.dtype, so the round trip is exact.
                canonicalConvolution
                    ? GLM5NextLinearAttention.State(
                        convolution: $0.convolution.asType(query.dtype), recurrent: $0.recurrent)
                    : $0
            }
            ?? .init(
                convolution: MLXArray.zeros(
                    [x.dim(0), 3 * linear.numHeads * linear.headDim, linear.shortConvKernelSize],
                    dtype: query.dtype),
                recurrent: MLXArray.zeros(
                    [x.dim(0), linear.numHeads, linear.headDim, linear.headDim], dtype: .float32))
        let conv = concatenated([qConv.weight, kConv.weight, vConv.weight], axis: 0).squeezed(
            axis: 1)
        let weights = GLM5NextLinearAttention.Weights(
            convolution: conv, aLog: aLog, dtBias: dtBias, outputNorm: outputNorm.weight)
        if let execution = wholeStep {
            let result = try GLM5NextLinearAttention.apply(
                query: query, key: key, value: value, forgetLogits: forget, betaLogits: b,
                outputGate: gate, weights: weights, state: state, validity: validity,
                normEpsilon: Float(c.rmsNormEps), gateLowerBound: Float(linear.gateLowerBound),
                execution: execution, capture: capture)
            return (
                try prefillProjections.project(
                    output,
                    result.values.reshaped(x.dim(0), x.dim(1), linear.numHeads * linear.headDim)),
                result.state,
                capture
                    ? CapturedStates(
                        convolution: stacked(result.captured.map(\.convolution), axis: 1),
                        recurrent: stacked(result.captured.map(\.recurrent), axis: 1)) : nil
            )
        }
        var rows: [MLXArray] = []
        var captured: [GLM5NextLinearAttention.State] = []
        for t in 0 ..< x.dim(1) {
            let slice = t ..< t + 1
            let result = try GLM5NextLinearAttention.apply(
                query: query[0..., slice], key: key[0..., slice], value: value[0..., slice],
                forgetLogits: forget[0..., slice], betaLogits: b[0..., slice],
                outputGate: gate[0..., slice],
                weights: weights, state: state, normEpsilon: Float(c.rmsNormEps),
                gateLowerBound: Float(linear.gateLowerBound), execution: recurrenceExecution)
            let active = validity[0..., t]
            state = .init(
                convolution: MLX.where(
                    active.reshaped(x.dim(0), 1, 1), result.state.convolution, state.convolution),
                recurrent: MLX.where(
                    active.reshaped(x.dim(0), 1, 1, 1), result.state.recurrent, state.recurrent))
            rows.append(MLX.where(active.reshaped(x.dim(0), 1, 1, 1), result.values, 0))
            if capture { captured.append(state) }
        }
        let result = output(
            concatenated(rows, axis: 1).reshaped(
                x.dim(0), x.dim(1), linear.numHeads * linear.headDim))
        return (
            result, state,
            capture
                ? CapturedStates(
                    convolution: stacked(captured.map(\.convolution), axis: 1),
                    recurrent: stacked(captured.map(\.recurrent), axis: 1)) : nil
        )
    }

    /// The fused projection output as in-place q/k/v/beta rows of a long
    /// fused step: q, k and v lead in that order, beta is the sixth member.
    /// Both `PrefillProjections` members write the fused output row-contiguous,
    /// so the long step reads the same values in place instead of copying each
    /// column view to contiguous storage first (same bits).
    private func projectionRows(_ output: MLXArray?) -> GLM5NextLinearDecodeKernel.ProjectionRows? {
        guard let output, let columns = fusedInputs?.columns, columns.count == 6
        else { return nil }
        let width = configuration.linearAttention.numHeads * configuration.linearAttention.headDim
        guard columns[1] == width, columns[2] == 2 * width else { return nil }
        return .init(buffer: output, betaColumn: columns[5])
    }

    /// The load-time policy's whole-step recurrence, or nil for the per-token
    /// reference loop. A per-call `recurrenceExecution` is the reference
    /// policy's qualification seam and never overrides a load-time choice.
    private func wholeStepExecution(
        length: Int, override: GLM5NextLinearAttention.ExecutionPolicy
    ) throws -> GLM5NextLinearAttention.ExecutionPolicy? {
        guard linearAttention != .reference else { return nil }
        guard override == .reference else {
            throw GLM5NextLoadError.invalid("KDA execution is fixed by the load-time policy")
        }
        switch linearAttention {
        case .reference: return nil
        case .nativeRecurrence: return .nativeRequired
        case .nativeRecurrenceChunkedPrefill:
            let size = GLM5NextExecutionPolicy.linearAttentionChunkSize
            guard length >= size else { return .nativeRequired }
            // The chunk operations could run anywhere; the member's contract
            // is the native one for every step length.
            try GLM5NextLinearKernel.requireDefaultGPUStream()
            return .chunked(size: size)
        }
    }
}
