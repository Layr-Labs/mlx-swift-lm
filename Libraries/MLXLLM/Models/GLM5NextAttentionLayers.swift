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
    init(weight: MLXArray, epsilon: Float) {
        self.weight = weight
        self.epsilon = epsilon
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
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
    let groupSize = 64
    let bits = 4
    let mode = QuantizationMode.affine
    var scales: MLXArray
    var biases: MLXArray
    init(weight: MLXArray, scales: MLXArray, biases: MLXArray) {
        self.scales = scales
        self.biases = biases
        super.init(weight: weight)
        freeze()
    }
    public override var shape: (Int, Int) { (weight.dim(0), weight.dim(1) * 8) }
    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let indices = x.flattened()
        return dequantized(
            weight[indices], scales: scales[indices], biases: biases[indices],
            groupSize: 64, bits: 4
        ).reshaped(shape + [-1])
    }
    public override func asLinear(_ x: MLXArray) -> MLXArray {
        quantizedMM(x, weight, scales: scales, biases: biases, groupSize: 64, bits: 4)
    }
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
                biases: try array(prefix + ".biases"))
            : Embedding(weight: weight)
    }
    func mlp(_ c: GLM5NextTextConfiguration, prefix: String, intermediate: Int) throws
        -> GLM5NextMLP
    {
        guard inventory.storage != .vontraQ4 || releaseSourceTensors != nil else {
            throw GLM5NextLoadError.invalid("Q4 MLP loading requires source-cache eviction")
        }
        return try GLM5NextMLP(
            hiddenSize: c.hiddenSize, intermediateSize: intermediate,
            gateProj: linear(prefix + ".gate_proj"), upProj: linear(prefix + ".up_proj"),
            downProj: linear(prefix + ".down_proj"), swigluLimit: Float(c.swigluLimit))
    }
    func moe(
        _ c: GLM5NextTextConfiguration, prefix: String, materialize: ((MLXArray) throws -> Void)?
    ) throws -> GLM5NextMoE {
        guard inventory.storage != .vontraQ4 || (materialize != nil && releaseSourceTensors != nil)
        else {
            throw GLM5NextLoadError.invalid(
                "Q4 expert loading requires materialization and source-cache eviction")
        }
        var projections: [String: SwitchLinear] = [:]
        for name in ["gate_proj", "up_proj", "down_proj"] {
            let output = name == "down_proj" ? c.hiddenSize : c.moeIntermediateSize
            let input = name == "down_proj" ? c.moeIntermediateSize : c.hiddenSize
            func bank(_ suffix: String) throws -> MLXArray {
                let names = (0 ..< c.nRoutedExperts).map {
                    prefix + ".experts.\($0).\(name).\(suffix)"
                }
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
                projections[name] = try QuantizedSwitchLinear(
                    inputDims: input, outputDims: output, numExperts: c.nRoutedExperts,
                    weight: weight, scales: bank("scales"), biases: bank("biases"),
                    groupSize: 64, bits: 4)
            } else {
                projections[name] = SwitchLinear(
                    inputDims: input, outputDims: output, numExperts: c.nRoutedExperts,
                    weight: weight)
            }
        }
        guard let gate = projections["gate_proj"], let up = projections["up_proj"],
            let down = projections["down_proj"]
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
                intermediate: c.moeIntermediateSize * c.nSharedExperts))
    }
}

final class GLM5NextIndexNorm: Module {
    var weight: MLXArray
    var bias: MLXArray
    init(weight: MLXArray, bias: MLXArray) {
        self.weight = weight
        self.bias = bias
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
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
    init(_ c: GLM5NextTextConfiguration, prefix: String, loader: GLM5NextTensorLoader) throws {
        configuration = c
        _query.wrappedValue = try loader.linear(prefix + ".wq_b")
        _key.wrappedValue = try loader.linear(prefix + ".wk")
        _headWeights.wrappedValue = try loader.linear(prefix + ".weights_proj")
        _keyNorm.wrappedValue = GLM5NextIndexNorm(
            weight: try loader.array(prefix + ".k_norm.weight"),
            bias: try loader.array(prefix + ".k_norm.bias"))
        _positionBias.wrappedValue = try loader.array(prefix + ".index_kpool_compress_ape")
        _poolGate.wrappedValue = try loader.array(prefix + ".index_kpool_compress_gate")
    }
    func project(_ x: MLXArray, residual: MLXArray) -> (
        queries: MLXArray, keys: MLXArray, gates: MLXArray, weights: MLXArray
    ) {
        let keys = keyNorm(key(x))
        return (
            query(residual).reshaped(
                x.dim(0), x.dim(1), configuration.indexNHeads, configuration.indexHeadDim),
            keys, matmul(x, poolGate.T), headWeights(x)
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
    public convenience init(
        configuration: GLM5NextTextConfiguration, sourcePrefix: String,
        inventory: GLM5NextWeightInventory, tensor: @escaping (String) throws -> MLXArray,
        releaseSourceTensors: (([String]) -> Void)? = nil
    ) throws {
        guard configuration == inventory.configuration.textConfig else {
            throw GLM5NextLoadError.invalid("DSA configuration/inventory mismatch")
        }
        guard inventory.storage != .vontraQ4 || releaseSourceTensors != nil else {
            throw GLM5NextLoadError.invalid("Q4 DSA loading requires source-cache eviction")
        }
        try self.init(
            configuration, prefix: sourcePrefix,
            loader: GLM5NextTensorLoader(
                inventory, tensor, releaseSourceTensors: releaseSourceTensors))
    }
    init(_ c: GLM5NextTextConfiguration, prefix: String, loader: GLM5NextTensorLoader) throws {
        let components = prefix.split(separator: ".")
        guard components.count == 5, components[0] == "model", components[1] == "language_model",
            components[2] == "layers", components[4] == "self_attn",
            let index = Int(components[3]), index >= 0
        else {
            throw GLM5NextLoadError.invalid("DSA requires an exact source layer prefix")
        }
        modelLayerIndex = index
        configuration = c
        _qA.wrappedValue = try loader.linear(prefix + ".q_a_proj")
        _qNorm.wrappedValue = try loader.norm(prefix + ".q_a_layernorm", epsilon: c.rmsNormEps)
        _qB.wrappedValue = try loader.linear(prefix + ".q_b_proj")
        _kvA.wrappedValue = try loader.linear(prefix + ".kv_a_proj_with_mqa")
        _kvNorm.wrappedValue = try loader.norm(prefix + ".kv_a_layernorm", epsilon: c.rmsNormEps)
        _kvB.wrappedValue = try loader.linear(prefix + ".kv_b_proj")
        _output.wrappedValue = try loader.linear(prefix + ".o_proj")
        indexer = try GLM5NextIndexer(c, prefix: prefix + ".indexer", loader: loader)
    }
    /// Graph-only compatibility cache, not an evaluated publication boundary.
    /// Admitted request owners use stageForward and retain its evaluation lease.
    public func forward(_ hidden: MLXArray, cache: GLM5NextDSACache, validity: MLXArray) throws
        -> MLXArray
    {
        let c = configuration
        guard hidden.ndim == 3, hidden.dim(2) == c.hiddenSize,
            validity.shape == Array(hidden.shape.prefix(2)), validity.dtype == .bool
        else { throw GLM5NextLoadError.invalid("DSA input geometry") }
        let residual = qNorm(qA(hidden))
        let q = qB(residual).reshaped(
            hidden.dim(0), hidden.dim(1), c.attentionHeads, c.qkNopeHeadDim)
        let projected = indexer.project(hidden, residual: residual)
        let working = cache.copy() as! GLM5NextDSACache
        let start = working.offset
        try working.append(
            latents: kvNorm(kvA(hidden)), keys: projected.keys, gates: projected.gates,
            validity: validity)
        guard let latents = working.latents, let keys = working.indexKeys,
            let gates = working.poolGates, let keyValidity = working.validity
        else {
            throw GLM5NextLoadError.invalid("DSA history is incomplete")
        }
        let selection = try GLM5NextSparseAttention.selectTokens(
            indexQueries: projected.queries,
            indexKeys: keys, poolGates: gates, poolPositionBias: indexer.positionBias,
            headWeights: projected.weights, keyValidity: keyValidity, queryValidity: validity,
            queryStart: start, tokenBudget: c.indexTopK)
        let result = try attend(q: q, selection: selection) { ids in
            takeAlong(
                latents, clip(ids, min: 0, max: working.offset - 1).expandedDimensions(axis: -1),
                axis: 1)
        }
        cache.adopt(working)
        return result
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
        let residual = qNorm(qA(hidden))
        let q = qB(residual).reshaped(1, hidden.dim(1), c.attentionHeads, c.qkNopeHeadDim)
        let projected = indexer.project(hidden, residual: residual)
        let start = state.committedTokenCount + state.stagedTokenCount
        try state.stage(
            transaction, at: start, latents: kvNorm(kvA(hidden)),
            indexKeys: projected.keys, poolGates: projected.gates, validity: validity)
        let index = try state.indexView(transaction)
        let selection = try GLM5NextSparseAttention.selectTokens(
            indexQueries: projected.queries, indexKeys: index.indexKeys,
            poolGates: index.poolGates, poolPositionBias: indexer.positionBias,
            headWeights: projected.weights, keyValidity: index.validity,
            queryValidity: validity, queryStart: start, tokenBudget: c.indexTopK)
        var leases: [AnyObject] = [index]
        let result = try attend(q: q, selection: selection) { ids in
            let selected = try state.gatherLatents(ids.squeezed(axis: 0), transaction: transaction)
            leases.append(selected)
            return selected.latents
        }
        return StagedOutput(values: result, readLeases: leases)
    }

    /// Native contiguous CBv2 storage is the DSA payload itself: latent K,
    /// and V = [raw index keys, raw pool gates, exact 0/1 validity].
    func forwardCBv2(_ hidden: MLXArray, cache: any CBv2AttendingLayerCache, validity: MLXArray,
                    targetIdentity: ObjectIdentifier) throws -> MLXArray {
        let c = configuration
        let starts = cache.rows.map(\.absoluteOffset)
        let residual = qNorm(qA(hidden))
        let queries = qB(residual).reshaped(hidden.dim(0), hidden.dim(1), c.attentionHeads, c.qkNopeHeadDim)
        let projected = indexer.project(hidden, residual: residual)
        let latents = kvNorm(kvA(hidden))
        let values = concatenated([
            projected.keys, projected.gates, validity.asType(projected.keys.dtype).expandedDimensions(axis: -1),
        ], axis: -1)
        let nativeKeys = latents.asType(.float32).expandedDimensions(axis: 1)
        let nativeValues = values.asType(.float32).expandedDimensions(axis: 1)
        let paged: GLM5NextPagedPayload?
        let history: [(keys: MLXArray, values: MLXArray)]
        if let cache = cache as? PagedLayerCache {
            paged = try cache.beginGLM5NextPayload(keys: nativeKeys, values: nativeValues, targetIdentity: targetIdentity)
            history = []
        } else if let cache = cache as? CBv2LayerCache {
            paged = nil
            history = cache.updateKVAndAdvanceOffsets(keys: nativeKeys, values: nativeValues)
        } else { throw GLM5NextLoadError.invalid("unbound GLM payload cache") }
        var outputs: [MLXArray] = []
        for row in 0..<hidden.dim(0) {
            // Canonical FP32 storage losslessly extends native activation values.
            // Restore each projection's dtype before its trained pooling/matmul.
            let metadata = paged?.values(row: row) ?? history[row].values.squeezed(axis: 1)
            let selection = try GLM5NextSparseAttention.selectTokens(
                indexQueries: projected.queries[row..<row + 1],
                indexKeys: metadata[0..., 0..., ..<c.indexHeadDim].asType(projected.keys.dtype),
                poolGates: metadata[0..., 0..., c.indexHeadDim..<(2 * c.indexHeadDim)].asType(projected.gates.dtype),
                poolPositionBias: indexer.positionBias,
                headWeights: projected.weights[row..<row + 1],
                keyValidity: metadata[0..., 0..., 2 * c.indexHeadDim] .> 0,
                queryValidity: validity[row..<row + 1], queryStart: starts[row], tokenBudget: c.indexTopK)
            outputs.append(try attend(q: queries[row..<row + 1], selection: selection) { ids in
                if let paged { return try paged.selectedKeys(row: row, indices: ids).asType(latents.dtype) }
                let keys = history[row].keys.squeezed(axis: 1).asType(latents.dtype)
                return takeAlong(keys, clip(ids, min: 0, max: keys.dim(1) - 1).expandedDimensions(axis: -1), axis: 1)
            })
        }
        let result = concatenated(outputs, axis: 0)
        paged?.retainResult(result)
        return result
    }

    private func attend(
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
    init(_ c: GLM5NextTextConfiguration, prefix: String, loader: GLM5NextTensorLoader) throws {
        configuration = c
        _q.wrappedValue = try loader.linear(prefix + ".q_proj")
        _k.wrappedValue = try loader.linear(prefix + ".k_proj")
        _v.wrappedValue = try loader.linear(prefix + ".v_proj")
        _fA.wrappedValue = try loader.linear(prefix + ".f_a_proj")
        _fB.wrappedValue = try loader.linear(prefix + ".f_b_proj")
        _gA.wrappedValue = try loader.linear(prefix + ".g_a_proj")
        _gB.wrappedValue = try loader.linear(prefix + ".g_b_proj")
        _beta.wrappedValue = try loader.linear(prefix + ".b_proj")
        _output.wrappedValue = try loader.linear(prefix + ".o_proj")
        _qConv.wrappedValue = GLM5NextWeight(try loader.array(prefix + ".q_conv1d.weight"))
        _kConv.wrappedValue = GLM5NextWeight(try loader.array(prefix + ".k_conv1d.weight"))
        _vConv.wrappedValue = GLM5NextWeight(try loader.array(prefix + ".v_conv1d.weight"))
        _outputNorm.wrappedValue = GLM5NextWeight(try loader.array(prefix + ".o_norm.weight"))
        _aLog.wrappedValue = try loader.array(prefix + ".A_log")
        _dtBias.wrappedValue = try loader.array(prefix + ".dt_bias")
    }
    func forward(
        _ x: MLXArray, cache: GLM5NextKDACache, validity: MLXArray,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy = .reference
    ) throws -> MLXArray {
        let result = try forward(
            x, state: cache.current, validity: validity, recurrenceExecution: recurrenceExecution)
        cache.advance(result.state, count: x.dim(1))
        return result.values
    }

    func forward(
        _ x: MLXArray, state initialState: GLM5NextLinearAttention.State?, validity: MLXArray,
        canonicalConvolution: Bool = false,
        recurrenceExecution: GLM5NextLinearAttention.ExecutionPolicy = .reference
    ) throws
        -> (values: MLXArray, state: GLM5NextLinearAttention.State)
    {
        let c = configuration
        let linear = c.linearAttention
        let shape = [x.dim(0), x.dim(1), linear.numHeads, linear.headDim]
        let masked = MLX.where(validity.expandedDimensions(axis: -1), x, 0)
        let query = q(masked).reshaped(shape)
        let key = k(masked).reshaped(shape)
        let value = v(masked).reshaped(shape)
        let forget = fB(fA(masked)).reshaped(shape)
        let gate = gB(gA(masked)).reshaped(shape)
        let b = beta(masked)
        let conv = concatenated([qConv.weight, kConv.weight, vConv.weight], axis: 0).squeezed(
            axis: 1)
        let weights = GLM5NextLinearAttention.Weights(
            convolution: conv, aLog: aLog, dtBias: dtBias, outputNorm: outputNorm.weight)
        var state: GLM5NextLinearAttention.State =
            initialState.map {
                // Native CBv2 stores raw convolution history in FP32; these
                // values came from query.dtype, so the round trip is exact.
                canonicalConvolution
                    ? GLM5NextLinearAttention.State(convolution: $0.convolution.asType(query.dtype), recurrent: $0.recurrent)
                    : $0
            }
            ?? .init(
                convolution: MLXArray.zeros(
                    [x.dim(0), 3 * linear.numHeads * linear.headDim, linear.shortConvKernelSize],
                    dtype: query.dtype),
                recurrent: MLXArray.zeros(
                    [x.dim(0), linear.numHeads, linear.headDim, linear.headDim], dtype: .float32))
        var rows: [MLXArray] = []
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
        }
        let result = output(
            concatenated(rows, axis: 1).reshaped(
                x.dim(0), x.dim(1), linear.numHeads * linear.headDim))
        return (result, state)
    }
}
