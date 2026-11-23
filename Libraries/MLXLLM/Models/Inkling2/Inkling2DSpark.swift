import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Port of the Inkling 2 DSpark drafter (`mtp.safetensors`): an anchor plus
// mask block runs through a small dense transformer whose attention reads K/V
// built from mixed target features, then a recurrent (`rnn_hidden`) Markov
// head proposes the block's tokens one by one. Numerics follow the vLLM and
// sglang ports: fp32 math, one bf16 rounding where their kernels round.

/// fp32 RMSNorm with one rounding by the caller (MLX's fused kernel rounds
/// `x * rms` to the input type before the weight multiply).
@inline(__always)
func inkling2DSparkRMS(_ x: MLXArray, _ weight: MLXArray, eps: Float) -> MLXArray {
    MLXFast.rmsNorm(x.asType(.float32), weight: weight.asType(.float32), eps: eps)
}

final class Inkling2DSparkAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "k_proj_ctx") var kProjContext: Linear
    @ModuleInfo(key: "v_proj_ctx") var vProjContext: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "k_norm_ctx") var kNormContext: RMSNorm

    init(_ c: Inkling2DSparkConfiguration) {
        let q = c.numAttentionHeads * c.headDim
        let kv = c.numKeyValueHeads * c.headDim
        _qProj.wrappedValue = Linear(c.hiddenSize, q, bias: false)
        _kProj.wrappedValue = Linear(c.hiddenSize, kv, bias: false)
        _vProj.wrappedValue = Linear(c.hiddenSize, kv, bias: false)
        _oProj.wrappedValue = Linear(q, c.hiddenSize, bias: false)
        _kProjContext.wrappedValue = Linear(c.hiddenSize, kv, bias: false)
        _vProjContext.wrappedValue = Linear(c.hiddenSize, kv, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        _kNormContext.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
    }
}

final class Inkling2DSparkMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ c: Inkling2DSparkConfiguration) {
        _gateProj.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = gateProj(x).asType(.float32)
        let up = upProj(x).asType(.float32)
        // silu(gate) * up in fp32, rounded once.
        return downProj((gate / (1 + exp(-gate)) * up).asType(x.dtype))
    }
}

final class Inkling2DSparkLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "dflare_norm") var featureNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attention: Inkling2DSparkAttention
    @ModuleInfo(key: "mlp") var mlp: Inkling2DSparkMLP

    init(_ c: Inkling2DSparkConfiguration) {
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _featureNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _attention.wrappedValue = Inkling2DSparkAttention(c)
        _mlp.wrappedValue = Inkling2DSparkMLP(c)
    }
}

final class Inkling2DSparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var tokenEmbedding: Embedding
    @ModuleInfo(key: "rnn_proj") var rnnProjection: Linear
    @ModuleInfo(key: "hidden_out") var hiddenOut: Linear
    @ModuleInfo(key: "dflare_norm") var featureNorm: RMSNorm

    init(_ c: Inkling2DSparkConfiguration) {
        let r = c.markovRank
        _tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: c.paddedVocabularySize, dimensions: r)
        _rnnProjection.wrappedValue = Linear(2 * r + 2 * c.hiddenSize, 3 * r, bias: true)
        _hiddenOut.wrappedValue = Linear(r, c.hiddenSize, bias: true)
        _featureNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }
}

/// Target tensors the drafter shares: the normed token embedding of the
/// anchor and the unembedding (LM head) for base logits.
public protocol Inkling2DSparkTarget: AnyObject {
    /// `embed_norm(embed(ids))` for `ids` [B, L]: [B, L, hidden].
    func dsparkNormedEmbedding(_ ids: MLXArray) -> MLXArray
    /// The unembedding `model.llm.unembed.weight`, [padded vocabulary, hidden].
    var dsparkUnembedWeight: MLXArray { get }
    /// The certified head over `dsparkUnembedWeight` (int6/int8 proxies and the
    /// exact row product), shared with the target's own certified sampling; nil
    /// when unavailable. The drafter then reads the full unembedding.
    func dsparkCertifiedHead() -> Inkling2CertifiedHead?
}

extension Inkling2DSparkTarget {
    public func dsparkCertifiedHead() -> Inkling2CertifiedHead? { nil }
}

/// Plain-array target used by tests and drafter-only runs.
public final class Inkling2DSparkArrayTarget: Inkling2DSparkTarget {
    public let embedding: MLXArray
    public let embeddingNorm: MLXArray
    public let unembedding: MLXArray
    public let eps: Float
    public let certifiedHead: Inkling2CertifiedHead?

    public init(
        embedding: MLXArray, embeddingNorm: MLXArray, unembedding: MLXArray, eps: Float,
        certifiedHead: Inkling2CertifiedHead? = nil
    ) {
        self.embedding = embedding
        self.embeddingNorm = embeddingNorm
        self.unembedding = unembedding
        self.eps = eps
        self.certifiedHead = certifiedHead
    }

    public func dsparkCertifiedHead() -> Inkling2CertifiedHead? { certifiedHead }

    public func dsparkNormedEmbedding(_ ids: MLXArray) -> MLXArray {
        inkling2DSparkRMS(embedding[ids], embeddingNorm, eps: eps).asType(embedding.dtype)
    }

    public var dsparkUnembedWeight: MLXArray { unembedding }
}

/// The Markov head folded through the target's unembedding:
/// `foldWeight = unembed @ hidden_out.weight`, `foldBias = unembed @ hidden_out.bias`.
public struct Inkling2DSparkFoldedHead {
    public let unembedding: MLXArray
    public let foldWeight: MLXArray
    public let foldBias: MLXArray
    /// Certified base logits (nil: every round reads the full unembedding).
    public var certified: Inkling2DSparkCertifiedUnembed? = nil
}

/// Certified draft selection over the unembedding: drafts equal to the full
/// product's (`rowProduct` base logits, the same bias and softcap, first
/// maximum) while reading the int6 proxy and only the rows in reach.
///
/// Per Markov step and row: the proxy bounds `lo <= b_j <= hi` of the bf16 base
/// logit hold for its FP32 accumulation in any order; bf16 addition is
/// monotone, so `bf16(lo + bias_j) <= s_j <= bf16(hi + bias_j)`; the softcap's
/// monotone envelopes (the GPU's tanh is not monotone everywhere) bound
/// `G(s_j)`. Every row whose upper bound reaches the best lower bound `tau`
/// gets its exact base logit (`exactProducts`, bit-equal to `rowProduct`), and
/// the first maximum among those is the draft: every other row is below `tau`
/// by proof, so it can neither win nor tie.
public struct Inkling2DSparkCertifiedUnembed {
    public let head: Inkling2CertifiedHead
    /// The drafter's softcap `bf16(tanh(f32(s) / cap) * cap)` on every bf16 `s`.
    public let softcap: Inkling2CertifiedHead.Envelope
    /// Bounds of the bf16 base product itself.
    public let identity: Inkling2CertifiedHead.Envelope
    /// Unpadded vocabulary: rows at or above it never win.
    public let vocabularySize: Int
    /// Largest `rows x steps` one proxy read serves.
    public static let rowLimit = 64

    /// Per `(r, j)`: the envelope bounds of `G(bf16(b + bias))` from the bf16
    /// bounds of `b` (bf16 addition is monotone), -inf off the vocabulary.
    /// The grid is exactly `R * N` threads (non-uniform last threadgroup), so
    /// the row count is not a template argument: one pipeline serves every
    /// batch size and the first round of a new row count compiles nothing.
    /// The bf16 sums use MLX's own `bfloat16_t` arithmetic, as the elementwise
    /// path does.
    static let reachKernel = MLXFast.metalKernel(
        name: "inkling2_dspark_certified_reach",
        inputNames: ["lower", "upper", "bias", "envLower", "envUpper"],
        outputNames: ["reachLow", "reachHigh"],
        source: """
            const uint i = thread_position_in_grid.x;
            const uint j = i % uint(N);
            if (j >= uint(V)) {
                reachLow[i] = -INFINITY;
                reachHigh[i] = -INFINITY;
                return;
            }
            const bfloat16_t b = bias[i];
            const bfloat16_t lo = static_cast<bfloat16_t>(lower[i]) + b;
            const bfloat16_t hi = static_cast<bfloat16_t>(upper[i]) + b;
            reachLow[i] = envLower[as_type<ushort>(lo)];
            reachHigh[i] = envUpper[as_type<ushort>(hi)];
            """)

    /// Per row (one threadgroup): over the listed rows in that row's reach
    /// (`reachHigh >= tau`, inside the vocabulary), the first maximum of
    /// `table[bf16(exact + bias)]`; MLX's argMax rule for NaN (the first NaN).
    static let selectKernel = MLXFast.metalKernel(
        name: "inkling2_dspark_certified_select",
        inputNames: ["exact", "bias", "table", "reachHigh", "tau", "list", "count"],
        outputNames: ["token"],
        source: """
            const uint r = threadgroup_position_in_grid.x;
            const uint tid = thread_index_in_threadgroup;
            const uint total = count[0];
            const float floorTau = tau[r];
            float best = -INFINITY;
            uint index = 0xFFFFFFFFu;
            uint firstNaN = 0xFFFFFFFFu;
            for (uint k = tid; k < total; k += THREADS) {
                const uint j = list[k];
                const size_t at = size_t(r) * N + j;
                if (j >= uint(V) || !(reachHigh[at] >= floorTau)) {
                    continue;
                }
                const bfloat16_t s = exact[at] + bias[at];
                const float f = table[as_type<ushort>(s)];
                if (isnan(f)) {
                    firstNaN = min(firstNaN, j);
                } else if (f > best || (f == best && j < index)) {
                    best = f;
                    index = j;
                }
            }
            threadgroup float bests[THREADS];
            threadgroup uint indices[THREADS];
            threadgroup uint nans[THREADS];
            bests[tid] = best;
            indices[tid] = index;
            nans[tid] = firstNaN;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint half_ = THREADS / 2; half_ > 0; half_ >>= 1) {
                if (tid < half_) {
                    const float ob = bests[tid + half_];
                    const uint oi = indices[tid + half_];
                    if (ob > bests[tid] || (ob == bests[tid] && oi < indices[tid])) {
                        bests[tid] = ob;
                        indices[tid] = oi;
                    }
                    nans[tid] = min(nans[tid], nans[tid + half_]);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (tid == 0) {
                token[r] = int(nans[0] != 0xFFFFFFFFu ? nans[0] : indices[0]);
            }
            """)

    /// Greedy drafts [R] int32 for bounds `lower`/`upper` [R, N] of the bf16
    /// base logits of `rows` [R, K], with the step's `bias` [R, N] bf16.
    func select(rows: MLXArray, lower: MLXArray, upper: MLXArray, bias: MLXArray) -> MLXArray {
        let (r, n) = (bias.dim(0), bias.dim(1))
        let reach = Self.reachKernel(
            [lower, upper, bias, softcap.lower, softcap.upper],
            template: [("N", n), ("V", vocabularySize)],
            grid: (r * n, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[r, n], [r, n]], outputDTypes: [.float32, .float32])
        let tau = reach[0].max(axis: -1)
        let list = head.candidates(upper: reach[1], atLeast: tau)
        let exact = head.exactProducts(rows, indices: list.indices, count: list.count)
        return Self.selectKernel(
            [exact, bias, softcap.table, reach[1], tau, list.indices, list.count],
            template: [("N", n), ("V", vocabularySize), ("THREADS", 256)],
            grid: (r * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[r]], outputDTypes: [.int32])[0]
    }

    public init(head: Inkling2CertifiedHead, cap: Float, vocabularySize: Int, dtype: DType) {
        self.head = head
        self.vocabularySize = vocabularySize
        self.identity = .identity()
        self.softcap = .init { x in
            let logits = x.asType(.float32)
            let capped = cap > 0 ? tanh(logits / cap) * cap : logits
            return capped.asType(dtype)
        }
    }
}

/// One proposal's per-row Markov inputs, built once per block.
struct Inkling2DSparkMarkovPlan {
    /// [B, S, vocabulary] bf16 base logits of the S mask rows (full path).
    let baseLogits: MLXArray?
    /// [B, S, 3r] fp32: `rnn_proj` bias plus its `[hidden, f_rnn]` term.
    let context: MLXArray
    /// Certified path: the mask rows [B, S, hidden] and their proxy bounds
    /// [B, S, vocabulary] (float32 holding bf16 values) of the base product.
    var rows: MLXArray? = nil
    var lower: MLXArray? = nil
    var upper: MLXArray? = nil
}

public final class Inkling2DSparkModel: Module {
    public let configuration: Inkling2DSparkConfiguration

    @ModuleInfo(key: "layers") var layers: [Inkling2DSparkLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "markov_head") var markovHead: Inkling2DSparkMarkovHead
    @ParameterInfo(key: "fuse_weight") var fuseWeight: MLXArray
    @ParameterInfo(key: "mask_embed") var maskEmbedding: MLXArray

    /// Derived after loading (`prepareDerivedWeights`).
    private(set) var fuseAlpha: MLXArray?
    private(set) var ropeFrequencies: MLXArray?
    private var rnnStateWeight: MLXArray?
    private var rnnContextWeight: MLXArray?
    private var rnnBias: MLXArray?

    public init(_ configuration: Inkling2DSparkConfiguration) {
        self.configuration = configuration
        _layers.wrappedValue = (0 ..< configuration.numHiddenLayers).map { _ in
            Inkling2DSparkLayer(configuration)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
        _markovHead.wrappedValue = Inkling2DSparkMarkovHead(configuration)
        _fuseWeight.wrappedValue = MLXArray.zeros([
            configuration.numHiddenLayers + 1, configuration.featureCount,
        ])
        _maskEmbedding.wrappedValue = MLXArray.zeros([configuration.hiddenSize])
    }

    var eps: Float { configuration.rmsNormEps }
    var dtype: DType { maskEmbedding.dtype }

    /// Weights derived from loaded parameters. Call after every weight update.
    public func prepareDerivedWeights() {
        let c = configuration
        let r = c.markovRank
        fuseAlpha = softmax(fuseWeight.asType(.float32), axis: -1)
        let exponents =
            MLXArray(stride(from: 0, to: c.headDim, by: 2).map { Float($0) })
            / Float(c.headDim)
        ropeFrequencies = pow(MLXArray(c.ropeTheta), exponents)
        let weight = markovHead.rnnProjection.weight.asType(.float32)
        rnnStateWeight = weight[0..., 0 ..< (2 * r)].transposed()
        rnnContextWeight = weight[0..., (2 * r)...].transposed()
        rnnBias = markovHead.rnnProjection.bias!.asType(.float32)
        eval(fuseAlpha!, ropeFrequencies!, rnnStateWeight!, rnnContextWeight!, rnnBias!)
    }

    /// Folds `hidden_out` into the unembedding in fp32 chunks (the reference
    /// folds in fp64; both round once to bf16).
    public func foldHead(unembedding: MLXArray, chunk: Int = 16384) -> Inkling2DSparkFoldedHead {
        let projection = markovHead.hiddenOut.weight.asType(.float32)
        let bias = markovHead.hiddenOut.bias!.asType(.float32)
        var weights: [MLXArray] = []
        var biases: [MLXArray] = []
        let rows = unembedding.dim(0)
        for start in stride(from: 0, to: rows, by: chunk) {
            let head = unembedding[start ..< min(start + chunk, rows)].asType(.float32)
            let w = matmul(head, projection).asType(unembedding.dtype)
            let b = matmul(head, bias).asType(unembedding.dtype)
            eval(w, b)
            weights.append(w)
            biases.append(b)
        }
        let foldWeight = concatenated(weights, axis: 0)
        let foldBias = concatenated(biases, axis: 0)
        eval(foldWeight, foldBias)
        return .init(unembedding: unembedding, foldWeight: foldWeight, foldBias: foldBias)
    }

    // MARK: Features

    /// RMSNorm of the `fuse_alpha[row]`-weighted sum of the captured target
    /// features. `features` is [N, featureCount * hidden]; returns [N, hidden].
    func mixFeatures(_ features: MLXArray, row: Int, weight: MLXArray) -> MLXArray {
        let c = configuration
        let n = features.dim(0)
        let taps = features.reshaped(n, c.featureCount, c.hiddenSize).asType(.float32)
        let alpha = fuseAlpha![row]
        var mixed = taps[0..., 0, 0...] * alpha[0]
        for tap in 1 ..< c.featureCount {
            mixed = mixed + taps[0..., tap, 0...] * alpha[tap]
        }
        return inkling2DSparkRMS(mixed, weight, eps: eps).asType(dtype)
    }

    func rope(_ x: MLXArray, offset: Int) -> MLXArray {
        MLXFast.RoPE(
            x.asType(.float32), dimensions: configuration.headDim, traditional: false,
            base: nil, scale: 1, offset: offset, freqs: ropeFrequencies!
        ).asType(dtype)
    }

    func rope(_ x: MLXArray, offsets: MLXArray) -> MLXArray {
        MLXFast.RoPE(
            x.asType(.float32), dimensions: configuration.headDim, traditional: false,
            base: nil, scale: 1, offset: offsets, freqs: ropeFrequencies!
        ).asType(dtype)
    }

    /// Context K/V of draft layer `layer` for `features` [N, featureCount *
    /// hidden] at positions `start ..< start + N`: [1, kvHeads, N, headDim].
    public func contextKV(layer index: Int, features: MLXArray, start: Int) -> (
        keys: MLXArray, values: MLXArray
    ) {
        let c = configuration
        let layer = layers[index]
        let n = features.dim(0)
        let context = mixFeatures(features, row: index, weight: layer.featureNorm.weight)
        var keys = layer.attention.kProjContext(context)
            .reshaped(1, n, c.numKeyValueHeads, c.headDim)
        let values = layer.attention.vProjContext(context)
            .reshaped(1, n, c.numKeyValueHeads, c.headDim).transposed(0, 2, 1, 3)
        keys = inkling2DSparkRMS(keys, layer.attention.kNormContext.weight, eps: eps)
            .asType(dtype).transposed(0, 2, 1, 3)
        return (rope(keys, offset: start), values)
    }

    // MARK: Block

    private func addNorm(_ x: MLXArray, _ residual: MLXArray, _ weight: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        let sum = residual.asType(.float32) + x.asType(.float32)
        return (inkling2DSparkRMS(sum, weight, eps: eps).asType(dtype), sum.asType(dtype))
    }

    /// Final draft hidden [B, block, hidden] for blocks whose anchors have the
    /// normed embeddings `anchors` [B, hidden] and sit at `positions` [B].
    /// `contexts[i]` holds layer i's context K/V [B, kvHeads, T, headDim]
    /// (already window-limited) and an optional additive or boolean mask
    /// [B, 1, block, T + block] for padded rows.
    func blockHidden(
        anchors: MLXArray, positions: [Int],
        contexts: [(keys: MLXArray, values: MLXArray, mask: MLXArray?)]
    ) -> MLXArray {
        let c = configuration
        let batch = anchors.dim(0)
        let block = c.blockSize
        let masks = broadcast(
            maskEmbedding.reshaped(1, 1, c.hiddenSize),
            to: [batch, block - 1, c.hiddenSize])
        var x = concatenated(
            [anchors.reshaped(batch, 1, c.hiddenSize).asType(dtype), masks], axis: 1)
        var residual = x
        var h = inkling2DSparkRMS(x, layers[0].inputNorm.weight, eps: eps).asType(dtype)
        let offsets = MLXArray(positions.map { Int32($0) })
        let scale = 1 / Float(c.headDim).squareRoot()
        for (index, layer) in layers.enumerated() {
            if index > 0 {
                (h, residual) = addNorm(x, residual, layer.inputNorm.weight)
            }
            let attention = layer.attention
            var q = attention.qProj(h).reshaped(batch, block, c.numAttentionHeads, c.headDim)
            var k = attention.kProj(h).reshaped(batch, block, c.numKeyValueHeads, c.headDim)
            let v = attention.vProj(h).reshaped(batch, block, c.numKeyValueHeads, c.headDim)
                .transposed(0, 2, 1, 3)
            q = inkling2DSparkRMS(q, attention.qNorm.weight, eps: eps).asType(dtype)
                .transposed(0, 2, 1, 3)
            k = inkling2DSparkRMS(k, attention.kNorm.weight, eps: eps).asType(dtype)
                .transposed(0, 2, 1, 3)
            if batch == 1 {
                q = rope(q, offset: positions[0])
                k = rope(k, offset: positions[0])
            } else {
                q = rope(q, offsets: offsets)
                k = rope(k, offsets: offsets)
            }
            let context = contexts[index]
            let keys = concatenated([context.keys, k], axis: 2)
            let values = concatenated([context.values, v], axis: 2)
            let output = MLXFast.scaledDotProductAttention(
                queries: q, keys: keys, values: values, scale: scale, mask: context.mask)
            let attended = attention.oProj(
                output.transposed(0, 2, 1, 3).reshaped(batch, block, -1))
            (h, residual) = addNorm(attended, residual, layer.postAttentionNorm.weight)
            x = layer.mlp(h)
        }
        return addNorm(x, residual, norm.weight).0
    }

    // MARK: Markov head

    /// `f_rnn` from the features of the last committed position [B, featureWidth].
    func markovFeatures(_ features: MLXArray) -> MLXArray {
        mixFeatures(
            features, row: configuration.numHiddenLayers,
            weight: markovHead.featureNorm.weight)
    }

    /// Base logits and recurrent context for the mask rows `rows` [B, S, hidden].
    func markovPlan(rows: MLXArray, markovFeatures: MLXArray, head: Inkling2DSparkFoldedHead)
        -> Inkling2DSparkMarkovPlan
    {
        let (batch, steps, hidden) = (rows.dim(0), rows.dim(1), rows.dim(2))
        let flat = rows.reshaped(batch * steps, hidden)
        let f = broadcast(
            markovFeatures.reshaped(batch, 1, hidden), to: [batch, steps, hidden])
        let inputs = concatenated([rows, f], axis: -1).asType(.float32)
        let context = rnnBias! + matmul(inputs, rnnContextWeight!)
        if let certified = head.certified, flat.dtype == .bfloat16,
            batch * steps <= Inkling2DSparkCertifiedUnembed.rowLimit,
            certified.head.columns == hidden, certified.head.rows == head.unembedding.dim(0)
        {
            // One int6 proxy read bounds every mask row's base logits.
            let bounds = certified.head.proxyBounds(
                flat, proxy: .int6, envelope: certified.identity)
            return .init(
                baseLogits: nil, context: context, rows: rows,
                lower: bounds.lower.reshaped(batch, steps, -1),
                upper: bounds.upper.reshaped(batch, steps, -1))
        }
        let base = Self.baseProduct(flat, head.unembedding).reshaped(batch, steps, -1)
        return .init(baseLogits: base, context: context)
    }

    /// The base logits' product: the row-exact tile the certified path
    /// reproduces (`Inkling2CertifiedHead.exactProducts`) when the kernels
    /// cover it, else MLX's product.
    static func baseProduct(_ rows: MLXArray, _ unembedding: MLXArray) -> MLXArray {
        Inkling2Kernels.rowProduct(rows, transposedWeight: unembedding)
            ?? matmul(rows, unembedding.transposed())
    }

    /// One recurrence step for every row: returns the new state [B, r] and
    /// the latent [B, r].
    func markovStep(state: MLXArray, previous: MLXArray, context: MLXArray) -> (
        state: MLXArray, latent: MLXArray
    ) {
        let r = configuration.markovRank
        let embedded = markovHead.tokenEmbedding(previous.reshaped(-1))
        let inputs = concatenated([state, embedded], axis: -1).asType(.float32)
        let raw = context + matmul(inputs, rnnStateWeight!)
        let gate = sigmoid(raw[0..., 0 ..< r].asType(dtype))
        let candidate = tanh(raw[0..., r ..< (2 * r)].asType(dtype))
        let out = raw[0..., (2 * r)...].asType(dtype)
        let next = gate * state + (1 - gate) * candidate
        return (next, tanh(out))
    }

    /// Softcapped draft logits over the unpadded vocabulary [B, vocabulary].
    func draftLogits(base: MLXArray, latent: MLXArray, head: Inkling2DSparkFoldedHead)
        -> MLXArray
    {
        let c = configuration
        let bias = addMM(head.foldBias, latent, head.foldWeight.transposed())
        var logits = (base + bias).asType(.float32)
        if c.finalLogitSoftcapping > 0 {
            let cap = c.finalLogitSoftcapping
            logits = tanh(logits / cap) * cap
        }
        return logits.asType(dtype)[0..., 0 ..< c.vocabularySize]
    }

    /// Greedy draft ids [B] int32.
    func draftTokens(base: MLXArray, latent: MLXArray, head: Inkling2DSparkFoldedHead)
        -> MLXArray
    {
        argMax(draftLogits(base: base, latent: latent, head: head), axis: -1).asType(.int32)
    }

    /// Greedy draft ids [B] int32 of Markov step `step`: the full product's
    /// draft, through the certified candidates when the plan holds bounds.
    func draftTokens(
        plan: Inkling2DSparkMarkovPlan, step: Int, latent: MLXArray,
        head: Inkling2DSparkFoldedHead
    ) -> MLXArray {
        guard let certified = head.certified, let lower = plan.lower, let upper = plan.upper,
            let rows = plan.rows
        else {
            return draftTokens(base: plan.baseLogits![0..., step, 0...], latent: latent, head: head)
        }
        let bias = addMM(head.foldBias, latent, head.foldWeight.transposed())
        return certified.select(
            rows: rows[0..., step, 0...], lower: lower[0..., step, 0...],
            upper: upper[0..., step, 0...], bias: bias)
    }

    /// Zero recurrent state for `batch` rows.
    func initialMarkovState(batch: Int) -> MLXArray {
        MLXArray.zeros([batch, configuration.markovRank], dtype: dtype)
    }

    // MARK: Whole proposal (reference path)

    /// Proposes `depth` tokens for one row whose committed context K/V per
    /// layer is `contexts` (window-limited), anchor token `anchor` [1] at
    /// `position`, with `lastFeatures` [1, featureWidth] of position - 1.
    /// Returns ([depth] int32 drafts, [1, block, hidden] draft hidden).
    func propose(
        anchor: MLXArray, position: Int, lastFeatures: MLXArray,
        contexts: [(keys: MLXArray, values: MLXArray, mask: MLXArray?)],
        target: any Inkling2DSparkTarget, head: Inkling2DSparkFoldedHead, depth: Int
    ) -> (drafts: MLXArray, hidden: MLXArray) {
        let anchors = target.dsparkNormedEmbedding(anchor.reshaped(1, 1)).reshaped(1, -1)
        let hidden = blockHidden(anchors: anchors, positions: [position], contexts: contexts)
        let plan = markovPlan(
            rows: hidden[0..., 1 ... depth, 0...], markovFeatures: markovFeatures(lastFeatures),
            head: head)
        var state = initialMarkovState(batch: 1)
        var previous = anchor.reshaped(1).asType(.int32)
        var drafts: [MLXArray] = []
        for step in 0 ..< depth {
            let next = markovStep(
                state: state, previous: previous, context: plan.context[0..., step, 0...])
            state = next.state
            previous = draftTokens(plan: plan, step: step, latent: next.latent, head: head)
            drafts.append(previous)
        }
        return (concatenated(drafts, axis: 0), hidden)
    }

    // MARK: Loading

    /// Checkpoint names map one to one onto this module tree.
    public func load(weights: [String: MLXArray]) throws {
        let expected = Set(parameters().flattened().map(\.0))
        let provided = Set(weights.keys)
        let missing = expected.subtracting(provided)
        let unexpected = provided.subtracting(expected)
        guard missing.isEmpty, unexpected.isEmpty else {
            throw Inkling2DSparkError.weights(
                "missing \(missing.sorted().prefix(8)), unexpected \(unexpected.sorted().prefix(8))"
            )
        }
        try update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(self)
        prepareDerivedWeights()
    }

    /// Loads `mtp.safetensors` from an Inkling 2 checkpoint directory.
    public static func load(directory: URL) throws -> Inkling2DSparkModel {
        let configuration = try Inkling2DSparkConfiguration.read(
            configURL: directory.appending(component: "config.json"))
        let model = Inkling2DSparkModel(configuration)
        let weights = try loadArrays(url: directory.appending(component: "mtp.safetensors"))
        try model.load(weights: weights)
        return model
    }
}
