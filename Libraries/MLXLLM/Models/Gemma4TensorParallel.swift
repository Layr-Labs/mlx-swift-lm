// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Refusal of a Gemma 4 tensor-parallel geometry or installation.
public struct Gemma4TensorParallelError: Error, CustomStringConvertible, Equatable, Sendable {
    public let description: String
    init(_ description: String) { self.description = description }
}

/// One rank's equal half of a channel width. With an odd group count both ranks store the
/// middle down-projection group, and zero gate/up rows make the other rank's half of it add 0.
public struct Gemma4ChannelShard: Sendable, Equatable {
    public let width: Int
    public let groupSize: Int
    public let rank: Int
    /// The channels of the full width this rank computes.
    public let channels: Range<Int>
    /// The down projection's input columns this rank stores: `channels`
    /// widened to whole quantization groups.
    public let storedColumns: Range<Int>

    public init(width: Int, groupSize: Int, rank: Int, size: Int = 2) throws {
        guard size == 2, (0 ..< size).contains(rank) else {
            throw Gemma4TensorParallelError("Gemma 4 tensor parallel runs two ranks")
        }
        guard width > 0, groupSize > 0, width % groupSize == 0, width % size == 0 else {
            throw Gemma4TensorParallelError(
                "channel width \(width) is not whole quantization groups of \(groupSize)")
        }
        let half = width / size
        self.width = width
        self.groupSize = groupSize
        self.rank = rank
        channels = rank == 0 ? 0 ..< half : half ..< width
        let lower = channels.lowerBound / groupSize * groupSize
        let upper = (channels.upperBound + groupSize - 1) / groupSize * groupSize
        storedColumns = lower ..< upper
    }

    /// Every rank's local width (equal for two ranks: an odd group count
    /// shares exactly the middle group).
    public var localWidth: Int { storedColumns.count }
    /// Where `channels` lie in the local width; the other local channels are zero.
    public var localChannels: Range<Int> {
        (channels.lowerBound - storedColumns.lowerBound)
            ..< (channels.upperBound - storedColumns.lowerBound)
    }
    /// Zero channels before and after `localChannels`.
    public var paddingBefore: Int { localChannels.lowerBound }
    public var paddingAfter: Int { localWidth - localChannels.upperBound }
}

/// One rank of a two-rank Gemma 4 MoE text model: half the query heads with their KV heads, half
/// of every channel width (`Gemma4ChannelShard`) and half the vocabulary rows of the tied head.
/// Each layer sums the attention output and one packed feed-forward collective across ranks.
public struct Gemma4TensorShard: Sendable, Equatable {
    public let rank: Int
    public let size: Int

    public init(rank: Int, size: Int = 2) throws {
        guard size == 2, (0 ..< size).contains(rank) else {
            throw Gemma4TensorParallelError("Gemma 4 tensor parallel runs two ranks")
        }
        self.rank = rank
        self.size = size
    }

    /// Group size of the channel-parallel down projections.
    static func groupSize(_ c: Gemma4TextConfiguration) -> Int { c.quantizationGroupSize ?? 64 }

    public func denseChannels(_ c: Gemma4TextConfiguration) throws -> Gemma4ChannelShard {
        try Gemma4ChannelShard(
            width: c.intermediateSize, groupSize: Self.groupSize(c), rank: rank, size: size)
    }

    public func expertChannels(_ c: Gemma4TextConfiguration) throws -> Gemma4ChannelShard {
        guard let width = c.moeIntermediateSize else {
            throw Gemma4TensorParallelError("Gemma 4 tensor parallel needs the MoE block")
        }
        return try Gemma4ChannelShard(
            width: width, groupSize: Self.groupSize(c), rank: rank, size: size)
    }

    /// Vocabulary rows this rank's head projects.
    public func vocabularyRange(_ c: Gemma4TextConfiguration) throws -> Range<Int> {
        guard c.vocabSize > 0, c.vocabSize % size == 0 else {
            throw Gemma4TensorParallelError("vocabulary \(c.vocabSize) does not divide by \(size)")
        }
        let half = c.vocabSize / size
        return (rank * half) ..< ((rank + 1) * half)
    }

    /// The configuration this rank's modules are built from: half the query
    /// and KV heads of each layer kind and the local channel widths. The
    /// vocabulary, embedding and every norm stay whole.
    public func localConfiguration(_ c: Gemma4TextConfiguration) throws
        -> Gemma4TextConfiguration
    {
        guard c.enableMoeBlock, let experts = c.numExperts, let topK = c.topKExperts,
            (1 ... experts).contains(topK)
        else { throw Gemma4TensorParallelError("Gemma 4 tensor parallel needs the MoE block") }
        guard c.numKvSharedLayers == 0, c.hiddenSizePerLayerInput == 0 else {
            throw Gemma4TensorParallelError(
                "shared-KV layers and per-layer embeddings are not covered")
        }
        guard c.tieWordEmbeddings else {
            throw Gemma4TensorParallelError("only the tied head is covered")
        }
        let global = c.numGlobalKeyValueHeads ?? c.numKeyValueHeads
        for (label, kv) in [("sliding", c.numKeyValueHeads), ("full", global)] {
            guard kv > 0, kv % size == 0, c.numAttentionHeads % kv == 0 else {
                throw Gemma4TensorParallelError(
                    "\(label) KV heads (\(kv)) do not split into whole query groups")
            }
        }
        guard c.numAttentionHeads % size == 0 else {
            throw Gemma4TensorParallelError("query heads do not divide by \(size)")
        }
        _ = try vocabularyRange(c)
        var local = c
        local.numAttentionHeads = c.numAttentionHeads / size
        local.numKeyValueHeads = c.numKeyValueHeads / size
        local.numGlobalKeyValueHeads = c.numGlobalKeyValueHeads.map { $0 / size }
        local.intermediateSize = try denseChannels(c).localWidth
        local.moeIntermediateSize = try expertChannels(c).localWidth
        return local
    }
}

/// Submits a CBv2 forward's hidden rows every `stride` decoder layers (0: never).
/// Evaluation order only: every value is the one the whole forward computes.
public enum Gemma4LayerSubmission {
    @TaskLocal public static var stride = 0
}

/// Unrounded FP32 partials of a row-parallel projection.
public enum Gemma4TensorParallelProjection {
    /// This rank's FP32 partial of `linear` over `x`: the input widened
    /// exactly and nothing rounded. Under the scoped batch-decode or
    /// registered prompt policy each row keeps its singleton (M=1) reduction,
    /// as the unsharded projection does there.
    public static func partial(_ linear: Linear, _ x: MLXArray) -> MLXArray {
        let input = x.asType(.float32)
        guard x.ndim == 3,
            Gemma4BatchDecodeProjection.applies(x) || Gemma4PrefillProjection.applies(x)
        else { return linear(input) }
        let rows = input.reshaped([x.dim(0) * x.dim(1), 1, x.dim(2)])
        guard type(of: linear) == QuantizedLinear.self, let packed = linear as? QuantizedLinear,
            packed.bias == nil,
            Gemma4BatchDecodeProjection.supports(
                packed.mode, bits: packed.bits, groupSize: packed.groupSize,
                input: input, scales: packed.scales)
        else {
            return Gemma4BatchDecodeProjection.perRow(rows, linear.callAsFunction)
                .reshaped([x.dim(0), x.dim(1), -1])
        }
        return Gemma4BatchDecodeProjection.project(
            rows, weight: packed.weight, scales: packed.scales, biases: packed.biases,
            groupSize: packed.groupSize, bits: packed.bits
        )
        .reshaped([x.dim(0), x.dim(1), -1])
    }
}

/// One rank's cross-rank sum, shared by every layer of a tensor-parallel
/// model. Not a module: it stays out of the parameter tree.
final class Gemma4TensorParallelContext {
    let shard: Gemma4TensorShard
    let allSum: (MLXArray) -> MLXArray
    /// Diagnostic only when false: the dense partial takes its own collective.
    let packsFeedForward: Bool

    init(shard: Gemma4TensorShard, packsFeedForward: Bool, allSum: @escaping (MLXArray) -> MLXArray)
    {
        self.shard = shard
        self.packsFeedForward = packsFeedForward
        self.allSum = allSum
    }

    /// A row-parallel projection: the FP32 partials summed across ranks,
    /// then one rounding to the activation dtype.
    func reduce(_ projection: Linear, _ x: MLXArray) -> MLXArray {
        allSum(Gemma4TensorParallelProjection.partial(projection, x)).asType(x.dtype)
    }
}

/// The tied head over this rank's vocabulary rows: the same packed rows the
/// whole head reads (row views of the replicated embedding, never copies),
/// through the kernel the whole head uses for the same hidden rows.
struct Gemma4TensorParallelHead {
    let rows: Range<Int>

    func project(_ embedding: Embedding, _ hidden: MLXArray) -> MLXArray {
        guard let packed = embedding as? QuantizedEmbedding else {
            precondition(
                type(of: embedding) == Embedding.self,
                "Gemma 4 tensor parallel splits a plain or affine-quantized tied head only")
            return matmul(hidden, embedding.weight[rows].T)
        }
        precondition(
            type(of: packed) == QuantizedEmbedding.self && packed.mode == .affine,
            "Gemma 4 tensor parallel splits a plain or affine-quantized tied head only")
        let weight = packed.weight[rows]
        let scales = packed.scales[rows]
        let biases = packed.biases?[rows]
        func whole(_ input: MLXArray) -> MLXArray {
            quantizedMM(
                input, weight, scales: scales, biases: biases, transpose: true,
                groupSize: packed.groupSize, bits: packed.bits)
        }
        guard Gemma4BatchDecodeProjection.applies(hidden) else { return whole(hidden) }
        guard
            Gemma4BatchDecodeProjection.supports(
                packed.mode, bits: packed.bits, groupSize: packed.groupSize,
                input: hidden, scales: packed.scales)
        else { return Gemma4BatchDecodeProjection.perRow(hidden, whole) }
        return Gemma4BatchDecodeProjection.project(
            hidden, weight: weight, scales: scales, biases: biases,
            groupSize: packed.groupSize, bits: packed.bits)
    }
}

extension Gemma4DecoderLayer {
    /// The feed-forward of a tensor-parallel rank: the dense branch's FP32
    /// partial rides in the experts' collective, and each branch's sum is
    /// rounded and normalized exactly where the unsharded layer rounds and
    /// normalizes that branch.
    func tensorParallelFeedForward(
        _ out: MLXArray, router: Gemma4Router, experts: Gemma4Experts,
        postFeedforwardLayernorm1: RMSNorm, preFeedforwardLayernorm2: RMSNorm,
        postFeedforwardLayernorm2: RMSNorm, context: Gemma4TensorParallelContext,
        isExpertPrefill: Bool
    ) -> MLXArray {
        let h1 = preFeedforwardLayernorm(out)
        let densePartial = mlp.tensorParallelPartial(h1)
        let (topKIndices, topKWeights) = router(out)
        let h2 = preFeedforwardLayernorm2(out)
        let expertOutput: MLXArray
        let denseSum: MLXArray
        if context.packsFeedForward {
            (expertOutput, denseSum) = experts.tensorParallelCall(
                h2, topKIndices: topKIndices, topKWeights: topKWeights,
                isExpertPrefill: isExpertPrefill, carrying: densePartial)
        } else {
            denseSum = context.allSum(densePartial)
            expertOutput = experts(
                h2, topKIndices: topKIndices, topKWeights: topKWeights,
                isExpertPrefill: isExpertPrefill)
        }
        let dense = postFeedforwardLayernorm1(denseSum.asType(h1.dtype))
        return dense + postFeedforwardLayernorm2(expertOutput)
    }
}

extension Gemma4Experts {
    /// Every selected expert's FP32 partial and `densePartial` summed in one collective: the
    /// weighted expert output and the dense FP32 sum. Scoped singleton slices pack into it too.
    func tensorParallelCall(
        _ x: MLXArray, topKIndices: MLXArray, topKWeights: MLXArray,
        isExpertPrefill: Bool, carrying densePartial: MLXArray
    ) -> (output: MLXArray, denseSum: MLXArray) {
        guard let down = switchGLU.downProjection as? TensorParallelSwitchLinear else {
            preconditionFailure("A tensor-parallel Gemma 4 rank sums its experts' partials")
        }
        let (B, S, H) = (x.dim(0), x.dim(1), x.dim(2))
        let K = topKIndices.dim(-1)
        let byToken = Gemma4PrefillProjection.appliesExperts(
            x, indices: topKIndices, weights: topKWeights, isExpertPrefill: isExpertPrefill)
        guard byToken || Gemma4BatchDecodeProjection.applies(x) else {
            let packed = down.carrying(densePartial) {
                switchGLU.callAndWeightedReduce(
                    x.reshaped(B * S, H), topKIndices.reshaped(B * S, K),
                    weights: topKWeights.reshaped(B * S, K),
                    fuseSortedReduction: fuseWeightedUnsort,
                    isProductionPrefill: isExpertPrefill)
            }
            return (packed.output.reshaped(B, S, H), packed.rowsSum)
        }
        let count = byToken ? S : B
        func slice(_ value: MLXArray, _ index: Int) -> MLXArray {
            byToken ? value[0..., index ..< index + 1, 0...] : value[index ..< index + 1]
        }
        var partials: [MLXArray] = []
        for index in 0 ..< count {
            let rows = slice(x, index)
            let routes = slice(topKIndices, index)
            let tokens = rows.dim(0) * rows.dim(1)
            partials.append(
                down.withoutCollective {
                    switchGLU(rows.reshaped(tokens, H), routes.reshaped(tokens, K))
                }
                .reshaped([-1, H]))
        }
        let expertRows = partials.reduce(0) { $0 + $1.dim(0) }
        let summed = down.sum(concatenated(partials + [densePartial.reshaped([-1, H])], axis: 0))
        var offset = 0
        let outputs = (0 ..< count).map { index -> MLXArray in
            let rows = slice(x, index)
            let tokens = rows.dim(0) * rows.dim(1)
            let experts = summed[offset ..< offset + tokens * K].reshaped([tokens, K, H])
                .asType(x.dtype)
            offset += tokens * K
            return weightedExpertSum(experts, slice(topKWeights, index).reshaped(tokens, K))
                .reshaped(rows.shape)
        }
        return (
            concatenated(outputs, axis: byToken ? 1 : 0),
            summed[expertRows...].reshaped(densePartial.shape)
        )
    }
}

extension Gemma4TextModel {
    /// The shard of an installed tensor-parallel rank; nil for a whole model.
    public var tensorParallelShard: Gemma4TensorShard? {
        model.layers.first?.tensorParallel?.shard
    }

    /// Vocabulary rows this model's head projects (all of them unless a
    /// tensor-parallel rank is installed).
    public var headVocabularyRange: Range<Int> {
        tensorParallelHead?.rows ?? 0 ..< configuration.vocabSize
    }

    /// Makes this model, built from `shard.localConfiguration(full)` with the rank's cut weights,
    /// one rank of a pair. `allSum` must stay lazy; `packFeedForward: false` is a diagnostic.
    public func installTensorParallel(
        shard: Gemma4TensorShard, full: Gemma4TextConfiguration,
        packFeedForward: Bool = true,
        allSum: @escaping (MLXArray) -> MLXArray
    ) throws {
        guard tensorParallelShard == nil else {
            throw Gemma4TensorParallelError("tensor parallel is already installed")
        }
        let local = try shard.localConfiguration(full)
        let own = configuration
        guard own.numAttentionHeads == local.numAttentionHeads,
            own.numKeyValueHeads == local.numKeyValueHeads,
            own.numGlobalKeyValueHeads == local.numGlobalKeyValueHeads,
            own.intermediateSize == local.intermediateSize,
            own.moeIntermediateSize == local.moeIntermediateSize,
            own.hiddenSize == full.hiddenSize, own.numHiddenLayers == full.numHiddenLayers,
            own.layerTypes == full.layerTypes, own.vocabSize == full.vocabSize
        else {
            throw Gemma4TensorParallelError(
                "the model was not built from rank \(shard.rank)'s local configuration")
        }
        let context = Gemma4TensorParallelContext(
            shard: shard, packsFeedForward: packFeedForward, allSum: allSum)
        var replacements: [(String, Module)] = []
        for (index, layer) in model.layers.enumerated() {
            guard let experts = layer.experts,
                let source = experts.switchGLU.downProjection as? QuantizedSwitchLinear
            else {
                throw Gemma4TensorParallelError(
                    "layer \(index) has no quantized expert down projection")
            }
            let projection: TensorParallelSwitchLinear
            do {
                projection = try TensorParallelSwitchLinear(source, allSum: allSum)
            } catch {
                throw Gemma4TensorParallelError(
                    "layer \(index)'s expert down projection is not the affine contract")
            }
            replacements.append(("model.layers.\(index).experts.switch_glu.down_proj", projection))
        }
        try update(modules: ModuleChildren.unflattened(replacements), verify: .none)
        for layer in model.layers {
            layer.tensorParallel = context
            layer.selfAttn.tensorParallel = context
        }
        tensorParallelHead = Gemma4TensorParallelHead(rows: try shard.vocabularyRange(full))
    }
}
