//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/25.
//
//  Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen3_5
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

private enum Qwen35VLError: Error {
    case featureTokenMismatch(expected: Int, actual: Int)
}

private let precomputedPositionIdsKey = LMOutput.Key<MLXArray>(
    "qwen35.precomputedPositionIds")
private let ropeDeltasKey = LMOutput.Key<MLXArray>(
    "qwen35.ropeDeltas")

private func supportsExactCompiledDecode(_ dtype: DType) -> Bool {
    dtype == .float16 || dtype == .bfloat16
}

// MARK: - Configuration

public struct Qwen35Configuration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public var modelType: String = ""
        public var hiddenSize: Int = 4096
        public var hiddenLayers: Int = 32
        public var intermediateSize: Int = 14_336
        public var attentionHeads: Int = 32
        public var kvHeads: Int = 8
        public var linearNumValueHeads: Int = 64
        public var linearNumKeyHeads: Int = 16
        public var linearKeyHeadDim: Int = 192
        public var linearValueHeadDim: Int = 128
        public var linearConvKernelDim: Int = 4
        public var rmsNormEps: Float = 1e-6
        public var vocabularySize: Int = 248_320
        public var ropeTheta: Float = 100_000.0
        public var partialRotaryFactor: Float = 0.25
        public var maxPositionEmbeddings: Int = 131_072
        public var tieWordEmbeddings: Bool = false
        public var attentionBias: Bool = false
        public var headDim: Int?
        public var ropeParameters: [String: StringOrNumber]?
        public var fullAttentionInterval: Int = 4
        public var layerTypes: [String]?
        var resolvedLayerTypes: [String] = []
        public var mtpNumHiddenLayers: Int = 0
        public var mtpUseDedicatedEmbeddings: Bool = false

        // MoE fields
        public var numExperts: Int = 0
        public var numExpertsPerTok: Int = 0
        public var decoderSparseStep: Int = 1
        public var sharedExpertIntermediateSize: Int = 0
        public var moeIntermediateSize: Int = 0
        public var normTopkProb: Bool = true

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case linearNumValueHeads = "linear_num_value_heads"
            case linearNumKeyHeads = "linear_num_key_heads"
            case linearKeyHeadDim = "linear_key_head_dim"
            case linearValueHeadDim = "linear_value_head_dim"
            case linearConvKernelDim = "linear_conv_kernel_dim"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
            case maxPositionEmbeddings = "max_position_embeddings"
            case tieWordEmbeddings = "tie_word_embeddings"
            case attentionBias = "attention_bias"
            case headDim = "head_dim"
            case ropeParameters = "rope_parameters"
            case fullAttentionInterval = "full_attention_interval"
            case layerTypes = "layer_types"
            case mtpNumHiddenLayers = "mtp_num_hidden_layers"
            case mtpUseDedicatedEmbeddings = "mtp_use_dedicated_embeddings"
            case numExperts = "num_experts"
            case numExpertsPerTok = "num_experts_per_tok"
            case decoderSparseStep = "decoder_sparse_step"
            case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case normTopkProb = "norm_topk_prob"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)

            self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
            self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
            self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
            self.intermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14_336
            self.attentionHeads =
                try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
            self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
            self.linearNumValueHeads =
                try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
            self.linearNumKeyHeads =
                try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
            self.linearKeyHeadDim =
                try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
            self.linearValueHeadDim =
                try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
            self.linearConvKernelDim =
                try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
            self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            self.vocabularySize =
                try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 248_320
            self.maxPositionEmbeddings =
                try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
            self.tieWordEmbeddings =
                try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
            self.attentionBias =
                try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
            self.fullAttentionInterval =
                try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
            self.layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
            let layerSchedule = try HybridAttentionSchedule(
                hiddenLayerCount: hiddenLayers,
                fullAttentionInterval: fullAttentionInterval,
                explicitLayerTypes: layerTypes
            )
            self.resolvedLayerTypes = layerSchedule.layerTypes
            self.mtpNumHiddenLayers =
                try container.decodeIfPresent(Int.self, forKey: .mtpNumHiddenLayers) ?? 0
            self.mtpUseDedicatedEmbeddings =
                try container.decodeIfPresent(Bool.self, forKey: .mtpUseDedicatedEmbeddings)
                ?? false

            self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
            self.numExpertsPerTok =
                try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
            self.decoderSparseStep =
                try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
            self.sharedExpertIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
            self.moeIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
            self.normTopkProb =
                try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true

            let defaultRopeParameters: [String: StringOrNumber] = [
                "type": .string("default"),
                "mrope_section": .ints([11, 11, 10]),
                "rope_theta": .float(100_000.0),
                "partial_rotary_factor": .float(0.25),
            ]

            var decodedRope = try container.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeParameters)

            if decodedRope == nil {
                let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                let partial = try container.decodeIfPresent(
                    Float.self, forKey: .partialRotaryFactor)
                if ropeTheta != nil || partial != nil {
                    decodedRope = defaultRopeParameters
                    if let ropeTheta {
                        decodedRope?["rope_theta"] = .float(ropeTheta)
                    }
                    if let partial {
                        decodedRope?["partial_rotary_factor"] = .float(partial)
                    }
                }
            }

            if var decodedRope {
                if decodedRope["type"] == nil, let ropeType = decodedRope["rope_type"] {
                    decodedRope["type"] = ropeType
                }
                self.ropeParameters = decodedRope
                self.ropeTheta = decodedRope["rope_theta"]?.asFloat() ?? 100_000.0
                self.partialRotaryFactor = decodedRope["partial_rotary_factor"]?.asFloat() ?? 0.25
            } else {
                self.ropeParameters = defaultRopeParameters
                self.ropeTheta = 100_000.0
                self.partialRotaryFactor = 0.25
            }

            if self.headDim == nil {
                self.headDim = self.hiddenSize / self.attentionHeads
            }
        }
    }

    public typealias VisionConfiguration = Qwen3VLConfiguration.VisionConfiguration

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _ignoreIndex: Int?
    public var ignoreIndex: Int { _ignoreIndex ?? -100 }
    private let _imageTokenId: Int?
    public var imageTokenId: Int { _imageTokenId ?? 248_056 }
    private let _videoTokenId: Int?
    public var videoTokenId: Int { _videoTokenId ?? 248_057 }
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int { _imageTokenIndex ?? imageTokenId }
    private let _videoTokenIndex: Int?
    public var videoTokenIndex: Int { _videoTokenIndex ?? videoTokenId }
    private let _visionStartTokenId: Int?
    public var visionStartTokenId: Int { _visionStartTokenId ?? 248_045 }
    private let _visionEndTokenId: Int?
    public var visionEndTokenId: Int { _visionEndTokenId ?? 248_046 }
    private let _vocabSize: Int?
    public var vocabSize: Int { _vocabSize ?? textConfiguration.vocabularySize }
    private let _eosTokenId: IntOrIntArray?
    public var eosTokenId: [Int]? { _eosTokenId?.values }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _ignoreIndex = "ignore_index"
        case _imageTokenId = "image_token_id"
        case _videoTokenId = "video_token_id"
        case _imageTokenIndex = "image_token_index"
        case _videoTokenIndex = "video_token_index"
        case _visionStartTokenId = "vision_start_token_id"
        case _visionEndTokenId = "vision_end_token_id"
        case _vocabSize = "vocab_size"
        case _eosTokenId = "eos_token_id"
    }
}

// MARK: - Public Serving Seams

/// Attention semantics for Qwen3.5 visual placeholder tokens in the language
/// model. Unlike Gemma, Qwen keeps these tokens on the ordinary causal path.
public enum Qwen35VisionTokenAttention: Sendable, Equatable {
    case causal
}

/// Public configuration facts needed to locate visual spans and reproduce
/// Qwen3.5 M-RoPE positions outside the legacy `prepare` path.
public struct Qwen35VisionSeamConfiguration: Sendable, Equatable {
    /// Token ids whose text embeddings are replaced by vision features.
    public let imagePlaceholderTokenId: Int
    public let videoPlaceholderTokenId: Int

    /// Token ids consumed by Qwen's M-RoPE position calculation. These are
    /// normally identical to the placeholder ids, but both config fields are
    /// exposed because checkpoints may encode them separately.
    public let imagePositionTokenId: Int
    public let videoPositionTokenId: Int
    public let visionStartTokenId: Int
    public let visionEndTokenId: Int

    public let spatialMergeSize: Int
    public let temporalPatchSize: Int
    public let attention: Qwen35VisionTokenAttention
}

/// One text-space slice of Qwen3.5 vision tower output.
public struct Qwen35VisionFeature: @unchecked Sendable {
    public enum Kind: Sendable, Equatable {
        /// One processor image, in image packing order.
        case image(index: Int)
        /// One temporal vision frame within one processor video grid.
        ///
        /// A temporal vision frame represents `temporalPatchSize` sampled
        /// source frames (the processor pads the last group when necessary).
        case videoFrame(videoIndex: Int, frameIndex: Int)
    }

    public let kind: Kind

    /// `[1, visualTokens, textHidden]`, in the language token-embedding dtype.
    public let features: MLXArray

    init(kind: Kind, features: MLXArray) {
        self.kind = kind
        self.features = features
    }
}

/// Exact final text-space vision features used by `Qwen35.prepare`.
///
/// `ordered` contains all images first, followed by videos in video order and
/// each video's temporal frames in time order. This is the same order in which
/// `prepare` packs image pixels followed by video pixels. Concatenating the
/// entries along their token axis yields `flattenedFeatures` exactly.
///
/// `@unchecked Sendable` is limited to immutable, request-owned `MLXArray`
/// handles. Callers crossing isolation must `eval` the arrays first and must
/// not mutate them after transfer.
public struct Qwen35VisionFeatures: @unchecked Sendable {
    public let ordered: [Qwen35VisionFeature]

    /// `[totalVisualTokens, textHidden]`, exactly what `prepare` scatters over
    /// image and video placeholder tokens.
    public let flattenedFeatures: MLXArray

    init(ordered: [Qwen35VisionFeature], flattenedFeatures: MLXArray) {
        self.ordered = ordered
        self.flattenedFeatures = flattenedFeatures
    }
}

public enum Qwen35VisionSeamError: Error, Equatable {
    case missingImagePixels
    case missingImageGrids
    case missingVideoPixels
    case missingVideoGrids
    case invalidGrid(kind: String, index: Int)
    case invalidPixelRank(kind: String, actual: Int)
    case pixelCountMismatch(kind: String, expected: Int, actual: Int)
    case featureCountMismatch(expected: Int, actual: Int)
}

/// Request-owned decode position state. It contains no model-global mutable
/// state and can therefore remain attached to one serving request while other
/// requests prefill or decode on the same model.
///
public struct Qwen35PositionState: Sendable, Equatable {
    /// Per-request M-RoPE deltas, one host value per batch row. Host values
    /// make the state safely transferable without sharing an MLX graph or a
    /// mutable model property across requests.
    public let deltas: [Int32]
    public var batchSize: Int { deltas.count }

    public init(deltas: [Int32]) {
        self.deltas = deltas
    }

    /// Build `[3, batch, length]` decode position ids from this request's
    /// delta and its current cache offset.
    public func decodePositionIds(cacheOffset: Int, length: Int = 1) -> MLXArray {
        precondition(cacheOffset >= 0, "cacheOffset must be non-negative")
        precondition(length > 0, "decode position length must be positive")

        var base = MLXArray(0 ..< length).asType(.int32)
        base = broadcast(base[.newAxis, 0...], to: [batchSize, length])
        let offset = MLXArray(cacheOffset).asType(.int32) + MLXArray(deltas)
        base = base + offset[0..., .newAxis]
        return broadcast(base[.newAxis, 0..., 0...], to: [3, batchSize, length])
    }
}

/// Prompt M-RoPE positions and the request-owned state used for subsequent
/// decode positions.
///
/// `@unchecked Sendable` is limited to the immutable prompt `MLXArray` handle.
/// Evaluate it under model-container isolation before transferring the result;
/// `decodeState` itself is ordinary host data with checked `Sendable` safety.
public struct Qwen35PositionResult: @unchecked Sendable {
    /// Full prompt position ids, shape `[3, batch, promptLength]`.
    public let promptPositionIds: MLXArray
    public let decodeState: Qwen35PositionState
    public let promptLength: Int

    public init(
        promptPositionIds: MLXArray,
        decodeState: Qwen35PositionState,
        promptLength: Int
    ) {
        self.promptPositionIds = promptPositionIds
        self.decodeState = decodeState
        self.promptLength = promptLength
    }

    /// Slice prompt positions for chunked prefill without recomputing or
    /// mutating request state.
    public func promptPositionIds(in range: Range<Int>) -> MLXArray {
        precondition(
            range.lowerBound >= 0 && range.upperBound <= promptLength,
            "prompt position range must be within 0..<promptLength")
        return promptPositionIds[0..., 0..., range]
    }
}

public enum Qwen35PositionSeamError: Error, Equatable {
    case invalidInputRank(Int)
    case invalidAttentionMaskRank(Int)
    case attentionMaskShapeMismatch
    case multimodalBatchUnsupported(Int)
    case invalidGrid(kind: String, index: Int)
    case visualTokenRunMismatch(kind: String, gridIndex: Int, expected: Int, actual: Int)
}

// MARK: - Language

public enum Qwen35Language {

    final class RotaryEmbedding {
        private let invFreq: MLXArray
        private let mropeIndices: MLXArray

        init(dim: Int, base: Float, mropeSection: [Int]) {
            let safeDim = max(1, dim)
            var freq = MLXArray(stride(from: 0, to: safeDim, by: 2)).asType(.float32)
            freq = freq / Float(safeDim)
            self.invFreq = 1.0 / pow(MLXArray(base), freq)

            // Precompute which of the three position planes (t/h/w) owns each
            // frequency so the interleave is a single takeAlong instead of a
            // per-frequency slice loop building ~dims lazy ops per forward.
            let sections = mropeSection.count >= 3 ? mropeSection : [11, 11, 10]
            var indices = [Int32](repeating: 0, count: freq.dim(0))
            for (dimension, offset) in [(1, 1), (2, 2)] {
                let end = min(sections[dimension] * 3, indices.count)
                for index in stride(from: offset, to: end, by: 3) {
                    indices[index] = Int32(dimension)
                }
            }
            self.mropeIndices = MLXArray(indices).reshaped(1, 1, 1, -1)
        }

        private func applyInterleavedMRope(_ freqs: MLXArray) -> MLXArray {
            takeAlong(freqs, mropeIndices, axis: 0).squeezed(axis: 0)
        }

        func callAsFunction(x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
            var positionIds = positionIds
            if positionIds.ndim == 2 {
                positionIds = broadcast(
                    positionIds[.newAxis, 0..., 0...],
                    to: [3, positionIds.dim(0), positionIds.dim(1)])
            }

            let pos = positionIds.asType(.float32)
            var inv = invFreq.asType(.float32)
            inv = inv[.newAxis, .newAxis, .newAxis, 0...]
            var freqs = pos[0..., 0..., 0..., .newAxis] * inv
            freqs = applyInterleavedMRope(freqs)

            let emb = concatenated([freqs, freqs], axis: -1)
            return (cos(emb).asType(x.dtype), sin(emb).asType(x.dtype))
        }
    }

    static func applyMultimodalRotaryPosEmb(
        q: MLXArray,
        k: MLXArray,
        cos: MLXArray,
        sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        let cos = expandedDimensions(cos, axis: 1)
        let sin = expandedDimensions(sin, axis: 1)

        let rotaryDim = cos.dim(-1)
        let qDim = q.dim(-1)
        let kDim = k.dim(-1)

        let qRot = q[.ellipsis, ..<rotaryDim]
        let kRot = k[.ellipsis, ..<rotaryDim]

        let qEmbedded = (qRot * cos) + (QwenVL.rotateHalf(qRot) * sin)
        let kEmbedded = (kRot * cos) + (QwenVL.rotateHalf(kRot) * sin)

        let qOut: MLXArray
        if rotaryDim < qDim {
            qOut = concatenated([qEmbedded, q[.ellipsis, rotaryDim...]], axis: -1)
        } else {
            qOut = qEmbedded
        }

        let kOut: MLXArray
        if rotaryDim < kDim {
            kOut = concatenated([kEmbedded, k[.ellipsis, rotaryDim...]], axis: -1)
        } else {
            kOut = kEmbedded
        }

        return (qOut, kOut)
    }

    final class RMSNormGated: Module {
        @ParameterInfo(key: "weight") var weight: MLXArray
        let eps: Float

        init(dimensions: Int, eps: Float = 1e-6) {
            self.eps = eps
            _weight.wrappedValue = MLXArray.ones([dimensions])
            super.init()
        }

        func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray? = nil) -> MLXArray {
            var x = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: eps)
            if let gate {
                x = x * silu(gate)
            }
            return x
        }
    }

    final class Attention: Module {
        let numKeyValueHeads: Int
        let numAttentionHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rotaryEmbedding: RotaryEmbedding

        init(_ args: Qwen35Configuration.TextConfiguration) {
            self.numKeyValueHeads = args.kvHeads
            self.numAttentionHeads = args.attentionHeads
            self.headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
            self.scale = pow(Float(headDim), -0.5)

            _qProj.wrappedValue = Linear(
                args.hiddenSize, numAttentionHeads * headDim * 2, bias: args.attentionBias)
            _kProj.wrappedValue = Linear(
                args.hiddenSize, numKeyValueHeads * headDim, bias: args.attentionBias)
            _vProj.wrappedValue = Linear(
                args.hiddenSize, numKeyValueHeads * headDim, bias: args.attentionBias)
            _oProj.wrappedValue = Linear(
                numAttentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

            let mrope = args.ropeParameters?["mrope_section"]?.asInts() ?? [11, 11, 10]
            let rotaryDim = Int(Float(headDim) * args.partialRotaryFactor)
            self.rotaryEmbedding = RotaryEmbedding(
                dim: rotaryDim, base: args.ropeTheta, mropeSection: mrope)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let (queries, gate, keys, values) = projectPreRope(x)
            let attention = attend(
                queries: queries,
                keys: keys,
                values: values,
                mask: mask,
                cache: cache,
                positionIds: positionIds)
            return mergeHeadsAndProject(attention: attention, gate: gate)
        }

        /// Projections up to M-RoPE, which remains outside compiled traces.
        func projectPreRope(
            _ x: MLXArray
        ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
            let B = x.dim(0)
            let L = x.dim(1)

            let qProjOutput = qProj(x)
            let qSplit = qProjOutput.reshaped(B, L, numAttentionHeads, -1).split(parts: 2, axis: -1)
            var queries = qSplit[0]
            let gate = qSplit[1].reshaped(B, L, -1)

            var keys = kProj(x)
            var values = vProj(x)

            queries = qNorm(queries).transposed(0, 2, 1, 3)
            keys = kNorm(keys.reshaped(B, L, numKeyValueHeads, -1)).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)

            return (queries, gate, keys, values)
        }

        /// Dynamic M-RoPE, cache mutation, and SDPA stay eager.
        func attend(
            queries initialQueries: MLXArray,
            keys initialKeys: MLXArray,
            values: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds providedPositionIds: MLXArray?
        ) -> MLXArray {
            let B = initialQueries.dim(0)
            let L = initialQueries.dim(2)
            var queries = initialQueries
            var keys = initialKeys

            var kvSeqLen = keys.dim(-2)
            var positionIds = providedPositionIds

            if positionIds == nil {
                let offset = cache?.offset ?? 0
                kvSeqLen += offset + 1
                var base = MLXArray(stride(from: offset, to: offset + L, by: 1)).asType(.int32)
                base = tiled(base[.newAxis, 0...], repetitions: [B, 1])
                positionIds = base[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds!, repetitions: [3, 1, 1])
            } else if let cache {
                kvSeqLen += cache.offset + 1
            }

            let (cosValues, sinValues) = rotaryEmbedding(x: values, positionIds: positionIds!)
            (queries, keys) = applyMultimodalRotaryPosEmb(
                q: queries, k: keys, cos: cosValues, sin: sinValues)

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                attentionMask = .array(mask[.ellipsis, 0 ..< kvSeqLen])
            } else {
                attentionMask = .none
            }

            return attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: attentionMask
            )
        }

        func mergeHeadsAndProject(attention: MLXArray, gate: MLXArray) -> MLXArray {
            let output =
                attention
                .transposed(0, 2, 1, 3)
                .reshaped(attention.dim(0), attention.dim(2), -1)
            return oProj(output * sigmoid(gate))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    open class GatedDeltaNet: Module {
        let hiddenSize: Int
        let numVHeads: Int
        let numKHeads: Int
        let headKDim: Int
        let headVDim: Int
        let keyDim: Int
        let valueDim: Int
        let convKernelSize: Int
        let convDim: Int

        @ModuleInfo(key: "conv1d") var conv1d: Conv1d
        @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
        @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
        @ModuleInfo(key: "in_proj_b") var inProjB: Linear
        @ModuleInfo(key: "in_proj_a") var inProjA: Linear

        // Inference-only physical projection. The four registered modules
        // remain as views so checkpoint and adapter paths stay unchanged.
        private let fusedInputProjection = FusedQuantizedLinearProjectionCache()
        var fusedInputProjectionEnabled = qwen35FourGDNEnabled

        private struct ProjectionTrace {
            weak var value: (any CompiledTraceInvalidating)?
        }
        // Submodule updates must also drop traces owned by parent decoders.
        private let fusedProjectionTracesLock = NSLock()
        private var fusedProjectionTraces: [ObjectIdentifier: ProjectionTrace] = [:]

        @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
        @ParameterInfo(key: "A_log") var aLog: MLXArray

        @ModuleInfo(key: "norm") var norm: RMSNormGated
        @ModuleInfo(key: "out_proj") var outProj: Linear

        public init(_ args: Qwen35Configuration.TextConfiguration) {
            self.hiddenSize = args.hiddenSize
            self.numVHeads = args.linearNumValueHeads
            self.numKHeads = args.linearNumKeyHeads
            self.headKDim = args.linearKeyHeadDim
            self.headVDim = args.linearValueHeadDim
            self.keyDim = headKDim * numKHeads
            self.valueDim = headVDim * numVHeads
            self.convKernelSize = args.linearConvKernelDim
            self.convDim = keyDim * 2 + valueDim

            precondition(
                numVHeads % numKHeads == 0,
                "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
            )

            _conv1d.wrappedValue = Conv1d(
                inputChannels: convDim,
                outputChannels: convDim,
                kernelSize: convKernelSize,
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: convDim,
                bias: false
            )

            _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
            _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
            _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
            _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

            _dtBias.wrappedValue = MLXArray.ones([numVHeads])
            let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
            _aLog.wrappedValue = log(a)

            _norm.wrappedValue = RMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
            _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)
            super.init()
        }

        @discardableResult
        open override func update(
            parameters: ModuleParameters, verify: VerifyUpdate,
            path: [String] = [], modulePath: [String] = []
        ) throws -> Self {
            let inputProjectionPrefixes = [
                "in_proj_qkv.", "in_proj_z.", "in_proj_b.", "in_proj_a.",
            ]
            let replacesInputProjection = parameters.flattened().contains { key, _ in
                inputProjectionPrefixes.contains(where: key.hasPrefix)
            }
            defer {
                if replacesInputProjection {
                    invalidateFusedInputProjection()
                }
            }
            return try super.update(
                parameters: parameters, verify: verify, path: path, modulePath: modulePath)
        }

        open override func updateModule(key: String, _ value: Any) throws {
            let replacesInputProjection =
                key == "in_proj_qkv" || key == "in_proj_z"
                || key == "in_proj_b" || key == "in_proj_a"
            defer {
                if replacesInputProjection {
                    invalidateFusedInputProjection()
                }
            }
            try super.updateModule(key: key, value)
        }

        var hasFusedInputProjection: Bool { fusedInputProjection.isPrepared }

        func registerFusedProjectionTrace(_ trace: any CompiledTraceInvalidating) {
            fusedProjectionTracesLock.withLock {
                fusedProjectionTraces[ObjectIdentifier(trace)] = ProjectionTrace(value: trace)
            }
        }

        private func invalidateFusedInputProjection() {
            fusedInputProjection.invalidate()
            let traces = fusedProjectionTracesLock.withLock {
                let traces = fusedProjectionTraces.values.compactMap(\.value)
                fusedProjectionTraces.removeAll()
                return traces
            }
            // Trace setup takes its own lock first, so invalidate after unlocking.
            for trace in traces {
                trace.invalidate()
            }
        }

        /// The fused projection as compile state for a trace that runs this
        /// layer. It is not a registered child because the four source
        /// projections retain the checkpoint topology.
        var fusedProjectionTraceState: [Module] {
            fusedInputProjection.fused.map { [$0] } ?? []
        }

        @discardableResult
        func prepareFusedInputProjection() throws -> Bool {
            try fusedInputProjection.prepare(
                enabled: fusedInputProjectionEnabled,
                linears: [
                    inProjQKV, inProjZ, inProjB, inProjA,
                ]
            ) { sourceViews in
                try update(
                    modules: ModuleChildren(values: [
                        "in_proj_qkv": .value(sourceViews[0]),
                        "in_proj_z": .value(sourceViews[1]),
                        "in_proj_b": .value(sourceViews[2]),
                        "in_proj_a": .value(sourceViews[3]),
                    ]), verify: [])
            }
        }

        func projectInputs(_ inputs: MLXArray, batch: Int, sequence: Int) -> (
            qkv: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray
        ) {
            guard fusedInputProjectionEnabled, let fusedInProj = fusedInputProjection.fused else {
                return (
                    inProjQKV(inputs),
                    inProjZ(inputs).reshaped(batch, sequence, numVHeads, headVDim),
                    inProjB(inputs),
                    inProjA(inputs)
                )
            }

            let projected = fusedInProj(inputs)
            let qkvEnd = keyDim * 2 + valueDim
            let zEnd = qkvEnd + valueDim
            let bEnd = zEnd + numVHeads
            let aEnd = bEnd + numVHeads
            return (
                projected[0..., 0..., ..<qkvEnd],
                projected[0..., 0..., qkvEnd ..< zEnd].reshaped(
                    batch, sequence, numVHeads, headVDim),
                projected[0..., 0..., zEnd ..< bEnd],
                projected[0..., 0..., bEnd ..< aEnd]
            )
        }

        open func callAsFunction(
            _ inputs: MLXArray,
            mask: MLXArray? = nil,
            cache: MambaCache? = nil,
            checkpointAfter: Int? = nil
        ) -> MLXArray {
            let convState =
                cache?[0] ?? zeroStates(batch: inputs.dim(0), dtype: inputs.dtype).conv
            let result = forward(
                inputs, convState: convState, recState: cache?[1], mask: mask,
                checkpointAfter: checkpointAfter)
            if let cache {
                if convKernelSize > 1 {
                    cache[0] = result.convState
                }
                cache[1] = result.recurrentState
                if let checkpoint = result.checkpoint, let checkpointAfter {
                    cache.saveSpeculativeCheckpoint(
                        convState: checkpoint.conv,
                        recurrentState: checkpoint.recurrent,
                        advancedBy: checkpointAfter)
                }
                cache.advance(inputs.dim(1))
            }
            return result.output
        }

        func zeroStates(batch: Int, dtype: DType) -> (conv: MLXArray, rec: MLXArray) {
            (
                MLXArray.zeros(
                    [batch, max(0, convKernelSize - 1), convDim], dtype: dtype),
                MLXArray.zeros(
                    [batch, numVHeads, headVDim, headKDim], dtype: .float32)
            )
        }

        /// The GDN body with cache state passed explicitly for compiled decode.
        func forward(
            _ x: MLXArray,
            convState: MLXArray,
            recState: MLXArray?,
            mask: MLXArray?,
            checkpointAfter: Int? = nil
        ) -> (
            output: MLXArray,
            convState: MLXArray,
            recurrentState: MLXArray,
            checkpoint: (conv: MLXArray, recurrent: MLXArray)?
        ) {
            let B = x.dim(0)
            let S = x.dim(1)

            var (mixedQKV, z, b, a) = projectInputs(x, batch: B, sequence: S)
            if let mask {
                mixedQKV = MLX.where(mask[.ellipsis, .newAxis], mixedQKV, 0)
            }

            let convInput = concatenated([convState, mixedQKV], axis: 1)
            let newConvState: MLXArray
            if convKernelSize > 1 {
                newConvState = contiguous(
                    convInput[0..., (-(convKernelSize - 1))..., 0...])
            } else {
                newConvState = MLXArray.zeros([B, 0, convDim], dtype: mixedQKV.dtype)
            }

            let convOut = silu(conv1d(convInput))
            let split = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
            let q = split[0].reshaped(B, S, numKHeads, headKDim)
            let k = split[1].reshaped(B, S, numKHeads, headKDim)
            let v = split[2].reshaped(B, S, numVHeads, headVDim)

            let dtype = q.dtype
            let invScale = pow(Float(headKDim), -0.5)
            // rmsNorm adds eps to mean(x^2). The reference l2norm adds 1e-6 to
            // sum(x^2), so divide the eps by the head dim.
            let qkEps = 1e-6 / Float(headKDim)
            let qNormed =
                MLXArray(pow(invScale, 2)).asType(dtype)
                * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: qkEps)
            let kNormed =
                MLXArray(invScale).asType(dtype)
                * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: qkEps)

            let out: MLXArray
            let newRecState: MLXArray
            let checkpoint: (conv: MLXArray, recurrent: MLXArray)?
            if let split = checkpointAfter, split > 0, split < S {
                let prefixMask = mask.map { $0[0..., ..<split] }
                let suffixMask = mask.map { $0[0..., split...] }
                let (prefixOut, prefixState) = gatedDeltaUpdate(
                    q: qNormed[0..., ..<split, 0..., 0...],
                    k: kNormed[0..., ..<split, 0..., 0...],
                    v: v[0..., ..<split, 0..., 0...],
                    a: a[0..., ..<split, 0...],
                    b: b[0..., ..<split, 0...],
                    aLog: aLog,
                    dtBias: dtBias,
                    state: recState,
                    mask: prefixMask,
                    useKernel: !training)
                let (suffixOut, suffixState) = gatedDeltaUpdate(
                    q: qNormed[0..., split..., 0..., 0...],
                    k: kNormed[0..., split..., 0..., 0...],
                    v: v[0..., split..., 0..., 0...],
                    a: a[0..., split..., 0...],
                    b: b[0..., split..., 0...],
                    aLog: aLog,
                    dtBias: dtBias,
                    state: prefixState,
                    mask: suffixMask,
                    useKernel: !training)
                out = concatenated([prefixOut, suffixOut], axis: 1)
                newRecState = suffixState

                let checkpointConv: MLXArray
                if convKernelSize > 1 {
                    checkpointConv = contiguous(
                        convInput[
                            0..., split ..< (split + convKernelSize - 1), 0...])
                } else {
                    checkpointConv = MLXArray.zeros(
                        [B, 0, convDim], dtype: mixedQKV.dtype)
                }
                checkpoint = (checkpointConv, prefixState)
            } else {
                (out, newRecState) = gatedDeltaUpdate(
                    q: qNormed,
                    k: kNormed,
                    v: v,
                    a: a,
                    b: b,
                    aLog: aLog,
                    dtBias: dtBias,
                    state: recState,
                    mask: mask,
                    useKernel: !training)
                checkpoint = nil
            }

            let gated = norm(out, gate: z)
            return (
                outProj(gated.reshaped(B, S, -1)), newConvState, newRecState, checkpoint
            )
        }
    }

    open class SparseMoeBlock: Module, UnaryLayer {
        let normTopkProb: Bool
        let numExperts: Int
        let topK: Int

        @ModuleInfo(key: "gate") var gate: Linear
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

        @ModuleInfo(key: "shared_expert") var sharedExpert: MLP
        @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

        public init(_ args: Qwen35Configuration.TextConfiguration) {
            self.normTopkProb = args.normTopkProb
            self.numExperts = args.numExperts
            self.topK = args.numExpertsPerTok

            _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
            // Fused gate_up: one gather_qmm serves gate+up. The wrapper's
            // sanitize concatenates split checkpoints into this layout
            // (`qwen35FuseSwitchMLPGateUp`).
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: args.hiddenSize,
                hiddenDims: args.moeIntermediateSize,
                numExperts: args.numExperts,
                fuseGateUp: true
            )

            _sharedExpert.wrappedValue = MLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.sharedExpertIntermediateSize
            )
            _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
            super.init()
        }

        open func callAsFunction(_ x: MLXArray) -> MLXArray {
            guard x.dim(1) == 1, !training, supportsExactCompiledDecode(x.dtype),
                supportsCompiledDecode
            else {
                return forward(x)
            }
            return compiledForward(self, x)
        }

        private let compiledForward = CompiledTrace<SparseMoeBlock> { block, arguments in
            [block.forward(arguments[0])]
        }

        var compiledDecodeIsCompiled: Bool { compiledForward.isCompiled }

        var supportsCompiledDecode: Bool {
            type(of: self) == SparseMoeBlock.self && type(of: switchMLP) == SwitchGLU.self
        }

        /// The eager body, also inlined by an enclosing decoder trace.
        func forward(_ x: MLXArray) -> MLXArray {
            var gates = gate(x)
            gates = MLX.softmax(gates, axis: -1, precise: true)

            let (inds, scores) = moeRouterTopK(
                gates, k: topK, normalize: normTopkProb)

            let y = switchMLP(x, inds)
            let combined = weightedExpertSum(y, scores.asType(y.dtype))

            var sharedY = sharedExpert(x)
            sharedY = sigmoid(sharedExpertGate(x)) * sharedY

            return combined + sharedY
        }
    }

    open class DecoderLayer: Module {
        let isLinear: Bool
        let allowsCompiledDecode: Bool

        @ModuleInfo(key: "self_attn") var selfAttn: Attention?
        @ModuleInfo(key: "linear_attn") var linearAttn: GatedDeltaNet?

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        @ModuleInfo(key: "mlp") var mlp: Module

        public init(
            _ args: Qwen35Configuration.TextConfiguration, layerIdx: Int,
            forceFullAttention: Bool = false
        ) {
            self.allowsCompiledDecode = !forceFullAttention
            self.isLinear =
                forceFullAttention
                ? false
                : args.resolvedLayerTypes[layerIdx] == HybridAttentionSchedule.linearAttention

            if isLinear {
                _linearAttn.wrappedValue = GatedDeltaNet(args)
            } else {
                _selfAttn.wrappedValue = Attention(args)
            }

            if args.numExperts > 0 {
                _mlp.wrappedValue = SparseMoeBlock(args)
            } else {
                _mlp.wrappedValue = MLP(
                    dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            }

            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)

            super.init()
        }

        open func callAsFunction(
            _ x: MLXArray,
            attentionMask: MLXArray?,
            ssmMask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?,
            checkpointAfter: Int? = nil
        ) -> MLXArray {
            forward(
                x,
                attentionMask: attentionMask,
                ssmMask: ssmMask,
                cache: cache,
                positionIds: positionIds,
                checkpointAfter: checkpointAfter,
                allowCompiledDecode: allowsCompiledDecode)
        }

        func forward(
            _ x: MLXArray,
            attentionMask: MLXArray?,
            ssmMask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?,
            checkpointAfter: Int?,
            allowCompiledDecode: Bool
        ) -> MLXArray {
            let canCompile =
                allowCompiledDecode && allowsCompiledDecode && !training
                && supportsExactCompiledDecode(x.dtype)
                && x.dim(0) == 1 && x.dim(1) == 1
                && attentionMask == nil && ssmMask == nil && checkpointAfter == nil
                && supportsCompiledDecode

            if canCompile, isLinear, let mambaCache = cache as? MambaCache,
                mambaCache[0] != nil, mambaCache[1] != nil
            {
                return decodeLinearLayer(x, cache: mambaCache)
            }

            let r: MLXArray
            if isLinear {
                r = linearAttn!(
                    inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache,
                    checkpointAfter: checkpointAfter)
            } else {
                r = selfAttn!(
                    inputLayerNorm(x), mask: attentionMask, cache: cache, positionIds: positionIds)
            }

            let h = x + r
            return h
                + mlpForward(
                    postAttentionLayerNorm(h), allowCompiledDecode: canCompile)
        }

        private let compiledLinearLayer: CompiledTrace<DecoderLayer> = CompiledTrace(
            state: { layer in
                layer.linearAttn?.registerFusedProjectionTrace(layer.compiledLinearLayer)
                return [layer] + (layer.linearAttn?.fusedProjectionTraceState ?? [])
            },
            body: { layer, arguments in
                let result = layer.linearLayerBody(
                    x: arguments[0], convState: arguments[1], recState: arguments[2])
                return [result.output, result.convState, result.recurrentState]
            })

        var compiledGDNIsCompiled: Bool { compiledLinearLayer.isCompiled }

        var supportsCompiledDecode: Bool {
            guard type(of: self) == DecoderLayer.self, !training else { return false }
            if isLinear {
                guard let linearAttn, type(of: linearAttn) == GatedDeltaNet.self,
                    !linearAttn.training
                else {
                    return false
                }
            } else if selfAttn == nil {
                return false
            }

            if let moe = mlp as? SparseMoeBlock {
                return moe.supportsCompiledDecode
            }
            return mlp is MLP
        }

        private func decodeLinearLayer(_ x: MLXArray, cache: MambaCache) -> MLXArray {
            let outputs = compiledLinearLayer(self, [x, cache[0]!, cache[1]!])
            cache[0] = outputs[1]
            cache[1] = outputs[2]
            cache.advance(1)
            return outputs[0]
        }

        func linearLayerBody(
            x: MLXArray, convState: MLXArray, recState: MLXArray
        ) -> (output: MLXArray, convState: MLXArray, recurrentState: MLXArray) {
            let result = linearAttn!.forward(
                inputLayerNorm(x), convState: convState, recState: recState, mask: nil)
            let h = x + result.output
            return (
                h + mlpForward(postAttentionLayerNorm(h), allowCompiledDecode: false),
                result.convState,
                result.recurrentState
            )
        }

        func attentionPreBody(
            x: MLXArray
        ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
            selfAttn!.projectPreRope(inputLayerNorm(x))
        }

        func attentionCacheStep(
            queries: MLXArray,
            keys: MLXArray,
            values: MLXArray,
            cache: KVCache,
            positionIds: MLXArray?
        ) -> MLXArray {
            selfAttn!.attend(
                queries: queries,
                keys: keys,
                values: values,
                mask: nil,
                cache: cache,
                positionIds: positionIds)
        }

        func attentionPostBody(
            x: MLXArray, attention: MLXArray, gate: MLXArray
        ) -> MLXArray {
            let r = selfAttn!.mergeHeadsAndProject(attention: attention, gate: gate)
            let h = x + r
            return h + mlpForward(postAttentionLayerNorm(h), allowCompiledDecode: false)
        }

        private func mlpForward(_ x: MLXArray, allowCompiledDecode: Bool) -> MLXArray {
            if let moe = mlp as? SparseMoeBlock, type(of: moe) == SparseMoeBlock.self {
                return allowCompiledDecode ? moe(x) : moe.forward(x)
            }
            return (mlp as! UnaryLayer)(x)
        }
    }

    open class Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") fileprivate var layers: [DecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        let ssmIdx: Int?
        let faIdx: Int?

        private let decodeSegments: [CompiledDecodeSegment]
        private let compiledSegments: CompiledDecodeSegmentCache<Model>

        public init(_ args: Qwen35Configuration.TextConfiguration) {
            precondition(args.vocabularySize > 0)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
            let layers = (0 ..< args.hiddenLayers).map {
                DecoderLayer(args, layerIdx: $0)
            }
            _layers.wrappedValue = layers
            _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

            self.ssmIdx = layers.firstIndex(where: { $0.isLinear })
            self.faIdx = layers.firstIndex(where: { !$0.isLinear })

            let segments = CompiledDecodeSegment.schedule(
                linearLayers: layers.map(\.isLinear))
            self.decodeSegments = segments
            self.compiledSegments = CompiledDecodeSegmentCache(
                count: segments.count,
                state: { model, index in
                    var modules = model.traceState(forLayers: segments[index].layerIndices)
                    if index == segments.count - 1 {
                        modules.append(model.norm)
                    }
                    return modules
                },
                body: { model, index, arguments in
                    model.segmentBody(at: index, arguments)
                })
            super.init()
        }

        open func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            positionIds: MLXArray? = nil,
            applyFinalNorm: Bool = true,
            checkpointAfter: Int? = nil
        ) -> MLXArray {
            forward(
                inputs,
                inputsEmbeds: inputsEmbeds,
                cache: cache,
                positionIds: positionIds,
                applyFinalNorm: applyFinalNorm,
                checkpointAfter: checkpointAfter,
                useCompiledDecode: true)
        }

        func forward(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            positionIds: MLXArray? = nil,
            applyFinalNorm: Bool = true,
            checkpointAfter: Int? = nil,
            useCompiledDecode: Bool = true
        ) -> MLXArray {
            if useCompiledDecode, type(of: self) == Model.self, !training,
                inputsEmbeds == nil, applyFinalNorm, checkpointAfter == nil,
                inputs.dim(0) == 1, inputs.dim(1) == 1, let cache,
                let step = decodeStep(inputs, cache, positionIds: positionIds)
            {
                return step
            }

            var hiddenStates: MLXArray
            if let inputsEmbeds {
                hiddenStates = inputsEmbeds
            } else {
                hiddenStates = embedTokens(inputs)
            }

            var cacheArray = cache
            if cacheArray == nil {
                cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
            }

            let faMask: MLXArray?
            if let faIdx {
                let faMaskMode = createAttentionMask(
                    h: hiddenStates, cache: cacheArray?[faIdx], returnArray: true)
                if case .array(let arrayMask) = faMaskMode {
                    faMask = arrayMask
                } else {
                    faMask = nil
                }
            } else {
                faMask = nil
            }
            let ssmMask = ssmIdx.flatMap {
                createSSMMask(h: hiddenStates, cache: cacheArray?[$0] as? MambaCache)
            }

            let allowLayerCompilation =
                useCompiledDecode && type(of: self) == Model.self && !training
                && inputsEmbeds == nil && applyFinalNorm
                && supportsExactCompiledDecode(hiddenStates.dtype)
                && hiddenStates.dim(0) == 1 && hiddenStates.dim(1) == 1
                && faMask == nil && ssmMask == nil && checkpointAfter == nil
                && layers.enumerated().allSatisfy { index, layer in
                    guard layer.supportsCompiledDecode else { return false }
                    if layer.isLinear {
                        guard let mambaCache = cacheArray?[index] as? MambaCache else {
                            return false
                        }
                        return mambaCache[0] != nil && mambaCache[1] != nil
                    }
                    guard let kvCache = cacheArray?[index] else { return false }
                    return kvCache is QuantizedKVCacheProtocol
                        || kvCache is TurboQuantKVCache
                }

            for (index, layer) in layers.enumerated() {
                let layerSSMMask = layer.isLinear ? ssmMask : nil
                if type(of: layer) == DecoderLayer.self {
                    hiddenStates = layer.forward(
                        hiddenStates,
                        attentionMask: faMask,
                        ssmMask: layerSSMMask,
                        cache: cacheArray?[index],
                        positionIds: positionIds,
                        checkpointAfter: checkpointAfter,
                        allowCompiledDecode: allowLayerCompilation)
                } else {
                    hiddenStates = layer(
                        hiddenStates,
                        attentionMask: faMask,
                        ssmMask: layerSSMMask,
                        cache: cacheArray?[index],
                        positionIds: positionIds,
                        checkpointAfter: checkpointAfter)
                }
            }

            return applyFinalNorm ? norm(hiddenStates) : hiddenStates
        }

        var compiledDecodeSegmentCount: Int { compiledSegments.compiledCount }

        var compiledGDNTraceCount: Int {
            layers.filter(\.compiledGDNIsCompiled).count
        }

        var compiledMoETraceCount: Int {
            layers.compactMap { $0.mlp as? SparseMoeBlock }
                .filter(\.compiledDecodeIsCompiled).count
        }

        private func traceState(forLayers indices: [Int]) -> [Module] {
            indices.flatMap { index in
                let layer = layers[index]
                layer.linearAttn?.registerFusedProjectionTrace(compiledSegments)
                return [layer] + (layer.linearAttn?.fusedProjectionTraceState ?? [])
            }
        }

        private func segmentBody(at index: Int, _ args: [MLXArray]) -> [MLXArray] {
            let segment = decodeSegments[index]
            var hiddenStates = args[0]

            if let post = segment.attentionPostLayer {
                hiddenStates = layers[post].attentionPostBody(
                    x: hiddenStates, attention: args[1], gate: args[2])
            }

            var states: [MLXArray] = []
            for (stateIndex, layerIndex) in segment.linearLayers.enumerated() {
                let slot = segment.stateInputOffset + 2 * stateIndex
                let result = layers[layerIndex].linearLayerBody(
                    x: hiddenStates, convState: args[slot], recState: args[slot + 1])
                hiddenStates = result.output
                states.append(result.convState)
                states.append(result.recurrentState)
            }

            if let pre = segment.attentionPreLayer {
                let (queries, gate, keys, values) = layers[pre].attentionPreBody(
                    x: hiddenStates)
                return [hiddenStates] + states + [queries, gate, keys, values]
            }

            if index == decodeSegments.count - 1 {
                hiddenStates = norm(hiddenStates)
            }
            return [hiddenStates] + states
        }

        private func decodeStep(
            _ inputs: MLXArray, _ cache: [KVCache?], positionIds: MLXArray?
        ) -> MLXArray? {
            let embedded = embedTokens(inputs)
            guard supportsExactCompiledDecode(embedded.dtype) else { return nil }
            guard cache.count == layers.count else { return nil }
            if let ssmIdx,
                createSSMMask(h: embedded, cache: cache[ssmIdx] as? MambaCache) != nil
            {
                return nil
            }
            if let faIdx {
                guard let faCache = cache[faIdx],
                    case .none = createAttentionMask(
                        h: embedded, cache: faCache, returnArray: true)
                else { return nil }
            }

            var mambaCaches = [MambaCache?](repeating: nil, count: layers.count)
            for (index, layer) in layers.enumerated() {
                guard layer.supportsCompiledDecode else { return nil }
                if layer.isLinear {
                    guard let mambaCache = cache[index] as? MambaCache,
                        mambaCache[0] != nil, mambaCache[1] != nil
                    else { return nil }
                    mambaCaches[index] = mambaCache
                } else {
                    guard let kvCache = cache[index], usesPlainAttentionCacheRoute(kvCache) else {
                        return nil
                    }
                }
            }

            var carry = embedded
            var pendingAttention: [MLXArray] = []

            for (segmentIndex, segment) in decodeSegments.enumerated() {
                var args = [carry] + pendingAttention
                for layerIndex in segment.linearLayers {
                    let mambaCache = mambaCaches[layerIndex]!
                    args.append(mambaCache[0]!)
                    args.append(mambaCache[1]!)
                }

                let outputs = compiledSegments(self, at: segmentIndex, args)
                carry = outputs[0]

                for (stateIndex, layerIndex) in segment.linearLayers.enumerated() {
                    let mambaCache = mambaCaches[layerIndex]!
                    mambaCache[0] = outputs[1 + 2 * stateIndex]
                    mambaCache[1] = outputs[2 + 2 * stateIndex]
                    mambaCache.advance(1)
                }

                pendingAttention = []
                if let pre = segment.attentionPreLayer {
                    let head = segment.attentionOutputOffset
                    let attention = layers[pre].attentionCacheStep(
                        queries: outputs[head],
                        keys: outputs[head + 2],
                        values: outputs[head + 3],
                        cache: cache[pre]!,
                        positionIds: positionIds)
                    pendingAttention = [attention, outputs[head + 1]]
                }
            }

            return carry
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        let config: Qwen35Configuration
        let textConfig: Qwen35Configuration.TextConfiguration
        let modelType: String
        let kvHeads: [Int]

        init(_ config: Qwen35Configuration) {
            self.config = config
            self.textConfig = config.textConfiguration
            self.modelType = config.textConfiguration.modelType
            self.model = Model(config.textConfiguration)
            self.kvHeads = Array(
                repeating: config.textConfiguration.kvHeads,
                count: config.textConfiguration.hiddenLayers
            )

            if !config.textConfiguration.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.textConfiguration.hiddenSize,
                    config.textConfiguration.vocabularySize,
                    bias: false)
            }
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            state: LMOutput.State?,
            mask: MLXArray? = nil,
            positionIds providedPositionIds: MLXArray? = nil,
            pixelValues: MLXArray? = nil,
            imageGridTHW: [THW]? = nil,
            videoGridTHW: [THW]? = nil,
            lastTokenOnly: Bool = false
        ) -> LMOutput {
            var state = state ?? .init()

            // Ensure inputs is 2D [batch, seq]. Text-only callers (e.g.
            // WiredMemoryUtils, TokenIterator) may pass 1D token arrays.
            let inputs = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs

            if pixelValues != nil {
                state[precomputedPositionIdsKey] = nil
                state[ropeDeltasKey] = nil
            }
            let precomputedPositionIds = state[precomputedPositionIdsKey]
            let ropeDeltas = state[ropeDeltasKey]

            let cacheAnchorIndex = model.faIdx ?? model.ssmIdx
            var cacheOffset = 0
            var hasCacheAnchor = false
            if let cache, let cacheAnchorIndex, cache.indices.contains(cacheAnchorIndex),
                let cacheAnchor = cache[cacheAnchorIndex]
            {
                cacheOffset = cacheAnchor.offset
                hasCacheAnchor = true
            }

            var ropeMask = mask
            if let mask, mask.dim(-1) != inputs.dim(-1) {
                ropeMask = nil
            }

            var positionIds = providedPositionIds
            if positionIds == nil && (ropeMask == nil || ropeMask?.ndim == 2) {
                // A stateless text continuation (no deltas in `state`, a warm
                // cache, no media) takes the offset branch below. Text rows
                // have zero rope deltas, so `cacheOffset + arange` is exact.
                // Recomputing here would restart positions at zero.
                let statelessTextContinuation =
                    ropeDeltas == nil && hasCacheAnchor && cacheOffset > 0
                    && imageGridTHW == nil && videoGridTHW == nil
                if (hasCacheAnchor && cacheOffset == 0)
                    || (ropeDeltas == nil && !statelessTextContinuation)
                    || cache == nil
                {
                    if let precomputedPositionIds {
                        let seqLength = inputs.dim(1)
                        positionIds =
                            precomputedPositionIds[
                                0..., 0..., cacheOffset ..< (cacheOffset + seqLength)]
                    } else {
                        let (computed, deltas) = Qwen3VLLanguage.getRopeIndex(
                            inputIds: inputs,
                            imageGridTHW: imageGridTHW,
                            videoGridTHW: videoGridTHW,
                            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                            imageTokenId: config.imageTokenId,
                            videoTokenId: config.videoTokenId,
                            visionStartTokenId: config.visionStartTokenId,
                            attentionMask: ropeMask)
                        positionIds = computed
                        state[precomputedPositionIdsKey] = computed
                        state[ropeDeltasKey] = deltas
                    }
                } else {
                    let batchSize = inputs.dim(0)
                    let seqLength = inputs.dim(1)

                    // Apply `ropeDeltas` only when it is a 1-D [batchSize]
                    // array. Under continuous batching a caller can hand this
                    // step a state from a prefill of a different batch. Those
                    // deltas belong to other rows. Repeating them gave
                    // [broadcast_shapes] (B,1) vs (prefillBatch*B,1) crashes
                    // (d-inference issue 513). Batched text rows have zero
                    // deltas, so skipping a mismatched array is exact for them.
                    var delta = MLXArray(cacheOffset).asType(.int32)
                    if let ropeDeltas, ropeDeltas.ndim == 1, ropeDeltas.dim(0) == batchSize {
                        delta = delta + ropeDeltas.asType(.int32)
                    }

                    var base = MLXArray(0 ..< seqLength).asType(.int32)
                    base = broadcast(base[.newAxis, 0...], to: [batchSize, seqLength])

                    if delta.ndim == 0 {
                        delta = broadcast(delta, to: [batchSize])
                    }

                    // Invariant, total by construction: delta is the scalar
                    // cacheOffset broadcast to [batchSize], optionally plus a
                    // ropeDeltas that the guard above proved is [batchSize].
                    // (Debug-only; compiles out of release builds.)
                    assert(
                        delta.ndim == 1 && delta.dim(0) == batchSize,
                        "position-id delta must be [batchSize] before broadcast")

                    base = base + delta[0..., .newAxis]
                    positionIds = broadcast(
                        base[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])
                }
            }

            let emitDrafterState = state[mtpEmitFlagKey] ?? false
            let preNormHidden = model(
                inputs,
                inputsEmbeds: inputsEmbeds,
                cache: cache,
                positionIds: positionIds,
                applyFinalNorm: !emitDrafterState,
                checkpointAfter: state[mtpCacheCheckpointIndexKey]
            )
            let hiddenStates = emitDrafterState ? model.norm(preNormHidden) : preNormHidden

            let projection: Module = lmHead ?? model.embedTokens
            var out =
                lastTokenOnly && !emitDrafterState
                ? quantizedVocabularyProjectionInput(hiddenStates, projection: projection)
                : hiddenStates
            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }

            if emitDrafterState {
                state[mtpLastHiddenStatesKey] = hiddenStates
                state[mtpSharedKVStatesKey] = qwen35VLMSharedKVState(
                    cache: cache, fullAttentionIndex: model.faIdx)
                state[mtpSharedKVOffsetsKey] = qwen35VLMSharedKVOffsets(
                    cache: cache, fullAttentionIndex: model.faIdx)
                state[mtpSharedKVSourceIndicesKey] =
                    model.faIdx.map {
                        ["full_attention": $0]
                    } ?? [:]
                state[mtpPositionDeltasKey] = state[ropeDeltasKey]
            }

            return LMOutput(logits: out, state: state)
        }

        func makeCache(parameters: GenerateParameters?) throws -> [KVCache] {
            try model.layers.map { layer in
                if layer.isLinear {
                    return MambaCache()
                }
                return try makeAttentionKVCache(parameters: parameters)
            }
        }

        func prepare() throws {
            for layer in model.layers {
                if let linearAttn = layer.linearAttn {
                    _ = try linearAttn.prepareFusedInputProjection()
                }
            }
        }
    }
}

private func qwen35VLMSharedKVState(
    cache: [KVCache?]?,
    fullAttentionIndex: Int?
) -> [String: (MLXArray, MLXArray)] {
    guard let cache, let fullAttentionIndex,
        fullAttentionIndex < cache.count,
        let faCache = cache[fullAttentionIndex]
    else {
        return [:]
    }
    let state = faCache.state
    guard state.count == 2 else {
        return [:]
    }
    return ["full_attention": (state[0], state[1])]
}

private func qwen35VLMSharedKVOffsets(
    cache: [KVCache?]?,
    fullAttentionIndex: Int?
) -> [String: Int]? {
    guard let cache, let fullAttentionIndex,
        fullAttentionIndex < cache.count,
        let faCache = cache[fullAttentionIndex]
    else {
        return nil
    }
    return ["full_attention": faCache.offset]
}

// MARK: - Model

/// The vision tower at the path it has in ``Qwen35`` when it loads separately.
private final class Qwen35Vision: Module {
    @ModuleInfo(key: "vision_tower") var model: Qwen3VLVision.VisionModel

    init(_ config: Qwen35Configuration) {
        _model.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfiguration)
        super.init()
    }
}

public class Qwen35: Module, VLMModel {
    @ModuleInfo(key: "vision_tower") private var visionModel: Qwen3VLVision.VisionModel?
    @ModuleInfo(key: "language_model") var languageModel: Qwen35Language.LanguageModel

    public let config: Qwen35Configuration
    private let onDemandVision: OnDemandComponent<Qwen35Vision>

    /// Checkpoint quantization policy staged by `loadWeights` (via
    /// `QuantizationPolicyReceiving`) before `sanitize` runs; drives the
    /// per-layer decision whether routed-expert gate/up halves may fuse
    /// (`Qwen35MoE.sanitize`). `nil` for unquantized checkpoints.
    public var checkpointPerLayerQuantization: BaseConfiguration.PerLayerQuantization?

    public init(_ config: Qwen35Configuration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Qwen35Language.LanguageModel(config)
        onDemandVision = OnDemandComponent(.vision) { Qwen35Vision(config) }
        super.init()
    }

    public var vocabularySize: Int { config.vocabSize }

    public var visionSeamConfiguration: Qwen35VisionSeamConfiguration {
        Qwen35VisionSeamConfiguration(
            imagePlaceholderTokenId: config.imageTokenIndex,
            videoPlaceholderTokenId: config.videoTokenIndex,
            imagePositionTokenId: config.imageTokenId,
            videoPositionTokenId: config.videoTokenId,
            visionStartTokenId: config.visionStartTokenId,
            visionEndTokenId: config.visionEndTokenId,
            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
            temporalPatchSize: config.visionConfiguration.temporalPatchSize,
            attention: .causal)
    }

    public var imagePlaceholderTokenId: Int { config.imageTokenIndex }
    public var videoPlaceholderTokenId: Int { config.videoTokenIndex }

    /// Compute all prompt and decode positions without reading or writing the
    /// legacy language module's `ropeDeltas` property.
    public func positionResult(
        tokens: MLXArray,
        imageGrids: [THW]? = nil,
        videoGrids: [THW]? = nil,
        attentionMask: MLXArray? = nil
    ) throws -> Qwen35PositionResult {
        guard tokens.ndim == 1 || tokens.ndim == 2 else {
            throw Qwen35PositionSeamError.invalidInputRank(tokens.ndim)
        }
        let tokens = tokens.ndim == 1 ? tokens.expandedDimensions(axis: 0) : tokens

        var mask = attentionMask
        if let attentionMask {
            guard attentionMask.ndim == 1 || attentionMask.ndim == 2 else {
                throw Qwen35PositionSeamError.invalidAttentionMaskRank(attentionMask.ndim)
            }
            mask =
                attentionMask.ndim == 1
                ? attentionMask.expandedDimensions(axis: 0) : attentionMask
            guard mask?.shape == tokens.shape else {
                throw Qwen35PositionSeamError.attentionMaskShapeMismatch
            }
        }

        let imageGrids = imageGrids?.nilIfEmpty
        let videoGrids = videoGrids?.nilIfEmpty
        if imageGrids != nil || videoGrids != nil, tokens.dim(0) != 1 {
            throw Qwen35PositionSeamError.multimodalBatchUnsupported(tokens.dim(0))
        }

        let merge = config.visionConfiguration.spatialMergeSize
        for (kind, grids) in [("image", imageGrids ?? []), ("video", videoGrids ?? [])] {
            for (index, grid) in grids.enumerated()
            where grid.t <= 0 || grid.h <= 0 || grid.w <= 0
                || grid.h % merge != 0 || grid.w % merge != 0
            {
                throw Qwen35PositionSeamError.invalidGrid(kind: kind, index: index)
            }
        }
        try validateVisualTokenRuns(
            tokens: tokens, attentionMask: mask,
            imageGrids: imageGrids ?? [], videoGrids: videoGrids ?? [], merge: merge)

        let (positionIds, delta) = Qwen3VLLanguage.getRopeIndex(
            inputIds: tokens,
            imageGridTHW: imageGrids,
            videoGridTHW: videoGrids,
            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
            imageTokenId: config.imageTokenId,
            videoTokenId: config.videoTokenId,
            visionStartTokenId: config.visionStartTokenId,
            attentionMask: mask)

        return Qwen35PositionResult(
            promptPositionIds: positionIds,
            decodeState: Qwen35PositionState(deltas: delta.asArray(Int32.self)),
            promptLength: tokens.dim(1))
    }

    private func validateVisualTokenRuns(
        tokens: MLXArray,
        attentionMask: MLXArray?,
        imageGrids: [THW],
        videoGrids: [THW],
        merge: Int
    ) throws {
        guard tokens.dim(0) == 1 else { return }
        let values = tokens[0, 0...].asArray(Int32.self).map(Int.init)
        let maskValues = attentionMask?.asType(.int32)[0, 0...].asArray(Int32.self)
        var imageIndex = 0
        var videoIndex = 0
        var cursor = 0

        while cursor < values.count {
            let visible = maskValues.map { $0[cursor] == 1 } ?? true
            let token = values[cursor]
            guard visible, token == config.imageTokenId || token == config.videoTokenId else {
                cursor += 1
                continue
            }

            let kind = token == config.imageTokenId ? "image" : "video"
            let gridIndex = kind == "image" ? imageIndex : videoIndex
            let grids = kind == "image" ? imageGrids : videoGrids
            var end = cursor + 1
            while end < values.count,
                maskValues.map { $0[end] == 1 } ?? true, values[end] == token
            {
                end += 1
            }
            let actual = end - cursor
            guard gridIndex < grids.count else {
                throw Qwen35PositionSeamError.visualTokenRunMismatch(
                    kind: kind, gridIndex: gridIndex, expected: 0, actual: actual)
            }
            let expected = try mergedTokenCount(
                grids[gridIndex], merge: merge, kind: kind, index: gridIndex)
            guard actual == expected else {
                throw Qwen35PositionSeamError.visualTokenRunMismatch(
                    kind: kind, gridIndex: gridIndex, expected: expected, actual: actual)
            }
            if kind == "image" { imageIndex += 1 } else { videoIndex += 1 }
            cursor = end
        }

        if imageIndex < imageGrids.count {
            let expected = try mergedTokenCount(
                imageGrids[imageIndex], merge: merge, kind: "image", index: imageIndex)
            throw Qwen35PositionSeamError.visualTokenRunMismatch(
                kind: "image", gridIndex: imageIndex, expected: expected, actual: 0)
        }
        if videoIndex < videoGrids.count {
            let expected = try mergedTokenCount(
                videoGrids[videoIndex], merge: merge, kind: "video", index: videoIndex)
            throw Qwen35PositionSeamError.visualTokenRunMismatch(
                kind: "video", gridIndex: videoIndex, expected: expected, actual: 0)
        }
    }

    private func mergedTokenCount(
        _ grid: THW, merge: Int, kind: String, index: Int
    ) throws -> Int {
        let (spatial, spatialOverflow) = (grid.h / merge).multipliedReportingOverflow(
            by: grid.w / merge)
        let (count, temporalOverflow) = grid.t.multipliedReportingOverflow(by: spatial)
        guard !spatialOverflow, !temporalOverflow else {
            throw Qwen35PositionSeamError.invalidGrid(kind: kind, index: index)
        }
        return count
    }

    /// Run the exact vision tower path used by `prepare`, cast its final
    /// output to the text embedding dtype, and return stable ordered slices.
    ///
    /// Images are emitted first in `imageGrids` order. Videos follow in
    /// `videoGrids` order, split into temporal vision frames. Qwen visual
    /// tokens remain ordinary causal language tokens; this API supplies no
    /// Gemma-style bidirectional span mask.
    public func visionFeatures(
        imagePixels: MLXArray? = nil,
        imageGrids: [THW]? = nil,
        videoPixels: MLXArray? = nil,
        videoGrids: [THW]? = nil
    ) throws -> Qwen35VisionFeatures {
        if imagePixels != nil, imageGrids?.isEmpty != false {
            throw Qwen35VisionSeamError.missingImageGrids
        }
        if imagePixels == nil, imageGrids?.isEmpty == false {
            throw Qwen35VisionSeamError.missingImagePixels
        }
        if videoPixels != nil, videoGrids?.isEmpty != false {
            throw Qwen35VisionSeamError.missingVideoGrids
        }
        if videoPixels == nil, videoGrids?.isEmpty == false {
            throw Qwen35VisionSeamError.missingVideoPixels
        }

        let imageGrids = imageGrids ?? []
        let videoGrids = videoGrids ?? []
        let merge = config.visionConfiguration.spatialMergeSize

        func validate(grids: [THW], kind: String) throws {
            for (index, grid) in grids.enumerated()
            where grid.t <= 0 || grid.h <= 0 || grid.w <= 0
                || grid.h % merge != 0 || grid.w % merge != 0
            {
                throw Qwen35VisionSeamError.invalidGrid(kind: kind, index: index)
            }
        }
        try validate(grids: imageGrids, kind: "image")
        try validate(grids: videoGrids, kind: "video")

        var pixelParts: [MLXArray] = []
        if let imagePixels {
            guard imagePixels.ndim == 2 else {
                throw Qwen35VisionSeamError.invalidPixelRank(
                    kind: "image", actual: imagePixels.ndim)
            }
            pixelParts.append(imagePixels)
        }
        if let videoPixels {
            guard videoPixels.ndim == 2 else {
                throw Qwen35VisionSeamError.invalidPixelRank(
                    kind: "video", actual: videoPixels.ndim)
            }
            pixelParts.append(videoPixels)
        }

        func validatePixels(
            _ pixels: MLXArray?, grids: [THW], kind: String
        ) throws {
            var expected = 0
            for (index, grid) in grids.enumerated() {
                let (spatial, spatialOverflow) = grid.h.multipliedReportingOverflow(by: grid.w)
                let (count, temporalOverflow) = grid.t.multipliedReportingOverflow(by: spatial)
                let (total, totalOverflow) = expected.addingReportingOverflow(count)
                guard !spatialOverflow, !temporalOverflow, !totalOverflow else {
                    throw Qwen35VisionSeamError.invalidGrid(kind: kind, index: index)
                }
                expected = total
            }
            let actual = pixels?.dim(0) ?? 0
            guard expected == actual else {
                throw Qwen35VisionSeamError.pixelCountMismatch(
                    kind: kind, expected: expected, actual: actual)
            }
        }
        try validatePixels(imagePixels, grids: imageGrids, kind: "image")
        try validatePixels(videoPixels, grids: videoGrids, kind: "video")

        let grids = imageGrids + videoGrids
        let textDType = languageModel.model.embedTokens(
            MLXArray([Int32(0)]).reshaped(1, 1)
        ).dtype
        guard !pixelParts.isEmpty else {
            return Qwen35VisionFeatures(
                ordered: [],
                flattenedFeatures: MLXArray.zeros(
                    [0, config.textConfiguration.hiddenSize], dtype: textDType))
        }

        let visionTower = try vision()
        let pixels = (pixelParts.count == 1 ? pixelParts[0] : concatenated(pixelParts))
            .asType(visionTower.patchEmbed.proj.weight.dtype)
        let (visionHidden, _) = visionTower(pixels, gridTHW: grids)
        let flattened = visionHidden.asType(textDType)
        let expectedFeatures = grids.reduce(0) { $0 + $1.product / (merge * merge) }
        guard flattened.dim(0) == expectedFeatures else {
            throw Qwen35VisionSeamError.featureCountMismatch(
                expected: expectedFeatures, actual: flattened.dim(0))
        }

        var ordered: [Qwen35VisionFeature] = []
        ordered.reserveCapacity(imageGrids.count + videoGrids.reduce(0) { $0 + $1.t })
        var cursor = 0

        for (index, grid) in imageGrids.enumerated() {
            let count = grid.product / (merge * merge)
            let slice = flattened[cursor ..< cursor + count, 0...]
                .expandedDimensions(axis: 0)
            ordered.append(.init(kind: .image(index: index), features: slice))
            cursor += count
        }
        for (videoIndex, grid) in videoGrids.enumerated() {
            let count = (grid.h / merge) * (grid.w / merge)
            for frameIndex in 0 ..< grid.t {
                let slice = flattened[cursor ..< cursor + count, 0...]
                    .expandedDimensions(axis: 0)
                ordered.append(
                    .init(
                        kind: .videoFrame(
                            videoIndex: videoIndex, frameIndex: frameIndex),
                        features: slice))
                cursor += count
            }
        }

        return Qwen35VisionFeatures(ordered: ordered, flattenedFeatures: flattened)
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        _ = try parameters?.kvCachePlan()
        return try languageModel.makeCache(parameters: parameters)
    }

    public func prepare() throws {
        try languageModel.prepare()
    }

    private func vision() throws -> Qwen3VLVision.VisionModel {
        if let visionModel { return visionModel }
        return try onDemandVision.load().model
    }

    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray,
        inputEmbeds: MLXArray,
        inputIds: MLXArray,
        imageTokenIndex: Int,
        videoTokenIndex: Int
    ) throws -> (MLXArray, MLXArray) {
        let imageMask = (inputIds .== MLXArray(imageTokenIndex))
        let videoMask = (inputIds .== MLXArray(videoTokenIndex))
        var specialMask = imageMask .|| videoMask

        let nImageTokens = specialMask.sum().item(Int.self)

        specialMask = expandedDimensions(specialMask, axis: -1)
        let maskExpanded = broadcast(specialMask, to: inputEmbeds.shape)

        let nImageFeatures = imageFeatures.dim(0)
        let nImageMaskElements = maskExpanded.sum().item(Int.self)
        let imageFeatureSize = imageFeatures.size

        guard nImageMaskElements == imageFeatureSize else {
            throw Qwen35VLError.featureTokenMismatch(expected: nImageTokens, actual: nImageFeatures)
        }

        let originalShape = inputEmbeds.shape
        let flattenedEmbeds = inputEmbeds.flattened()
        let flattenedFeatures = imageFeatures.flattened()
        let flattenedMask = maskExpanded.flattened()

        let indices = nonZero(flattenedMask.asType(.bool))

        var result = flattenedEmbeds
        if !indices.isEmpty && indices.count == flattenedFeatures.size {
            let indexArray = MLXArray(indices.map { UInt32($0) })
            result[indexArray] = flattenedFeatures
        }

        result = result.reshaped(originalShape)
        let visualMask = specialMask.squeezed(axis: -1).asType(.bool)
        return (result, visualMask)
    }

    private func nonZero(_ mask: MLXArray) -> [Int] {
        let values = mask.asArray(Bool.self)
        var indices: [Int] = []
        indices.reserveCapacity(values.count)
        for (idx, value) in values.enumerated() where value {
            indices.append(idx)
        }
        return indices
    }

    private func combinedFrames(imageFrames: [THW]?, videoFrames: [THW]?) -> [THW] {
        var frames: [THW] = []
        if let imageFrames { frames.append(contentsOf: imageFrames) }
        if let videoFrames { frames.append(contentsOf: videoFrames) }
        return frames
    }

    /// Vision tower + image→token merge over the input, shared by `prepare`
    /// and `prepareContinuation` (mirrors `Qwen25VL.inputEmbeddings`).
    private func visionInputEmbeddings(
        _ input: LMInput
    ) throws -> (
        pixelValues: MLXArray?, imageFrames: [THW]?, videoFrames: [THW]?,
        inputEmbeddings: MLXArray?
    ) {
        var pixelValues: MLXArray?
        var imageFrames: [THW]?
        var videoFrames: [THW]?

        var pixelParts: [MLXArray] = []

        if let image = input.image {
            pixelParts.append(image.pixels)
            imageFrames = image.frames
        }
        if let video = input.video {
            pixelParts.append(video.pixels)
            videoFrames = video.frames
        }

        var inputEmbeddings: MLXArray?

        if !pixelParts.isEmpty,
            combinedFrames(imageFrames: imageFrames, videoFrames: videoFrames).nilIfEmpty != nil
        {
            let vision = try vision()
            let typedPixels = concatenated(pixelParts).asType(vision.patchEmbed.proj.weight.dtype)
            pixelValues = typedPixels
            let inputIds = input.text.tokens
            let textEmbeds = languageModel.model.embedTokens(inputIds)
            // The public vision seam runs the same tower and packing order
            // that CBv2 serving uses, and validates grids and pixel counts.
            let features = try visionFeatures(
                imagePixels: input.image?.pixels,
                imageGrids: imageFrames,
                videoPixels: input.video?.pixels,
                videoGrids: videoFrames)

            let (mergedEmbeds, _) = try mergeInputIdsWithImageFeatures(
                imageFeatures: features.flattenedFeatures,
                inputEmbeds: textEmbeds,
                inputIds: inputIds,
                imageTokenIndex: config.imageTokenIndex,
                videoTokenIndex: config.videoTokenIndex
            )
            inputEmbeddings = mergedEmbeds
        }

        return (pixelValues, imageFrames, videoFrames, inputEmbeddings)
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let inputIds = input.text.tokens
        let inputIds2D = inputIds.ndim == 1 ? inputIds[.newAxis, 0...] : inputIds
        let cacheOffset = anchorCacheOffset(cache)
        // Resolved before the routing decision so a warm cache fails closed on either path.
        let positionOffset = try QwenVL.continuationAnchor(
            model: "Qwen35", key: ropeDeltasKey, cacheOffset: cacheOffset,
            batchSize: inputIds2D.dim(0), state: state)

        // Windowed (chunked) prefill — the remaining #344 deferred item for
        // Qwen3.5 — with the same default as the sibling chunked prefills
        // (Gemma3/LLMModel: `prefill.resolvedStepSize()`). The windowed forward also
        // owns every warm continuation (multi-turn chat, tool restart,
        // restored prompt cache): a cold cache is just a continuation
        // anchored at offset 0, and a warm one anchors M-RoPE positions at
        // the cache offset plus the rope delta carried in `state` — never
        // back at zero. The windowed forward is single-sequence; batched
        // inputs keep the single-shot path below.
        let window = prefill.resolvedStepSize()
        if inputIds2D.dim(0) == 1, inputIds2D.dim(-1) > 0,
            cacheOffset > 0 || inputIds2D.dim(-1) > window
        {
            return try prepareContinuation(
                input, inputIds: inputIds2D, cache: cache, cacheOffset: cacheOffset,
                positionOffset: positionOffset, prefill: prefill)
        }

        let (pixelValues, imageFrames, videoFrames, inputEmbeddings) =
            try visionInputEmbeddings(input)

        let typedCache = castCache(cache)
        let output = withPreparedCache(cache, lengths: input.text.sequenceLengths) {
            languageModel(
                inputIds,
                inputsEmbeds: inputEmbeddings,
                cache: typedCache,
                state: state,
                mask: input.text.mask,
                positionIds: nil,
                pixelValues: pixelValues,
                imageGridTHW: imageFrames,
                videoGridTHW: videoFrames,
                lastTokenOnly: true
            )
        }

        let total = inputIds.dim(-1)
        prefill.progress?(total, total)
        return .logits(output)
    }

    /// Cached token count from a full-attention layer, or a linear layer when no full attention exists.
    private func anchorCacheOffset(_ cache: [any KVCache]) -> Int {
        guard let index = languageModel.model.faIdx ?? languageModel.model.ssmIdx,
            cache.indices.contains(index)
        else {
            return 0
        }
        return cache[index].offset
    }

    /// Warm, windowed continuation through an image-bearing remainder — the
    /// windowed forward that `prepare` also delegates to for long prompts.
    ///
    /// It runs the vision tower and the image→token merge **once** over the
    /// remainder, computes the new image's M-RoPE positions **once** from the
    /// seeded **Position Anchor** (so the image's diverging t/h/w indices
    /// start at the anchor, not zero), then drives the language-model forward
    /// in chunks from the prefill parameters. The full-attention scratch is bounded to
    /// `[heads, chunk, L]` instead of `[heads, L, L]`, so it cannot crash on
    /// a long prefix or a large image.
    ///
    /// `cache` must already be warmed to the restore offset `P` (a fresh cache
    /// means `P = 0`, a crash-safe cold prefill). `state` carries the anchor's
    /// rope delta in the `qwen35.ropeDeltas` slot (zero / absent for an
    /// image-free prefix). The returned state carries the rope delta the
    /// post-image text tail resumes with (`getRopeIndex` delta − `P`), so a
    /// caller threading state end-to-end continues that tail with the ordinary
    /// flat-continuation branch. Decode stays caller-owned.
    private func prepareContinuation(
        _ input: LMInput,
        inputIds: MLXArray,
        cache: [any KVCache],
        cacheOffset: Int,
        positionOffset: Int,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let remainderLength = inputIds.dim(-1)
        precondition(remainderLength > 0, "prepareContinuation needs a non-empty remainder")

        // Vision tower + image→token merge — once, over the whole remainder;
        // chunks slice the merged embeddings below.
        let (_, imageFrames, videoFrames, inputEmbeddings) =
            try visionInputEmbeddings(input)

        // Offset-aware M-RoPE positions for the whole remainder — once. The new
        // image's t/h/w indices diverge from the anchor; the returned delta is
        // in the same offset frame.
        let (positionIds, ropeDeltas) = Qwen3VLLanguage.getRopeIndex(
            inputIds: inputIds,
            imageGridTHW: imageFrames,
            videoGridTHW: videoFrames,
            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
            imageTokenId: config.imageTokenId,
            videoTokenId: config.videoTokenId,
            visionStartTokenId: config.visionStartTokenId,
            attentionMask: input.text.mask,
            positionOffset: positionOffset
        )

        // Chunk the forward. Each window forwards `chunk` query tokens against
        // the growing cache, so the full-attention scratch stays `[heads,
        // chunk, L]`. Chunk logits are dropped un-evaluated (only the cache is
        // realized between windows) and the logits the iterator samples come
        // from the reserved final position, so `lm_head` materializes
        // `[1, 1, vocab]`, never `[1, L, vocab]`. `asyncEval` bounds the
        // un-evaluated graph while letting the GPU run window i as the CPU
        // builds window i+1 (same shape as the sibling chunked prefills).
        let typedCache = castCache(cache)

        /// One forward over `range`, slicing positions and embeddings in lockstep.
        func forward(_ range: Range<Int>) -> LMOutput {
            languageModel(
                inputIds[0..., range],
                inputsEmbeds: inputEmbeddings.map { $0[0..., range, 0...] },
                cache: typedCache,
                state: nil,
                mask: nil,
                positionIds: positionIds[0..., 0..., range],
                // Never the pixels: a non-nil value here clears the carried
                // anchor and restarts positions at zero.
                pixelValues: nil,
                imageGridTHW: nil,
                videoGridTHW: nil
            )
        }

        let processed = try prefill.forEachChunk(total: remainderLength) { range in
            _ = forward(range)
            if let typedCache {
                asyncEval(typedCache)
            }
        }
        if processed > 0, let typedCache {
            eval(typedCache)
        }

        let lastLogits = forward(processed ..< remainderLength).logits
        prefill.progress?(remainderLength, remainderLength)

        // Seed the post-image text tail's anchor. The vendor's flat-continuation
        // branch positions tail token j at `tailCacheOffset + ropeDeltas + j`;
        // after this remainder `tailCacheOffset = P + remainderLength`, so the
        // delta the tail needs is the offset-frame `getRopeIndex` delta minus
        // `P` (which `getRopeIndex` implicitly counted into `remainderLength`).
        return .logits(
            LMOutput(
                logits: lastLogits,
                state: QwenVL.continuationResumeState(
                    ropeDeltas: ropeDeltas, cacheOffset: cacheOffset, key: ropeDeltasKey)))
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        precondition(
            anchorCacheOffset(cache ?? []) == 0 || state?[ropeDeltasKey] != nil,
            "Qwen35 cannot continue a warm prompt cache without \(ropeDeltasKey.id)")
        let typedCache = castCacheOptional(cache)
        let result = languageModel(
            input.tokens,
            inputsEmbeds: nil,
            cache: typedCache,
            state: state,
            mask: nil,
            positionIds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil
        )
        return result
    }

    /// Stateless text forward. Fork tests and serving probes call this form.
    /// A warm cache continues at `cacheOffset + arange` (text rope deltas are
    /// zero); a fresh cache computes positions from the tokens.
    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let typedCache = castCacheOptional(cache)
        let result = languageModel(
            inputs,
            inputsEmbeds: nil,
            cache: typedCache,
            state: nil,
            mask: nil,
            positionIds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil
        )
        return result.logits
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = try Qwen35CheckpointPolicy.prepareTarget(
            checkpoint, layout: .vision,
            tiedWordEmbeddings: config.textConfiguration.tieWordEmbeddings)
        checkpoint.weights = try sanitize(
            weights: checkpoint.weights, metadata: checkpoint.metadata)
        return checkpoint
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) throws -> [String:
        MLXArray]
    {
        if metadata["format"]?.lowercased() == "mlx" {
            // Converted checkpoints can keep the MTP head; the drafter loads it, not this model.
            return Qwen35CheckpointPolicy.targetWeights(weights)
        }
        return try sanitize(weights: weights)
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        // Whether the checkpoint stores RMSNorm weights in the raw (un-shifted)
        // convention that needs the `+1` offset. Mirrors the MLXLLM Qwen35 gate
        // so the VLM and text paths agree: a pre-converted MLX checkpoint
        // (conv1d already sanitized) has already been shifted at
        // conversion time and must NOT be shifted again — doing so double-shifts
        // every layernorm and produces garbage tokens. Computed on the incoming
        // weights, before the `mtp.` filter below.
        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        // MTP tensors are not proof of a raw checkpoint: a converted checkpoint can
        // keep them (the framework uses them for speculative decoding), and shifting
        // its already-shifted norms a second time produces garbage tokens. The conv1d
        // layout is the reliable signal on its own.
        let shouldShiftNormWeights = hasUnsanitizedConv1d

        let checkpoint = try Qwen35CheckpointPolicy.prepareTarget(
            .init(weights: weights), layout: .vision,
            tiedWordEmbeddings: config.textConfiguration.tieWordEmbeddings)
        let weights = checkpoint.weights

        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for (key, originalValue) in weights {
            var value = originalValue

            if key.contains("conv1d.weight") && value.dim(-1) != 1 {
                value = value.movedAxis(source: 2, destination: 1)
            }
            if shouldShiftNormWeights
                && normKeys.contains(where: { key.hasSuffix($0) }) && value.ndim == 1
            {
                value = value + MLXArray(1, dtype: value.dtype)
            }

            sanitized[key] = value
        }

        return Qwen3VLVision.VisionModel.sanitize(
            weights: sanitized, config: config.visionConfiguration)
    }
}

extension Qwen35: ModelComponentsProviding {
    public var modelComponents: [ModelComponent: [CheckpointComponent]] {
        [
            .vision: [
                CheckpointComponent(
                    name: "vision_tower",
                    namespaces: ["vision_tower", "model.visual", "visual"],
                    destination: "vision_tower")
            ]
        ]
    }

    public var onDemandComponents: [AnyOnDemandComponent] { [onDemandVision] }
}

extension Qwen35: SpeculativeCacheRewindModel {
    public var maximumNativeTargetCacheRewind: Int { 1 }
}

extension Array where Element == THW {
    fileprivate var nilIfEmpty: [THW]? { isEmpty ? nil : self }
}

extension Qwen35 {
    fileprivate func castCache(_ cache: [any KVCache]) -> [KVCache]? {
        guard !cache.isEmpty else { return nil }
        return cache.map { $0 }
    }

    fileprivate func castCacheOptional(_ cache: [any KVCache]?) -> [KVCache]? {
        guard let cache else { return nil }
        return castCache(cache)
    }
}

// MARK: - Chat conventions

// `Qwen35MoE` subclasses `Qwen35` and inherits both declarations.
extension Qwen35 {
    public var toolCallFormat: ToolCallFormat? { .qwen35 }
    public var reasoningConfig: ReasoningConfig? { QwenReasoningProtocol.tagged }
}
