// Copyright © 2026 Eigen Labs.
// Native MiMo V2.6 text component, deliberately not a public factory alias.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum MiMoV26ExecutionError: Error, Equatable, Sendable {
    case unsupportedConfiguration(String)
    case invalidInput(String)
    case unsupportedCache(layer: Int)
    case cacheGeometry(layer: Int)
    case contextExceeded
}

final class MiMoV26DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: MiMoV26Attention
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    let mlp: UnaryLayer
    let geometry: MiMoV26AttentionGeometry

    init(_ config: MiMoV26Configuration, layer: Int) throws {
        geometry = try config.attentionGeometry(at: layer)
        _selfAttention.wrappedValue = MiMoV26Attention(config, geometry: geometry)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize,
                                         eps: Float(config.layernormEpsilon))
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize,
                                                 eps: Float(config.layernormEpsilon))
        mlp = config.moeLayerFrequency[layer] == 1 ? MiMoV26MoE(config)
            : MiMoV26DenseMLP(hiddenSize: config.hiddenSize,
                             intermediateSize: config.intermediateSize)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode,
                        cache: KVCache?) -> MLXArray {
        let residual = x + selfAttention(inputNorm(x), mask: mask, cache: cache)
        return residual + mlp(postAttentionNorm(residual))
    }
}

public final class MiMoV26TextBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "norm") var norm: RMSNorm
    let layers: [MiMoV26DecoderLayer]
    let configuration: MiMoV26Configuration

    // Opt-in until this exact source/binary passes the native numerical and
    // complete-state gates. Tests set this on a fresh model before evaluation.
    var useFusedDecodeNorms =
        ProcessInfo.processInfo.environment["DARKBLOOM_MIMO_FUSED_DECODE_NORMS"] == "1"

    init(_ config: MiMoV26Configuration) throws {
        configuration = config
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabularySize,
                                               dimensions: config.hiddenSize)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize,
                                     eps: Float(config.layernormEpsilon))
        layers = try (0..<config.numHiddenLayers).map { try MiMoV26DecoderLayer(config, layer: $0) }
    }

    func forward(embeddings: MLXArray, cache: [KVCache]?, captureLayers: Set<Int>)
        -> (normalized: MLXArray, features: [Int: MLXArray]) {
        var hidden = embeddings
        var nextInput: MLXArray?
        var features: [Int: MLXArray] = [:]
        for (index, layer) in layers.enumerated() {
            let layerCache = cache?[index]
            let mask = createAttentionMask(h: hidden, cache: layerCache,
                                           windowSize: layer.geometry.slidingWindow)
            let normalized = nextInput ?? layer.inputNorm(hidden)
            let attention = layer.selfAttention(normalized, mask: mask, cache: layerCache)
            let nextNorm = index + 1 < layers.count ? layers[index + 1].inputNorm : norm
            if let fused = MiMoV26DecodeKernels.finishLayer(
                hidden, attentionOutput: attention, layer: layer,
                nextNorm: nextNorm, enabled: useFusedDecodeNorms) {
                hidden = fused.residual
                nextInput = fused.normalized
            } else {
                let residual = hidden + attention
                hidden = residual + layer.mlp(layer.postAttentionNorm(residual))
                nextInput = nil
            }
            if captureLayers.contains(index) { features[index] = hidden }
        }
        return (nextInput ?? norm(hidden), features)
    }
}

/// Request-owned outputs. Captured features are opt-in; the model never retains
/// a previous request's hidden tensors, rotary offsets or speculative state.
public struct MiMoV26TextOutput {
    public let logits: MLXArray
    public let normalizedHiddenStates: MLXArray
    public let layerFeatures: [Int: MLXArray]
    public let firstPosition: Int
    let ownerIdentity: UUID
}

/// Target-text building block for the future native omnimodal wrapper. It does
/// not register `mimo_v2`, discard non-text weights, advertise CBv2/paged support,
/// or infer MTP activation from the presence of trained heads in a checkpoint.
public final class MiMoV26TextModel: Module {
    public let configuration: MiMoV26Configuration
    public let model: MiMoV26TextBackbone
    let activationDType: DType
    let identity = UUID()
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    /// Packed storage dtype is not the activation dtype. Validate all declared
    /// affine parameters without materializing or dequantizing the full table.
    var hasLoadedEmbeddingPrecision: Bool {
        let embedding = model.embedTokens
        if let packed = embedding as? QuantizedEmbedding {
            return packed.mode == .affine && packed.bits == 8 && packed.groupSize == 64
                && packed.weight.dtype == .uint32 && packed.scales.dtype == activationDType
                && packed.biases?.dtype == activationDType
                && packed.weight.shape == [configuration.vocabularySize, configuration.hiddenSize / 4]
                && packed.scales.shape == [configuration.vocabularySize, configuration.hiddenSize / 64]
                && packed.biases?.shape == packed.scales.shape
        }
        return embedding.weight.dtype == activationDType
            && embedding.weight.shape == [configuration.vocabularySize, configuration.hiddenSize]
    }

    var hasLoadedReadoutPrecision: Bool {
        guard let readout = lmHead else { return hasLoadedEmbeddingPrecision }
        if let packed = readout as? QuantizedLinear {
            return packed.mode == .affine && packed.bits == 8 && packed.groupSize == 64
                && packed.weight.dtype == .uint32 && packed.scales.dtype == activationDType
                && packed.biases?.dtype == activationDType
                && packed.weight.shape == [configuration.vocabularySize, configuration.hiddenSize / 4]
                && packed.scales.shape == [configuration.vocabularySize, configuration.hiddenSize / 64]
                && packed.biases?.shape == packed.scales.shape
        }
        return readout.weight.dtype == activationDType
            && readout.weight.shape == [configuration.vocabularySize, configuration.hiddenSize]
    }

    public init(_ config: MiMoV26Configuration) throws {
        // Decoding preserves future metadata; construction must separately
        // reject operational semantics not implemented by this component.
        var dimensions: [Int] = [config.hiddenSize, config.intermediateSize, config.moeIntermediateSize,
                                 config.vocabularySize, config.numHiddenLayers, config.maxPositionEmbeddings,
                                 config.routedExpertCount]
        for geometry in [config.fullAttention, config.slidingAttention] {
            dimensions.append(contentsOf: [geometry.queryHeads, geometry.keyValueHeads,
                                           geometry.headDim, geometry.valueHeadDim])
            dimensions.append(geometry.queryHeads * geometry.headDim)
            dimensions.append(geometry.queryHeads * geometry.valueHeadDim)
            dimensions.append(geometry.keyValueHeads * geometry.headDim)
            dimensions.append(geometry.keyValueHeads * geometry.valueHeadDim)
        }
        guard dimensions.allSatisfy({ $0 > 0 && $0 <= Int(Int32.max) }) else {
            throw MiMoV26ExecutionError.unsupportedConfiguration("MLX shape exceeds Int32")
        }
        var tensorShapes: [[Int]] = [[config.vocabularySize, config.hiddenSize],
                                    [config.hiddenSize, config.intermediateSize],
                                    [config.routedExpertCount, config.hiddenSize, config.moeIntermediateSize]]
        for geometry in [config.fullAttention, config.slidingAttention] {
            tensorShapes.append([geometry.queryHeads * geometry.headDim, config.hiddenSize])
            tensorShapes.append([geometry.keyValueHeads * geometry.headDim, config.hiddenSize])
            tensorShapes.append([geometry.keyValueHeads * geometry.valueHeadDim, config.hiddenSize])
            tensorShapes.append([geometry.queryHeads * geometry.valueHeadDim, config.hiddenSize])
        }
        for shape in tensorShapes {
            var bytes = 4 // Initial module parameters are FP32 before strict loading.
            for dimension in shape {
                let product = bytes.multipliedReportingOverflow(by: dimension)
                guard !product.overflow else {
                    throw MiMoV26ExecutionError.unsupportedConfiguration("tensor byte geometry overflow")
                }
                bytes = product.partialValue
            }
        }
        let scalars = [config.attentionValueScale, config.layernormEpsilon,
                       config.fullAttention.ropeTheta, config.slidingAttention.ropeTheta,
                       config.routedScalingFactor ?? 1]
        guard scalars.allSatisfy({ Float($0).isFinite && Float($0) > 0 }) else {
            throw MiMoV26ExecutionError.unsupportedConfiguration("nonrepresentable Float scalar")
        }
        guard config.sharedExpertCount == nil || config.sharedExpertCount == 0 else {
            throw MiMoV26ExecutionError.unsupportedConfiguration("shared experts")
        }
        if let scaling = config.rawFields["rope_scaling"], scaling != .null {
            throw MiMoV26ExecutionError.unsupportedConfiguration("rope_scaling")
        }
        guard config.attentionDropout == 0 else {
            throw MiMoV26ExecutionError.unsupportedConfiguration("nonzero attention dropout")
        }
        configuration = config
        activationDType = config.dtype == "bfloat16" ? .bfloat16
            : (config.dtype == "float16" ? .float16 : .float32)
        model = try MiMoV26TextBackbone(config)
        _lmHead.wrappedValue = config.tieWordEmbeddings ? nil
            : Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    public func newCache() -> [KVCache] {
        model.layers.map { layer in
            if let window = layer.geometry.slidingWindow {
                return RotatingKVCache(maxSize: window)
            }
            return KVCacheSimple()
        }
    }

    /// Supply token IDs or already prepared native embeddings, never both.
    /// Native processors must preserve media order/positions before this seam.
    /// logitsStart avoids projecting discarded prefix rows without changing
    /// attention, retained hidden states or cache updates for those rows.
    public func forward(inputIDs: MLXArray? = nil, embeddings: MLXArray? = nil,
                        cache: [KVCache]? = nil, logitsStart: Int = 0,
                        captureLayers: Set<Int> = []) throws -> MiMoV26TextOutput {
        guard (inputIDs != nil) != (embeddings != nil) else {
            throw MiMoV26ExecutionError.invalidInput("provide IDs or embeddings, not both")
        }
        let batch: Int
        let length: Int
        if let inputIDs {
            guard inputIDs.ndim == 2, inputIDs.dtype == .int32 || inputIDs.dtype == .uint32 else {
                throw MiMoV26ExecutionError.invalidInput("IDs require rank-two int32/uint32")
            }
            batch = inputIDs.dim(0)
            length = inputIDs.dim(1)
        } else if let embeddings {
            guard embeddings.ndim == 3, embeddings.dim(2) == configuration.hiddenSize,
                  embeddings.dtype == activationDType else {
                throw MiMoV26ExecutionError.invalidInput("invalid native embeddings")
            }
            batch = embeddings.dim(0)
            length = embeddings.dim(1)
        } else {
            throw MiMoV26ExecutionError.invalidInput("missing input")
        }
        guard batch > 0, length > 0, logitsStart >= 0, logitsStart < length,
              captureLayers.allSatisfy({ model.layers.indices.contains($0) }) else {
            throw MiMoV26ExecutionError.invalidInput("invalid forward geometry")
        }
        if let inputIDs {
            // Direct SDK inputs must not inherit MLX's negative-index behavior.
            // Validate before embedding access or mutation of any layer cache.
            let minimum = inputIDs.dtype == .int32 ? Int64(inputIDs.min().item(Int32.self))
                : Int64(inputIDs.min().item(UInt32.self))
            let maximum = inputIDs.dtype == .int32 ? Int64(inputIDs.max().item(Int32.self))
                : Int64(inputIDs.max().item(UInt32.self))
            guard minimum >= 0, maximum < Int64(configuration.vocabularySize) else {
                throw MiMoV26ExecutionError.invalidInput("token ID outside vocabulary")
            }
            guard hasLoadedEmbeddingPrecision, hasLoadedReadoutPrecision else {
                throw MiMoV26ExecutionError.unsupportedConfiguration("embedding precision is not loaded")
            }
        }
        try validateCache(cache, batch: batch, length: length)
        let firstPosition = cache?.first?.offset ?? 0
        let hidden = embeddings ?? model.embedTokens(inputIDs!)
        let result = model.forward(embeddings: hidden, cache: cache, captureLayers: captureLayers)
        let projected = result.normalized[0..., logitsStart..., 0...]
        let logits = lmHead.map { $0(projected) } ?? model.embedTokens.asLinear(projected)
        return MiMoV26TextOutput(logits: logits, normalizedHiddenStates: result.normalized,
                                layerFeatures: result.features, firstPosition: firstPosition,
                                ownerIdentity: identity)
    }

    private func validateCache(_ cache: [KVCache]?, batch: Int, length: Int) throws {
        guard length <= configuration.maxPositionEmbeddings else {
            throw MiMoV26ExecutionError.contextExceeded
        }
        guard let cache else { return }
        guard cache.count == model.layers.count else {
            throw MiMoV26ExecutionError.invalidInput("one cache per target layer is required")
        }
        let absoluteOffset = cache.first?.offset ?? 0
        guard absoluteOffset >= 0, absoluteOffset <= configuration.maxPositionEmbeddings - length else {
            throw MiMoV26ExecutionError.contextExceeded
        }
        // Validate the complete list before the first layer can append anything.
        var owners = Set<ObjectIdentifier>()
        for (index, entry) in cache.enumerated() {
            let geometry = model.layers[index].geometry
            guard let base = entry as? BaseKVCache,
                  type(of: base) == KVCacheSimple.self || type(of: base) == RotatingKVCache.self else {
                throw MiMoV26ExecutionError.unsupportedCache(layer: index)
            }
            guard owners.insert(ObjectIdentifier(base)).inserted,
                  entry.offset == absoluteOffset, entry.maxSize == geometry.slidingWindow else {
                throw MiMoV26ExecutionError.cacheGeometry(layer: index)
            }
            // Raw ownership state, before a malformed offset can create a slice.
            let state = base.innerState()
            guard state.isEmpty || (state.count == 2 && state.allSatisfy { $0.ndim == 4 }) else {
                throw MiMoV26ExecutionError.cacheGeometry(layer: index)
            }
            if state.isEmpty && absoluteOffset != 0 {
                throw MiMoV26ExecutionError.cacheGeometry(layer: index)
            }
            if let simple = base as? KVCacheSimple {
                guard simple.step > 0, simple.step <= Int(Int32.max), simple.step <= Int.max - length,
                      state.isEmpty || absoluteOffset <= state[0].dim(2) else {
                    throw MiMoV26ExecutionError.cacheGeometry(layer: index)
                }
            } else {
                let raw = base.metaState
                let metadata = raw.compactMap(Int.init)
                guard raw.count == 5, metadata.count == 5, metadata[0] == 0,
                      metadata[1] == geometry.slidingWindow, metadata[2] > 0,
                      metadata[2] <= Int(Int32.max),
                      metadata[2] <= Int.max - length, metadata[3] == absoluteOffset,
                      metadata[4] >= 0, metadata[4] <= min(absoluteOffset, state.first?.dim(2) ?? 0),
                      state.isEmpty || state[0].dim(2) >= min(absoluteOffset, metadata[1]) else {
                    throw MiMoV26ExecutionError.cacheGeometry(layer: index)
                }
            }
            if !state.isEmpty {
                let key = state[0], value = state[1]
                guard key.dim(0) == batch, value.dim(0) == batch,
                      key.dim(1) == geometry.keyValueHeads, value.dim(1) == geometry.keyValueHeads,
                      key.dim(2) == value.dim(2), key.dim(3) == geometry.headDim,
                      value.dim(3) == geometry.valueHeadDim,
                      key.dtype == activationDType, value.dtype == activationDType else {
                    throw MiMoV26ExecutionError.cacheGeometry(layer: index)
                }
            }
        }
    }
}
