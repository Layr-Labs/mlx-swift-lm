// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Refusal of a MiMo V2.6 tensor-parallel geometry or installation.
public struct MiMoV26TensorParallelError: Error, CustomStringConvertible, Equatable, Sendable {
    public let description: String
    init(_ description: String) { self.description = description }
}

/// One rank of a two-rank tensor-parallel MiMo V2.6 text model: half the query heads with the KV
/// heads they read, half of each channel width and of the head's vocabulary; the rest replicated.
public struct MiMoV26TensorShard: Sendable, Equatable {
    public let rank: Int
    public let size: Int

    public init(rank: Int, size: Int = 2) throws {
        guard size == 2, (0 ..< size).contains(rank) else {
            throw MiMoV26TensorParallelError("MiMo tensor parallel runs two ranks")
        }
        self.rank = rank
        self.size = size
    }

    /// The axis of a text-model parameter (module path) that each rank keeps half of; nil
    /// replicates it. Names outside the text model, MTP included, throw.
    public static func splitAxis(_ name: String) throws -> Int? {
        let parts = name.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        func refuse() -> MiMoV26TensorParallelError {
            MiMoV26TensorParallelError("not a MiMo text parameter: \(name)")
        }
        let packed: Set<String> = ["weight", "scales", "biases"]
        guard let leaf = parts.last, !parts.contains("") else { throw refuse() }
        let module = parts.dropLast().joined(separator: ".")
        switch module {
        case "model.embed_tokens":
            guard packed.contains(leaf) else { throw refuse() }
            return nil
        case "model.norm":
            guard leaf == "weight" else { throw refuse() }
            return nil
        case "lm_head":
            guard packed.contains(leaf) else { throw refuse() }
            return 0
        default:
            break
        }
        guard parts.count >= 4, parts[0] == "model", parts[1] == "layers",
            let layer = Int(parts[2]), layer >= 0, String(layer) == parts[2]
        else { throw refuse() }
        let rest = parts[3...].joined(separator: ".")
        switch rest {
        case "input_layernorm.weight", "post_attention_layernorm.weight",
            "mlp.gate.weight", "mlp.gate.e_score_correction_bias":
            return nil
        case "self_attn.attention_sink_bias":
            return 0
        default:
            break
        }
        let projection = parts[3 ..< parts.count - 1].joined(separator: ".")
        switch projection {
        case "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj":
            guard packed.contains(leaf) || leaf == "bias" else { throw refuse() }
            return 0
        case "mlp.gate_proj", "mlp.up_proj":
            guard packed.contains(leaf) else { throw refuse() }
            return 0
        case "self_attn.o_proj", "mlp.down_proj":
            guard packed.contains(leaf) else { throw refuse() }
            return 1
        case "mlp.switch_mlp.gate_proj", "mlp.switch_mlp.up_proj":
            guard packed.contains(leaf) else { throw refuse() }
            return 1
        case "mlp.switch_mlp.down_proj":
            guard packed.contains(leaf) else { throw refuse() }
            return 2
        default:
            throw refuse()
        }
    }

    /// The shape this rank stores for a tensor of `shape` cut on `axis`.
    /// An odd cut axis would split a packed word or a quantization group.
    public func localShape(_ shape: [Int], axis: Int?) throws -> [Int] {
        guard let axis else { return shape }
        guard shape.indices.contains(axis), shape[axis] > 0, shape[axis] % size == 0 else {
            throw MiMoV26TensorParallelError(
                "axis \(axis) of \(shape) does not halve into whole groups")
        }
        var local = shape
        local[axis] /= size
        return local
    }

    /// This rank's half of a stored tensor (2-D or a 3-D expert bank) along
    /// `axis`, row-major. Nil keeps the whole tensor.
    public func slice(_ value: MLXArray, axis: Int?) throws -> MLXArray {
        let local = try localShape(value.shape, axis: axis)
        guard let axis else { return value }
        let part = contiguous(value.split(parts: size, axis: axis)[rank])
        guard part.shape == local else {
            throw MiMoV26TensorParallelError("cut \(value.shape) to \(part.shape), not \(local)")
        }
        return part
    }

    /// The bytes of a row-major payload of `shape` this rank reads: one run
    /// per index of the dimensions before `axis`.
    public func byteRuns(shape: [Int], axis: Int?, bytes: Int) throws -> GLM5NextByteRuns {
        guard let axis else { return .whole(bytes) }
        _ = try localShape(shape, axis: axis)
        let elements = shape.reduce(1, *)
        guard bytes > 0, bytes % elements == 0 else {
            throw MiMoV26TensorParallelError("payload of \(bytes) bytes is not \(shape)")
        }
        let leading = shape[..<axis].reduce(1, *)
        let row = shape[axis...].reduce(1, *) * (bytes / elements)
        let part = row / size
        return GLM5NextByteRuns(offset: rank * part, length: part, count: leading, stride: row)
    }

    /// Vocabulary rows this rank's head projects.
    public func vocabularyRange(_ c: MiMoV26Configuration) throws -> Range<Int> {
        guard c.vocabularySize > 0, c.vocabularySize % size == 0 else {
            throw MiMoV26TensorParallelError(
                "vocabulary \(c.vocabularySize) does not divide by \(size)")
        }
        let half = c.vocabularySize / size
        return (rank * half) ..< ((rank + 1) * half)
    }

    /// This rank's configuration: half of each layer kind's heads and of both channel widths,
    /// without the MTP heads or the towers.
    public func localConfiguration(_ c: MiMoV26Configuration) throws -> MiMoV26Configuration {
        guard !c.tieWordEmbeddings else {
            throw MiMoV26TensorParallelError("MiMo tensor parallel splits the untied head only")
        }
        for (label, geometry) in [("full", c.fullAttention), ("sliding", c.slidingAttention)] {
            guard geometry.queryHeads % size == 0, geometry.keyValueHeads % size == 0 else {
                throw MiMoV26TensorParallelError(
                    "\(label) attention (\(geometry.queryHeads) query, \(geometry.keyValueHeads) KV heads) "
                        + "does not halve into whole query groups")
            }
        }
        guard c.intermediateSize % size == 0, c.moeIntermediateSize % size == 0 else {
            throw MiMoV26TensorParallelError("channel widths do not divide by \(size)")
        }
        _ = try vocabularyRange(c)
        var fields = c.rawFields
        for key in [
            "num_attention_heads", "num_key_value_heads", "swa_num_attention_heads",
            "swa_num_key_value_heads", "intermediate_size", "moe_intermediate_size",
        ] {
            guard case .number(let value) = fields[key] else {
                throw MiMoV26TensorParallelError("configuration lacks \(key)")
            }
            fields[key] = .number(value / Decimal(size))
        }
        fields["num_nextn_predict_layers"] = .number(0)
        for key in ["omlx_mimo_mtp", "vision_config", "audio_config", "processor_config"] {
            fields[key] = nil
        }
        return try MiMoV26Configuration(rawFields: fields)
    }
}

/// One rank's cross-rank sum, shared by every layer. Not a module: it stays
/// out of the parameter tree.
final class MiMoV26TensorParallelContext {
    let reduce: (MLXArray) -> MLXArray
    let vocabularyRows: Range<Int>

    init(reduce: @escaping (MLXArray) -> MLXArray, vocabularyRows: Range<Int>) {
        self.reduce = reduce
        self.vocabularyRows = vocabularyRows
    }

    /// A partial stored in the activation dtype, widened exactly, summed in
    /// FP32 and rounded once.
    func sum(_ partial: MLXArray) -> MLXArray {
        reduce(partial.asType(.float32)).asType(partial.dtype)
    }
}

extension MiMoV26MoE {
    /// The FP32 weighted combine of expert rows `[..., K, H]`: the single-Mac combine without its
    /// final rounding (each product rounded to FP32, then an FP32 sum).
    static func combinePartial(_ experts: MLXArray, weights: MLXArray) -> MLXArray {
        (experts * weights[.ellipsis, .newAxis]).sum(axis: -2)
    }

    /// This rank's FP32 partial: one combined row, not `TensorParallelSwitchLinear`'s per-expert
    /// rows, so the MXFP4 kernels run; each expert's half-width row is rounded to BF16 first.
    func tensorParallelPartial(_ x: MLXArray, rowLocalRouter: Bool = false) -> MLXArray {
        let routed = gate(x, rowLocal: rowLocalRouter)
        return Self.combinePartial(switchMLP(x, routed.indices), weights: routed.weights)
    }
}

extension MiMoV26DecoderLayer {
    /// The feed-forward of a tensor-parallel rank, summed across ranks and
    /// rounded once to the activation dtype.
    func tensorParallelFeedForward(
        _ x: MLXArray, _ context: MiMoV26TensorParallelContext, rowLocal: Bool = false
    ) -> MLXArray {
        if let moe = mlp as? MiMoV26MoE {
            return context.reduce(moe.tensorParallelPartial(x, rowLocalRouter: rowLocal))
                .asType(x.dtype)
        }
        return context.sum(
            rowLocal ? MiMoV26RectangularDense.mlp(mlp, x, enabled: true) : mlp(x))
    }

    /// The feed-forward output: summed across ranks when tensor parallel is installed.
    func feedForward(_ x: MLXArray) -> MLXArray {
        guard let tensorParallel else { return mlp(x) }
        return tensorParallelFeedForward(x, tensorParallel)
    }
}

extension MiMoV26TextModel {
    /// Vocabulary rows of this rank's head; nil for a whole model.
    public var tensorParallelVocabularyRows: Range<Int>? { tensorParallel?.vocabularyRows }

    /// Vocabulary rows this model's logits cover.
    public var headVocabularyRange: Range<Int> {
        tensorParallel?.vocabularyRows ?? 0 ..< configuration.vocabularySize
    }

    /// Makes this model, built from `MiMoV26TensorShard.localConfiguration`, one rank before its
    /// weights load: each layer's two partials sum through the lazy `reduce`, and `lm_head` holds
    /// `vocabularyRows`. CBv2 and MTP owners refuse the rank.
    public func installTensorParallel(
        reduce: @escaping (MLXArray) -> MLXArray, vocabularyRows: Range<Int>
    ) throws {
        guard tensorParallel == nil else {
            throw MiMoV26TensorParallelError("tensor parallel is already installed")
        }
        guard !configuration.tieWordEmbeddings, let head = lmHead,
            type(of: head) == Linear.self,
            head.weight.shape == [configuration.vocabularySize, configuration.hiddenSize]
        else {
            throw MiMoV26TensorParallelError(
                "install tensor parallel on an untied model before its weights load")
        }
        guard !vocabularyRows.isEmpty, vocabularyRows.lowerBound >= 0,
            vocabularyRows.upperBound <= configuration.vocabularySize
        else {
            throw MiMoV26TensorParallelError("vocabulary rows \(vocabularyRows) are not the head's")
        }
        try update(
            modules: ModuleChildren.unflattened([
                ("lm_head", Linear(configuration.hiddenSize, vocabularyRows.count, bias: false))
            ]), verify: .none)
        let context = MiMoV26TensorParallelContext(reduce: reduce, vocabularyRows: vocabularyRows)
        tensorParallel = context
        for layer in model.layers { layer.tensorParallel = context }
    }
}
