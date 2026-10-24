import Cmlx
import MLX
import MLXLMCommon

/// Durable prompt-history checkpoints of the Super assistant, for complete
/// (SSD) prefix checkpoints. A checkpoint at `count` target inputs is a
/// `PrefixCheckpoint`: the FP8 history codes of every whole
/// `historyChunkRows` block, the queued partial block and the frontier. The
/// split depends only on `count`, so the tensor layout does too. The codes
/// are stored widened to float16 integers in 0...255, which is exact: the
/// generic recurrent manifest admits only floating assistant tensors. Decode
/// checks every value before narrowing it back, so a restored request writes
/// the same codes, and the mirror decoded from them, as the request the
/// checkpoint was taken from.
extension NemotronSuperMTPAssistant: CBv2MTPPrefixCheckpointCoding {
    public var prefixCheckpointCodecID: String {
        let c = target.configuration
        let (heads, headDim) = historyGeometry
        return "nemotron-super-mtp-fp8-history-v1:h\(c.hiddenSize):v\(c.vocabSize)"
            + ":kv\(heads)x\(headDim):rows\(workspace.graph.historyChunkRows)"
            + ":\(head.weightDType):codes-f16"
    }

    /// K and V codes when whole blocks were drained (layer tags 0 and 1 keep
    /// the two roles distinct), the queued pairs when any, then the frontier.
    public func prefixCheckpointTensorDescriptors(targetInputCount count: Int)
        -> [CBv2CheckpointTensorDescriptor]?
    {
        guard let split = historySplit(targetInputCount: count),
            let activation = CBv2CheckpointDType(head.weightDType), activation.isFloatingPoint
        else { return nil }
        let width = target.configuration.hiddenSize
        let (heads, headDim) = historyGeometry
        var descriptors: [CBv2CheckpointTensorDescriptor] = []
        do {
            if split.drained > 0 {
                for layer in 0 ..< 2 {
                    descriptors.append(
                        try .init(
                            role: .assistantHidden, layer: layer,
                            shape: [1, heads, split.drained, headDim], dtype: .float16))
                }
            }
            if split.queued > 0 {
                descriptors.append(
                    try .init(role: .assistantHidden, shape: [1, split.queued, width], dtype: activation))
                descriptors.append(
                    try .init(role: .assistantTokens, shape: [1, split.queued], dtype: .int32))
            }
            descriptors.append(
                try .init(role: .assistantFrontier, shape: [1, 1, width], dtype: activation))
        } catch {
            return nil
        }
        return descriptors
    }

    public func encodePrefixCheckpoint(_ checkpoint: any CBv2MTPPrefixCheckpoint) -> [MLXArray]? {
        guard let checkpoint = checkpoint as? PrefixCheckpoint,
            checkpoint.owner == ObjectIdentifier(self),
            let descriptors = prefixCheckpointTensorDescriptors(
                targetInputCount: checkpoint.targetInputCount)
        else { return nil }
        let arrays =
            checkpoint.codes.map { $0.asType(.float16) }
            + [checkpoint.hidden, checkpoint.tokens, checkpoint.frontier].compactMap { $0 }
        return Self.conforms(arrays, to: descriptors) ? arrays : nil
    }

    /// The arrays are the evaluated host buffers of an authenticated import.
    /// Nil unless they match this assistant's layout at `prefixTokens.count`,
    /// every code is an integer in 0...255 and the queued tokens are the
    /// prompt's own.
    public func decodePrefixCheckpoint(tensors: [MLXArray], prefixTokens: [Int])
        -> (any CBv2MTPPrefixCheckpoint)?
    {
        let count = prefixTokens.count
        guard let split = historySplit(targetInputCount: count),
            let descriptors = prefixCheckpointTensorDescriptors(targetInputCount: count),
            Self.conforms(tensors, to: descriptors),
            prefixTokens.allSatisfy({ $0 >= 0 && $0 < target.configuration.vocabSize })
        else { return nil }
        var cursor = 0
        var codes: [MLXArray] = []
        if split.drained > 0 {
            for _ in 0 ..< 2 {
                guard let narrowed = Self.narrowedCodes(tensors[cursor]) else { return nil }
                codes.append(narrowed)
                cursor += 1
            }
        }
        var hidden: MLXArray?
        var tokens: MLXArray?
        if split.queued > 0 {
            // Pair i holds input i + 1: the queue starts after the drained pairs.
            guard Self.tokens(tensors[cursor + 1], equal: prefixTokens[(split.drained + 1)...])
            else { return nil }
            hidden = tensors[cursor]
            tokens = tensors[cursor + 1]
            cursor += 2
        }
        return PrefixCheckpoint(
            owner: ObjectIdentifier(self), targetInputCount: count, codes: codes,
            hidden: hidden, tokens: tokens, frontier: tensors[cursor])
    }

    private var historyGeometry: (heads: Int, headDim: Int) {
        let c = target.configuration
        return (c.numKeyValueHeads, c.headDim ?? (c.hiddenSize / c.numAttentionHeads))
    }

    /// Pairs drained into whole history blocks, and pairs still queued, at
    /// `count` target inputs: the invariant `capturePrefixCheckpoint` holds.
    private func historySplit(targetInputCount count: Int) -> (drained: Int, queued: Int)? {
        let chunk = workspace.graph.historyChunkRows
        guard count > 1, chunk > 0 else { return nil }
        let pairs = count - 1
        return (pairs / chunk * chunk, pairs % chunk)
    }

    private static func conforms(
        _ arrays: [MLXArray], to descriptors: [CBv2CheckpointTensorDescriptor]
    ) -> Bool {
        arrays.count == descriptors.count
            && zip(arrays, descriptors).allSatisfy {
                $0.shape == $1.shape && $0.dtype == $1.dtype.mlxDType
            }
    }

    /// The uint8 codes of a widened tensor, read from its evaluated host
    /// buffer without GPU work; nil unless every value is an integer in 0...255.
    private static func narrowedCodes(_ widened: MLXArray) -> MLXArray? {
        guard widened.dtype == .float16, let info = try? widened.evaluatedBufferInfo(),
            info.isRowContiguous, info.dataElements == widened.size,
            let pointer = mlx_array_data_float16(widened.ctx)
        else { return nil }
        return withExtendedLifetime(widened) {
            var codes = [UInt8](repeating: 0, count: widened.size)
            for index in 0 ..< widened.size {
                let value = Float(pointer[index])
                guard value >= 0, value <= 255, value == value.rounded() else { return nil }
                codes[index] = UInt8(value)
            }
            return MLXArray(codes, widened.shape)
        }
    }

    private static func tokens(_ tokens: MLXArray, equal expected: ArraySlice<Int>) -> Bool {
        guard tokens.size == expected.count, let info = try? tokens.evaluatedBufferInfo(),
            info.isRowContiguous, info.dataElements == tokens.size,
            let pointer = mlx_array_data_int32(tokens.ctx)
        else { return false }
        return withExtendedLifetime(tokens) {
            for (index, token) in expected.enumerated() where Int(pointer[index]) != token {
                return false
            }
            return true
        }
    }
}
