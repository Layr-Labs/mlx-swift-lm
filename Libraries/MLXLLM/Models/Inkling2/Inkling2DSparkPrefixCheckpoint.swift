import Foundation
import MLX
import MLXLMCommon

/// Drafter history at a prompt boundary of `targetInputCount` positions: the
/// context K/V of positions `0 ..< targetInputCount - 1` (sliding layers keep
/// only their window), the frontier feature row of the last position, which a
/// restored request writes or replaces with its round seed exactly as a cold
/// request would, and the boundary's prompt tokens as identity evidence.
final class Inkling2DSparkPrefixCheckpoint: CBv2MTPPrefixCheckpoint {
    weak var owner: Inkling2DSparkDrafter?
    let targetInputCount: Int
    /// Positions held as context K/V (`targetInputCount - 1`).
    var contextCount: Int { targetInputCount - 1 }
    /// Per layer `[1, kvHeads, retained, headDim]`, the newest `retained`
    /// context positions.
    let keys: [MLXArray]
    let values: [MLXArray]
    /// `[1, 1, featureWidth]`
    let frontier: MLXArray
    /// `[1, targetInputCount]` int32
    let tokens: MLXArray
    let expectedTokens: [Int32]
    let materializedBytes: Int
    let imported: Bool
    var importedStorageOwner: AnyObject?

    init(
        owner: Inkling2DSparkDrafter, targetInputCount: Int, keys: [MLXArray],
        values: [MLXArray], frontier: MLXArray, tokens: MLXArray, expectedTokens: [Int32],
        materializedBytes: Int, imported: Bool = false
    ) {
        self.owner = owner
        self.targetInputCount = targetInputCount
        self.keys = keys
        self.values = values
        self.frontier = frontier
        self.tokens = tokens
        self.expectedTokens = expectedTokens
        self.materializedBytes = materializedBytes
        self.imported = imported
    }

    /// Descriptor order: keys and values per layer, frontier, tokens.
    var evaluationTargets: [MLXArray] {
        zip(keys, values).flatMap { [$0, $1] } + [frontier, tokens]
    }
}

/// A fresh buffer with the same bits (a view may alias a larger donor).
private func inkling2DSparkCopy(_ x: MLXArray) -> MLXArray {
    MLX.where(MLXArray(true), x, x)
}

extension Inkling2DSparkDrafter: CBv2HistoricalMTPPrefixCheckpointCoding {
    public var prefixCheckpointCodecID: String {
        let c = configuration
        return "inkling2-dspark-ctx-kv-l\(c.numHiddenLayers)-kv\(c.numKeyValueHeads)"
            + "x\(c.headDim)-w\(c.slidingWindowSize)-f\(c.featureWidth)-\(model.dtype)-v1"
    }

    public func prefixCheckpointTensorDescriptors(targetInputCount count: Int)
        -> [CBv2CheckpointTensorDescriptor]?
    {
        let c = configuration
        guard count >= 4, let dtype = CBv2CheckpointDType(model.dtype) else { return nil }
        do {
            var result: [CBv2CheckpointTensorDescriptor] = []
            for layer in 0 ..< c.numHiddenLayers {
                let rows =
                    c.isLocal(layer: layer)
                    ? min(count - 1, c.slidingWindowSize) : count - 1
                let shape = [1, c.numKeyValueHeads, rows, c.headDim]
                result.append(
                    try .init(role: .assistantKeys, layer: layer, shape: shape, dtype: dtype))
                result.append(
                    try .init(role: .assistantValues, layer: layer, shape: shape, dtype: dtype))
            }
            result.append(
                try .init(role: .assistantFrontier, shape: [1, 1, c.featureWidth], dtype: dtype))
            result.append(try .init(role: .assistantTokens, shape: [1, count], dtype: .int32))
            return result
        } catch { return nil }
    }

    /// Called once on a fresh state before it observes the prompt, or on a
    /// restored state before its first suffix input. The prompt is copied.
    package func installPrefixCaptureContext(
        requestState: any CBv2MTPRequestState, promptTokens: [Int]
    ) throws {
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released,
            s.prefixMayInstall, s.promptOnly, s.stagedInputCount == 0, !promptTokens.isEmpty,
            promptTokens.allSatisfy({
                $0 >= 0 && $0 < configuration.paddedVocabularySize && $0 <= Int(Int32.max)
            })
        else { throw Inkling2DSparkError.target("invalid or late prefix capture context") }
        if let restored = s.restored {
            guard restored.targetInputCount == s.committedInputCount,
                restored.targetInputCount < promptTokens.count,
                restored.expectedTokens.enumerated().allSatisfy({
                    Int($0.element) == promptTokens[$0.offset]
                })
            else { throw Inkling2DSparkError.target("restored prompt prefix changed") }
        } else {
            guard s.committedInputCount == 0, s.pending.isEmpty, s.frontier == nil,
                s.observedTokens.isEmpty, (s.paged?.written ?? 0) == 0
            else { throw Inkling2DSparkError.target("context did not precede observations") }
        }
        s.prefixContext = promptTokens.map { Int32($0) }
        s.prefixMayInstall = false
    }

    package func bindImportedPrefixCheckpointOwner(
        _ checkpoint: any CBv2MTPPrefixCheckpoint, owner: AnyObject
    ) throws {
        guard let value = checkpoint as? Inkling2DSparkPrefixCheckpoint, value.imported,
            value.owner === self,
            value.importedStorageOwner == nil || value.importedStorageOwner === owner
        else { throw Inkling2DSparkError.target("foreign imported checkpoint") }
        value.importedStorageOwner = owner
    }

    /// The boundary's context K/V per layer, window-limited, plus the copied
    /// frontier: restored or written context first, then queued rows.
    private func boundaryContext(_ s: State, count: Int) -> (
        keys: [MLXArray], values: [MLXArray]
    )? {
        let c = configuration
        let base: Int
        var keys: [[MLXArray]] = Array(repeating: [], count: c.numHiddenLayers)
        var values: [[MLXArray]] = Array(repeating: [], count: c.numHiddenLayers)
        if let restored = s.restored {
            base = restored.contextCount
            for layer in 0 ..< c.numHiddenLayers {
                keys[layer].append(restored.keys[layer])
                values[layer].append(restored.values[layer])
            }
        } else if let paged = s.paged, paged.written > 0 {
            base = paged.written
            for (layer, row) in paged.rows.enumerated() {
                let snapshot = row.snapshot()
                keys[layer].append(snapshot.keys)
                values[layer].append(snapshot.values)
            }
        } else {
            base = 0
        }
        guard base + s.pendingRows + 1 == count else { return nil }
        if !s.pending.isEmpty {
            let rows = concatenated(s.pending, axis: 1)[0]
            for layer in 0 ..< c.numHiddenLayers {
                let context = model.contextKV(layer: layer, features: rows, start: base)
                keys[layer].append(context.keys)
                values[layer].append(context.values)
            }
        }
        var outKeys: [MLXArray] = []
        var outValues: [MLXArray] = []
        for layer in 0 ..< c.numHiddenLayers {
            guard !keys[layer].isEmpty else { return nil }
            var k = keys[layer].count == 1 ? keys[layer][0] : concatenated(keys[layer], axis: 2)
            var v =
                values[layer].count == 1 ? values[layer][0] : concatenated(values[layer], axis: 2)
            let rows = c.isLocal(layer: layer) ? min(count - 1, c.slidingWindowSize) : count - 1
            guard k.dim(2) >= rows else { return nil }
            if k.dim(2) > rows {
                k = k[0..., 0..., (k.dim(2) - rows)..., 0...]
                v = v[0..., 0..., (v.dim(2) - rows)..., 0...]
            }
            outKeys.append(inkling2DSparkCopy(k.asType(model.dtype)))
            outValues.append(inkling2DSparkCopy(v.asType(model.dtype)))
        }
        return (outKeys, outValues)
    }

    public func capturePrefixCheckpoint(
        requestState: any CBv2MTPRequestState, targetInputCount count: Int
    ) -> (any CBv2MTPPrefixCheckpoint)? {
        guard !refusesPrefixCaptureForTesting,
            let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released,
            let context = s.prefixContext, s.promptOnly, s.stagedInputCount == 0, !s.seedWritten,
            count >= 4, count < context.count, s.committedInputCount == count,
            let frontier = s.frontier,
            let descriptors = prefixCheckpointTensorDescriptors(targetInputCount: count),
            let bound = allocationBound(descriptors)
        else { return nil }
        // Identity evidence: the observed inputs are exactly the prompt prefix.
        let observed =
            s.observedTokens.isEmpty
            ? [Int32]()
            : concatenated(s.observedTokens, axis: 1).asType(.int32).asArray(Int32.self)
        guard observed.count == count, observed == Array(context.prefix(count)),
            let boundary = boundaryContext(s, count: count)
        else { return nil }
        let value = Inkling2DSparkPrefixCheckpoint(
            owner: self, targetInputCount: count, keys: boundary.keys, values: boundary.values,
            frontier: inkling2DSparkCopy(frontier.asType(model.dtype)),
            tokens: MLXArray(observed).reshaped(1, count), expectedTokens: observed,
            materializedBytes: bound)
        guard matches(value.evaluationTargets, descriptors) else { return nil }
        return value
    }

    public func encodePrefixCheckpoint(_ checkpoint: any CBv2MTPPrefixCheckpoint) -> [MLXArray]? {
        guard let value = checkpoint as? Inkling2DSparkPrefixCheckpoint, value.owner === self,
            let descriptors = prefixCheckpointTensorDescriptors(
                targetInputCount: value.targetInputCount),
            matches(value.evaluationTargets, descriptors)
        else { return nil }
        return value.evaluationTargets
    }

    public func decodePrefixCheckpoint(tensors: [MLXArray], prefixTokens: [Int])
        -> (any CBv2MTPPrefixCheckpoint)?
    {
        let c = configuration
        let count = prefixTokens.count
        guard let descriptors = prefixCheckpointTensorDescriptors(targetInputCount: count),
            matches(tensors, descriptors), let bound = allocationBound(descriptors),
            prefixTokens.allSatisfy({
                $0 >= 0 && $0 < c.paddedVocabularySize && $0 <= Int(Int32.max)
            })
        else { return nil }
        let layers = c.numHiddenLayers
        let tokens = tensors[2 * layers + 1]
        let decoded = tokens.asArray(Int32.self)
        guard decoded.count == count, zip(decoded, prefixTokens).allSatisfy({ Int($0) == $1 })
        else { return nil }
        // Bound to THIS drafter; serialized identifiers are never trusted.
        return Inkling2DSparkPrefixCheckpoint(
            owner: self, targetInputCount: count,
            keys: (0 ..< layers).map { tensors[2 * $0] },
            values: (0 ..< layers).map { tensors[2 * $0 + 1] },
            frontier: tensors[2 * layers], tokens: tokens, expectedTokens: decoded,
            materializedBytes: bound, imported: true)
    }

    public func restorePrefixCheckpoint(_ checkpoint: any CBv2MTPPrefixCheckpoint)
        -> (any CBv2MTPRequestState)?
    {
        guard let value = checkpoint as? Inkling2DSparkPrefixCheckpoint, value.owner === self,
            let descriptors = prefixCheckpointTensorDescriptors(
                targetInputCount: value.targetInputCount),
            matches(value.evaluationTargets, descriptors)
        else { return nil }
        // A NEW mutable owner; donor and imported payload are never mutated.
        let s = State(owner: ObjectIdentifier(self))
        s.committedInputCount = value.targetInputCount
        s.frontier = inkling2DSparkCopy(value.frontier)
        s.restored = value
        s.observedTokens = [inkling2DSparkCopy(value.tokens)]
        return s
    }

    /// Writes a restored checkpoint's context into a fresh paged state. Older
    /// sliding positions outside the window are written as zeros so every
    /// layer's cursor reaches the same position; they are never read.
    func writeRestored(_ checkpoint: Inkling2DSparkPrefixCheckpoint, into s: State) {
        guard let paged = s.paged else { return }
        precondition(paged.written == 0, "Inkling2 DSpark restore into a used state")
        let c = configuration
        for (layer, cache) in paged.caches.enumerated() {
            let keys = checkpoint.keys[layer]
            let values = checkpoint.values[layer]
            let retained = keys.dim(2)
            var skipped = checkpoint.contextCount - retained
            while skipped > 0 {
                let n = min(skipped, Self.contextChunk)
                let zeros = MLXArray.zeros(
                    [1, c.numKeyValueHeads, n, c.headDim], dtype: model.dtype)
                s.roots += cache.appendTrusted(keys: zeros, values: zeros)
                skipped -= n
            }
            for start in stride(from: 0, to: retained, by: Self.contextChunk) {
                let end = min(start + Self.contextChunk, retained)
                s.roots += cache.appendTrusted(
                    keys: keys[0..., 0..., start ..< end, 0...],
                    values: values[0..., 0..., start ..< end, 0...])
            }
        }
    }

    private func matches(_ arrays: [MLXArray], _ descriptors: [CBv2CheckpointTensorDescriptor])
        -> Bool
    {
        arrays.count == descriptors.count
            && zip(arrays, descriptors).allSatisfy {
                $0.shape == $1.shape && $0.dtype == $1.dtype.mlxDType
            }
    }

    private func allocationBound(_ descriptors: [CBv2CheckpointTensorDescriptor]) -> Int? {
        var result = 0
        for descriptor in descriptors {
            guard
                let bytes = try? Memory.allocationFootprintUpperBound(
                    byteCount: descriptor.byteCount)
            else { return nil }
            let next = result.addingReportingOverflow(bytes)
            guard !next.overflow else { return nil }
            result = next.partialValue
        }
        return result
    }
}
