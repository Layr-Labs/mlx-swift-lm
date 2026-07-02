// ContiguousKVBackend.swift
//
// The v1 `CBv2KVBackend`: per-sequence contiguous MLX buffers
// (`CBv2FullSequenceKV` / `CBv2WindowedSequenceKV` / `CBv2QuantizedSequenceKV`).
// The paged backend (workstream C) implements the same protocol behind a
// Metal kernel; the scheduler and models never see the difference.

import Foundation
import MLX

/// Configuration for `CBv2ContiguousKVBackend`.
public struct CBv2ContiguousBackendConfig: Sendable {
    /// Byte budget for all live sequence KV (admission ceiling).
    public var bytesCapacity: Int
    /// Optional KV quantization for FULL-attention layers (group size, bits).
    /// Windowed layers stay unquantized (small ring, quantization gains
    /// nothing and the modular writes would fight group boundaries).
    public var quantization: (groupSize: Int, bits: Int)?
    /// dtype assumed for admission estimates (actual allocation adopts the
    /// dtype of the first appended K/V).
    public var kvDType: DType

    public init(
        bytesCapacity: Int,
        quantization: (groupSize: Int, bits: Int)? = nil,
        kvDType: DType = .float16
    ) {
        self.bytesCapacity = bytesCapacity
        self.quantization = quantization
        self.kvDType = kvDType
    }
}

/// Factory + accounting for per-sequence contiguous KV state.
///
/// Thread-safe: the live-row registry is lock-protected (`makeSequenceState`
/// runs on the admission path while `release` runs on the engine loop).
/// `bytesInUse` is truthful — it sums the ACTUAL allocated bytes of live
/// rows (which grow by doubling), not a worst-case estimate.
public final class CBv2ContiguousKVBackend: CBv2KVBackend {

    public let config: CBv2ContiguousBackendConfig

    private let lock = NSLock()
    private var live: [ObjectIdentifier: CBv2SequenceKV] = [:]

    public init(config: CBv2ContiguousBackendConfig) {
        self.config = config
    }

    public var bytesCapacity: Int { config.bytesCapacity }

    public var bytesInUse: Int {
        lock.lock()
        defer { lock.unlock() }
        return live.values.reduce(0) { $0 + $1.byteCount }
    }

    public func makeSequenceState(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        try validate(layerKinds: layerKinds)
        guard promptLength <= maxLength else {
            throw CBv2KVError.backendIneligible(
                reason: "promptLength \(promptLength) exceeds maxLength \(maxLength)")
        }

        let needed = estimatedInitialBytes(
            layerKinds: layerKinds, promptLength: promptLength, maxLength: maxLength)
        try admit(needed: needed)

        let state = layerKinds.map { kind -> CBv2SequenceKV? in
            makeRow(kind: kind, promptLength: promptLength, maxLength: maxLength)
        }
        register(state)
        return state
    }

    public func makeSequenceState(
        adopting prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?],
        layerKinds: [CBv2LayerKind], maxLength: Int
    ) throws -> [CBv2SequenceKV?] {
        try validate(layerKinds: layerKinds)
        guard prefix.count == layerKinds.count else {
            throw CBv2KVError.backendIneligible(
                reason: "prefix count \(prefix.count) != layer count \(layerKinds.count)")
        }

        // The matched prefix length: uniform across all donated full-attention
        // layers by construction of the prefix cache.
        var matched = 0
        for (index, entry) in prefix.enumerated() {
            guard let entry else { continue }
            let kind = layerKinds[index]
            guard kind.sharesKVWithLayer == nil else {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index) is KV-shared but received a prefix snapshot")
            }
            guard case .full = kind.attention else {
                throw CBv2KVError.backendIneligible(
                    reason:
                        "layer \(index) is windowed but received a prefix snapshot (windowed layers are recomputed)"
                )
            }
            if matched == 0 { matched = entry.offset }
            guard entry.offset == matched else {
                throw CBv2KVError.backendIneligible(
                    reason:
                        "non-uniform prefix offsets (\(entry.offset) vs \(matched)) at layer \(index)"
                )
            }
        }
        guard matched <= maxLength else {
            throw CBv2KVError.backendIneligible(
                reason: "prefix offset \(matched) exceeds maxLength \(maxLength)")
        }

        let needed = estimatedInitialBytes(
            layerKinds: layerKinds, promptLength: matched, maxLength: maxLength)
        try admit(needed: needed)

        let state = layerKinds.enumerated().map { index, kind -> CBv2SequenceKV? in
            guard kind.sharesKVWithLayer == nil else { return nil }
            switch kind.attention {
            case .slidingWindow(let window):
                // Windowed layers never receive a snapshot. Reconciled
                // adoption semantics (contract `makeSequenceState(adopting:)`):
                // the engine already sliced the prefix down by
                // `cbv2RequiredRecompute`, so EVERY row starts at the uniform
                // adopted offset and the engine replays [matched, prompt)
                // through all layers.
                return CBv2WindowedSequenceKV(
                    window: window, kvHeads: kind.kvHeads, headDim: kind.headDim,
                    initialOffset: matched)
            case .full:
                let row = makeRow(kind: kind, promptLength: matched, maxLength: maxLength)!
                if let entry = prefix[index], entry.offset > 0 {
                    // Bounded one-time copy of the donated arrays into fresh
                    // sequence state (the paged backend makes this free).
                    _ = row.update(keys: entry.keys, values: entry.values)
                }
                return row
            }
        }
        register(state)
        return state
    }

    public func release(_ state: [CBv2SequenceKV?]) {
        lock.lock()
        defer { lock.unlock() }
        for row in state {
            guard let row else { continue }
            live.removeValue(forKey: ObjectIdentifier(row))
        }
    }

    // MARK: - Private

    private func makeRow(kind: CBv2LayerKind, promptLength: Int, maxLength: Int)
        -> CBv2SequenceKV?
    {
        guard kind.sharesKVWithLayer == nil else { return nil }
        switch kind.attention {
        case .slidingWindow(let window):
            return CBv2WindowedSequenceKV(
                window: window, kvHeads: kind.kvHeads, headDim: kind.headDim)
        case .full:
            if let quantization = config.quantization {
                return CBv2QuantizedSequenceKV(
                    promptLength: promptLength, maxLength: maxLength,
                    kvHeads: kind.kvHeads, headDim: kind.headDim,
                    groupSize: quantization.groupSize, bits: quantization.bits)
            }
            return CBv2FullSequenceKV(
                promptLength: promptLength, maxLength: maxLength,
                kvHeads: kind.kvHeads, headDim: kind.headDim)
        }
    }

    private func register(_ state: [CBv2SequenceKV?]) {
        lock.lock()
        defer { lock.unlock() }
        for row in state {
            guard let row else { continue }
            live[ObjectIdentifier(row)] = row
        }
    }

    private func admit(needed: Int) throws {
        let available = bytesCapacity - bytesInUse
        guard needed <= available else {
            throw CBv2KVError.capacityExhausted(needed: needed, available: max(0, available))
        }
    }

    private func validate(layerKinds: [CBv2LayerKind]) throws {
        for (index, kind) in layerKinds.enumerated() {
            if case .slidingWindow(let window) = kind.attention, window <= 0 {
                throw CBv2KVError.backendIneligible(
                    reason: "layer \(index): non-positive window \(window)")
            }
            if let source = kind.sharesKVWithLayer {
                guard source >= 0, source < layerKinds.count, source != index else {
                    throw CBv2KVError.backendIneligible(
                        reason: "layer \(index): invalid KV-share source \(source)")
                }
                guard layerKinds[source].sharesKVWithLayer == nil else {
                    throw CBv2KVError.backendIneligible(
                        reason:
                            "layer \(index): KV-share source \(source) is itself a shared layer")
                }
            }
            // The v1 contiguous backend attends through MLXFast SDPA, which
            // supports attention sinks natively — sink models are eligible.
        }
    }

    /// Bytes the initial allocations will take (full layers allocate
    /// `promptLength + 256` slots capped at maxLength; windowed layers
    /// allocate their full ring up front).
    private func estimatedInitialBytes(
        layerKinds: [CBv2LayerKind], promptLength: Int, maxLength: Int
    ) -> Int {
        let itemSize = config.kvDType.size
        var total = 0
        for kind in layerKinds where kind.sharesKVWithLayer == nil {
            switch kind.attention {
            case .slidingWindow(let window):
                total += window * kind.kvHeads * kind.headDim * itemSize * 2
            case .full:
                let slots = min(maxLength, max(1, promptLength + CBv2FullSequenceKV.initialSlack))
                if let quantization = config.quantization {
                    let groupSize = CBv2QuantizedSequenceKV.resolvedGroupSize(
                        requested: quantization.groupSize, headDim: kind.headDim)
                    // Packed weights + fp scales/biases per group.
                    let perToken =
                        kind.headDim * quantization.bits / 8
                        + 2 * (kind.headDim / groupSize) * itemSize
                    total += slots * kind.kvHeads * perToken * 2
                } else {
                    total += slots * kind.kvHeads * kind.headDim * itemSize * 2
                }
            }
        }
        return total
    }
}
