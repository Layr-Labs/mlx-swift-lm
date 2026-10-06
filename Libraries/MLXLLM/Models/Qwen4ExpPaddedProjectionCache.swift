import Cmlx
import Foundation
import MLX

/// Padding is exact concatenation, but its cached constants belong to one
/// live projection and one generation of its parameter descriptors. A raw
/// ObjectIdentifier is insufficient: allocator reuse after model eviction can
/// otherwise substitute a different layer's same-shaped weights on reload.
final class Qwen4ExpPaddedProjectionCache: @unchecked Sendable {
    struct Bank {
        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray
    }

    private final class Entry {
        weak var owner: AnyObject?
        let identities: [UInt]
        // Pin descriptors independently of their mutable Swift wrappers to
        // prevent descriptor-address reuse and detect _updateInternal/update.
        let sources: [MLXArray]
        let compute: DType
        let targetRows: Int
        let stream: StreamOrDevice
        let bank: Bank
        let bytes: Int

        init(owner: AnyObject, identities: [UInt], sources: [MLXArray], compute: DType,
             targetRows: Int, stream: StreamOrDevice, bank: Bank, bytes: Int) {
            self.owner = owner
            self.identities = identities
            self.sources = sources
            self.compute = compute
            self.targetRows = targetRows
            self.stream = stream
            self.bank = bank
            self.bytes = bytes
        }
    }

    private let lock = NSLock()
    private let maximumEntries: Int
    private let maximumBytes: Int
    // Injectable bucket selection makes address-collision tests deterministic;
    // the weak owner identity is always checked independently of this key.
    private let keyForOwner: (AnyObject) -> ObjectIdentifier
    private var entries: [ObjectIdentifier: Entry] = [:]

    init(maximumEntries: Int = 128, maximumBytes: Int = 64 * 1_024 * 1_024,
         keyForOwner: @escaping (AnyObject) -> ObjectIdentifier = { ObjectIdentifier($0) }) {
        precondition(maximumEntries > 0 && maximumBytes >= 0)
        self.maximumEntries = maximumEntries
        self.maximumBytes = maximumBytes
        self.keyForOwner = keyForOwner
    }

    func pruneDeadOwners() {
        lock.withLock { entries = entries.filter { $0.value.owner != nil } }
    }

    var retainedEntryCount: Int { lock.withLock { entries.count } }
    var retainedBytes: Int { lock.withLock { entries.values.reduce(0) { $0 + $1.bytes } } }

    func bank(owner: AnyObject, weight: MLXArray, scales: MLXArray, biases: MLXArray,
              compute: DType, targetRows: Int) -> Bank {
        precondition(weight.ndim == 2 && scales.ndim == 2 && biases.shape == scales.shape)
        precondition(weight.dim(0) == scales.dim(0) && targetRows > weight.dim(0))
        return lock.withLock {
            // Bounded scan, only on the prefill padding path. Dead owners never
            // keep model instances alive, and their banks leave on the next use.
            entries = entries.filter { $0.value.owner != nil }
            let key = keyForOwner(owner)
            let inputs = [weight, scales, biases]
            let stream = StreamOrDevice.default
            let identities = inputs.compactMap { source -> UInt? in
                var identity: UInt = 0
                var canCache = false
                guard _mlx_array_constant_cache_identity(&identity, &canCache, source.ctx) == 0,
                      canCache else { return nil }
                return identity
            }
            let canCache = identities.count == inputs.count
            if canCache, let entry = entries[key], entry.owner === owner,
               entry.identities == identities, entry.compute == compute,
               entry.targetRows == targetRows, entry.stream == stream {
                return entry.bank
            }
            entries.removeValue(forKey: key)
            // Never retain compile/grad/vmap tracers. A separate context also
            // prevents later Swift parameter-wrapper updates changing a bank.
            let sources = inputs.map { source -> MLXArray in
                var context = mlx_array_new()
                mlx_array_set(&context, source.ctx)
                return MLXArray(context)
            }
            let padding = targetRows - weight.dim(0)
            let bank = Bank(
                weight: concatenated([sources[0], MLXArray.zeros(
                    [padding, weight.dim(1)], dtype: weight.dtype, stream: stream)], axis: 0, stream: stream),
                scales: concatenated([sources[1], MLXArray.zeros(
                    [padding, scales.dim(1)], dtype: compute, stream: stream)], axis: 0, stream: stream),
                biases: concatenated([sources[2], MLXArray.zeros(
                    [padding, biases.dim(1)], dtype: compute, stream: stream)], axis: 0, stream: stream))
            let arrays = sources + [bank.weight, bank.scales, bank.biases]
            let bytes = arrays.reduce(0) { total, array in
                let (sum, overflow) = total.addingReportingOverflow(array.nbytes)
                return overflow ? Int.max : sum
            }
            if canCache && bytes <= maximumBytes {
                // Capacity pressure affects reuse only, never projection math.
                // Clear the bounded bank instead of introducing an unbounded
                // recency queue or pinning inactive models through ownership.
                let retained = entries.values.reduce(0) { $0 + $1.bytes }
                if entries.count >= maximumEntries || retained > maximumBytes - bytes {
                    entries.removeAll(keepingCapacity: true)
                }
                entries[key] = Entry(owner: owner, identities: identities, sources: sources,
                    compute: compute, targetRows: targetRows, stream: stream, bank: bank, bytes: bytes)
            }
            return bank
        }
    }
}
