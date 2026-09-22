import Foundation
import MLXLLM
import MLXLMCommon

/// No model owner and no disk reads. Mutable setup ends before the engine is
/// exposed; request adoption/donation run on its execution queue.
final class DiffusionGemmaPrefixPersistence: @unchecked Sendable {
    let store: any CBv2NativeBlockPrefixCache
    let codec: DiffusionGemmaPersistentPrefixCodec
    let chunkSize: Int
    weak var engine: CBv2NativeBlockEngine?

    init(store: any CBv2NativeBlockPrefixCache, codec: DiffusionGemmaPersistentPrefixCodec, chunkSize: Int) {
        self.store = store; self.codec = codec; self.chunkSize = chunkSize
    }

    func take(request: CBv2Request, identity: DiffusionGemmaPrefixIdentity,
              geometry: DiffusionGemmaPrefillGeometry? = nil)
        -> (checkpoint: DiffusionGemmaPrefixCheckpoint?, failed: Bool) {
        guard let receipt = request.prefixCacheReceiptID else { return (nil, false) }
        let (maximum, overflow) = request.promptTokens.count.addingReportingOverflow(request.maxTokens)
        guard !overflow, let staged = store.takeNativeStaged(requestID: receipt,
            tokens: request.promptTokens, cacheSalt: request.checkpointCacheSalt, maximumSequenceLength: maximum)
        else { return (nil, false) }
        do {
            guard identity.mediaIdentity == nil || geometry?.permitsRestore(position: staged.manifest.position) == true
            else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
            return (try codec.adopt(staged, prefixIdentity: identity), false)
        }
        catch { staged.close(); return (nil, true) }
    }

    func accepts(_ checkpoint: DiffusionGemmaPrefixCheckpoint, request: CBv2Request) -> Bool {
        if let media = request.multimodal {
            guard let geometry = try? DiffusionGemmaPrefillGeometry(
                promptCount: request.promptTokens.count, chunkSize: chunkSize, spans: media.spans),
                geometry.permitsPersistentCapture(position: checkpoint.tokenCount) else { return false }
        }
        return (request.multimodal != nil || checkpoint.tokenCount % chunkSize == 0)
            && store.acceptsCheckpoint(position: checkpoint.tokenCount, packedBytes: checkpoint.retainedBytes)
    }

    func donate(_ checkpoint: DiffusionGemmaPrefixCheckpoint, request: CBv2Request) {
        guard let engine, accepts(checkpoint, request: request) else { return }
        do {
            // The LRU may evict this snapshot while I/O runs. Fund independent
            // ownership BEFORE the request/shared-cache reservation can retire.
            let lease = try engine.reserveNativeCheckpoint(bytes: checkpoint.physicalRetainedBytes)
            let donor = DiffusionGemmaCheckpointDonor(checkpoint: checkpoint, lease: lease)
            do {
                let source = try codec.export(checkpoint, chunkSize: chunkSize, engine: engine)
                store.donate(source, requestID: request.prefixCacheReceiptID, tokens: request.promptTokens,
                    cacheSalt: request.checkpointCacheSalt) { [donor] _ in donor.close() }
            } catch { donor.close() }
        } catch {
            // Optional cache capacity must not turn successful generation into
            // a failure or spend beyond the slot grant.
        }
    }
}

private final class DiffusionGemmaCheckpointDonor: @unchecked Sendable {
    private let lock = NSLock()
    private var checkpoint: DiffusionGemmaPrefixCheckpoint?
    private var lease: CBv2NativeBlockCheckpointLease?
    init(checkpoint: DiffusionGemmaPrefixCheckpoint, lease: CBv2NativeBlockCheckpointLease) {
        self.checkpoint = checkpoint; self.lease = lease
    }
    func close() {
        let retiring = lock.withLock {
            checkpoint = nil
            let value = lease; lease = nil; return value
        }
        retiring?.close()
    }
    deinit { close() }
}
