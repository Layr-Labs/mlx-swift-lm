import Foundation
import MLX
import Testing

@testable import MLXLMCommon

final class CBv2OptionalDeadlineSampler: CBv2StepSampler {
    private let base = CBv2GreedySampler()
    var onFirstConfirmation: (() -> Void)?

    func sample(logits: MLXArray, params: [CBv2SamplingParams], requestIDs: [CBv2RequestID],
                stepIndex: Int, pendingSampledTokens: MLXArray?,
                rowContext: () -> [CBv2SamplerRow]) -> MLXArray {
        base.sample(logits: logits, params: params, requestIDs: requestIDs,
            stepIndex: stepIndex, pendingSampledTokens: pendingSampledTokens, rowContext: rowContext)
    }

    func confirmSampledTokens(_ tokens: [Int], requestIDs: [CBv2RequestID]) {
        let callback = onFirstConfirmation
        onFirstConfirmation = nil
        callback?()
        base.confirmSampledTokens(tokens, requestIDs: requestIDs)
    }

    func requestDidFinish(_ id: CBv2RequestID) { base.requestDidFinish(id) }
}

final class CBv2OptionalDeadlineGate: @unchecked Sendable {
    private let lock = NSLock()
    private let semaphore = DispatchSemaphore(value: 0)
    private var _entered = false
    var entered: Bool { lock.withLock { _entered } }
    func wait() {
        lock.withLock { _entered = true }
        semaphore.wait()
    }
    func release() { semaphore.signal() }
}

final class CBv2OptionalDeadlineModel: CBv2PrefillSteppableModel {
    var widths: [Int] = [] // engine queue only

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        widths.append(tokens.dim(1))
        var output = broadcast(MLXArray([Float(0), Float(1)]).reshaped([1, 1, 2]),
            to: [tokens.dim(0), tokens.dim(1), 2])
        for cache in caches {
            let qkv = MLXArray.ones([tokens.dim(0), 1, tokens.dim(1), 64], dtype: .float32)
            let attention = cache.updateAndAttend(queries: qkv, keys: qkv, values: qkv,
                scale: 0.125, sinks: nil)
            output = output + sum(attention, axes: [1, 3]).expandedDimensions(axis: 2)
        }
        return output
    }

    func prefill(tokens: MLXArray, inputEmbeddings: MLXArray?,
                 caches: [CBv2AttendingLayerCache], requirement: CBv2PrefillRequirement) -> MLXArray {
        let output = forward(tokens: tokens, caches: caches)
        switch requirement {
        case .evaluationOnly: return output[0..., -1, 0..<1]
        case .lastPositionLogits: return output[0..., -1, 0...]
        }
    }
}

final class CBv2OptionalPrefillDeadlineFixture: @unchecked Sendable {
    let clock: CBv2SchedFakeClock
    let sampler: CBv2OptionalDeadlineSampler
    let model: CBv2OptionalDeadlineModel
    let kinds: [CBv2LayerKind]
    let backend: PagedKVBackend
    let engine: EngineV2
    let blocker: CBv2Request
    let target = CBv2Request(id: .init(91_002), promptTokens: Array(repeating: 1, count: 9),
        sampling: .init(temperature: 0), maxTokens: 1, prefixCacheEnabled: false)
    var filler: CBv2CheckpointReservation?
    var gate: CBv2OptionalDeadlineGate?

    var admission: AdmissionV2 { engine.loopForTesting.capacity as! AdmissionV2 }

    init(partial: Bool = false, mode: PagedQuantizedPrefillMode = .opportunisticSDPA) throws {
        let clock = CBv2SchedFakeClock(), sampler = CBv2OptionalDeadlineSampler()
        let model = CBv2OptionalDeadlineModel()
        let kinds = [CBv2LayerKind(attention: .full, headDim: 64, kvHeads: 1, queryHeads: 1)]
        self.clock = clock
        self.sampler = sampler
        self.model = model
        self.kinds = kinds
        blocker = CBv2Request(id: .init(91_001),
            promptTokens: Array(repeating: 1, count: partial ? 66 : 33),
            sampling: .init(temperature: 0), maxTokens: partial ? 2 : 1, prefixCacheEnabled: false)
        let backend = try PagedKVBackend(layerKinds: kinds, config: .init(
            capacityBytes: 4 << 20, dtype: .float32, maxPrefillChunk: 33,
            nominalMaxSequenceLength: 128, maxBufferLength: 4 << 20,
            segmentSizeBytes: 32_768, layerDTypes: [.float32], quantization: .init(),
            quantizedPrefillMode: mode))
        self.backend = backend
        let engine = EngineV2(model: model, layerKinds: kinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()), sampler: sampler,
            schedulerConfig: .init(maxConcurrentRequests: 2, maxBatchedTokensPerStep: 33,
                prefillChunkSize: 33, maxConcurrentPartialPrefills: 1, maxWaiting: 4,
                enablePrefixCache: false),
            loopConfig: .init(clock: clock.clock), admissionConfig: .init(watermarkFraction: 0))
        self.engine = engine
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.suspendStepExecutionAtCountForTesting = 1
        }
    }

    func holdFirstStep(fillFreeGrant: Bool = true) async throws -> AsyncStream<CBv2Event> {
        let stream = try engine.submit(blocker)
        #expect(await cbv2SchedWait {
            self.engine.loopForTesting.onEngineQueueSync { self.engine.loopForTesting.stepCount == 1 }
        })
        try engine.loopForTesting.onEngineQueueSync {
            #expect(model.widths == [33])
            if fillFreeGrant {
                #expect(backend.pool.quantizedPrefillStatistics.currentAdditionalWorkspaceBytes > 0)
                // Another persistent owner fills the remaining grant. It has
                // no GPU arrays and remains unchanged across the arrival;
                // only the real fast step's O can make the newcomer fit.
                filler = try admission.reserveTransient(
                    bytes: admission.admissibleBytesCapacity - admission.bytesReserved)
                #expect(!targetFits())
                engine.loopForTesting.publishGauges()
            }
        }
        return stream
    }

    func targetFits() -> Bool {
        admission.canGuarantee(projectedOperations: [.reserve(.init(id: target.id,
            additionalTokens: target.promptTokens.count + target.maxTokens, additionalBytes: 0))],
            projectedPhysicalBytes: { self.backend.pool.projectedPhysicalBytes(reservedTokens: $0, layerKinds: self.kinds) })
    }

    func policy(seconds: Double = 30) -> CBv2FirstTokenDeadlineAdmission {
        .init(deadline: clock.clock.now().advanced(by: .seconds(seconds)),
              conservativePrefillTokensPerSecond: 1_000, conservativeDecodeTokensPerSecond: 1_000)
    }

    func resume() {
        engine.loopForTesting.onEngineQueueSync {
            engine.loopForTesting.suspendStepExecutionAtCountForTesting = nil
        }
    }

    func close() async {
        gate?.release()
        engine.cancel(blocker.id)
        engine.cancel(target.id)
        resume()
        filler?.release()
        filler = nil
        await engine.shutdown()
    }
}
