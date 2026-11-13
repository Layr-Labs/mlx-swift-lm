import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@testable import MLXVLM

/// The witnesses own accounting only; the strict-loaded model, KV, sampler and engine are real.
private final class BudgetNativeOwner: CBv2ProcessMemoryOwner, @unchecked Sendable {
    enum Failure: Error { case invalidCharge }
    private let lock = NSLock()
    private var charge: UInt64 = 0
    private var materialized: UInt64 = 0
    private var retired = false
    var bytes: UInt64 { lock.withLock { charge } }
    func replaceCharge(_ bytes: UInt64) throws {
        try lock.withLock {
            guard !retired, bytes >= materialized, bytes <= 128 << 20 else {
                throw Failure.invalidCharge
            }
            charge = bytes
        }
    }
    func recordMaterialization(_ bytes: UInt64) throws {
        try lock.withLock {
            guard !retired, bytes >= materialized, bytes <= charge else {
                throw Failure.invalidCharge
            }
            materialized = bytes
        }
    }
    func withdrawCoverage(_ bytes: UInt64) throws {
        try lock.withLock {
            guard bytes <= materialized else { throw Failure.invalidCharge }
            materialized -= bytes
        }
    }
    func retire() { lock.withLock { retired = true } }
}

private final class BudgetNativeStore: CBv2NativeCompletePrefixCache, @unchecked Sendable {
    let identity = CBv2CompleteCheckpointIdentity(
        modelAggregateHash: "budget-tiny-synthetic",
        promptContractID: "fixture", buildID: "test", numericsFingerprint: "native")
    func acceptsCheckpoint(position: Int, packedBytes: Int) -> Bool { false }
    func takeStaged(
        requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?,
        maximumSequenceLength: Int
    ) -> CBv2StagedCompleteCheckpoint? { nil }
    func donate(
        _ source: CBv2CompleteCheckpointExport, requestID: CBv2RequestID?, tokens: [Int],
        cacheSalt: String?, completion: @escaping @Sendable ([Int]) -> Void
    ) {
        source.close()
        completion([])
    }
    func close() {}
    func closeAndWait() async {}
}

private final class BudgetMediaPermit: MiMoV26MediaWorkReservation, @unchecked Sendable {
    private let lock = NSLock()
    private var fingerprint: String?
    private var charge = 0
    private var retired = 0
    private var retainedFaults: [MiMoV26FailedMediaWork] = []
    var bytes: Int { lock.withLock { charge } }
    var retirements: Int { lock.withLock { retired } }
    func admit(_ plan: MiMoV26MultimodalPlan, bytes: Int) throws {
        try lock.withLock {
            guard fingerprint == nil, bytes > 0 else {
                throw BudgetNativeOwner.Failure.invalidCharge
            }
            fingerprint = plan.preparationSHA256
            charge = bytes
        }
    }
    func validate(plan: MiMoV26MultimodalPlan) throws {
        try lock.withLock {
            guard fingerprint == plan.preparationSHA256, charge > 0, retired == 0 else {
                throw BudgetNativeOwner.Failure.invalidCharge
            }
        }
    }
    func retainAfterFailedDrain(_ work: MiMoV26FailedMediaWork) {
        lock.withLock { retainedFaults.append(work) }
    }
    func retire() throws {
        try lock.withLock {
            guard retired == 0, retainedFaults.isEmpty else {
                throw BudgetNativeOwner.Failure.invalidCharge
            }
            retired = 1
            charge = 0
        }
    }
}

@Suite("Reasoning budget on a strict-loaded native MiMo", .tags(.integration), .serialized)
struct CBv2ReasoningBudgetNativeMiMoTests {
    @Test("A protected native MTP engine enforces N through ordinary target decoding and retires")
    func nativeOwnerBudget() async throws {
        typealias Fixture = MiMoV26TinyCheckpoint
        var fields = Fixture.nativeFields()
        // The native historical codec requires head widths >=64 and unequal K/V.
        fields["head_dim"] = 128
        fields["v_head_dim"] = 64
        fields["swa_head_dim"] = 128
        fields["swa_v_head_dim"] = 64
        let root = try Fixture.writeNativeBundle(
            to: Fixture.temporaryRoot("budget"), fields: fields)
        defer { try? FileManager.default.removeItem(at: root) }
        let session = try MiMoV26SerialLoadSession(plan: Fixture.preflight(root))
        let prepared = try await MiMoV26ModelFactory.prepare(
            request: session.request,
            configuration: .init(directory: root),
            tokenizerLoader: Fixture.Loader(tokenizer: .init()))
        let construction = NativeConstructionWork()
        defer {
            if construction.snapshot.isRetainedFault { _ = Unmanaged.passRetained(construction) }
        }
        let container = try await MiMoV26ModelFactory.loadContainer(
            session: session,
            reservation: Fixture.Permit(session.request), prepared: prepared,
            retaining: construction)
        try await construction.acknowledgeContainerAdoption(container)
        let process = BudgetNativeOwner()
        let store = BudgetNativeStore()
        let engine = try await MiMoV26ModelFactory.withNativeConstruction(
            container: container, retaining: construction
        ) { model, scope in
            let binding = try model.makeCBv2Binding(enableMTP: true)
            _ = try binding.adapter.probeNativeKVTypes(retaining: scope)
            let metadata = try model.nativeCompletePrefixMetadata(
                binding: binding, retaining: scope)
            let resources = try model.makeManagedMediaCompletePrefixExecutionResources(
                binding: binding,
                bytesCapacity: 32 << 20, expectedMetadata: metadata, completePrefixCache: store,
                processMemoryOwner: process,
                limits: .init(
                    maximumMedia: 1, maximumVideoFrames: 1, maximumPromptTokens: 128,
                    maximumMetadataBytes: 1 << 20, maximumMetadataNodes: 1000,
                    maximumMetadataDepth: 16,
                    pixels: .init(
                        maximumInputElements: 10000, maximumOutputElements: 10000,
                        maximumWorkingBytes: 1 << 20),
                    vision: .init(maximumPatches: 64, maximumAttentionScoreElements: 32768)),
                retaining: scope)
            let engine = EngineV2(
                model: binding.adapter, layerKinds: binding.adapter.layerKinds,
                backend: resources.backend, cacheProvider: resources.cacheProvider,
                schedulerConfig: .init(
                    maxConcurrentRequests: 1, maxBatchedTokensPerStep: 16,
                    prefillChunkSize: 16, maxConcurrentPartialPrefills: 1, enablePrefixCache: true),
                completePrefixCache: store, mtpDrafter: binding.assistant,
                mtpConfig: .init(
                    enabled: true, maxDraftTokens: 3, maxSpeculativeBatch: 1,
                    fixedDraftTokens: 3, verificationMode: .serialTarget),
                processMemoryOwner: process, nativeCompletionTracking: true,
                nativeExecutionContract: resources.contract, reasoningBudgetsEnabled: true)
            try scope.retainOwner(engine)
            return engine
        }
        guard case .completed(let completion) = construction.snapshot.disposition else {
            Issue.record("native construction did not finish")
            return
        }
        try await construction.sealForPublication(completion)
        #expect(engine.nativeCompletionFault == nil)
        func retire() async throws {
            guard case .quiescent(let receipt) = await engine.shutdownReportingNativeCompletion()
            else {
                _ = Unmanaged.passRetained(engine)
                _ = Unmanaged.passRetained(container)
                _ = Unmanaged.passRetained(construction)
                throw BudgetNativeOwner.Failure.invalidCharge
            }
            #expect(receipt.engineID == engine.nativeShutdownEngineID)
            try await container.perform { context in
                let model = try #require(context.model as? MiMoV26LoadedModel)
                try model.releaseNativeCompletePrefixAfterNativeRetirement(receipt)
            }
            #expect(process.bytes == 0)
        }
        do {
            let control = CBv2Request(
                id: .init(810), promptTokens: [20, 21],
                sampling: .init(temperature: 0), maxTokens: 12, prefixCacheEnabled: false)
            let controlSubmission = try engine.submitWithNativeRetirement(control)
            let baseline = await cbv2SchedCollect(controlSubmission.events)
            await controlSubmission.retirement.wait()
            #expect(baseline.finishReason == .length)
            #expect((baseline.usage?.timing.mtpRounds ?? 0) > 0)
            #expect(!baseline.tokens.prefix(2).contains(126))
            #expect(!baseline.tokens.prefix(2).contains(127))
            var request = control
            request.id = .init(811)
            request.maxTokens = 6
            request.reasoningBudget = try .init(
                maximumTokens: 2, openTokenID: 126, closeTokenID: 127,
                startsInsideThought: true)
            let submission = try engine.submitWithNativeRetirement(request)
            let result = await cbv2SchedCollect(submission.events)
            await submission.retirement.wait()
            #expect(result.finishReason == .length)
            #expect(result.tokens.count == 6)
            #expect(Array(result.tokens.prefix(2)) == Array(baseline.tokens.prefix(2)))
            #expect(result.tokens.dropFirst(2).first == 127)
            #expect(result.usage?.completionTokens == 6)
            #expect(result.usage?.timing.mtpRounds == 0)
            #expect(result.usage?.timing.mtpProposed == 0)

            for cancel in [false, true] {
                let permit = BudgetMediaPermit()
                let input = MiMoV26MultimodalInput(
                    messages: [
                        .init(
                            role: .user,
                            content: [
                                .image(
                                    .init(
                                        height: 4, width: 4,
                                        planarRGB: Array(repeating: 0.5, count: 48))),
                                .text("describe"),
                            ])
                    ],
                    additionalContext: ["enable_thinking": true],
                    maximumOutputTokens: cancel ? 128 : 6)
                var media = try await container.perform { context in
                    let model = try #require(context.model as? MiMoV26LoadedModel)
                    return try model.prepareManagedDecodedMedia(
                        input, engine: engine,
                        authorize: { plan, bytes in
                            try permit.admit(plan, bytes: bytes)
                            return permit
                        },
                        retire: { _ in try permit.retire() }, isCancelled: { false })
                }
                media.id = .init(cancel ? 813 : 812)
                media.sampling = .init(temperature: 0, logitBias: [126: 50000, 21: 5000])
                media.reasoningBudget = try .init(
                    maximumTokens: 2, openTokenID: 126, closeTokenID: 127,
                    startsInsideThought: false)
                let mediaSubmission = try engine.submitWithNativeRetirement(media)
                var mediaTokens: [Int] = []
                var mediaFinish: CBv2FinishReason?
                var mediaUsage: CBv2Usage?
                var requestedCancellation = false
                for await event in mediaSubmission.events {
                    switch event {
                    case .delta(_, let tokens, _):
                        mediaTokens += tokens
                        if cancel && !requestedCancellation && !tokens.isEmpty {
                            requestedCancellation = true
                            engine.cancel(media.id)
                        }
                    case .finished(let reason, let usage):
                        mediaFinish = reason
                        mediaUsage = usage
                    }
                }
                await mediaSubmission.retirement.wait()
                if cancel {
                    #expect(requestedCancellation)
                    #expect(mediaTokens.first == 126)
                    #expect(mediaTokens.count < 128)
                    #expect(mediaFinish == .cancelled)
                } else {
                    #expect(mediaTokens == [126, 21, 21, 127, 21, 21])
                    #expect(mediaFinish == .length)
                }
                #expect(mediaUsage?.completionTokens == mediaTokens.count)
                #expect(permit.retirements == 1)
                #expect(permit.bytes == 0)
                #expect(throws: (any Error).self) { try engine.submitWithNativeRetirement(media) }
            }

        } catch {
            try await retire()
            throw error
        }
        try await retire()
    }
}
