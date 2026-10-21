import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// The Super assistant's admission split on synthetic weights: a fixed
/// declaration plus per-token buffers the engine prices for each request's own
/// reserved tokens. At a request's full bound the split covers the
/// configured-window declaration it replaces, a shorter request pays less, the
/// linear price shared reservations use bounds the projection, and EngineV2
/// installs both parts.
@Suite(.tags(.integration))
struct NemotronSuperMTPSequenceChargeTests {
    typealias Fixture = NemotronSuperMTPFixture
    static let limits = CBv2MTPAllocationLimits(maximumPrefillTokens: 16, maximumDraftTokens: 3)
    static let bound = 1_000

    static func resolved(_ spec: CBv2MTPBoundedAllocationSpec?, policy: AllocationFootprintPolicy)
        throws -> CBv2ResolvedMTPAdmission
    {
        guard
            case .bounded(let value) = CBv2MTPBoundedAdmission.resolve(
                spec: spec, limits: limits, policy: policy)
        else { throw CBv2MTPAdmissionRefusal.invalidDeclaration }
        return value
    }

    @Test(arguments: [
        NemotronSuperMTPAssistant.Settings.Verification.serial, .rectangularExact,
    ])
    func splitCoversTheWindowAtTheRequestBound(
        _ verification: NemotronSuperMTPAssistant.Settings.Verification
    ) throws {
        let stack = try Fixture.stack(
            kind: .random, depth: 3, verification: verification, maximumSequenceLength: Self.bound)
        let assistant = stack.assistant
        let policy = try #require(Memory.allocationFootprintPolicy())
        let window = try Self.resolved(
            assistant.windowRequestAllocation(limits: Self.limits), policy: policy)
        let fixed = try Self.resolved(
            assistant.boundedRequestAllocation(limits: Self.limits), policy: policy)
        let specs = try #require(assistant.boundedRequestTokenAllocation(limits: Self.limits))
        let projection = try #require(
            CBv2AuxiliaryAllocationProjection(policy: policy, buffers: specs))
        // One request: its fixed carries, the owner's round work and its
        // per-token state cover every term of the window declaration at the
        // full bound; splitting buffers adds at most one allocator rounding
        // per piece.
        #expect(fixed.workingBytes == 0 && fixed.ownerBytes > 0)
        let owned = fixed.fixedBytesPerRequest + fixed.ownerBytes
        let full = owned + (try #require(projection.bytes(forTokens: Self.bound)))
        let extra = try #require(policy.maximumExtraBytes)
        #expect(full + 16 * extra >= window.fixedBytesPerRequest)
        #expect(full <= window.fixedBytesPerRequest + 64 << 10 + 16 * extra)
        // Shorter requests pay for their own length only.
        let short = owned + (try #require(projection.bytes(forTokens: 64)))
        #expect(short < window.fixedBytesPerRequest)
        let rate = try #require(projection.linearBytesPerToken)
        let slack = try #require(projection.linearSlackBytes)
        for tokens in [1, 2, 63, 255, 256, 257, 511, 999, Self.bound] {
            let bytes = try #require(projection.bytes(forTokens: tokens))
            #expect(bytes <= tokens * rate + slack, "\(tokens)")
            #expect(
                bytes >= tokens * specs.reduce(0) { $0 + $1.bytesPerToken * $1.allocationCount })
        }
        print(
            "SUPER_MTP_SEQUENCE_CHARGE \(verification): window \(window.fixedBytesPerRequest) "
                + "fixed \(fixed.fixedBytesPerRequest) at_bound \(full) at_64 \(short) "
                + "rate \(rate) slack \(slack)")
        #expect(
            assistant.boundedRequestTokenAllocation(
                limits: .init(maximumPrefillTokens: 16, maximumDraftTokens: 4)) == nil)
    }

    /// EngineV2 installs the fixed part in its per-request charge and the
    /// rest as the auxiliary per-token ledger with its exact projection; the
    /// resolution reports both for consumers that price linearly.
    @Test func engineChargesTheAssistantPerReservedToken() async throws {
        let stack = try Fixture.stack(kind: .random, depth: 3, maximumSequenceLength: Self.bound)
        let (engine, _) = NemotronSuperMTPEngineTests.speculativeEngine(stack, depth: 3)
        guard case .bounded(let resolution)? = engine.resolvedMTPAdmission else {
            Issue.record("the speculative engine refused its MTP declaration")
            await engine.shutdown()
            return
        }
        let admission = try #require(engine.loopForTesting.capacity as? AdmissionV2)
        #expect(resolution.auxiliaryBytesPerToken > 0)
        #expect(admission.auxiliaryBytesPerToken == resolution.auxiliaryBytesPerToken)
        #expect(admission.auxiliaryTokenGranularity == 1)
        #expect(admission.auxiliaryTokenAllocationPadding == 0)
        #expect(resolution.auxiliaryLinearSlackBytes > 0)
        #expect(resolution.ownerBytes > 0 && resolution.workingBytes == 0)
        #expect(engine.resolvedFixedBytesPerRequest >= resolution.fixedBytesPerRequest)
        // 256 more reserved tokens are charged at least their history rows
        // beside the KV; the configured window is no longer charged up front.
        let row = try NemotronSuperFP8KVHistory.bytesPerToken(
            heads: 1, headDimension: 16, dtype: .float32)
        let grown =
            admission.estimatedBytes(forTokens: 257) - admission.estimatedBytes(forTokens: 1)
        #expect(grown >= 256 * row)
        let window = admission.estimatedBytes(forTokens: Self.bound)
        #expect(admission.estimatedBytes(forTokens: 1) < window - (Self.bound - 1) * row)
        await engine.shutdown()
    }

    /// A one-request owner whose depth table never drafts its maximum (`[1,
    /// 2]` drafts one token for one request) declares its bounded charge, so
    /// it admits and serves the plain greedy tokens. The declaration still
    /// refuses limits below the deepest round they state.
    @Test func aDepthTableBelowItsMaximumKeepsTheOwnerServing() async throws {
        let stack = try Fixture.stack(
            kind: .random, depth: 2, verification: .rectangularExact,
            maximumSequenceLength: Self.bound)
        func limits(roundDepth: Int?) -> CBv2MTPAllocationLimits {
            .init(
                maximumPrefillTokens: 16, maximumDraftTokens: 2, maximumWindowPositions: 2,
                maximumRoundDraftSteps: 1, maximumHistoryCopies: 2, maximumRoundDepth: roundDepth)
        }
        #expect(stack.assistant.boundedRequestAllocation(limits: limits(roundDepth: 1)) != nil)
        #expect(stack.assistant.boundedRequestAllocation(limits: limits(roundDepth: nil)) == nil)

        let prompt = [5, 9, 1, 4, 22, 7, 3, 8]
        let (plain, _) = NemotronSuperMTPEngineTests.plainEngine(stack.model)
        let expected = await NemotronSuperMTPEngineTests.collect(
            try plain.submit(CBv2Request(id: .init(1), promptTokens: prompt, maxTokens: 8)))
        await plain.shutdown()
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 64 << 20, kvDType: .float32))
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(stack.target),
            layerKinds: stack.target.cbv2LayerKinds, backend: backend,
            cacheProvider: CBv2LayerCacheBank(
                caches: stack.target.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) }),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 16, prefillChunkSize: 8,
                maxWaiting: 4),
            admissionConfig: .init(watermarkFraction: 0), mtpDrafter: stack.assistant,
            mtpConfig: .init(
                enabled: true, maxDraftTokens: 2, maxSpeculativeBatch: 1, fixedDraftTokens: 2,
                fixedDraftTokensByDecodeRows: [1, 2], verificationMode: .rectangularExact))
        guard case .bounded = engine.resolvedMTPAdmission else {
            Issue.record("the one-request owner refused its MTP declaration")
            await engine.shutdown()
            return
        }
        #expect(engine.resolvedFixedBytesPerRequest < Int.max)
        let served = await NemotronSuperMTPEngineTests.collect(
            try engine.submit(CBv2Request(id: .init(2), promptTokens: prompt, maxTokens: 8)))
        #expect(served.finish == .length)
        #expect(served.tokens == expected.tokens)
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
    }

    /// With captured verify windows the engine keeps each request's ordinary
    /// three recurrent generations and sizes the two live windows once per
    /// engine, from the depth policy's largest round, in `ownerBytes`, beside
    /// the worst-case whole-history copies over every decode width.
    @Test(arguments: [
        (1, 3, [Int]?.none, 4, 5, 4), (8, 3, [Int]?.none, 32, 40, 32),
        (4, 3, [3, 2, 2, 2, 1], 12, 12, 12), (8, 3, [3, 2, 2, 2, 1], 16, 12, 16),
        (8, 3, [3, 2, 0], 8, 6, 12),
    ])
    func capturedWindowsAreOwnerWide(
        rows: Int, depth: Int, table: [Int]?, positions: Int, generations: Int, copies: Int
    ) async throws {
        let stack = try Fixture.stack(
            kind: .random, depth: depth, verification: .rectangularExact,
            maximumSequenceLength: Self.bound)
        let config = CBv2MTPConfig(
            enabled: true, maxDraftTokens: depth, maxSpeculativeBatch: rows,
            fixedDraftTokens: depth, fixedDraftTokensByDecodeRows: table,
            verificationMode: .rectangularExact)
        #expect(CBv2MTPAllocationLimits.roundBounds(config).positions == positions)
        // Beyond each request's three: 8 x 4 - 8 x 3 at fixed depth 3 (the
        // per-request 2 x (depth + 1) it replaces), 2 x 12 - 12 at four rows
        // of the width table.
        #expect(
            CBv2MTPAllocationLimits.capturedWindowGenerations(config, perRequest: 3) == generations)
        // Each speculating row may copy its history once per draft step plus
        // once more; every other request once. A zero-depth width (speculation
        // off at three or more rows) still writes one row per request, so it
        // counts once each: 2 x (2 + 1) + 6 = 12 bounds 8 x 1 and 1 x 4 + 7.
        #expect(CBv2MTPAllocationLimits.historyCopies(config, requests: rows) == copies)
        let policy = try #require(Memory.allocationFootprintPolicy())
        let perGeneration = try stack.target.cbv2RecurrentStateSpec.allocationBytesPerGeneration(
            policy: policy)
        let (engine, _) = NemotronSuperMTPEngineTests.speculativeEngine(stack, depth: depth)
        guard case .bounded(let resolution)? = engine.resolvedMTPAdmission else {
            Issue.record("the speculative engine refused its MTP declaration")
            await engine.shutdown()
            return
        }
        #expect(
            engine.resolvedFixedBytesPerRequest
                == 3 * perGeneration + resolution.fixedBytesPerRequest)
        let windows = CBv2MTPAllocationLimits.capturedWindowGenerations(
            engine.loopForTesting.mtp!.config, perRequest: 3)
        #expect(windows == 2 * (depth + 1) - 3)
        let engineCopies = CBv2MTPAllocationLimits.historyCopies(
            engine.loopForTesting.mtp!.config,
            requests: engine.loopForTesting.scheduler.config.maxConcurrentRequests)
        #expect(engineCopies == depth + 1)
        let copy = try #require(stack.assistant.historyCopyBytes)
        #expect(resolution.ownerBytes >= windows * perGeneration + engineCopies * copy)
        await engine.shutdown()
    }
}
