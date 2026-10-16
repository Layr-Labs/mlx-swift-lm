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
        let full =
            fixed.fixedBytesPerRequest + (try #require(projection.bytes(forTokens: Self.bound)))
        // Every term of the window declaration is present at the full bound;
        // splitting buffers adds at most one allocator rounding per piece.
        let extra = try #require(policy.maximumExtraBytes)
        #expect(full + 16 * extra >= window.fixedBytesPerRequest)
        #expect(full <= window.fixedBytesPerRequest + 64 << 10 + 16 * extra)
        // Shorter requests pay for their own length only.
        let short = fixed.fixedBytesPerRequest + (try #require(projection.bytes(forTokens: 64)))
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
        #expect(engine.resolvedFixedBytesPerRequest >= resolution.fixedBytesPerRequest)
        // 256 more reserved tokens are charged at least their FP32 backlog rows
        // beside the KV; the configured window is no longer charged up front.
        let grown =
            admission.estimatedBytes(forTokens: 257) - admission.estimatedBytes(forTokens: 1)
        #expect(grown >= 256 * Fixture.hidden * 4)
        let window = admission.estimatedBytes(forTokens: Self.bound)
        #expect(
            admission.estimatedBytes(forTokens: 1) < window - (Self.bound - 1) * Fixture.hidden * 4)
        await engine.shutdown()
    }
}
