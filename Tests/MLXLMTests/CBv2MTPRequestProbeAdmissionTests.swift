import MLX
import Testing

@testable import MLXLMCommon

/// The first-token deadline projection for request-probe mode (`CBv2MTPConfig.requestProbeDepth`): the projection
/// cannot run the controller, so it charges each decoding row a width. Request-probe mode chooses its depth from
/// measured goodput, and admission prices a projected step at the measured mean step (whose time already includes
/// the deep rounds that ran), so the projection charges what the adaptive depth-1 controller charges: raising the cap
/// must not raise any projection. Host-only: the driver's model and drafter never run a graph.
@Suite("CBv2MTP request probe admission")
struct CBv2MTPRequestProbeAdmissionTests {
    private static func driver(cap: Int, probe: Int?, envelope: Int) throws -> CBv2MTPRoundDriver {
        let model = ProbeAdmissionModel()
        return try #require(
            CBv2MTPRoundDriver.build(
                model: model, drafter: ProbeAdmissionDrafter(target: model),
                config: CBv2MTPConfig(
                    enabled: true, maxDraftTokens: cap, maxSpeculativeBatch: 8,
                    verificationMode: .automatic, maxAutomaticRectangularTokens: envelope,
                    requestProbeDepth: probe)))
    }

    /// Gemma 4's production geometry: 2,048-token chunks, a 4,096-token step budget.
    private static func scheduler(maxConcurrent: Int) -> SchedulerV2 {
        SchedulerV2(
            config: CBv2SchedulerConfig(
                maxConcurrentRequests: maxConcurrent, maxBatchedTokensPerStep: 4096,
                prefillChunkSize: 2048, maxConcurrentPartialPrefills: 1, maxWaiting: 64))
    }

    /// The decode steps, by shape, that the projection charges before a `prompt`-token request's first token,
    /// with `running` requests decoding, wired to `driver` by the engine's own `wireSpeculationProjection`.
    private static func projectedDecodeSteps(
        driver: CBv2MTPRoundDriver, running: Int, maxConcurrent: Int, runningMaxTokens: Int,
        prompt: Int
    ) throws -> [CBv2DecodeStepShape: Int] {
        let scheduler = scheduler(maxConcurrent: maxConcurrent)
        for _ in 0 ..< running {
            try scheduler.enqueue(
                CBv2SchedFixtures.request(prompt: [1], maxTokens: runningMaxTokens))
        }
        let plan = scheduler.plan()
        for (id, _) in plan.assignments {
            guard let rec = scheduler.record(for: id),
                rec.numComputedTokens == rec.effectiveTokenCount
            else { continue }
            scheduler.markPendingSamples(ids: [id])
            scheduler.recordSampled(id: id, token: (rec.tokens.last ?? 0) + 1)
        }
        scheduler.speculationPlanner = { _ in 0 }
        EngineLoopV2.wireSpeculationProjection(scheduler, to: driver)
        let target = CBv2SchedFixtures.request(prompt: Array(1 ... prompt), maxTokens: 64)
        try scheduler.enqueue(target)
        guard
            case .bounded(let work, _) = scheduler.firstTokenWorkProjection(
                for: target.id, mirrorsSpeculationDepth: true)
        else {
            Issue.record("unbounded projection")
            return [:]
        }
        return try #require(work.decodeSteps)
    }

    /// 1-5 decoding rows beside a 6,000-token prompt, on M3-M5 (envelope 8) and M1/M2 (envelope 4).
    @Test(arguments: [8, 4], [1, 2, 3, 4, 5])
    func aLongPromptBesideDecodingRowsIsProjectedAsAtDepthOne(envelope: Int, running: Int) throws {
        let probe = try Self.driver(cap: 4, probe: 2, envelope: envelope)
        let depthOne = try Self.driver(cap: 1, probe: nil, envelope: envelope)
        let withProbe = try Self.projectedDecodeSteps(
            driver: probe, running: running, maxConcurrent: 8, runningMaxTokens: 400, prompt: 6000)
        let atDepthOne = try Self.projectedDecodeSteps(
            driver: depthOne, running: running, maxConcurrent: 8, runningMaxTokens: 400,
            prompt: 6000)
        #expect(withProbe == atDepthOne)
    }

    /// A request queued behind full slots of width 1 or 2 (the 199-step case).
    @Test(arguments: [(8, 1), (8, 2), (4, 1), (4, 2)])
    func aRequestQueuedBehindFullSlotsIsProjectedAsAtDepthOne(envelope: Int, running: Int) throws {
        let probe = try Self.driver(cap: 4, probe: 2, envelope: envelope)
        let depthOne = try Self.driver(cap: 1, probe: nil, envelope: envelope)
        let withProbe = try Self.projectedDecodeSteps(
            driver: probe, running: running, maxConcurrent: running, runningMaxTokens: 200,
            prompt: 1000)
        let atDepthOne = try Self.projectedDecodeSteps(
            driver: depthOne, running: running, maxConcurrent: running, runningMaxTokens: 200,
            prompt: 1000)
        #expect(withProbe == atDepthOne)
    }

    /// Only request-probe mode changes the charge: the incremental controller and fixed depths keep their cap.
    @Test func otherModesKeepTheirConfiguredBound() throws {
        #expect(try Self.driver(cap: 4, probe: 2, envelope: 8).projectionDraftTokenUpperBound == 1)
        #expect(
            try Self.driver(cap: 4, probe: nil, envelope: 8).projectionDraftTokenUpperBound == 4)
        let model = ProbeAdmissionModel()
        let fixed = try #require(
            CBv2MTPRoundDriver.build(
                model: model, drafter: ProbeAdmissionDrafter(target: model),
                config: CBv2MTPConfig(enabled: true, maxDraftTokens: 4, fixedDraftTokens: 3)))
        #expect(fixed.projectionDraftTokenUpperBound == 3)
    }

    /// `plannedDepthSteps` never steps back: a plan is counted once, at its final depth, when the next plan begins.
    /// A tail clamp moves `depthSelections` (the plan's live view) back, but not the counter.
    @Test func plannedDepthStepsCountEachPlanOnceAtItsFinalDepth() throws {
        let model = ProbeAdmissionModel()
        let driver = try #require(
            CBv2MTPRoundDriver.build(
                model: model, drafter: ProbeAdmissionDrafter(target: model),
                config: CBv2MTPConfig(
                    enabled: true, maxDraftTokens: 4, fixedDraftTokens: 2,
                    verificationMode: .automatic, maxAutomaticRectangularTokens: 8)))
        let rows = [CBv2RequestID(1)]
        driver.beginPlan(plannedDecodeRows: 1, canSpeculate: true, rowIDs: rows)
        #expect(driver.planDecision.depth == 2)
        #expect(driver.metricsSnapshot().plannedDepthSteps.isEmpty)
        driver.beginPlan(plannedDecodeRows: 1, canSpeculate: true, rowIDs: rows)
        #expect(driver.metricsSnapshot().plannedDepthSteps == [2: 1])
        #expect(driver.metricsSnapshot().depthSelections == [2: 2])
        driver.clampPlanDepth(to: 0, reason: "tail_depth")
        // The live view moved back (2: 2 -> 2: 1); the counter is unchanged until the plan is final.
        #expect(driver.metricsSnapshot().depthSelections == [2: 1, 0: 1])
        #expect(driver.metricsSnapshot().plannedDepthSteps == [2: 1])
        driver.beginPlan(plannedDecodeRows: 1, canSpeculate: true, rowIDs: rows)
        #expect(driver.metricsSnapshot().plannedDepthSteps == [2: 1, 0: 1])
    }
}

private final class ProbeAdmissionCapture: CBv2MTPPreparedCapture {}
private final class ProbeAdmissionDrafter: CBv2MTPDrafter {
    let mtpTargetIdentity: ObjectIdentifier?
    let supportsTargetPrefixAcceptance = true
    init(target: ProbeAdmissionModel) { mtpTargetIdentity = ObjectIdentifier(target) }
    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture { ProbeAdmissionCapture() }
    func draftStep(tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture)
        -> (tokens: MLXArray, hidden: MLXArray)
    { (tokens, hidden) }
}
private final class ProbeAdmissionModel: CBv2MTPSteppableModel {
    let mtpCaptureLayers: CBv2MTPCaptureLayers? = .init(full: 0, sliding: 0)
    var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(self) }
    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        fatalError("admission projection tests do not execute model graphs")
    }
    func forwardWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache])
        -> (logits: MLXArray, lastHidden: MLXArray)
    {
        fatalError("admission projection tests do not execute model graphs")
    }
}
