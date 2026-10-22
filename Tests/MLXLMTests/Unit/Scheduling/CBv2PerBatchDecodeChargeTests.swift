// Copyright © 2026 Eigen Labs.
//
// Per-batch decode charging for first-token deadline admission: pure
// scheduler projections, the MTP depth ceiling and the conversion
// arithmetic. No MLX arrays, engine or model forward. The engine's own
// conversion is CBv2PerBatchDecodeConversionTests (Integration/Engine).

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension UnitTests {
    @Suite("Per-batch decode charging")
    struct CBv2PerBatchDecodeChargeTests {
        private static let ids = CBv2PerBatchIDs()

        private func scheduler(maxConcurrent: Int = 8, budget: Int = 64, chunk: Int = 4)
            -> SchedulerV2
        {
            SchedulerV2(
                config: CBv2SchedulerConfig(
                    maxConcurrentRequests: maxConcurrent, maxBatchedTokensPerStep: budget,
                    prefillChunkSize: chunk, maxConcurrentPartialPrefills: 1, maxWaiting: 64))
        }

        private func request(prompt: Int, maxTokens: Int) -> CBv2Request {
            CBv2Request(
                id: CBv2RequestID(Self.ids.next()), promptTokens: Array(1 ... prompt),
                maxTokens: maxTokens)
        }

        /// Admit `count` one-token prompts and confirm their first samples, so
        /// each is a decode-ready running row, as the engine loop would.
        private func addDecoders(_ scheduler: SchedulerV2, count: Int, maxTokens: Int) throws {
            for _ in 0 ..< count {
                try scheduler.enqueue(request(prompt: 1, maxTokens: maxTokens))
            }
            let plan = scheduler.plan()
            for (id, _) in plan.assignments {
                guard let rec = scheduler.record(for: id),
                    rec.numComputedTokens == rec.effectiveTokenCount
                else { continue }
                scheduler.markPendingSamples(ids: [id])
                scheduler.recordSampled(id: id, token: (rec.tokens.last ?? 0) + 1)
            }
        }

        /// Install an MTP planner whose live depth ceiling is `ceiling`.
        private func speculate(
            _ scheduler: SchedulerV2, upperBound: Int = 4, ceiling: ((Int) -> Int)?
        ) {
            scheduler.speculationPlanner = { _ in upperBound }
            scheduler.speculationDraftTokenUpperBound = upperBound
            scheduler.speculationDepthCeiling = ceiling
        }

        private func work(_ projection: CBv2FirstTokenWorkProjection) throws
            -> CBv2FirstTokenScheduledWork
        {
            guard case .bounded(let work, _) = projection else {
                Issue.record("expected a bounded projection, got \(projection)")
                throw CBv2PerBatchTestFailure()
            }
            return work
        }

        @Test("each projected step records its decode rows once, at the widest row")
        func projectionRecordsDecodeStepsNotRows() throws {
            let scheduler = scheduler()
            try addDecoders(scheduler, count: 3, maxTokens: 20)
            let target = request(prompt: 8, maxTokens: 1)
            try scheduler.enqueue(target)

            // Two 4-token chunks, each beside the three decode rows.
            let work = try work(scheduler.firstTokenWorkProjection(for: target.id))
            #expect(work.prefillTokens == 8)
            #expect(work.decodeTokens == 6)
            #expect(work.scheduledSteps == 2)
            #expect(work.mixedSteps == 2)
            #expect(work.decodeSteps == [.init(rows: 3, width: 1): 2])
            // Measured at three rows the batch costs two steps; measured solo the
            // linear bound reproduces the legacy per-row sum.
            #expect(work.batchDecodeTokens(rate: .init(batchRows: 3)) == 2)
            #expect(work.batchDecodeTokens(rate: .init(batchRows: 1)) == 6)
            #expect(work.batchDecodeTokens(rate: .init(batchRows: 1.5)) == 4)
        }

        @Test("rows beyond the MTP batch gate are charged one target token")
        func speculativeBatchGateMirrorsLivePlanner() throws {
            let batchGate: (Int) -> Int = { $0 > 1 ? 0 : 4 }
            let gated = scheduler()
            try addDecoders(gated, count: 2, maxTokens: 20)
            speculate(gated, ceiling: batchGate)
            let target = request(prompt: 4, maxTokens: 1)
            try gated.enqueue(target)

            let legacy = try work(gated.firstTokenWorkProjection(for: target.id))
            #expect(legacy.decodeTokens == 10)
            #expect(legacy.decodeSteps == [.init(rows: 2, width: 5): 1])
            let gatedWork = try work(
                gated.firstTokenWorkProjection(for: target.id, mirrorsSpeculationDepth: true))
            #expect(gatedWork.decodeTokens == 2)
            #expect(gatedWork.decodeSteps == [.init(rows: 2, width: 1): 1])
            #expect(gatedWork.prefillTokens == legacy.prefillTokens)
            #expect(gatedWork.scheduledSteps == legacy.scheduledSteps)

            // One decode row fits the gate and keeps its full 1+k ceiling.
            let solo = scheduler()
            try addDecoders(solo, count: 1, maxTokens: 20)
            speculate(solo, ceiling: batchGate)
            let soloTarget = request(prompt: 4, maxTokens: 1)
            try solo.enqueue(soloTarget)
            let soloWork = try work(
                solo.firstTokenWorkProjection(for: soloTarget.id, mirrorsSpeculationDepth: true))
            #expect(soloWork.decodeTokens == 5)
            #expect(soloWork.decodeSteps == [.init(rows: 1, width: 5): 1])
        }

        @Test("the automatic verification envelope lowers the charged depth as rows grow")
        func automaticEnvelopeMirrorsLivePlanner() throws {
            // M3-M5 production envelope: 8 rectangular target rows per plan.
            let envelope: (Int) -> Int = { rows in max(0, 8 / rows - 1) }
            for (rows, width) in [(1, 4), (2, 4), (3, 2), (4, 2), (5, 1), (7, 1)] {
                let mirrored = scheduler()
                try addDecoders(mirrored, count: rows, maxTokens: 20)
                speculate(mirrored, upperBound: 3, ceiling: envelope)
                let target = request(prompt: 4, maxTokens: 1)
                try mirrored.enqueue(target)
                let ceiled = try work(
                    mirrored.firstTokenWorkProjection(for: target.id, mirrorsSpeculationDepth: true)
                )
                #expect(ceiled.decodeTokens == rows * width, "rows=\(rows)")
                #expect(ceiled.decodeSteps == [.init(rows: rows, width: width): 1], "rows=\(rows)")
                let legacy = try work(mirrored.firstTokenWorkProjection(for: target.id))
                #expect(legacy.decodeTokens == rows * 4, "rows=\(rows)")
            }
        }

        @Test("an unknown depth ceiling keeps the conservative 1+k charge")
        func ceilingNeedsAMirroredPlanner() throws {
            let unknown = scheduler()
            try addDecoders(unknown, count: 2, maxTokens: 20)
            speculate(unknown, ceiling: nil)
            let target = request(prompt: 4, maxTokens: 1)
            try unknown.enqueue(target)
            let work = try work(
                unknown.firstTokenWorkProjection(for: target.id, mirrorsSpeculationDepth: true))
            #expect(work.decodeTokens == 10)
        }

        @Test("a full-slot MTP stretch with no live depth projects exactly like plain decode")
        func gatedFullSlotStretchMatchesPlainDecode() throws {
            func projection(planner: Bool, mirrored: Bool) throws -> CBv2FirstTokenScheduledWork {
                let scheduler = scheduler(maxConcurrent: 2)
                try addDecoders(scheduler, count: 2, maxTokens: 10)
                if planner { speculate(scheduler, ceiling: { $0 > 1 ? 0 : 4 }) }
                let target = request(prompt: 4, maxTokens: 1)
                try scheduler.enqueue(target)
                return try work(
                    scheduler.firstTokenWorkProjection(
                        for: target.id, mirrorsSpeculationDepth: mirrored))
            }
            let plain = try projection(planner: false, mirrored: false)
            // Nine lockstep steps until both rows finish, one possible chained
            // successor, then the target's solo prompt step.
            #expect(plain.decodeTokens == 20)
            #expect(plain.decodeSteps == [.init(rows: 2, width: 1): 10])
            #expect(plain.prefillTokens == 4)
            #expect(try projection(planner: true, mirrored: true) == plain)
            let legacy = try projection(planner: true, mirrored: false)
            #expect(legacy.decodeTokens > plain.decodeTokens)
        }

        @Test(
            "the driver's depth ceiling bounds every live plan depth and is reached at fixed depth")
        func driverCeilingBoundsLivePlans() throws {
            let configurations: [(CBv2MTPVerificationMode, envelope: Int, batch: Int)] = [
                (.automatic, 8, 8), (.automatic, 4, 8), (.automatic, 8, 1), (.rectangular, 8, 2),
            ]
            for (mode, envelope, batch) in configurations {
                let model = CBv2CeilingTestModel()
                let driver = try #require(
                    CBv2MTPRoundDriver.build(
                        model: model, drafter: CBv2CeilingTestDrafter(target: model),
                        config: CBv2MTPConfig(
                            enabled: true, maxDraftTokens: 3, maxSpeculativeBatch: batch,
                            fixedDraftTokens: 3, verificationMode: mode,
                            maxAutomaticRectangularTokens: envelope)))
                for rows in 1 ... 9 {
                    // `beginMTPPlan` passes the batch gate as `canSpeculate`.
                    driver.beginPlan(plannedDecodeRows: rows, canSpeculate: rows <= batch)
                    #expect(
                        driver.planDepth
                            == min(3, driver.speculationDepthCeiling(decodeRows: rows)),
                        "mode=\(mode) envelope=\(envelope) batch=\(batch) rows=\(rows)")
                }
            }
        }

        @Test("steps scale linearly past the measured rows and width; malformed input stays legacy")
        func batchConversionArithmetic() {
            let work = CBv2FirstTokenScheduledWork(
                prefillTokens: 0, decodeTokens: 4 + 12 + 16, scheduledSteps: 9, mixedSteps: 0,
                decodeSteps: [
                    .init(rows: 1, width: 1): 4, .init(rows: 4, width: 1): 3,
                    .init(rows: 8, width: 1): 2,
                ])
            #expect(work.batchDecodeTokens(rate: .init(batchRows: 8)) == 9)
            // 4 + 3 * 2 + 2 * 4, then 4 * 2 + 3 * 8 + 2 * 16.
            #expect(work.batchDecodeTokens(rate: .init(batchRows: 2)) == 18)
            #expect(work.batchDecodeTokens(rate: .init(batchRows: 0.5)) == nil)
            for invalid in [0, -1, .nan, .infinity] as [Double] {
                #expect(work.batchDecodeTokens(rate: .init(batchRows: invalid)) == nil)
                #expect(
                    work.batchDecodeTokens(rate: .init(batchRows: 1, tokensPerStep: invalid)) == nil
                )
                #expect(
                    work.batchDecodeTokens(rate: .init(batchRows: 1, stepWidth: invalid)) == nil)
            }
            let unprojected = CBv2FirstTokenScheduledWork(
                prefillTokens: 0, decodeTokens: 8, scheduledSteps: 2, mixedSteps: 0)
            #expect(unprojected.batchDecodeTokens(rate: .init(batchRows: 4)) == nil)

            let deadline = ContinuousClock.now
            #expect(
                !CBv2FirstTokenDeadlineAdmission(
                    deadline: deadline, conservativePrefillTokensPerSecond: 1,
                    conservativeDecodeTokensPerSecond: 1
                ).chargesDecodePerBatch)
            #expect(
                CBv2FirstTokenDeadlineAdmission(
                    deadline: deadline, conservativePrefillTokensPerSecond: 1,
                    conservativeDecodeTokensPerSecond: 1,
                    decodeRateShape: .init(batchRows: 2)
                ).chargesDecodePerBatch)
            #expect(
                !CBv2FirstTokenDeadlineAdmission(
                    deadline: deadline, conservativePrefillTokensPerSecond: 1,
                    conservativeDecodeTokensPerSecond: 1,
                    decodeRateShape: .init(batchRows: .nan)
                ).chargesDecodePerBatch)
        }

        @Test("a reviewed calibration keeps its configured MTP work vector")
        func calibrationKeepsLegacyWorkVector() {
            let deadline = ContinuousClock.now
            let shape = CBv2DecodeRateShape(batchRows: 2)
            let observed = CBv2FirstTokenDeadlineAdmission(
                deadline: deadline, conservativePrefillTokensPerSecond: 1,
                conservativeDecodeTokensPerSecond: 1, decodeRateShape: shape)
            #expect(observed.mirrorsSpeculationDepth)
            let reviewed = CBv2FirstTokenDeadlineAdmission(
                deadline: deadline, conservativePrefillTokensPerSecond: 1,
                conservativeDecodeTokensPerSecond: 1,
                calibration: CBv2FirstContentCalibration(
                    cells: [], evidenceGuard: CBv2FirstContentEvidenceGuard(),
                    sameModelRequests: 1, otherModelRequests: 0, otherModelServiceFraction: 0,
                    competitorProfileIDs: []),
                decodeRateShape: shape)
            #expect(reviewed.chargesDecodePerBatch)
            #expect(!reviewed.mirrorsSpeculationDepth)
        }

        @Test(
            "an MTP-measured rate prices verify steps at confirmed tokens and plain steps at a full step"
        )
        func mtpMeasuredShapeConversion() {
            // Measured solo with depth-3 rounds confirming 2.5 tokens each.
            let measured = CBv2DecodeRateShape(batchRows: 1, tokensPerStep: 2.5, stepWidth: 4)
            let verify = CBv2FirstTokenScheduledWork(
                prefillTokens: 0, decodeTokens: 4 * 4, scheduledSteps: 4, mixedSteps: 0,
                decodeSteps: [.init(rows: 1, width: 4): 4])
            // Four matching rounds cost four measured steps, not sixteen tokens.
            #expect(verify.batchDecodeTokens(rate: measured) == 10)
            // A plain step is never priced below one measured step.
            let plain = CBv2FirstTokenScheduledWork(
                prefillTokens: 0, decodeTokens: 2 * 3, scheduledSteps: 3, mixedSteps: 0,
                decodeSteps: [.init(rows: 2, width: 1): 3])
            #expect(plain.batchDecodeTokens(rate: measured) == 15)
            // A wider round than measured scales with its width.
            let wider = CBv2FirstTokenScheduledWork(
                prefillTokens: 0, decodeTokens: 8, scheduledSteps: 1, mixedSteps: 0,
                decodeSteps: [.init(rows: 1, width: 8): 1])
            #expect(wider.batchDecodeTokens(rate: measured) == 5)
        }
    }
}

private struct CBv2PerBatchTestFailure: Error {}

private final class CBv2PerBatchIDs: @unchecked Sendable {
    private let lock = NSLock()
    private var value: UInt64 = 20_260_000
    func next() -> UInt64 {
        lock.withLock {
            value += 1
            return value
        }
    }
}

// Graph-free MTP target and drafter: the depth tests only plan, never forward.

private final class CBv2CeilingTestPrepared: CBv2MTPPreparedCapture {}

private final class CBv2CeilingTestDrafter: CBv2MTPDrafter {
    let mtpTargetIdentity: ObjectIdentifier?
    let supportsTargetPrefixAcceptance = false

    init(target: CBv2CeilingTestModel) {
        mtpTargetIdentity = ObjectIdentifier(target)
    }

    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture {
        CBv2CeilingTestPrepared()
    }

    func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        (tokens, hidden)
    }
}

private final class CBv2CeilingTestModel: CBv2MTPSteppableModel {
    let mtpCaptureLayers: CBv2MTPCaptureLayers? = .init(full: 0, sliding: 0)
    var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(self) }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        fatalError("depth ceiling tests do not execute model graphs")
    }

    func forwardWithHidden(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache]
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        fatalError("depth ceiling tests do not execute model graphs")
    }
}
