// CBv2MTPCaptureVerifyTests.swift
//
// MTP capture-verify (MTPLX GDN capture-commit pattern) — targeted tests:
//  1. Captured recurrent transaction lifecycle (stageCaptured /
//     commit(keepPositions:) / rollback / bind fencing).
//  2. Qwen35GatedDeltaNet captured-window forward equivalence against the
//     serial per-column oracle, for both the rejected (keep 1) and accepted
//     (keep full window) commit paths.
//  3. Compact legacy-cache prefix replay across all accepted boundaries,
//     including fallback and terminal tape cleanup.
//  4. Default-sampler MTP verify pre-sampling reproduces the ordinary
//     per-token draw (greedy bitwise; stochastic same keyed RNG stream).

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class CBv2MTPCaptureVerifyTests: XCTestCase {

    // MARK: - 1. Captured transaction lifecycle

    private var scalarSpec: CBv2RecurrentStateSpec {
        CBv2RecurrentStateSpec(layers: [
            CBv2RecurrentLayerStateSpec(
                modelLayerIndex: 0,
                convShape: [1, 2, 3], convDType: .float32,
                ssmShape: [1, 1, 2, 2], ssmDType: .float32)
        ])
    }

    private func capturedStacks(positions: Int, base: Float) -> (conv: MLXArray, ssm: MLXArray) {
        let conv = concatenated(
            (0 ..< positions).map {
                MLXArray.full([1, 2, 3], values: MLXArray(base + Float($0 + 1)))
            }, axis: 0)
        let ssm = concatenated(
            (0 ..< positions).map {
                MLXArray.full([1, 1, 2, 2], values: MLXArray(base + Float($0 + 1)))
            }, axis: 0)
        return (conv, ssm)
    }

    func testCapturedCommitSelectsPositionAndRollbackRestores() throws {
        let state = try CBv2RecurrentRequestState(spec: scalarSpec)

        // Commit a plain baseline generation with value 10.
        let baseline = try state.bind()
        try baseline.stage(
            modelLayerIndex: 0,
            conv: MLXArray.full([1, 2, 3], values: MLXArray(Float(10))),
            ssm: MLXArray.full([1, 1, 2, 2], values: MLXArray(Float(10))))
        _ = try baseline.evaluate()
        try baseline.commit()

        // Captured window of 3 positions (values 11, 12, 13); commit keep=2.
        let captured = try state.bind()
        let stacks = capturedStacks(positions: 3, base: 10)
        try captured.stageCaptured(
            modelLayerIndex: 0, conv: stacks.conv, ssm: stacks.ssm, positions: 3)
        XCTAssertTrue(captured.isCaptured)
        _ = try captured.evaluate()
        try captured.commit(keepPositions: 2)

        let afterCommit = state.state(modelLayerIndex: 0)!
        XCTAssertEqual(afterCommit.conv!.shape, [1, 2, 3])
        XCTAssertEqual(afterCommit.ssm!.shape, [1, 1, 2, 2])
        XCTAssertEqual(afterCommit.conv![0, 0, 0].item(Float.self), 12)
        XCTAssertEqual(afterCommit.ssm![0, 0, 0, 0].item(Float.self), 12)

        // Rollback of a captured window restores the committed state.
        let rejected = try state.bind()
        let rejectedStacks = capturedStacks(positions: 2, base: 12)
        try rejected.stageCaptured(
            modelLayerIndex: 0, conv: rejectedStacks.conv, ssm: rejectedStacks.ssm,
            positions: 2)
        _ = try rejected.evaluate()
        try rejected.rollback()
        let afterRollback = state.state(modelLayerIndex: 0)!
        XCTAssertEqual(afterRollback.conv![0, 0, 0].item(Float.self), 12)
        XCTAssertEqual(afterRollback.ssm![0, 0, 0, 0].item(Float.self), 12)
    }

    func testCapturedLifecycleFencing() throws {
        let state = try CBv2RecurrentRequestState(spec: scalarSpec)
        let captured = try state.bind()
        let stacks = capturedStacks(positions: 2, base: 0)

        // Plain stage after a captured stage is a violation.
        try captured.stageCaptured(
            modelLayerIndex: 0, conv: stacks.conv, ssm: stacks.ssm, positions: 2)
        XCTAssertThrowsError(
            try captured.stage(
                modelLayerIndex: 0,
                conv: MLXArray.zeros([1, 2, 3]),
                ssm: MLXArray.zeros([1, 1, 2, 2])))

        _ = try captured.evaluate()

        // A chained bind over an unresolved captured window must fail loud:
        // its per-position stacks are not a readable input state.
        XCTAssertThrowsError(try state.bind())

        // Plain commit on a captured generation is a violation; out-of-range
        // keeps are violations; a valid keep succeeds.
        XCTAssertThrowsError(try captured.commit())
        XCTAssertThrowsError(try captured.commit(keepPositions: 3))
        try captured.commit(keepPositions: 1)
        XCTAssertEqual(state.state(modelLayerIndex: 0)!.conv![0, 0, 0].item(Float.self), 1)
    }

    func testCapturedWindowResidencyChargesPerPosition() throws {
        // The bare-loop capacity gauge derives from materializedByteCount:
        // a captured verify window of P positions holds P full conv/SSM
        // copies, so it must be charged P× — not as one generation.
        let state = try CBv2RecurrentRequestState(spec: scalarSpec)
        XCTAssertEqual(state.materializedByteCount, 0)

        let baseline = try state.bind()
        try baseline.stage(
            modelLayerIndex: 0,
            conv: MLXArray.full([1, 2, 3], values: MLXArray(Float(10))),
            ssm: MLXArray.full([1, 1, 2, 2], values: MLXArray(Float(10))))
        _ = try baseline.evaluate()
        try baseline.commit()
        XCTAssertEqual(state.materializedByteCount, state.byteCount)

        // Committed state + captured window of 2 positions = 3 copies.
        let captured = try state.bind()
        let stacks = capturedStacks(positions: 2, base: 10)
        try captured.stageCaptured(
            modelLayerIndex: 0, conv: stacks.conv, ssm: stacks.ssm, positions: 2)
        _ = try captured.evaluate()
        XCTAssertEqual(state.materializedByteCount, 3 * state.byteCount)

        // Committing one position collapses the window back to one copy.
        try captured.commit(keepPositions: 1)
        XCTAssertEqual(state.materializedByteCount, state.byteCount)
    }

    func testCompactReplayStrictPrefixFullCommitAndRollback() throws {
        func makeState() throws -> CBv2RecurrentRequestState {
            let state = try CBv2RecurrentRequestState(spec: scalarSpec)
            let baseline = try state.bind()
            try baseline.stage(
                modelLayerIndex: 0,
                conv: MLXArray.full([1, 2, 3], values: MLXArray(Float(10))),
                ssm: MLXArray.full([1, 1, 2, 2], values: MLXArray(Float(10))))
            _ = try baseline.evaluate()
            try baseline.commit()
            return state
        }

        let strictState = try makeState()
        let strict = try strictState.bind()
        var strictReplayKeeps: [Int] = []
        let root = MLXArray(Float(123))
        try strict.stagePrefixReplay(
            modelLayerIndex: 0, positions: 4,
            finalConv: MLXArray.full([1, 2, 3], values: MLXArray(Float(14))),
            finalSSM: MLXArray.full([1, 1, 2, 2], values: MLXArray(Float(14))),
            materializedByteCount: 37, evaluationRoots: [root],
            strictReplayRetainedByteCount: 17,
            strictReplayRetainedRoots: [root]
        ) { keep in
            strictReplayKeeps.append(keep)
            return CBv2RecurrentLayerState(
                conv: MLXArray.full([1, 2, 3], values: MLXArray(Float(10 + keep))),
                ssm: MLXArray.full(
                    [1, 1, 2, 2], values: MLXArray(Float(10 + keep))))
        }
        XCTAssertTrue(strict.isCaptured)
        let roots = try strict.evaluate()
        XCTAssertEqual(roots.count, 3)
        XCTAssertEqual(roots.last!.item(Float.self), 123)
        XCTAssertEqual(strictState.materializedByteCount, strictState.byteCount + 37)
        XCTAssertThrowsError(try strictState.bind())
        try strict.commit(keepPositions: 2)
        XCTAssertEqual(strictReplayKeeps, [2])
        XCTAssertEqual(
            strictState.state(modelLayerIndex: 0)!.ssm![0, 0, 0, 0].item(Float.self), 12)
        XCTAssertEqual(
            strictState.materializedByteCount, strictState.byteCount + 17)
        let settled = try strictState.bind()
        let strictCommitted = strictState.state(modelLayerIndex: 0)!
        try settled.stage(
            modelLayerIndex: 0,
            conv: try XCTUnwrap(strictCommitted.conv),
            ssm: try XCTUnwrap(strictCommitted.ssm))
        eval(try settled.evaluate())
        try settled.commit()
        XCTAssertEqual(strictState.materializedByteCount, strictState.byteCount)

        let fullState = try makeState()
        let full = try fullState.bind()
        let fullRetentionRoot = MLXArray(Float(456))
        var fullMaterializationCount = 0
        var fullReplayCount = 0
        try full.stagePrefixReplay(
            modelLayerIndex: 0, positions: 4,
            finalConv: MLXArray.full([1, 2, 3], values: MLXArray(Float(99))),
            finalSSM: MLXArray.full([1, 1, 2, 2], values: MLXArray(Float(99))),
            materializedByteCount: 41, evaluationRoots: [],
            strictReplayRetainedByteCount: 0,
            strictReplayRetainedRoots: [],
            fullAcceptanceRetainedByteCount: 19,
            fullAcceptanceRetainedRoots: [fullRetentionRoot],
            fullAcceptance: {
                fullMaterializationCount += 1
                return CBv2RecurrentLayerState(
                    conv: MLXArray.full([1, 2, 3], values: MLXArray(Float(14))),
                    ssm: MLXArray.full(
                        [1, 1, 2, 2], values: MLXArray(Float(14))))
            }
        ) { _ in
            fullReplayCount += 1
            return CBv2RecurrentLayerState(conv: nil, ssm: nil)
        }
        _ = try full.evaluate()
        XCTAssertEqual(fullMaterializationCount, 0)
        try full.commit(keepPositions: 4)
        XCTAssertEqual(fullMaterializationCount, 1)
        XCTAssertEqual(fullReplayCount, 0)
        XCTAssertEqual(
            fullState.state(modelLayerIndex: 0)!.conv![0, 0, 0].item(Float.self), 14)
        XCTAssertEqual(fullState.materializedByteCount, fullState.byteCount + 19)
        let fullSettled = try fullState.bind()
        let fullCommitted = fullState.state(modelLayerIndex: 0)!
        try fullSettled.stage(
            modelLayerIndex: 0,
            conv: try XCTUnwrap(fullCommitted.conv),
            ssm: try XCTUnwrap(fullCommitted.ssm))
        eval(try fullSettled.evaluate())
        try fullSettled.commit()
        XCTAssertEqual(fullState.materializedByteCount, fullState.byteCount)

        let rolledBackState = try makeState()
        let rolledBack = try rolledBackState.bind()
        var rollbackFullAcceptanceCount = 0
        var rollbackReplayCount = 0
        try rolledBack.stagePrefixReplay(
            modelLayerIndex: 0, positions: 3,
            finalConv: MLXArray.full([1, 2, 3], values: MLXArray(Float(13))),
            finalSSM: MLXArray.full([1, 1, 2, 2], values: MLXArray(Float(13))),
            materializedByteCount: 29, evaluationRoots: [],
            strictReplayRetainedByteCount: 0,
            strictReplayRetainedRoots: [],
            fullAcceptance: {
                rollbackFullAcceptanceCount += 1
                return CBv2RecurrentLayerState(conv: nil, ssm: nil)
            }
        ) { _ in
            rollbackReplayCount += 1
            return CBv2RecurrentLayerState(conv: nil, ssm: nil)
        }
        _ = try rolledBack.evaluate()
        try rolledBack.rollback()
        XCTAssertEqual(rollbackFullAcceptanceCount, 0)
        XCTAssertEqual(rollbackReplayCount, 0)
        XCTAssertEqual(
            rolledBackState.state(modelLayerIndex: 0)!.conv![0, 0, 0].item(Float.self), 10)
    }

    func testCompactReplayRejectsMixedModesLayersAndWidths() throws {
        let twoLayerSpec = CBv2RecurrentStateSpec(layers: [
            scalarSpec.layers[0],
            CBv2RecurrentLayerStateSpec(
                modelLayerIndex: 1,
                convShape: [1, 2, 3], convDType: .float32,
                ssmShape: [1, 1, 2, 2], ssmDType: .float32),
        ])
        let state = try CBv2RecurrentRequestState(spec: twoLayerSpec)
        let transaction = try state.bind()
        func stageReplay(layer: Int, positions: Int) throws {
            try transaction.stagePrefixReplay(
                modelLayerIndex: layer, positions: positions,
                finalConv: MLXArray.zeros([1, 2, 3]),
                finalSSM: MLXArray.zeros([1, 1, 2, 2]),
                materializedByteCount: 1, evaluationRoots: [],
                strictReplayRetainedByteCount: 0,
                strictReplayRetainedRoots: []
            ) { _ in
                CBv2RecurrentLayerState(
                    conv: MLXArray.zeros([1, 2, 3]),
                    ssm: MLXArray.zeros([1, 1, 2, 2]))
            }
        }
        try stageReplay(layer: 0, positions: 4)
        XCTAssertThrowsError(
            try transaction.stage(
                modelLayerIndex: 1, conv: MLXArray.zeros([1, 2, 3]),
                ssm: MLXArray.zeros([1, 1, 2, 2])))
        XCTAssertThrowsError(
            try transaction.stageCaptured(
                modelLayerIndex: 1, conv: MLXArray.zeros([4, 2, 3]),
                ssm: MLXArray.zeros([4, 1, 2, 2]), positions: 4))
        XCTAssertThrowsError(try stageReplay(layer: 1, positions: 3))
        try stageReplay(layer: 1, positions: 4)
        _ = try transaction.evaluate()
        XCTAssertThrowsError(try transaction.commit())
        try transaction.commit(keepPositions: 4)
    }

    // MARK: - 2. GDN captured-window equivalence

    private func smallGDNConfiguration() throws -> Qwen35TextConfiguration {
        // linear_key_head_dim must be >= 32: the gated-delta Metal kernel
        // tiles the state as Dk/32 floats per thread.
        let json = Data(
            """
            {
              "model_type": "qwen3_5_moe",
              "hidden_size": 16,
              "num_hidden_layers": 4,
              "intermediate_size": 32,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "linear_num_value_heads": 4,
              "linear_num_key_heads": 2,
              "linear_key_head_dim": 32,
              "linear_value_head_dim": 16,
              "linear_conv_kernel_dim": 4,
              "vocab_size": 32,
              "head_dim": 8,
              "full_attention_interval": 4,
              "num_experts": 0,
              "num_experts_per_tok": 0
            }
            """.utf8)
        return try JSONDecoder.json5().decode(Qwen35TextConfiguration.self, from: json)
    }

    private func gdnSpec(_ config: Qwen35TextConfiguration) -> CBv2RecurrentStateSpec {
        let keyDim = config.linearNumKeyHeads * config.linearKeyHeadDim
        let valueDim = config.linearNumValueHeads * config.linearValueHeadDim
        return CBv2RecurrentStateSpec(layers: [
            CBv2RecurrentLayerStateSpec(
                modelLayerIndex: 0,
                convShape: [1, config.linearConvKernelDim - 1, 2 * keyDim + valueDim],
                convDType: .float32,
                ssmShape: [
                    1, config.linearNumValueHeads, config.linearValueHeadDim,
                    config.linearKeyHeadDim,
                ],
                ssmDType: .float32)
        ])
    }

    /// After a REJECTED draft the captured commit at keep=1 must reproduce
    /// the state of a control that never speculated (only consumed the seed
    /// column); after an ACCEPTED draft the captured commit at keep=2 must
    /// match a control that decoded the same two tokens serially. Outputs
    /// are compared within distributional tolerance (batched projections
    /// change accumulation geometry; bitwise parity is NOT the contract —
    /// see Qwen35MTP.swift NUMERICS POLICY).
    func testGDNCapturedWindowMatchesSerialOracle() throws {
        MLXRandom.seed(4711)
        let config = try smallGDNConfiguration()
        let layer = Qwen35GatedDeltaNet(config)
        eval(layer)

        let hidden = config.hiddenSize
        MLXRandom.seed(90210)
        let x0 = MLXRandom.normal([1, 1, hidden])
        let x1 = MLXRandom.normal([1, 1, hidden])
        let window = concatenated([x0, x1], axis: 1)

        // Warm history so the states are non-trivial before the window.
        let history = MLXRandom.normal([1, 3, hidden])

        func freshState() throws -> CBv2RecurrentRequestState {
            let state = try CBv2RecurrentRequestState(spec: gdnSpec(config))
            let warm = try state.bind()
            _ = layer.cbv2Forward(history, modelLayerIndex: 0, recurrentState: [warm])
            _ = try warm.evaluate()
            try warm.commit()
            return state
        }

        // Serial oracle: two [1, 1] columns through plain transactions.
        let serialState = try freshState()
        let serial0 = try serialState.bind()
        let y0 = layer.cbv2Forward(x0, modelLayerIndex: 0, recurrentState: [serial0])
        _ = try serial0.evaluate()
        try serial0.commit()
        let afterSeed = serialState.state(modelLayerIndex: 0)!
        let afterSeedConv = afterSeed.conv! + 0
        let afterSeedSSM = afterSeed.ssm! + 0
        let serial1 = try serialState.bind()
        let y1 = layer.cbv2Forward(x1, modelLayerIndex: 0, recurrentState: [serial1])
        _ = try serial1.evaluate()
        try serial1.commit()
        let afterDraft = serialState.state(modelLayerIndex: 0)!
        eval(y0, y1, afterSeedConv, afterSeedSSM, afterDraft.conv!, afterDraft.ssm!)

        // Rejected draft: captured window, keep 1 → state == control that
        // never consumed the draft.
        let rejectedState = try freshState()
        let rejected = try rejectedState.bind()
        let rejectedOut = layer.cbv2ForwardCaptured(
            window, modelLayerIndex: 0, recurrentState: [rejected])
        XCTAssertTrue(rejected.isCaptured)
        _ = try rejected.evaluate()
        eval(rejectedOut)
        try rejected.commit(keepPositions: 1)
        let rejectedFinal = rejectedState.state(modelLayerIndex: 0)!
        XCTAssertEqual(rejectedFinal.conv!.shape, afterSeedConv.shape)
        XCTAssertEqual(rejectedFinal.ssm!.shape, afterSeedSSM.shape)
        XCTAssertTrue(
            allClose(rejectedFinal.conv!, afterSeedConv, rtol: 1e-4, atol: 1e-5)
                .item(Bool.self))
        XCTAssertTrue(
            allClose(rejectedFinal.ssm!, afterSeedSSM, rtol: 1e-4, atol: 1e-5)
                .item(Bool.self))

        // Accepted draft: captured window, keep 2 → state == control that
        // decoded both tokens serially; window outputs match the serial
        // column outputs.
        let acceptedState = try freshState()
        let accepted = try acceptedState.bind()
        let acceptedOut = layer.cbv2ForwardCaptured(
            window, modelLayerIndex: 0, recurrentState: [accepted])
        _ = try accepted.evaluate()
        eval(acceptedOut)
        try accepted.commit(keepPositions: 2)
        let acceptedFinal = acceptedState.state(modelLayerIndex: 0)!
        XCTAssertTrue(
            allClose(acceptedFinal.conv!, afterDraft.conv!, rtol: 1e-4, atol: 1e-5)
                .item(Bool.self))
        XCTAssertTrue(
            allClose(acceptedFinal.ssm!, afterDraft.ssm!, rtol: 1e-4, atol: 1e-5)
                .item(Bool.self))
        XCTAssertTrue(acceptedOut.shape == [1, 2, hidden])
        XCTAssertTrue(
            allClose(
                acceptedOut, concatenated([y0, y1], axis: 1), rtol: 1e-3, atol: 1e-4
            ).item(Bool.self))
        for value in [rejectedOut, acceptedOut] {
            XCTAssertTrue(isFinite(value).all().item(Bool.self))
        }
    }

    func testGDNWideWindowUsesCompactReplayAndMatchesEverySerialPrefix() throws {
        MLXRandom.seed(0xC0FFEE)
        let config = try smallGDNConfiguration()
        let layer = Qwen35GatedDeltaNet(config)
        eval(layer)
        let hidden = config.hiddenSize
        let columns = (0 ..< 4).map { _ in MLXRandom.normal([1, 1, hidden]) }
        let window = concatenated(columns, axis: 1)
        let history = MLXRandom.normal([1, 3, hidden])

        func freshState() throws -> CBv2RecurrentRequestState {
            let state = try CBv2RecurrentRequestState(spec: gdnSpec(config))
            let warm = try state.bind()
            _ = layer.cbv2Forward(history, modelLayerIndex: 0, recurrentState: [warm])
            _ = try warm.evaluate()
            try warm.commit()
            return state
        }

        let serial = try freshState()
        var oracle: [(conv: MLXArray, ssm: MLXArray)] = []
        for column in columns {
            let step = try serial.bind()
            _ = layer.cbv2Forward(column, modelLayerIndex: 0, recurrentState: [step])
            _ = try step.evaluate()
            try step.commit()
            let state = serial.state(modelLayerIndex: 0)!
            let conv = state.conv! + 0
            let ssm = state.ssm! + 0
            eval(conv, ssm)
            oracle.append((conv, ssm))
        }

        for keep in 1 ... columns.count {
            let compactState = try freshState()
            let compact = try compactState.bind()
            _ = layer.cbv2ForwardCaptured(
                window, modelLayerIndex: 0, recurrentState: [compact])
            _ = try compact.evaluate()
            XCTAssertTrue(compact.isCaptured)
            XCTAssertLessThan(
                compactState.materializedByteCount,
                (columns.count + 1) * compactState.byteCount,
                "wide production verify must not materialize a full state stack")
            try compact.commit(keepPositions: keep)
            let actual = compactState.state(modelLayerIndex: 0)!
            XCTAssertTrue(
                allClose(actual.conv!, oracle[keep - 1].conv, rtol: 1e-4, atol: 1e-5)
                    .item(Bool.self))
            XCTAssertTrue(
                allClose(actual.ssm!, oracle[keep - 1].ssm, rtol: 1e-4, atol: 1e-5)
                    .item(Bool.self))
        }
    }

    func testProductionWidthCompactReplayFitsFourGenerationContinuationCharge() throws {
        let json = Data(
            """
            {
              "model_type": "qwen3_5_moe",
              "hidden_size": 16,
              "num_hidden_layers": 4,
              "intermediate_size": 32,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "linear_num_value_heads": 32,
              "linear_num_key_heads": 16,
              "linear_key_head_dim": 128,
              "linear_value_head_dim": 128,
              "linear_conv_kernel_dim": 4,
              "vocab_size": 32,
              "head_dim": 8,
              "full_attention_interval": 4,
              "num_experts": 0,
              "num_experts_per_tok": 0
            }
            """.utf8)
        let config = try JSONDecoder.json5().decode(
            Qwen35TextConfiguration.self, from: json)
        let layer = Qwen35GatedDeltaNet(config)
        eval(layer)
        let spec = gdnSpec(config)
        let state = try CBv2RecurrentRequestState(spec: spec)

        let warm = try state.bind()
        _ = layer.cbv2Forward(
            MLXRandom.normal([1, 3, config.hiddenSize]),
            modelLayerIndex: 0, recurrentState: [warm])
        _ = try warm.evaluate()
        try warm.commit()

        let verify = try state.bind()
        _ = layer.cbv2ForwardCaptured(
            MLXRandom.normal([1, 5, config.hiddenSize]),
            modelLayerIndex: 0, recurrentState: [verify])
        _ = try verify.evaluate()

        let fixedThreeGenerations = try spec.peakBytesPerRequest()
        XCTAssertEqual(fixedThreeGenerations, 3 * state.byteCount)
        XCTAssertLessThanOrEqual(
            state.materializedByteCount, fixedThreeGenerations,
            "the compact verify itself must fit the base recurrent charge")

        // A strict-prefix commit retains its replay tape until a successor
        // generation commits. That reachable transition exceeds 3F but must
        // fit the compact driver's resolved 4F admission reservation.
        try verify.commit(keepPositions: 3)
        let successor = try state.bind()
        _ = layer.cbv2Forward(
            MLXRandom.normal([1, 1, config.hiddenSize]),
            modelLayerIndex: 0, recurrentState: [successor])
        eval(try successor.evaluate())
        let fixedFourGenerations = 4 * state.byteCount
        XCTAssertGreaterThan(state.materializedByteCount, fixedThreeGenerations)
        XCTAssertLessThanOrEqual(
            state.materializedByteCount, fixedFourGenerations,
            "strict-prefix continuation must fit resolved compact 4F headroom")
        try successor.commit()
        XCTAssertEqual(state.materializedByteCount, state.byteCount)

        // Full acceptance defers the exact conv-tail copy until commit. Its
        // lazy graph keeps the verify-window backing charged until the next
        // evaluated generation provides the release fence.
        let fullVerify = try state.bind()
        _ = layer.cbv2ForwardCaptured(
            MLXRandom.normal([1, 5, config.hiddenSize]),
            modelLayerIndex: 0, recurrentState: [fullVerify])
        eval(try fullVerify.evaluate())
        try fullVerify.commit(keepPositions: 5)
        XCTAssertGreaterThan(state.materializedByteCount, state.byteCount)
        XCTAssertLessThanOrEqual(state.materializedByteCount, fixedThreeGenerations)

        let fullSuccessor = try state.bind()
        _ = layer.cbv2Forward(
            MLXRandom.normal([1, 1, config.hiddenSize]),
            modelLayerIndex: 0, recurrentState: [fullSuccessor])
        eval(try fullSuccessor.evaluate())
        XCTAssertLessThanOrEqual(
            state.materializedByteCount, fixedFourGenerations)
        try fullSuccessor.commit()
        XCTAssertEqual(state.materializedByteCount, state.byteCount)
    }

    // MARK: - 3. Compact recurrent prefix replay

    private func legacyCaches(_ model: Qwen35TextModel) -> [any KVCache] {
        model.newCache(parameters: nil).map { $0 as any KVCache }
    }

    private func copiedCaches(_ caches: [any KVCache]) -> [any KVCache] {
        caches.map { $0.copy() }
    }

    @discardableResult
    private func legacyForward(
        _ model: Qwen35TextModel,
        tokens: [Int32],
        caches: [any KVCache],
        nConfirmed: Int
    ) -> MLXArray {
        let input = MLXArray(tokens).reshaped(1, tokens.count)
        let output = model.model(
            input, cache: caches.map { Optional($0) }, nConfirmed: nConfirmed)
        eval([output] + caches.flatMap { $0.innerState() })
        return output
    }

    private func assertRecurrentStateEqual(
        _ lhs: [any KVCache],
        _ rhs: [any KVCache],
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(lhs.count, rhs.count, file: file, line: line)
        for index in lhs.indices {
            guard let left = lhs[index] as? MambaCache,
                  let right = rhs[index] as? MambaCache
            else { continue }
            XCTAssertEqual(left.state.count, right.state.count, file: file, line: line)
            for (leftArray, rightArray) in zip(left.state, right.state) {
                XCTAssertEqual(leftArray.shape, rightArray.shape, file: file, line: line)
                XCTAssertTrue(
                    allClose(leftArray, rightArray, rtol: 1e-5, atol: 1e-6)
                        .item(Bool.self),
                    file: file,
                    line: line)
            }
        }
    }

    /// A partially accepted wide verify rebuilds each recurrent layer from its
    /// compact tape, while attention remains at the speculative offset until
    /// the caller trims exactly the rejected suffix.
    func testRecurrentPrefixReplayMatchesSerialPrefixAndAttentionTrim() throws {
        MLXRandom.seed(61_337)
        let model = Qwen35TextModel(try smallGDNConfiguration())
        eval(model)

        let base = legacyCaches(model)
        _ = legacyForward(
            model, tokens: [1, 2, 3], caches: base, nConfirmed: 0)
        let baseAttentionOffsets = base.map(\.offset)
        let verifyTokens: [Int32] = [4, 5, 6, 7]

        // One accepted verify row is a total draft rejection; larger proper
        // prefixes exercise partial acceptance at every available boundary.
        for committedRows in 1 ..< verifyTokens.count {
            let verify = copiedCaches(base)
            let oracle = copiedCaches(base)
            _ = legacyForward(
                model, tokens: verifyTokens, caches: verify, nConfirmed: 1)
            for case let recurrent as MambaCache in verify {
                let tape = try XCTUnwrap(recurrent.prefixReplayTape)
                XCTAssertEqual(tape.rowCount, verifyTokens.count)
                XCTAssertNil(recurrent.rollbackState)
            }
            for index in verify.indices where !(verify[index] is ArraysCache) {
                XCTAssertEqual(
                    verify[index].offset,
                    baseAttentionOffsets[index] + verifyTokens.count)
            }

            XCTAssertTrue(
                model.replayRecurrentPrefix(
                    cache: verify, committedRows: committedRows))
            for case let recurrent as MambaCache in verify {
                XCTAssertNil(recurrent.prefixReplayTape)
                XCTAssertNil(recurrent.rollbackState)
            }
            // Recurrent replay must not mutate attention bookkeeping.
            let rejectedRows = verifyTokens.count - committedRows
            for index in verify.indices where !(verify[index] is ArraysCache) {
                XCTAssertEqual(
                    verify[index].offset,
                    baseAttentionOffsets[index] + verifyTokens.count)
                XCTAssertEqual(verify[index].trim(rejectedRows), rejectedRows)
                XCTAssertEqual(
                    verify[index].offset,
                    baseAttentionOffsets[index] + committedRows)
            }

            _ = legacyForward(
                model,
                tokens: Array(verifyTokens.prefix(committedRows)),
                caches: oracle,
                nConfirmed: 0)
            assertRecurrentStateEqual(verify, oracle)
            for index in oracle.indices where !(oracle[index] is ArraysCache) {
                XCTAssertEqual(verify[index].offset, oracle[index].offset)
            }
        }
    }

    /// Shape-incompatible tapes fail closed before recurrent state mutation and
    /// release every layer's verify-only arrays for the fallback path.
    func testIneligibleRecurrentReplayCleansAllTapesWithoutStateMutation() throws {
        MLXRandom.seed(61_338)
        let model = Qwen35TextModel(try smallGDNConfiguration())
        eval(model)

        let caches = legacyCaches(model)
        _ = legacyForward(
            model, tokens: [1, 2, 3, 4], caches: caches, nConfirmed: 1)
        let recurrent = caches.compactMap { $0 as? MambaCache }
        let before = recurrent.map { $0.state.map { $0 + 0 } }
        eval(before.flatMap { $0 })

        // Corrupt the final recurrent layer so a mutate-as-it-walks replay
        // would damage earlier layers before discovering ineligibility.
        let invalid = try XCTUnwrap(recurrent.last)
        let tape = try XCTUnwrap(invalid.prefixReplayTape)
        invalid.prefixReplayTape = ArraysCache.PrefixReplayTape(
            convInput: tape.convInput,
            q: MLXArray.zeros([1]),
            k: tape.k,
            v: tape.v,
            a: tape.a,
            b: tape.b,
            ssmPre: tape.ssmPre,
            mask: tape.mask,
            rowCount: tape.rowCount,
            convStateRows: tape.convStateRows)

        XCTAssertFalse(
            model.replayRecurrentPrefix(cache: caches, committedRows: 2))
        for (index, cache) in recurrent.enumerated() {
            XCTAssertNil(cache.prefixReplayTape)
            XCTAssertNil(cache.rollbackState)
            for (actual, expected) in zip(cache.state, before[index]) {
                XCTAssertTrue(
                    allClose(actual, expected, rtol: 0, atol: 0)
                        .item(Bool.self))
            }
        }
    }

    /// Full acceptance explicitly discards the tape without touching state;
    /// an ordinary next forward and a direct cache reset also cannot retain it.
    func testRecurrentReplayTapeCleanupOnTerminalPaths() throws {
        MLXRandom.seed(61_339)
        let model = Qwen35TextModel(try smallGDNConfiguration())
        eval(model)

        let base = legacyCaches(model)
        let accepted = copiedCaches(base)
        _ = legacyForward(
            model, tokens: [1, 2, 3], caches: accepted, nConfirmed: 1)
        let acceptedState = accepted.compactMap { ($0 as? MambaCache)?.state.map { $0 + 0 } }
        eval(acceptedState.flatMap { $0 })
        model.clearRecurrentPrefixReplay(cache: accepted)
        for (index, cache) in accepted.compactMap({ $0 as? MambaCache }).enumerated() {
            XCTAssertNil(cache.prefixReplayTape)
            XCTAssertNil(cache.rollbackState)
            for (actual, expected) in zip(cache.state, acceptedState[index]) {
                XCTAssertTrue(
                    allClose(actual, expected, rtol: 0, atol: 0)
                        .item(Bool.self))
            }
        }

        let continued = copiedCaches(base)
        _ = legacyForward(
            model, tokens: [1, 2, 3], caches: continued, nConfirmed: 1)
        _ = legacyForward(
            model, tokens: [4], caches: continued, nConfirmed: 0)
        for case let cache as MambaCache in continued {
            XCTAssertNil(cache.prefixReplayTape)
            XCTAssertNil(cache.rollbackState)
        }

        let reset = copiedCaches(base)
        _ = legacyForward(
            model, tokens: [1, 2, 3], caches: reset, nConfirmed: 1)
        let resetFirst = try XCTUnwrap(reset.first)
        let resetCache = try XCTUnwrap(resetFirst as? MambaCache)
        XCTAssertNotNil(resetCache.prefixReplayTape)
        resetCache.rollbackState = (
            try XCTUnwrap(resetCache[0]),
            try XCTUnwrap(resetCache[1]))
        resetCache.state = []
        XCTAssertNil(resetCache.prefixReplayTape)
        XCTAssertNil(resetCache.rollbackState)
    }

    // MARK: - 4. Verify pre-sampling reproduces the ordinary draw

    func testVerifySampleMatchesOrdinaryDraws() throws {
        let vocab = 64
        let id = CBv2RequestID(7)
        let params = CBv2SamplingParams(
            temperature: 0.7, topP: 0.9, topK: 12, seed: 1234)
        let row = CBv2SamplerRow(
            id: id, params: params, promptTokens: [1, 2, 3], outputTokens: [])

        MLXRandom.seed(5)
        let logits0 = MLXRandom.normal([1, vocab])
        let logits1 = MLXRandom.normal([1, vocab])
        eval(logits0, logits1)

        // Ordinary path: two consecutive draws through the stateful sampler.
        let ordinary = CBv2DefaultSampler(fallbackSeed: 99)
        let draw0 = ordinary.sample(
            logits: logits0, params: [params], requestIDs: [id], stepIndex: 0,
            pendingSampledTokens: nil, rowContext: { [row] })
        let draw1 = ordinary.sample(
            logits: logits1, params: [params], requestIDs: [id], stepIndex: 1,
            pendingSampledTokens: nil, rowContext: { [row] })
        eval(draw0, draw1)

        // Verify pre-sampling: both positions in one [1, 2, vocab] window,
        // stepBase 0 (no tokens generated yet).
        let verifier = CBv2DefaultSampler(fallbackSeed: 99)
        XCTAssertTrue(verifier.supportsMTPTargetPrefix)
        let window = concatenated(
            [logits0.reshaped([1, 1, vocab]), logits1.reshaped([1, 1, vocab])], axis: 1)
        let sampled = try XCTUnwrap(
            verifier.mtpVerifySample(
                logits: window, params: [params], requestIDs: [id], stepBases: [0]))
        eval(sampled)
        XCTAssertEqual(sampled.shape, [1, 2])
        XCTAssertEqual(sampled[0, 0].item(Int32.self), draw0[0].item(Int32.self))
        XCTAssertEqual(sampled[0, 1].item(Int32.self), draw1[0].item(Int32.self))

        // Greedy rows go through the bit-identical argmax path.
        let greedy = CBv2SamplingParams(temperature: 0)
        let greedySampled = try XCTUnwrap(
            verifier.mtpVerifySample(
                logits: window, params: [greedy], requestIDs: [id], stepBases: [0]))
        let expected = argMax(window.reshaped([2, vocab]), axis: -1).asType(.int32)
        eval(greedySampled, expected)
        XCTAssertEqual(greedySampled[0, 0].item(Int32.self), expected[0].item(Int32.self))
        XCTAssertEqual(greedySampled[0, 1].item(Int32.self), expected[1].item(Int32.self))
    }
}
