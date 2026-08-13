// CBv2MTPCaptureVerifyTests.swift
//
// MTP capture-verify (MTPLX GDN capture-commit pattern) — targeted tests:
//  1. Captured recurrent transaction lifecycle (stageCaptured /
//     commit(keepPositions:) / rollback / bind fencing).
//  2. Qwen35GatedDeltaNet captured-window forward equivalence against the
//     serial per-column oracle, for both the rejected (keep 1) and accepted
//     (keep full window) commit paths.
//  3. Default-sampler MTP verify pre-sampling reproduces the ordinary
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

    // MARK: - 3. Verify pre-sampling reproduces the ordinary draw

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
