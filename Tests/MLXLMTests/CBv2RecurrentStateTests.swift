import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class CBv2RecurrentStateTests: XCTestCase {
    private final class RecurrentFixtureModel: CBv2RecurrentSteppableModel {
        let cbv2Capabilities = CBv2ModelCapabilities.initialRecurrentTarget
        let recurrentStateSpec: CBv2RecurrentStateSpec? = CBv2RecurrentStateSpec(layers: [
            CBv2RecurrentLayerStateSpec(
                modelLayerIndex: 0,
                convShape: [1, 1, 1], convDType: .float32,
                ssmShape: [1, 1, 1, 1], ssmDType: .float32)
        ])

        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
            preconditionFailure("fixture requires recurrent forward")
        }

        func forward(
            tokens: MLXArray, caches: [CBv2AttendingLayerCache],
            recurrentState: [CBv2RecurrentStateEvaluation]
        ) -> MLXArray {
            let B = tokens.dim(0)
            let L = tokens.dim(1)
            let qkv = MLXArray.zeros([B, 1, L, 1])
            for cache in caches {
                _ = cache.updateAndAttend(
                    queries: qkv, keys: qkv, values: qkv,
                    scale: 1, sinks: nil)
            }
            for evaluation in recurrentState {
                let previous = evaluation.inputState(modelLayerIndex: 0)
                let conv = (previous?.conv ?? MLXArray.zeros([1, 1, 1])) + 1
                let ssm = (previous?.ssm ?? MLXArray.zeros([1, 1, 1, 1])) + 1
                try! evaluation.stage(modelLayerIndex: 0, conv: conv, ssm: ssm)
            }
            // Greedy token 1 for every row/position.
            return broadcast(
                MLXArray([Float(0), Float(1)]).reshaped(1, 1, 2),
                to: [B, L, 2])
        }
    }

    private final class RecurrentMultimodalFixtureModel:
        CBv2PositionedMultimodalSteppableModel, CBv2PositionedRecurrentSteppableModel
    {
        let cbv2Capabilities = CBv2ModelCapabilities.initialRecurrentTarget
        let recurrentStateSpec: CBv2RecurrentStateSpec? = CBv2RecurrentStateSpec(layers: [
            CBv2RecurrentLayerStateSpec(
                modelLayerIndex: 0,
                convShape: [1, 1, 1], convDType: .float32,
                ssmShape: [1, 1, 1, 1], ssmDType: .float32)
        ])
        var supportsMultimodalPrefill: Bool { true }

        func embedPromptTokens(_ tokens: MLXArray) -> MLXArray {
            broadcast(tokens.asType(.float32)[0..., 0..., .newAxis], to: [1, tokens.dim(1), 1])
        }

        func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
            preconditionFailure("fixture requires recurrent state")
        }

        func forward(
            tokens: MLXArray, inputEmbeddings: MLXArray,
            caches: [CBv2AttendingLayerCache]
        ) -> MLXArray {
            preconditionFailure("fixture requires recurrent state")
        }

        func forward(
            tokens: MLXArray, caches: [CBv2AttendingLayerCache],
            recurrentState: [CBv2RecurrentStateEvaluation]
        ) -> MLXArray {
            positionedForward(
                tokens: tokens, caches: caches, recurrentState: recurrentState,
                positionIds: nil)
        }

        func forward(
            tokens: MLXArray, caches: [CBv2AttendingLayerCache],
            recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
        ) -> MLXArray {
            positionedForward(
                tokens: tokens, caches: caches, recurrentState: recurrentState,
                positionIds: positionIds)
        }

        func forward(
            tokens: MLXArray, inputEmbeddings: MLXArray,
            caches: [CBv2AttendingLayerCache],
            recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
        ) -> MLXArray {
            positionedForward(
                tokens: tokens, caches: caches, recurrentState: recurrentState,
                positionIds: positionIds)
        }

        private func positionedForward(
            tokens: MLXArray, caches: [CBv2AttendingLayerCache],
            recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
        ) -> MLXArray {
            precondition(positionIds?.shape == [3, 1, tokens.dim(1)])
            let qkv = MLXArray.zeros([1, 1, tokens.dim(1), 1])
            for cache in caches {
                _ = cache.updateAndAttend(
                    queries: qkv, keys: qkv, values: qkv, scale: 1, sinks: nil)
            }
            for evaluation in recurrentState {
                let previous = evaluation.inputState(modelLayerIndex: 0)
                try! evaluation.stage(
                    modelLayerIndex: 0,
                    conv: (previous?.conv ?? MLXArray.zeros([1, 1, 1])) + 1,
                    ssm: (previous?.ssm ?? MLXArray.zeros([1, 1, 1, 1])) + 1)
            }
            return broadcast(
                MLXArray([Float(0), Float(1)]).reshaped(1, 1, 2),
                to: [1, tokens.dim(1), 2])
        }
    }

    private var oneLayerSpec: CBv2RecurrentStateSpec {
        CBv2RecurrentStateSpec(layers: [
            CBv2RecurrentLayerStateSpec(
                modelLayerIndex: 0,
                convShape: [1, 3, 8], convDType: .float16,
                ssmShape: [1, 2, 4, 4], ssmDType: .float32)
        ])
    }

    private func stage(
        _ value: Float, on state: CBv2RecurrentRequestState
    ) throws -> CBv2RecurrentStateEvaluation {
        let evaluation = try state.bind()
        try evaluation.stage(
            modelLayerIndex: 0,
            conv: MLXArray.full([1, 3, 8], values: MLXArray(value), dtype: .float16),
            ssm: MLXArray.full([1, 2, 4, 4], values: MLXArray(value), dtype: .float32))
        _ = try evaluation.evaluate()
        return evaluation
    }

    private func scalar(_ array: MLXArray?) -> Float {
        array![0, 0, 0].item(Float.self)
    }

    func testIndependentRowsAndJoinLeaveComposition() throws {
        let a = try CBv2RecurrentRequestState(spec: oneLayerSpec)
        let b = try CBv2RecurrentRequestState(spec: oneLayerSpec)
        try stage(11, on: a).commit()
        try stage(29, on: b).commit()

        XCTAssertEqual(scalar(a.state(modelLayerIndex: 0)?.conv), 11)
        XCTAssertEqual(scalar(b.state(modelLayerIndex: 0)?.conv), 29)

        // Join in reverse order, leave A, then rejoin. Object ownership, not
        // batch position, determines which recurrent history advances.
        let joined = [b, a]
        try stage(30, on: joined[0]).commit()
        XCTAssertEqual(scalar(a.state(modelLayerIndex: 0)?.conv), 11)
        XCTAssertEqual(scalar(b.state(modelLayerIndex: 0)?.conv), 30)
        try stage(12, on: a).commit()
        XCTAssertEqual(scalar(a.state(modelLayerIndex: 0)?.conv), 12)
        XCTAssertEqual(scalar(b.state(modelLayerIndex: 0)?.conv), 30)
    }

    func testChainedOneTokenRollbackRestoresExactPriorState() throws {
        let state = try CBv2RecurrentRequestState(spec: oneLayerSpec)
        try stage(1, on: state).commit()

        let accepted = try stage(2, on: state)
        let speculative = try stage(3, on: state)
        try accepted.commit()
        try speculative.rollback()

        let restored = state.state(modelLayerIndex: 0)!
        XCTAssertEqual(scalar(restored.conv), 2)
        XCTAssertEqual(restored.ssm![0, 0, 0, 0].item(Float.self), 2)
        try state.release()
        XCTAssertTrue(state.isReleased)
    }

    func testAdmissionChargesFixedStateOnceAndReleaseRestoresCapacity() throws {
        let kind = CBv2LayerKind(
            attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        let admission = AdmissionV2(
            layerKinds: [kind], bytesCapacity: 216,
            config: .init(
                watermarkFraction: 0, elementBytes: 2,
                fixedBytesPerRequest: 100))

        XCTAssertEqual(admission.estimatedBytes(forTokens: 4), 116)
        XCTAssertFalse(admission.canEverFit(promptTokens: 30, maxTokens: 0))
        try admission.reserve(id: CBv2RequestID(1), additionalTokens: 4)
        XCTAssertEqual(admission.bytesReserved, 116)
        XCTAssertThrowsError(
            try admission.reserve(id: CBv2RequestID(2), additionalTokens: 1))
        admission.releaseAll(id: CBv2RequestID(1))
        XCTAssertEqual(admission.bytesReserved, 0)
        try admission.reserve(id: CBv2RequestID(2), additionalTokens: 1)
        XCTAssertEqual(admission.bytesReserved, 104)
        admission.releaseAll(id: CBv2RequestID(2))
    }

    func testSchedulerPreemptionReleasesVictimFixedStateAccounting() throws {
        let kind = CBv2LayerKind(
            attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        let admission = AdmissionV2(
            layerKinds: [kind], bytesCapacity: 208,
            config: .init(
                watermarkFraction: 0, elementBytes: 2,
                fixedBytesPerRequest: 100))
        let scheduler = SchedulerV2(
            config: .init(
                maxConcurrentRequests: 2,
                maxBatchedTokensPerStep: 2,
                prefillChunkSize: 1,
                maxWaiting: 2),
            capacity: admission)
        try scheduler.enqueue(
            CBv2Request(
                id: CBv2RequestID(1), promptTokens: [1, 2], maxTokens: 1,
                priority: 10))
        try scheduler.enqueue(
            CBv2Request(
                id: CBv2RequestID(2), promptTokens: [3, 4], maxTokens: 1,
                priority: 0))

        let first = scheduler.plan()
        XCTAssertEqual(first.assignments.count, 2)
        XCTAssertEqual(admission.bytesReserved, 208)

        let second = scheduler.plan()
        XCTAssertEqual(second.preemptions, [CBv2RequestID(2)])
        XCTAssertEqual(second.assignments.map(\.id), [CBv2RequestID(1)])
        XCTAssertEqual(admission.bytesReserved, 108)
        admission.releaseAll(id: CBv2RequestID(1))
        XCTAssertEqual(admission.bytesReserved, 0)
    }

    func testEngineFinishRollsBackChainedGenerationAndReleasesState() async throws {
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        ]
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 1 << 20, kvDType: .float32))
        let engine = EngineV2(
            model: RecurrentFixtureModel(),
            layerKinds: kinds,
            backend: backend,
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: 8,
                prefillChunkSize: 8,
                maxWaiting: 2),
            admissionConfig: .init(watermarkFraction: 0))

        let stream = try engine.submit(
            CBv2Request(
                id: CBv2RequestID(99), promptTokens: [1], maxTokens: 1))
        var finished = false
        for await event in stream {
            if case .finished = event { finished = true }
        }
        XCTAssertTrue(finished)
        await engine.shutdown()

        XCTAssertTrue(engine.loopForTesting.recurrentStates.isEmpty)
        XCTAssertEqual(backend.bytesInUse, 0)
        XCTAssertEqual(backend.bytesReserved, 0)
        XCTAssertEqual(engine.capacity().kvBytesInUse, 0)
        XCTAssertEqual(engine.capacity().kvBytesReserved, 0)
    }

    func testCausalVisionPrefillUsesAndCleansRequestOwnedRecurrentState() async throws {
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: 1, kvHeads: 1, queryHeads: 1)
        ]
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 1 << 20, kvDType: .float32))
        let engine = EngineV2(
            model: RecurrentMultimodalFixtureModel(),
            layerKinds: kinds,
            backend: backend,
            cacheProvider: CBv2LayerCacheBank(layerKinds: kinds),
            sampler: CBv2GreedySampler(),
            schedulerConfig: .init(
                maxConcurrentRequests: 1,
                maxBatchedTokensPerStep: 2,
                prefillChunkSize: 2,
                maxWaiting: 2),
            admissionConfig: .init(watermarkFraction: 0))
        let prompt = [1, 7, 7, 7, 2]
        let positions = CBv2PositionState(
            promptPositionIds: MLXArray(Array(repeating: Int32(0), count: 15))
                .reshaped([3, 1, 5]),
            decodeDeltas: [-1])
        let media = CBv2MultimodalInput(
            spans: [CBv2ImageSpan(tokenOffset: 1, length: 3)],
            attention: .causal,
            positionState: positions
        ) { [MLXArray.ones([1, 3, 1])] }
        let stream = try engine.submit(
            CBv2Request(
                id: CBv2RequestID(199), promptTokens: prompt,
                maxTokens: 1, multimodal: media,
                positionState: positions))
        for await _ in stream {}
        await engine.shutdown()

        XCTAssertTrue(engine.loopForTesting.recurrentStates.isEmpty)
        XCTAssertEqual(backend.bytesReserved, 0)
        XCTAssertEqual(engine.capacity().kvBytesInUse, 0)
    }

    private func deltaSpec() -> CBv2RecurrentStateSpec {
        CBv2RecurrentStateSpec(layers: [
            CBv2RecurrentLayerStateSpec(
                modelLayerIndex: 0,
                convShape: [1, 3, 64], convDType: .bfloat16,
                ssmShape: [1, 4, 16, 32], ssmDType: .float32)
        ])
    }

    private struct DeltaInputs {
        let q: MLXArray
        let k: MLXArray
        let v: MLXArray
        let a: MLXArray
        let b: MLXArray
        let aLog: MLXArray
        let dtBias: MLXArray
    }

    private func makeDeltaInputs(B: Int = 1, T: Int, seed: UInt64) -> DeltaInputs {
        MLXRandom.seed(seed)
        return DeltaInputs(
            q: MLXRandom.normal([B, T, 2, 32]).asType(.bfloat16),
            k: MLXRandom.normal([B, T, 2, 32]).asType(.bfloat16),
            v: MLXRandom.normal([B, T, 4, 16]).asType(.bfloat16),
            a: MLXRandom.normal([B, T, 4]).asType(.bfloat16),
            b: MLXRandom.normal([B, T, 4]).asType(.bfloat16),
            aLog: (MLXArray.ones([4]) * Float(0.1)).asType(.bfloat16),
            dtBias: MLXArray.zeros([4], dtype: .bfloat16))
    }

    private func runDelta(
        inputs: DeltaInputs, chunks: [Int], state: CBv2RecurrentRequestState
    ) throws -> MLXArray {
        var outputs: [MLXArray] = []
        var start = 0
        for count in chunks {
            let evaluation = try state.bind()
            let inputState = evaluation.inputState(modelLayerIndex: 0)
            let (output, newSSM) = gatedDeltaUpdate(
                q: inputs.q[0..., start ..< start + count],
                k: inputs.k[0..., start ..< start + count],
                v: inputs.v[0..., start ..< start + count],
                a: inputs.a[0..., start ..< start + count],
                b: inputs.b[0..., start ..< start + count],
                aLog: inputs.aLog,
                dtBias: inputs.dtBias,
                state: inputState?.ssm)
            // The transaction treats conv and SSM tensors uniformly. The
            // conv sentinel advances independently of the recurrence.
            let conv = (inputState?.conv ?? MLXArray.zeros([1, 3, 64], dtype: .bfloat16))
                + Float(count)
            try evaluation.stage(modelLayerIndex: 0, conv: conv, ssm: newSSM)
            _ = try evaluation.evaluate()
            eval(output)
            try evaluation.commit()
            outputs.append(output)
            start += count
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 1)
    }

    func testGatedDeltaChunkPartitionEquivalenceThroughTransactions() throws {
        let inputs = makeDeltaInputs(T: 16, seed: 90210)
        let wholeState = try CBv2RecurrentRequestState(spec: deltaSpec())
        let chunkedState = try CBv2RecurrentRequestState(spec: deltaSpec())

        let whole = try runDelta(inputs: inputs, chunks: [16], state: wholeState)
        let chunked = try runDelta(inputs: inputs, chunks: [3, 5, 1, 7], state: chunkedState)
        XCTAssertTrue(allClose(whole, chunked, rtol: 1e-4, atol: 1e-4).item(Bool.self))

        let wholeFinal = wholeState.state(modelLayerIndex: 0)!
        let chunkedFinal = chunkedState.state(modelLayerIndex: 0)!
        XCTAssertEqual(wholeFinal.conv!.dtype, .bfloat16)
        XCTAssertEqual(wholeFinal.ssm!.dtype, .float32)
        XCTAssertTrue(
            allClose(wholeFinal.conv!, chunkedFinal.conv!, rtol: 0, atol: 0).item(Bool.self))
        XCTAssertTrue(
            allClose(wholeFinal.ssm!, chunkedFinal.ssm!, rtol: 1e-5, atol: 1e-6)
                .item(Bool.self))
    }

    func testRectangularDeltaDecodeWithUnequalHistoriesMatchesSoloRows() throws {
        let historyA = makeDeltaInputs(T: 4, seed: 100)
        let historyB = makeDeltaInputs(T: 9, seed: 200)
        let decodeA = makeDeltaInputs(T: 1, seed: 300)
        let decodeB = makeDeltaInputs(T: 1, seed: 400)
        let soloA = try CBv2RecurrentRequestState(spec: deltaSpec())
        let soloB = try CBv2RecurrentRequestState(spec: deltaSpec())
        _ = try runDelta(inputs: historyA, chunks: [4], state: soloA)
        _ = try runDelta(inputs: historyB, chunks: [3, 6], state: soloB)
        let expectedA = try runDelta(inputs: decodeA, chunks: [1], state: soloA)
        let expectedB = try runDelta(inputs: decodeB, chunks: [1], state: soloB)

        let batchedA = try CBv2RecurrentRequestState(spec: deltaSpec())
        let batchedB = try CBv2RecurrentRequestState(spec: deltaSpec())
        _ = try runDelta(inputs: historyA, chunks: [2, 2], state: batchedA)
        _ = try runDelta(inputs: historyB, chunks: [9], state: batchedB)
        let evalA = try batchedA.bind()
        let evalB = try batchedB.bind()
        let states = [evalA, evalB].map { $0.inputState(modelLayerIndex: 0)! }
        let batchedInputs = DeltaInputs(
            q: concatenated([decodeA.q, decodeB.q], axis: 0),
            k: concatenated([decodeA.k, decodeB.k], axis: 0),
            v: concatenated([decodeA.v, decodeB.v], axis: 0),
            a: concatenated([decodeA.a, decodeB.a], axis: 0),
            b: concatenated([decodeA.b, decodeB.b], axis: 0),
            aLog: decodeA.aLog,
            dtBias: decodeA.dtBias)
        let (batched, newSSM) = gatedDeltaUpdate(
            q: batchedInputs.q, k: batchedInputs.k, v: batchedInputs.v,
            a: batchedInputs.a, b: batchedInputs.b,
            aLog: batchedInputs.aLog, dtBias: batchedInputs.dtBias,
            state: concatenated(states.map { $0.ssm! }, axis: 0))
        try evalA.stage(modelLayerIndex: 0, conv: states[0].conv!, ssm: newSSM[0 ..< 1])
        try evalB.stage(modelLayerIndex: 0, conv: states[1].conv!, ssm: newSSM[1 ..< 2])
        _ = try evalA.evaluate()
        _ = try evalB.evaluate()
        eval(batched)
        try evalA.commit()
        try evalB.commit()

        XCTAssertTrue(
            allClose(batched[0 ..< 1], expectedA, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        XCTAssertTrue(
            allClose(batched[1 ..< 2], expectedB, rtol: 1e-4, atol: 1e-4).item(Bool.self))
    }
}
