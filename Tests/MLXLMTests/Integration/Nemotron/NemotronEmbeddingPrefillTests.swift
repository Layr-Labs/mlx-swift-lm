import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite(.tags(.integration))
struct NemotronEmbeddingPrefillTests {
    private func model() -> NemotronHModel {
        NemotronHModel(
            NemotronHConfiguration(
                vocabSize: 32, hiddenSize: 64,
                numHiddenLayers: 2, numAttentionHeads: 2, numKeyValueHeads: 1,
                mambaNumHeads: 4, mambaHeadDim: 16, ssmStateSize: 16,
                convKernel: 4, nGroups: 2, intermediateSize: 128,
                moeIntermediateSize: 64, moeSharedExpertIntermediateSize: 64,
                nRoutedExperts: 4, numExpertsPerTok: 2, hybridOverridePattern: "M*",
                headDim: 64))
    }

    private struct Result {
        let last: MLXArray
        let decode: MLXArray
        let conv: MLXArray
        let ssm: MLXArray
        let keys: MLXArray
        let values: MLXArray
        let offset: Int
    }

    private func run(
        _ model: NemotronHModel, tokens: [Int32], replacements: [Int32]?,
        chunks: [Int], injectInvalid: Bool = false
    ) throws -> Result {
        let kinds = model.cbv2LayerKinds
        let backend = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 16 << 20, dtype: .float32, maxPrefillChunk: 32,
                nominalMaxSequenceLength: 64))
        let rows = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: tokens.count, maxLength: 64)
        defer { backend.release(rows) }
        let storage = backend.makeLayerCaches()
        for i in storage.indices { storage[i].setRows([rows[i]!]) }
        let caches: [KVCache] = storage.map { $0 as KVCache }
        let state = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
        var start = 0
        var last = MLXArray.zeros([1])
        for count in chunks {
            let input = MLXArray(Array(tokens[start ..< start + count])).reshaped(1, count)
            let embedding = replacements.map {
                model.inputEmbeddings(
                    MLXArray(Array($0[start ..< start + count])).reshaped(1, count))
            }
            if let embedding { try model.validateInputEmbeddings(embedding, tokens: input) }
            let binding = try state.bind()
            let requirement: CBv2PrefillRequirement =
                start + count == tokens.count
                ? .lastPositionLogits : .evaluationOnly
            last = model.cbv2RecurrentPrefill(
                input, inputEmbedding: embedding, cache: caches,
                recurrentState: [binding], positionIds: nil, requirement: requirement)
            eval([last] + (try binding.evaluate()))
            try binding.commit()
            start += count
        }
        #expect(start == tokens.count)
        let next = try state.bind()
        if injectInvalid {
            let existing = try #require(state.state(modelLayerIndex: 0))
            let beforeConv = try #require(existing.conv) + 0
            let beforeSSM = try #require(existing.ssm) + 0
            let before = rows[0]!.snapshot()
            let beforeKeys = before.keys + 0
            let beforeValues = before.values + 0
            eval(beforeConv, beforeSSM, beforeKeys, beforeValues)
            let attempted = MLXArray([Int32(7)]).reshaped(1, 1)
            for bad in [
                MLXArray.zeros([1, 2, 64]), MLXArray.zeros([1, 1, 63]),
                MLXArray.zeros([1, 1, 64], dtype: .bfloat16),
            ] {
                #expect(throws: NemotronHModel.InputEmbeddingError.self) {
                    try model.prefillWithEmbeddings(
                        attempted, embeddings: bad, cache: caches,
                        recurrentState: [next], requirement: .lastPositionLogits)
                }
                let after = rows[0]!.snapshot()
                let recurrent = try #require(state.state(modelLayerIndex: 0))
                let afterConv = try #require(recurrent.conv)
                let afterSSM = try #require(recurrent.ssm)
                #expect(after.offset == before.offset)
                #expect(arrayEqual(beforeKeys, after.keys).item(Bool.self))
                #expect(arrayEqual(beforeValues, after.values).item(Bool.self))
                #expect(arrayEqual(beforeConv, afterConv).item(Bool.self))
                #expect(arrayEqual(beforeSSM, afterSSM).item(Bool.self))
            }
        }
        let decoded = model.cbv2Forward(
            MLXArray([Int32(7)]).reshaped(1, 1), caches: caches,
            recurrentState: [next])
        eval([decoded] + (try next.evaluate()))
        try next.commit()
        let recurrent = try #require(state.state(modelLayerIndex: 0))
        let conv = try #require(recurrent.conv)
        let ssm = try #require(recurrent.ssm)
        let snapshot = rows[0]!.snapshot()
        let keys = snapshot.keys + 0
        let values = snapshot.values + 0
        eval(keys, values)
        return Result(
            last: last, decode: decoded, conv: conv, ssm: ssm,
            keys: keys, values: values, offset: snapshot.offset)
    }

    private func compare(_ a: Result, _ b: Result, tolerance: Float = 0.0001) {
        #expect(a.offset == b.offset)
        for (lhs, rhs) in [
            (a.last, b.last), (a.decode, b.decode), (a.conv, b.conv),
            (a.ssm, b.ssm), (a.keys, b.keys), (a.values, b.values),
        ] {
            #expect(lhs.shape == rhs.shape)
            #expect(abs(lhs - rhs).max().item(Float.self) < tolerance)
        }
    }

    @Test func identicalEmbeddingsMatchTokenPrefill() throws {
        let m = model()
        let prompt: [Int32] = [1, 2, 3, 4]
        compare(
            try run(m, tokens: prompt, replacements: nil, chunks: [4]),
            try run(m, tokens: prompt, replacements: prompt, chunks: [4]))
    }

    @Test func replacementAndSplitSpanMatchIndependentTokenOracle() throws {
        let m = model()
        let placeholders: [Int32] = [1, 0, 0, 4]
        let actual: [Int32] = [1, 9, 10, 4]
        let oracle = try run(m, tokens: actual, replacements: nil, chunks: [4])
        compare(oracle, try run(m, tokens: placeholders, replacements: actual, chunks: [4]))
        compare(oracle, try run(m, tokens: placeholders, replacements: actual, chunks: [2, 2]))
    }

    @Test func invalidEmbeddingsThrowBeforeCacheMutation() throws {
        let m = model()
        let input = MLXArray([Int32(1), 2]).reshaped(1, 2)
        for bad in [
            MLXArray.zeros([1, 1, 64]), MLXArray.zeros([1, 2, 63]),
            MLXArray.zeros([1, 2, 64], dtype: .int32),
        ] {
            #expect(throws: NemotronHModel.InputEmbeddingError.self) {
                try m.prefillWithEmbeddings(
                    input, embeddings: bad, cache: [],
                    recurrentState: [], requirement: .lastPositionLogits)
            }
        }
        #expect(throws: NemotronHModel.InputEmbeddingError.self) {
            try m.validateInputEmbeddings(MLXArray.zeros([1, 2, 64]), tokens: input.reshaped(2))
        }
        // Invalid shapes cannot reach even the missing attention-cache invariant;
        // no request state is required to reject them at the public boundary.
    }

    @Test func failedPrefillPreservesPopulatedStateAndContinuation() throws {
        let m = model()
        let prompt: [Int32] = [1, 2, 3, 4]
        compare(
            try run(m, tokens: prompt, replacements: prompt, chunks: [4]),
            try run(m, tokens: prompt, replacements: prompt, chunks: [4], injectInvalid: true))
    }

    @Test func validationUsesLoadedActivationDtypeIncludingQuantizedEmbedding() throws {
        let m = model()
        let tokens = MLXArray([Int32(1), 2]).reshaped(1, 2)
        let cast = m.castPredicate
        m.update(
            parameters: ModuleParameters.unflattened(
                m.parameters().flattened().map {
                    ($0.0, cast?($0.0) == false ? $0.1 : $0.1.asType(.bfloat16))
                }))
        let native = m.inputEmbeddings(tokens)
        #expect(native.dtype == .bfloat16)
        try m.validateInputEmbeddings(native, tokens: tokens)
        #expect(throws: NemotronHModel.InputEmbeddingError.self) {
            try m.validateInputEmbeddings(native.asType(.float32), tokens: tokens)
        }
        quantize(model: m, groupSize: 32, bits: 4) { _, module in module is Embedding }
        eval(m)
        let quantizedNative = m.inputEmbeddings(tokens)
        #expect(quantizedNative.dtype == .bfloat16)
        try m.validateInputEmbeddings(quantizedNative, tokens: tokens)
        #expect(throws: NemotronHModel.InputEmbeddingError.self) {
            try m.validateInputEmbeddings(
                MLXArray.zeros([1, 2, 64], dtype: .uint32), tokens: tokens)
        }
    }
}
