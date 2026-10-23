import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

private struct Failure: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}

private struct Result: Codable {
    let rows: Int
    let dtype: String
    let paged: Bool
    let tied: Bool
    let entry: String
    let comparisons: Int
    let backingChecks: Int
    let mismatches: [String]
}

private final class State {
    let kv: [CBv2SequenceKV?]
    let recurrent: CBv2RecurrentRequestState
    init(model: NemotronH35Model, backend: any CBv2KVBackend, prompt: Int) throws {
        kv = try backend.makeSequenceState(
            layerKinds: model.cbv2LayerKinds, promptLength: prompt, maxLength: 64)
        recurrent = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
    }
    func retire(_ backend: any CBv2KVBackend) throws {
        backend.release(kv)
        try recurrent.release()
    }
}

private func runCase(
    rows: Int, dtype: DType, paged: Bool, tied: Bool, logits: Bool,
    check: () throws -> Void
) throws -> Result {
    let config: [String: Any] = [
        "model_type": "nemotron_h", "vocab_size": 256, "hidden_size": 256,
        "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 64, "mamba_num_heads": 4, "mamba_head_dim": 64,
        "ssm_state_size": 32, "conv_kernel": 4, "n_groups": 2,
        "intermediate_size": 512, "moe_intermediate_size": 256,
        "moe_shared_expert_intermediate_size": 256, "n_routed_experts": 16,
        "num_experts_per_tok": 8, "layers_block_type": ["mamba", "moe", "mamba", "attention"],
        "norm_eps": 0.00001, "mamba_ssm_cache_dtype": "float32", "tie_word_embeddings": tied,
    ]
    MLXRandom.seed(350035)
    let model = NemotronH35Model(
        try JSONDecoder().decode(
            NemotronH35Configuration.self,
            from: JSONSerialization.data(withJSONObject: config)))
    let cast = model.castPredicate
    model.update(
        parameters: ModuleParameters.unflattened(
            model.parameters().flattened().map { name, value in
                // Router weights initialize to zero because real loading replaces them.
                // Use deterministic nonzero rows here to exercise request-dependent routes.
                let tensor =
                    name.hasSuffix("mixer.gate.weight")
                    ? MLXArray((0 ..< value.size).map { sin(Float($0) * 0.17) / 32 }, value.shape)
                    : value
                return (name, cast?(name) == false ? tensor : tensor.asType(dtype))
            }))
    quantize(model: model, groupSize: 64, bits: 4)
    eval(model)
    try check()
    let kinds = model.cbv2LayerKinds
    let backend: any CBv2KVBackend
    let caches: [any CBv2AttendingLayerCache]
    if paged {
        let pages = try PagedKVBackend(
            layerKinds: kinds,
            config: .init(
                capacityBytes: 128 << 20, dtype: dtype, maxPrefillChunk: 1,
                nominalMaxSequenceLength: 64, segmentSizeBytes: 64 << 10,
                layerDTypes: Array(repeating: dtype, count: kinds.count)))
        backend = pages
        let storage = pages.makeLayerCaches()
        let index = Dictionary(
            uniqueKeysWithValues: kinds.enumerated().map {
                ($0.element.modelLayerIndex!, $0.offset)
            })
        caches = model.newCacheV2 { layer, _ in storage[index[layer]!] }
    } else {
        backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 128 << 20, kvDType: dtype))
        caches = model.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) }
    }
    var reference = try (0 ..< rows).map {
        try State(model: model, backend: backend, prompt: $0 + 2)
    }
    var candidate = try (0 ..< rows).map {
        try State(model: model, backend: backend, prompt: $0 + 2)
    }
    defer {
        for cache in caches { cache.setRows([]) }
        for state in reference + candidate { try? state.retire(backend) }
    }
    func bytes(_ array: MLXArray) -> Data { array.asData(access: .copy).data }
    func forward(_ states: [State], tokens: [Int32], batch: Bool) throws -> MLXArray {
        for index in caches.indices { caches[index].setRows(states.map { $0.kv[index]! }) }
        let evaluations = try states.map { try $0.recurrent.bind() }
        let input = MLXArray(tokens, [states.count, 1])
        let kv = caches.map { $0 as! any KVCache }
        let output: MLXArray
        if logits {
            let residual = model.inputEmbeddings(input)
            output =
                try batch
                ? model.cbv2BatchedStageLogits(
                    input, inputEmbedding: residual, caches: kv, recurrentState: evaluations)
                : model.cbv2StageLogits(
                    input, inputEmbedding: residual, caches: kv,
                    recurrentState: evaluations, requirement: nil)
        } else {
            output =
                try batch
                ? model.cbv2BatchedStageResidual(input, caches: kv, recurrentState: evaluations)
                : model.cbv2StageResidual(
                    input, inputEmbedding: nil, caches: kv, recurrentState: evaluations)
        }
        let staged = try evaluations.flatMap { try $0.evaluate() }
        eval([output] + staged + kv.flatMap { $0.innerState() })
        try check()
        for evaluation in evaluations { try evaluation.commit() }
        return output
    }
    func warm(_ row: Int, length: Int, salt: Int) throws {
        for step in 0 ..< length {
            let token = Int32((row * 19 + step * 7 + salt) % 256)
            let a = try forward([reference[row]], tokens: [token], batch: false)
            let b = try forward([candidate[row]], tokens: [token], batch: false)
            guard bytes(a) == bytes(b) else { throw Failure("Singleton warmup oracle differs") }
        }
    }
    for row in 0 ..< rows { try warm(row, length: row + 2, salt: 3) }
    var comparisons = 0
    var backingChecks = 0
    var mismatches: [String] = []
    func compare(_ a: MLXArray, _ b: MLXArray, _ label: String) {
        comparisons += 1
        if a.shape != b.shape || a.dtype != b.dtype || bytes(a) != bytes(b) {
            mismatches.append(label)
        }
    }
    for step in 0 ..< 6 {
        // Leave one request idle, then bring it back at its own frontier.
        var order = (0 ..< rows).map { (step + (step % 2 == 0 ? $0 : rows - 1 - $0)) % rows }
        if rows > 1 && step % 3 == 1 { order.removeLast() }
        let tokens = (0 ..< rows).map { Int32(($0 * 29 + step * 11 + 5) % 256) }
        var expected: [Int: MLXArray] = [:]
        for row in order {
            expected[row] = try forward([reference[row]], tokens: [tokens[row]], batch: false)
        }
        let actual = try forward(
            order.map { candidate[$0] }, tokens: order.map { tokens[$0] }, batch: true)
        for (slot, row) in order.enumerated() {
            compare(expected[row]!, actual[slot ..< slot + 1], "step=\(step) row=\(row) output")
            for index in reference[row].kv.indices {
                let a = reference[row].kv[index]!.snapshot()
                let b = candidate[row].kv[index]!.snapshot()
                guard a.offset == b.offset else { throw Failure("KV frontier differs") }
                compare(a.keys, b.keys, "step=\(step) row=\(row) layer=\(index) keys")
                compare(a.values, b.values, "step=\(step) row=\(row) layer=\(index) values")
            }
            let a = reference[row].recurrent.confirmedStateSnapshot()!
            let b = candidate[row].recurrent.confirmedStateSnapshot()!
            for layer in a.keys.sorted() {
                guard a[layer]!.ssm!.dtype == .float32 && b[layer]!.ssm!.dtype == .float32 else {
                    throw Failure("Persistent SSM state is not float32")
                }
                if order.count > 1 {
                    for state in [b[layer]!.conv!, b[layer]!.ssm!] {
                        backingChecks += 1
                        guard let info = try state.evaluatedBufferInfo(), info.dataOffset == 0,
                            info.dataElements == state.size, info.isRowContiguous,
                            info.allocatedBytes
                                <= (try Memory.allocationFootprintUpperBound(
                                    byteCount: state.nbytes))
                        else {
                            throw Failure("Recurrent state retains another lane's backing")
                        }
                    }
                }
                compare(
                    a[layer]!.conv!, b[layer]!.conv!, "step=\(step) row=\(row) layer=\(layer) conv")
                compare(
                    a[layer]!.ssm!, b[layer]!.ssm!, "step=\(step) row=\(row) layer=\(layer) ssm")
            }
        }
        if step == 2 {
            // Retire one request, reuse its scheduler slot with a new history,
            // and verify every survivor against its independent singleton.
            for cache in caches { cache.setRows([]) }
            let row = rows - 1
            try reference[row].retire(backend)
            try candidate[row].retire(backend)
            reference[row] = try State(model: model, backend: backend, prompt: 5)
            candidate[row] = try State(model: model, backend: backend, prompt: 5)
            try warm(row, length: 5, salt: 71)
        }
    }
    return .init(
        rows: rows, dtype: String(describing: dtype), paged: paged, tied: tied,
        entry: logits ? "stageLogits" : "stageResidual", comparisons: comparisons,
        backingChecks: backingChecks, mismatches: mismatches)
}

@Suite(.tags(.integration), .serialized)
struct Nemotron35BatchDecodeModelTests {
    @Test func nativeOutputsAndOwnedStateMatchSingleton() throws {
        try Device.withDefaultDevice(.gpu) {
            try MLX.withError { error in
                for rows in [1, 2, 4, 8] {
                    for dtype in [DType.bfloat16, .float32] {
                        for paged in [false, true] {
                            for tied in [false, true] {
                                for logits in [false, true] {
                                    let result = try autoreleasepool {
                                        try runCase(
                                            rows: rows, dtype: dtype, paged: paged, tied: tied,
                                            logits: logits, check: { try error.check() })
                                    }
                                    #expect(
                                        result.mismatches.isEmpty,
                                        "rows=\(rows), dtype=\(dtype), paged=\(paged), tied=\(tied), logits=\(logits): \(result.mismatches)"
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
