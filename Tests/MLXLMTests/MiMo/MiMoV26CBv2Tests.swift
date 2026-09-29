import Foundation
import MLX
import MLXLMCommon
import MLXNN

@testable import MLXLLM

#if !MIMO_CBV2_COMPONENT_PROBE
    import XCTest
#endif

private func withMiMoConstructionScope<Value>(
    _ body: (NativeConstructionScope) throws -> Value
) rethrows -> Value {
    let work = NativeConstructionScope()
    defer {
        // Unexpected failed completion is restart-only, including in this
        // dedicated native test process. Never deallocate its sole SDK owner.
        if work.snapshot.isRetainedFault { _ = Unmanaged.passRetained(work) }
    }
    return try body(work)
}

private final class MiMoV26ReadoutSpy: Linear {
    var inputShapes: [[Int]] = []
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        inputShapes.append(x.shape)
        return super.callAsFunction(x)
    }
}

private enum MiMoV26CBv2Checks {
    struct Failure: Error { let message: String }
    static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw Failure(message: message) }
    }
    static func rejected(_ body: () throws -> Void) throws {
        do {
            try body()
            throw Failure(message: "invalid adapter input accepted")
        } catch is MiMoV26CBv2Error {}
    }
    static func configuration(_ dtype: String = "float32") throws -> MiMoV26Configuration {
        var fields = try JSONDecoder().decode(
            [String: MiMoV26JSONValue].self,
            from: Data(
                """
                {"model_type":"mimo_v2","architectures":["MiMoV2ForCausalLM"],
                 "hidden_size":16,"intermediate_size":32,"moe_intermediate_size":8,
                 "vocab_size":64,"num_hidden_layers":2,"max_position_embeddings":64,
                 "sliding_window_size":3,"sliding_window":3,"num_nextn_predict_layers":3,
                 "hybrid_layer_pattern":[0,1],"moe_layer_freq":[0,1],
                 "partial_rotary_factor":0.5,"attention_value_scale":0.707,
                 "layernorm_epsilon":0.000001,"attention_projection_layout":"split",
                 "moe_router_dtype":"bfloat16","hidden_act":"silu","dtype":"float32",
                 "attention_bias":false,"tie_word_embeddings":false,"attention_dropout":0,
                 "scoring_func":"sigmoid","topk_method":"noaux_tc","n_routed_experts":4,
                 "num_experts_per_tok":2,"n_group":1,"topk_group":1,"norm_topk_prob":true,
                 "n_shared_experts":null,"routed_scaling_factor":null,
                 "num_attention_heads":4,"num_key_value_heads":1,"head_dim":8,"v_head_dim":4,
                 "swa_num_attention_heads":4,"swa_num_key_value_heads":2,"swa_head_dim":8,
                 "swa_v_head_dim":4,"rope_theta":10000000,"swa_rope_theta":10000,
                 "add_full_attention_sink_bias":false,"add_swa_attention_sink_bias":true,
                 "eos_token_id":3,"pad_token_id":0}
                """.utf8))
        fields["dtype"] = .string(dtype)
        return try MiMoV26Configuration(rawFields: fields)
    }
    static func fill(_ target: MiMoV26TextModel) throws {
        let values = target.parameters().flattened().map { name, array in
            let salt = name.utf8.reduce(0) { ($0 + Int($1)) % 997 }
            let data = (0 ..< array.size).map { i -> Float in
                if name.contains("norm.weight") { return 1 }
                return sin(Float((i * 13 + salt) % 109)) * 0.15
            }
            let dtype: DType =
                name.hasSuffix("e_score_correction_bias")
                ? .float32
                : name.hasSuffix("mlp.gate.weight") ? .bfloat16 : target.activationDType
            return (name, MLXArray(data, array.shape).asType(dtype))
        }
        try target.update(parameters: .unflattened(values), verify: .all)
    }
    static func fixture(_ type: String = "float32") throws -> (
        MiMoV26TextModel, MiMoV26CBv2Adapter, MiMoV26CBv2Backend
    ) {
        let target = try MiMoV26TextModel(configuration(type))
        try fill(target)
        let adapter = try MiMoV26CBv2Adapter(target: target)
        _ = try withMiMoConstructionScope { constructionWork in
            try adapter.probeNativeKVTypes(retaining: constructionWork)
        }
        return (target, adapter, try adapter.makeBackend(bytesCapacity: 16 << 20))
    }
    static func tokens(_ values: [Int]) -> MLXArray {
        MLXArray(values.map(Int32.init), [1, values.count])
    }
    static func near(_ actual: MLXArray, _ expected: MLXArray, dtype: DType, label: String) throws {
        eval(actual, expected)
        try require(actual.shape == expected.shape, label + " shape")
        let a = actual.asType(.float32)
        let b = expected.asType(.float32)
        let error = abs(a - b)
        let absolute: Float = dtype == .bfloat16 ? 0.02 : 5e-5
        let relative: Float = dtype == .bfloat16 ? 0.02 : 1e-4
        let maximum = error.max().item(Float.self)
        try require(
            maximum.isFinite && all(error .<= (absolute + relative * abs(b))).item(Bool.self),
            "\(label) maxAbsoluteError=\(maximum)")
    }
    static func rows(_ adapter: MiMoV26CBv2Adapter, _ backend: MiMoV26CBv2Backend) throws
        -> [CBv2SequenceKV?]
    {
        try backend.makeSequenceState(
            layerKinds: adapter.layerKinds, promptLength: 8, maxLength: 64)
    }
    static func cases() -> [(String, () throws -> Void)] {
        [
            (
                "native layer metadata and all unqualified serving capabilities fail closed",
                {
                    let (target, adapter, _) = try fixture()
                    try require(adapter.target === target, "adapter duplicated target")
                    let full = adapter.layerKinds[0]
                    let sliding = adapter.layerKinds[1]
                    try require(
                        full.headDim == 8 && full.valueHeadDim == 4 && full.kvHeads == 1
                            && full.attention == .full && !full.hasSinks, "full metadata")
                    try require(
                        sliding.headDim == 8 && sliding.valueHeadDim == 4 && sliding.kvHeads == 2
                            && sliding.attention == .slidingWindow(3) && sliding.hasSinks,
                        "SWA metadata")
                    try require(
                        full.kvGeometry?.bytesPerToken(elementBytes: 4) == 48
                            && sliding.kvGeometry?.bytesPerToken(elementBytes: 4) == 96,
                        "native byte metadata")
                    let c = adapter.cbv2Capabilities
                    try require(
                        !c.supportsPrefixReuse && !c.supportsPagedKV && !c.supportsCompiledDecode
                            && !c.supportsMTP && !c.supportsPackedPrefill
                            && !c.supportsRecurrentCheckpointReuse, "premature capability")
                    let bank = CBv2LayerCacheBank(caches: adapter.makeCaches())
                    try require(
                        !bank.supportsMultimodalSpans && !bank.supportsPackedPrefill
                            && !bank.supportsMTPRectangularVerification, "cache capability widened")
                }
            ),
            (
                "FP32 and BF16 ordinary versus CBv2 chunks wrap and independent decode rows",
                {
                    for type in ["float32", "bfloat16"] {
                        let (target, adapter, backend) = try fixture(type)
                        let caches = adapter.makeCaches()
                        let a = try rows(adapter, backend)
                        let b = try rows(adapter, backend)
                        let ordinaryA = target.newCache()
                        let ordinaryB = target.newCache()
                        for (state, ordinary, chunks) in [
                            (a, ordinaryA, [[1, 2], [3, 4, 5]]),
                            (b, ordinaryB, [[8, 9, 10, 11], [12, 13, 14, 15, 16]]),
                        ] {
                            for ids in chunks {
                                try adapter.bindRows([state], caches: caches)
                                let actual = try adapter.forwardValidated(
                                    tokens: tokens(ids), caches: caches)
                                let expected = try target.forward(
                                    inputIDs: tokens(ids), cache: ordinary
                                ).logits
                                try near(
                                    actual, expected, dtype: target.activationDType,
                                    label: "\(type) chunk")
                            }
                        }
                        try adapter.bindRows([a, b], caches: caches)
                        let actual = try adapter.forwardValidated(
                            tokens: MLXArray([Int32(6), 17], [2, 1]), caches: caches)
                        let expectedA = try target.forward(inputIDs: tokens([6]), cache: ordinaryA)
                            .logits
                        let expectedB = try target.forward(inputIDs: tokens([17]), cache: ordinaryB)
                            .logits
                        try near(
                            actual[0 ..< 1], expectedA, dtype: target.activationDType,
                            label: "\(type) rowA offset5")
                        try near(
                            actual[1 ..< 2], expectedB, dtype: target.activationDType,
                            label: "\(type) rowB offset9")
                        try adapter.bindRows([b, a], caches: caches)
                        let reordered = try adapter.forwardValidated(
                            tokens: MLXArray([Int32(18), 7], [2, 1]), caches: caches)
                        try near(
                            reordered[0 ..< 1],
                            target.forward(inputIDs: tokens([18]), cache: ordinaryB).logits,
                            dtype: target.activationDType, label: "\(type) reorderedB")
                        try near(
                            reordered[1 ..< 2],
                            target.forward(inputIDs: tokens([7]), cache: ordinaryA).logits,
                            dtype: target.activationDType, label: "\(type) reorderedA")
                        try adapter.bindRows([a], caches: caches)
                        try near(
                            adapter.forwardValidated(tokens: tokens([8]), caches: caches),
                            target.forward(inputIDs: tokens([8]), cache: ordinaryA).logits,
                            dtype: target.activationDType, label: "\(type) shrink")
                        backend.release(a)
                        backend.release(b)
                    }
                }
            ),
            (
                "explicit B2 prefill rows equal isolated ordinary targets without enabling packing",
                {
                    let (target, adapter, backend) = try fixture()
                    let caches = adapter.makeCaches()
                    let a = try rows(adapter, backend)
                    let b = try rows(adapter, backend)
                    try adapter.bindRows([a, b], caches: caches)
                    let idsA = [1, 2, 3, 4, 5]
                    let idsB = [9, 8, 7, 6, 5]
                    let batched = MLXArray((idsA + idsB).map(Int32.init), [2, 5])
                    let output = try adapter.forwardValidated(tokens: batched, caches: caches)
                    try near(
                        output[0 ..< 1], target.forward(inputIDs: tokens(idsA)).logits,
                        dtype: .float32, label: "B2 prefill A")
                    try near(
                        output[1 ..< 2], target.forward(inputIDs: tokens(idsB)).logits,
                        dtype: .float32, label: "B2 prefill B")
                    backend.release(a)
                    backend.release(b)
                }
            ),
            (
                "causal prepared embeddings equal text without any span capability",
                {
                    let (target, adapter, backend) = try fixture()
                    let ids = tokens([1, 2, 3, 4, 5])
                    let textRows = try rows(adapter, backend)
                    let embeddedRows = try rows(adapter, backend)
                    let textCaches = adapter.makeCaches()
                    let embeddedCaches = adapter.makeCaches()
                    try adapter.bindRows([textRows], caches: textCaches)
                    try adapter.bindRows([embeddedRows], caches: embeddedCaches)
                    let text = try adapter.forwardValidated(tokens: ids, caches: textCaches)
                    let embedded = try adapter.forwardValidated(
                        tokens: ids, inputEmbeddings: target.model.embedTokens(ids),
                        caches: embeddedCaches)
                    try near(text, embedded, dtype: .float32, label: "embedding seam")
                    try require(
                        embeddedCaches.allSatisfy { !($0 is any CBv2SpanMaskBinding) },
                        "Gemma span overlay exposed")
                    backend.release(textRows)
                    backend.release(embeddedRows)
                }
            ),
            (
                "prompt intermediate skips readout and frontier projects one row while preserving all KV",
                {
                    let (target, adapter, backend) = try fixture()
                    let state = try rows(adapter, backend)
                    let caches = adapter.makeCaches()
                    let spy = MiMoV26ReadoutSpy(16, 64, bias: false)
                    try spy.update(parameters: target.lmHead!.parameters(), verify: .all)
                    target.update(modules: ModuleChildren.unflattened([("lm_head", spy as Module)]))
                    try adapter.bindRows([state], caches: caches)
                    let intermediate = try adapter.prefillValidated(
                        tokens: tokens([1, 2, 3, 4]), caches: caches, requirement: .evaluationOnly)
                    eval(intermediate)
                    try require(
                        spy.inputShapes.isEmpty && intermediate.shape == [1],
                        "intermediate constructed vocabulary logits")
                    let frontier = try adapter.prefillValidated(
                        tokens: tokens([5, 6]), caches: caches, requirement: .lastPositionLogits)
                    eval(frontier)
                    try require(
                        spy.inputShapes == [[1, 16]] && frontier.shape == [1, 64],
                        "frontier projected discarded positions")
                    try require(
                        state.allSatisfy { $0?.absoluteOffset == 6 },
                        "prompt did not commit all positions")
                    let reference = try target.forward(inputIDs: tokens([1, 2, 3, 4, 5, 6]))
                    try near(
                        frontier, reference.logits[0..., -1, 0...], dtype: .float32,
                        label: "frontier logits")
                    backend.release(state)
                }
            ),
            (
                "real native probe accepts empty FP16 placeholders and records FP32 BF16 storage",
                {
                    for type in ["float32", "bfloat16"] {
                        let (target, adapter, _) = try fixture(type)
                        let result = try withMiMoConstructionScope { constructionWork in
                            try adapter.probeNativeKVTypes(retaining: constructionWork)
                        }
                        try require(
                            result.layerDTypes == [target.activationDType, target.activationDType]
                                && result.observations.count == 4,
                            "native probe did not observe both phases")
                        try require(
                            result.observations.allSatisfy {
                                $0.keysShape.last == 8 && $0.valuesShape.last == 4
                            }, "probe widths")
                    }
                }
            ),
            (
                "malformed cache owner alias layout history or embeddings reject before writes",
                {
                    let (target, adapter, backend) = try fixture()
                    let state = try rows(adapter, backend)
                    let caches = adapter.makeCaches()
                    try adapter.bindRows([state], caches: caches)
                    let (_, other, _) = try fixture()
                    try rejected {
                        _ = try adapter.forwardValidated(
                            tokens: tokens([1]), caches: other.makeCaches())
                    }
                    try rejected {
                        _ = try adapter.forwardValidated(
                            tokens: tokens([1]), caches: [caches[0], caches[0]])
                    }
                    try rejected { try adapter.bindRows([[state[0], state[0]]], caches: caches) }
                    var wrong = state
                    wrong[1] = CBv2WindowedSequenceKV(
                        window: 3, kvHeads: 2, headDim: 8, valueHeadDim: 8)
                    try rejected { try adapter.bindRows([wrong], caches: caches) }
                    try rejected {
                        _ = try adapter.forwardValidated(
                            tokens: tokens([1]), inputEmbeddings: MLXArray.zeros([1, 1, 15]),
                            caches: caches)
                    }
                    try rejected { try adapter.validateTokenIDs([-1]) }
                    try rejected {
                        try adapter.validateTokenIDs([target.configuration.vocabularySize])
                    }
                    try require(
                        state.allSatisfy { $0?.absoluteOffset == 0 && $0?.byteCount == 0 },
                        "failed preflight mutated an earlier layer")
                    _ = try adapter.forwardValidated(tokens: tokens([1, 2]), caches: caches)
                    state[0]!.rollback(1)
                    try rejected {
                        _ = try adapter.forwardValidated(tokens: tokens([3]), caches: caches)
                    }
                    try require(
                        state[1]!.absoluteOffset == 2,
                        "stale-position failure advanced another layer")
                    backend.release(state)
                }
            ),
            (
                "identical-geometry foreign empty and nonempty rows cannot cross target ownership",
                {
                    let (_, a, backendA) = try fixture()
                    let (_, b, backendB) = try fixture()
                    let stateA = try rows(a, backendA)
                    let stateB = try rows(b, backendB)
                    let cachesA = a.makeCaches()
                    let cachesB = b.makeCaches()
                    try b.bindRows([stateB], caches: cachesB)
                    try rejected { try b.bindRows([stateA], caches: cachesB) }
                    try require(
                        cachesB[0].rows[0] === stateB[0]!,
                        "foreign empty bind partially replaced owner")
                    try a.bindRows([stateA], caches: cachesA)
                    let first = try a.forwardValidated(tokens: tokens([1, 2]), caches: cachesA)
                    eval(first)
                    let offsets = stateA.map { $0!.absoluteOffset }
                    let bytes = stateA.map { $0!.byteCount }
                    try rejected { try b.bindRows([stateA], caches: cachesB) }
                    // Simulate an incorrect internal composition caller bypassing
                    // external bindRows. Forward must independently reject it.
                    for index in cachesB.indices { cachesB[index].setRows([stateA[index]!]) }
                    try rejected {
                        _ = try b.forwardValidated(tokens: tokens([3]), caches: cachesB)
                    }
                    try require(
                        stateA.map { $0!.absoluteOffset } == offsets
                            && stateA.map { $0!.byteCount } == bytes,
                        "foreign populated row rejection mutated target A")
                    try require(
                        stateB.allSatisfy { $0!.absoluteOffset == 0 && $0!.byteCount == 0 },
                        "foreign rejection mutated target B")
                    try rejected { try backendB.releaseValidated(stateA) }
                    backendA.release(stateA)
                    backendB.release(stateB)
                }
            ),
            (
                "owner registration survives reorder join and cancellation but rejects released or spliced rows",
                {
                    let (_, adapter, backend) = try fixture()
                    let caches = adapter.makeCaches()
                    let a = try rows(adapter, backend)
                    let b = try rows(adapter, backend)
                    try rejected { try adapter.bindRows([[a[0], b[1]]], caches: caches) }
                    try adapter.bindRows([a, b], caches: caches)
                    eval(
                        try adapter.forwardValidated(
                            tokens: MLXArray([Int32(1), 2], [2, 1]), caches: caches))
                    try adapter.bindRows([b, a], caches: caches)
                    eval(
                        try adapter.forwardValidated(
                            tokens: MLXArray([Int32(3), 4], [2, 1]), caches: caches))
                    try backend.releaseValidated(a)
                    try rejected {
                        _ = try adapter.forwardValidated(
                            tokens: MLXArray([Int32(5), 6], [2, 1]), caches: caches)
                    }
                    try require(
                        b.allSatisfy { $0!.absoluteOffset == 2 },
                        "released-row failure advanced survivor")
                    try rejected { try adapter.bindRows([a], caches: caches) }
                    try adapter.bindRows([b], caches: caches)
                    eval(try adapter.forwardValidated(tokens: tokens([5]), caches: caches))
                    let c = try rows(adapter, backend)
                    try adapter.bindRows([c, b], caches: caches)
                    eval(
                        try adapter.forwardValidated(
                            tokens: MLXArray([Int32(7), 6], [2, 1]), caches: caches))
                    try require(
                        c.allSatisfy { $0!.absoluteOffset == 1 }
                            && b.allSatisfy { $0!.absoluteOffset == 4 },
                        "join did not preserve independent histories")
                    backend.release(b)
                    backend.release(c)
                    try require(
                        backend.bytesReserved == 0 && backend.bytesInUse == 0,
                        "cancel lifecycle leaked native owner")
                }
            ),
            (
                "registered sliding rows reject nonzero empty or inadequate retained history",
                {
                    let (_, adapter, backend) = try fixture()
                    let caches = adapter.makeCaches()
                    for inadequate in [false, true] {
                        let state = try rows(adapter, backend)
                        let count = inadequate ? 4 : 2
                        _ = state[0]!.update(
                            keys: MLXArray.zeros([1, 1, count, 8]),
                            values: MLXArray.zeros([1, 1, count, 4]))
                        if inadequate {
                            _ = state[1]!.update(
                                keys: MLXArray.zeros([1, 2, 5, 8]),
                                values: MLXArray.zeros([1, 2, 5, 4]))
                            state[1]!.rollback(1)
                        } else {
                            state[1]!.fastForward(to: 2)
                        }
                        let offsets = state.map { $0!.absoluteOffset }
                        try rejected { try adapter.bindRows([state], caches: caches) }
                        try require(
                            state.map { $0!.absoluteOffset } == offsets
                                && caches.allSatisfy { $0.rows.isEmpty },
                            "missing history rejection partially rebound caches")
                        backend.release(state)
                    }
                }
            ),
            (
                "weak row metadata is bounded and pruned on registration and removal",
                {
                    let ledger = MiMoV26CBv2RowLedger()
                    let owner = UUID()
                    weak var observed: CBv2FullSequenceKV?
                    for _ in 0 ..< 64 {
                        var row: CBv2FullSequenceKV? = CBv2FullSequenceKV(
                            promptLength: 1, maxLength: 2,
                            kvHeads: 1, headDim: 8, valueHeadDim: 4)
                        observed = row
                        try ledger.register([row], backend: owner)
                        try require(
                            ledger.metadataEntryCount == 1,
                            "dead metadata accumulated during registration")
                        row = nil
                        try require(observed == nil, "ledger became a strong native row owner")
                    }
                    let live = CBv2FullSequenceKV(
                        promptLength: 1, maxLength: 2, kvHeads: 1, headDim: 8, valueHeadDim: 4)
                    try ledger.register([live], backend: owner)
                    var retiring: CBv2FullSequenceKV? = CBv2FullSequenceKV(
                        promptLength: 1, maxLength: 2,
                        kvHeads: 1, headDim: 8, valueHeadDim: 4)
                    try ledger.register([retiring], backend: UUID())
                    try require(
                        ledger.metadataEntryCount == 2 && live.byteCount == 0
                            && retiring!.byteCount == 0,
                        "metadata-only test allocated/duplicated tensor state")
                    retiring = nil
                    try ledger.remove([live], backend: owner)
                    try require(
                        ledger.metadataEntryCount == 0,
                        "retirement did not prune other dead weak entries")
                }
            ),
            (
                "serving backend requires completed native probe before reserving precision",
                {
                    for type in ["float32", "bfloat16"] {
                        let target = try MiMoV26TextModel(configuration(type))
                        try fill(target)
                        let adapter = try MiMoV26CBv2Adapter(target: target)
                        try rejected { _ = try adapter.makeBackend(bytesCapacity: 16 << 20) }
                        try rejected { try adapter.bindRows([], caches: adapter.makeCaches()) }
                        let observed = try withMiMoConstructionScope { constructionWork in
                            try adapter.probeNativeKVTypes(retaining: constructionWork)
                        }
                        let backend = try adapter.makeBackend(bytesCapacity: 16 << 20)
                        let state = try rows(adapter, backend)
                        let elementBytes =
                            Set(observed.layerDTypes).count == 1 ? observed.layerDTypes[0].size : 4
                        let expected = (64 * 1 * (8 + 4) + 3 * 2 * (8 + 4)) * elementBytes
                        try require(
                            backend.bytesReserved == expected && backend.bytesInUse == 0,
                            "backend reserved activation-based rather than observed native bytes")
                        backend.release(state)
                        try require(
                            backend.bytesReserved == 0, "probe-bound backend reservation leaked")
                    }
                }
            ),
        ]
    }
}

#if MIMO_CBV2_COMPONENT_PROBE
    @main
    private struct MiMoV26CBv2Probe {
        static func main() throws {
            let cases = MiMoV26CBv2Checks.cases()
            var failures = 0
            for (name, body) in cases {
                do {
                    try body()
                    print("PASS \(name)")
                } catch {
                    failures += 1
                    print("FAIL \(name): \(error)")
                }
            }
            print(
                "RESULT discovered=\(cases.count) passed=\(cases.count-failures) failed=\(failures) skipped=0"
            )
            if failures > 0 { throw MiMoV26CBv2Checks.Failure(message: "MiMo CBv2 probe failed") }
        }
    }
#else
    final class MiMoV26CBv2Tests: XCTestCase {
        func testNativeContiguousAdapter() throws {
            for (name, body) in MiMoV26CBv2Checks.cases() {
                do { try body() } catch { XCTFail("\(name): \(error)") }
            }
        }
    }
#endif
