import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Rectangular `[B, 1]` Super text decode against each request decoded alone,
/// on tiny synthetic ModelOpt fixtures. Requests have different prompt
/// lengths, one finishes early, one joins mid-stream and one is cancelled
/// while its row is in flight. Decode inputs are teacher-forced, so every
/// comparison sees identical token histories.
@Suite(.tags(.integration))
struct NemotronSuperBatchedDecodeTests {
    typealias Fixture = NemotronSuperBatchFixture

    /// A prompt, its decode inputs, the batched step before which it is
    /// prefilled and joins, and the decode during which it is cancelled (its
    /// row is computed, rolled back and released), if any.
    struct Request {
        let prompt: [Int32]
        let inputs: [Int32]
        var joinsAtStep = 0
        var cancelledAtDecode: Int? = nil
    }

    static let requests: [Request] = [
        Request(prompt: Fixture.tokens(5, seed: 1), inputs: Fixture.tokens(8, seed: 2)),
        Request(prompt: Fixture.tokens(11, seed: 3), inputs: Fixture.tokens(10, seed: 4)),
        Request(
            prompt: Fixture.tokens(3, seed: 5), inputs: Fixture.tokens(6, seed: 6),
            joinsAtStep: 3),
        Request(
            prompt: Fixture.tokens(8, seed: 7), inputs: Fixture.tokens(6, seed: 8),
            cancelledAtDecode: 3),
    ]

    struct Trace {
        /// Per request: logits of every committed decode.
        var logits: [[MLXArray]]
        /// Per request: committed recurrent state and KV at its end (after
        /// the rollback for the cancelled request).
        var snapshots: [[MLXArray]]
        var widestStep = 0
    }

    /// Each request alone: prefill, then one `[1, 1]` decode per input.
    static func serial(_ model: NemotronSuperModel, _ requests: [Request]) throws -> Trace {
        let runner = try NemotronSuperBatchRunner(model: model, maximumRows: 1, kvDType: .float32)
        var trace = Trace(logits: [], snapshots: [])
        for request in requests {
            let lane = try runner.lane(prompt: request.prompt, inputs: request.inputs)
            try runner.prefill(lane)
            for _ in 0 ..< (request.cancelledAtDecode ?? request.inputs.count) {
                try runner.decode([lane])
            }
            trace.logits.append(lane.logits)
            trace.snapshots.append(try lane.snapshot(kv: request.cancelledAtDecode == nil))
            try runner.release(lane)
        }
        return trace
    }

    /// The scripted concurrent schedule. Row order is rotated every step so
    /// no request keeps a fixed row.
    static func batched(
        _ model: NemotronSuperModel, _ requests: [Request], maximumRows: Int = 4
    ) throws -> Trace {
        let runner = try NemotronSuperBatchRunner(
            model: model, maximumRows: maximumRows, kvDType: .float32)
        let lanes = try requests.map { try runner.lane(prompt: $0.prompt, inputs: $0.inputs) }
        var trace = Trace(
            logits: Array(repeating: [], count: lanes.count),
            snapshots: Array(repeating: [], count: lanes.count))
        var active: [Int] = []
        var step = 0
        let lastJoin = requests.map(\.joinsAtStep).max() ?? 0
        while !active.isEmpty || step <= lastJoin {
            for index in lanes.indices where requests[index].joinsAtStep == step {
                try runner.prefill(lanes[index])
                active.append(index)
            }
            guard !active.isEmpty else {
                step += 1
                continue
            }
            let rotation = step % active.count
            let rows = Array(active[rotation...] + active[..<rotation])
            let outcomes = rows.map { index -> NemotronSuperBatchRunner.Outcome in
                requests[index].cancelledAtDecode == lanes[index].decoded ? .rollback : .commit
            }
            try runner.decode(rows.map { lanes[$0] }, outcomes: outcomes)
            trace.widestStep = max(trace.widestStep, rows.count)
            for (row, index) in rows.enumerated() {
                let lane = lanes[index]
                guard outcomes[row] == .rollback || lane.decoded == lane.inputs.count else {
                    continue
                }
                trace.logits[index] = lane.logits
                trace.snapshots[index] = try lane.snapshot(kv: outcomes[row] == .commit)
                try runner.release(lane)
                active.removeAll { $0 == index }
            }
            step += 1
        }
        return trace
    }

    private static func exact(_ batched: Trace, _ serial: Trace) {
        #expect(batched.logits.count == serial.logits.count)
        for (request, (lhs, rhs)) in zip(batched.logits, serial.logits).enumerated() {
            #expect(lhs.count == rhs.count, "request \(request) decode count")
            for (step, (a, b)) in zip(lhs, rhs).enumerated() {
                #expect(modelOptBitwiseEqual(a, b), "request \(request) decode \(step) logits")
            }
        }
        for (request, (lhs, rhs)) in zip(batched.snapshots, serial.snapshots).enumerated() {
            #expect(lhs.count == rhs.count && !lhs.isEmpty, "request \(request) snapshot")
            for (index, (a, b)) in zip(lhs, rhs).enumerated() {
                #expect(modelOptBitwiseEqual(a, b), "request \(request) state array \(index)")
            }
        }
    }

    // MARK: Exactness with row-exact products

    /// With every row-mixing product at its serial shape, batching, joining,
    /// finishing and cancelling never change a request's logits or state.
    @Test(arguments: [DType.bfloat16, .float32])
    func rowExactProductsMakeBatchedDecodeBitwiseSerial(dtype: DType) throws {
        let model = try Fixture.model(seed: 31, dtype: dtype, products: .exactSettings)
        let wrapped = try model.languageModel.installRowExactProducts(.all)
        // FP8: two Mamba layers, attention, output head; NVFP4 shared experts,
        // latent projections and routers of the two MoE layers.
        #expect(wrapped == 2 * 2 + 4 + 1 + 2 * 2 + 2 * 2 + 2)
        let serial = try Self.serial(model, Self.requests)
        let batched = try Self.batched(model, Self.requests)
        #expect(batched.widestStep == 4)
        for logits in serial.logits.joined() {
            #expect(!isNaN(logits).any().item(Bool.self))
        }
        Self.exact(batched, serial)
    }

    /// Sixteen rows: MLX switches BF16 products to GEMM and the routed experts
    /// to their sorted path at this width.
    @Test func rowExactProductsStayBitwiseAtSixteenRows() throws {
        let model = try Fixture.model(seed: 5, dtype: .bfloat16, products: .exactSettings)
        try model.languageModel.installRowExactProducts(.all)
        let requests = (0 ..< 16).map {
            Request(
                prompt: Fixture.tokens(2 + $0, seed: 40 + $0),
                inputs: Fixture.tokens(3, seed: 70 + $0))
        }
        let serial = try Self.serial(model, requests)
        let runner = try NemotronSuperBatchRunner(model: model, maximumRows: 16, kvDType: .float32)
        let lanes = try requests.map { try runner.lane(prompt: $0.prompt, inputs: $0.inputs) }
        for lane in lanes { try runner.prefill(lane) }
        for _ in 0 ..< 3 { try runner.decode(lanes) }
        var trace = Trace(logits: lanes.map(\.logits), snapshots: [])
        trace.snapshots = try lanes.map { try $0.snapshot() }
        Self.exact(trace, serial)
    }

    // MARK: Stock product deviation

    struct Deviation: CustomStringConvertible {
        var maxRelativeLogit: Float = 0
        var maxAbsoluteLogit: Float = 0
        var argmaxDisagreements = 0
        var decodes = 0
        var maxRelativeState: Float = 0
        var bitwiseDecodes = 0

        var description: String {
            "max |dlogit|/max|logit| \(maxRelativeLogit) (abs \(maxAbsoluteLogit)), "
                + "argmax disagreements \(argmaxDisagreements)/\(decodes), "
                + "bitwise decodes \(bitwiseDecodes)/\(decodes), "
                + "max state |d|/max|state| \(maxRelativeState)"
        }
    }

    static func deviation(_ batched: Trace, _ serial: Trace) -> Deviation {
        var result = Deviation()
        for (lhs, rhs) in zip(batched.logits, serial.logits) {
            for (a, b) in zip(lhs, rhs) {
                let x = a.asType(.float32)
                let y = b.asType(.float32)
                let difference = abs(x - y).max().item(Float.self)
                let scale = max(abs(y).max().item(Float.self), .leastNormalMagnitude)
                result.maxAbsoluteLogit = max(result.maxAbsoluteLogit, difference)
                result.maxRelativeLogit = max(result.maxRelativeLogit, difference / scale)
                result.decodes += 1
                if modelOptBitwiseEqual(a, b) { result.bitwiseDecodes += 1 }
                if argMax(x).item(Int32.self) != argMax(y).item(Int32.self) {
                    result.argmaxDisagreements += 1
                }
            }
        }
        for (lhs, rhs) in zip(batched.snapshots, serial.snapshots) {
            for (a, b) in zip(lhs, rhs) {
                let x = a.asType(.float32)
                let y = b.asType(.float32)
                let scale = max(abs(y).max().item(Float.self), .leastNormalMagnitude)
                result.maxRelativeState = max(
                    result.maxRelativeState, abs(x - y).max().item(Float.self) / scale)
            }
        }
        return result
    }

    /// Stock batched products read each weight once for all rows; MLX then
    /// picks wider matrix kernels (`gemv_wide`, `qmv_wide`, GEMM) whose
    /// float32 summation order differs from M = 1. Producer bytes and scales
    /// are unchanged. With exact settings in float32 only the dense FP8
    /// product differs and the logits stay within 1e-5 of the largest logit.
    /// Elsewhere a last-bit difference can flip a downstream NVFP4 activation
    /// code or expert choice, as any two valid summation orders could, so that
    /// deviation is recorded, not bounded.
    @Test(arguments: [
        (DType.float32, Fixture.Products.packedTiled), (.float32, .exactSettings),
        (.bfloat16, .packedTiled), (.bfloat16, .exactSettings),
    ])
    func stockProductDeviationIsMeasured(dtype: DType, products: Fixture.Products) throws {
        let model = try Fixture.model(seed: 31, dtype: dtype, products: products)
        let serial = try Self.serial(model, Self.requests)
        let batched = try Self.batched(model, Self.requests)
        let measured = Self.deviation(batched, serial)
        print("BATCH deviation \(dtype) \(products): \(measured)")
        for logits in batched.logits.joined() {
            #expect(!isNaN(logits).any().item(Bool.self))
        }
        #expect(measured.decodes == 8 + 10 + 6 + 3)
        #expect(measured.maxRelativeState.isFinite && measured.maxRelativeLogit.isFinite)
        if dtype == .float32, products == .exactSettings {
            #expect(measured.maxRelativeLogit <= 1e-5)
        }
    }

    /// Diagnostic: which product slots differ between `[B, 1]` and `[1, 1]`
    /// on this device when they alone keep the batched shape.
    @Test func rowVariantProductSlotsAreRecorded() throws {
        let slots: [(String, NemotronSuperRowProducts)] = [
            ("fp8", .fp8), ("nvfp4Dense", .nvfp4Dense), ("latent", .latent),
            ("router", .router),
        ]
        for (name, slot) in slots {
            let model = try Fixture.model(seed: 31, dtype: .bfloat16, products: .exactSettings)
            try model.languageModel.installRowExactProducts(
                NemotronSuperRowProducts.all.subtracting(slot))
            let measured = Self.deviation(
                try Self.batched(model, Self.requests), try Self.serial(model, Self.requests))
            print("BATCH row-variant slot \(name) alone batched: \(measured)")
            #expect(measured.decodes == 27)
        }
    }

    // MARK: Contract

    @Test func adapterAndModelRefuseInvalidBatchesBeforeStaging() throws {
        let model = try Fixture.model(seed: 3, dtype: .float32, products: .exactSettings)
        #expect(NemotronSuperRecurrentAdapter(model).maximumDecodeRows == 1)
        #expect(throws: NemotronSuperRecurrentAdapter.Failure.batch) {
            try NemotronSuperRecurrentAdapter(model, maximumDecodeRows: 0)
        }
        #expect(throws: NemotronSuperRecurrentAdapter.Failure.batch) {
            try NemotronSuperRecurrentAdapter(
                model, maximumDecodeRows: NemotronSuperRecurrentAdapter.decodeRowLimit + 1)
        }
        let runner = try NemotronSuperBatchRunner(model: model, maximumRows: 2, kvDType: .float32)
        let lanes = try (0 ..< 2).map {
            try runner.lane(prompt: Fixture.tokens(4, seed: $0), inputs: [1])
        }
        for lane in lanes { try runner.prefill(lane) }
        let before = try lanes.map { try $0.snapshot() }
        // The bank rebinds its caches on every call: bind right before use.
        func caches(_ rows: Int) -> [KVCache] {
            runner.bank.layerCaches(rowStates: lanes.prefix(rows).map(\.rows))
                .map { $0 as! KVCache }
        }
        let invalid: [(tokens: [Int32], cacheRows: Int, duplicateOwner: Bool)] = [
            ([1, Int32(Fixture.vocabulary)], 2, false),
            ([1, -1], 2, false),
            ([1, 2, 3], 2, false),
            ([1, 2], 1, false),
            ([1, 2], 2, true),
        ]
        for (tokens, cacheRows, duplicateOwner) in invalid {
            let first = try lanes[0].state.bind()
            let second = duplicateOwner ? first : try lanes[1].state.bind()
            let cache = caches(cacheRows)
            #expect(throws: NemotronSuperModel.Failure.self) {
                try model.decode(tokens: tokens, cache: cache, recurrentState: [first, second])
            }
            #expect(throws: CBv2RecurrentStateError.self) { try first.evaluate() }
            if !duplicateOwner {
                #expect(throws: CBv2RecurrentStateError.self) { try second.evaluate() }
            }
        }
        // Prompt chunks, embeddings and explicit positions never batch.
        let attention = model.cbv2LayerKinds.count
        for (rows, length, positions) in [(2, 2, false), (3, 1, false), (2, 1, true)] {
            #expect(throws: NemotronSuperRecurrentAdapter.Failure.self) {
                try NemotronSuperRecurrentAdapter.validateLayout(
                    batchSize: rows, sequenceLength: length, hasExplicitPositions: positions,
                    recurrentOwnerCount: rows, attentionCacheCount: attention,
                    expectedAttentionCacheCount: attention, maximumDecodeRows: 2)
            }
        }
        try NemotronSuperRecurrentAdapter.validateLayout(
            batchSize: 2, sequenceLength: 1, hasExplicitPositions: false,
            recurrentOwnerCount: 2, attentionCacheCount: attention,
            expectedAttentionCacheCount: attention, maximumDecodeRows: 2)
        let after = try lanes.map { try $0.snapshot() }
        for (lhs, rhs) in zip(before, after) {
            for (a, b) in zip(lhs, rhs) { #expect(modelOptBitwiseEqual(a, b)) }
        }
        let logits = try model.decode(
            tokens: [3, 4], cache: caches(2),
            recurrentState: try lanes.map { try $0.state.bind() })
        #expect(logits.shape == [2, 1, Fixture.vocabulary])
        for lane in lanes { try runner.release(lane) }
    }
}
