import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Super attention KV in segmented native paged storage, paired with
/// contiguous storage on the tiny synthetic ModelOpt trunk (the Lightning
/// backend oracle ported to Super). Paged and contiguous attention reduce
/// differently (the paged decode kernel and blocked prefill over gathered
/// pages against SDPA), so across backends logits are held to a stated
/// tolerance with equal greedy choices and equal offsets. Within paged storage
/// everything is exact: repeated runs, batched rows against each request
/// alone, and rollback of a speculative write.
@Suite(.tags(.integration))
struct NemotronSuperPagedParityTests {
    typealias Fixture = NemotronSuperBatchFixture
    typealias Decode = NemotronSuperBatchedDecodeTests

    /// Largest accepted cross-backend logit deviation of one decode, relative
    /// to that decode's largest contiguous logit. The attention outputs
    /// themselves differ only in rounding (`NemotronSuperPagedAttentionTests`),
    /// but a last-bit difference can flip a downstream NVFP4 activation code
    /// or expert choice; on this fixture float32 measured 3.7e-2, bfloat16 0.
    static let logitTolerance: Float = 0.1

    /// Prompts long enough to fill several KV pages and, with 64-token
    /// chunks, to cross prefill chunk boundaries.
    static let longRequests: [Decode.Request] = [
        .init(prompt: Fixture.tokens(200, seed: 11), inputs: Fixture.tokens(12, seed: 12)),
        .init(prompt: Fixture.tokens(129, seed: 13), inputs: Fixture.tokens(9, seed: 14)),
    ]

    static func chunkedSerial(
        _ model: NemotronSuperModel, _ requests: [Decode.Request], paged: Bool, chunk: Int
    ) throws -> Decode.Trace {
        let runner = try NemotronSuperBatchRunner(
            model: model, maximumRows: 1, kvDType: .float32, paged: paged)
        var trace = Decode.Trace(logits: [], snapshots: [])
        for request in requests {
            let lane = try runner.lane(prompt: request.prompt, inputs: request.inputs)
            trace.logits.append([try runner.prefill(lane, chunk: chunk)[0]])
            for _ in request.inputs { try runner.decode([lane]) }
            trace.logits[trace.logits.count - 1] += lane.logits
            trace.snapshots.append(try lane.snapshot())
            try runner.release(lane)
        }
        return trace
    }

    @Test(arguments: [DType.bfloat16, .float32])
    func pagedStorageRepeatsExactly(dtype: DType) throws {
        let model = try Fixture.model(seed: 31, dtype: dtype, products: .rowwise)
        Decode.exact(
            try Self.chunkedSerial(model, Self.longRequests, paged: true, chunk: 64),
            try Self.chunkedSerial(model, Self.longRequests, paged: true, chunk: 64))
    }

    /// The default row-exact batched trunk over paged rows: joins, an early
    /// finish and a cancelled in-flight row leave every request's logits,
    /// recurrent state and paged KV equal to that request decoded alone.
    @Test(arguments: [DType.bfloat16, .float32])
    func pagedBatchedRowsEqualEachRequestAlone(dtype: DType) throws {
        let model = try Fixture.model(seed: 31, dtype: dtype, products: .rowwise)
        let serial = try Decode.serial(model, Decode.requests, paged: true)
        let batched = try Decode.batched(model, Decode.requests, products: .rowExact, paged: true)
        #expect(batched.widestStep == 4)
        Decode.exact(batched, serial)
        let wide = (0 ..< 16).map {
            Decode.Request(
                prompt: Fixture.tokens(2 + 7 * $0, seed: 40 + $0),
                inputs: Fixture.tokens(3, seed: 70 + $0))
        }
        Decode.exact(
            try Decode.batched(model, wide, maximumRows: 16, products: .rowExact, paged: true),
            try Decode.serial(model, wide, paged: true))
    }

    /// Short and long prompts, chunked prefill and decode: equal greedy
    /// choices, KV offsets and committed lengths, logits within tolerance.
    @Test(arguments: [DType.bfloat16, .float32])
    func pagedStaysWithinToleranceOfContiguous(dtype: DType) throws {
        let model = try Fixture.model(seed: 31, dtype: dtype, products: .rowwise)
        for (name, requests, chunk) in [
            (
                "short",
                Decode.requests.map { Decode.Request(prompt: $0.prompt, inputs: $0.inputs) },
                512
            ),
            ("long", Self.longRequests, 64),
        ] {
            let contiguous = try Self.chunkedSerial(model, requests, paged: false, chunk: chunk)
            let paged = try Self.chunkedSerial(model, requests, paged: true, chunk: chunk)
            let measured = Decode.deviation(paged, contiguous)
            print("PAGED deviation \(dtype) \(name): \(measured)")
            #expect(measured.decodes == requests.map { $0.inputs.count + 1 }.reduce(0, +))
            #expect(measured.argmaxDisagreements == 0)
            #expect(measured.maxRelativeLogit <= Self.logitTolerance)
            #expect(measured.maxRelativeState.isFinite)
            for (lhs, rhs) in zip(paged.snapshots, contiguous.snapshots) {
                #expect(lhs.count == rhs.count)
                for (a, b) in zip(lhs, rhs) { #expect(a.shape == b.shape && a.dtype == b.dtype) }
            }
        }
    }

    /// A pending decode written speculatively to paged rows is rolled back
    /// to the exact prior bytes; readmitting the same token reproduces the
    /// committed decode exactly.
    @Test(arguments: [DType.bfloat16, .float32])
    func pagedSpeculativeWriteRollsBackExactly(dtype: DType) throws {
        let model = try Fixture.model(seed: 31, dtype: dtype, products: .rowwise)
        let request = Self.longRequests[0]
        let runner = try NemotronSuperBatchRunner(
            model: model, maximumRows: 1, kvDType: .float32, paged: true)
        let lane = try runner.lane(prompt: request.prompt, inputs: request.inputs)
        let reference = try runner.lane(prompt: request.prompt, inputs: request.inputs)
        for item in [lane, reference] {
            try runner.prefill(item, chunk: 64)
            for _ in 0 ..< 3 { try runner.decode([item]) }
        }
        let before = try lane.snapshot()
        for row in lane.rows.compactMap({ $0 }) {
            #expect(row.supportsSpeculativeWrites)
            row.beginSpeculativeWrite()
        }
        try runner.decode([lane], outcomes: [.rollback])
        for row in lane.rows.compactMap({ $0 }) {
            row.rollback(1)
            row.commitSpeculativeWrite()
        }
        runner.bank.invalidateBoundComposition()
        let restored = try lane.snapshot()
        #expect(restored.count == before.count)
        for (a, b) in zip(restored, before) { #expect(modelOptBitwiseEqual(a, b)) }
        try runner.decode([lane])
        try runner.decode([reference])
        #expect(modelOptBitwiseEqual(lane.logits.last!, reference.logits.last!))
        let after = try lane.snapshot()
        for (a, b) in zip(after, try reference.snapshot()) { #expect(modelOptBitwiseEqual(a, b)) }
        try runner.release(lane)
        try runner.release(reference)
        #expect(runner.backend.bytesReserved == 0)
    }
}
