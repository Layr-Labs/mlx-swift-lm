import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// One captured `[1, W]` verification forward against W one-token decode
/// steps of the same target: every position's logits and hidden, and the
/// recurrent plus attention state after committing any accepted prefix, must be
/// bit-identical. Covers the FP32 fixture and the BF16 ModelOpt precision mix.
@Suite(.tags(.integration))
struct NemotronSuperMTPCaptureTests {
    final class Request {
        let rows: [CBv2SequenceKV?]
        let state: CBv2RecurrentRequestState
        init(_ model: NemotronSuperModel, backend: any CBv2KVBackend) throws {
            rows = try backend.makeSequenceState(
                layerKinds: model.cbv2LayerKinds, promptLength: 16, maxLength: 64)
            state = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
        }
    }

    struct Snapshot {
        let offsets: [Int]
        let tensors: [MLXArray]
    }

    static func snapshot(_ request: Request) throws -> Snapshot {
        let confirmed = try #require(request.state.confirmedStateSnapshot())
        var tensors: [MLXArray] = []
        for spec in request.state.spec.layers {
            let layer = try #require(confirmed[spec.modelLayerIndex])
            tensors += [try #require(layer.conv) + 0, try #require(layer.ssm) + 0]
        }
        var offsets: [Int] = []
        for row in request.rows.compactMap({ $0 }) {
            let kv = row.snapshot()
            offsets.append(kv.offset)
            tensors += [kv.keys + 0, kv.values + 0]
        }
        eval(tensors)
        return Snapshot(offsets: offsets, tensors: tensors)
    }

    static func exact(_ lhs: Snapshot, _ rhs: Snapshot, _ context: String) {
        #expect(lhs.offsets == rhs.offsets, "\(context)")
        #expect(lhs.tensors.count == rhs.tensors.count)
        for (a, b) in zip(lhs.tensors, rhs.tensors) {
            #expect(a.shape == b.shape && arrayEqual(a, b).item(Bool.self), "\(context)")
        }
    }

    /// One serial (`captured == false`) or captured forward in its own
    /// recurrent transaction, evaluated but not finalized.
    static func forward(
        _ target: NemotronSuperMTPCaptureTarget, bank: CBv2LayerCacheBank, request: Request,
        tokens: [Int], captured: Bool
    ) throws -> (logits: MLXArray, hidden: MLXArray, transaction: CBv2RecurrentStateEvaluation) {
        let attending = try bank.layerCaches(rowStates: [request.rows])
        let caches = try attending.map { try #require($0 as? KVCache) }
        let serializing = attending.compactMap { $0 as? any CBv2MTPRectangularSerializing }
        if captured { for cache in serializing { cache.mtpSerializesRectangularAttention = true } }
        defer { for cache in serializing { cache.mtpSerializesRectangularAttention = false } }
        let transaction = try request.state.bind()
        let input = MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count)
        let output =
            captured
            ? target.cbv2ForwardWithHiddenCaptured(
                input, caches: caches, recurrentState: [transaction], positionIds: nil)
            : target.cbv2ForwardWithHidden(
                input, caches: caches, recurrentState: [transaction], positionIds: nil)
        eval(
            [output.logits, output.lastHidden] + (try transaction.evaluate())
                + caches.flatMap { $0.innerState() })
        return (output.logits, output.lastHidden, transaction)
    }

    @Test(
        .timeLimit(.minutes(3)),
        arguments: [(false, 2), (false, 4), (true, 2), (true, 3), (true, 4)])
    func capturedWindowMatchesDecodeAndCommitsEveryPrefix(quantized: Bool, width: Int) throws {
        let model = try NemotronSuperMTPFixture.target(quantized: quantized)
        let target = try NemotronSuperMTPCaptureTarget(capturing: model)
        let backend = CBv2ContiguousKVBackend(
            config: .init(
                bytesCapacity: 16 << 20, kvDType: quantized ? .bfloat16 : .float32))
        let bank = CBv2LayerCacheBank(
            caches: target.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
        let prompt = [5, 9, 1, 4, 22, 7, 3, 8, 30, 12, 2]
        let window = [17, 44, 3, 61].prefix(width).map { $0 }
        var requests: [Request] = []
        defer {
            bank.releaseBoundRows()
            for request in requests { backend.release(request.rows) }
        }
        func prefilled() throws -> Request {
            let request = try Request(model, backend: backend)
            requests.append(request)
            let prefill = try Self.forward(
                target, bank: bank, request: request, tokens: prompt, captured: false)
            try prefill.transaction.commit()
            return request
        }
        let oracle = try prefilled()
        var states = [try Self.snapshot(oracle)]
        var logits: [MLXArray] = []
        var hidden: [MLXArray] = []
        for token in window {
            let step = try Self.forward(
                target, bank: bank, request: oracle, tokens: [token], captured: false)
            try step.transaction.commit()
            logits.append(step.logits)
            hidden.append(step.hidden)
            states.append(try Self.snapshot(oracle))
        }
        for keep in 0 ... width {
            let request = try prefilled()
            for row in request.rows.compactMap({ $0 }) { row.beginSpeculativeWrite() }
            let verify = try Self.forward(
                target, bank: bank, request: request, tokens: window, captured: true)
            #expect(verify.transaction.isCaptured)
            for position in 0 ..< width {
                let context = "quantized \(quantized) W=\(width) position \(position)"
                #expect(
                    arrayEqual(verify.logits[0..., position ..< (position + 1)], logits[position])
                        .item(Bool.self), "logits \(context)")
                #expect(
                    arrayEqual(verify.hidden[0..., position ..< (position + 1)], hidden[position])
                        .item(Bool.self), "hidden \(context)")
            }
            if keep == 0 {
                try verify.transaction.rollback()
            } else {
                try verify.transaction.commit(keepPositions: keep)
            }
            for row in request.rows.compactMap({ $0 }) {
                if width > keep { row.rollback(width - keep) }
                row.commitSpeculativeWrite()
            }
            bank.invalidateBoundComposition()
            Self.exact(
                try Self.snapshot(request), states[keep],
                "quantized \(quantized) W=\(width) keep \(keep)")
            try request.state.release()
        }
        try oracle.state.release()
    }
}
