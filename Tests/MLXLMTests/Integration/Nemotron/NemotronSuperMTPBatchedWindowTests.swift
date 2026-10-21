import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Several requests in one captured `[B, W]` verification forward, and in one
/// hidden-returning `[B, 1]` decode, against each request's own one-request
/// forwards: every position's logits and hidden, and each request's recurrent
/// and attention state after committing its own accepted prefix, must be
/// bit-identical. Requests carry different histories and windows.
@Suite(.tags(.integration))
struct NemotronSuperMTPBatchedWindowTests {
    typealias Capture = NemotronSuperMTPCaptureTests

    enum Fixture: Sendable, CustomTestStringConvertible {
        /// The MTP fixture in FP32 and in the BF16 ModelOpt precision mix.
        case mtp(quantized: Bool)
        /// The latent-MoE trunk on the graph path, rowwise or (the default)
        /// tiled accumulation.
        case graph(NemotronSuperBatchFixture.Products)

        var testDescription: String {
            switch self {
            case .mtp(let quantized): "mtp \(quantized ? "bf16-modelopt" : "fp32")"
            case .graph(let products): "graph bf16-modelopt \(products)"
            }
        }

        func model() throws -> NemotronSuperModel {
            switch self {
            case .mtp(let quantized): try NemotronSuperMTPFixture.target(quantized: quantized)
            case .graph(let products):
                try NemotronSuperBatchFixture.model(seed: 23, dtype: .bfloat16, products: products)
            }
        }

        var kvDType: DType {
            if case .mtp(false) = self { return .float32 }
            return .bfloat16
        }

        /// Token ids below both fixtures' vocabularies that avoid the image id.
        func tokens(_ count: Int, seed: Int) -> [Int] {
            (0 ..< count).map { (index: Int) -> Int in
                let value = (index * 37 + seed * 11 + 5) % NemotronSuperMTPFixture.vocabulary
                return value == 18 ? 19 : value
            }
        }
    }

    final class Harness {
        let model: NemotronSuperModel
        let target: NemotronSuperMTPCaptureTarget
        let backend: CBv2ContiguousKVBackend
        let bank: CBv2LayerCacheBank
        var requests: [Capture.Request] = []

        init(_ fixture: Fixture, maximumRequests: Int) throws {
            model = try fixture.model()
            target = try NemotronSuperMTPCaptureTarget(
                capturing: model, maximumRequests: maximumRequests)
            backend = CBv2ContiguousKVBackend(
                config: .init(bytesCapacity: 32 << 20, kvDType: fixture.kvDType))
            bank = CBv2LayerCacheBank(
                caches: target.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
        }

        deinit {
            bank.releaseBoundRows()
            for request in requests { backend.release(request.rows) }
        }

        func prefilled(_ prompt: [Int]) throws -> Capture.Request {
            let request = try Capture.Request(model, backend: backend)
            requests.append(request)
            let prefill = try Capture.forward(
                target, bank: bank, request: request, tokens: prompt, captured: false)
            try prefill.transaction.commit()
            return request
        }

        /// One forward of every request's `tokens` row in one `[B, W]` batch,
        /// evaluated but not finalized.
        func batched(_ requests: [Capture.Request], tokens: [[Int]], captured: Bool) throws
            -> (logits: MLXArray, hidden: MLXArray, transactions: [CBv2RecurrentStateEvaluation])
        {
            let attending = bank.layerCaches(rowStates: requests.map(\.rows))
            let caches = try attending.map { try #require($0 as? KVCache) }
            let serializing = attending.compactMap { $0 as? any CBv2MTPRectangularSerializing }
            if captured {
                for cache in serializing { cache.mtpSerializesRectangularAttention = true }
            }
            defer { for cache in serializing { cache.mtpSerializesRectangularAttention = false } }
            let transactions = try requests.map { try $0.state.bind() }
            let input = MLXArray(tokens.flatMap { $0.map(Int32.init) })
                .reshaped(tokens.count, tokens[0].count)
            let output =
                captured
                ? target.cbv2ForwardWithHiddenCaptured(
                    input, caches: caches, recurrentState: transactions, positionIds: nil)
                : target.cbv2ForwardWithHidden(
                    input, caches: caches, recurrentState: transactions, positionIds: nil)
            var roots = [output.logits, output.lastHidden] + caches.flatMap { $0.innerState() }
            for transaction in transactions { roots += try transaction.evaluate() }
            eval(roots)
            return (output.logits, output.lastHidden, transactions)
        }
    }

    static let prompts = [11, 5, 8, 3]

    /// Each request decodes its window one token at a time; the state after
    /// every accepted prefix and each position's outputs are the oracle.
    static func oracle(_ harness: Harness, prompt: [Int], window: [Int]) throws
        -> (states: [Capture.Snapshot], logits: [MLXArray], hidden: [MLXArray])
    {
        let request = try harness.prefilled(prompt)
        var states = [try Capture.snapshot(request)]
        var logits: [MLXArray] = []
        var hidden: [MLXArray] = []
        for token in window {
            let step = try Capture.forward(
                harness.target, bank: harness.bank, request: request, tokens: [token],
                captured: false)
            try step.transaction.commit()
            logits.append(step.logits)
            hidden.append(step.hidden)
            states.append(try Capture.snapshot(request))
        }
        try request.state.release()
        return (states, logits, hidden)
    }

    @Test(
        .timeLimit(.minutes(5)),
        arguments: [
            (Fixture.mtp(quantized: false), 3, 3), (.mtp(quantized: true), 4, 4),
            (.mtp(quantized: true), 2, 2), (.graph(.rowwise), 3, 4), (.graph(.tiled), 4, 3),
            (.graph(.tiled), 4, 4),
        ])
    func batchedWindowMatchesEachRequestAndCommitsItsOwnPrefix(
        fixture: Fixture, requests count: Int, width: Int
    ) throws {
        let harness = try Harness(fixture, maximumRequests: 4)
        let prompts = (0 ..< count).map { fixture.tokens(Self.prompts[$0], seed: $0 + 1) }
        let windows = (0 ..< count).map { fixture.tokens(width, seed: $0 + 7) }
        let oracles = try (0 ..< count).map {
            try Self.oracle(harness, prompt: prompts[$0], window: windows[$0])
        }
        // Rotate the accepted lengths so every request commits a different
        // prefix, including a full rollback and the whole window.
        for rotation in 0 ... width {
            let requests = try prompts.map { try harness.prefilled($0) }
            for request in requests {
                for row in request.rows.compactMap({ $0 }) { row.beginSpeculativeWrite() }
            }
            let verify = try harness.batched(requests, tokens: windows, captured: true)
            #expect(verify.logits.shape.prefix(2) == [count, width])
            for (index, oracle) in oracles.enumerated() {
                #expect(verify.transactions[index].isCaptured)
                for position in 0 ..< width {
                    let context =
                        "\(fixture) B=\(count) W=\(width) request \(index) pos \(position)"
                    #expect(
                        arrayEqual(
                            verify.logits[index ..< (index + 1), position ..< (position + 1)],
                            oracle.logits[position]
                        ).item(Bool.self), "logits \(context)")
                    #expect(
                        arrayEqual(
                            verify.hidden[index ..< (index + 1), position ..< (position + 1)],
                            oracle.hidden[position]
                        ).item(Bool.self), "hidden \(context)")
                }
            }
            for (index, request) in requests.enumerated() {
                let keep = (rotation + index) % (width + 1)
                if keep == 0 {
                    try verify.transactions[index].rollback()
                } else {
                    try verify.transactions[index].commit(keepPositions: keep)
                }
                for row in request.rows.compactMap({ $0 }) {
                    if width > keep { row.rollback(width - keep) }
                    row.commitSpeculativeWrite()
                }
            }
            harness.bank.invalidateBoundComposition()
            for (index, request) in requests.enumerated() {
                let keep = (rotation + index) % (width + 1)
                Capture.exact(
                    try Capture.snapshot(request), oracles[index].states[keep],
                    "\(fixture) B=\(count) W=\(width) request \(index) keep \(keep)")
                try request.state.release()
            }
        }
    }

    @Test(
        .timeLimit(.minutes(3)),
        arguments: [
            Fixture.mtp(quantized: false), .mtp(quantized: true), .graph(.rowwise), .graph(.tiled),
        ])
    func batchedHiddenDecodeMatchesEachRequestsDecode(fixture: Fixture) throws {
        let harness = try Harness(fixture, maximumRequests: 4)
        let prompts = (0 ..< 4).map { fixture.tokens(Self.prompts[$0], seed: $0 + 1) }
        let inputs = (0 ..< 4).map { fixture.tokens(3, seed: $0 + 13) }
        let oracles = try (0 ..< 4).map {
            try Self.oracle(harness, prompt: prompts[$0], window: inputs[$0])
        }
        let requests = try prompts.map { try harness.prefilled($0) }
        for step in 0 ..< 3 {
            let decoded = try harness.batched(
                requests, tokens: inputs.map { [$0[step]] }, captured: false)
            for (index, transaction) in decoded.transactions.enumerated() {
                #expect(!transaction.isCaptured)
                let context = "\(fixture) request \(index) step \(step)"
                #expect(
                    arrayEqual(decoded.logits[index ..< (index + 1)], oracles[index].logits[step])
                        .item(Bool.self), "logits \(context)")
                #expect(
                    arrayEqual(decoded.hidden[index ..< (index + 1)], oracles[index].hidden[step])
                        .item(Bool.self), "hidden \(context)")
                try transaction.commit()
            }
            harness.bank.invalidateBoundComposition()
            for (index, request) in requests.enumerated() {
                Capture.exact(
                    try Capture.snapshot(request), oracles[index].states[step + 1],
                    "\(fixture) request \(index) after step \(step)")
            }
        }
        for request in requests { try request.state.release() }
    }

    @Test func targetAndAssistantRefuseUnservedRows() throws {
        let stack = try NemotronSuperMTPFixture.stack(
            kind: .predictive, quantized: true, verification: .rectangularExact)
        #expect(throws: NemotronHModel.SuperMTPCaptureError.self) {
            try NemotronSuperMTPCaptureTarget(
                capturing: stack.model,
                maximumRequests: NemotronHModel.superMTPMaximumWindowRequests + 1)
        }
        let target = try NemotronSuperMTPCaptureTarget(capturing: stack.model, maximumRequests: 2)
        var settings = NemotronSuperMTPFixture.settings(depth: 2, verification: .rectangularExact)
        settings.maximumSpeculativeRows = 3
        #expect(throws: NemotronSuperMTPAssistant.Failure.rows) {
            try target.makeAssistant(head: stack.head, settings: settings)
        }
        settings.maximumSpeculativeRows = 2
        let assistant = try target.makeAssistant(head: stack.head, settings: settings)
        #expect(assistant.maximumSpeculativeBatch == 2)
        #expect(assistant.commitsAcceptedPrefixPerRow)
        settings.verification = .serial
        #expect(throws: NemotronSuperMTPAssistant.Failure.rows) {
            try NemotronSuperMTPTextTarget(stack.model, maximumRequests: 2)
                .makeAssistant(head: stack.head, settings: settings)
        }
    }
}
