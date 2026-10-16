import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

private let nemotronSuperServingBenchmarkMode =
    ProcessInfo.processInfo.environment["NEMOTRON_SUPER_SERVING_BENCH"]

/// Opt-in serving timings at real Super dimensions with synthetic weights
/// (the batched decode benchmark's stacks: FP8 Mamba/attention projections and
/// head, 512 NVFP4 experts, graph path with its tiled default). Two stacks per
/// measurement separate the marginal cost of Mamba+MoE pairs from the fixed
/// cost, projected to the 40 Mamba, 40 MoE and 8 attention layers of the
/// checkpoint. Run with NEMOTRON_SUPER_SERVING_BENCH=1 through the GPU lane,
/// in release. Reports only; never the checkpoint.
@Suite(.serialized, .enabled(if: nemotronSuperServingBenchmarkMode == "1"))
struct NemotronSuperMTPServingBenchmark {
    typealias Bench = NemotronSuperBatchedDecodeBenchmark
    typealias Capture = NemotronSuperMTPCaptureTests

    /// Base stack, two more Mamba+MoE pairs, one more attention layer.
    static let patterns = ["ME*", "MEMEME*", "ME**"]
    static let repetitions = 9

    static func median(_ body: () throws -> Void) rethrows -> Double {
        try body()
        var samples: [Double] = []
        for _ in 0 ..< repetitions {
            let started = DispatchTime.now().uptimeNanoseconds
            try body()
            samples.append(Double(DispatchTime.now().uptimeNanoseconds - started) / 1e6)
        }
        return samples.sorted()[repetitions / 2]
    }

    /// 88-layer step: the base stack (one pair, one attention layer, embedding
    /// and head) plus 39 more pairs and 7 more attention layers, each priced
    /// by the marginal stacks.
    static func project(_ values: [Double]) -> Double {
        let pair = (values[1] - values[0]) / 2
        let attention = values[2] - values[0]
        return values[0] + 39 * pair + 7 * attention
    }

    final class Requests {
        let model: NemotronSuperModel
        let target: NemotronSuperMTPCaptureTarget
        let backend: CBv2ContiguousKVBackend
        let bank: CBv2LayerCacheBank
        var requests: [Capture.Request] = []

        init(_ model: NemotronSuperModel, count: Int) throws {
            self.model = model
            target = try NemotronSuperMTPCaptureTarget(capturing: model, maximumRequests: 8)
            backend = CBv2ContiguousKVBackend(
                config: .init(bytesCapacity: 1 << 30, kvDType: .bfloat16))
            bank = CBv2LayerCacheBank(
                caches: target.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
            for index in 0 ..< count {
                let request = try Capture.Request(model, backend: backend)
                requests.append(request)
                let prompt = Bench.stream(16, row: index, salt: 11).map(Int.init)
                let prefill = try Capture.forward(
                    target, bank: bank, request: request, tokens: prompt, captured: false)
                try prefill.transaction.commit()
            }
        }

        deinit {
            bank.releaseBoundRows()
            for request in requests { backend.release(request.rows) }
        }

        /// One `[B, W]` forward of the first `rows` requests, evaluated and
        /// rolled back so it can be repeated on the same state.
        func step(rows: Int, width: Int, captured: Bool) throws {
            let lanes = Array(requests.prefix(rows))
            let attending = bank.layerCaches(rowStates: lanes.map(\.rows))
            let caches = attending.map { $0 as! KVCache }
            let serializing = attending.compactMap { $0 as? any CBv2MTPRectangularSerializing }
            for cache in serializing where captured {
                cache.mtpSerializesRectangularAttention = true
            }
            defer { for cache in serializing { cache.mtpSerializesRectangularAttention = false } }
            for lane in lanes {
                for row in lane.rows.compactMap({ $0 }) { row.beginSpeculativeWrite() }
            }
            let transactions = try lanes.map { try $0.state.bind() }
            let tokens = MLXArray(
                (0 ..< rows).flatMap { Bench.stream(width, row: $0, salt: 997) }
            ).reshaped(rows, width)
            let output =
                captured
                ? target.cbv2ForwardWithHiddenCaptured(
                    tokens, caches: caches, recurrentState: transactions, positionIds: nil)
                : target.cbv2ForwardWithHidden(
                    tokens, caches: caches, recurrentState: transactions, positionIds: nil)
            var roots = [output.logits, output.lastHidden] + caches.flatMap { $0.innerState() }
            for transaction in transactions { roots += try transaction.evaluate() }
            eval(roots)
            for transaction in transactions { try transaction.rollback() }
            for lane in lanes {
                for row in lane.rows.compactMap({ $0 }) {
                    row.rollback(width)
                    row.commitSpeculativeWrite()
                }
            }
            bank.invalidateBoundComposition()
        }
    }

    /// One captured `[B, 1 + k]` window against B one-request windows and
    /// against the `[B, 1]` decode step of the same rows.
    @Test func batchedVerifyWindowsAtRealDimensions() throws {
        var lines: [String] = []
        var steps: [String: [Double]] = [:]
        for pattern in Self.patterns {
            let harness = try Requests(try Bench.model(pattern, seed: 41), count: 4)
            for rows in [1, 2, 4] {
                steps["decode B=\(rows)", default: []].append(
                    try Self.median { try harness.step(rows: rows, width: 1, captured: false) })
                for width in [2, 4] {
                    steps["window B=\(rows) W=\(width)", default: []].append(
                        try Self.median {
                            try harness.step(rows: rows, width: width, captured: true)
                        })
                }
            }
            Memory.clearCache()
        }
        for key in steps.keys.sorted() {
            let values = steps[key]!
            lines.append(
                "\(key): stacks \(values.map { String(format: "%.2f", $0) }) ms, "
                    + "88-layer projection \(String(format: "%.1f", Self.project(values))) ms")
        }
        print("SYNTHETIC super serving windows\n" + lines.joined(separator: "\n"))
    }

    /// Plain single-request decode: a bare unchained loop, a bare loop that
    /// builds step N + 1 while N runs, and EngineV2 with the production
    /// sampler. The engine's extra time per token is its serialized overhead.
    @Test func engineDecodeAgainstBareLoopsAtRealDimensions() async throws {
        var lines: [String] = []
        var projected: [String: [Double]] = [:]
        for pattern in Self.patterns {
            let model = try Bench.model(pattern, seed: 43)
            let steps = 48
            let runner = try NemotronSuperBatchRunner(
                model: model, maximumRows: 1, kvDType: .bfloat16)
            func bare(chained: Bool) throws -> Double {
                let lane = try runner.lane(
                    prompt: Bench.stream(16, row: 0, salt: 3),
                    inputs: Bench.stream(steps, row: 0, salt: 5))
                try runner.prefill(lane)
                var token = MLXArray([Int32(7)]).reshaped(1, 1)
                var pending: (MLXArray, CBv2RecurrentStateEvaluation)?
                var times: [UInt64] = []
                for _ in 0 ..< steps {
                    let cache = runner.caches([lane])
                    let binding = try lane.state.bind()
                    let logits = runner.adapter.cbv2Forward(
                        token, caches: cache, recurrentState: [binding])
                    let next = argMax(logits[0..., -1, 0...], axis: -1).asType(.int32).reshaped(
                        1, 1)
                    let roots =
                        [next] + cache.flatMap { $0.innerState() } + (try binding.evaluate())
                    if chained {
                        asyncEval(roots)
                        if let (previous, previousBinding) = pending {
                            eval(previous)
                            try previousBinding.commit()
                        }
                        pending = (next, binding)
                    } else {
                        eval(roots)
                        try binding.commit()
                    }
                    token = next
                    times.append(DispatchTime.now().uptimeNanoseconds)
                }
                if let (previous, previousBinding) = pending {
                    eval(previous)
                    try previousBinding.commit()
                }
                try runner.release(lane)
                return Double(times.last! - times[8]) / 1e6 / Double(steps - 9)
            }
            let unchained = try bare(chained: false)
            let chained = try bare(chained: true)
            let engine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(runner.adapter),
                layerKinds: runner.adapter.cbv2LayerKinds,
                backend: CBv2ContiguousKVBackend(
                    config: .init(bytesCapacity: 1 << 30, kvDType: .bfloat16)),
                cacheProvider: CBv2LayerCacheBank(
                    caches: runner.adapter.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) }),
                sampler: CBv2DefaultSampler(),
                schedulerConfig: .init(
                    maxConcurrentRequests: 1, maxBatchedTokensPerStep: 2048, prefillChunkSize: 512),
                admissionConfig: .init(watermarkFraction: 0))
            var stamps: [UInt64] = []
            for await event in try engine.submit(
                CBv2Request(
                    id: CBv2RequestID(1),
                    promptTokens: Bench.stream(16, row: 0, salt: 3).map(Int.init),
                    sampling: .init(temperature: 0), maxTokens: steps))
            {
                if case .delta(_, let tokens, _) = event {
                    for _ in tokens { stamps.append(DispatchTime.now().uptimeNanoseconds) }
                }
            }
            await engine.shutdown()
            let engineStep = Double(stamps.last! - stamps[8]) / 1e6 / Double(stamps.count - 9)
            lines.append(
                "\(pattern): bare unchained \(String(format: "%.2f", unchained)) ms/token, bare chained "
                    + "\(String(format: "%.2f", chained)), engine \(String(format: "%.2f", engineStep))"
            )
            for (name, value) in [
                ("unchained", unchained), ("chained", chained), ("engine", engineStep),
            ] {
                projected[name, default: []].append(value)
            }
            Memory.clearCache()
        }
        for name in ["unchained", "chained", "engine"] {
            lines.append(
                "88-layer projection \(name): "
                    + String(format: "%.1f ms/token", Self.project(projected[name]!)))
        }
        print("SYNTHETIC super serving decode\n" + lines.joined(separator: "\n"))
    }
}
