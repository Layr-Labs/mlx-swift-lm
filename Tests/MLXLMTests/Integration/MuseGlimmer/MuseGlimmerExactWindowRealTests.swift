// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Exact verify windows on the pinned NVIDIA checkpoint with the pinned
/// quantized drafter. Needs `MUSE_GLIMMER_NVFP4_DIR`, `MUSE_GLIMMER_ORACLE_DIR`
/// and (for the drafter) `MUSE_GLIMMER_DFLASH_QUANTIZED_DIR`; hold the GPU lane.
/// The tower is staged here, apart from the other real suites, because the
/// switch is per model.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerExactWindowRealTests {
    static let environment = ProcessInfo.processInfo.environment
    static let drafterDirectory = environment["MUSE_GLIMMER_DFLASH_QUANTIZED_DIR"]
    static var hasDrafter: Bool { MuseGlimmerRealArtifact.hasOracle && drafterDirectory != nil }
    static let newTokens = 64

    nonisolated(unsafe) static let staged: Result<MuseGlimmerTextModel, Error> = Result {
        guard let directory = MuseGlimmerRealArtifact.directory else {
            throw MuseGlimmerNPZ.Failure.malformed("no directory")
        }
        let model = try MuseGlimmerInstallation.stage(directory: directory).model
        try model.setRowExactWindows(true)
        return model
    }

    nonisolated(unsafe) static let drafterModel: Result<MuseGlimmerDFlashDraftModel, Error> =
        Result {
            guard let drafterDirectory else { throw MuseGlimmerDFlashError.artifact("environment") }
            return try MuseGlimmerDFlashDraftModel.load(
                directory: URL(fileURLWithPath: drafterDirectory),
                pin: .museGlimmer30BQuantized, maximumBytes: 2 << 30)
        }

    static func oracle(_ name: String) throws -> (prompt: [Int], tokens: [Int], margins: [Float]) {
        let file = try MuseGlimmerNPZ(
            url: MuseGlimmerRealArtifact.oracleDirectory!.appendingPathComponent(
                "\(name)_greedy.npz"))
        return (
            try file.ints("input_ids").map(Int.init), try file.ints("new_ids").map(Int.init),
            try file.floats("top1_margin")
        )
    }

    /// An 8-token window (the oracle's own continuation) after each oracle
    /// prompt, prefilled as the engine prefills, equals eight one-row steps in
    /// logits and K/V, byte for byte, with attention as decode rows (one
    /// dispatch per layer) and as the column loop.
    @Test(
        .enabled(if: MuseGlimmerRealArtifact.hasOracle), .timeLimit(.minutes(20)),
        arguments: ["P1", "P2", "P3"])
    func windowsEqualStepsOnTheOraclePrompts(name: String) throws {
        let model = try Self.staged.get()
        #expect(model.verifiesWindowsExactly)
        let (prompt, continuation, _) = try Self.oracle(name)
        for (width, columnsAsRows) in [(8, true), (3, true), (8, false)] {
            let window = continuation.prefix(width).map(Int32.init)
            func paged() throws -> MuseGlimmerExactWindowTests.Paged {
                let paged = try MuseGlimmerExactWindowTests.Paged(
                    model, capacityBytes: 1 << 30, chunk: 1024, segmentBytes: 8 << 20,
                    maximumLength: 8192)
                _ = try paged.addPrompt(prompt.map(Int32.init), extra: width)
                return paged
            }
            let windowed = try paged()
            let scored = windowed.forward(
                [window], rectangular: true, columnsAsRows: columnsAsRows)
            let calls = windowed.columnsAsRowsCalls
            let stepped = try paged()
            var rows = 0
            for (column, token) in window.enumerated() {
                let step = stepped.forward([[token]], rectangular: false)
                if MuseGlimmerFixture.sameBits(scored[0..., column ..< column + 1], step) {
                    rows += 1
                }
            }
            let sameKV = windowed.kv(0) == stepped.kv(0)
            MuseGlimmerRealArtifact.record(
                "\(name) exact_window width=\(width) prompt=\(prompt.count) "
                    + "attention=\(columnsAsRows ? "decode_rows" : "column_loop") "
                    + "decode_row_layers=\(calls) "
                    + "equal_logit_rows=\(rows)/\(width) equal_kv=\(sameKV)")
            #expect(rows == width && sameKV, "\(name) width \(width) \(columnsAsRows)")
            #expect(calls == (columnsAsRows ? windowed.caches.count : 0))
        }
    }

    private func engine(
        _ model: MuseGlimmerTextModel, drafter: (any CBv2MTPDrafter)?
    ) throws -> (EngineV2, [PagedLayerCache]) {
        let backend = try PagedKVBackend(
            layerKinds: model.cbv2LayerKinds,
            config: .init(
                capacityBytes: 1 << 30, dtype: .bfloat16, maxPrefillChunk: 1024,
                nominalMaxSequenceLength: 8192, segmentSizeBytes: 8 << 20,
                layerDTypes: model.kvStorageDTypes))
        let caches = backend.makeLayerCaches()
        let engine = EngineV2(
            model: MuseGlimmerCBv2Adapter(model), layerKinds: model.cbv2LayerKinds,
            backend: backend, cacheProvider: CBv2LayerCacheBank(caches: caches),
            schedulerConfig: .init(
                maxConcurrentRequests: 1, maxBatchedTokensPerStep: 1024, prefillChunkSize: 1024,
                soloPrefillStripeTokens: 1024, maxWaiting: 2, enablePrefixCache: false),
            admissionConfig: .init(watermarkFraction: 0), mtpDrafter: drafter,
            mtpConfig: CBv2MTPConfig(
                enabled: drafter != nil, maxDraftTokens: 7, maxSpeculativeBatch: 1,
                fixedDraftTokens: 7, verificationMode: .rectangularExact,
                maxAutomaticRectangularTokens: 16))
        return (engine, caches)
    }

    /// Greedy and sampled (temperature 1, top-p 0.95, top-k 64) output with
    /// the drafter equals the output without it; the greedy tokens track the
    /// oracle up to its near-ties (margin under 1.5, as in the plain test).
    @Test(
        .enabled(if: MuseGlimmerExactWindowRealTests.hasDrafter), .timeLimit(.minutes(30)),
        arguments: ["P1", "P2", "P3"])
    func drafterOnEqualsDrafterOffAndGreedyTracksTheOracle(name: String) async throws {
        let model = try Self.staged.get()
        let (prompt, oracle, margins) = try Self.oracle(name)
        let samplings: [(String, CBv2SamplingParams)] = [
            ("greedy", .init(temperature: 0, seed: 0)),
            ("sampled", .init(temperature: 1, topP: 0.95, topK: 64, seed: 99)),
        ]
        for (label, sampling) in samplings {
            let request = CBv2Request(
                id: .init(1), promptTokens: prompt, sampling: sampling, maxTokens: Self.newTokens)
            let (plain, _) = try engine(model, drafter: nil)
            let off = await cbv2SchedCollect(try plain.submit(request), timeoutSeconds: 1200)
            await plain.shutdown()
            let drafter = try MuseGlimmerDFlashDrafter(
                target: model, model: try Self.drafterModel.get(), settings: .verifying(model))
            #expect(drafter.batchesVerifyColumnsAsDecodeRows)
            let (speculative, caches) = try engine(model, drafter: drafter)
            #expect(speculative.mtpInactiveReason == nil)
            let start = Date()
            let on = await cbv2SchedCollect(try speculative.submit(request), timeoutSeconds: 1200)
            let seconds = Date().timeIntervalSince(start)
            let metrics = try #require(speculative.mtpMetricsSnapshot())
            await speculative.shutdown()
            let calls = caches.map(\.mtpColumnsAsRowsCalls).reduce(0, +)
            let firstMismatch = zip(off.tokens, oracle).enumerated().first {
                $0.element.0 != $0.element.1
            }?.offset
            MuseGlimmerRealArtifact.record(
                "\(name) exact_drafter \(label) prompt=\(prompt.count) off=\(off.tokens.count) "
                    + "on=\(on.tokens.count) equal=\(on.tokens == off.tokens) "
                    + "rounds=\(metrics.rounds) rectangular=\(metrics.rectangularVerificationRounds) "
                    + "serial=\(metrics.serialVerificationRounds) drafted=\(metrics.draftedTokens) "
                    + "accepted=\(metrics.acceptedTokens) emitted=\(metrics.emittedTokens) "
                    + "decode_row_layer_calls=\(calls) "
                    + "seconds_on=\(String(format: "%.1f", seconds)) "
                    + "oracle_first_mismatch=\(firstMismatch ?? -1) tokens=\(off.tokens.prefix(24))"
            )
            #expect(off.finishReason != nil && on.finishReason == off.finishReason)
            #expect(on.tokens == off.tokens, "\(name) \(label)")
            #expect(metrics.rounds > 0 && metrics.serialVerificationRounds == 0)
            #expect(calls > 0)
            if label == "greedy", let firstMismatch, firstMismatch < oracle.count {
                #expect(margins[firstMismatch] < 1.5, "\(name) parts from the oracle")
            }
        }
    }
}
