// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Mixed batches on the tiny Inkling 2 VLM with the DSpark drafter
/// (`CBv2RecurrentMTPRequestScopedMediaFallback`): text rows keep drafting
/// while an image row decodes beside them, the image row never drafts,
/// verifies, carries or owns assistant state, and every row's greedy tokens
/// equal the drafter-off run. Without the opt-in, a batch holding an image
/// row ran no drafting at all.
@Suite(.serialized, .tags(.integration))
struct Inkling2MediaFallbackEngineTests {
    typealias VLM = Inkling2VLMDSparkEngineTests

    static let imageID = CBv2RequestID(2)

    /// Text carries stored by the driver, each with the image request's
    /// decoded-token count at that moment, plus the image request's driver
    /// state sampled at the same engine-queue points.
    final class Log: @unchecked Sendable {
        private let lock = NSLock()
        private var carries: [(id: CBv2RequestID, imageTokens: Int?)] = []
        private(set) var imageSamples = 0
        private(set) var imageTargetOnly = true
        private(set) var imageAcquiredState = false

        func record(_ id: CBv2RequestID, imageTokens: Int?) {
            lock.withLock { carries.append((id, imageTokens)) }
        }

        func sampleImage(targetOnly: Bool, acquiredState: Bool) {
            lock.withLock {
                imageSamples += 1
                imageTargetOnly = imageTargetOnly && targetOnly
                imageAcquiredState = imageAcquiredState || acquiredState
            }
        }

        var carryIDs: Set<CBv2RequestID> { lock.withLock { Set(carries.map(\.id)) } }

        /// Distinct image decode counts (>= 1) seen while text rows stored
        /// carries: two or more mean text rows drafted across image decode steps.
        var imageDecodeCountsBesideDrafting: Int {
            lock.withLock { Set(carries.compactMap(\.imageTokens).filter { $0 >= 1 }).count }
        }
    }

    static func observe(_ engine: EngineV2, into log: Log) {
        let loop = engine.loopForTesting
        loop.onEngineQueueSync {
            loop.mtp?.carryStoredObserverForTesting = { [unowned loop] carry in
                let image = loop.scheduler.record(for: imageID)
                log.record(carry.id, imageTokens: image?.generatedTokenCount)
                guard let image, let driver = loop.mtp else { return }
                let acquired =
                    driver.assistantStateCountsForTesting(imageID) != nil
                    || driver.roundMark(for: imageID) != nil || driver.isSeedMarked(imageID)
                    || driver.hasValidCarry(for: image)
                log.sampleImage(
                    targetOnly: driver.isTargetOnlyMediaForTesting(imageID),
                    acquiredState: acquired)
            }
        }
    }

    static func run(
        _ engine: EngineV2, _ prompts: [VLM.Prompt], maxTokens: [Int]
    ) async throws -> [[Int]] {
        let streams = try prompts.enumerated().map { index, prompt in
            try engine.submit(
                .init(
                    id: .init(UInt64(index + 1)), promptTokens: prompt.tokens,
                    sampling: .init(temperature: 0), maxTokens: maxTokens[index],
                    prefixCacheEnabled: false,
                    multimodal: prompt.features.map { features in
                        CBv2MultimodalInput(spans: [prompt.span!], attention: .causal) {
                            [features]
                        }
                    }))
        }
        var outputs: [[Int]] = []
        for (index, stream) in streams.enumerated() {
            outputs.append(
                try await cbv2CollectFinished(
                    stream, timeoutSeconds: 300, label: "row \(index)",
                    expectedTokens: maxTokens[index]
                )
                .tokens)
        }
        return outputs
    }

    @Test func textRowsDraftBesideAnImageRowAndEveryRowMatchesTheDrafterOff() async throws {
        let model = try await VLM.model()
        // Request 2 (`imageID`) is the image row, between two text rows.
        let prompts = [
            VLM.textPrompt(41, count: 22), try VLM.imagePrompt(model),
            VLM.textPrompt(42, count: 30),
        ]
        let maxTokens = [40, 24, 40]

        let baseline = try VLM.engine(model, drafter: nil, rows: prompts.count)
        let off = try await Self.run(baseline, prompts, maxTokens: maxTokens)
        await baseline.shutdown()
        #expect(off.map(\.count) == maxTokens)

        let drafter = try Inkling2DSparkTargetEngineTests.drafter(
            target: model.languageModel, dtype: VLM.dtype)
        let speculative = try VLM.engine(model, drafter: drafter, rows: prompts.count)
        let driver = try #require(speculative.loopForTesting.mtp)
        #expect(driver.supportsRequestScopedMediaFallback, "the VLM opts into the media fallback")
        let log = Log()
        Self.observe(speculative, into: log)
        let on = try await Self.run(speculative, prompts, maxTokens: maxTokens)
        let metrics = try #require(speculative.mtpMetricsSnapshot())
        #expect(speculative.loopForTesting.mtp?.requestStateCountForTesting == 0)
        await speculative.shutdown()

        // Every row equals the drafter-off run, image row included.
        #expect(on == off)
        // Text rows draft, and keep drafting while the image row decodes.
        #expect(metrics.rounds > 0)
        #expect(log.carryIDs.isSuperset(of: [CBv2RequestID(1), CBv2RequestID(3)]))
        #expect(log.imageDecodeCountsBesideDrafting >= 2)
        // The image row never drafts, verifies, carries or owns assistant state.
        #expect(!log.carryIDs.contains(Self.imageID))
        #expect(log.imageSamples > 0)
        #expect(log.imageTargetOnly)
        #expect(!log.imageAcquiredState)
        print(
            "inkling2 media fallback rounds=\(metrics.rounds) accepted=\(metrics.acceptedTokens) "
                + "imageDecodeCountsBesideDrafting=\(log.imageDecodeCountsBesideDrafting) "
                + "imageSamples=\(log.imageSamples)")
    }
}
