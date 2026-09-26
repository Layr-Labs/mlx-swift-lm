import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

@testable import MLXVLM

@Suite("Native diffusion resident prefix lifecycle", .serialized)
struct DiffusionGemmaResidentPrefixTests {
    private func config() throws -> DiffusionGemmaResidentPrefixConfiguration {
        try .init(
            maximumBytes: 1 << 20, artifactIdentity: "generated-fixture",
            templateIdentity: "fixture-template", numericalProfile: "canonical-strict-metal")
    }
    private func run(
        _ engine: CBv2NativeBlockEngine, id: UInt64, tokens: [Int],
        scope: String?, enabled: Bool = true
    ) async throws -> (String, [Int], CBv2Usage) {
        let stream = try engine.submit(
            .init(
                id: .init(id), promptTokens: tokens,
                sampling: .init(seed: 341), maxTokens: 4, cacheSalt: scope,
                prefixCacheEnabled: enabled))
        var text = ""
        var raw = [Int]()
        var terminal: CBv2Usage?
        for await event in stream {
            switch event {
            case .delta(let value, let ids, _):
                text += value
                raw += ids
            case .finished(let reason, let usage):
                #expect(reason == .stop || reason == .length)
                #expect(terminal == nil)
                terminal = usage
            }
        }
        return (text, raw, try #require(terminal))
    }

    @Test func exactAppendScopesDisabledAndShutdownAreLossless() async throws {
        let (directory, _) = try DiffusionGemmaFactoryTests().fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await DiffusionGemmaModelFactory.shared.load(
            from: directory,
            using: DiffusionGemmaFactoryTests.Loader(fail: false))
        let engine = try context.makeNativeEngine(
            kvBytesCapacity: 16 << 20, prefillChunkSize: 4, prefixCache: config())
        let control = try context.makeNativeEngine(kvBytesCapacity: 16 << 20, prefillChunkSize: 4)
        let prompt = [2, 5, 7, 9, 11, 13]
        let cold = try await run(engine, id: 1, tokens: prompt, scope: "tenant-a")
        #expect(cold.2.prefixCacheOutcome == .miss && cold.2.prefixCachePrefillTokensSaved == 0)
        let warm = try await run(engine, id: 2, tokens: prompt, scope: "tenant-a")
        #expect(warm.0 == cold.0 && warm.1 == cold.1)
        #expect(warm.2.prefixCachePrefillTokensSaved == prompt.count)
        #expect(warm.2.prefixCacheTier == .memorySnapshot)
        #expect(warm.2.prefixCacheStrategy == .direct && warm.2.prefixCacheReplayTokens == 0)
        let appended = prompt + [17, 19, 23]
        let reference = try await run(control, id: 1, tokens: appended, scope: nil)
        let hit = try await run(engine, id: 3, tokens: appended, scope: "tenant-a")
        #expect(hit.0 == reference.0 && hit.1 == reference.1)
        #expect(
            hit.2.prefixCachePrefillTokensSaved == 4,
            "Append must preserve the cold chunk boundary, not reuse unaligned6")
        let other = try await run(engine, id: 4, tokens: prompt, scope: "tenant-b")
        #expect(other.1 == cold.1 && other.2.prefixCachePrefillTokensSaved == 0)
        let branch = Array(prompt.prefix(4)) + [31, 33, 35, 37]
        let branchCold = try await run(control, id: 2, tokens: branch, scope: nil)
        let branchHit = try await run(engine, id: 7, tokens: branch, scope: "tenant-a")
        #expect(branchHit.1 == branchCold.1 && branchHit.2.prefixCachePrefillTokensSaved == 4)
        let disabled = try await run(
            engine, id: 5, tokens: prompt, scope: "tenant-a", enabled: false)
        #expect(disabled.1 == cold.1 && disabled.2.prefixCacheOutcome == .skippedPolicy)
        let unscoped = try await run(engine, id: 6, tokens: prompt, scope: nil)
        #expect(unscoped.1 == cold.1 && unscoped.2.prefixCacheOutcome == .skippedPolicy)
        #expect(engine.capacity().kvBytesInUse > 0)
        #expect(
            engine.capacity().kvBytesReserved == 1 << 20,
            "Resident partition stays inside the slot grant")
        await engine.shutdown()
        await control.shutdown()
        #expect(engine.capacity().kvBytesInUse == 0 && engine.capacity().kvBytesReserved == 0)
    }

    @Test func stagedCancellationCapacityAndIdentityChangesCannotPublishHits() async throws {
        let (directory, _) = try DiffusionGemmaFactoryTests().fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await DiffusionGemmaModelFactory.shared.load(
            from: directory,
            using: DiffusionGemmaFactoryTests.Loader(fail: false))
        let store = DiffusionGemmaResidentPrefixCache(configuration: try config(), chunkSize: 4)
        let tokens: [Int32] = [2, 3, 4, 5]
        let request = CBv2Request(
            id: .init(1), promptTokens: tokens.map(Int.init), maxTokens: 4, cacheSalt: "a")
        let identity = try #require(try store.identity(for: request))
        let state = try context.model.makeCache(expectedPromptLength: tokens.count)
        _ = try context.model.encode(
            tokenIds: MLXArray(tokens).reshaped(1, tokens.count), cache: state)
        eval(state.stateArrays())
        let cancelled = UUID()
        try store.capture(
            model: context.model, cache: state, promptTokens: tokens,
            request: cancelled, identity: identity)
        #expect(store.retainedBytes > 0 && store.lookup(tokens: tokens, identity: identity) == nil)
        store.finish(request: cancelled, successful: false)
        #expect(store.retainedBytes == 0 && store.lookup(tokens: tokens, identity: identity) == nil)
        let completed = UUID()
        try store.capture(
            model: context.model, cache: state, promptTokens: tokens,
            request: completed, identity: identity)
        store.finish(request: completed, successful: true)
        let checkpoint = try #require(store.lookup(tokens: tokens, identity: identity))
        #expect(checkpoint.compact)
        let changed = try DiffusionGemmaPrefixIdentity(
            tenantScope: identity.tenantScope,
            artifact: identity.artifact, template: identity.template + "-changed",
            media: identity.media,
            numericalProfile: identity.numericalProfile, epoch: identity.epoch)
        #expect(store.lookup(tokens: tokens, identity: changed) == nil)
        #expect(store.lookup(tokens: [2, 3, 4, 6], identity: identity) == nil)
        let reloaded = DiffusionGemmaResidentPrefixCache(configuration: try config(), chunkSize: 4)
        let nextEpoch = try #require(try reloaded.identity(for: request))
        #expect(store.lookup(tokens: tokens, identity: nextEpoch) == nil)
        store.trim(to: 0)
        #expect(store.retainedBytes == 0 && store.lookup(tokens: tokens, identity: identity) == nil)
    }

    @Test func shrinkEvictsSnapshotsAndShutdownReleasesWeightsWhileEngineStaysAlive() async throws {
        let (directory, _) = try DiffusionGemmaFactoryTests().fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        var context: DiffusionGemmaContext? = try await DiffusionGemmaModelFactory.shared.load(
            from: directory,
            using: DiffusionGemmaFactoryTests.Loader(fail: false))
        weak var model = context?.model
        let engine = try context!.makeNativeEngine(
            kvBytesCapacity: 16 << 20, prefillChunkSize: 4, prefixCache: config())
        let prompt = [2, 5, 7, 9, 11, 13]
        let cold = try await run(engine, id: 1, tokens: prompt, scope: "a")
        #expect(engine.capacity().kvBytesInUse > 0)
        engine.updateKVBytesCapacity(1 << 19)
        let deadline = ContinuousClock.now.advanced(by: .seconds(2))
        while engine.capacity().kvBytesInUse != 0 && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(1))
        }
        #expect(engine.capacity().kvBytesInUse == 0 && engine.capacity().kvBytesReserved == 0)
        let afterShrink = try await run(engine, id: 2, tokens: prompt, scope: "a")
        #expect(afterShrink.1 == cold.1 && afterShrink.2.prefixCacheOutcome == .skippedCapacity)
        context = nil
        await engine.shutdown()
        #expect(model == nil, "Cache hooks must not keep the model alive after shutdown")
        engine.updateKVBytesCapacity(16 << 20)
        try await Task.sleep(for: .milliseconds(10))
        #expect(engine.capacity().kvBytesInUse == 0 && engine.capacity().kvBytesReserved == 0)
    }

    @Test func aGrownCacheCannotExceedTheAlreadyClaimedProcessReservation() async throws {
        let (directory, _) = try DiffusionGemmaFactoryTests().fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try await DiffusionGemmaModelFactory.shared.load(from: directory,
            using: DiffusionGemmaFactoryTests.Loader(fail: false))
        let engine = try context.makeNativeEngine(kvBytesCapacity: 16 << 20,
            prefillChunkSize: 4, prefixCache: config())
        engine.updateKVBytesCapacity(1 << 19)
        var request = CBv2Request(id: .init(901), promptTokens: [2, 3, 4, 5, 6, 7],
            sampling: .init(seed: 341), maxTokens: 4, cacheSalt: "a")
        // The bridge claims this amount while the resident cache cannot fit.
        request.nativeReservationBytes = try engine.estimatedRequestBytes(request)
        engine.updateKVBytesCapacity(16 << 20)
        #expect(try engine.estimatedRequestBytes(request) > request.nativeReservationBytes!)
        var usage: CBv2Usage?
        for await event in try engine.submit(request) {
            if case .finished(let reason, let value) = event {
                #expect(reason == .stop || reason == .length)
                usage = value
            }
        }
        #expect(usage?.prefixCachePrefillTokensSaved == 0)
        #expect(engine.capacity().kvBytesInUse == 0,
            "The larger post-claim cache budget must not create unfunded snapshots")
        request.id = .init(902)
        request.nativeReservationBytes = 0
        #expect(throws: CBv2KVError.self) { try engine.submit(request) }
        await engine.shutdown()
    }
}
