import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    /// Buffers one MTP history chunk really allocates, against the history work
    /// `GLM5NextMTPExecution.memoryPlan` declares for it. The counts come from
    /// process-global allocator counters, so run this suite alone:
    /// `DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST=1 scripts/run-nested-suite.sh
    /// GLM5NextMTPHistoryWorkTests --no-parallel`.
    @Suite(.serialized)
    struct GLM5NextMTPHistoryWorkTests {
        private struct Measurement {
            let plan: GLM5NextMTPExecution.MemoryPlan
            /// Allocator resources the chunk created, live or cached. The cache
            /// starts empty and nothing is evicted, so this bounds the most
            /// buffers the chunk held at once.
            let buffers: Int
            /// Bytes of those resources.
            let bytes: Int
            /// Peak active bytes above the evaluated inputs.
            let peak: Int
        }

        /// Waits for completion handlers to release finished work, then empties
        /// the cache so no earlier buffer can be reused by the measured chunk.
        private func settle() {
            Stream.gpu.synchronize()
            var active = Memory.snapshot().activeMemory
            for _ in 0 ..< 100 {
                Thread.sleep(forTimeInterval: 0.01)
                let next = Memory.snapshot().activeMemory
                if next == active { break }
                active = next
            }
            Memory.clearCache()
        }

        /// One chunk as `Owner.stageHistory` builds it: `prepareHistory` over
        /// `rows` rows after `sourceRows` evaluated history rows, then the bank
        /// arrays of `Owner.bankArrays` (a first chunk copies its four planes;
        /// every bank adds its uint8 validity).
        private func measure(
            _ mtp: GLM5NextMTP, dtype: DType, sourceRows: Int, rows: Int,
            limits: CBv2MTPAllocationLimits
        ) throws -> Measurement {
            let c = mtp.configuration
            func tokens(_ count: Int, from start: Int) -> MLXArray {
                MLXArray((start ..< start + count).map { Int32($0 % c.vocabularySize) })
                    .reshaped([1, count])
            }
            func bank(_ cache: GLM5NextDSACache, appendedTo source: GLM5NextDSACache?)
                -> [MLXArray]
            {
                let state = cache.state
                let history =
                    (source?.offset ?? 0) > 0
                    ? state : state.map { MLX.where(MLXArray(true), $0, $0) }
                return history + [state[3].asType(.uint8)]
            }
            var source: GLM5NextDSACache?
            var sourceRoots: [MLXArray] = []
            if sourceRows > 0 {
                let primed = try mtp.prepareHistory(
                    shiftedTokens: tokens(sourceRows, from: 1),
                    hidden: MLXArray.ones([1, sourceRows, c.hiddenSize], dtype: dtype),
                    cache: GLM5NextDSACache(growth: .exact))
                sourceRoots = bank(primed, appendedTo: nil)
                let cache = GLM5NextDSACache(growth: .exact)
                cache.state = Array(sourceRoots.prefix(4))
                source = cache
            }
            let chunkTokens = tokens(rows, from: sourceRows + 1)
            let chunkHidden = MLXArray.ones([1, rows, c.hiddenSize], dtype: dtype)
            try withError { error in
                eval(sourceRoots + [chunkTokens, chunkHidden])
                try error.check()
            }
            let plan = try GLM5NextMTPExecution.memoryPlan(
                configuration: c, maximumInputs: sourceRows + rows, limits: limits,
                executionPolicy: mtp.executionPolicy)
            settle()
            let before = Memory.snapshot()
            let resources = Memory.numResources
            Memory.peakMemory = 0
            let roots = try withError { error -> [MLXArray] in
                let result = try mtp.prepareHistory(
                    shiftedTokens: chunkTokens, hidden: chunkHidden,
                    cache: source ?? GLM5NextDSACache(growth: .exact))
                let roots = bank(result, appendedTo: source)
                eval(roots)
                Stream.gpu.synchronize()
                try error.check()
                return roots
            }
            let after = Memory.snapshot()
            let measurement = Measurement(
                plan: plan, buffers: Memory.numResources - resources,
                bytes: after.activeMemory - before.activeMemory + after.cacheMemory
                    - before.cacheMemory,
                peak: after.peakMemory - before.activeMemory)
            withExtendedLifetime((roots, sourceRoots, source)) {}
            return measurement
        }

        /// Measures each (sourceRows, rows) case with freed buffers kept cached,
        /// checks it against its plan and returns the buffer counts.
        private func check(
            _ mtp: GLM5NextMTP, dtype: DType, label: String, limits: CBv2MTPAllocationLimits,
            cases: [(sourceRows: Int, rows: Int)]
        ) throws -> [Int] {
            let policy = try #require(Memory.allocationFootprintPolicy())
            let extra = try #require(policy.maximumExtraBytes)
            func charged(_ specs: [CBv2MTPFixedBufferSpec]) throws -> Int {
                var total = 0
                for spec in specs {
                    total +=
                        try #require(policy.upperBound(byteCount: spec.logicalBytes))
                        * spec.allocationCount
                }
                return total
            }
            let cacheLimit = Memory.cacheLimit
            Memory.cacheLimit = 64 << 30
            defer {
                Memory.cacheLimit = cacheLimit
                Memory.clearCache()
            }
            var counts: [Int] = []
            for (sourceRows, rows) in cases {
                let m = try measure(
                    mtp, dtype: dtype, sourceRows: sourceRows, rows: rows, limits: limits)
                let declared =
                    try charged(m.plan.historyWorkBuffers) + charged(m.plan.cacheBuffers)
                // The per-chunk term has the largest count in the history work.
                let small = try #require(m.plan.historyWorkBuffers.map(\.allocationCount).max())
                print(
                    "GLM53_MTP_HISTORY \(label) source=\(sourceRows) rows=\(rows) "
                        + "buffers=\(m.buffers) small_term=\(small) bytes=\(m.bytes) "
                        + "peak=\(m.peak) declared=\(declared)")
                // Even if every buffer were sub-page, the per-chunk term alone
                // covers all of them, the bank arrays included.
                #expect(m.buffers <= small)
                #expect(m.peak <= declared)
                // A buffer's footprint bound exceeds its length by at most `extra`.
                #expect(m.bytes + m.buffers * extra <= declared)
                counts.append(m.buffers)
            }
            return counts
        }

        private func evaluateParameters(_ modules: [Module]) throws {
            try withError { error in
                eval(modules.flatMap { $0.parameters().flattened().map(\.1) })
                try error.check()
            }
        }

        /// The production loader on the packed Q4 fixture: packed embedding,
        /// BF16 activations and each policy's projections and norms. Every
        /// buffer here is small, so this counts everything a chunk allocates.
        @Test(
            .enabled(
                if: ProcessInfo.processInfo.environment["DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST"]
                    == "1",
                "Set DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST=1 and run this allocator check alone"),
            arguments: ["reference", "serving"])
        func packedChunkAllocatesAFixedNumberOfBuffers(policyName: String) throws {
            let fixture = try GLM5NextLocalArtifactFixture(q4: true)
            let loader = try GLM5NextArtifactLoader(
                directory: fixture.directory, expectedIdentity: fixture.identity(),
                storage: fixture.storage,
                executionPolicy: policyName == "serving" ? .serving : .reference)
            let target = try loader.loadTextModel()
            let mtp = try loader.loadMTP(target: target)
            try evaluateParameters([target, mtp])
            let dtype = target.embed(MLXArray([Int32(1)]).reshaped([1, 1])).dtype
            let counts = try check(
                mtp, dtype: dtype, label: "packed-\(policyName)",
                limits: .init(maximumPrefillTokens: 128, maximumDraftTokens: 1),
                cases: [(0, 1), (0, 128), (64, 1), (64, 128)])
            // 128 rows allocate no more buffers than one row, bar kernel scratch.
            #expect(counts[1] <= counts[0] + 16 && counts[3] <= counts[2] + 16)
        }

        /// GLM-5.3's history widths (hidden 4096, latent 512, index 128) at the
        /// serving prefill envelope, on a one-layer float32 model with zero
        /// weights. A 2,048-row chunk holds the plan's largest buffers.
        @Test(
            .enabled(
                if: ProcessInfo.processInfo.environment["DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST"]
                    == "1",
                "Set DARKBLOOM_EXCLUSIVE_NATIVE_GPU_TEST=1 and run this allocator check alone"),
            arguments: ["reference", "serving"])
        func envelopeChunkAtOfficialWidthsFitsItsPlan(policyName: String) throws {
            let base = try GLM5NextInventoryFixture.configuration(layers: 1)
            var root = try #require(
                JSONSerialization.jsonObject(with: JSONEncoder().encode(base)) as? [String: Any])
            var text = try #require(root["text_config"] as? [String: Any])
            for (key, value) in [
                "hidden_size": 4096, "kv_lora_rank": 512, "index_head_dim": 128,
                "max_position_embeddings": 8192,
            ] { text[key] = value }
            root["text_config"] = text
            var vision = try #require(root["vision_config"] as? [String: Any])
            vision["out_hidden_size"] = 4096
            root["vision_config"] = vision
            let configuration = try JSONDecoder().decode(
                GLM5NextConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
            let headers = try GLM5NextWeightInventory.expectedTensors(
                configuration: configuration, storage: .float32,
                shard: "official-widths.safetensors")
            let inventory = try GLM5NextWeightInventory.validate(
                configuration: configuration, tensors: headers, storage: .float32)
            func tensor(_ name: String) throws -> MLXArray {
                let shape = try #require(inventory.entries[name]).tensor.shape
                return name.contains("norm.weight") || name.hasSuffix("_scale")
                    ? MLXArray.ones(shape, dtype: .float32)
                    : MLXArray.zeros(shape, dtype: .float32)
            }
            let target = try GLM5NextTextModel(
                configuration: configuration.textConfig, inventory: inventory, tensor: tensor,
                executionPolicy: policyName == "serving" ? .serving : .reference)
            let mtp = try GLM5NextMTP(target: target, tensor: tensor)
            try evaluateParameters([target, mtp])
            let counts = try check(
                mtp, dtype: .float32, label: "official-\(policyName)",
                limits: .init(maximumPrefillTokens: 2048, maximumDraftTokens: 3),
                cases: [(0, 1), (0, 2048), (2048, 1), (2048, 2048)])
            #expect(counts[1] <= counts[0] + 16 && counts[3] <= counts[2] + 16)
        }
    }
}
