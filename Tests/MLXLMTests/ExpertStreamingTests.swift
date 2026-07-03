// Copyright © 2026 Eigen Labs Inc.

import Foundation
import MLX
import MLXNN
import MLXRandom
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// Tests for MoE expert SSD streaming (ExpertStreaming/*.swift):
///  (a) safetensors layout parsing (sharded, index.json-driven)
///  (b) per-expert byte-range math vs a whole-tensor load
///  (c) LRU cache eviction respects its byte budget
///  (d) end-to-end numeric parity: resident SwitchGLU vs
///      StreamingQuantizedSwitchGLU, decode (L=1) and prefill (L=16) shapes
final class ExpertStreamingTests: XCTestCase {

    private func makeTempDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dsv4-expert-streaming-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - (a) SafetensorsLayout: sharded, index.json-driven

    func testSafetensorsLayoutParsesShardedCheckpointViaIndex() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        let shard0: [String: MLXArray] = [
            "a.weight": MLXArray((0 ..< 24).map { Float($0) }, [2, 3, 4]),
            // Regression: real DeepSeek-V4 shards carry int64 hash-routing
            // tables (tid2eid). The parser scans whole shard headers, so it
            // must decode dtypes streaming never reads without failing.
            "a.tid2eid": MLXArray((0 ..< 8).map { Int64($0) }, [4, 2]),
        ]
        let shard1: [String: MLXArray] = [
            "b.weight": MLXArray((0 ..< 16).map { UInt32($0) }, [2, 8])
        ]
        try MLX.save(arrays: shard0, url: dir.appendingPathComponent("model-00001-of-00002.safetensors"))
        try MLX.save(arrays: shard1, url: dir.appendingPathComponent("model-00002-of-00002.safetensors"))

        let index: [String: [String: String]] = [
            "weight_map": [
                "a.weight": "model-00001-of-00002.safetensors",
                "a.tid2eid": "model-00001-of-00002.safetensors",
                "b.weight": "model-00002-of-00002.safetensors",
            ]
        ]
        let indexData = try JSONSerialization.data(withJSONObject: index)
        try indexData.write(to: dir.appendingPathComponent("model.safetensors.index.json"))

        let layout = try SafetensorsLayout.load(modelDirectory: dir)

        let aLoc = try XCTUnwrap(layout["a.weight"])
        XCTAssertEqual(aLoc.shape, [2, 3, 4])
        XCTAssertEqual(aLoc.dtype, .float32)
        XCTAssertEqual(aLoc.byteRange.count, 24 * 4)

        let tidLoc = try XCTUnwrap(layout["a.tid2eid"], "int64 tensors must parse, not fail the shard")
        XCTAssertEqual(tidLoc.dtype, .int64)
        XCTAssertEqual(tidLoc.byteRange.count, 8 * 8)

        let bLoc = try XCTUnwrap(layout["b.weight"])
        XCTAssertEqual(bLoc.shape, [2, 8])
        XCTAssertEqual(bLoc.dtype, .uint32)
        XCTAssertEqual(bLoc.byteRange.count, 16 * 4)

        // Read the raw bytes back at the computed absolute byte range and
        // confirm they reproduce the original array exactly -- this is the
        // real end-to-end assertion: byte-range math must agree with what
        // MLX.save actually wrote, not just be internally self-consistent.
        let handle = try FileHandle(forReadingFrom: bLoc.fileURL)
        handle.seek(toFileOffset: UInt64(bLoc.byteRange.lowerBound))
        let raw = handle.readData(ofLength: bLoc.byteRange.count)
        try handle.close()
        let recovered = MLXArray(raw, bLoc.shape, dtype: .uint32)
        let original = shard1["b.weight"]!
        eval(recovered, original)
        XCTAssertEqual(recovered.asArray(UInt32.self), original.asArray(UInt32.self))
    }

    /// No `model.safetensors.index.json` present: layout must fall back to
    /// scanning every `*.safetensors` file in the directory directly.
    func testSafetensorsLayoutFallsBackToDirectoryScanWithoutIndex() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        let arrays: [String: MLXArray] = [
            "solo.weight": MLXArray((0 ..< 12).map { Float($0) }, [3, 4])
        ]
        try MLX.save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))

        let layout = try SafetensorsLayout.load(modelDirectory: dir)
        let loc = try XCTUnwrap(layout["solo.weight"])
        XCTAssertEqual(loc.shape, [3, 4])
        XCTAssertEqual(loc.dtype, .float32)
    }

    // MARK: - (b) Per-expert byte-range math vs whole-tensor load

    func testExpertByteRangeMathMatchesWholeTensorLoad() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        let numExperts = 6
        MLXRandom.seed(41)
        // Packed-looking uint32 "weights" and small uint8 "scales", stacked
        // expert-major -- exactly the mlx-canonical switch_mlp layout.
        let gateW = MLXRandom.randInt(low: 0, high: 1 << 30, [numExperts, 8, 4], type: UInt32.self)
        let gateS = MLXRandom.randInt(low: 0, high: 255, [numExperts, 8, 2], type: UInt8.self)
        let upW = MLXRandom.randInt(low: 0, high: 1 << 30, [numExperts, 8, 4], type: UInt32.self)
        let upS = MLXRandom.randInt(low: 0, high: 255, [numExperts, 8, 2], type: UInt8.self)
        let downW = MLXRandom.randInt(low: 0, high: 1 << 30, [numExperts, 4, 4], type: UInt32.self)
        let downS = MLXRandom.randInt(low: 0, high: 255, [numExperts, 4, 2], type: UInt8.self)
        eval(gateW, gateS, upW, upS, downW, downS)

        let prefix = "model.layers.0.ffn.switch_mlp"
        try MLX.save(
            arrays: [
                "\(prefix).gate_proj.weight": gateW, "\(prefix).gate_proj.scales": gateS,
                "\(prefix).up_proj.weight": upW, "\(prefix).up_proj.scales": upS,
                "\(prefix).down_proj.weight": downW, "\(prefix).down_proj.scales": downS,
            ], url: dir.appendingPathComponent("model.safetensors"))

        let layout = try SafetensorsLayout.load(modelDirectory: dir)
        let store = ExpertShardStore(layout: layout, numExperts: numExperts)

        // Pure byte-range math, independent of any I/O.
        let gateWLoc = try XCTUnwrap(layout["\(prefix).gate_proj.weight"])
        for e in 0 ..< numExperts {
            let range = try ExpertShardStore.expertByteRange(
                of: gateWLoc, expert: e, numExperts: numExperts)
            let stride = gateWLoc.byteRange.count / numExperts
            XCTAssertEqual(range.count, stride)
            XCTAssertEqual(range.lowerBound, gateWLoc.byteRange.lowerBound + e * stride)
        }

        // Full read pipeline: fetch every expert and compare against slicing
        // the fully-loaded whole tensor.
        let fetched = try store.fetch(layerIndex: 0, experts: Array(0 ..< numExperts))
        for e in 0 ..< numExperts {
            let expert = try XCTUnwrap(fetched[e])
            let wantGateW = gateW[e]
            let wantGateS = gateS[e]
            let wantDownW = downW[e]
            eval(expert.gateWeight, wantGateW, expert.gateScales, wantGateS,
                expert.downWeight, wantDownW)
            XCTAssertEqual(expert.gateWeight.shape, wantGateW.shape)
            XCTAssertEqual(expert.gateWeight.asArray(UInt32.self), wantGateW.asArray(UInt32.self))
            XCTAssertEqual(expert.gateScales.asArray(UInt8.self), wantGateS.asArray(UInt8.self))
            XCTAssertEqual(expert.downWeight.asArray(UInt32.self), wantDownW.asArray(UInt32.self))
            XCTAssertNil(expert.gateBiases, "checkpoint has no bias tensors -- must resolve to nil")
        }
    }

    func testFetchThrowsOnMissingTensor() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        // Empty directory: no shards at all.
        let layout = try SafetensorsLayout.load(modelDirectory: dir)
        let store = ExpertShardStore(layout: layout, numExperts: 4)
        XCTAssertThrowsError(try store.fetch(layerIndex: 0, experts: [0])) { error in
            guard case ExpertShardStoreError.missingTensor = error else {
                XCTFail("expected .missingTensor, got \(error)")
                return
            }
        }
    }

    // MARK: - fd cache: one fd per shard file, reused across reads

    /// Regression for the fd-churn fix: before caching, every tensor read
    /// did its own `open()`/`close()` -- a single expert fetch (6 tensors:
    /// gate/up/down weight+scales) would have opened the shard file 6
    /// times, and fetching multiple experts across multiple layers sharing
    /// one shard file would open it once per tensor per expert per layer.
    /// After caching, the fd count must stay pinned at 1 for the whole
    /// store's lifetime, no matter how many tensors/experts/layers are
    /// read from that single file.
    func testExpertShardStoreReusesFileDescriptorAcrossReads() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        let numExperts = 6
        MLXRandom.seed(7)
        func mkStack(_ rows: Int, _ cols: Int) -> MLXArray {
            MLXRandom.randInt(low: 0, high: 1000, [numExperts, rows, cols], type: UInt32.self)
        }
        // Two layers sharing the SAME shard file (single-file checkpoint),
        // so the test exercises fd reuse across layers too, not just
        // within one layer's fetch.
        var arrays: [String: MLXArray] = [:]
        for layer in [0, 1] {
            let prefix = "model.layers.\(layer).ffn.switch_mlp"
            arrays["\(prefix).gate_proj.weight"] = mkStack(4, 4)
            arrays["\(prefix).gate_proj.scales"] = mkStack(4, 1)
            arrays["\(prefix).up_proj.weight"] = mkStack(4, 4)
            arrays["\(prefix).up_proj.scales"] = mkStack(4, 1)
            arrays["\(prefix).down_proj.weight"] = mkStack(4, 4)
            arrays["\(prefix).down_proj.scales"] = mkStack(4, 1)
        }
        eval(Array(arrays.values))
        try MLX.save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))

        let layout = try SafetensorsLayout.load(modelDirectory: dir)
        let store = ExpertShardStore(layout: layout, numExperts: numExperts)

        XCTAssertEqual(store.openFileDescriptorCountForTesting, 0, "no fd opened before first read")

        _ = try store.fetch(layerIndex: 0, experts: [0, 1, 2])
        XCTAssertEqual(
            store.openFileDescriptorCountForTesting, 1,
            "first fetch (3 experts x 6 tensors = 18 reads) must open exactly one fd")

        _ = try store.fetch(layerIndex: 0, experts: [3, 4, 5])
        _ = try store.fetch(layerIndex: 1, experts: [0, 1, 2, 3, 4, 5])
        XCTAssertEqual(
            store.openFileDescriptorCountForTesting, 1,
            "further fetches (including a different layer's tensors, same shard file) "
                + "must reuse the same cached fd, not open new ones")
    }

    /// Two DIFFERENT shard files must each get their own cached fd -- the
    /// cache is keyed by path, not a single global fd.
    func testExpertShardStoreCachesOneFileDescriptorPerDistinctShardFile() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        let numExperts = 2
        MLXRandom.seed(11)
        func mkStack() -> MLXArray {
            MLXRandom.randInt(low: 0, high: 1000, [numExperts, 4, 2], type: UInt32.self)
        }
        let shard0: [String: MLXArray] = [
            "model.layers.0.ffn.switch_mlp.gate_proj.weight": mkStack(),
            "model.layers.0.ffn.switch_mlp.gate_proj.scales": mkStack(),
            "model.layers.0.ffn.switch_mlp.up_proj.weight": mkStack(),
            "model.layers.0.ffn.switch_mlp.up_proj.scales": mkStack(),
            "model.layers.0.ffn.switch_mlp.down_proj.weight": mkStack(),
            "model.layers.0.ffn.switch_mlp.down_proj.scales": mkStack(),
        ]
        let shard1: [String: MLXArray] = [
            "model.layers.1.ffn.switch_mlp.gate_proj.weight": mkStack(),
            "model.layers.1.ffn.switch_mlp.gate_proj.scales": mkStack(),
            "model.layers.1.ffn.switch_mlp.up_proj.weight": mkStack(),
            "model.layers.1.ffn.switch_mlp.up_proj.scales": mkStack(),
            "model.layers.1.ffn.switch_mlp.down_proj.weight": mkStack(),
            "model.layers.1.ffn.switch_mlp.down_proj.scales": mkStack(),
        ]
        eval(Array(shard0.values) + Array(shard1.values))
        try MLX.save(arrays: shard0, url: dir.appendingPathComponent("model-00001-of-00002.safetensors"))
        try MLX.save(arrays: shard1, url: dir.appendingPathComponent("model-00002-of-00002.safetensors"))

        let layout = try SafetensorsLayout.load(modelDirectory: dir)
        let store = ExpertShardStore(layout: layout, numExperts: numExperts)

        _ = try store.fetch(layerIndex: 0, experts: [0, 1])
        XCTAssertEqual(store.openFileDescriptorCountForTesting, 1)

        _ = try store.fetch(layerIndex: 1, experts: [0, 1])
        XCTAssertEqual(
            store.openFileDescriptorCountForTesting, 2,
            "a second distinct shard file must get its own cached fd")
    }

    // MARK: - (c) LRU eviction respects byte budget

    func testExpertCacheEvictsLeastRecentlyUsedUnderByteBudget() {
        let placeholder = MLXArray.zeros([1])
        func fakeWeights(bytes: Int) -> ExpertWeights {
            ExpertWeights(
                gateWeight: placeholder, gateScales: placeholder, gateBiases: nil,
                upWeight: placeholder, upScales: placeholder, upBiases: nil,
                downWeight: placeholder, downScales: placeholder, downBiases: nil,
                byteCount: bytes)
        }

        // Budget fits exactly two 100-byte entries.
        let cache = ExpertCache(byteBudget: 250)

        cache.insert(layer: 0, expert: 0, weights: fakeWeights(bytes: 100))
        cache.insert(layer: 0, expert: 1, weights: fakeWeights(bytes: 100))
        XCTAssertNotNil(cache.get(layer: 0, expert: 0))  // touch: 0 becomes MRU, 1 is LRU
        cache.insert(layer: 0, expert: 2, weights: fakeWeights(bytes: 100))
        // Budget exceeded (300 > 250) -> evicts the LRU entry, which is now expert 1.
        XCTAssertNil(cache.get(layer: 0, expert: 1), "least-recently-used entry must be evicted")
        XCTAssertNotNil(cache.get(layer: 0, expert: 0), "recently-touched entry must survive")
        XCTAssertNotNil(cache.get(layer: 0, expert: 2), "just-inserted entry must survive")
        XCTAssertLessThanOrEqual(cache.stats.residentBytes, 250)
        XCTAssertEqual(cache.stats.residentCount, 2)

        // Different layers with the same expert index must not collide --
        // use a separate cache (budget pressure is already exercised above;
        // this is purely a key-identity check).
        let disambiguationCache = ExpertCache(byteBudget: 1000)
        disambiguationCache.insert(layer: 0, expert: 0, weights: fakeWeights(bytes: 11))
        disambiguationCache.insert(layer: 1, expert: 0, weights: fakeWeights(bytes: 22))
        XCTAssertEqual(disambiguationCache.get(layer: 0, expert: 0)?.byteCount, 11)
        XCTAssertEqual(disambiguationCache.get(layer: 1, expert: 0)?.byteCount, 22)
    }

    func testExpertCacheHitMissCounters() {
        let placeholder = MLXArray.zeros([1])
        let weights = ExpertWeights(
            gateWeight: placeholder, gateScales: placeholder, gateBiases: nil,
            upWeight: placeholder, upScales: placeholder, upBiases: nil,
            downWeight: placeholder, downScales: placeholder, downBiases: nil,
            byteCount: 10)
        let cache = ExpertCache(byteBudget: 1000)

        XCTAssertNil(cache.get(layer: 0, expert: 0))  // miss
        cache.insert(layer: 0, expert: 0, weights: weights)
        XCTAssertNotNil(cache.get(layer: 0, expert: 0))  // hit
        XCTAssertNotNil(cache.get(layer: 0, expert: 0))  // hit

        let stats = cache.stats
        XCTAssertEqual(stats.hits, 2)
        XCTAssertEqual(stats.misses, 1)
    }

    func testExpertCacheFetchResolvesMissesFromStore() throws {
        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        let numExperts = 3
        let gateW = MLXRandom.randInt(low: 0, high: 1000, [numExperts, 4, 2], type: UInt32.self)
        let gateS = MLXRandom.randInt(low: 0, high: 255, [numExperts, 4, 1], type: UInt8.self)
        let upW = MLXRandom.randInt(low: 0, high: 1000, [numExperts, 4, 2], type: UInt32.self)
        let upS = MLXRandom.randInt(low: 0, high: 255, [numExperts, 4, 1], type: UInt8.self)
        let downW = MLXRandom.randInt(low: 0, high: 1000, [numExperts, 2, 2], type: UInt32.self)
        let downS = MLXRandom.randInt(low: 0, high: 255, [numExperts, 2, 1], type: UInt8.self)
        eval(gateW, gateS, upW, upS, downW, downS)

        let prefix = "model.layers.2.ffn.switch_mlp"
        try MLX.save(
            arrays: [
                "\(prefix).gate_proj.weight": gateW, "\(prefix).gate_proj.scales": gateS,
                "\(prefix).up_proj.weight": upW, "\(prefix).up_proj.scales": upS,
                "\(prefix).down_proj.weight": downW, "\(prefix).down_proj.scales": downS,
            ], url: dir.appendingPathComponent("model.safetensors"))

        let layout = try SafetensorsLayout.load(modelDirectory: dir)
        let store = ExpertShardStore(layout: layout, numExperts: numExperts)
        let cache = ExpertCache(byteBudget: 1_000_000)

        let first = try cache.fetch(layer: 2, experts: [0, 1], from: store)
        XCTAssertEqual(Set(first.keys), Set([0, 1]))
        XCTAssertEqual(cache.stats.misses, 2)
        XCTAssertEqual(cache.stats.hits, 0)

        // Second call for the same experts must be all cache hits (no
        // additional store fetch needed -- store still resolves correctly
        // even if called again, but the cache should short-circuit first).
        let second = try cache.fetch(layer: 2, experts: [0, 1], from: store)
        XCTAssertEqual(cache.stats.hits, 2)
        eval(first[0]!.gateWeight, second[0]!.gateWeight)
        XCTAssertEqual(first[0]!.gateWeight.asArray(UInt32.self), second[0]!.gateWeight.asArray(UInt32.self))
    }

    // MARK: - (d) End-to-end parity: resident SwitchGLU vs StreamingQuantizedSwitchGLU

    /// Tiny config mirroring DeepseekV4Tests.makeConfig -- duplicated here
    /// (rather than shared) since the source file's helper is `private`.
    private func makeMoEConfig() throws -> DeepseekV4Configuration {
        let json = """
            {
                "vocab_size": 64,
                "hidden_size": 32,
                "moe_intermediate_size": 32,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "head_dim": 16,
                "q_lora_rank": 16,
                "qk_rope_head_dim": 8,
                "rms_norm_eps": 1e-6,
                "o_groups": 2,
                "o_lora_rank": 8,
                "sliding_window": 8,
                "compress_ratios": [0, 4, 128, 0],
                "compress_rope_theta": 160000.0,
                "n_routed_experts": 8,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "scoring_func": "sqrtsoftplus",
                "routed_scaling_factor": 1.5,
                "swiglu_limit": 10.0,
                "num_hash_layers": 1,
                "num_nextn_predict_layers": 0,
                "norm_topk_prob": true,
                "hc_mult": 4,
                "hc_sinkhorn_iters": 20,
                "hc_eps": 1e-6,
                "rope_theta": 10000.0,
                "rope_scaling": {
                    "type": "yarn",
                    "factor": 16,
                    "original_max_position_embeddings": 512,
                    "beta_fast": 32,
                    "beta_slow": 1
                },
                "max_position_embeddings": 8192,
                "index_n_heads": 4,
                "index_head_dim": 8,
                "index_topk": 4
            }
            """.data(using: .utf8)!
        return try JSONDecoder().decode(DeepseekV4Configuration.self, from: json)
    }

    /// Builds a `DeepseekV4MoE` (streaming disabled, so it gets a resident
    /// `SwitchGLU`), quantizes just its `switch_mlp` sub-tree to mxfp4 g32
    /// (the real checkpoint's routed-expert recipe), and randomizes the gate
    /// so different tokens route to different experts.
    private func makeQuantizedMoE(config: DeepseekV4Configuration, layerIdx: Int) throws
        -> DeepseekV4MoE
    {
        // Belt-and-suspenders: this test constructs the resident path
        // directly regardless of global state, but don't let a leaked
        // `enabled = true` from another test change which branch fires.
        DeepseekV4ExpertStreaming.enabled = false

        MLXRandom.seed(17)
        let moe = DeepseekV4MoE(config: config, layerIdx: layerIdx)

        try moe.gate.update(
            parameters: ModuleParameters.unflattened([
                "weight": MLXRandom.normal([config.nRoutedExperts, config.hiddenSize]),
                "e_score_correction_bias": MLXRandom.normal([config.nRoutedExperts]) * 0.1,
            ]), verify: .none)

        quantize(model: moe) { path, _ in
            path.contains("switch_mlp") ? (32, 4, .mxfp4) : nil
        }
        eval(moe)
        return moe
    }

    /// Save a quantized resident `SwitchGLU`'s tensors to a temp safetensors
    /// directory under the same key naming `ExpertShardStore` expects.
    private func saveSwitchMLP(
        _ switchMLP: SwitchGLU, layerIdx: Int, to dir: URL
    ) throws -> (groupSize: Int, bits: Int, mode: QuantizationMode) {
        guard let gateProj = switchMLP.gateProj as? QuantizedSwitchLinear,
            let upProj = switchMLP.upProj as? QuantizedSwitchLinear,
            let downProj = switchMLP.downProj as? QuantizedSwitchLinear
        else {
            XCTFail("switch_mlp sub-projections must be quantized")
            return (32, 4, .mxfp4)
        }

        let prefix = "model.layers.\(layerIdx).ffn.switch_mlp"
        try MLX.save(
            arrays: [
                "\(prefix).gate_proj.weight": gateProj.weight,
                "\(prefix).gate_proj.scales": gateProj.scales,
                "\(prefix).up_proj.weight": upProj.weight,
                "\(prefix).up_proj.scales": upProj.scales,
                "\(prefix).down_proj.weight": downProj.weight,
                "\(prefix).down_proj.scales": downProj.scales,
            ], url: dir.appendingPathComponent("model.safetensors"))

        return (gateProj.groupSize, gateProj.bits, gateProj.mode)
    }

    private func assertStreamingMatchesResident(seqLen: Int) throws {
        let config = try makeMoEConfig()
        let layerIdx = 3  // >= numHashLayers(1) -> bias routing, matches gate randomization below
        let moe = try makeQuantizedMoE(config: config, layerIdx: layerIdx)
        guard let switchMLP = moe.switchMLP else {
            XCTFail("resident switchMLP must be present when streaming is disabled")
            return
        }

        let dir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let (groupSize, bits, mode) = try saveSwitchMLP(switchMLP, layerIdx: layerIdx, to: dir)

        let layout = try SafetensorsLayout.load(modelDirectory: dir)
        let store = ExpertShardStore(layout: layout, numExperts: config.nRoutedExperts)
        let cache = ExpertCache(byteBudget: 1_000_000_000)

        let limit = config.swiguLimit
        let activationProduct: @Sendable (MLXArray, MLXArray) -> MLXArray = { gate, up in
            guard limit > 0 else { return silu(gate) * up }
            let g = MLX.minimum(gate, MLXArray(limit))
            let u = clip(up, min: MLXArray(-limit), max: MLXArray(limit))
            return silu(g) * u
        }

        // Force multiple fetch chunks (8 routed experts, chunk size 3) so
        // the run-length-encode + bucket-into-chunks path is genuinely
        // exercised, not just the single-chunk fast case.
        let streaming = StreamingQuantizedSwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            layerIndex: layerIdx,
            groupSize: groupSize, bits: bits, mode: mode,
            cache: cache, store: store,
            activationProduct: activationProduct,
            maxExpertsPerChunk: 3)

        MLXRandom.seed(99)
        let x = MLXRandom.normal([1, seqLen, config.hiddenSize])
        let (indices, _) = moe.gate(x, inputIds: nil)

        let resident = switchMLP(x, indices)
        let streamed = streaming(x, indices)
        eval(resident, streamed)

        XCTAssertEqual(resident.shape, streamed.shape)
        let maxDiff = abs(resident.asType(.float32) - streamed.asType(.float32)).max().item(Float.self)
        XCTAssertLessThan(
            maxDiff, 1e-5,
            "streamed switch_mlp output must match resident output within float precision "
                + "(seqLen=\(seqLen), maxDiff=\(maxDiff))")

        let stats = cache.stats
        XCTAssertGreaterThan(stats.misses, 0, "first run must miss and fetch from disk")
    }

    func testStreamingMatchesResidentDecode() throws {
        try assertStreamingMatchesResident(seqLen: 1)
    }

    func testStreamingMatchesResidentPrefill() throws {
        try assertStreamingMatchesResident(seqLen: 16)
    }

    // MARK: - (e) shouldEvalChunk: conditional per-chunk sync

    func testShouldEvalChunkAlwaysEvalsWhenMultipleChunks() {
        // Multiple chunks in one forward call -- must ALWAYS eval so the
        // next chunk's stacked tensors can reuse the freed memory, even
        // when this chunk is tiny.
        XCTAssertTrue(
            StreamingQuantizedSwitchGLU.shouldEvalChunk(
                totalChunksInThisCall: 2, chunkBytes: 1, evalThresholdBytes: 400 * 1024 * 1024))
        XCTAssertTrue(
            StreamingQuantizedSwitchGLU.shouldEvalChunk(
                totalChunksInThisCall: 5, chunkBytes: 0, evalThresholdBytes: 400 * 1024 * 1024))
    }

    func testShouldEvalChunkSkipsSingleSmallChunk() {
        // The decode fast path: exactly one chunk, comfortably under the
        // threshold -- must NOT eval (this is the whole point of the fix).
        let decodeChunkBytes = 6 * 13_600_000  // 6 experts, ~13.6 MB each
        XCTAssertFalse(
            StreamingQuantizedSwitchGLU.shouldEvalChunk(
                totalChunksInThisCall: 1, chunkBytes: decodeChunkBytes,
                evalThresholdBytes: 400 * 1024 * 1024))
    }

    func testShouldEvalChunkEvalsSingleChunkAboveThreshold() {
        // Single chunk, but big enough to matter (e.g. an unusually large
        // unique-expert count from batched decode) -- the byte-threshold
        // safety net must still force an eval.
        XCTAssertTrue(
            StreamingQuantizedSwitchGLU.shouldEvalChunk(
                totalChunksInThisCall: 1, chunkBytes: 500 * 1024 * 1024,
                evalThresholdBytes: 400 * 1024 * 1024))
    }

    func testShouldEvalChunkBoundaryIsExclusive() {
        let threshold = 400 * 1024 * 1024
        XCTAssertFalse(
            StreamingQuantizedSwitchGLU.shouldEvalChunk(
                totalChunksInThisCall: 1, chunkBytes: threshold, evalThresholdBytes: threshold),
            "exactly at threshold must NOT force an eval (strictly greater-than)")
        XCTAssertTrue(
            StreamingQuantizedSwitchGLU.shouldEvalChunk(
                totalChunksInThisCall: 1, chunkBytes: threshold + 1, evalThresholdBytes: threshold))
    }
}
