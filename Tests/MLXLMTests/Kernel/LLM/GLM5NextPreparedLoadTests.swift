// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {
    /// `GLM5NextArtifactLoader.prepare`: the single verified read must build
    /// exactly the arrays the per-tensor loader builds (every parameter's
    /// shape, dtype and bytes), for each tensor-parallel rank and whole.
    @Suite(.serialized)
    struct GLM5NextPreparedLoadTests {
        private static let layers = "model.language_model.layers."

        /// The NVFP4 hybrid fixture with two attention and two KDA heads and
        /// 128 routed-expert channels, so a two-rank tensor-parallel plan
        /// divides every split tensor, the affine shared experts' one-group
        /// scales included.
        private static func tensorParallelConfiguration() throws -> GLM5NextConfiguration {
            let encoded = try JSONEncoder().encode(GLM5NextNVFP4InventoryFixture.configuration())
            var root = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
            var text = try #require(root["text_config"] as? [String: Any])
            text["num_attention_heads"] = 2
            text["num_key_value_heads"] = 2
            text["moe_intermediate_size"] = 128
            var linear = try #require(text["linear_attn_config"] as? [String: Any])
            linear["num_heads"] = 2
            text["linear_attn_config"] = linear
            root["text_config"] = text
            return try JSONDecoder().decode(
                GLM5NextConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
        }

        private static func identity(_ fixture: GLM5NextNVFP4ArtifactFixture) throws
            -> GLM5NextArtifactIdentity
        {
            let files = try Dictionary(
                uniqueKeysWithValues: (["config.json", "model.safetensors.index.json"]
                    + fixture.shards).map { name in
                    let bytes = try Data(contentsOf: fixture.url(name))
                    let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }
                        .joined()
                    return (name, GLM5NextArtifactIdentity.File(bytes: bytes.count, sha256: digest))
                })
            return .init(
                repository: "synthetic/glm5-next-nvfp4",
                revision: String(repeating: "c", count: 40), files: files)
        }

        private static func loader(
            _ fixture: GLM5NextNVFP4ArtifactFixture, _ identity: GLM5NextArtifactIdentity,
            _ verification: GLM5NextArtifactVerification, roots: GLM5NextArtifactLoadRoots
        ) throws -> GLM5NextArtifactLoader {
            try GLM5NextArtifactLoader(
                directory: fixture.directory, expectedIdentity: identity, storage: .nvfp4Hybrid,
                constructionRoots: roots, verification: verification)
        }

        /// Every parameter's shape, dtype and SHA-256 of its bytes, by key path;
        /// with `gpu`, also a sum the GPU computes from the buffer itself.
        private static func digests(_ modules: [Module], gpu: Bool) -> [String: String] {
            var result: [String: String] = [:]
            for (index, module) in modules.enumerated() {
                for (key, array) in module.parameters().flattened() {
                    let data = array.asData(access: .copy).data
                    let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }
                        .joined()
                    let sum = gpu ? " \(array.asType(.float32).sum(stream: .gpu).item(Float.self))" : ""
                    result["\(index).\(key)"] = "\(array.shape) \(array.dtype) \(digest)\(sum)"
                }
            }
            return result
        }

        /// Loads `plan` (and NextN with a final plan) both ways and returns
        /// both digests.
        private static func bothWays(
            _ fixture: GLM5NextNVFP4ArtifactFixture, plan: GLM5NextTextStagePlan, mtp: Bool,
            gpu: Bool = false
        ) throws -> (legacy: [String: String], prepared: [String: String]) {
            let identity = try identity(fixture)
            func load(_ verification: GLM5NextArtifactVerification) throws -> [String: String] {
                let roots = GLM5NextArtifactLoadRoots()
                defer {
                    StreamOrDevice.default.stream.synchronize()
                    roots.releaseAfterConfirmedFence()
                }
                let loader = try Self.loader(fixture, identity, verification, roots: roots)
                if verification == .withPreparedPayload {
                    try loader.prepare(plan: plan, includeMTP: mtp)
                }
                let stage = try loader.loadTextStage(plan: plan)
                let head = mtp ? try loader.loadMTP(stage: stage) : nil
                if mtp { try loader.requireConsumed(.mtp) }
                return digests([stage] + (head.map { [$0] } ?? []), gpu: gpu)
            }
            return (try load(.beforeLoad), try load(.withPreparedPayload))
        }

        // MARK: - Byte selection

        @Test func byteRunsSelectExactlyTheSliceBytesOfEveryRank() throws {
            try Device.withDefaultDevice(.cpu) {
                let p = Self.layers + "3."
                let cases: [(String, [Int], DType)] = [
                    (p + "self_attn.q_proj.weight", [8, 6], .uint32),
                    (p + "self_attn.A_log", [4], .float32),
                    (p + "self_attn.q_conv1d.weight", [6, 1, 4], .bfloat16),
                    (p + "mlp.experts.1.gate_proj.scales", [4, 6], .uint8),
                    (p + "mlp.experts.1.down_proj.weight", [6, 4], .uint32),
                    (p + "mlp.experts.1.down_proj.scales", [3, 8], .uint8),
                    (p + "self_attn.o_proj.weight", [5, 8, 3], .bfloat16),
                    (p + "mlp.experts.0.gate_proj.weight_scale_2", [], .float32),
                    ("model.language_model.embed_tokens.weight", [6, 4], .uint32),
                ]
                for rank in 0 ..< 2 {
                    let shard = try GLM5NextTensorShard(rank: rank, size: 2)
                    for (name, shape, dtype) in cases {
                        let bytes = shape.reduce(1, *) * dtype.size
                        let payload = Data((0 ..< bytes).map { UInt8(($0 * 37 + 11) % 251) })
                        let expected = try shard.slice(name, MLXArray(payload, shape, dtype: dtype))
                        let runs = try shard.byteRuns(name, shape: shape, bytes: bytes)
                        #expect(try shard.localShape(name, shape) == expected.shape, "\(name)")
                        #expect(runs.bytes == expected.nbytes, "\(name)")
                        // Any windowing of the payload writes the same selection.
                        for width in [1, 3, 7, 16, bytes] {
                            var selected = Data(count: runs.bytes)
                            selected.withUnsafeMutableBytes { destination in
                                payload.withUnsafeBytes { source in
                                    for start in stride(from: 0, to: bytes, by: width) {
                                        runs.copy(
                                            window: start ..< min(bytes, start + width),
                                            from: source.baseAddress! + start,
                                            to: destination.baseAddress!)
                                    }
                                }
                            }
                            #expect(
                                selected == expected.asData(access: .copy).data,
                                "\(name) rank \(rank) window \(width)")
                        }
                    }
                }
            }
        }

        @Test func segmentsCopyAtTheirFileOffsetFromAnyWindow() {
            // A column split of a [4, 6]-byte payload at file offset 37.
            let runs = GLM5NextByteRuns(offset: 3, length: 3, count: 4, stride: 6)
            let file = (0 ..< 100).map { UInt8($0) }
            var expected: [UInt8] = []
            for row in 0 ..< 4 { expected += file[(37 + row * 6 + 3) ..< (37 + row * 6 + 6)] }
            for width in [1, 2, 5, 8, 16, 100] {
                var out = [UInt8](repeating: 0, count: 12)
                out.withUnsafeMutableBytes { destination in
                    let segment = GLM5NextPreparedSegment(
                        base: 37, runs: runs, destination: destination.baseAddress!)
                    #expect(segment.start == 40 && segment.end == 61)
                    file.withUnsafeBytes { source in
                        for start in stride(from: 0, to: file.count, by: width) {
                            segment.copy(
                                window: start ..< min(file.count, start + width),
                                from: source.baseAddress! + start)
                        }
                    }
                }
                #expect(out == expected, "window \(width)")
            }
        }

        @Test func banksAreCompleteRoutedExpertStacksInExpertOrder() {
            let p = Self.layers + "3.mlp."
            let names: Set<String> = [
                p + "experts.1.gate_proj.weight", p + "experts.0.gate_proj.weight",
                p + "experts.0.down_proj.scales", p + "experts.1.down_proj.scales",
                p + "experts.0.up_proj.weight",  // expert 1 missing: not a bank
                p + "shared_experts.gate_proj.weight", p + "gate.weight",
                p + "experts.01.gate_proj.scales", p + "experts.1.gate_proj.scales",
            ]
            let banks = Set(GLM5NextPreparedBanks.group(names, experts: 2))
            #expect(
                banks == [
                    [p + "experts.0.gate_proj.weight", p + "experts.1.gate_proj.weight"],
                    [p + "experts.0.down_proj.scales", p + "experts.1.down_proj.scales"],
                ])
        }

        /// Names the byte matchers and Foundation's search must read alike:
        /// every split and replicated pattern, at several layers and experts,
        /// near misses, and the same with non-ASCII bytes (the fallback).
        static func nameCorpus() -> [String] {
            let parts = [
                "self_attn.q_proj.weight", "self_attn.k_proj.scales", "self_attn.v_proj.biases",
                "self_attn.q_conv1d.weight", "self_attn.f_b_proj.weight", "self_attn.g_b_proj.scales",
                "self_attn.b_proj.weight", "self_attn.q_b_proj.weight", "self_attn.kv_b_proj.biases",
                "self_attn.o_proj.weight", "self_attn.o_proj.weight_scale_2", "self_attn.A_log",
                "self_attn.dt_bias", "self_attn.A_logx", "mlp.gate_proj.weight", "mlp.up_proj.scales",
                "mlp.down_proj.weight", "mlp.down_proj.weight_scale_2", "mlp.shared_experts.down_proj.weight",
                "mlp.experts.7.gate_proj.weight", "mlp.experts.287.down_proj.scales",
                "mlp.experts.07.up_proj.weight", "mlp.experts.+5.up_proj.weight", "mlp.experts..up_proj",
                "mlp.experts.12", "mlp.experts.99999999999999999999.up_proj.weight",
                "mlp.gate.weight", "self_attn.indexer.wk.weight", "input_layernorm.weight", "down_proj.",
            ]
            var names: [String] = []
            for layer in ["0", "3", "44", "45"] {
                for part in parts { names.append("model.language_model.layers.\(layer).\(part)") }
            }
            names += [
                "model.language_model.embed_tokens.weight", "lm_head.weight", "model.visual.blocks.0.mlp.up_proj.weight",
                "model.language_model.layers.", "", ".self_attn.q_proj.", "model.language_model.layers.3.self_attn.q_proj",
            ]
            // The same with a non-ASCII byte in a few places (Foundation decides those).
            names += names.prefix(40).flatMap { name in
                ["\u{e9}" + name, name + "\u{301}", name.replacingOccurrences(of: ".", with: "\u{2024}")]
            }
            return names
        }

        @Test func byteMatchersAgreeWithFoundationOnEveryName() throws {
            let fixture = try GLM5NextNVFP4ArtifactFixture(configuration: Self.tensorParallelConfiguration())
            for name in Self.nameCorpus() + Array(fixture.inventory.entries.keys) {
                #expect(GLM5NextTensorShard.splitAxis(name) == GLM5NextTensorShard.referenceSplitAxis(name), "\(name)")
                let fast = GLM5NextPreparedBanks.member(name), reference = GLM5NextPreparedBanks.referenceMember(name)
                #expect(fast?.0 == reference?.0 && fast?.1 == reference?.1, "\(name)")
            }
            #expect(GLM5NextTensorShard.splitAxis(Self.layers + "3.mlp.experts.7.down_proj.weight") == 1)
            #expect(GLM5NextPreparedBanks.member(Self.layers + "3.mlp.experts.287.down_proj.scales")?.1 == 287)
        }

        // MARK: - Stage exactness

        /// On the GPU the per-tensor loader slices and stacks there, and the
        /// prepared arrays (no-copy Metal buffers) are read by a GPU sum.
        @Test(arguments: [false, true])
        func preparedTensorParallelRanksAreBitwiseThePerTensorLoad(gpu: Bool) throws {
            try Device.withDefaultDevice(gpu ? .gpu : .cpu) {
                let fixture = try GLM5NextNVFP4ArtifactFixture(
                    configuration: Self.tensorParallelConfiguration())
                let c = fixture.declared.textConfig
                var ranks: [[String: String]] = []
                for rank in 0 ..< 2 {
                    let plan = try GLM5NextTextStagePlan(
                        configuration: c, layers: 0 ..< c.hiddenLayers,
                        shard: GLM5NextTensorShard(rank: rank, size: 2))
                    let (legacy, prepared) = try Self.bothWays(fixture, plan: plan, mtp: true, gpu: gpu)
                    #expect(!legacy.isEmpty)
                    print("GLM53_PREPARED rank \(rank) gpu \(gpu): \(legacy.count) arrays compared")
                    #expect(Set(prepared.keys) == Set(legacy.keys))
                    for (key, value) in legacy where prepared[key] != value {
                        Issue.record("rank \(rank) \(key): \(value) != \(prepared[key] ?? "missing")")
                    }
                    ranks.append(legacy)
                }
                // Each rank keeps its own half of the split tensors.
                #expect(ranks[0].keys == ranks[1].keys)
                #expect(ranks[0].contains { ranks[1][$0.key] != $0.value })
            }
        }

        @Test(arguments: [false, true])
        func preparedWholeStagesAreBitwiseThePerTensorLoad(final: Bool) throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextNVFP4ArtifactFixture()
                let c = fixture.declared.textConfig
                let plan = try GLM5NextTextStagePlan(
                    configuration: c, layers: final ? 2 ..< c.hiddenLayers : 0 ..< 2,
                    loadMTPEmbedding: final)
                let (legacy, prepared) = try Self.bothWays(fixture, plan: plan, mtp: final)
                #expect(!legacy.isEmpty && prepared == legacy)
            }
        }

        // MARK: - Integrity

        @Test func aCorruptedFileFailsThePassAndNothingLoads() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextNVFP4ArtifactFixture()
                let identity = try Self.identity(fixture)
                let c = fixture.declared.textConfig
                let plan = try GLM5NextTextStagePlan(
                    configuration: c, layers: 0 ..< c.hiddenLayers)
                // One payload byte, past the header, in place (same size).
                let url = fixture.url(fixture.shards[1])
                var bytes = try Data(contentsOf: url)
                bytes[bytes.count - 3] ^= 0x5a
                try bytes.write(to: url)
                let roots = GLM5NextArtifactLoadRoots()
                defer { roots.releaseAfterConfirmedFence() }
                // Size-matching files pass the initializer, which hashes nothing.
                let loader = try Self.loader(
                    fixture, identity, .withPreparedPayload, roots: roots)
                #expect(throws: GLM5NextArtifactError.invalid("SHA-256 \(fixture.shards[1])")) {
                    try loader.prepare(plan: plan, includeMTP: true)
                }
                #expect(throws: GLM5NextArtifactError.invalid("prepare has not verified this artifact")) {
                    _ = try loader.loadTextStage(plan: plan)
                }
                let name = try #require(fixture.inventory.entries.keys.min())
                #expect(throws: GLM5NextArtifactError.invalid("payload read before the artifact is verified")) {
                    _ = try loader.tensor(name)
                }
                // The same corruption also fails the verify-first owner.
                #expect(throws: GLM5NextArtifactError.invalid("SHA-256 \(fixture.shards[1])")) {
                    _ = try Self.loader(fixture, identity, .beforeLoad, roots: roots)
                }
            }
        }

        @Test func aFileChangedAfterParsingFailsThePass() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextNVFP4ArtifactFixture()
                let identity = try Self.identity(fixture)
                let c = fixture.declared.textConfig
                let plan = try GLM5NextTextStagePlan(
                    configuration: c, layers: 0 ..< c.hiddenLayers)
                let roots = GLM5NextArtifactLoadRoots()
                defer { roots.releaseAfterConfirmedFence() }
                let loader = try Self.loader(
                    fixture, identity, .withPreparedPayload, roots: roots)
                // Rewritten with the same bytes: a new inode and change time.
                let url = fixture.url(fixture.shards[0])
                let bytes = try Data(contentsOf: url)
                try FileManager.default.removeItem(at: url)
                try bytes.write(to: url)
                #expect(throws: GLM5NextArtifactError.self) {
                    try loader.prepare(plan: plan, includeMTP: false)
                }
            }
        }

        @Test func aPreparedOwnerBuildsOnlyThePlanItPrepared() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextNVFP4ArtifactFixture()
                let identity = try Self.identity(fixture)
                let c = fixture.declared.textConfig
                let whole = try GLM5NextTextStagePlan(configuration: c, layers: 0 ..< c.hiddenLayers)
                let roots = GLM5NextArtifactLoadRoots()
                defer {
                    StreamOrDevice.default.stream.synchronize()
                    roots.releaseAfterConfirmedFence()
                }
                let loader = try Self.loader(
                    fixture, identity, .withPreparedPayload, roots: roots)
                try loader.prepare(plan: whole, includeMTP: false)
                #expect(throws: GLM5NextArtifactError.invalid("prepare needs a fresh .withPreparedPayload owner")) {
                    try loader.prepare(plan: whole, includeMTP: false)
                }
                let other = try GLM5NextTextStagePlan(configuration: c, layers: 0 ..< 2)
                #expect(throws: GLM5NextArtifactError.invalid("text stage differs from the prepared plan")) {
                    _ = try loader.loadTextStage(plan: other)
                }
                #expect(throws: GLM5NextArtifactError.invalid("prepared payload was admitted before its pass")) {
                    _ = try loader.loadTextStage(plan: whole, beforeTensorRead: { _, _ in })
                }
                // Not prepared with NextN: it is read per tensor, after the pass verified it.
                let stage = try loader.loadTextStage(plan: whole)
                _ = try loader.loadMTP(stage: stage)
                try loader.requireConsumed(.mtp)
            }
        }
    }
}

extension KernelTests {
    /// The prepared pass's byte selection on the real checkpoint's layouts:
    /// for every tensor of a KDA and dense layer, a DSA and MoE layer (four of
    /// its experts), the embedding, norm and head, each rank's bytes copied
    /// out of 4 MiB file windows equal `GLM5NextTensorShard.slice` of the whole
    /// tensor. Reads those tensors only (a few GB, cache bypassed); runs when
    /// `GLM53_REAL_MODEL_DIR` names the NVFP4 model directory.
    @Suite(.serialized)
    struct GLM5NextPreparedLoadRealTests {
        static let directory = ProcessInfo.processInfo.environment["GLM53_REAL_MODEL_DIR"]
            .map { URL(fileURLWithPath: $0, isDirectory: true) }

        @Test(.enabled(if: directory != nil))
        func realLayoutsSelectTheSliceBytesOfEachRank() throws {
            let directory = try #require(Self.directory)
            let index = try #require(
                JSONSerialization.jsonObject(
                    with: Data(contentsOf: directory.appendingPathComponent("model.safetensors.index.json")))
                    as? [String: Any])
            let map = try #require(index["weight_map"] as? [String: String])
            let layers = "model.language_model.layers."
            let names = map.keys.filter { name in
                if name.hasPrefix(layers + "0.") { return true }
                if name.hasPrefix(layers + "3.") {
                    guard let marker = name.range(of: ".experts.") else { return true }
                    let rest = name[marker.upperBound...]
                    return Int(rest.prefix { $0.isNumber }).map { $0 < 4 } ?? false
                }
                return !name.hasPrefix(layers) && !name.contains("visual")
            }.sorted()
            #expect(names.count > 100)
            let window = GLM5NextArtifactLoader.preparedWindowBytes
            var headers: [String: (start: Int, tensors: [String: Any], bytes: Int)] = [:]
            var checked = 0
            for name in names {
                let file = map[name]!
                let url = directory.appendingPathComponent(file)
                if headers[file] == nil {
                    let handle = try FileHandle(forReadingFrom: url)
                    defer { try? handle.close() }
                    let prefix = try #require(try handle.read(upToCount: 8))
                    let length = prefix.enumerated().reduce(0) { $0 | Int($1.element) << ($1.offset * 8) }
                    let header = try #require(try handle.read(upToCount: length))
                    let size = try #require(
                        FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int)
                    headers[file] = (
                        8 + length,
                        try #require(JSONSerialization.jsonObject(with: header) as? [String: Any]), size
                    )
                }
                let (start, tensors, fileBytes) = headers[file]!
                let tensor = try #require(tensors[name] as? [String: Any])
                let shape = try #require(tensor["shape"] as? [Int])
                let offsets = try #require(tensor["data_offsets"] as? [Int])
                let dtypeName = try #require(tensor["dtype"] as? String)
                let stored = try #require(GLM5NextWeightInventory.storedTypes[dtypeName])
                let (offset, bytes) = (start + offsets[0], offsets[1] - offsets[0])
                let fd = open(url.path, O_RDONLY)
                defer { close(fd) }
                _ = fcntl(fd, F_NOCACHE, 1)
                let payload = try GLM5NextAlignedPayloadReader.read(
                    descriptor: fd, fileSize: fileBytes, offset: offset, count: bytes, checkUnchanged: {})
                try Device.withDefaultDevice(.cpu) {
                    let full = MLXArray(payload, shape, dtype: stored.dtype)
                    for rank in 0 ..< 2 {
                        let shard = try GLM5NextTensorShard(rank: rank, size: 2)
                        let expected = try shard.slice(name, full).asData(access: .copy).data
                        let runs = try shard.byteRuns(name, shape: shape, bytes: bytes)
                        var selected = Data(count: runs.bytes)
                        selected.withUnsafeMutableBytes { destination in
                            payload.withUnsafeBytes { source in
                                let segment = GLM5NextPreparedSegment(
                                    base: offset, runs: runs, destination: destination.baseAddress!)
                                var position = offset / window * window
                                while position < offset + bytes {
                                    let lower = max(position, offset)
                                    let upper = min(position + window, offset + bytes)
                                    segment.copy(
                                        window: lower ..< upper,
                                        from: source.baseAddress! + (lower - offset))
                                    position += window
                                }
                            }
                        }
                        #expect(selected == expected, "\(name) rank \(rank)")
                    }
                }
                checked += 1
            }
            print("GLM53_PREPARED_REAL tensors=\(checked)")
        }
    }
}

extension KernelTests.GLM5NextPreparedLoadRealTests {
    /// The prepared owner's initializer on the real checkpoint: metadata only
    /// (stamps, strict JSON of config, index and every header, the inventory),
    /// no payload byte read or hashed. Prints its time, the load's `open` phase.
    @Test(.enabled(if: directory != nil))
    func realPreparedOwnerParsesMetadataOnly() throws {
        let directory = try #require(Self.directory)
        let manifest = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
                as? [String: Any])
        var files: [String: GLM5NextArtifactIdentity.File] = [:]
        for file in try #require(manifest["files"] as? [[String: Any]]) {
            let path = try #require(file["path"] as? String)
            guard path == "config.json" || path == "model.safetensors.index.json" || path.hasSuffix(".safetensors")
            else { continue }
            files[path] = .init(
                bytes: try #require(file["size_bytes"] as? Int), sha256: try #require(file["sha256"] as? String))
        }
        let start = ContinuousClock.now
        let loader = try GLM5NextArtifactLoader(
            directory: directory,
            expectedIdentity: .init(
                repository: "EigenLabs/GLM-5.3-Flash-MLX-NVFP4-mtp", revision: String(repeating: "0", count: 40),
                files: files),
            storage: .nvfp4Hybrid, verification: .withPreparedPayload)
        let elapsed = start.duration(to: .now)
        print("GLM53_PREPARED_REAL owner_init=\(elapsed) tensors=\(loader.inventory.entries.count)")
        #expect(loader.inventory.entries.count == 114_160)
        // Every real name: the byte matchers read it as Foundation does.
        var splits = 0
        for name in loader.inventory.entries.keys {
            let axis = GLM5NextTensorShard.splitAxis(name)
            #expect(axis == GLM5NextTensorShard.referenceSplitAxis(name), "\(name)")
            let fast = GLM5NextPreparedBanks.member(name), reference = GLM5NextPreparedBanks.referenceMember(name)
            #expect(fast?.0 == reference?.0 && fast?.1 == reference?.1, "\(name)")
            if axis != nil { splits += 1 }
        }
        print("GLM53_PREPARED_REAL names=\(loader.inventory.entries.count) split=\(splits)")
    }
}
