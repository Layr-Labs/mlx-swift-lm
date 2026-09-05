// EmbeddedHeadProvenanceTests.swift
//
// The one embedded-head provenance rule (contract §12c), over a synthetic
// checkpoint: two shards and an index that puts the `mtp.*` tensors in one of
// them. What the rule claims is that the digest names the HEAD, so the shard
// that holds no head tensor must not reach the hash at all.
//
// Model-free by construction: the shards are bytes on disk, and nothing here
// loads weights or touches Metal.

import Foundation
import Testing

@testable import MLXRunners

@Suite("Embedded head provenance")
struct EmbeddedHeadProvenanceTests {

    /// A checkpoint directory: `shardA` carries the head, `shardB` does not.
    private struct Checkpoint {
        let directory: URL
        let shardA: URL
        let shardB: URL
    }

    private func makeCheckpoint(
        headTensors: [String], shardA: Data, shardB: Data
    ) throws -> Checkpoint {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("mtp-provenance-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)

        let a = directory.appendingPathComponent("model-00001-of-00002.safetensors")
        let b = directory.appendingPathComponent("model-00002-of-00002.safetensors")
        try shardA.write(to: a)
        try shardB.write(to: b)

        var weightMap: [String: String] = [
            "model.layers.0.self_attn.q_proj.weight": a.lastPathComponent,
            "lm_head.weight": b.lastPathComponent,
        ]
        for tensor in headTensors { weightMap[tensor] = a.lastPathComponent }
        let index = try JSONSerialization.data(withJSONObject: ["weight_map": weightMap])
        try index.write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))

        return Checkpoint(directory: directory, shardA: a, shardB: b)
    }

    @Test("Only the shards carrying mtp.* are hashed, sized and counted")
    func hashesTheHeadShardAlone() throws {
        let head = ["mtp.fc_embedding.weight", "mtp.layers.0.mlp.down_proj.weight"]
        let checkpoint = try makeCheckpoint(
            headTensors: head,
            shardA: Data(repeating: 0xA1, count: 4096),
            shardB: Data(repeating: 0xB2, count: 8192))
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }

        let provenance = try #require(
            try RunnerCheckpoint.provenance(ofEmbeddedHeadAt: checkpoint.directory))
        // The head shard alone: its 4096 bytes, and one file.
        #expect(provenance.bytes == 4096)
        #expect(provenance.fileCount == 1)

        // Rewriting the shard that holds no head tensor cannot move the
        // digest. This is the claim the rule makes, and the reason the whole
        // checkpoint is not hashed.
        try Data(repeating: 0xCC, count: 8192).write(to: checkpoint.shardB)
        #expect(
            try RunnerCheckpoint.provenance(ofEmbeddedHeadAt: checkpoint.directory)?
                .sha256 == provenance.sha256)

        // Rewriting the head shard does move it.
        try Data(repeating: 0xA2, count: 4096).write(to: checkpoint.shardA)
        #expect(
            try RunnerCheckpoint.provenance(ofEmbeddedHeadAt: checkpoint.directory)?
                .sha256 != provenance.sha256)
    }

    @Test("A different head tensor name is a different digest over the same bytes")
    func hashesTheIndexEntries() throws {
        let bytes = Data(repeating: 0xA1, count: 4096)
        let other = Data(repeating: 0xB2, count: 8192)
        let first = try makeCheckpoint(
            headTensors: ["mtp.fc_embedding.weight"], shardA: bytes, shardB: other)
        let second = try makeCheckpoint(
            headTensors: ["mtp.fc_hidden.weight"], shardA: bytes, shardB: other)
        defer {
            try? FileManager.default.removeItem(at: first.directory)
            try? FileManager.default.removeItem(at: second.directory)
        }

        #expect(
            try RunnerCheckpoint.provenance(ofEmbeddedHeadAt: first.directory)?.sha256
                != RunnerCheckpoint.provenance(ofEmbeddedHeadAt: second.directory)?.sha256)
    }

    @Test("A checkpoint with no mtp.* has no head provenance")
    func noHeadIsNil() throws {
        let checkpoint = try makeCheckpoint(
            headTensors: [],
            shardA: Data(repeating: 0xA1, count: 4096),
            shardB: Data(repeating: 0xB2, count: 8192))
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }

        // Not a digest of the target weights, and not a refusal: a checkpoint
        // without a head is a serial-only model.
        #expect(try RunnerCheckpoint.provenance(ofEmbeddedHeadAt: checkpoint.directory) == nil)
    }

    /// FAIL CLOSED. A checkpoint with no head gives nil, but a checkpoint
    /// whose index cannot be read is BROKEN, and the rule refuses it. This is
    /// what `adopt` relies on: it calls the helper with `try`, so this throw
    /// stops the adoption instead of shipping an unattributed head.
    @Test("A corrupt index refuses instead of reporting no provenance")
    func corruptIndexRefuses() throws {
        let checkpoint = try makeCheckpoint(
            headTensors: ["mtp.fc_embedding.weight"],
            shardA: Data(repeating: 0xA1, count: 4096),
            shardB: Data(repeating: 0xB2, count: 8192))
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }
        try Data("{ not json".utf8).write(
            to: checkpoint.directory.appendingPathComponent(
                "model.safetensors.index.json"))

        #expect(throws: RunnerError.self) {
            try RunnerCheckpoint.provenance(ofEmbeddedHeadAt: checkpoint.directory)
        }
    }
}
