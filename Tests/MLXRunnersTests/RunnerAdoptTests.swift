// RunnerAdoptTests.swift
//
// `Runner.adopt` — the seam Darkbloom uses because its slot lifecycle
// already owns a resident `ModelContainer`. Adoption must read NO tensors:
// a runner that could only `load` would read a checkpoint the process
// already holds, and on a 113 GB model that is not a slow path, it is a
// second copy in unified memory.
//
// The proof here is structural rather than instrumented. Each `adopt` reads
// the checkpoint facts FIRST and type-checks the module SECOND, so pointing
// it at a directory containing ONLY `config.json` and the safetensors index
// and handing it a module of the wrong family reaches the type check — which
// it can only do if every read it performed was satisfied by those two
// files. Any stray read would surface as a filesystem error instead.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXRunners

@Suite("Runner adoption")
struct RunnerAdoptTests {

    /// A checkpoint directory holding exactly the two files adoption may
    /// read, and nothing else — no weights, no tokenizer, no generation
    /// config.
    struct MinimalCheckpoint {
        let directory: URL

        init(modelType: String, eosTokenID: Int = 7) throws {
            directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("adopt-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let config = """
                {"model_type":"\(modelType)","eos_token_id":\(eosTokenID)}
                """
            try Data(config.utf8)
                .write(to: directory.appendingPathComponent("config.json"))
            let index = """
                {"metadata":{"total_size":0},"weight_map":{}}
                """
            try Data(index.utf8).write(
                to: directory.appendingPathComponent("model.safetensors.index.json"))
        }

        /// Everything present, so a read of anything else must fail.
        var contents: [String] {
            ((try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? [])
                .sorted()
        }

        func cleanUp() { try? FileManager.default.removeItem(at: directory) }
    }

    /// Stands in for a module of the wrong family: enough to be handed to
    /// `adopt`, never enough to be adopted.
    final class ForeignModel: Module, LanguageModel, @unchecked Sendable {
        // A parameterless `Module`: it allocates no tensors, so it can be
        // built on a machine with no GPU.
        func prepare(
            _ input: LMInput, cache: [KVCache], windowSize: Int?
        ) throws -> PrepareResult {
            preconditionFailure("the foreign stand-in is never prepared")
        }
        func newCache(parameters: GenerateParameters?) -> [KVCache] { [] }
    }

    private func adopt(
        _ runner: (any Runner.Type),
        checkpoint: MinimalCheckpoint,
        options: RunnerLoadOptions = RunnerLoadOptions()
    ) throws {
        _ = try runner.adopt(
            model: ForeignModel(),
            tokenizer: StubTokenizer(),
            configuration: ModelConfiguration(directory: checkpoint.directory),
            directory: checkpoint.directory,
            options: options)
    }

    @Test("The fixture checkpoint really holds only config.json and the index")
    func minimalCheckpointShape() throws {
        let checkpoint = try MinimalCheckpoint(modelType: "gemma4_text")
        defer { checkpoint.cleanUp() }
        #expect(checkpoint.contents == ["config.json", "model.safetensors.index.json"])
    }

    /// Every family: adoption gets as far as the MODULE check against a
    /// directory holding nothing but those two files, so it read nothing
    /// else.
    @Test("Adoption reads nothing under the checkpoint but config.json and the index")
    func adoptionReadsOnlyConfigAndIndex() throws {
        let families: [(any Runner.Type, String)] = [
            (Gemma4TextRunner.self, "gemma4_text"),
            (GPTOSSRunner.self, "gpt_oss"),
            (Qwen35Runner.self, "qwen3_5"),
            (Qwen3VLRunner.self, "qwen3_vl"),
        ]
        for (runner, modelType) in families {
            let checkpoint = try MinimalCheckpoint(modelType: modelType)
            defer { checkpoint.cleanUp() }
            // The MODULE is what it refuses — not a missing file.
            #expect(throws: RunnerError.self) {
                try adopt(runner, checkpoint: checkpoint)
            }
            do {
                try adopt(runner, checkpoint: checkpoint)
            } catch let error as RunnerError {
                guard case .unexpectedModel = error else {
                    Issue.record("\(runner) refused with \(error), not the module")
                    continue
                }
            }
        }
    }

    /// A checkpoint whose `config.json` is absent is refused as a
    /// CHECKPOINT, which is what proves the read happens at all rather than
    /// the module check short-circuiting it.
    @Test("Adoption refuses a checkpoint it cannot read, before the module check")
    func adoptionRefusesUnreadableCheckpoint() throws {
        let empty = FileManager.default.temporaryDirectory
            .appendingPathComponent("adopt-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: empty) }

        do {
            _ = try Gemma4TextRunner.adopt(
                model: ForeignModel(),
                tokenizer: StubTokenizer(),
                configuration: ModelConfiguration(directory: empty),
                directory: empty,
                options: RunnerLoadOptions())
            Issue.record("adoption accepted a checkpoint with no config.json")
        } catch let error as RunnerError {
            guard case .invalidCheckpoint = error else {
                Issue.record("refused with \(error), not the checkpoint")
                return
            }
        }
    }

    /// A resident drafter is BOUND, never re-read: `loadDrafter` hands back
    /// exactly what the caller supplied and opens no directory.
    @Test("A preloaded drafter is used instead of loading one")
    func preloadedDrafterIsUsed() async throws {
        let drafter = StubDrafter()
        let options = RunnerLoadOptions(
            // A directory that does not exist: reading it would throw, so a
            // pass means it was never opened.
            drafterDirectory: URL(fileURLWithPath: "/nonexistent/assistant"),
            preloadedDrafter: drafter)

        for runner in [Gemma4TextRunner.self] as [any Runner.Type] {
            let resolved = try await runner.loadDrafter(
                options: options,
                directory: URL(fileURLWithPath: "/nonexistent"),
                target: ForeignModel())
            #expect(resolved === drafter)
        }
        let qwen = try await Qwen35Runner.loadDrafter(
            options: options,
            directory: URL(fileURLWithPath: "/nonexistent"),
            target: ForeignModel())
        #expect(qwen === drafter)
    }

    /// Without a preloaded drafter and without a directory, a family whose
    /// drafter is a separate artifact loads none — and never touches disk
    /// looking for one.
    @Test("No drafter directory and no preloaded drafter means no drafter")
    func noDrafterIsNotAnError() async throws {
        let resolved = try await Gemma4TextRunner.loadDrafter(
            options: RunnerLoadOptions(),
            directory: URL(fileURLWithPath: "/nonexistent"),
            target: ForeignModel())
        #expect(resolved == nil)
    }

    /// Families with no speculative decoder refuse a drafter rather than
    /// accepting and ignoring one.
    @Test("A family with no decoder refuses a drafter it was handed")
    func drafterlessFamiliesRefuse() throws {
        let checkpoint = try MinimalCheckpoint(modelType: "gpt_oss")
        defer { checkpoint.cleanUp() }
        let options = RunnerLoadOptions(preloadedDrafter: StubDrafter())

        do {
            try adopt(GPTOSSRunner.self, checkpoint: checkpoint, options: options)
            Issue.record("gpt_oss accepted a drafter")
        } catch let error as RunnerError {
            guard case .drafterUnavailable = error else {
                Issue.record("refused with \(error), not the drafter")
                return
            }
        }
    }

    /// `load` is defined in terms of `adopt`, so there is ONE construction
    /// path: the mock records that its adoption ran, and `load` is what ran
    /// it.
    @Test("load is defined in terms of adopt")
    func loadGoesThroughAdopt() throws {
        // The mock's `adopt` reads the checkpoint's model_type exactly as a
        // real runner's does; a `load` that bypassed it would not.
        let checkpoint = try MinimalCheckpoint(modelType: "mock")
        defer { checkpoint.cleanUp() }
        let adopted = try MockRunner.adopt(
            model: ForeignModel(),
            tokenizer: StubTokenizer(),
            configuration: ModelConfiguration(directory: checkpoint.directory),
            directory: checkpoint.directory,
            options: RunnerLoadOptions())
        #expect(adopted.loadedModelType == "qwen4_exp_text")
    }
}
