// BenchWorkerLaunchTests.swift
//
// bench-worker's argv ORDER. Every argument refusal must be answerable from
// argv alone: benchd points the worker at a checkpoint that can be 113 GB,
// so a wrong flag reported as "cannot read <weights>/config.json" both names
// the wrong thing and, on a real box, only after the operator has waited.
//
// The `--weights` path used throughout is deliberately one that does not
// exist: if any check reached the checkpoint, these tests would report a
// filesystem error instead of the flag.

import Foundation
import Testing

@testable import MLXRunners

@Suite("bench-worker launch arguments")
struct BenchWorkerLaunchTests {

    private let absentCheckpoint = "/nonexistent"

    private func parse(_ arguments: [String]) throws -> BenchWorkerLaunchOptions {
        try BenchWorkerLaunchOptions.parse(arguments)
    }

    @Test("The official spawn parses")
    func officialSpawn() throws {
        let launch = try parse([
            "runtime-worker", "--weights", absentCheckpoint,
            "--speculative-protocol", "v1.1",
        ])
        #expect(launch.weights.path == absentCheckpoint)
        #expect(launch.speculative)
        #expect(launch.trusted == false)
        #expect(launch.runnerID == nil)
    }

    @Test("Without the flag the launch is plain v1")
    func plainV1Spawn() throws {
        let launch = try parse(["runtime-worker", "--weights", absentCheckpoint])
        #expect(launch.speculative == false)
    }

    /// The reported defect: a wrong `--speculative-protocol` answered with a
    /// checkpoint read.
    @Test("A wrong --speculative-protocol refuses before the checkpoint is read")
    func speculativeProtocolRefusedFirst() {
        #expect(throws: SpeculativeProtocol.ArgumentError.unsupportedVersion("v9")) {
            _ = try parse([
                "runtime-worker", "--weights", absentCheckpoint,
                "--speculative-protocol", "v9",
            ])
        }
    }

    @Test("A duplicate --resource refuses before the checkpoint is read")
    func duplicateResourceRefusedFirst() throws {
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("launch-\(UUID().uuidString).bin")
        try Data([0x00]).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }

        #expect(throws: RunnerResourceArgumentError.duplicateName("ngram")) {
            _ = try parse([
                "runtime-worker", "--weights", absentCheckpoint,
                "--resource", "ngram=\(file.path)",
                "--resource", "ngram=\(file.path)",
            ])
        }
    }

    @Test("A missing --resource path refuses before the checkpoint is read")
    func missingResourceRefusedFirst() {
        #expect(
            throws: RunnerResourceArgumentError.pathMissing(
                name: "ngram", path: "/nonexistent/table.bin")
        ) {
            _ = try parse([
                "runtime-worker", "--weights", absentCheckpoint,
                "--resource", "ngram=/nonexistent/table.bin",
            ])
        }
    }

    @Test("An empty --runner id refuses before the checkpoint is read")
    func emptyRunnerIDRefusedFirst() {
        #expect(throws: WorkerLaunchError.emptyRunnerID) {
            _ = try parse([
                "runtime-worker", "--weights", absentCheckpoint, "--runner", "   ",
            ])
        }
    }

    @Test("An unknown argument, a missing value and a bad --kv-bytes all refuse")
    func argumentShapeRefusals() {
        #expect(throws: WorkerLaunchError.unknownArgument("--nope")) {
            _ = try parse(["runtime-worker", "--weights", absentCheckpoint, "--nope"])
        }
        #expect(throws: WorkerLaunchError.missingValue("--weights")) {
            _ = try parse(["runtime-worker", "--weights"])
        }
        #expect(throws: WorkerLaunchError.badValue("--kv-bytes")) {
            _ = try parse([
                "runtime-worker", "--weights", absentCheckpoint, "--kv-bytes", "lots",
            ])
        }
        #expect(throws: WorkerLaunchError.missingWeights) {
            _ = try parse(["runtime-worker", "--speculative-protocol", "v1.1"])
        }
    }

    /// The ordering stated as an invariant rather than inferred from one
    /// case: parsing a launch whose checkpoint does not exist SUCCEEDS, so
    /// nothing in it can have read the checkpoint. Resolution is the first
    /// thing that does, and it is a separate call.
    @Test("Parsing never reads the checkpoint; resolveRunner is what does")
    func parsingNeverReadsTheCheckpoint() throws {
        let launch = try parse([
            "runtime-worker", "--weights", absentCheckpoint,
            "--speculative-protocol", "v1.1", "--trusted",
        ])
        #expect(launch.trusted)

        #expect(throws: RunnerError.self) {
            _ = try launch.resolveRunner()
        }
    }

    @Test("An unknown --runner id refuses without reading the checkpoint")
    func unknownRunnerID() throws {
        let launch = try parse([
            "runtime-worker", "--weights", absentCheckpoint, "--runner", "layr/nope",
        ])
        #expect(throws: WorkerLaunchError.unknownRunner("layr/nope")) {
            _ = try launch.resolveRunner()
        }
    }
}
