// Copyright © 2026 Eigen Labs.
//
// MLXRunners — bench-worker's argv, parsed and VALIDATED before the worker
// touches the checkpoint.
//
// The ordering is the point of this file. benchd points the worker at a
// checkpoint that can be 113 GB; a bad argument must be reported from argv
// alone, before any read under `--weights` — `config.json` included. A
// worker that resolves the runner first answers a wrong
// `--speculative-protocol` with "cannot read /nonexistent/config.json",
// which names the wrong thing and, on a real box, only after the operator
// has waited for a checkpoint read.
//
// So `BenchWorkerLaunchOptions.parse` performs EVERY argument check and
// performs NO read under `--weights`. Resolving the runner and loading the
// model happen afterwards, in the executable, over an already-validated
// launch.
//
// In `MLXRunners` rather than in the executable so the ordering is testable
// in process; a test target that depended on the executable would be run by
// SwiftPM as the swift-testing host (see Package.swift).

import Foundation

/// Refusals raised from argv, before any checkpoint access.
public enum WorkerLaunchError: Error, CustomStringConvertible, Equatable {
    case missingValue(String)
    case badValue(String)
    case unknownArgument(String)
    case missingWeights
    case emptyRunnerID
    case unknownRunner(String)
    /// `resident` was asked for without the socket it must listen on.
    case missingSocket
    /// `--socket` was given to a subcommand that does not listen. The
    /// attaching worker is told where the resident is by the environment,
    /// not by argv, so a `--socket` on `runtime-worker` means the caller
    /// believes something about this process that is not true.
    case socketNotAccepted(String)
    /// More than one subcommand, or none.
    case ambiguousSubcommand(String)

    public var description: String {
        switch self {
        case .missingValue(let name): return "\(name) requires a value"
        case .badValue(let name): return "\(name) has an unusable value"
        case .unknownArgument(let name): return "unknown argument \(name)"
        case .missingWeights: return "--weights <dir> is required"
        case .emptyRunnerID: return "--runner requires a runner id"
        case .unknownRunner(let id): return "no runner with id \(id)"
        case .missingSocket: return "resident requires --socket <path>"
        case .socketNotAccepted(let subcommand):
            return "--socket is only accepted by resident, not \(subcommand)"
        case .ambiguousSubcommand(let detail):
            return "expected exactly one subcommand (\(detail))"
        }
    }
}

/// What this invocation is: a per-phase worker, or the window's resident.
public enum BenchWorkerSubcommand: String, Sendable, Equatable {
    /// One phase. Serves in process, or attaches to a resident when
    /// `BENCH_WORKER_RESIDENT_SOCKET` names one.
    case runtimeWorker = "runtime-worker"
    /// One window. Loads ONCE and listens; see `BenchWorkerResident`.
    case resident
}

/// One validated invocation.
public struct BenchWorkerLaunchOptions: Sendable {
    /// Which mode was asked for. Defaults to `runtime-worker`, which is what
    /// benchd spawns.
    public var subcommand: BenchWorkerSubcommand
    /// The resident's socket path. Required by `resident`, refused on
    /// `runtime-worker` (which reads the environment instead).
    public var socket: String?
    /// Checkpoint directory. NOT read during parsing — only carried.
    public var weights: URL
    /// Explicit runner id, or nil to resolve by the checkpoint's
    /// `model_type`.
    public var runnerID: String?
    public var drafter: URL?
    public var trusted: Bool
    public var kvBytesCapacity: Int
    /// `--speculative-protocol v1.1` was given. See ``SpeculativeProtocol``.
    public var speculative: Bool
    /// Validated resource bag, ready for `RunnerLoadOptions`.
    public var resources: RunnerResources

    /// Default KV grant when `--kv-bytes` is absent.
    public static let defaultKVBytesCapacity = 8 << 30

    /// Parse and validate `arguments` (argv minus the executable name).
    ///
    /// Every refusal this can raise is raised HERE, from argv and from the
    /// resource paths the caller named. Nothing under `--weights` is opened,
    /// stat'ed or listed, so a wrong flag is reported as a wrong flag rather
    /// than as a checkpoint that could not be read.
    public static func parse(
        _ arguments: [String],
        fileManager: FileManager = .default
    ) throws -> BenchWorkerLaunchOptions {
        var weights: URL?
        var runnerID: String?
        var drafter: URL?
        var trusted = false
        var kvBytesCapacity = defaultKVBytesCapacity
        var speculativeProtocol: String?
        var rawResources: [String] = []
        var subcommand: BenchWorkerSubcommand?
        var socket: String?

        var index = 0
        while index < arguments.count {
            let argument = arguments[index]
            index += 1
            func next(_ name: String) throws -> String {
                guard index < arguments.count else {
                    throw WorkerLaunchError.missingValue(name)
                }
                defer { index += 1 }
                return arguments[index]
            }
            switch argument {
            case "runtime-worker", "resident":
                guard subcommand == nil else {
                    throw WorkerLaunchError.ambiguousSubcommand(
                        "\(subcommand!.rawValue) then \(argument)")
                }
                subcommand = BenchWorkerSubcommand(rawValue: argument)
            case "--socket": socket = try next("--socket")
            case "--weights": weights = URL(fileURLWithPath: try next("--weights"))
            case "--runner": runnerID = try next("--runner")
            case "--drafter": drafter = URL(fileURLWithPath: try next("--drafter"))
            case "--trusted": trusted = true
            case "--resource": rawResources.append(try next("--resource"))
            case "--speculative-protocol":
                speculativeProtocol = try next("--speculative-protocol")
            case "--kv-bytes":
                guard let bytes = Int(try next("--kv-bytes")), bytes > 0 else {
                    throw WorkerLaunchError.badValue("--kv-bytes")
                }
                kvBytesCapacity = bytes
            default: throw WorkerLaunchError.unknownArgument(argument)
            }
        }

        // ARGUMENT validation. None of it reads the checkpoint.
        let speculative = try SpeculativeProtocol.isEnabled(speculativeProtocol)
        // benchd spawns `runtime-worker`, and the flag has been optional
        // since v1; an invocation naming no subcommand is that one.
        let resolvedSubcommand = subcommand ?? .runtimeWorker
        switch resolvedSubcommand {
        case .resident:
            guard let socket, !socket.trimmingCharacters(in: .whitespaces).isEmpty else {
                throw WorkerLaunchError.missingSocket
            }
        case .runtimeWorker:
            guard socket == nil else {
                throw WorkerLaunchError.socketNotAccepted(
                    resolvedSubcommand.rawValue)
            }
        }
        if let runnerID {
            guard !runnerID.trimmingCharacters(in: .whitespaces).isEmpty else {
                throw WorkerLaunchError.emptyRunnerID
            }
        }
        let resources = try RunnerResourceArguments.parse(
            rawResources, fileManager: fileManager)
        guard let weights else { throw WorkerLaunchError.missingWeights }

        return BenchWorkerLaunchOptions(
            subcommand: resolvedSubcommand,
            socket: socket,
            weights: weights,
            runnerID: runnerID,
            drafter: drafter,
            trusted: trusted,
            kvBytesCapacity: kvBytesCapacity,
            speculative: speculative,
            resources: resources)
    }

    /// The runner this launch names: the explicit `--runner` id, or the one
    /// claiming the checkpoint's `model_type`.
    ///
    /// The FIRST thing that reads under `--weights`, and deliberately a
    /// separate call from `parse` so the ordering is visible at the call
    /// site rather than buried.
    public func resolveRunner(
        registry: RunnerRegistry = .shared
    ) throws -> any Runner.Type {
        guard let runnerID else {
            return try registry.resolve(checkpoint: weights)
        }
        guard
            let match = registry.manifests().first(where: { $0.runnerID == runnerID }),
            let modelType = match.modelTypes.first
        else {
            throw WorkerLaunchError.unknownRunner(runnerID)
        }
        return try registry.resolve(modelType: modelType)
    }

    public init(
        subcommand: BenchWorkerSubcommand = .runtimeWorker,
        socket: String? = nil,
        weights: URL,
        runnerID: String? = nil,
        drafter: URL? = nil,
        trusted: Bool = false,
        kvBytesCapacity: Int = BenchWorkerLaunchOptions.defaultKVBytesCapacity,
        speculative: Bool = false,
        resources: RunnerResources = RunnerResources()
    ) {
        self.subcommand = subcommand
        self.socket = socket
        self.weights = weights
        self.runnerID = runnerID
        self.drafter = drafter
        self.trusted = trusted
        self.kvBytesCapacity = kvBytesCapacity
        self.speculative = speculative
        self.resources = resources
    }
}
