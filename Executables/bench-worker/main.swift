// Copyright © 2026 Eigen Labs.
//
// bench-worker — Engine Protocol v1 server over `Runner`.
//
// benchd spawns `<engine> runtime-worker --weights <DIR>`. One binary serves
// every runner: the checkpoint's `config.json` `model_type` picks it out of
// `RunnerRegistry`, or `--runner <id>` names it.
//
// A SHIM. The loop lives in `MLXRunners/BenchWorker.swift` so the
// conformance test can drive it in process; a test target that depended on
// this executable would be run by SwiftPM as the swift-testing host and
// would abort the package's whole `@Test` pass (see Package.swift's
// BenchCBv2Core/BenchCBv2 note).

import Foundation
import MLXLMCommon
import MLXRunners

#if canImport(Darwin)
import Darwin
#endif

/// Device identity for the hello.
///
/// The Metal device name is not exposed by mlx-swift at this pin, so the
/// probe reads the machine's CPU brand string, which names the same chip
/// ("Apple M5 Max" → "m5"). A machine that answers neither reports
/// "unknown" rather than a guess.
func probeDevice() -> String {
    #if canImport(Darwin)
    var size = 0
    guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 else {
        return "unknown"
    }
    var buffer = [CChar](repeating: 0, count: size)
    guard sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0) == 0 else {
        return "unknown"
    }
    let brand = String(cString: buffer).lowercased()
    for token in brand.split(separator: " ") where token.count >= 2 && token.hasPrefix("m") {
        if token.dropFirst().allSatisfy(\.isNumber) { return String(token) }
    }
    return brand.isEmpty ? "unknown" : brand
    #else
    return "unknown"
    #endif
}

struct Options {
    var weights: URL?
    var runnerID: String?
    var drafter: URL?
    var trusted = false
    var kvBytesCapacity = 8 << 30
    /// Raw `--speculative-protocol` value, nil when the flag was absent.
    var speculativeProtocol: String?
    /// Raw `<name>=<path>` values, in argv order. Validated once, together,
    /// AFTER parsing so a duplicate name is reported against the whole
    /// command rather than against whichever occurrence came second.
    var resources: [String] = []
}

func parseOptions(_ arguments: [String]) throws -> Options {
    var options = Options()
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
        case "runtime-worker": continue
        case "--weights": options.weights = URL(fileURLWithPath: try next("--weights"))
        case "--runner": options.runnerID = try next("--runner")
        case "--drafter": options.drafter = URL(fileURLWithPath: try next("--drafter"))
        case "--trusted": options.trusted = true
        case "--resource": options.resources.append(try next("--resource"))
        case "--speculative-protocol":
            options.speculativeProtocol = try next("--speculative-protocol")
        case "--kv-bytes":
            guard let bytes = Int(try next("--kv-bytes")) else {
                throw WorkerLaunchError.badValue("--kv-bytes")
            }
            options.kvBytesCapacity = bytes
        default: throw WorkerLaunchError.unknownArgument(argument)
        }
    }
    return options
}

enum WorkerLaunchError: Error, CustomStringConvertible {
    case missingValue(String)
    case badValue(String)
    case unknownArgument(String)
    case missingWeights
    case unknownRunner(String)

    var description: String {
        switch self {
        case .missingValue(let name): return "\(name) requires a value"
        case .badValue(let name): return "\(name) has an unusable value"
        case .unknownArgument(let name): return "unknown argument \(name)"
        case .missingWeights: return "--weights <dir> is required"
        case .unknownRunner(let id): return "no runner with id \(id)"
        }
    }
}

let options = try parseOptions(Array(CommandLine.arguments.dropFirst()))
guard let weights = options.weights else { throw WorkerLaunchError.missingWeights }

let runnerType: any Runner.Type
if let runnerID = options.runnerID {
    guard
        let match = RunnerRegistry.shared.manifests().first(where: { $0.runnerID == runnerID }),
        let modelType = match.modelTypes.first
    else {
        throw WorkerLaunchError.unknownRunner(runnerID)
    }
    runnerType = try RunnerRegistry.shared.resolve(modelType: modelType)
} else {
    runnerType = try RunnerRegistry.shared.resolve(checkpoint: weights)
}

// Both of these are validated BEFORE the load: a refusal an operator could
// have been told at argv must not cost a multi-minute weight load first.
let speculative = try SpeculativeProtocol.isEnabled(options.speculativeProtocol)

// Resources are validated BEFORE the load: a bad path discovered after a
// multi-minute weight load has wasted the box's time to report something
// argv already knew.
let resources = try RunnerResourceArguments.parse(options.resources)

// ONE load, before the hello.
let runner = try await runnerType.load(
    weights,
    options: RunnerLoadOptions(
        drafterDirectory: options.drafter,
        kvBytesCapacity: options.kvBytesCapacity,
        resources: resources))

let server = BenchWorkerServer(
    runner: runner,
    transport: StdioTransport(),
    trusted: options.trusted,
    speculative: speculative,
    // Stamped by the BenchRevisionStamp prebuild plugin: the revision that
    // PRODUCED this binary, not whatever the checkout moved to afterwards.
    build: BenchBuildRevision.value,
    device: probeDevice(),
    kvBytesCapacity: options.kvBytesCapacity)
await server.run()
