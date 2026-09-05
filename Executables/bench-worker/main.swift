// Copyright © 2026 Eigen Labs.
//
// bench-worker — Engine Protocol v1 server over `Runner`.
//
// benchd spawns `<engine> runtime-worker --weights <DIR> --speculative-protocol v1.1`.
// One binary serves every runner: the checkpoint's `config.json` `model_type`
// picks it out of `RunnerRegistry`, or `--runner <id>` names it.
//
// A SHIM. Argument parsing and validation live in
// `MLXRunners/BenchWorkerLaunch.swift` and the protocol loop in
// `MLXRunners/BenchWorker.swift`, so both are testable in process; a test
// target that depended on this executable would be run by SwiftPM as the
// swift-testing host and would abort the package's whole `@Test` pass (see
// Package.swift's BenchCBv2Core/BenchCBv2 note).
//
// The ORDER below is load bearing and reads top to bottom: validate argv,
// then resolve the runner (the first read under `--weights`), then load the
// weights ONCE, then serve. A checkpoint here can be 113 GB, so a bad flag
// must be answered from argv alone.

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

/// One line on stderr and a non-zero status — never a Swift runtime trap.
/// benchd reads this line to tell an operator what to fix, and a crash
/// report buries it under a backtrace.
func fail(_ error: some Error) -> Never {
    let message: String =
        (error as? CustomStringConvertible)?.description ?? "\(error)"
    FileHandle.standardError.write(Data(("bench-worker: " + message + "\n").utf8))
    exit(2)
}

// 1. ARGV. Every argument refusal happens here, with nothing under
//    `--weights` read yet.
let launch: BenchWorkerLaunchOptions
do {
    launch = try BenchWorkerLaunchOptions.parse(Array(CommandLine.arguments.dropFirst()))
} catch {
    fail(error)
}

// 2. The checkpoint. This is the first read under `--weights`.
let runnerType: any Runner.Type
do {
    runnerType = try launch.resolveRunner()
} catch {
    fail(error)
}

// 3. ONE load, before the hello.
let runner: any Runner
do {
    runner = try await runnerType.load(
        launch.weights,
        options: RunnerLoadOptions(
            drafterDirectory: launch.drafter,
            kvBytesCapacity: launch.kvBytesCapacity,
            resources: launch.resources))
} catch {
    fail(error)
}

// 4. Serve.
let server = BenchWorkerServer(
    runner: runner,
    transport: StdioTransport(),
    trusted: launch.trusted,
    speculative: launch.speculative,
    // Stamped by the BenchRevisionStamp prebuild plugin: the revision that
    // PRODUCED this binary, not whatever the checkout moved to afterwards.
    build: BenchBuildRevision.value,
    device: probeDevice(),
    kvBytesCapacity: launch.kvBytesCapacity)
await server.run()
