import Foundation
import MLX
import MLXDecisions

// Local checkpoint qualification entrypoint; diagnostics never enter the HTTP API.
@main struct LayaProbe {
    static func main() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard arguments.count >= 2 else {
            FileHandle.standardError.write(
                Data(
                    "Usage: laya-probe MODEL_DIRECTORY REQUEST.json [--diagnostics] [--repeat N]\n"
                        .utf8))
            Foundation.exit(2)
        }
        let request = try Data(contentsOf: URL(fileURLWithPath: arguments[1]))
        let start = ContinuousClock.now
        let runtime = try await LayaRuntime.load(directory: URL(fileURLWithPath: arguments[0]))
        let loadTime = start.duration(to: .now)
        if arguments.contains("--diagnostics") {
            let result = try await runtime.diagnostics(data: request)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            FileHandle.standardOutput.write(try encoder.encode(result))
        } else {
            var count = 1
            if let index = arguments.firstIndex(of: "--repeat"), index + 1 < arguments.count {
                guard let value = Int(arguments[index + 1]), (1 ... 10000).contains(value) else {
                    throw LayaError.invalidRequest("Repeat count must be 1...10000")
                }
                count = value
            }
            if count > 1 { for _ in 0 ..< 5 { _ = try await runtime.predict(data: request) } }
            var elapsed: [Double] = []
            var response = Data()
            for _ in 0 ..< count {
                let before = ContinuousClock.now
                response = try await runtime.predict(data: request)
                let delta = before.duration(to: .now).components
                elapsed.append(Double(delta.seconds) * 1000 + Double(delta.attoseconds) / 1e15)
            }
            elapsed.sort()
            FileHandle.standardOutput.write(response)
            let timing: [String: Any] = [
                "iterations": count, "warmup": count > 1 ? 5 : 0,
                "median_ms": elapsed[count / 2], "min_ms": elapsed.first!, "max_ms": elapsed.last!,
                "load_duration": String(describing: loadTime),
                "active_memory_bytes": Memory.activeMemory,
                "peak_memory_bytes": Memory.peakMemory,
            ]
            FileHandle.standardError.write(
                try JSONSerialization.data(withJSONObject: timing, options: [.sortedKeys]))
            FileHandle.standardError.write(Data("\n".utf8))
        }
        FileHandle.standardOutput.write(Data("\n".utf8))
    }
}
