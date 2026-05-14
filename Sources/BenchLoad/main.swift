// BenchLoad — measures model load time from a local directory.
// Reports per-run elapsed and aggregate stats (mean / median / stddev / range).

import BenchmarkHelpers
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM

@main
struct BenchLoad {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            print("usage: BenchLoad <model-directory> [runs=5] [warmup=1]")
            exit(64)
        }
        let directory = URL(fileURLWithPath: args[1])
        let runs = args.count > 2 ? Int(args[2]) ?? 5 : 5
        let warmup = args.count > 3 ? Int(args[3]) ?? 1 : 1

        guard FileManager.default.fileExists(atPath: directory.path) else {
            print("error: directory does not exist: \(directory.path)")
            exit(66)
        }

        // Report what's on disk.
        let weightFiles = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        let shards = weightFiles.filter { $0.pathExtension == "safetensors" }
        let totalBytes = shards.reduce(Int64(0)) { acc, url in
            // attributesOfItem follows symlinks (resourceValues does not).
            let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
            let size = (attrs?[.size] as? NSNumber)?.int64Value ?? 0
            return acc + size
        }
        let totalGiB = Double(totalBytes) / (1024 * 1024 * 1024)
        print("Model directory: \(directory.lastPathComponent)")
        print("  Shards: \(shards.count)")
        print("  Total weight bytes: \(String(format: "%.2f", totalGiB)) GiB")
        print("  Warmup runs: \(warmup)   Timed runs: \(runs)")
        print("")

        let tokenizerLoader = NoOpTokenizerLoader()

        do {
            // Warmup runs (page cache warming, JIT-ish overhead, etc.)
            for i in 1...max(warmup, 0) {
                let start = CFAbsoluteTimeGetCurrent()
                _ = try await loadModelContainer(from: directory, using: tokenizerLoader)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                print("  [warmup \(i)] \(String(format: "%.1f", elapsed)) ms")
                Memory.clearCache()
            }

            // Timed runs
            var times: [Double] = []
            for i in 1...runs {
                let start = CFAbsoluteTimeGetCurrent()
                _ = try await loadModelContainer(from: directory, using: tokenizerLoader)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                times.append(elapsed)
                print("  [run \(i)] \(String(format: "%.1f", elapsed)) ms")
                Memory.clearCache()
            }

            print("")
            let stats = BenchmarkStats(times: times)
            stats.printSummary(label: "Load \(directory.lastPathComponent)")
            print("")
            print("RAW_MS:\(times.map { String(format: "%.1f", $0) }.joined(separator: ","))")
        } catch {
            print("error: \(error)")
            exit(70)
        }
    }
}
