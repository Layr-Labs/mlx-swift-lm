// BenchLoad — measures model load time from a local directory.
// Reports per-run elapsed and aggregate stats (mean / median / stddev / range).

import BenchmarkHelpers
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM

/// Loads the model, walks every weight tensor in deterministic order, and
/// prints a digest derived from each tensor's shape, dtype, and sum. Two
/// runs that produce the same digest have provably identical weights.
func runChecksum(directory: URL) async {
    let tokenizerLoader = NoOpTokenizerLoader()
    do {
        let container = try await loadModelContainer(from: directory, using: tokenizerLoader)
        await container.perform { context in
            let params = context.model.parameters().flattened()
            let sorted = params.sorted { $0.0 < $1.0 }

            // Deterministic combination of per-tensor (key, shape, dtype, sum).
            var combinedHash: UInt64 = 0xcbf29ce484222325  // FNV-1a basis
            var totalElements: Int64 = 0
            for (key, array) in sorted {
                let s = sum(array.asType(.float32)).item(Float.self)
                let shape = array.shape.map { String($0) }.joined(separator: "x")
                let line = "\(key)|\(shape)|\(array.dtype)|\(s)"
                for b in line.utf8 {
                    combinedHash = (combinedHash ^ UInt64(b)) &* 0x100000001b3
                }
                totalElements += Int64(array.size)
            }
            print("digest=\(String(format: "%016x", combinedHash))")
            print("tensors=\(sorted.count) elements=\(totalElements)")
        }
    } catch {
        print("error: \(error)")
        exit(70)
    }
}

/// Time loading an .mlxpack via the mmap+zero-copy path.
func runMLXPackBench(file: URL, runs: Int, warmup: Int) {
    let size = (try? FileManager.default.attributesOfItem(atPath: file.path)[.size] as? NSNumber)?
        .int64Value ?? 0
    let gb = Double(size) / 1024 / 1024 / 1024
    print("MLXPack: \(file.lastPathComponent)")
    print("  Size: \(String(format: "%.2f", gb)) GiB")
    print("  Warmup: \(warmup)   Runs: \(runs)")
    print("")

    do {
        if warmup > 0 {
            for i in 1...warmup {
                let start = CFAbsoluteTimeGetCurrent()
                _ = try MLXPackLoader.load(url: file)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                print("  [warmup \(i)] \(String(format: "%.1f", elapsed)) ms")
                Memory.clearCache()
            }
        }
        var times: [Double] = []
        for i in 1...runs {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try MLXPackLoader.load(url: file)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            times.append(elapsed)
            print("  [run \(i)] \(String(format: "%.1f", elapsed)) ms")
            Memory.clearCache()
        }
        print("")
        let stats = BenchmarkStats(times: times)
        stats.printSummary(label: "Load \(file.lastPathComponent)")
        print("RAW_MS:\(times.map { String(format: "%.1f", $0) }.joined(separator: ","))")
    } catch {
        print("error: \(error)")
        exit(70)
    }
}

/// Compute a digest of the .mlxpack's tensors using the same algorithm as
/// --checksum so we can compare with the safetensors loader.
func runMLXPackChecksum(file: URL) {
    do {
        let (weights, _) = try MLXPackLoader.load(url: file)
        let sorted = weights.sorted { $0.key < $1.key }
        var combinedHash: UInt64 = 0xcbf29ce484222325
        var totalElements: Int64 = 0
        for (key, array) in sorted {
            let s = sum(array.asType(.float32)).item(Float.self)
            let shape = array.shape.map { String($0) }.joined(separator: "x")
            let line = "\(key)|\(shape)|\(array.dtype)|\(s)"
            for b in line.utf8 {
                combinedHash = (combinedHash ^ UInt64(b)) &* 0x100000001b3
            }
            totalElements += Int64(array.size)
        }
        print("digest=\(String(format: "%016x", combinedHash))")
        print("tensors=\(sorted.count) elements=\(totalElements)")
    } catch {
        print("error: \(error)")
        exit(70)
    }
}

@main
struct BenchLoad {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            print("usage: BenchLoad <model-directory> [runs=5] [warmup=1]")
            print("       BenchLoad --checksum <model-directory>")
            exit(64)
        }

        // --checksum mode: load model, print a stable digest of weight tensor
        // statistics. Useful for proving the loader's output is bit-identical
        // between variants.
        if args[1] == "--checksum" {
            guard args.count >= 3 else {
                print("usage: BenchLoad --checksum <model-directory>")
                exit(64)
            }
            await runChecksum(directory: URL(fileURLWithPath: args[2]))
            return
        }

        // --convert mode: read all safetensors shards in <dir> and write a
        // single page-aligned .mlxpack file.
        if args[1] == "--convert" {
            guard args.count >= 4 else {
                print("usage: BenchLoad --convert <model-directory> <output.mlxpack>")
                exit(64)
            }
            let src = URL(fileURLWithPath: args[2])
            let dst = URL(fileURLWithPath: args[3])
            let t0 = CFAbsoluteTimeGetCurrent()
            do {
                try convertSafetensorsDirectoryToMLXPack(directory: src, outputFile: dst)
                let secs = CFAbsoluteTimeGetCurrent() - t0
                let size = (try? FileManager.default.attributesOfItem(atPath: dst.path)[.size] as? NSNumber)?
                    .int64Value ?? 0
                let gb = Double(size) / 1024 / 1024 / 1024
                print(String(
                    format: "converted in %.2f s → %@ (%.2f GiB)",
                    secs, dst.path, gb))
            } catch {
                print("convert error: \(error)")
                exit(70)
            }
            return
        }

        // --load-mlxpack mode: time loading from a converted .mlxpack file.
        if args[1] == "--load-mlxpack" {
            guard args.count >= 3 else {
                print("usage: BenchLoad --load-mlxpack <file.mlxpack> [runs=5] [warmup=1]")
                exit(64)
            }
            let pack = URL(fileURLWithPath: args[2])
            let runs = args.count > 3 ? Int(args[3]) ?? 5 : 5
            let warmup = args.count > 4 ? Int(args[4]) ?? 1 : 1
            runMLXPackBench(file: pack, runs: runs, warmup: warmup)
            return
        }

        // --checksum-mlxpack: load .mlxpack and compute the same digest as
        // --checksum so we can verify bit-identical loading.
        if args[1] == "--checksum-mlxpack" {
            guard args.count >= 3 else {
                print("usage: BenchLoad --checksum-mlxpack <file.mlxpack>")
                exit(64)
            }
            runMLXPackChecksum(file: URL(fileURLWithPath: args[2]))
            return
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
            if warmup > 0 {
                for i in 1...warmup {
                    let start = CFAbsoluteTimeGetCurrent()
                    _ = try await loadModelContainer(from: directory, using: tokenizerLoader)
                    let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                    print("  [warmup \(i)] \(String(format: "%.1f", elapsed)) ms")
                    Memory.clearCache()
                }
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
