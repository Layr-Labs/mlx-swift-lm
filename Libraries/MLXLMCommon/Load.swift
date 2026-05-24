// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

#if canImport(Darwin)
import Darwin

/// Tell the kernel to start prefetching every shard into the unified buffer
/// cache. Non-blocking; the actual reads happen in the SSD controller's own
/// queue. Safe to call even when files are already cached (it's just a hint).
private func prefetchShards(_ urls: [URL]) {
    DispatchQueue.concurrentPerform(iterations: urls.count) { idx in
        let path = urls[idx].path
        let fd = open(path, O_RDONLY)
        guard fd >= 0 else { return }
        defer { close(fd) }

        var sb = stat()
        guard fstat(fd, &sb) == 0 else { return }

        var ra = radvisory(ra_offset: 0, ra_count: Int32(min(Int(sb.st_size), Int(Int32.max))))
        _ = fcntl(fd, F_RDADVISE, &ra)
    }
}
#else
private func prefetchShards(_ urls: [URL]) { }
#endif

/// Lock-protected scratch space used by the parallel shard reader. Wrapped in
/// a final class so Swift 6 strict concurrency can see we mean to share it
/// across the concurrent closures.
private final class ParallelShardState: @unchecked Sendable {
    typealias ShardResult = (weights: [String: MLXArray], metadata: [String: String])
    let lock = NSLock()
    var results: [ShardResult?]
    var firstError: Error?

    init(shardCount: Int) {
        self.results = Array(repeating: nil, count: shardCount)
    }

    func store(index: Int, result: ShardResult) {
        lock.lock()
        defer { lock.unlock() }
        results[index] = result
    }

    func recordError(_ error: Error) {
        lock.lock()
        defer { lock.unlock() }
        if firstError == nil { firstError = error }
    }
}

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    let bench = ProcessInfo.processInfo.environment["BENCH_VERBOSE"] != nil
    var t = CFAbsoluteTimeGetCurrent()
    func mark(_ label: String) {
        if bench {
            let now = CFAbsoluteTimeGetCurrent()
            FileHandle.standardError.write(
                Data("    [stage] \(label): \(String(format: "%.1f", (now - t) * 1000)) ms\n"
                    .utf8))
            t = now
        }
    }

    // Gather the shard URLs first so we can fan them out concurrently.
    var shardURLs: [URL] = []
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator where url.pathExtension == "safetensors" {
        shardURLs.append(url)
    }
    shardURLs.sort { $0.lastPathComponent < $1.lastPathComponent }

    // Hand the kernel a head start on every shard. F_RDADVISE is Darwin's
    // async-prefetch primitive — it issues a non-blocking advisory read into
    // the unified buffer cache, letting the SSD start streaming pages before
    // we ask for them. Net cost is one open/fcntl/close per shard. By the
    // time the DispatchQueue.concurrentPerform tasks below try to read, the
    // pages may already be resident.
    prefetchShards(shardURLs)
    mark("rdadvise")

    // Load shards in parallel. Each task forces eval() on its arrays so MLX
    // actually performs the disk read inside the task rather than deferring all
    // of it to a single later eval(model) call. This lets concurrent shards
    // overlap their I/O.
    typealias ShardResult = (weights: [String: MLXArray], metadata: [String: String])
    let shared = ParallelShardState(shardCount: shardURLs.count)
    let urls = shardURLs  // immutable Sendable snapshot for the closure

    DispatchQueue.concurrentPerform(iterations: urls.count) { idx in
        do {
            let (w, m) = try loadArraysAndMetadata(url: urls[idx])
            if !w.isEmpty {
                eval(Array(w.values))
            }
            shared.store(index: idx, result: (w, m))
        } catch {
            shared.recordError(error)
        }
    }
    if let err = shared.firstError { throw err }

    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    for (i, slot) in shared.results.enumerated() {
        guard let (w, m) = slot else { continue }
        for (key, value) in w { weights[key] = value }
        // Match the original "first iterated shard's metadata" semantics by
        // taking shard 0's, falling back to next non-empty for safety.
        if i == 0 || metadata.isEmpty { metadata = m }
    }
    mark("read shards (parallel)")

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)
    mark("sanitize")

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }
    mark("quantize wire")

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])
    mark("update params")

    eval(model)
    mark("eval")
}
