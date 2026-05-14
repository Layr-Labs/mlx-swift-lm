// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

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

    // Load shards in parallel. Each task forces eval() on its arrays so MLX
    // actually performs the disk read inside the task rather than deferring all
    // of it to a single later eval(model) call. This lets concurrent shards
    // overlap their I/O.
    typealias ShardResult = (weights: [String: MLXArray], metadata: [String: String])
    let resultsLock = NSLock()
    var shardResults: [ShardResult?] = Array(repeating: nil, count: shardURLs.count)
    var shardError: Error?

    DispatchQueue.concurrentPerform(iterations: shardURLs.count) { idx in
        do {
            let (w, m) = try loadArraysAndMetadata(url: shardURLs[idx])
            // Force materialization of this shard now so the disk read happens
            // concurrently with other shards' reads, instead of all being
            // deferred to a single later eval(model) call.
            if !w.isEmpty {
                eval(Array(w.values))
            }
            resultsLock.lock()
            shardResults[idx] = (w, m)
            resultsLock.unlock()
        } catch {
            resultsLock.lock()
            if shardError == nil { shardError = error }
            resultsLock.unlock()
        }
    }
    if let shardError { throw shardError }

    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    for case let (w, m)? in shardResults {
        for (key, value) in w { weights[key] = value }
        if metadata.isEmpty { metadata = m }
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
