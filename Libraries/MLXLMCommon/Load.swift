// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - Concurrent weight loading

// MLX's safetensors loader is lazy: `loadArraysAndMetadata` reads only the header, and a
// tensor's bytes are read when its array is evaluated. A single `eval` of everything
// serializes that I/O and the copies into unified memory no matter how fast the disk is.
// Splitting each file into contiguous byte-balanced ranges and evaluating the ranges from
// concurrent `eval` calls overlaps read, copy, and allocation. Measured on an M4 Pro
// (14 cores, 5 GB shards, NVMe at ~6.1 GB/s sequential): the serial loader moves ~3-4.5 GB/s
// while the concurrent one reaches the disk ceiling cold (~5.9 GB/s) and >10 GB/s from the
// page cache -- a 30-45% faster cold load, about 2x warm. `F_RDADVISE`/read-ahead variants
// measured *slower* than the serial baseline because the advised I/O competes with the
// loader's own reads.
//
// The fork keeps an `F_RDADVISE` hint for every selected file before this loader starts
// (`prefetchShards`). With the earlier per-file loader it measured 13% faster on GPT-OSS-20B
// with a warm cache (309 ms to 269 ms, 5 runs). A model that keeps excluded tensors cold on
// external storage turns the hint off through
// ``CheckpointWeightLoadFiltering/skipWholeShardPrefetch``.

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
private func prefetchShards(_ urls: [URL]) {}
#endif

/// One tensor's byte range in a safetensors file, from the file's own header.
struct SafetensorSpan {
    let name: String
    let byteCount: Int64
}

/// The tensors of the safetensors file at `url`, ordered by their position in the file.
///
/// Reads the 8-byte header length and the JSON header only. Throws when the file is not a
/// well-formed safetensors file; callers fall back to loading the file whole.
func safetensorSpansInFileOrder(url: URL) throws -> [SafetensorSpan] {
    struct Malformed: Error {}

    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }

    guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
        throw Malformed()
    }
    let headerLength = lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        .littleEndian
    // a header bigger than this is not a header
    guard headerLength > 0, headerLength <= 512 * 1024 * 1024 else { throw Malformed() }
    guard let headerData = try handle.read(upToCount: Int(headerLength)),
        headerData.count == headerLength,
        let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
    else {
        throw Malformed()
    }

    var spans = [(name: String, begin: Int64, byteCount: Int64)]()
    for (name, value) in header {
        guard name != "__metadata__" else { continue }
        guard let entry = value as? [String: Any],
            let offsets = entry["data_offsets"] as? [Any], offsets.count == 2,
            let begin = (offsets[0] as? NSNumber)?.int64Value,
            let end = (offsets[1] as? NSNumber)?.int64Value,
            end >= begin
        else {
            throw Malformed()
        }
        spans.append((name, begin, end - begin))
    }
    spans.sort { $0.begin < $1.begin }
    return spans.map { SafetensorSpan(name: $0.name, byteCount: $0.byteCount) }
}

/// Contiguous index ranges of `byteCounts` whose byte totals are balanced around
/// `total / groupCount`, preserving order.
func contiguousLoadGroups(byteCounts: [Int64], groupCount: Int) -> [Range<Int>] {
    guard !byteCounts.isEmpty else { return [] }
    let total = byteCounts.reduce(0, +)
    guard groupCount > 1, total > 0 else { return [0 ..< byteCounts.count] }

    let groups = Int64(groupCount)
    var ranges = [Range<Int>]()
    var start = 0
    var cumulative: Int64 = 0
    var boundary: Int64 = 1
    for (index, byteCount) in byteCounts.enumerated() {
        cumulative += byteCount
        if boundary < groups, cumulative >= total * boundary / groups {
            ranges.append(start ..< index + 1)
            start = index + 1
            boundary += 1
        }
    }
    if start < byteCounts.count {
        ranges.append(start ..< byteCounts.count)
    }
    return ranges
}

/// How many concurrent evaluations to spread a model's weight loading across.
///
/// Throughput rises with concurrent readers until the disk (cold) or the memory system (warm)
/// saturates -- around 8-16 in-flight readers on Apple silicon. More workers than cores only
/// adds contention.
func weightLoadConcurrency(processorCount: Int = ProcessInfo.processInfo.activeProcessorCount)
    -> Int
{
    max(4, min(16, processorCount))
}

/// Below this size a file is loaded whole: splitting cannot beat a single sequential read.
private let minimumBytesPerLoadGroup: Int64 = 256 * 1024 * 1024

/// Lock-guarded shared state for the concurrent load.
private final class ConcurrentLoadState: @unchecked Sendable {
    private let lock = NSLock()
    private var perFile: [[String: MLXArray]]
    private var perFileMetadata: [[String: String]]
    private var firstError: Error?

    init(fileCount: Int) {
        perFile = Array(repeating: [:], count: fileCount)
        perFileMetadata = Array(repeating: [:], count: fileCount)
    }

    func merge(file: Int, weights: [String: MLXArray], metadata: [String: String]) {
        lock.lock()
        defer { lock.unlock() }
        perFile[file].merge(weights) { _, new in new }
        perFileMetadata[file] = metadata
    }

    func record(error: Error) {
        lock.lock()
        defer { lock.unlock() }
        if firstError == nil { firstError = error }
    }

    /// Preserve file order and tensor metadata, including later-file-wins duplicates.
    func result() throws -> ModelCheckpoint {
        lock.lock()
        defer { lock.unlock() }
        if let firstError { throw firstError }
        var weights = [String: MLXArray]()
        var weightMetadata = [String: [String: String]]()
        for (index, fileWeights) in perFile.enumerated() {
            weights.merge(fileWeights) { _, new in new }
            for name in fileWeights.keys {
                weightMetadata[name] = perFileMetadata[index]
            }
        }
        let metadata = perFileMetadata.first { !$0.isEmpty } ?? [:]
        return ModelCheckpoint(weights: weights, metadata: metadata, weightMetadata: weightMetadata)
    }
}

/// Load and materialize the weights of every file in `urls`, evaluating contiguous byte
/// ranges of each file concurrently.
///
/// Each work item lazily opens its file, evaluates only its assigned tensors (forcing that
/// range's I/O inside the work item), and the results are merged in file order. A file whose
/// header cannot be parsed is loaded whole by one work item, which is exactly the serial
/// loader's behavior for that file.
func loadWeightArrays(urls: [URL]) throws -> (
    weights: [String: MLXArray], metadata: [String: String]
) {
    let checkpoint = try loadModelCheckpoint(urls: urls)
    return (checkpoint.weights, checkpoint.metadata)
}

/// Tensors under `excludedNamespaces` are never read. A file that holds only excluded tensors
/// still supplies its metadata.
func loadModelCheckpoint(urls: [URL], excludedNamespaces: [String] = []) throws
    -> ModelCheckpoint
{
    let exclusions = CheckpointNameMapping(excludedNamespaces.map { .excludePrefix($0) })
    return try loadModelCheckpoint(urls: urls) { exclusions.mapName($0) != nil }
}

/// Only tensors whose names pass `isRead` are read.
func loadModelCheckpoint(urls: [URL], isRead: @escaping @Sendable (String) -> Bool) throws
    -> ModelCheckpoint
{
    struct WorkItem {
        let file: Int
        let url: URL
        /// tensors this item evaluates; nil evaluates every tensor in the file that is read
        let names: [String]?
    }

    let items: [WorkItem] = {
        var spansPerFile = [[SafetensorSpan]?]()
        var totalBytes: Int64 = 0
        for url in urls {
            let spans = (try? safetensorSpansInFileOrder(url: url))?.filter { isRead($0.name) }
            spansPerFile.append(spans)
            totalBytes += spans?.reduce(0) { $0 + $1.byteCount } ?? 0
        }

        let concurrency = weightLoadConcurrency()
        let groupBytes = max(minimumBytesPerLoadGroup, totalBytes / Int64(concurrency))
        var items = [WorkItem]()
        for (file, url) in urls.enumerated() {
            if let spans = spansPerFile[file], !spans.isEmpty {
                let bytes = spans.reduce(0) { $0 + $1.byteCount }
                let groupCount = max(1, Int(bytes / groupBytes))
                for range in contiguousLoadGroups(
                    byteCounts: spans.map(\.byteCount), groupCount: groupCount)
                {
                    items.append(
                        WorkItem(file: file, url: url, names: spans[range].map(\.name)))
                }
            } else {
                // An empty list reads only the metadata of a file with nothing to read.
                let names: [String]? = spansPerFile[file] == nil ? nil : []
                items.append(WorkItem(file: file, url: url, names: names))
            }
        }
        return items
    }()

    let state = ConcurrentLoadState(fileCount: urls.count)
    DispatchQueue.concurrentPerform(iterations: items.count) { index in
        let item = items[index]
        do {
            // A caller's withError does not reach this worker thread.
            try withLoaderThreadError {
                // Explicitly the CPU stream: `Load` has no GPU implementation and the arrays
                // land in unified memory either way. The concurrency comes from evaluating
                // disjoint groups from many threads, not from the stream itself.
                let (all, metadata) = try loadArraysAndMetadata(url: item.url, stream: .cpu)

                var selected = [String: MLXArray]()
                if let names = item.names {
                    for name in names {
                        if let array = all[name] { selected[name] = array }
                    }
                } else {
                    selected = all.filter { isRead($0.key) }
                }

                // force this range's I/O here, on this stream, in file-offset order
                if !selected.isEmpty { eval(Array(selected.values)) }
                state.merge(file: item.file, weights: selected, metadata: metadata)
            }
        } catch {
            state.record(error: error)
        }
    }
    return try state.result()
}

/// A model-supplied checkpoint-key filter applied immediately after each
/// safetensors header is decoded and BEFORE any array from that shard is
/// evaluated/materialized.
///
/// This differs intentionally from `sanitize(weights:)`: sanitization may
/// rename, reshape, or fuse arrays that have already been loaded. A loading
/// filter is only for checkpoint tensors the model proves it will never
/// materialize (for example Qwen4's SSD-backed learned PLE table).
public typealias CheckpointWeightLoadFilter = @Sendable (String) -> Bool

/// Implemented by models that must exclude some checkpoint tensors before
/// bulk materialization. The returned closure must capture only immutable,
/// Sendable configuration; shard reads invoke it concurrently.
public protocol CheckpointWeightLoadFiltering {
    /// Return true for a checkpoint key that should enter the ordinary
    /// in-memory load/update path, false for one owned by an external exact
    /// row source or otherwise absent from this model topology.
    var checkpointWeightLoadFilter: CheckpointWeightLoadFilter { get }

    /// Whole-file read-ahead must be skipped when an excluded tensor is
    /// intentionally kept cold on external storage. Filters used only to
    /// remove topology-absent tensors may leave read-ahead enabled.
    var skipWholeShardPrefetch: Bool { get }
}

extension CheckpointWeightLoadFiltering {
    public var skipWholeShardPrefetch: Bool { true }
}

/// Implemented by models whose ``BaseLanguageModel/sanitize(weights:)``
/// renames or fuses modules relative to the checkpoint layout.
///
/// Per-layer quantization tables in `config.json` are keyed on the
/// checkpoint's module paths. When a sanitizer renames a module (e.g. fusing
/// split `gate_proj`/`up_proj` experts into one `gate_up_proj`), the new path
/// no longer matches its explicit overrides, and
/// ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:weightFileSelection:excludedComponents:componentLoading:)-858j8``
/// would fall back to the default quantization — or to none at all for
/// mixed-precision configs without a default — and the subsequent strict
/// update would reject the renamed `.scales`/`.biases` tensors. Aliases
/// restore the lookup: the first alias with an explicit per-layer entry wins.
public protocol QuantizationPathAliasing {
    /// Checkpoint-layout config paths to consult, in order, when `path`
    /// itself has no explicit per-layer quantization entry.
    func quantizationPathAliases(for path: String) -> [String]
}

/// Resolve the quantization for one module path during weight loading,
/// consulting `aliasing` when the per-layer table has no explicit entry for
/// a post-sanitize (renamed/fused) module path.
public func resolveQuantization(
    path: String,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization,
    aliasing: QuantizationPathAliasing?
) -> BaseConfiguration.Quantization? {
    if perLayerQuantization.perLayerQuantization[path] == nil, let aliasing {
        for alias in aliasing.quantizationPathAliases(for: path)
        where perLayerQuantization.perLayerQuantization[alias] != nil {
            return perLayerQuantization.quantization(layer: alias)
        }
    }
    return perLayerQuantization.quantization(layer: path)
}

/// Implemented by models whose ``BaseLanguageModel/sanitize(weights:)``
/// makes module-topology decisions that depend on the checkpoint's
/// quantization policy.
///
/// Fusing two checkpoint modules into one (e.g. split `gate_proj`/`up_proj`
/// experts into a fused `gate_up_proj`) is only representable when both
/// halves share one quantization policy — a single quantized projection has
/// one bits/group_size/mode. Sanitizers consult the staged policy to keep
/// such pairs split (and reshape the module tree accordingly) whenever the
/// halves' effective policies differ.
///
/// ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:weightFileSelection:excludedComponents:componentLoading:)-858j8``
/// stages the checkpoint's resolved policy here before calling `sanitize`;
/// a uniform `quantization` is staged as a table with only a default.
public protocol QuantizationPolicyReceiving: AnyObject {
    /// Quantization policy of the checkpoint currently being loaded, or
    /// `nil` for unquantized checkpoints.
    var checkpointPerLayerQuantization: BaseConfiguration.PerLayerQuantization? { get set }
}

/// Opt-in load-time materialization for a model whose sanitizer creates
/// large lazy packed-weight copies. Other models retain the existing load
/// sequence. The hook runs after strict update and removal of both loader
/// staging owners, before dtype conversion and the final model eval.
public protocol IncrementalCheckpointMaterializing: AnyObject {
    var needsIncrementalCheckpointMaterialization: Bool { get }
    func materializeCheckpointWeightsIncrementally() throws
}

private struct SafetensorsIndex: Decodable {
    let weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}

/// How the safetensors files holding a model's weights are chosen.
///
/// ## See Also
/// - ``ModelConfiguration/weightFileSelection``
public enum WeightFileSelection: Sendable, Equatable {
    /// Use `model.safetensors.index.json` when it names files that exist, otherwise the
    /// conventional `model*.safetensors` (then `weight*.safetensors`) names.
    ///
    /// This is what a well-packaged checkpoint wants and it is the default.
    case automatic

    /// Load every safetensors file in the model directory.
    ///
    /// This is an escape hatch for a checkpoint whose index is known to be wrong in a way
    /// ``automatic`` cannot detect -- an index that names files that all exist but omits
    /// weights the model needs. It loads files that may not belong to this model, and a
    /// stray tensor whose name collides with one the model's `sanitize(weights:)` rewrites
    /// is loaded silently rather than reported, so prefer a model that declares its own
    /// extra files (see ``AdditionalWeightFilesProviding``) where that is possible.
    case allFilesPresent
}

/// The safetensors files in `modelDirectory` that hold the model's weights.
///
/// Only the top level of the directory is considered. Checkpoints keep auxiliary weights that
/// belong to a different module in subdirectories (for example `mlx-community/Qwen3.5-4B-OptiQ-4bit`
/// and its `optiq/mtp.safetensors`), and a nested Hugging Face snapshot cache under a local
/// checkpoint directory would otherwise be pulled in as well.
///
/// With ``WeightFileSelection/automatic`` the files are chosen in this order:
///
/// 1. The files named by `model.safetensors.index.json`, when it exists and every file it names
///    exists. The index is precise about which of several weight files belong to the model, which
///    matters for a repo that ships both a consolidated file and shards.
/// 2. The conventional `model*.safetensors` names, matching `mlx_lm.utils.load_model`. Uploads
///    regularly ship an index carried over from an unquantized source repo that names shards the
///    repo does not contain, and the convention is what those repos actually follow.
/// 3. `weight*.safetensors`, then every safetensors file present, so a directory that follows no
///    convention at all still loads.
///
/// `additionalFiles` names files the model requires that no rule above selects, for example the
/// Jina reranker's `projector.safetensors`. They are appended, so a file the index already names
/// is not loaded twice, and names that are not present are ignored.
///
/// - Parameters:
///   - modelDirectory: directory holding the weight files
///   - selection: how to choose the files, see ``WeightFileSelection``
///   - additionalFiles: file names, relative to `modelDirectory`, to load in addition to the
///     selected ones. See ``AdditionalWeightFilesProviding/additionalWeightFiles``.
package func safetensorWeightURLs(
    in modelDirectory: URL,
    selection: WeightFileSelection = .automatic,
    additionalFiles: [String] = []
) throws -> [URL] {
    let present = topLevelSafetensorURLs(in: modelDirectory)

    let selected: [URL]
    switch selection {
    case .allFilesPresent:
        selected = present
    case .automatic:
        selected = try indexedWeightURLs(in: modelDirectory) ?? conventionalWeightURLs(in: present)
    }

    var seen = Set(selected.map(\.standardizedFileURL.path))
    var urls = selected
    for name in additionalFiles {
        let url = modelDirectory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path),
            seen.insert(url.standardizedFileURL.path).inserted
        else {
            continue
        }
        urls.append(url)
    }
    return urls
}

/// The files named by `model.safetensors.index.json`, or `nil` when there is no index or it
/// names a file the directory does not contain.
///
/// Existence is checked against the file system rather than the top-level listing: an index may
/// legitimately map weights into a subdirectory, and that is a deliberate statement about where
/// this model's weights live rather than an unrelated file that happens to be nearby.
private func indexedWeightURLs(in modelDirectory: URL) throws -> [URL]? {
    let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
    guard FileManager.default.fileExists(atPath: indexURL.path) else {
        return nil
    }

    let data = try Data(contentsOf: indexURL)
    let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
    let urls = Set(index.weightMap.values)
        .sorted()
        .map { modelDirectory.appendingPathComponent($0) }

    guard !urls.isEmpty,
        urls.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) })
    else {
        return nil
    }
    return urls
}

/// The conventionally named weight files among `present`, matching `mlx_lm.utils.load_model`'s
/// `model*.safetensors` glob, with `weight*.safetensors` and then everything as fallbacks.
private func conventionalWeightURLs(in present: [URL]) -> [URL] {
    for prefix in ["model", "weight"] {
        let matches = present.filter { $0.lastPathComponent.hasPrefix(prefix) }
        if !matches.isEmpty {
            return matches
        }
    }
    return present
}

private func topLevelSafetensorURLs(in modelDirectory: URL) -> [URL] {
    let contents =
        (try? FileManager.default.contentsOfDirectory(
            at: modelDirectory, includingPropertiesForKeys: nil)) ?? []
    return
        contents
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
}

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads model weight `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/prepareCheckpoint(_:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights. Derived inference-only state is prepared after the
/// checkpoint update and before the model is evaluated and returned to callers.
///
/// The weight files are chosen from `model.safetensors.index.json` when it names files that
/// exist, and otherwise by the conventional `model*.safetensors` names. A model can name extra
/// files it needs by conforming to ``AdditionalWeightFilesProviding``, and a caller can override
/// the choice with ``ModelConfiguration/weightFileSelection``.
///
/// The model loads without the `excludedComponents` it supports, see
/// ``ModelComponentsProviding``. Its on-demand components load the first time an input
/// needs them, unless `componentLoading` is ``ComponentLoading/immediate``. Tensors of components
/// that do not load now and of the model's ``BaseLanguageModel/excludedCheckpointNamespaces`` are
/// not read.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    weightFileSelection: WeightFileSelection = .automatic,
    excludedComponents: Set<ModelComponent> = [],
    componentLoading: ComponentLoading = .onFirstUse
) throws {
    // Set BENCH_VERBOSE to write the time of each load stage to standard error.
    let bench = ProcessInfo.processInfo.environment["BENCH_VERBOSE"] != nil
    var t = CFAbsoluteTimeGetCurrent()
    func mark(_ label: String) {
        if bench {
            let now = CFAbsoluteTimeGetCurrent()
            FileHandle.standardError.write(
                Data(
                    "    [stage] \(label): \(String(format: "%.1f", (now - t) * 1000)) ms\n"
                        .utf8))
            t = now
        }
    }

    let additionalFiles = (model as? any AdditionalWeightFilesProviding)?.additionalWeightFiles
    let weightURLs = try safetensorWeightURLs(
        in: modelDirectory,
        selection: weightFileSelection,
        additionalFiles: additionalFiles ?? [])
    let perLayerQuantization =
        perLayerQuantization
        ?? quantization.map { .init(quantization: $0, perLayerQuantization: [:]) }
    let excludedNamespaces =
        try prepareComponents(
            of: model, urls: weightURLs, excluded: excludedComponents,
            loading: componentLoading, perLayerQuantization: perLayerQuantization)
        + model.excludedCheckpointNamespaces

    let filterOwner = model as? any CheckpointWeightLoadFiltering
    let checkpointFilter = filterOwner?.checkpointWeightLoadFilter

    // Hand the kernel a head start on every file. F_RDADVISE is Darwin's
    // async-prefetch primitive. It issues a non-blocking advisory read into
    // the unified buffer cache, so the SSD can stream pages before the
    // concurrent loader asks for them. The cost is one open/fcntl/close per file.
    //
    // Do NOT advise whole files for an externally-backed model. Qwen4's PLE
    // tensors share safetensor files with compute weights; whole-file advice
    // would explicitly pull the 30 GiB SSD table into the unified buffer
    // cache even though the key filter below never evaluates those arrays.
    if filterOwner?.skipWholeShardPrefetch != true {
        prefetchShards(weightURLs)
        mark("rdadvise")
    } else {
        mark("rdadvise skipped (checkpoint filter)")
    }

    // The model's checkpoint filter applies before any array is evaluated:
    // excluded tensors are never read.
    let exclusions = CheckpointNameMapping(excludedNamespaces.map { .excludePrefix($0) })
    var checkpoint = try loadModelCheckpoint(
        urls: weightURLs,
        isRead: { name in
            exclusions.mapName(name) != nil && checkpointFilter?(name) != false
        })
    mark("read shards (parallel)")

    let prism = try (model as? any PrismHadamardLoading).map {
        try PrismHadamardCheckpoint(
            directory: modelDirectory,
            configuration: $0.prismCheckpoint, weights: checkpoint.weights)
    }
    if prism != nil {
        checkpoint.weights = checkpoint.weights.filter { !$0.key.hasSuffix(".signs") }
    }

    // Stage the checkpoint's quantization policy for sanitizers whose
    // module-topology decisions depend on it (e.g. the Qwen3.5 routed-expert
    // gate/up fusion, which must keep heterogeneous pairs split).
    checkpoint.perLayerQuantization = perLayerQuantization
    if let policyReceiving = model as? QuantizationPolicyReceiving {
        policyReceiving.checkpointPerLayerQuantization = perLayerQuantization
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    checkpoint = try model.prepareCheckpoint(checkpoint)
    mark("sanitize")

    try applyCheckpoint(checkpoint, to: model) { weights in
        mark("quantize wire")
        try prism?.install(model: model, weights: weights)
    }
    mark("update params")

    // Drop the staging checkpoint before dtype conversion so we don't keep
    // two copies of safetensor arrays alive during the bf16 pass.
    checkpoint.weights.removeAll(keepingCapacity: false)
    if let materializing = model as? IncrementalCheckpointMaterializing,
        materializing.needsIncrementalCheckpointMaterialization
    {
        try materializing.materializeCheckpointWeightsIncrementally()
        mark("incremental checkpoint materialization")
    }
    MLX.Memory.clearCache()

    // Convert fp16 parameters to bf16 to eliminate AsType cascades.
    // Metal's kernel dispatcher promotes mixed fp16/fp32 operations to full fp32,
    // causing extra kernel dispatches across all layers. bf16 shares fp32's exponent
    // range, so the promotion is eliminated. Quantization scales/biases are also
    // converted — QuantizedMatmul uses scales dtype to determine output dtype.
    //
    // Controlled by DARKBLOOM_BF16_WEIGHTS (default: ON, set to "0" to disable).
    // The conversion runs before the derived inference state is prepared, so
    // that state is built from the converted weights.
    let bf16Env = ProcessInfo.processInfo.environment["DARKBLOOM_BF16_WEIGHTS"] ?? "1"
    if bf16Env == "1" && prism == nil {
        convertToBFloat16(model: model)
        mark("bf16 convert")
    }

    // Build derived inference-only state and realize the model while the loader
    // still has exclusive access. Forward passes must remain read-only.
    materializeModelForInference(model)
    mark("eval")
}

/// Quantize the layers `checkpoint` has scales for, then update `module` with strict validation.
///
/// A module that conforms to ``QuantizationPathAliasing`` resolves renamed or fused layers
/// through its aliases. `beforeUpdate` runs after quantization and before the update, with
/// the checkpoint tensors.
func applyCheckpoint(
    _ checkpoint: ModelCheckpoint, to module: Module,
    beforeUpdate: ([String: MLXArray]) throws -> Void = { _ in }
) throws {
    let weights = checkpoint.weights
    if let perLayerQuantization = checkpoint.perLayerQuantization {
        let aliasing = module as? QuantizationPathAliasing
        quantize(model: module) { path, _ in
            guard weights["\(path).scales"] != nil else { return nil }
            return resolveQuantization(
                path: path, perLayerQuantization: perLayerQuantization,
                aliasing: aliasing)?.asTuple
        }
    }
    try beforeUpdate(weights)
    try module.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
}

/// Async variant of
/// ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:weightFileSelection:excludedComponents:componentLoading:)-858j8``.
///
/// Loading blocks its thread on file I/O and fans out with `DispatchQueue.concurrentPerform`.
/// Swift concurrency's cooperative threads must never block, so this overload runs the load
/// on a global queue and suspends the caller instead. Async callers resolve to this overload;
/// the synchronous one remains for synchronous code such as model conversion.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    weightFileSelection: WeightFileSelection = .automatic,
    excludedComponents: Set<ModelComponent> = [],
    componentLoading: ComponentLoading = .onFirstUse
) async throws {
    let model = SendableBox(model)
    try await withCheckedThrowingContinuation {
        (continuation: CheckedContinuation<Void, any Error>) in
        DispatchQueue.global(qos: .userInitiated).async {
            // The caller's withError does not reach this thread.
            continuation.resume(
                with: Result {
                    try withLoaderThreadError {
                        try loadWeights(
                            modelDirectory: modelDirectory, model: model.consume(),
                            quantization: quantization,
                            perLayerQuantization: perLayerQuantization,
                            weightFileSelection: weightFileSelection,
                            excludedComponents: excludedComponents,
                            componentLoading: componentLoading)
                    }
                })
        }
    }
}

/// Run `body` in an MLX error scope on the current thread. An MLX error is thrown
/// in place of a later error it caused in `body`, such as a shape mismatch.
func withLoaderThreadError<R>(_ body: () throws -> R) throws -> R {
    try withError { mlxError in
        do {
            return try body()
        } catch {
            try mlxError.check()
            throw error
        }
    }
}

// MARK: - BFloat16 Weight Conversion

/// Convert float16 model parameters to bfloat16 to prevent AsType cascades.
///
/// Metal's kernel dispatcher promotes mixed float16/float32 operations to full float32,
/// causing speed regression for models where routing/normalization runs at float32.
/// bfloat16 avoids this because it shares float32's exponent range.
///
/// Quantization scales/biases are also converted — QuantizedMatmul uses scales dtype to
/// determine output dtype, so float16 scales → float16 output → AsType when multiplied
/// with bfloat16 norms. Converting scales to bfloat16 eliminates this cascade.
///
/// The conversion is chunked to bound transient memory. Large quantized models carry
/// tens of GB of scale/bias metadata; converting all arrays at once keeps both fp16 and
/// bf16 copies alive until the final eval.
private func convertToBFloat16(model: Module) {
    let t0 = CFAbsoluteTimeGetCurrent()

    let convertibleParams: [(key: String, convertedBytes: Int)] = {
        let flat = model.parameters().flattened()
        return flat.compactMap { key, array in
            guard array.dtype == .float16 else { return nil }
            return (key: key, convertedBytes: estimatedByteCount(array, as: .bfloat16))
        }
    }()

    guard !convertibleParams.isEmpty else { return }

    let totalBytes = convertibleParams.reduce(0) { $0 + $1.convertedBytes }
    let chunkLimit = bfloat16ConversionChunkLimit()
    var index = 0
    var convertedCount = 0

    while index < convertibleParams.count {
        var converted = [String: MLXArray]()
        var chunkBytes = 0

        do {
            let current = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            while index < convertibleParams.count {
                let entry = convertibleParams[index]
                if !converted.isEmpty,
                    chunkBytes + entry.convertedBytes > chunkLimit
                {
                    break
                }

                guard let array = current[entry.key],
                    array.dtype == DType.float16
                else {
                    index += 1
                    continue
                }

                converted[entry.key] = array.asType(DType.bfloat16)
                chunkBytes += entry.convertedBytes
                index += 1
            }
        }

        guard !converted.isEmpty else { continue }

        let values = Array(converted.values)
        MLX.eval(values)
        convertedCount += converted.count

        let params = ModuleParameters.unflattened(converted)
        do {
            try model.update(parameters: params, verify: [])
        } catch {
            FileHandle.standardError.write(
                Data("[bf16] model.update failed: \(error)\n".utf8))
        }
        MLX.Memory.clearCache()
    }

    let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
    let mb = Double(totalBytes) / (1024 * 1024)
    FileHandle.standardError.write(
        Data(
            "[bf16] converted \(convertedCount) params (\(String(format: "%.1f", mb)) MB) fp16→bf16 in \(String(format: "%.0f", elapsed)) ms\n"
                .utf8))
}

/// Maximum bytes of bf16 output to accumulate per conversion chunk.
/// Override with DARKBLOOM_BF16_CHUNK_MB environment variable.
private func bfloat16ConversionChunkLimit() -> Int {
    if let raw = ProcessInfo.processInfo.environment["DARKBLOOM_BF16_CHUNK_MB"],
        let mb = Int(raw), mb > 0
    {
        return mb * 1024 * 1024
    }
    return 256 * 1024 * 1024
}

private func estimatedByteCount(_ array: MLXArray, as dtype: DType) -> Int {
    let elements = array.shape.reduce(1) { partial, dim in
        partial * max(dim, 1)
    }
    return elements * dtypeByteWidth(dtype)
}

private func dtypeByteWidth(_ dtype: DType) -> Int {
    if dtype == .bool || dtype == .int8 || dtype == .uint8 {
        return 1
    }
    if dtype == .float16 || dtype == .bfloat16
        || dtype == .int16 || dtype == .uint16
    {
        return 2
    }
    if dtype == .int64 || dtype == .uint64 {
        return 8
    }
    return 4  // float32, int32, uint32, complex64
}
