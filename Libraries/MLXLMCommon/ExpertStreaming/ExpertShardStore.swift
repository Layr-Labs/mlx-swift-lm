// Reads individual routed-expert weight slices directly off disk from a
// sharded safetensors checkpoint, without ever materializing the full
// [numExperts, out, in] stacked tensor.
//
// Stacked expert tensors in the mlx-canonical MoE layout are EXPERT-MAJOR
// (`[numExperts, out, in]`), so a single expert's slice is one contiguous
// byte range within its parent tensor's file region: no gather/scatter, just
// `pread(fd, buf, stride, tensorStart + e * stride)`. This is what makes SSD
// streaming viable — a "gather this expert" operation is a single
// sequential read, not 2048 scattered reads of `in`-sized rows.

import Foundation
import MLX

/// One resident expert's worth of streamed switch_mlp tensors. Bias fields
/// are optional (nil for mxfp4 — DeepSeek-V4's routed experts have none —
/// but plumbed through generically so an affine-quantized MoE checkpoint
/// reusing this store, e.g. a future Qwen port, gets bias support for free).
///
/// `@unchecked Sendable`: `MLXArray` is reference-counted GPU-backed storage,
/// not itself `Sendable`, but these arrays are only ever read after being
/// fully constructed (never mutated in place) and are safe to hand across
/// the `DispatchQueue.concurrentPerform` / cache-lock boundaries this type
/// crosses — same rationale as `GenerationBatchResponse` in GenerationBatch.swift.
public struct ExpertWeights: @unchecked Sendable {
    public let gateWeight: MLXArray
    public let gateScales: MLXArray
    public let gateBiases: MLXArray?
    public let upWeight: MLXArray
    public let upScales: MLXArray
    public let upBiases: MLXArray?
    public let downWeight: MLXArray
    public let downScales: MLXArray
    public let downBiases: MLXArray?

    /// Total bytes across all resident arrays — used for cache-budget
    /// accounting. Computed once at fetch time (not derived lazily from
    /// `.nbytes` on every cache operation) so eviction bookkeeping is O(1).
    public let byteCount: Int

    public init(
        gateWeight: MLXArray, gateScales: MLXArray, gateBiases: MLXArray?,
        upWeight: MLXArray, upScales: MLXArray, upBiases: MLXArray?,
        downWeight: MLXArray, downScales: MLXArray, downBiases: MLXArray?,
        byteCount: Int
    ) {
        self.gateWeight = gateWeight
        self.gateScales = gateScales
        self.gateBiases = gateBiases
        self.upWeight = upWeight
        self.upScales = upScales
        self.upBiases = upBiases
        self.downWeight = downWeight
        self.downScales = downScales
        self.downBiases = downBiases
        self.byteCount = byteCount
    }
}

public enum ExpertShardStoreError: Error, Equatable {
    case missingTensor(String)
    case shortRead(String)
    case notExpertStacked(String)
}

/// Lock-protected scratch space for `ExpertShardStore.fetch`'s parallel
/// reads. Wrapped in a final class (mirroring `ParallelShardState` in
/// Load.swift) so Swift 6 strict concurrency can see the sharing is
/// intentional and lock-protected, not a data race.
private final class ExpertFetchState: @unchecked Sendable {
    let lock = NSLock()
    var results: [ExpertWeights?]
    var firstError: Error?

    init(count: Int) {
        self.results = Array(repeating: nil, count: count)
    }

    func store(index: Int, weights: ExpertWeights) {
        lock.lock()
        defer { lock.unlock() }
        results[index] = weights
    }

    func recordError(_ error: Error) {
        lock.lock()
        defer { lock.unlock() }
        if firstError == nil { firstError = error }
    }
}

/// Given a layout and a `switch_mlp` weight prefix, resolves per-expert byte
/// ranges and reads them with `pread`, fanning misses out across
/// `DispatchQueue.concurrentPerform` so a chunk of N experts issues N
/// concurrent reads instead of a serial loop (SSDs need queue depth to hit
/// their rated IOPS/throughput).
public final class ExpertShardStore: @unchecked Sendable {
    private let layout: SafetensorsLayout
    private let numExperts: Int

    public init(layout: SafetensorsLayout, numExperts: Int) {
        self.layout = layout
        self.numExperts = numExperts
    }

    /// Byte sub-range (within the tensor's already-absolute `byteRange`) that
    /// expert `e` occupies in a `[numExperts, ...]` stacked tensor. Exposed
    /// as a pure function (no I/O) so the offset math is independently
    /// unit-testable against a whole-tensor load.
    static func expertByteRange(
        of location: SafetensorsTensorLocation, expert e: Int, numExperts: Int
    ) throws -> Range<Int> {
        guard let leading = location.shape.first, leading == numExperts else {
            throw ExpertShardStoreError.notExpertStacked(
                "expected leading dim \(numExperts), got shape \(location.shape)")
        }
        let totalBytes = location.byteRange.count
        precondition(totalBytes % numExperts == 0, "stacked tensor size not divisible by expert count")
        let stride = totalBytes / numExperts
        let base = location.byteRange.lowerBound
        return (base + e * stride) ..< (base + (e + 1) * stride)
    }

    /// Shape of one expert's slice once the leading `numExperts` axis is
    /// dropped, e.g. `[256, 2048, 512] → [2048, 512]`.
    private static func perExpertShape(_ shape: [Int]) -> [Int] { Array(shape.dropFirst()) }

    /// Fetch a batch of expert indices for one layer's `switch_mlp`
    /// sub-tree, in parallel. Throws if the checkpoint is missing any of the
    /// gate/up/down weight or scale tensors for this layer (layer-index or
    /// checkpoint-shape mismatch) — biases are optional and silently absent
    /// rather than an error, matching the resident `QuantizedSwitchLinear`
    /// load path where `biases` is `MLXArray?`.
    public func fetch(layerIndex: Int, experts: [Int]) throws -> [Int: ExpertWeights] {
        let prefix = "model.layers.\(layerIndex).ffn.switch_mlp"

        func required(_ name: String) throws -> SafetensorsTensorLocation {
            guard let loc = layout["\(prefix).\(name)"] else {
                throw ExpertShardStoreError.missingTensor("\(prefix).\(name)")
            }
            return loc
        }

        let gateW = try required("gate_proj.weight")
        let gateS = try required("gate_proj.scales")
        let upW = try required("up_proj.weight")
        let upS = try required("up_proj.scales")
        let downW = try required("down_proj.weight")
        let downS = try required("down_proj.scales")
        let gateB = layout["\(prefix).gate_proj.biases"]
        let upB = layout["\(prefix).up_proj.biases"]
        let downB = layout["\(prefix).down_proj.biases"]

        let n = experts.count
        let state = ExpertFetchState(count: n)

        DispatchQueue.concurrentPerform(iterations: n) { i in
            do {
                let e = experts[i]
                let gw = try Self.readExpertTensor(gateW, expert: e, numExperts: numExperts)
                let gs = try Self.readExpertTensor(gateS, expert: e, numExperts: numExperts)
                let gb = try gateB.map { try Self.readExpertTensor($0, expert: e, numExperts: numExperts) }
                let uw = try Self.readExpertTensor(upW, expert: e, numExperts: numExperts)
                let us = try Self.readExpertTensor(upS, expert: e, numExperts: numExperts)
                let ub = try upB.map { try Self.readExpertTensor($0, expert: e, numExperts: numExperts) }
                let dw = try Self.readExpertTensor(downW, expert: e, numExperts: numExperts)
                let ds = try Self.readExpertTensor(downS, expert: e, numExperts: numExperts)
                let db = try downB.map { try Self.readExpertTensor($0, expert: e, numExperts: numExperts) }

                let bytes =
                    gw.nbytes + gs.nbytes + (gb?.nbytes ?? 0)
                    + uw.nbytes + us.nbytes + (ub?.nbytes ?? 0)
                    + dw.nbytes + ds.nbytes + (db?.nbytes ?? 0)

                state.store(
                    index: i,
                    weights: ExpertWeights(
                        gateWeight: gw, gateScales: gs, gateBiases: gb,
                        upWeight: uw, upScales: us, upBiases: ub,
                        downWeight: dw, downScales: ds, downBiases: db,
                        byteCount: bytes))
            } catch {
                state.recordError(error)
            }
        }
        if let firstError = state.firstError { throw firstError }

        var out: [Int: ExpertWeights] = [:]
        out.reserveCapacity(n)
        for (i, e) in experts.enumerated() {
            guard let w = state.results[i] else { throw ExpertShardStoreError.shortRead("expert \(e)") }
            out[e] = w
        }
        return out
    }

    /// pread one expert's contiguous byte range for a tensor and wrap it as
    /// an MLXArray with the per-expert shape (leading expert axis dropped).
    static func readExpertTensor(
        _ location: SafetensorsTensorLocation, expert e: Int, numExperts: Int
    ) throws -> MLXArray {
        let range = try expertByteRange(of: location, expert: e, numExperts: numExperts)
        let data = try preadRange(fileURL: location.fileURL, range: range)
        let shape = perExpertShape(location.shape)
        return MLXArray(data, shape, dtype: location.dtype.mlxDType)
    }

    /// Raw `pread(2)` of an absolute byte range. Uses the POSIX call
    /// directly (not `FileHandle.seek(toOffset:)` + `read`) so concurrent
    /// reads issued from different `DispatchQueue.concurrentPerform` threads
    /// against the same fd never race on a shared file offset — pread takes
    /// the offset as an explicit argument, so it's safe to call from many
    /// threads on independently-opened file descriptors (or even the same
    /// fd) with no coordination.
    private static func preadRange(fileURL: URL, range: Range<Int>) throws -> Data {
        let fd = open(fileURL.path, O_RDONLY)
        guard fd >= 0 else { throw ExpertShardStoreError.shortRead(fileURL.path) }
        defer { close(fd) }

        let count = range.count
        var buffer = Data(count: count)
        let bytesRead = buffer.withUnsafeMutableBytes { ptr -> Int in
            guard let base = ptr.baseAddress else { return -1 }
            return pread(fd, base, count, off_t(range.lowerBound))
        }
        guard bytesRead == count else {
            throw ExpertShardStoreError.shortRead(
                "\(fileURL.lastPathComponent): wanted \(count) bytes, got \(bytesRead)")
        }
        return buffer
    }
}

extension SafetensorsDType {
    var mlxDType: DType {
        switch self {
        case .bool: return .bool
        case .uint8: return .uint8
        case .uint16: return .uint16
        case .uint32: return .uint32
        case .uint64: return .uint64
        case .int8: return .int8
        case .int16: return .int16
        case .int32: return .int32
        case .int64: return .int64
        case .bfloat16: return .bfloat16
        case .float16: return .float16
        case .float32: return .float32
        case .float64: return .float64
        }
    }
}
