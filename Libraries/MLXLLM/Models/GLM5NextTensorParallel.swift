// Copyright © 2026 Eigen Labs.
import Foundation
import MLX

/// One rank of a tensor-parallel GLM text model: every layer on every rank,
/// each holding `1/size` of each head- or channel-parallel weight.
///
/// Split (rank-local contiguous half):
/// - KDA: per-head q/k/v projections and short convolutions, f/g second
///   projections, `b_proj`, `A_log`, `dt_bias` (axis 0); `o_proj` input columns.
/// - DSA: `q_b_proj` and `kv_b_proj` head rows (axis 0); `o_proj` input columns.
///   The latent projections, their norms and the whole indexer are replicated,
///   so every rank selects the same tokens.
/// - Dense MLP, shared experts and every routed expert: gate/up output rows,
///   down input columns. The router is replicated, so ranks route identically.
/// Everything else (embedding, hyper-connections, norms, router, head) is
/// replicated. Each layer then needs two sums across ranks: the attention
/// output and the feed-forward output (`GLM5NextTextLayer.reduce`).
public struct GLM5NextTensorShard: Sendable, Equatable {
    public let rank: Int
    public let size: Int

    public init(rank: Int, size: Int) throws {
        guard size == 2, (0 ..< size).contains(rank) else {
            throw GLM5NextLoadError.invalid("GLM tensor parallel supports two ranks")
        }
        self.rank = rank
        self.size = size
    }

    /// Which axis of a stored tensor this rank keeps a part of; nil replicates it.
    ///
    /// An ASCII name (every inventory name) is matched on its bytes: the same
    /// answer as `referenceSplitAxis`, about 50 times faster (a rank asks it for
    /// each of the 114,160 tensors three times while it loads: 1.0 s each time
    /// through Foundation's string search, M5 Ultra).
    public static func splitAxis(_ name: String) -> Int? {
        var contiguous = name
        return contiguous.withUTF8 { bytes in
            bytes.allSatisfy({ $0 < 0x80 }) ? asciiSplitAxis(bytes) : referenceSplitAxis(name)
        }
    }

    /// `splitAxis` as written against Foundation's string search; the
    /// definition the byte matcher must agree with.
    static func referenceSplitAxis(_ name: String) -> Int? {
        let layer = "model.language_model.layers."
        guard name.hasPrefix(layer) else { return nil }
        // weight / scales / biases of packed linears split together; the NVFP4
        // second-level scale is one scalar per matrix and stays replicated.
        guard !name.hasSuffix(".weight_scale_2") else { return nil }
        if rowSplit.contains(where: { name.contains($0) }) { return 0 }
        if name.hasSuffix(".self_attn.A_log") || name.hasSuffix(".self_attn.dt_bias") { return 0 }
        if name.contains(".self_attn.o_proj.") || name.contains(".down_proj.") { return 1 }
        return nil
    }

    private static let rowSplit = [
        ".self_attn.q_proj.", ".self_attn.k_proj.", ".self_attn.v_proj.",
        ".self_attn.q_conv1d.", ".self_attn.k_conv1d.", ".self_attn.v_conv1d.",
        ".self_attn.f_b_proj.", ".self_attn.g_b_proj.", ".self_attn.b_proj.",
        ".self_attn.q_b_proj.", ".self_attn.kv_b_proj.",
        ".gate_proj.", ".up_proj.",
    ]
    private static let asciiRowSplit = rowSplit.map { Array($0.utf8) }
    private static let asciiLayer = Array("model.language_model.layers.".utf8)
    private static let asciiColumnSplit = [".self_attn.o_proj.", ".down_proj."].map { Array($0.utf8) }
    private static let asciiReplicatedSuffix = Array(".weight_scale_2".utf8)
    private static let asciiRowSuffixes = [".self_attn.A_log", ".self_attn.dt_bias"].map { Array($0.utf8) }

    private static func asciiSplitAxis(_ name: UnsafeBufferPointer<UInt8>) -> Int? {
        func hasPrefix(_ p: [UInt8]) -> Bool { name.count >= p.count && name.prefix(p.count).elementsEqual(p) }
        func hasSuffix(_ p: [UInt8]) -> Bool { name.count >= p.count && name.suffix(p.count).elementsEqual(p) }
        func contains(_ p: [UInt8]) -> Bool {
            guard let base = name.baseAddress, name.count >= p.count else { return false }
            return p.withUnsafeBytes { memmem(base, name.count, $0.baseAddress, p.count) != nil }
        }
        guard hasPrefix(asciiLayer), !hasSuffix(asciiReplicatedSuffix) else { return nil }
        if asciiRowSplit.contains(where: contains) { return 0 }
        if asciiRowSuffixes.contains(where: hasSuffix) { return 0 }
        if asciiColumnSplit.contains(where: contains) { return 1 }
        return nil
    }

    /// This rank's part of a full stored tensor, as an array owning its own
    /// storage (a slice alone would keep the whole source buffer alive).
    public func slice(_ name: String, _ full: MLXArray) throws -> MLXArray {
        guard let axis = Self.splitAxis(name) else { return full }
        guard full.ndim > axis, full.dim(axis) % size == 0 else {
            throw GLM5NextLoadError.invalid("tensor-parallel split of \(name) \(full.shape)")
        }
        let width = full.dim(axis) / size
        let range = (rank * width) ..< ((rank + 1) * width)
        let part = axis == 0 ? full[range] : full[0..., range]
        return stacked([part]).squeezed(axis: 0)
    }

    /// The shape of `slice(name, full)` for a stored tensor of `shape`, with
    /// the same refusal.
    public func localShape(_ name: String, _ shape: [Int]) throws -> [Int] {
        guard let axis = Self.splitAxis(name) else { return shape }
        guard shape.count > axis, shape[axis] % size == 0 else {
            throw GLM5NextLoadError.invalid("tensor-parallel split of \(name) \(shape)")
        }
        var local = shape
        local[axis] /= size
        return local
    }

    /// Where the bytes of `slice(name, full)` lie in the stored row-major
    /// payload (`bytes` long) of a tensor of `shape`, in order: `count` runs of
    /// `length` bytes, the first at `offset`, each `stride` after the one
    /// before. A row split keeps one contiguous run; a column split keeps a
    /// run of every row (its columns are contiguous within the row).
    public func byteRuns(_ name: String, shape: [Int], bytes: Int) throws -> GLM5NextByteRuns {
        guard let axis = Self.splitAxis(name) else { return .whole(bytes) }
        _ = try localShape(name, shape)
        guard bytes > 0, bytes % shape[0] == 0 else {
            throw GLM5NextLoadError.invalid("tensor-parallel payload of \(name) \(shape)")
        }
        if axis == 0 {
            let part = bytes / size
            return GLM5NextByteRuns(offset: rank * part, length: part, count: 1, stride: part)
        }
        let row = bytes / shape[0]
        let part = row / size
        return GLM5NextByteRuns(offset: rank * part, length: part, count: shape[0], stride: row)
    }

    /// The configuration every module of this rank is built from: the same
    /// model with half the attention heads and feed-forward channels.
    public func localConfiguration(_ c: GLM5NextTextConfiguration) throws -> GLM5NextTextConfiguration {
        guard c.attentionHeads % size == 0, c.linearAttention.numHeads % size == 0,
            c.intermediateSize % size == 0, c.moeIntermediateSize % size == 0
        else { throw GLM5NextLoadError.invalid("GLM geometry does not divide across ranks") }
        var object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(c)) as? [String: Any] ?? [:]
        func halve(_ key: String, in dictionary: inout [String: Any]) throws {
            guard let value = dictionary[key] as? Int else {
                throw GLM5NextLoadError.invalid("tensor-parallel configuration lacks \(key)")
            }
            dictionary[key] = value / size
        }
        try halve("num_attention_heads", in: &object)
        // MLA: one latent key/value head per query head.
        try halve("num_key_value_heads", in: &object)
        try halve("intermediate_size", in: &object)
        try halve("moe_intermediate_size", in: &object)
        guard var linear = object["linear_attn_config"] as? [String: Any] else {
            throw GLM5NextLoadError.invalid("tensor-parallel configuration lacks linear_attn_config")
        }
        try halve("num_heads", in: &linear)
        object["linear_attn_config"] = linear
        let local = try JSONDecoder().decode(
            GLM5NextTextConfiguration.self, from: JSONSerialization.data(withJSONObject: object))
        guard local.attentionHeads * size == c.attentionHeads,
            local.linearAttention.numHeads * size == c.linearAttention.numHeads,
            local.hiddenSize == c.hiddenSize, local.hiddenLayers == c.hiddenLayers,
            local.layerTypes == c.layerTypes, local.hcMult == c.hcMult
        else { throw GLM5NextLoadError.invalid("tensor-parallel local configuration differs") }
        return local
    }
}

/// A strided byte selection of a stored payload (`GLM5NextTensorShard.byteRuns`):
/// `count` runs of `length` bytes, the first at `offset`, each `stride` after
/// the one before; the selected bytes, in order, are a contiguous array.
public struct GLM5NextByteRuns: Sendable, Equatable {
    public let offset: Int
    public let length: Int
    public let count: Int
    public let stride: Int

    public init(offset: Int, length: Int, count: Int, stride: Int) {
        self.offset = offset
        self.length = length
        self.count = count
        self.stride = stride
    }

    /// The whole payload as one run.
    public static func whole(_ bytes: Int) -> Self {
        Self(offset: 0, length: bytes, count: 1, stride: bytes)
    }

    /// Selected bytes.
    public var bytes: Int { length * count }
    /// One past the last selected byte, relative to the payload.
    public var end: Int { offset + (count - 1) * stride + length }

    /// Copies the selected bytes that fall in `window` (payload-relative
    /// positions `window.lowerBound ..< window.upperBound`, whose bytes start
    /// at `source`) to their places in the contiguous selection at
    /// `destination`. Calling it for consecutive windows covering the payload
    /// writes every selected byte exactly once.
    public func copy(
        window: Range<Int>, from source: UnsafeRawPointer, to destination: UnsafeMutableRawPointer
    ) {
        guard window.lowerBound < end, window.upperBound > offset else { return }
        // Run i covers [offset + i * stride, + length): the first run ending
        // after the window starts, through the last run starting before it ends.
        let behind = window.lowerBound - offset - length
        let first = behind < 0 ? 0 : behind / stride + 1
        let last = min(count - 1, (window.upperBound - offset + stride - 1) / stride - 1)
        guard first <= last else { return }
        for run in first ... last {
            let start = offset + run * stride
            let lower = max(start, window.lowerBound)
            let upper = min(start + length, window.upperBound)
            guard lower < upper else { continue }
            (destination + (run * length + lower - start)).copyMemory(
                from: source + (lower - window.lowerBound), byteCount: upper - lower)
        }
    }
}
