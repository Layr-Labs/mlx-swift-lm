// Copyright © 2026 Eigen Labs / Darkbloom Assignment 1.
//
// Qwen4-Exp n-gram PLE: hashed IDs, oQ affine gather, mmap-first residency.
// Source of truth: Fusion `mlx_vlm/models/qwen4_exp/language.py`.
// Default mmap gather is Fusion #3372 (unique + concurrent shard copy +
// one dequant). `DARKBLOOM_QWEN4_PLE_GATHER=0` restores serial dequant.

import CoreFoundation
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Geometry (pure; tested against Fusion defaults)

public enum Qwen4ExpNGramGeometry: Sendable {
    public static let defaultSeed = 1234
    public static let splitmixGamma: UInt64 = 0x9E37_79B9_7F4A_7C15
    public static let splitmixM1: UInt64 = 0xBF58_476D_1CE4_E5B9
    public static let splitmixM2: UInt64 = 0x94D0_49BB_1331_11EB
    public static let prime1: Int = 10_007
    public static let pleRecurrentLayerBase = 10_000

    public static func recurrentLayerIndex(_ modelLayerIndex: Int) -> Int {
        pleRecurrentLayerBase + modelLayerIndex
    }

    public static func isPrime(_ value: Int) -> Bool {
        if value < 2 { return false }
        if value % 2 == 0 { return value == 2 }
        var divisor = 3
        while divisor &* divisor <= value {
            if value % divisor == 0 { return false }
            divisor += 2
        }
        return true
    }

    public static func nthPrimeAfter(start: Int, count: Int) -> Int {
        var prime = start
        for _ in 0 ..< count {
            prime += 1
            while !isPrime(prime) { prime += 1 }
        }
        return prime
    }

    public static func splitmix64(_ value: UInt64) -> UInt64 {
        var current = value &+ splitmixGamma
        current = (current ^ (current >> 30)) &* splitmixM1
        current = (current ^ (current >> 27)) &* splitmixM2
        return current ^ (current >> 31)
    }

    public static func layerMultipliers(
        unigramVocabSize: Int, ngramSize: Int, pleLayerIndex: Int, seed: Int
    ) -> [Int64] {
        let maxLong = Int64.max
        let multiplierMax = maxLong / Int64(max(unigramVocabSize, 1))
        let halfBound = max(Int64(1), multiplierMax / 2)
        let baseSeed = UInt64(bitPattern: Int64(seed)) &+ UInt64(prime1) &* UInt64(pleLayerIndex)
        return (0 ..< ngramSize).map { index in
            let mixed = (baseSeed &+ splitmixGamma &* UInt64(index + 1))
            let odd = 2 * Int64(splitmix64(mixed) % UInt64(halfBound)) + 1
            return odd
        }
    }

    public static func headVocabSizes(
        ngramHeads: Int, pleLayerIndex: Int, base: Int
    ) -> [Int] {
        (0 ..< ngramHeads).map { head in
            nthPrimeAfter(start: base - 1, count: pleLayerIndex * ngramHeads + head + 1)
        }
    }

    public static func paddedVocabSize(total: Int, divisor: Int) -> Int {
        let step = max(divisor, 1)
        return ((total + step - 1) / step) * step
    }

    public static func floorMod(_ value: Int64, _ modulus: Int64) -> Int64 {
        let remainder = value % modulus
        return remainder >= 0 ? remainder : remainder + modulus
    }
}

public struct Qwen4ExpNGramTables: Sendable {
    public let ngramSize: Int
    public let headsPerNgram: Int
    public let ngramHeads: Int
    public let contextLen: Int
    public let eosTokenId: Int
    public let multipliers: [Int64]
    public let headVocabSizes: [Int]
    public let headOffsets: [Int]
    public let paddedVocabSize: Int
    public let shardCount: Int
    public let shardSizes: [Int]
    public let shardOffsets: [Int]
    public let embedDim: Int
    public let headEmbedDim: Int

    public init(_ args: Qwen4ExpTextConfiguration, pleIndex: Int) {
        self.ngramSize = args.ngramSize
        self.headsPerNgram = args.headsPerNgram
        self.contextLen = max(0, args.ngramSize - 1)
        self.ngramHeads = contextLen * args.headsPerNgram
        self.eosTokenId = args.eosTokenId.first ?? 248_044
        self.embedDim = args.pleEmbedDim
        self.headEmbedDim = args.pleEmbedDim / max(ngramHeads, 1)
        self.multipliers = Qwen4ExpNGramGeometry.layerMultipliers(
            unigramVocabSize: args.vocabularySize,
            ngramSize: args.ngramSize,
            pleLayerIndex: pleIndex,
            seed: args.seed)
        let sizes = Qwen4ExpNGramGeometry.headVocabSizes(
            ngramHeads: ngramHeads,
            pleLayerIndex: pleIndex,
            base: args.ngramVocabSizeBase)
        self.headVocabSizes = sizes
        var offsets: [Int] = []
        var total = 0
        for size in sizes {
            offsets.append(total)
            total += size
        }
        self.headOffsets = offsets
        self.paddedVocabSize = Qwen4ExpNGramGeometry.paddedVocabSize(
            total: total, divisor: args.makeNgramVocabSizeDivisibleBy)
        self.shardCount = args.splitNgramParts
        let base = paddedVocabSize / shardCount
        let remainder = paddedVocabSize % shardCount
        self.shardSizes = (0 ..< shardCount).map { $0 < remainder ? base + 1 : base }
        var shardOff = [0]
        for size in shardSizes { shardOff.append(shardOff[shardOff.count - 1] + size) }
        self.shardOffsets = shardOff
    }

    public func shardIndex(for id: Int) -> Int {
        var low = 0
        var high = shardSizes.count
        while low + 1 < high {
            let mid = (low + high) / 2
            if shardOffsets[mid] <= id { low = mid } else { high = mid }
        }
        return low
    }
}

// MARK: - Host n-gram IDs (same wrapping / floor-mod as Fusion int64)

enum Qwen4ExpNGramIDs {
    static func shiftRightIgnoreEOS(tokens: [Int], shift: Int, eos: Int) -> [Int] {
        if shift == 0 { return tokens }
        let seq = tokens.count
        var eosPositions = [Int](repeating: -1, count: seq)
        for index in 0 ..< seq where tokens[index] == eos {
            eosPositions[index] = index
        }
        var inclusive = [Int](repeating: -1, count: seq)
        var running = -1
        for index in 0 ..< seq {
            running = max(running, eosPositions[index])
            inclusive[index] = running
        }
        var previousEOS = [Int](repeating: -1, count: seq)
        if seq > 1 {
            for index in 1 ..< seq { previousEOS[index] = inclusive[index - 1] }
        }
        var shifted = [Int](repeating: eos, count: seq)
        for index in 0 ..< seq {
            let segmentStart = previousEOS[index] + 1
            let positionInSegment = index - segmentStart
            let source = index - shift
            if positionInSegment >= shift && source >= 0 {
                shifted[index] = tokens[source]
            }
        }
        return shifted
    }

    static func ids(
        history: [Int], inputWidth: Int, tables: Qwen4ExpNGramTables
    ) -> [[Int]] {
        let shifted = (0 ..< tables.ngramSize).map {
            shiftRightIgnoreEOS(tokens: history, shift: $0, eos: tables.eosTokenId)
        }
        let seq = history.count
        var blocks = [[[Int]]]()
        for ngram in 2 ... tables.ngramSize {
            let start = (ngram - 2) * tables.headsPerNgram
            var row = [[Int]](repeating: [Int](repeating: 0, count: tables.headsPerNgram), count: seq)
            for position in 0 ..< seq {
                var mixed = Int64(shifted[0][position]) &* tables.multipliers[0]
                if ngram > 1 {
                    for inner in 1 ..< ngram {
                        mixed ^= Int64(shifted[inner][position]) &* tables.multipliers[inner]
                    }
                }
                for head in 0 ..< tables.headsPerNgram {
                    let size = Int64(tables.headVocabSizes[start + head])
                    let offset = tables.headOffsets[start + head]
                    row[position][head] =
                        Int(Qwen4ExpNGramGeometry.floorMod(mixed, size)) + offset
                }
            }
            blocks.append(row)
        }
        let start = max(0, seq - inputWidth)
        return (start ..< seq).map { position in
            blocks.flatMap { $0[position] }
        }
    }

    /// Fusion `_ngram_ids_from_history` / `_shift_right_ignore_eos` on GPU.
    /// `history` is `[B, context+T]` token ids. Result is `[B, T, ngramHeads]`.
    static func gpuIds(
        history: MLXArray, inputWidth: Int, tables: Qwen4ExpNGramTables
    ) -> MLXArray {
        let tokens = history.asType(.int64)
        let seq = tokens.dim(1)
        precondition(inputWidth > 0 && inputWidth <= seq, "PLE GPU ids need a positive input window")
        var shifted: [MLXArray] = []
        shifted.reserveCapacity(tables.ngramSize)
        for shift in 0 ..< tables.ngramSize {
            shifted.append(gpuShiftRightIgnoreEOS(tokens, shift: shift, eos: tables.eosTokenId))
        }
        var blocks: [MLXArray] = []
        for ngram in 2 ... tables.ngramSize {
            let start = (ngram - 2) * tables.headsPerNgram
            var mixed = shifted[0] * tables.multipliers[0]
            if ngram > 1 {
                for inner in 1 ..< ngram {
                    mixed = mixed ^ (shifted[inner] * tables.multipliers[inner])
                }
            }
            let sizes = MLXArray(
                tables.headVocabSizes[start ..< (start + tables.headsPerNgram)].map { Int64($0) })
            let offsets = MLXArray(
                tables.headOffsets[start ..< (start + tables.headsPerNgram)].map { Int64($0) })
            blocks.append(mixed.expandedDimensions(axis: -1) % sizes + offsets)
        }
        let concatenatedIds = concatenated(blocks, axis: -1)
        return concatenatedIds[0..., (seq - inputWidth)..., 0...]
    }

    static func gpuShiftRightIgnoreEOS(_ tokenIds: MLXArray, shift: Int, eos: Int) -> MLXArray {
        if shift == 0 { return tokenIds }
        let batch = tokenIds.dim(0)
        let seq = tokenIds.dim(1)
        let positions = arange(seq, dtype: .int64)
        let eosPositions = MLX.where(
            tokenIds .== Int64(eos), broadcast(positions, to: [batch, seq]),
            MLXArray.full([batch, seq], values: MLXArray(Int64(-1)), type: Int64.self))
        let inclusive = cummax(eosPositions, axis: 1)
        let previousEOS = concatenated(
            [
                MLXArray.full([batch, 1], values: MLXArray(Int64(-1)), type: Int64.self),
                inclusive[0..., 0 ..< (seq - 1)],
            ], axis: 1)
        let segmentStart = previousEOS + 1
        let positionInSegment = broadcast(positions, to: [batch, seq]) - segmentStart
        let sourcePositions = positions - Int64(shift)
        let gatherPositions = broadcast(MLX.maximum(sourcePositions, 0), to: [batch, seq])
        let shifted = takeAlong(tokenIds, gatherPositions, axis: 1)
        let valid = logicalAnd(
            positionInSegment .>= Int64(shift),
            broadcast(sourcePositions, to: [batch, seq]) .>= 0)
        return MLX.where(
            valid, shifted,
            MLXArray.full([batch, seq], values: MLXArray(Int64(eos)), type: Int64.self))
    }
}

// MARK: - Safetensors row mmap

final class Qwen4ExpSafeTensorMMap {
    static let maxHeaderBytes = 16 * 1024 * 1024

    struct Tensor {
        let dtype: String
        let shape: [Int]
        let dataStart: Int
        let dataEnd: Int
    }

    let url: URL
    private let data: Data
    private let dataOrigin: Int
    private let tensors: [String: Tensor]

    init(url: URL) throws {
        self.url = url
        // `.mappedIfSafe` may silently fall back to a full heap copy.
        // Checkpoints are verified local regular files; require an actual
        // read-only VM mapping so load/reload residency cannot unexpectedly
        // duplicate a 30 GiB learned table.
        let mapped = try Data(contentsOf: url, options: [.alwaysMapped])
        guard mapped.count >= 8 else {
            throw Qwen4ExpPLEError.safetensors("header too small: \(url.lastPathComponent)")
        }
        let rawHeaderSize = mapped.prefix(8).withUnsafeBytes {
            $0.load(as: UInt64.self).littleEndian
        }
        guard rawHeaderSize <= UInt64(Self.maxHeaderBytes) else {
            throw Qwen4ExpPLEError.safetensors(
                "header exceeds \(Self.maxHeaderBytes) bytes: \(url.lastPathComponent)")
        }
        let headerSize = Int(rawHeaderSize)
        let headerEnd = 8 + headerSize
        guard headerEnd <= mapped.count else {
            throw Qwen4ExpPLEError.safetensors("header overflow: \(url.lastPathComponent)")
        }
        let headerJSON = mapped.subdata(in: 8 ..< headerEnd)
        guard let object = try JSONSerialization.jsonObject(with: headerJSON) as? [String: Any]
        else {
            throw Qwen4ExpPLEError.safetensors("header JSON: \(url.lastPathComponent)")
        }
        var parsed: [String: Tensor] = [:]
        for (key, raw) in object {
            if key == "__metadata__" { continue }
            guard let entry = raw as? [String: Any],
                let dtype = entry["dtype"] as? String,
                !dtype.isEmpty,
                let shapeRaw = entry["shape"] as? [Any],
                let offsets = entry["data_offsets"] as? [Any],
                offsets.count == 2
            else {
                throw Qwen4ExpPLEError.safetensors(
                    "malformed tensor entry \(key) in \(url.lastPathComponent)")
            }
            let shape = try shapeRaw.map { value in
                guard let dimension = Self.nonnegativeInteger(value) else {
                    throw Qwen4ExpPLEError.safetensors(
                        "invalid shape for \(key) in \(url.lastPathComponent)")
                }
                return dimension
            }
            guard let dataStart = Self.nonnegativeInteger(offsets[0]),
                let dataEnd = Self.nonnegativeInteger(offsets[1]),
                dataEnd >= dataStart
            else {
                throw Qwen4ExpPLEError.safetensors(
                    "invalid data offsets for \(key) in \(url.lastPathComponent)")
            }
            parsed[key] = Tensor(
                dtype: dtype,
                shape: shape,
                dataStart: dataStart,
                dataEnd: dataEnd)
        }
        self.data = mapped
        self.dataOrigin = headerEnd
        self.tensors = parsed
        Qwen4ExpPLEResourceMetrics.adjustMappedFiles(by: 1)
    }

    private static func nonnegativeInteger(_ value: Any) -> Int? {
        guard let number = value as? NSNumber,
            CFGetTypeID(number) != CFBooleanGetTypeID(),
            let integer = Int(number.stringValue), integer >= 0
        else { return nil }
        // Parsing the integer spelling avoids Double(Int.max) rounding up
        // to 2^63, which made malformed headers trap at Int(double).
        return integer
    }

    deinit {
        Qwen4ExpPLEResourceMetrics.adjustMappedFiles(by: -1)
    }

    func tensor(_ key: String) throws -> Tensor {
        guard let tensor = tensors[key] else {
            throw Qwen4ExpPLEError.safetensors("missing tensor \(key) in \(url.lastPathComponent)")
        }
        guard tensor.dataStart >= 0, tensor.dataEnd >= tensor.dataStart,
            dataOrigin <= data.count, tensor.dataEnd <= data.count - dataOrigin
        else {
            throw Qwen4ExpPLEError.safetensors(
                "invalid data offsets for \(key) in \(url.lastPathComponent)")
        }
        // Validate the selected PLE tensor BEFORE row output allocation or
        // affine geometry arithmetic. A tiny payload with a huge claimed
        // shape must throw, not overflow or allocate the claimed size.
        let itemBytes: Int? = switch tensor.dtype {
        case "U32": 4
        case "BF16": 2
        default: nil
        }
        if let itemBytes {
            var bytes = itemBytes
            for dimension in tensor.shape {
                let (next, overflow) = bytes.multipliedReportingOverflow(by: dimension)
                guard !overflow else {
                    throw Qwen4ExpPLEError.safetensors("PLE tensor byte count overflows")
                }
                bytes = next
            }
            guard bytes == tensor.dataEnd - tensor.dataStart else {
                throw Qwen4ExpPLEError.safetensors("PLE tensor byte count does not match its shape")
            }
        }
        return tensor
    }

    func rowsUInt32(key: String, rows: [Int]) throws -> [UInt32] {
        let tensor = try self.tensor(key)
        guard tensor.dtype == "U32", tensor.shape.count == 2 else {
            throw Qwen4ExpPLEError.safetensors("expected U32 rank-2 \(key)")
        }
        let cols = tensor.shape[1]
        let (count, overflow) = rows.count.multipliedReportingOverflow(by: cols)
        guard !overflow else { throw Qwen4ExpPLEError.safetensors("PLE row output size overflows") }
        var out = [UInt32](repeating: 0, count: count)
        try copyRows(tensor: tensor, rows: rows, itemSize: 4, destination: &out)
        return out
    }

    func rowsUInt16(key: String, rows: [Int]) throws -> [UInt16] {
        let tensor = try self.tensor(key)
        guard tensor.dtype == "BF16", tensor.shape.count == 2 else {
            throw Qwen4ExpPLEError.safetensors("expected BF16 rank-2 \(key)")
        }
        let cols = tensor.shape[1]
        let (count, overflow) = rows.count.multipliedReportingOverflow(by: cols)
        guard !overflow else { throw Qwen4ExpPLEError.safetensors("PLE row output size overflows") }
        var out = [UInt16](repeating: 0, count: count)
        try copyRows(tensor: tensor, rows: rows, itemSize: 2, destination: &out)
        return out
    }

    func copyIndexedRows<T>(
        key: String,
        sourceRows: [Int],
        destRows: [Int],
        destinationRowCount: Int,
        destination: UnsafeMutablePointer<T>
    ) throws {
        precondition(sourceRows.count == destRows.count)
        try copyIndexedRows(
            tensor: try tensor(key),
            sourceRows: sourceRows,
            destRows: destRows,
            destinationRowCount: destinationRowCount,
            destination: destination)
    }

    private func copyIndexedRows<T>(
        tensor: Tensor,
        sourceRows: [Int],
        destRows: [Int],
        destinationRowCount: Int,
        destination: UnsafeMutablePointer<T>
    ) throws {
        let itemSize = MemoryLayout<T>.stride
        let expectedDType =
            if T.self == UInt32.self {
                "U32"
            } else if T.self == UInt16.self {
                "BF16"
            } else {
                ""
            }
        guard !expectedDType.isEmpty, tensor.dtype == expectedDType,
            tensor.shape.count == 2, tensor.shape[0] >= 0, tensor.shape[1] >= 0
        else {
            throw Qwen4ExpPLEError.safetensors(
                "unexpected \(tensor.dtype) tensor for \(MemoryLayout<T>.stride)-byte PLE rows")
        }
        let cols = tensor.shape[1]
        let (rowBytes, rowOverflow) = cols.multipliedReportingOverflow(by: itemSize)
        let (tensorBytes, tensorOverflow) = tensor.shape[0].multipliedReportingOverflow(
            by: rowBytes)
        guard !rowOverflow, !tensorOverflow,
            tensor.dataEnd - tensor.dataStart == tensorBytes
        else {
            throw Qwen4ExpPLEError.safetensors("PLE tensor byte count does not match its shape")
        }
        let origin = dataOrigin + tensor.dataStart
        let limit = tensor.shape[0]
        try data.withUnsafeBytes { raw in
            guard let srcBase = raw.baseAddress else {
                throw Qwen4ExpPLEError.safetensors("PLE mmap row gather has a null buffer")
            }
            let dstBase = UnsafeMutableRawPointer(destination)
            for (output, row) in sourceRows.enumerated() {
                guard row >= 0, row < limit else {
                    throw Qwen4ExpPLEError.safetensors(
                        "row \(row) out of range for shape \(tensor.shape)")
                }
                let destinationRow = destRows[output]
                guard destinationRow >= 0, destinationRow < destinationRowCount else {
                    throw Qwen4ExpPLEError.safetensors(
                        "destination row \(destinationRow) out of range \(destinationRowCount)")
                }
                let start = origin + row * rowBytes
                let end = start + rowBytes
                guard end <= dataOrigin + tensor.dataEnd else {
                    throw Qwen4ExpPLEError.safetensors("row slice overflow")
                }
                dstBase.advanced(by: destinationRow * rowBytes)
                    .copyMemory(from: srcBase.advanced(by: start), byteCount: rowBytes)
            }
        }
    }

    private func copyRows<T>(
        tensor: Tensor, rows: [Int], itemSize: Int, destination: inout [T]
    ) throws {
        precondition(itemSize == MemoryLayout<T>.stride)
        try destination.withUnsafeMutableBufferPointer { dest in
            guard let destBase = dest.baseAddress else {
                throw Qwen4ExpPLEError.safetensors("PLE mmap row gather has a null buffer")
            }
            try copyIndexedRows(
                tensor: tensor,
                sourceRows: rows,
                destRows: Array(0 ..< rows.count),
                destinationRowCount: rows.count,
                destination: destBase)
        }
    }
}

enum Qwen4ExpPLEError: Error, CustomStringConvertible {
    case missingModelDirectory
    case safetensors(String)
    case gather(String)

    var description: String {
        switch self {
        case .missingModelDirectory:
            return "Qwen4 PLE mmap has no model directory bound by the checkpoint loader"
        case .safetensors(let message), .gather(let message):
            return "Qwen4 PLE: \(message)"
        }
    }
}

/// Fusion #3372 batched mmap gather: unique ids, concurrent shard copies,
/// one affine dequant. `DARKBLOOM_QWEN4_PLE_GATHER=0` restores the serial
/// per-shard dequant loop.
enum Qwen4ExpPLEGather: Sendable {
    static let envFlag = "DARKBLOOM_QWEN4_PLE_GATHER"

    static func isEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        let raw = environment[envFlag]?.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if raw == "0" || raw == "false" || raw == "no" || raw == "off" {
            return false
        }
        return true
    }

    /// First-seen unique ids and the inverse map back to the original order.
    static func uniqueInverse(_ ids: [Int]) -> (unique: [Int], inverse: [Int32]) {
        var first: [Int: Int] = [:]
        first.reserveCapacity(ids.count)
        var unique: [Int] = []
        unique.reserveCapacity(ids.count)
        var inverse = [Int32](repeating: 0, count: ids.count)
        for (index, id) in ids.enumerated() {
            if let seen = first[id] {
                inverse[index] = Int32(seen)
            } else {
                let dest = unique.count
                first[id] = dest
                unique.append(id)
                inverse[index] = Int32(dest)
            }
        }
        return (unique, inverse)
    }
}

/// mlx-serve 26.9.1 "graph before the n-gram lookup" for the chained decode.
/// Kill: `DARKBLOOM_QWEN4_PLE_DEFERRED=0` restores the eager `eval(ids)`
/// gather inside the forward (which serializes graph build behind the
/// previous step's GPU work).
public enum Qwen4ExpPLEDeferred: Sendable {
    public static let envFlag = "DARKBLOOM_QWEN4_PLE_DEFERRED"

    public static func isEnabled(
        environment: [String: String] = Qwen4ExpEnvironment.snapshot
    ) -> Bool {
        let raw = environment[envFlag]?.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if raw == "0" || raw == "false" || raw == "no" || raw == "off" {
            return false
        }
        return true
    }
}

/// Byte bound for reusable packed/scales/biases buffers behind deferred PLE
/// row reads. Long-lived providers see arbitrary final-chunk widths; keeping
/// two buffers for every width would be process-lifetime and unbounded.
public enum Qwen4ExpPLEDeferredBufferPolicy: Sendable {
    public static let byteBudgetFlag = "DARKBLOOM_QWEN4_PLE_ROW_BUFFER_MB"
    public static let defaultByteBudget = 256 * 1_024 * 1_024

    public static func byteBudget(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        guard let raw = environment[byteBudgetFlag],
            let mib = Int(raw.trimmingCharacters(in: .whitespacesAndNewlines)),
            mib >= 0
        else { return defaultByteBudget }
        let (bytes, overflow) = mib.multipliedReportingOverflow(by: 1_024 * 1_024)
        return overflow ? Int.max : bytes
    }

    public static func slotBytes(
        rows: Int, packedCols: Int, scaleCols: Int
    ) -> Int {
        guard rows >= 0, packedCols >= 0, scaleCols >= 0 else { return Int.max }
        let (packedBytes, packedOverflow) = packedCols.multipliedReportingOverflow(by: 4)
        let (affineCols, affineOverflow) = scaleCols.multipliedReportingOverflow(by: 2)
        let (affineBytes, affineBytesOverflow) = affineCols.multipliedReportingOverflow(by: 2)
        guard !packedOverflow, !affineOverflow, !affineBytesOverflow else { return Int.max }
        let (rowBytes, rowOverflow) = packedBytes.addingReportingOverflow(affineBytes)
        guard !rowOverflow else { return Int.max }
        let (total, totalOverflow) = rows.multipliedReportingOverflow(by: rowBytes)
        return totalOverflow ? Int.max : total
    }
}

/// Process-wide count of PLE decode forwards that took the deferred slot
/// path vs the eager in-forward gather (prefill, unscoped forwards, kill
/// switch). Surfaced on the provider's local `/metrics`.
public enum Qwen4ExpPLEDeferredInvocation: Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var deferred = 0
    nonisolated(unsafe) private static var eager = 0

    public struct Snapshot: Sendable, Equatable {
        public var deferred: Int
        public var eager: Int

        public init(deferred: Int, eager: Int) {
            self.deferred = deferred
            self.eager = eager
        }

        public var line: String { "pleGather deferred=\(deferred) eager=\(eager)" }
    }

    static func recordDeferred() { lock.withLock { deferred += 1 } }
    static func recordEager() { lock.withLock { eager += 1 } }

    public static func snapshot() -> Snapshot {
        lock.withLock { Snapshot(deferred: deferred, eager: eager) }
    }

    public static func resetForTesting() {
        lock.withLock {
            deferred = 0
            eager = 0
        }
    }
}

/// Process-local, prompt-free PLE resource gauges for qualification and leak
/// detection. Counts include cached and one-shot deferred buffers; mapped
/// files are VM mappings, not a claim that all mapped pages are resident.
public enum Qwen4ExpPLEResourceMetrics: Sendable {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var mappedFiles = 0
    nonisolated(unsafe) private static var activeRowBufferBytes = 0
    nonisolated(unsafe) private static var cachedRowBufferBytes = 0

    public struct Snapshot: Sendable, Equatable {
        public let mappedFiles: Int
        public let activeRowBufferBytes: Int
        public let cachedRowBufferBytes: Int

        public init(
            mappedFiles: Int,
            activeRowBufferBytes: Int,
            cachedRowBufferBytes: Int
        ) {
            self.mappedFiles = mappedFiles
            self.activeRowBufferBytes = activeRowBufferBytes
            self.cachedRowBufferBytes = cachedRowBufferBytes
        }
    }

    static func adjustMappedFiles(by delta: Int) {
        lock.withLock { mappedFiles = max(0, mappedFiles + delta) }
    }

    static func adjustActiveRowBufferBytes(by delta: Int) {
        lock.withLock { activeRowBufferBytes = max(0, activeRowBufferBytes + delta) }
    }

    static func adjustCachedRowBufferBytes(by delta: Int) {
        lock.withLock { cachedRowBufferBytes = max(0, cachedRowBufferBytes + delta) }
    }

    public static func snapshot() -> Snapshot {
        lock.withLock {
            Snapshot(
                mappedFiles: mappedFiles,
                activeRowBufferBytes: activeRowBufferBytes,
                cachedRowBufferBytes: cachedRowBufferBytes)
        }
    }

    public static func resetForTesting() {
        lock.withLock {
            mappedFiles = 0
            activeRowBufferBytes = 0
            cachedRowBufferBytes = 0
        }
    }
}

// MARK: - Embedding table (mmap or resident shards)

/// Fusion `DiskBackedShardedEmbedding` / `ShardedEmbedding`. Lives under
/// `ple_embedding.ngram_embedding` so oQ4e `weight_scale` matches the card.
final class Qwen4ExpShardedEmbedding: Module {
    @ParameterInfo(key: "weight_scale") var weightScale: MLXArray
    @ModuleInfo(key: "shards") var shards: [Embedding]

    init(mmap: Bool, shardSizes: [Int], headEmbedDim: Int) {
        _weightScale.wrappedValue = MLXArray.ones([1], dtype: .bfloat16)
        if mmap {
            _shards.wrappedValue = []
        } else {
            _shards.wrappedValue = shardSizes.map { size in
                Embedding(embeddingCount: size, dimensions: headEmbedDim)
            }
        }
        super.init()
    }
}

final class Qwen4ExpNGramEmbedding: Module {
    let tables: Qwen4ExpNGramTables
    let mmap: Bool
    let layerIndex: Int
    let weightPrefix: String

    @ModuleInfo(key: "ngram_embedding") var ngramEmbedding: Qwen4ExpShardedEmbedding

    private var mmapFiles: [String: Qwen4ExpSafeTensorMMap] = [:]
    private var shardKeys: [Int: (weight: String, scales: String, biases: String, file: String)] = [:]
    private var mmapReady = false
    private let deferredByteBudget: Int

    init(
        _ args: Qwen4ExpTextConfiguration,
        layerIndex: Int,
        pleIndex: Int,
        mmap: Bool,
        deferredByteBudget: Int? = nil
    ) {
        self.tables = Qwen4ExpNGramTables(args, pleIndex: pleIndex)
        self.mmap = mmap
        self.layerIndex = layerIndex
        self.weightPrefix =
            "language_model.model.layers.\(layerIndex).ple.ple_embedding.ngram_embedding"
        self.deferredByteBudget =
            deferredByteBudget ?? Qwen4ExpPLEDeferredBufferPolicy.byteBudget()
        _ngramEmbedding.wrappedValue = Qwen4ExpShardedEmbedding(
            mmap: mmap, shardSizes: tables.shardSizes, headEmbedDim: tables.headEmbedDim)
        super.init()
    }

    func gather(ids: [[Int]]) -> MLXArray {
        gatherFlat(ids.flatMap { $0 }, rows: ids.count)
    }

    func gather(_ ids: MLXArray) -> MLXArray {
        let heads = tables.ngramHeads
        precondition(ids.dim(-1) == heads, "PLE GPU ids must be [..., ngramHeads]")
        let rows = ids.size / heads
        eval(ids)
        let flat = ids.asArray(Int64.self).map { Int($0) }
        return gatherFlat(flat, rows: rows)
    }

    private func gatherFlat(_ flat: [Int], rows: Int) -> MLXArray {
        let heads = tables.ngramHeads
        let gathered: MLXArray
        if mmap {
            gathered = mmapGather(flat)
        } else {
            gathered = residentGather(flat)
        }
        return gathered.reshaped(rows, heads * tables.headEmbedDim) * ngramEmbedding.weightScale
    }

    private func residentGather(_ ids: [Int]) -> MLXArray {
        precondition(!ngramEmbedding.shards.isEmpty, "resident PLE shards were not constructed")
        let grouped = Dictionary(grouping: ids.enumerated()) { tables.shardIndex(for: $0.element) }
        var chunks: [MLXArray] = []
        var positions: [Int32] = []
        chunks.reserveCapacity(grouped.count)
        positions.reserveCapacity(ids.count)
        for (shard, items) in grouped.sorted(by: { $0.key < $1.key }) {
            let local = items.map { $0.element - tables.shardOffsets[shard] }
            chunks.append(ngramEmbedding.shards[shard](MLXArray(local)))
            positions.append(contentsOf: items.map { Int32($0.offset) })
        }
        return scatterGatheredRows(chunks, positions: positions, count: ids.count)
    }

    private func mmapGather(_ ids: [Int]) -> MLXArray {
        do {
            try ensureMmapCatalog()
            if Qwen4ExpPLEGather.isEnabled() {
                return try mmapGatherBatched(ids)
            }
            return try mmapGatherSerial(ids)
        } catch {
            preconditionFailure("Qwen4 PLE mmap gather failed: \(error)")
        }
    }

    /// Fusion #3372: unique rows, concurrent per-shard memcpy, one dequant.
    private func mmapGatherBatched(_ ids: [Int]) throws -> MLXArray {
        if ids.isEmpty {
            return MLXArray.zeros([0, tables.headEmbedDim], dtype: .bfloat16)
        }
        let (unique, inverse) = Qwen4ExpPLEGather.uniqueInverse(ids)
        var sourceByShard: [Int: [Int]] = [:]
        var destByShard: [Int: [Int]] = [:]
        sourceByShard.reserveCapacity(min(unique.count, tables.shardCount))
        destByShard.reserveCapacity(min(unique.count, tables.shardCount))
        for (dest, id) in unique.enumerated() {
            let shard = tables.shardIndex(for: id)
            sourceByShard[shard, default: []].append(id - tables.shardOffsets[shard])
            destByShard[shard, default: []].append(dest)
        }
        let shardOrder = sourceByShard.keys.sorted()
        guard let firstShard = shardOrder.first, let firstSpec = shardKeys[firstShard] else {
            throw Qwen4ExpPLEError.gather("batched PLE gather has no shards")
        }
        let firstFile = try file(named: firstSpec.file)
        let weightTensor = try firstFile.tensor(firstSpec.weight)
        let scaleTensor = try firstFile.tensor(firstSpec.scales)
        let packedCols = weightTensor.shape[1]
        let scaleCols = scaleTensor.shape[1]
        let groupSize = tables.headEmbedDim / scaleCols
        let bits = packedCols * 32 / tables.headEmbedDim
        var packed = [UInt32](repeating: 0, count: unique.count * packedCols)
        var scaleBits = [UInt16](repeating: 0, count: unique.count * scaleCols)
        var biasBits = [UInt16](repeating: 0, count: unique.count * scaleCols)

        struct ShardCopy: @unchecked Sendable {
            let file: Qwen4ExpSafeTensorMMap
            let weight: String
            let scales: String
            let biases: String
            let sourceRows: [Int]
            let destRows: [Int]
        }
        var copies: [ShardCopy] = []
        copies.reserveCapacity(shardOrder.count)
        for shard in shardOrder {
            guard let spec = shardKeys[shard],
                let sourceRows = sourceByShard[shard],
                let destRows = destByShard[shard]
            else {
                throw Qwen4ExpPLEError.gather("no mmap spec for shard \(shard)")
            }
            copies.append(
                ShardCopy(
                    file: try file(named: spec.file),
                    weight: spec.weight,
                    scales: spec.scales,
                    biases: spec.biases,
                    sourceRows: sourceRows,
                    destRows: destRows))
        }

        let failure = NSLock()
        nonisolated(unsafe) var captured: Error?
        packed.withUnsafeMutableBufferPointer { packedDest in
            scaleBits.withUnsafeMutableBufferPointer { scaleDest in
                biasBits.withUnsafeMutableBufferPointer { biasDest in
                    guard let packedBase = packedDest.baseAddress,
                        let scaleBase = scaleDest.baseAddress,
                        let biasBase = biasDest.baseAddress
                    else { return }
                    DispatchQueue.concurrentPerform(iterations: copies.count) { index in
                        let work = copies[index]
                        do {
                            try work.file.copyIndexedRows(
                                key: work.weight,
                                sourceRows: work.sourceRows,
                                destRows: work.destRows,
                                destinationRowCount: unique.count,
                                destination: packedBase)
                            try work.file.copyIndexedRows(
                                key: work.scales,
                                sourceRows: work.sourceRows,
                                destRows: work.destRows,
                                destinationRowCount: unique.count,
                                destination: scaleBase)
                            try work.file.copyIndexedRows(
                                key: work.biases,
                                sourceRows: work.sourceRows,
                                destRows: work.destRows,
                                destinationRowCount: unique.count,
                                destination: biasBase)
                        } catch {
                            failure.lock()
                            if captured == nil { captured = error }
                            failure.unlock()
                        }
                    }
                }
            }
        }
        if let captured { throw captured }

        let weight = MLXArray(packed, [unique.count, packedCols])
        let scales = MLXArray(scaleBits, [unique.count, scaleCols]).view(dtype: .bfloat16)
        let biases = MLXArray(biasBits, [unique.count, scaleCols]).view(dtype: .bfloat16)
        let values = dequantized(
            weight, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: .affine
        ).asType(.bfloat16)
        return values[MLXArray(inverse)]
    }

    // MARK: Deferred (graph-before-lookup) gather

    /// Packed-row geometry shared by every PLE shard (all shards carry the
    /// same width / group / bits; `mmapGatherBatched` reads it the same way).
    struct MmapGeometry {
        let packedCols: Int
        let scaleCols: Int
        let groupSize: Int
        let bits: Int
    }

    private var cachedGeometry: MmapGeometry?

    private func mmapGeometry() throws -> MmapGeometry {
        if let cachedGeometry { return cachedGeometry }
        try ensureMmapCatalog()
        guard let spec = shardKeys[0] else {
            throw Qwen4ExpPLEError.gather("PLE shard 0 missing from the mmap catalog")
        }
        let file = try file(named: spec.file)
        let weightTensor = try file.tensor(spec.weight)
        let scaleTensor = try file.tensor(spec.scales)
        guard weightTensor.dtype == "U32", weightTensor.shape.count == 2,
            scaleTensor.shape.count == 2, scaleTensor.shape[1] > 0
        else {
            throw Qwen4ExpPLEError.gather("PLE shard 0 is not a packed affine table")
        }
        let geometry = MmapGeometry(
            packedCols: weightTensor.shape[1],
            scaleCols: scaleTensor.shape[1],
            groupSize: tables.headEmbedDim / scaleTensor.shape[1],
            bits: weightTensor.shape[1] * 32 / tables.headEmbedDim)
        cachedGeometry = geometry
        return geometry
    }

    /// One host-writable placeholder: the raw packed rows, scales and biases
    /// the step's dequant reads. Materialized once; the host rewrites the
    /// bytes in place before every submit that uses the slot.
    final class DeferredSlot {
        let rows: Int
        let geometry: MmapGeometry
        let packed: MLXArray
        let scales: MLXArray
        let biases: MLXArray
        let byteCount: Int

        init(rows: Int, geometry: MmapGeometry) {
            self.rows = rows
            self.geometry = geometry
            self.byteCount = Qwen4ExpPLEDeferredBufferPolicy.slotBytes(
                rows: rows, packedCols: geometry.packedCols, scaleCols: geometry.scaleCols)
            self.packed = MLXArray.zeros([rows, geometry.packedCols], dtype: .uint32)
            self.scales = MLXArray.zeros([rows, geometry.scaleCols], dtype: .uint16)
            self.biases = MLXArray.zeros([rows, geometry.scaleCols], dtype: .uint16)
            eval(packed, scales, biases)
            Qwen4ExpPLEResourceMetrics.adjustActiveRowBufferBytes(by: byteCount)
        }

        deinit {
            Qwen4ExpPLEResourceMetrics.adjustActiveRowBufferBytes(by: -byteCount)
        }

        /// Contiguous backing of a materialized placeholder for in-place
        /// host writes. `asData(access: .noCopy)` wraps the array's own
        /// buffer (MLX Metal buffers are shared-storage), so a CPU write here
        /// is what the GPU dequant reads once the step is submitted.
        fileprivate func withMutableBytes<R>(
            of array: MLXArray, _ body: (UnsafeMutableRawPointer) throws -> R
        ) throws -> R {
            let view = array.asData(access: .noCopy)
            let expected = (0 ..< view.shape.count).map { axis in
                view.shape[(axis + 1)...].reduce(1, *)
            }
            guard view.strides == expected, view.data.count == array.nbytes else {
                throw Qwen4ExpPLEError.gather("PLE deferred slot lost its contiguous backing")
            }
            return try view.data.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else {
                    throw Qwen4ExpPLEError.gather("PLE deferred slot has a null buffer")
                }
                return try body(UnsafeMutableRawPointer(mutating: base))
            }
        }
    }

    /// Two slots per row count: the slot a step reads is never the slot the
    /// host is writing for the next step (the write also only happens after
    /// the previous step's token has been read back, which orders it behind
    /// every GPU read of either slot).
    private var deferredSlots: [Int: [DeferredSlot]] = [:]
    private var deferredCursor: [Int: Int] = [:]
    private var deferredLastUse: [Int: UInt64] = [:]
    private var deferredUseClock: UInt64 = 0
    private var deferredCachedBytes = 0

    private func nextDeferredSlot(rows: Int, geometry: MmapGeometry) -> DeferredSlot {
        deferredUseClock &+= 1
        if let slots = deferredSlots[rows] {
            deferredLastUse[rows] = deferredUseClock
            let cursor = (deferredCursor[rows] ?? 1) ^ 1
            deferredCursor[rows] = cursor
            return slots[cursor]
        }

        let slotBytes = Qwen4ExpPLEDeferredBufferPolicy.slotBytes(
            rows: rows, packedCols: geometry.packedCols, scaleCols: geometry.scaleCols)
        let (pairBytes, pairOverflow) = slotBytes.multipliedReportingOverflow(by: 2)
        // A width whose double buffer exceeds the process budget is one-shot.
        // The returned graph/fill closure owns this fresh slot, so it cannot
        // race a previous GPU reader and is released with that request.
        guard !pairOverflow, pairBytes <= deferredByteBudget else {
            return DeferredSlot(rows: rows, geometry: geometry)
        }

        while deferredCachedBytes > deferredByteBudget - pairBytes,
            let victim = deferredLastUse.min(by: { $0.value < $1.value })?.key
        {
            let removed = deferredSlots.removeValue(forKey: victim) ?? []
            deferredCursor[victim] = nil
            deferredLastUse[victim] = nil
            let removedBytes = removed.reduce(0) { $0 + $1.byteCount }
            deferredCachedBytes = max(0, deferredCachedBytes - removedBytes)
            Qwen4ExpPLEResourceMetrics.adjustCachedRowBufferBytes(by: -removedBytes)
        }

        let slots = [
            DeferredSlot(rows: rows, geometry: geometry),
            DeferredSlot(rows: rows, geometry: geometry),
        ]
        deferredSlots[rows] = slots
        deferredLastUse[rows] = deferredUseClock
        deferredCachedBytes += pairBytes
        Qwen4ExpPLEResourceMetrics.adjustCachedRowBufferBytes(by: pairBytes)
        let cursor = (deferredCursor[rows] ?? 1) ^ 1
        deferredCursor[rows] = cursor
        return slots[cursor]
    }

    /// Internal observability for qualification tests. It exposes sizes, not
    /// row ids or model data.
    var deferredBufferCacheSnapshot: (bytes: Int, budget: Int, rowCounts: [Int]) {
        (deferredCachedBytes, deferredByteBudget, deferredSlots.keys.sorted())
    }

    /// mlx-serve 26.9.1 "graph before the n-gram lookup": hand back the
    /// step's `[tokenRows, embedDim]` PLE embeddings as a lazy dequant of a
    /// host-writable slot, plus the fill that copies the resolved rows into
    /// it. Bit-identical to `gatherFlat` — the same `dequantized(...)`
    /// primitive on the same bytes; only `values[inverse]` (a row copy) is
    /// gone because every row is written in place.
    ///
    /// `nil` when the table is resident (no mmap rows to defer) or the mmap
    /// catalog is unavailable; callers then take the eager path.
    func deferredGather(tokenRows: Int) -> (values: MLXArray, fill: ([Int]) -> Void)? {
        guard mmap, tokenRows > 0 else { return nil }
        let geometry: MmapGeometry
        do {
            geometry = try mmapGeometry()
        } catch {
            return nil
        }
        let heads = tables.ngramHeads
        let rows = tokenRows * heads
        let slot = nextDeferredSlot(rows: rows, geometry: geometry)
        let values = dequantized(
            slot.packed,
            scales: slot.scales.view(dtype: .bfloat16),
            biases: slot.biases.view(dtype: .bfloat16),
            groupSize: geometry.groupSize, bits: geometry.bits, mode: .affine
        ).asType(.bfloat16)
        let embeddings = values.reshaped(tokenRows, heads * tables.headEmbedDim)
            * ngramEmbedding.weightScale
        let fill: ([Int]) -> Void = { [self] ids in
            do {
                try self.fillDeferredSlot(slot, ids: ids)
            } catch {
                preconditionFailure("Qwen4 PLE deferred fill failed: \(error)")
            }
        }
        return (embeddings, fill)
    }

    /// Same per-shard concurrent row copy as `mmapGatherBatched`, written
    /// straight into the slot's backing instead of a fresh host array.
    private func fillDeferredSlot(_ slot: DeferredSlot, ids: [Int]) throws {
        guard ids.count == slot.rows else {
            throw Qwen4ExpPLEError.gather(
                "PLE deferred fill got \(ids.count) ids for a \(slot.rows)-row slot")
        }
        var sourceByShard: [Int: [Int]] = [:]
        var destByShard: [Int: [Int]] = [:]
        for (dest, id) in ids.enumerated() {
            guard id >= 0, id < tables.shardOffsets[tables.shardCount] else {
                throw Qwen4ExpPLEError.gather("PLE id \(id) outside the padded vocabulary")
            }
            let shard = tables.shardIndex(for: id)
            sourceByShard[shard, default: []].append(id - tables.shardOffsets[shard])
            destByShard[shard, default: []].append(dest)
        }
        struct ShardCopy: @unchecked Sendable {
            let file: Qwen4ExpSafeTensorMMap
            let weight: String
            let scales: String
            let biases: String
            let sourceRows: [Int]
            let destRows: [Int]
        }
        var copies: [ShardCopy] = []
        for shard in sourceByShard.keys.sorted() {
            guard let spec = shardKeys[shard], let sourceRows = sourceByShard[shard],
                let destRows = destByShard[shard]
            else {
                throw Qwen4ExpPLEError.gather("no mmap spec for shard \(shard)")
            }
            copies.append(
                ShardCopy(
                    file: try file(named: spec.file), weight: spec.weight,
                    scales: spec.scales, biases: spec.biases,
                    sourceRows: sourceRows, destRows: destRows))
        }
        let failure = NSLock()
        nonisolated(unsafe) var captured: Error?
        try slot.withMutableBytes(of: slot.packed) { packedBase in
            try slot.withMutableBytes(of: slot.scales) { scaleBase in
                try slot.withMutableBytes(of: slot.biases) { biasBase in
                    let packedDest = packedBase.assumingMemoryBound(to: UInt32.self)
                    let scaleDest = scaleBase.assumingMemoryBound(to: UInt16.self)
                    let biasDest = biasBase.assumingMemoryBound(to: UInt16.self)
                    DispatchQueue.concurrentPerform(iterations: copies.count) { index in
                        let work = copies[index]
                        do {
                            try work.file.copyIndexedRows(
                                key: work.weight, sourceRows: work.sourceRows,
                                destRows: work.destRows, destinationRowCount: slot.rows,
                                destination: packedDest)
                            try work.file.copyIndexedRows(
                                key: work.scales, sourceRows: work.sourceRows,
                                destRows: work.destRows, destinationRowCount: slot.rows,
                                destination: scaleDest)
                            try work.file.copyIndexedRows(
                                key: work.biases, sourceRows: work.sourceRows,
                                destRows: work.destRows, destinationRowCount: slot.rows,
                                destination: biasDest)
                        } catch {
                            failure.lock()
                            if captured == nil { captured = error }
                            failure.unlock()
                        }
                    }
                }
            }
        }
        if let captured { throw captured }
    }

    private func mmapGatherSerial(_ ids: [Int]) throws -> MLXArray {
        let grouped = Dictionary(grouping: ids.enumerated()) { tables.shardIndex(for: $0.element) }
        var chunks: [MLXArray] = []
        var positions: [Int32] = []
        chunks.reserveCapacity(grouped.count)
        positions.reserveCapacity(ids.count)
        for (shard, items) in grouped.sorted(by: { $0.key < $1.key }) {
            guard let spec = shardKeys[shard] else {
                throw Qwen4ExpPLEError.gather("no mmap spec for shard \(shard)")
            }
            let file = try file(named: spec.file)
            let local = items.map { $0.element - tables.shardOffsets[shard] }
            let packed = try file.rowsUInt32(key: spec.weight, rows: local)
            let scaleBits = try file.rowsUInt16(key: spec.scales, rows: local)
            let biasBits = try file.rowsUInt16(key: spec.biases, rows: local)
            let tensor = try file.tensor(spec.weight)
            let scaleTensor = try file.tensor(spec.scales)
            let packedCols = tensor.shape[1]
            let scaleCols = scaleTensor.shape[1]
            let groupSize = tables.headEmbedDim / scaleCols
            let bits = packedCols * 32 / tables.headEmbedDim
            let weight = MLXArray(packed).reshaped(local.count, packedCols)
            let scales = MLXArray(scaleBits, [local.count, scaleCols]).view(dtype: .bfloat16)
            let biases = MLXArray(biasBits, [local.count, scaleCols]).view(dtype: .bfloat16)
            chunks.append(
                dequantized(
                    weight, scales: scales, biases: biases,
                    groupSize: groupSize, bits: bits, mode: .affine
                ).asType(.bfloat16))
            positions.append(contentsOf: items.map { Int32($0.offset) })
        }
        return scatterGatheredRows(chunks, positions: positions, count: ids.count)
    }

    /// Fusion #3372-style one write: shard reads stay per-file, the destination
    /// scatter is a single indexed update instead of one GPU write per shard.
    private func scatterGatheredRows(
        _ chunks: [MLXArray], positions: [Int32], count: Int
    ) -> MLXArray {
        var result = MLXArray.zeros([count, tables.headEmbedDim], dtype: .bfloat16)
        guard !chunks.isEmpty else { return result }
        result[MLXArray(positions)] = concatenated(chunks, axis: 0)
        return result
    }

    private func ensureMmapCatalog() throws {
        if mmapReady { return }
        guard let directory = Qwen4ExpPLEResidency.resolvedModelDirectory else {
            throw Qwen4ExpPLEError.missingModelDirectory
        }
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        let data = try Data(contentsOf: indexURL)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let weightMap = object["weight_map"] as? [String: String]
        else {
            throw Qwen4ExpPLEError.safetensors("index.json missing weight_map")
        }
        var catalog: [Int: (weight: String, scales: String, biases: String, file: String)] = [:]
        var expectedGeometry: MmapGeometry?
        for shard in 0 ..< tables.shardCount {
            let bases = [
                "\(weightPrefix).shards.\(shard)",
                "\(weightPrefix).shard_\(shard)",
                "model.language_model.layers.\(layerIndex).ple.ple_embedding.ngram_embedding.shards.\(shard)",
                "model.language_model.layers.\(layerIndex).ple.ple_embedding.ngram_embedding.shard_\(shard)",
            ]
            guard let base = bases.first(where: { weightMap["\($0).weight"] != nil }) else {
                throw Qwen4ExpPLEError.safetensors("PLE shard \(shard) missing from weight_map")
            }
            let weightKey = "\(base).weight"
            let scalesKey = "\(base).scales"
            let biasesKey = "\(base).biases"
            guard let fileName = weightMap[weightKey],
                let scalesFile = weightMap[scalesKey],
                let biasesFile = weightMap[biasesKey]
            else {
                throw Qwen4ExpPLEError.safetensors("incomplete affine PLE tensors for \(base)")
            }
            guard scalesFile == fileName, biasesFile == fileName else {
                throw Qwen4ExpPLEError.safetensors(
                    "PLE affine tensors for shard \(shard) span multiple files")
            }

            let mapped = try file(named: fileName)
            let weight = try mapped.tensor(weightKey)
            let scales = try mapped.tensor(scalesKey)
            let biases = try mapped.tensor(biasesKey)
            let rows = tables.shardSizes[shard]
            guard weight.dtype == "U32", scales.dtype == "BF16", biases.dtype == "BF16",
                weight.shape.count == 2, scales.shape.count == 2,
                biases.shape == scales.shape,
                weight.shape[0] == rows, scales.shape[0] == rows,
                weight.shape[1] > 0, scales.shape[1] > 0,
                tables.headEmbedDim % scales.shape[1] == 0
            else {
                throw Qwen4ExpPLEError.safetensors(
                    "PLE shard \(shard) does not match exact affine geometry")
            }
            let (packedBits, packedOverflow) = weight.shape[1].multipliedReportingOverflow(by: 32)
            guard !packedOverflow, packedBits % tables.headEmbedDim == 0 else {
                throw Qwen4ExpPLEError.safetensors("PLE packed affine width overflows or is misaligned")
            }
            let geometry = MmapGeometry(
                packedCols: weight.shape[1],
                scaleCols: scales.shape[1],
                groupSize: tables.headEmbedDim / scales.shape[1],
                bits: packedBits / tables.headEmbedDim)
            if let expectedGeometry {
                guard geometry.packedCols == expectedGeometry.packedCols,
                    geometry.scaleCols == expectedGeometry.scaleCols,
                    geometry.groupSize == expectedGeometry.groupSize,
                    geometry.bits == expectedGeometry.bits
                else {
                    throw Qwen4ExpPLEError.safetensors(
                        "PLE shard \(shard) geometry differs from shard 0")
                }
            } else {
                expectedGeometry = geometry
            }
            catalog[shard] = (
                weight: weightKey,
                scales: scalesKey,
                biases: biasesKey,
                file: fileName)
        }
        shardKeys = catalog
        cachedGeometry = expectedGeometry
        mmapReady = true
    }

    private func file(named name: String) throws -> Qwen4ExpSafeTensorMMap {
        if let existing = mmapFiles[name] { return existing }
        guard let directory = Qwen4ExpPLEResidency.resolvedModelDirectory else {
            throw Qwen4ExpPLEError.missingModelDirectory
        }
        let root = directory.standardizedFileURL
        let url = root.appendingPathComponent(name).standardizedFileURL
        guard url.path.hasPrefix(root.path + "/") else {
            throw Qwen4ExpPLEError.safetensors(
                "PLE shard path escapes the resolved model directory")
        }
        let opened = try Qwen4ExpSafeTensorMMap(url: url)
        mmapFiles[name] = opened
        return opened
    }

    /// Open and validate every exact affine shard without reading row payloads.
    /// Resident PLE has no external resources to validate.
    func validateExternalResources() throws {
        guard mmap else { return }
        try ensureMmapCatalog()
    }

    /// Unmap PLE safetensors and release every reusable row buffer. Call only
    /// after the serving engine has drained; `deinit` is the final backstop.
    func releaseExternalResources() {
        mmapFiles.removeAll(keepingCapacity: false)
        shardKeys.removeAll(keepingCapacity: false)
        mmapReady = false
        cachedGeometry = nil
        Qwen4ExpPLEResourceMetrics.adjustCachedRowBufferBytes(by: -deferredCachedBytes)
        deferredSlots.removeAll(keepingCapacity: false)
        deferredCursor.removeAll(keepingCapacity: false)
        deferredLastUse.removeAll(keepingCapacity: false)
        deferredCachedBytes = 0
    }

    deinit {
        releaseExternalResources()
    }
}

// MARK: - PLE layer (Fusion Qwen4ExpPLELayer)

final class Qwen4ExpPLELayer: Module {
    let tables: Qwen4ExpNGramTables
    let hcCount: Int
    let hiddenSize: Int
    let convDilation: Int
    let shortConvStateLen: Int
    let modelLayerIndex: Int

    @ModuleInfo(key: "ple_embedding") var pleEmbedding: Qwen4ExpNGramEmbedding
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "norm_key") var normKey: Qwen4ExpRMSNorm
    @ModuleInfo(key: "norm_query") var normQuery: Qwen4ExpRMSNorm
    @ModuleInfo(key: "norm_conv") var normConv: Qwen4ExpRMSNorm
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d

    init(_ args: Qwen4ExpTextConfiguration, layerIndex: Int, pleIndex: Int, mmap: Bool) {
        self.tables = Qwen4ExpNGramTables(args, pleIndex: pleIndex)
        self.hcCount = args.hcCount
        self.hiddenSize = args.hiddenSize
        self.convDilation = args.ngramSize
        self.shortConvStateLen = (args.pleConvKernelSize - 1) * args.ngramSize
        self.modelLayerIndex = layerIndex
        let hcHidden = args.hcCount * args.hiddenSize
        _pleEmbedding.wrappedValue = Qwen4ExpNGramEmbedding(
            args, layerIndex: layerIndex, pleIndex: pleIndex, mmap: mmap)
        _keyProj.wrappedValue = Linear(args.pleEmbedDim, hcHidden, bias: false)
        _valueProj.wrappedValue = Linear(args.pleEmbedDim, args.hiddenSize, bias: false)
        _normKey.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: hcHidden, eps: args.rmsNormEps, groupSize: args.hiddenSize)
        _normQuery.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: hcHidden, eps: args.rmsNormEps, groupSize: args.hiddenSize)
        _normConv.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: hcHidden, eps: args.rmsNormEps, groupSize: args.hiddenSize)
        _conv1d.wrappedValue = Conv1d(
            inputChannels: hcHidden,
            outputChannels: hcHidden,
            kernelSize: args.pleConvKernelSize,
            stride: 1,
            padding: 0,
            dilation: args.ngramSize,
            groups: hcHidden,
            bias: false)
        super.init()
    }

    func callAsFunction(
        _ hidden: MLXArray, inputIds: MLXArray, cache: ArraysCache? = nil
    ) -> MLXArray {
        let batch = hidden.dim(0)
        let width = inputIds.dim(1)
        let tokens = inputIds.asType(.int64)
        let history = concatenated([legacyContext(cache: cache, batch: batch), tokens], axis: 1)
        let embeddings = embedGPU(hidden: hidden, history: history, inputWidth: width)
        if let cache {
            let rows = tables.contextLen == 0 ? [[Int]](repeating: [], count: batch)
                : hostTokenRows(history[0..., (-tables.contextLen)...])
            cache[0] = MLXArray(rows.flatMap { $0.map { Int64($0) } })
                .reshaped([batch, tables.contextLen])
        }
        let (gated, normed) = mix(hidden: hidden, embeddings: embeddings, mask: nil)
        let convState =
            cache?[1]
            ?? MLXArray.zeros([batch, shortConvStateLen, hidden.dim(-1)], dtype: hidden.dtype)
        let convOut = shortConv(normed, state: convState)
        let convInput = concatenated([convState, normed], axis: 1)
        cache?[1] = shortConvStateLen == 0
            ? MLXArray.zeros([batch, 0, hidden.dim(-1)], dtype: hidden.dtype)
            : convInput[0..., (-shortConvStateLen)..., 0...]
        if let cache { cache.offset += width }
        return gated + convOut
    }

    func cbv2Forward(
        _ hidden: MLXArray,
        inputIds: MLXArray,
        recurrentState: [CBv2RecurrentStateEvaluation],
        captureRecurrentWindow: Bool = false
    ) -> MLXArray {
        if captureRecurrentWindow {
            return cbv2ForwardCaptured(
                hidden, inputIds: inputIds, recurrentState: recurrentState)
        }
        let batch = hidden.dim(0)
        precondition(recurrentState.count == batch, "Qwen4 PLE CBv2 row count mismatch")
        let width = inputIds.dim(1)
        let tokens = inputIds.asType(.int64)
        let pleIndex = Qwen4ExpNGramGeometry.recurrentLayerIndex(modelLayerIndex)
        var previousRows: [MLXArray] = []
        var convRows: [MLXArray] = []
        previousRows.reserveCapacity(batch)
        convRows.reserveCapacity(batch)
        for evaluation in recurrentState {
            let state = evaluation.inputState(modelLayerIndex: pleIndex)
            previousRows.append(contextFromSSM(state?.ssm))
            convRows.append(
                state?.conv
                    ?? MLXArray.zeros(
                        [1, shortConvStateLen, hidden.dim(-1)], dtype: hidden.dtype))
        }
        let previous = previousRows.count == 1 ? previousRows[0] : concatenated(previousRows, axis: 0)
        let history = concatenated([previous, tokens], axis: 1)
        let embeddings: MLXArray
        if width == 1, Qwen4ExpPLEDeferred.isEnabled(),
            let scope = CBv2DeferredHostFill.current,
            let deferred = pleEmbedding.deferredGather(tokenRows: batch)
        {
            // Graph before the lookup: the step's PLE rows come from a
            // host-writable slot the dequant already references, so this
            // forward never reads the (still lazy) input token. The fill
            // below runs after the whole step graph is built.
            embeddings = deferred.values.reshaped(batch, 1, tables.embedDim).asType(hidden.dtype)
            let tables = self.tables
            let ssmInputs = recurrentState.map {
                $0.inputState(modelLayerIndex: pleIndex)?.ssm
            }
            scope.register {
                let newest = inputIds.reshaped(-1).asArray(Int32.self)
                var flat: [Int] = []
                flat.reserveCapacity(batch * tables.ngramHeads)
                for row in 0 ..< batch {
                    let context = Self.hostContext(ssmInputs[row], tables: tables)
                    flat.append(contentsOf:
                        Qwen4ExpNGramIDs.ids(
                            history: context + [Int(newest[row])], inputWidth: 1, tables: tables
                        )[0])
                }
                deferred.fill(flat)
            }
            Qwen4ExpPLEDeferredInvocation.recordDeferred()
        } else {
            embeddings = embedGPU(hidden: hidden, history: history, inputWidth: width)
            Qwen4ExpPLEDeferredInvocation.recordEager()
        }
        Qwen4ExpDecodeProfile.dumpTensor("ple.history", layer: modelLayerIndex, history)
        Qwen4ExpDecodeProfile.dumpTensor("ple.embeddings", layer: modelLayerIndex, embeddings)
        let (gated, normed) = mix(hidden: hidden, embeddings: embeddings, mask: nil)
        let convState = convRows.count == 1 ? convRows[0] : concatenated(convRows, axis: 0)
        Qwen4ExpDecodeProfile.dumpTensor("ple.convState", layer: modelLayerIndex, convState)
        Qwen4ExpDecodeProfile.dumpTensor("ple.gated", layer: modelLayerIndex, gated)
        Qwen4ExpDecodeProfile.dumpTensor("ple.normed", layer: modelLayerIndex, normed)
        let convOut = shortConv(normed, state: convState)
        Qwen4ExpDecodeProfile.dumpTensor("ple.convOut", layer: modelLayerIndex, convOut)
        let convInput = concatenated([convState, normed], axis: 1)
        let newConv = convInput[0..., (-shortConvStateLen)..., 0...]
        let nextSSM = history[0..., (-tables.contextLen)...]
            .asType(.float32)
            .reshaped([batch, 1, 1, tables.contextLen])
        for (row, evaluation) in recurrentState.enumerated() {
            do {
                try evaluation.stage(
                    modelLayerIndex: pleIndex,
                    conv: newConv[row ..< row + 1],
                    ssm: nextSSM[row ..< row + 1])
            } catch {
                preconditionFailure(
                    "Qwen4 PLE CBv2 stage failed at layer \(modelLayerIndex): \(error)")
            }
        }
        return gated + convOut
    }

    /// Per-position conv/SSM stacks for Lightning MTP rectangular verify.
    /// Same recurrence as `cbv2Forward`; commit picks the accepted column.
    func cbv2ForwardCaptured(
        _ hidden: MLXArray,
        inputIds: MLXArray,
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        let batch = hidden.dim(0)
        let width = inputIds.dim(1)
        precondition(recurrentState.count == batch, "Qwen4 PLE CBv2 row count mismatch")
        precondition(width >= 1, "Qwen4 PLE capture-verify window must be non-empty")
        let tokens = inputIds.asType(.int64)
        let pleIndex = Qwen4ExpNGramGeometry.recurrentLayerIndex(modelLayerIndex)
        var previousRows: [MLXArray] = []
        var convRows: [MLXArray] = []
        previousRows.reserveCapacity(batch)
        convRows.reserveCapacity(batch)
        for evaluation in recurrentState {
            let state = evaluation.inputState(modelLayerIndex: pleIndex)
            previousRows.append(contextFromSSM(state?.ssm))
            convRows.append(
                state?.conv
                    ?? MLXArray.zeros(
                        [1, shortConvStateLen, hidden.dim(-1)], dtype: hidden.dtype))
        }
        let previous = previousRows.count == 1 ? previousRows[0] : concatenated(previousRows, axis: 0)
        let history = concatenated([previous, tokens], axis: 1)
        let embeddings: MLXArray
        if Qwen4ExpPLEDeferred.isEnabled(),
            let scope = CBv2DeferredHostFill.current,
            let deferred = pleEmbedding.deferredGather(tokenRows: batch * width)
        {
            // Lightning verify window: the column tokens are the lazy draft
            // chain. The eager path evaluated them here, mid-build, which
            // serialised the round (drafts on the GPU, then ~6 ms of host
            // gather, then the rest of the build). Same deferred slot as
            // T=1 decode, `batch * width` rows; the fill runs after the
            // whole round graph is built, once the drafts are already in
            // flight. Host ids are Fusion's int64 n-gram math, bit-identical
            // to `gpuIds` (Qwen4ExpPLEDeferredTests).
            embeddings = deferred.values
                .reshaped(batch, width, tables.embedDim).asType(hidden.dtype)
            let tables = self.tables
            let ssmInputs = recurrentState.map {
                $0.inputState(modelLayerIndex: pleIndex)?.ssm
            }
            scope.register {
                let newest = inputIds.reshaped(-1).asArray(Int32.self)
                var flat: [Int] = []
                flat.reserveCapacity(batch * width * tables.ngramHeads)
                for row in 0 ..< batch {
                    let context = Self.hostContext(ssmInputs[row], tables: tables)
                    let window = (0 ..< width).map { Int(newest[row * width + $0]) }
                    for ids in Qwen4ExpNGramIDs.ids(
                        history: context + window, inputWidth: width, tables: tables)
                    {
                        flat.append(contentsOf: ids)
                    }
                }
                deferred.fill(flat)
            }
            Qwen4ExpPLEDeferredInvocation.recordDeferred()
        } else {
            embeddings = embedGPU(hidden: hidden, history: history, inputWidth: width)
            Qwen4ExpPLEDeferredInvocation.recordEager()
        }
        Qwen4ExpDecodeProfile.dumpTensor("ple.history", layer: modelLayerIndex, history)
        Qwen4ExpDecodeProfile.dumpTensor("ple.embeddings", layer: modelLayerIndex, embeddings)
        let (gated, normed) = mix(hidden: hidden, embeddings: embeddings, mask: nil)
        let convState = convRows.count == 1 ? convRows[0] : concatenated(convRows, axis: 0)
        Qwen4ExpDecodeProfile.dumpTensor("ple.convState", layer: modelLayerIndex, convState)
        Qwen4ExpDecodeProfile.dumpTensor("ple.gated", layer: modelLayerIndex, gated)
        Qwen4ExpDecodeProfile.dumpTensor("ple.normed", layer: modelLayerIndex, normed)
        let convOut = shortConv(normed, state: convState)
        Qwen4ExpDecodeProfile.dumpTensor("ple.convOut", layer: modelLayerIndex, convOut)
        let convInput = concatenated([convState, normed], axis: 1)
        for (row, evaluation) in recurrentState.enumerated() {
            let convStack = concatenated(
                (0 ..< width).map { position in
                    convInput[
                        row ..< (row + 1),
                        (position + 1) ..< (position + 1 + shortConvStateLen),
                        0...]
                }, axis: 0)
            let ssmStack = concatenated(
                (0 ..< width).map { position in
                    history[
                        row ..< (row + 1),
                        (position + 1) ..< (position + 1 + tables.contextLen)]
                    .asType(.float32)
                    .reshaped([1, 1, 1, tables.contextLen])
                }, axis: 0)
            do {
                try evaluation.stageCaptured(
                    modelLayerIndex: pleIndex,
                    conv: convStack, ssm: ssmStack, positions: width)
            } catch {
                preconditionFailure(
                    "Qwen4 PLE CBv2 captured stage failed at layer \(modelLayerIndex): \(error)")
            }
        }
        return gated + convOut
    }

    private func embedGPU(hidden: MLXArray, history: MLXArray, inputWidth: Int) -> MLXArray {
        let ids = Qwen4ExpNGramIDs.gpuIds(
            history: history, inputWidth: inputWidth, tables: tables)
        let gathered = pleEmbedding.gather(ids)
        return gathered.reshaped(hidden.dim(0), hidden.dim(1), tables.embedDim)
            .asType(hidden.dtype)
    }

    private func mix(
        hidden: MLXArray, embeddings: MLXArray, mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        // Exact affine QMV like every other Qwen4 projection: stock
        // QuantizedLinear switches kernels between one row and a verify
        // window (qmv vs qmv_wide), so MTP verify columns would score with
        // different key/value bits than serial decode.
        let keys = normKey(Qwen4ExpAffineQMM.apply(keyProj, embeddings)).reshaped(
            hidden.dim(0), hidden.dim(1), hcCount, hiddenSize)
        let values = Qwen4ExpAffineQMM.apply(valueProj, embeddings)
        let queries = normQuery(hidden).reshaped(
            hidden.dim(0), hidden.dim(1), hcCount, hiddenSize)
        var gate = (keys * queries).sum(axis: -1, keepDims: true)
        gate = gate / sqrt(Float(hiddenSize))
        // Fusion `mx.maximum(mx.abs(gate), 1e-6)`: the Python scalar is weak
        // and takes the gate's dtype. A bare `MLXArray(Float(1e-6))` is a
        // strong float32 scalar that promoted the whole PLE output — and the
        // residual entering the next hyper-connection — to float32, so that
        // layer computed at a different precision than the qwen4 reference
        // and the HC hybrid kernel declined its bf16-only shape there.
        let floor = MLXArray(Float(1e-6)).asType(gate.dtype)
        gate = MLX.sign(gate) * sqrt(MLX.maximum(MLX.abs(gate), floor))
        var gated = sigmoid(gate) * values.expandedDimensions(axis: -2)
        gated = gated.reshaped(hidden.shape)
        var normed = normConv(gated)
        if let mask, mask.ndim == 2 {
            let keep = mask.expandedDimensions(axis: -1)
            gated = MLX.where(keep, gated, MLXArray.zeros(like: gated))
            normed = MLX.where(keep, normed, MLXArray.zeros(like: normed))
        }
        return (gated, normed)
    }

    private func shortConv(_ x: MLXArray, state: MLXArray) -> MLXArray {
        silu(conv1d(concatenated([state, x], axis: 1)))
    }

    private func eosContext(batch: Int) -> MLXArray {
        MLXArray.full(
            [batch, tables.contextLen],
            values: MLXArray(Int64(tables.eosTokenId)),
            type: Int64.self)
    }

    private func legacyContext(cache: ArraysCache?, batch: Int) -> MLXArray {
        guard let history = cache?[0] else { return eosContext(batch: batch) }
        precondition(history.shape == [batch, tables.contextLen], "Qwen4 PLE history shape mismatch")
        return history.asType(.int64)
    }

    /// CBv2 stores PLE history as float32 `[1,1,1,contextLen]` (exact for token ids).
    private func contextFromSSM(_ tensor: MLXArray?) -> MLXArray {
        guard let tensor else { return eosContext(batch: 1) }
        return tensor.asType(.int32).reshaped([1, tables.contextLen]).asType(.int64)
    }

    /// Host twin of `contextFromSSM`: the same float32 → int truncation,
    /// EOS-filled when the row has no committed history yet.
    static func hostContext(_ tensor: MLXArray?, tables: Qwen4ExpNGramTables) -> [Int] {
        guard let tensor else {
            return [Int](repeating: tables.eosTokenId, count: tables.contextLen)
        }
        let values = tensor.asArray(Float32.self)
        precondition(values.count == tables.contextLen, "PLE history row width mismatch")
        return values.map { Int(Int32($0)) }
    }

    func validateExternalResources() throws {
        try pleEmbedding.validateExternalResources()
    }

    /// Called after the engine drains, before the model container is dropped.
    /// Mutable request state belongs to the caller's cache, not this module.
    func releaseExternalResources() {
        pleEmbedding.releaseExternalResources()
    }

    private func hostTokenRows(_ tokens: MLXArray) -> [[Int]] {
        let ids = tokens.asType(.int64)
        eval(ids)
        let batch = ids.dim(0)
        let length = ids.dim(1)
        let flat = ids.asArray(Int64.self)
        return (0 ..< batch).map { row in
            (0 ..< length).map { col in Int(flat[row * length + col]) }
        }
    }
}
