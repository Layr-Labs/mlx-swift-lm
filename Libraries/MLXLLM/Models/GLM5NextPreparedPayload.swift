// Copyright © 2026 Eigen Labs.

import Darwin
import Foundation
import MLX

/// Payloads `GLM5NextArtifactLoader.prepare(plan:includeMTP:)` read and
/// verified in its single pass over the artifact, each already in the shape
/// the loading rank keeps and in its final buffer. Construction takes each
/// exactly once (`GLM5NextTensorLoader.prepared`): a tensor by its name, a
/// routed-expert bank by its member names in expert order.
public final class GLM5NextPreparedParts {
    private var tensors: [String: MLXArray]
    private var banks: [[String]: MLXArray]
    /// Every stored tensor construction has taken, bank members included.
    public private(set) var taken: Set<String> = []

    init(tensors: [String: MLXArray], banks: [[String]: MLXArray]) {
        self.tensors = tensors
        self.banks = banks
    }

    /// Arrays not yet taken (a complete construction leaves none).
    public var remaining: Int { tensors.count + banks.count }

    func take(_ name: String) throws -> MLXArray {
        guard let value = tensors.removeValue(forKey: name) else {
            throw GLM5NextArtifactError.invalid("unprepared tensor \(name)")
        }
        taken.insert(name)
        return value
    }

    func takeBank(_ names: [String]) throws -> MLXArray {
        guard let value = banks.removeValue(forKey: names) else {
            throw GLM5NextArtifactError.invalid("unprepared expert bank \(names.first ?? "")")
        }
        taken.formUnion(names)
        return value
    }
}

/// One prepared array's bytes before it becomes an `MLXArray`: whole pages of
/// anonymous memory the pass writes from the CPU (so the process is charged
/// for them, as for MLX's own allocations). A page-multiple size becomes the
/// array's Metal buffer without a copy; any other size, or a buffer Metal
/// does not accept, is copied into an MLX allocation and unmapped.
final class GLM5NextPreparedBuffer: @unchecked Sendable {
    let bytes: Int
    let pointer: UnsafeMutableRawPointer
    private let mappedBytes: Int
    private var owned = true

    init(bytes: Int) throws {
        let page = Int(getpagesize())
        guard bytes > 0, bytes <= Int.max - page else {
            throw GLM5NextArtifactError.invalid("prepared buffer size")
        }
        let mapped = (bytes + page - 1) / page * page
        guard
            let allocation = mmap(
                nil, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0),
            allocation != MAP_FAILED
        else { throw GLM5NextArtifactError.limit("prepared buffer of \(bytes) bytes") }
        self.bytes = bytes
        self.mappedBytes = mapped
        self.pointer = allocation
    }

    deinit { if owned { munmap(pointer, mappedBytes) } }

    /// The array over these bytes. Call once, after every byte is written.
    func array(_ shape: [Int], dtype: DType) -> MLXArray {
        guard bytes == mappedBytes else {
            return MLXArray(
                Data(bytesNoCopy: pointer, count: bytes, deallocator: .none), shape, dtype: dtype)
        }
        owned = false
        let (pointer, mapped) = (self.pointer, mappedBytes)
        // MLX wraps the pages as a no-copy Metal buffer and unmaps them when
        // the array is freed, or copies them and unmaps at once.
        return MLXArray(rawPointer: pointer, shape, dtype: dtype) { munmap(pointer, mapped) }
    }
}

/// Where one stored tensor's selected bytes go: `runs` of the payload that
/// starts at file offset `base`, copied in order to `destination`.
struct GLM5NextPreparedSegment: @unchecked Sendable {
    let base: Int
    let runs: GLM5NextByteRuns
    let destination: UnsafeMutableRawPointer

    var start: Int { base + runs.offset }
    var end: Int { base + runs.end }

    /// Copies the selected bytes inside the file window `window` (whose bytes
    /// start at `source`).
    func copy(window: Range<Int>, from source: UnsafeRawPointer) {
        runs.copy(
            window: (window.lowerBound - base) ..< (window.upperBound - base), from: source,
            to: destination)
    }
}

/// The routed-expert banks `GLM5NextTensorLoader.moe` stacks, found by name:
/// `<prefix>.experts.<e>.<projection>.<suffix>` for every expert `e`.
enum GLM5NextPreparedBanks {
    /// Each complete bank's member names in expert order; names outside a
    /// complete bank stay single tensors.
    static func group(_ names: Set<String>, experts: Int) -> [[String]] {
        var groups: [String: [Int: String]] = [:]
        for name in names {
            guard let (key, index) = member(name) else { continue }
            groups[key, default: [:]][index] = name
        }
        return groups.values.compactMap { members in
            guard experts > 0, members.count == experts,
                (0 ..< experts).allSatisfy({ members[$0] != nil })
            else { return nil }
            return (0 ..< experts).map { members[$0]! }
        }
    }

    /// The bank key (`<prefix>.experts.*<rest>`) and expert index of `name`,
    /// or nil: an ASCII name (every inventory name) is read on its bytes, any
    /// other by `referenceMember`, with the same answer.
    static func member(_ name: String) -> (String, Int)? {
        var contiguous = name
        return contiguous.withUTF8 { bytes -> (String, Int)? in
            guard bytes.allSatisfy({ $0 < 0x80 }) else { return referenceMember(name) }
            guard let base = bytes.baseAddress, bytes.count >= marker.count,
                let found = marker.withUnsafeBytes({
                    memmem(base, bytes.count, $0.baseAddress, marker.count)
                })
            else { return nil }
            let digits = base.distance(to: found.assumingMemoryBound(to: UInt8.self)) + marker.count
            var end = digits
            while end < bytes.count && bytes[end] != 0x2E { end += 1 }
            guard end < bytes.count, end > digits,
                bytes[digits ..< end].allSatisfy({ (0x30 ... 0x39).contains($0) }),
                end - digits == 1 || bytes[digits] != 0x30
            else { return nil }
            var index = 0
            for digit in bytes[digits ..< end] {
                let (times, overflow) = index.multipliedReportingOverflow(by: 10)
                let (sum, carry) = times.addingReportingOverflow(Int(digit - 0x30))
                guard !overflow, !carry else { return nil }
                index = sum
            }
            let key =
                String(decoding: bytes[..<digits], as: UTF8.self) + "*"
                + String(decoding: bytes[end...], as: UTF8.self)
            return (key, index)
        }
    }

    /// `member` through Foundation's string search; the definition the byte
    /// reader must agree with.
    static func referenceMember(_ name: String) -> (String, Int)? {
        guard let marker = name.range(of: ".experts.") else { return nil }
        let rest = name[marker.upperBound...]
        guard let dot = rest.firstIndex(of: "."), let index = Int(rest[..<dot]), index >= 0,
            String(index) == rest[..<dot]
        else { return nil }
        return (String(name[..<marker.upperBound]) + "*" + String(rest[dot...]), index)
    }

    private static let marker = Array(".experts.".utf8)
}
