// Copyright © 2026 Eigen Labs.

import Darwin
import Foundation

/// Page-aligned tensor payload reads for an F_NOCACHE/F_RDAHEAD descriptor.
///
/// Darwin only bypasses the unified buffer cache when the file offset, length
/// and destination are page aligned; an unaligned `read` into `Data` is served
/// through the file cache even with F_NOCACHE set. Payloads are therefore read
/// in aligned windows into one bounded anonymous scratch extent, and only the
/// selected bytes are copied into the returned `Data`. Mirrors DarkBloom's
/// shared checkpoint reader without depending on it.
enum GLM5NextAlignedPayloadReader {
    typealias Read = (Int32, UnsafeMutableRawPointer, Int, off_t) -> Int

    static let alignmentBytes = 16_384
    static let maximumScratchRequestBytes = 8 * 1024 * 1024
    /// Host allowance callers must price on top of the returned `Data`: one
    /// scratch extent plus one alignment page.
    static let scratchAllowanceBytes = maximumScratchRequestBytes + alignmentBytes

    struct Plan: Equatable {
        let selectedStart: Int, selectedEnd: Int, alignedStart: Int, alignedEnd: Int
        let scratchBytes: Int

        init(fileSize: Int, offset: Int, count: Int) throws {
            guard fileSize >= 0, offset >= 0, offset <= fileSize, count > 0,
                count <= fileSize - offset
            else { throw GLM5NextArtifactError.invalid("payload read range") }
            let a = GLM5NextAlignedPayloadReader.alignmentBytes
            selectedStart = offset
            selectedEnd = offset + count
            alignedStart = offset - offset % a
            let remainder = selectedEnd % a
            let (end, overflow) = selectedEnd.addingReportingOverflow(
                remainder == 0 ? 0 : a - remainder)
            guard !overflow else { throw GLM5NextArtifactError.invalid("payload read range") }
            alignedEnd = end
            scratchBytes = min(
                GLM5NextAlignedPayloadReader.maximumScratchRequestBytes, alignedEnd - alignedStart)
        }
    }

    /// Reads exactly `count` bytes at `offset` from the caller's open
    /// descriptor. `checkUnchanged` runs before and after every read call; the
    /// scratch is unmapped on every exit path, before the caller builds an array.
    static func read(
        descriptor: Int32, fileSize: Int, offset: Int, count: Int,
        checkUnchanged: () throws -> Void,
        readCall: Read = { Darwin.pread($0, $1, $2, $3) }
    ) throws -> Data {
        let plan = try Plan(fileSize: fileSize, offset: offset, count: count)
        guard descriptor >= 0, Int(getpagesize()) <= alignmentBytes,
            alignmentBytes % Int(getpagesize()) == 0
        else { throw GLM5NextArtifactError.invalid("payload read descriptor or page size") }
        try checkUnchanged()
        // Anonymous private pages only (fd=-1), never a mapping of the payload file.
        let allocation = mmap(
            nil, plan.scratchBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0)
        guard allocation != MAP_FAILED, let scratch = allocation,
            UInt(bitPattern: scratch) % UInt(alignmentBytes) == 0
        else {
            if let allocation, allocation != MAP_FAILED { munmap(allocation, plan.scratchBytes) }
            throw GLM5NextArtifactError.invalid("payload read scratch")
        }
        defer { munmap(scratch, plan.scratchBytes) }
        var data = Data(count: count)
        try data.withUnsafeMutableBytes { destination in
            var current = plan.alignedStart
            var written = 0
            while current < plan.alignedEnd {
                try Task.checkCancellation()
                let requested = min(plan.scratchBytes, plan.alignedEnd - current)
                let expected = min(requested, fileSize - current)
                try checkUnchanged()
                let n = readCall(descriptor, scratch, requested, off_t(current))
                if n < 0 && errno == EINTR { continue }  // same aligned pointer, offset, length
                // Only end-of-file may shorten a window; any other short read fails.
                guard n > 0, n == expected else {
                    throw GLM5NextArtifactError.invalid("truncated read")
                }
                try checkUnchanged()
                let lower = max(current, plan.selectedStart)
                let upper = min(current + n, plan.selectedEnd)
                guard lower < upper, lower == plan.selectedStart + written else {
                    throw GLM5NextArtifactError.invalid("payload read frontier")
                }
                destination.baseAddress!.advanced(by: written).copyMemory(
                    from: scratch.advanced(by: lower - current), byteCount: upper - lower)
                written += upper - lower
                current += requested
            }
            guard written == count else { throw GLM5NextArtifactError.invalid("truncated read") }
        }
        return data
    }
}
