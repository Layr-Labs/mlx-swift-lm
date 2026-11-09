// Copyright © 2026 Eigen Labs.

@preconcurrency import AVFoundation
import Foundation

@testable import MLXLMCommon

/// Every failed reader pass, in order. Disarms `fault` when a pass fails and a
/// second pass follows, so that pass reads every byte.
final class FailedPasses: @unchecked Sendable {
    private let lock = NSLock()
    private let fault: VideoDataRequestFault?
    private var recorded: [(failure: VideoReaderFailure, retrying: Bool)] = []

    init(disarming fault: VideoDataRequestFault? = nil) { self.fault = fault }

    var all: [(failure: VideoReaderFailure, retrying: Bool)] { lock.withLock { recorded } }
    var retrying: [Bool] { all.map { $0.retrying } }

    var observer: VideoReaderRetry.Observer {
        { [self] failure, retrying in
            lock.withLock { recorded.append((failure, retrying)) }
            if retrying { fault?.disarm() }
        }
    }
}

/// Cancels the readers in `ending` (1 is the first reader the task opens)
/// before their read number `read`, on the reading thread between two reads.
/// Each then ends `.cancelled`, as a reader that fails partway ends `.failed`.
final class ReaderEnder: @unchecked Sendable {
    private let lock = NSLock()
    private let ending: Set<Int>
    private let read: Int
    private var order = ReaderOrder()
    private var endedCount = 0

    init(ending: Set<Int>, atRead read: Int) {
        self.ending = ending
        self.read = read
    }

    var ended: Int { lock.withLock { endedCount } }

    var hook: @Sendable (AVAssetReader) -> Void {
        { [self] reader in
            let end = lock.withLock { () -> Bool in
                let (ordinal, reads) = order.count(readBy: reader)
                guard reads == read, ending.contains(ordinal) else { return false }
                endedCount += 1
                return true
            }
            if end { reader.cancelReading() }
        }
    }
}

/// The readers that have read, in the order of their first read. It keeps each
/// reader for as long as the order lives. A freed reader's address can go to
/// the next reader, so the `ObjectIdentifier` of a reader that is not kept can
/// name a later reader.
struct ReaderOrder {
    private var readers: [(reader: AVAssetReader, reads: Int)] = []

    /// Counts one read by `reader`. Returns the reader's ordinal (1 is the
    /// first reader) and its reads so far, this read included.
    mutating func count(readBy reader: AVAssetReader) -> (ordinal: Int, reads: Int) {
        let index: Int
        if let seen = readers.firstIndex(where: { $0.reader === reader }) {
            index = seen
        } else {
            readers.append((reader, 0))
            index = readers.count - 1
        }
        readers[index].reads += 1
        return (index + 1, readers[index].reads)
    }
}

/// Runs `body` with `passes` observing every failed reader pass, and with `ender`
/// ending readers partway when one is given.
func withReaderFaults<Output>(
    _ passes: FailedPasses, ending ender: ReaderEnder? = nil,
    isolation: isolated (any Actor)? = #isolation, _ body: () async throws -> Output
) async throws -> Output {
    try await VideoReaderRetry.$failedPassObserver.withValue(passes.observer) {
        try await VideoReaderRetry.$beforeReadForTesting.withValue(ender?.hook, operation: body)
    }
}
