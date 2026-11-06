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
    private var order: [ObjectIdentifier: (ordinal: Int, reads: Int)] = [:]
    private var endedCount = 0

    init(ending: Set<Int>, atRead read: Int) {
        self.ending = ending
        self.read = read
    }

    var ended: Int { lock.withLock { endedCount } }

    var hook: @Sendable (AVAssetReader) -> Void {
        { [self] reader in
            let end = lock.withLock { () -> Bool in
                let id = ObjectIdentifier(reader)
                var entry = order[id] ?? (ordinal: order.count + 1, reads: 0)
                entry.reads += 1
                order[id] = entry
                guard entry.reads == read, ending.contains(entry.ordinal) else { return false }
                endedCount += 1
                return true
            }
            if end { reader.cancelReading() }
        }
    }
}

/// Runs `body` with `passes` observing every failed reader pass, and with `ender`
/// ending readers partway when one is given.
func withReaderFaults<Output>(
    _ passes: FailedPasses, ending ender: ReaderEnder? = nil, _ body: () async throws -> Output
) async throws -> Output {
    try await VideoReaderRetry.$failedPassObserver.withValue(passes.observer) {
        try await VideoReaderRetry.$beforeReadForTesting.withValue(ender?.hook, operation: body)
    }
}
