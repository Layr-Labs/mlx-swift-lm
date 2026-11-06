// Copyright © 2026 Eigen Labs.

#if canImport(AVFoundation)
@preconcurrency import AVFoundation
import Foundation
import VideoToolbox

/// An `AVAssetReader` pass that ended in a state other than `.completed`.
///
/// Holds the reader's status and the domain and code of its error and of each
/// underlying error. It never holds media bytes, frames or error descriptions.
public struct VideoReaderFailure: Error, Equatable, Sendable, CustomStringConvertible {
    public struct Code: Equatable, Sendable {
        public let domain: String
        public let code: Int

        public init(domain: String, code: Int) {
            self.domain = domain
            self.code = code
        }
    }

    /// `AVAssetReader.Status.rawValue` when the pass ended.
    public let status: Int
    /// The reader's error first, then its `NSUnderlyingErrorKey` chain.
    public let codes: [Code]

    static let maximumChainDepth = 4

    public init(status: Int, error: (any Error)?) {
        self.status = status
        var codes: [Code] = []
        var next = error.map { $0 as NSError }
        while let current = next, codes.count < Self.maximumChainDepth {
            codes.append(Code(domain: current.domain, code: current.code))
            next = current.userInfo[NSUnderlyingErrorKey] as? NSError
        }
        self.codes = codes
    }

    public init(reader: AVAssetReader) {
        self.init(status: reader.status.rawValue, error: reader.error)
    }

    /// `AVErrorMediaServicesWereReset` and `AVErrorOperationInterrupted`. `AVError.Code`
    /// declares them on iOS only; the codes mean the same in any SDK.
    static let mediaServicesWereReset = -11819
    static let operationInterrupted = -11847

    /// Decoder, hardware and interruption faults of the decoding machine. Another
    /// provider can decode the same bytes, so these refuse as retryable.
    static let transientCodes: [String: Set<Int>] = [
        AVFoundationErrorDomain: [
            AVError.Code.outOfMemory.rawValue,
            AVError.Code.decoderTemporarilyUnavailable.rawValue,
            mediaServicesWereReset,
            operationInterrupted,
        ],
        NSOSStatusErrorDomain: [
            Int(kVTInvalidSessionErr),
            Int(kVTAllocationFailedErr),
            Int(kVTVideoDecoderMalfunctionErr),
            Int(kVTVideoDecoderNotAvailableNowErr),
            Int(kVTVideoDecoderRemovedErr),
        ],
    ]

    /// True when any code in the chain is a known decoder, hardware or
    /// interruption fault. Every other failure is a property of the media.
    public var isTransient: Bool {
        codes.contains { Self.transientCodes[$0.domain]?.contains($0.code) == true }
    }

    public var description: String {
        let chain = codes.map { "\($0.domain) \($0.code)" }.joined(separator: " < ")
        return "video reader status \(status): \(chain.isEmpty ? "no error" : chain)"
    }
}

/// Runs a reader pass a second time, with a new reader over the same asset,
/// when the first pass throws a `VideoReaderFailure`. The second pass decodes
/// the same bytes, so a success returns the same output.
public enum VideoReaderRetry {
    public typealias Observer = @Sendable (_ failure: VideoReaderFailure, _ retrying: Bool) -> Void

    /// Receives every failed pass. `retrying` is true when a second pass follows.
    @TaskLocal public static var failedPassObserver: Observer?

    public static func run<Output>(
        isolation: isolated (any Actor)? = #isolation, _ pass: () async throws -> Output
    ) async throws -> Output {
        do {
            return try await pass()
        } catch let failure as VideoReaderFailure {
            try retrying(after: failure)
            do {
                return try await pass()
            } catch let failure as VideoReaderFailure {
                report(failure, retrying: false)
                throw failure
            }
        }
    }

    /// Reports `failure` to the task's observer.
    public static func report(_ failure: VideoReaderFailure, retrying: Bool) {
        failedPassObserver?(failure, retrying)
    }

    /// A cancelled task does not retry: the reader failed because it was stopped.
    private static func retrying(after failure: VideoReaderFailure) throws {
        guard !Task.isCancelled else {
            report(failure, retrying: false)
            throw CancellationError()
        }
        report(failure, retrying: true)
    }
}

/// Fails data requests of the in-memory video resource loaders, for
/// fault-injection tests. A loader reads `current` when its asset is created;
/// production code never binds it. AVFoundation requests a failed range again,
/// so a reader pass fails only while the fault stays armed.
public final class VideoDataRequestFault: @unchecked Sendable {
    @TaskLocal public static var current: VideoDataRequestFault?

    private let offset: Int
    private let error: any Error & Sendable
    private let lock = NSLock()
    private var armed: Bool
    private var count = 0

    /// While armed, fails every data request whose range holds byte `offset`,
    /// after serving the bytes before it.
    public init(offset: Int, error: any Error & Sendable, armed: Bool = true) {
        self.offset = offset
        self.error = error
        self.armed = armed
    }

    public func arm() { lock.withLock { armed = true } }
    public func disarm() { lock.withLock { armed = false } }

    /// Requests failed so far.
    public var injected: Int { lock.withLock { count } }

    /// The bytes to serve before failing a request for `range`, and the error to
    /// fail it with; nil to serve the request in full.
    public func intercept(_ range: Range<Int>) -> (served: Range<Int>, error: any Error)? {
        guard range.contains(offset) else { return nil }
        return lock.withLock {
            guard armed else { return nil }
            count += 1
            return (range.lowerBound ..< offset, error)
        }
    }
}
#endif
