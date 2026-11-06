// Copyright © 2026 Eigen Labs.

#if canImport(AVFoundation)
@preconcurrency import AVFoundation
import CoreMedia
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

    /// Tests only: runs before each sample read with that reader, on the reading
    /// thread, so a test can end a reader partway.
    @TaskLocal public static var beforeReadForTesting: (@Sendable (AVAssetReader) -> Void)?

    public static func run<Output>(
        isolation: isolated (any Actor)? = #isolation, _ pass: () async throws -> Output
    ) async throws -> Output {
        do {
            return try await pass()
        } catch let failure as VideoReaderFailure {
            // A cancelled task does not retry: the reader failed because it was stopped.
            guard !Task.isCancelled else {
                report(failure, retrying: false)
                throw CancellationError()
            }
            report(failure, retrying: true)
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

    /// The next sample of `reader`'s `output`, after the task's test hook.
    public static func nextSample(_ output: AVAssetReaderOutput, of reader: AVAssetReader)
        -> CMSampleBuffer?
    {
        beforeReadForTesting?(reader)
        return output.copyNextSampleBuffer()
    }
}
#endif
