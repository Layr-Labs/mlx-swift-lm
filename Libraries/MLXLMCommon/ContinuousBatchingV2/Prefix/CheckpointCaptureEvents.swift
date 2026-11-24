import Foundation
import os

/// Where a durable checkpoint candidate went: the funnel (prepared, staged,
/// exported) and every graceful refusal in capture, commit and publication.
/// Refusals stay graceful; this only makes them visible, so a store that
/// stays empty says why. Positions and reasons only: no tokens, salts or
/// tensor contents are recorded.
enum CBv2CheckpointCaptureEvent: String, Sendable, CaseIterable {
    // Funnel.
    case prepared
    case staged
    case exported
    // Step eligibility of a prompt range (historical-recurrent donors).
    case requestIneligible
    case noRecurrentEvaluation
    case durableCheckpointNotPermitted
    case noChunkCap
    case geometryDisarmed
    case promptEnd
    // Candidate preparation.
    case layoutMismatch
    case packedInteriorCut
    case storeRefused
    case rowMismatch
    case windowFootprintOverflow
    case slotStagingCap
    case prepareFailed
    // Commit.
    case assistantCaptureRefused
    case stepDiscarded
    case recurrentStateMissing
    case commitFailed
    case duplicateBoundary
    // Publication.
    case publicationNotAllowed
    case exportFailed
    case exportScratchUnavailable
}

/// Counts are taken on refusal and funnel edges only (a lock and a dictionary
/// increment per checkpoint boundary, never per token). The debug log is off
/// unless the subsystem's debug level is enabled.
final class CBv2CheckpointCaptureEvents: @unchecked Sendable {
    private static let log = Logger(subsystem: "darkbloom", category: "CBv2Checkpoint")
    private let lock = NSLock()
    private var counts: [CBv2CheckpointCaptureEvent: Int] = [:]
    private var details: [CBv2CheckpointCaptureEvent: String] = [:]

    /// `detail` is a short numeric note (bytes, caps); never tokens or salts.
    func record(
        _ event: CBv2CheckpointCaptureEvent, position: Int? = nil, error: Error? = nil,
        detail: String? = nil
    ) {
        let at = position ?? -1
        let note = detail ?? error.map { String(describing: $0) }
        lock.withLock {
            counts[event, default: 0] += 1
            if let note { details[event] = "at \(at): \(note)" }
        }
        if let note {
            let detail = note
            Self.log.debug(
                "checkpoint \(event.rawValue, privacy: .public) at \(at, privacy: .public): \(detail, privacy: .public)"
            )
        } else {
            Self.log.debug(
                "checkpoint \(event.rawValue, privacy: .public) at \(at, privacy: .public)")
        }
    }

    var snapshot: [CBv2CheckpointCaptureEvent: Int] { lock.withLock { counts } }
    /// The most recent note per event, for diagnostics.
    var lastDetails: [CBv2CheckpointCaptureEvent: String] { lock.withLock { details } }
}
