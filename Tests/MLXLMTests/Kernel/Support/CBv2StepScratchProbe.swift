// Copyright © 2026 Eigen Labs.

import Foundation

@testable import MLXLMCommon

/// Engine-queue state for tests that make a real step scratch reservation
/// fail. `exhaust` holds every byte the admission still grants; `release`,
/// called at the next observed reservation, gives them back, so exactly one
/// reservation attempt is refused by the production admission path.
final class CBv2StepScratchProbe: @unchecked Sendable {
    var armed = false
    var suspendedAt: Int?
    var refusedRequests: Set<CBv2RequestID>?
    var refusedSite: CBv2StepScratchSite?
    /// A reservation was observed while both requests of interest ran.
    var overlapped = false
    /// Observed reservations that included a held-back request while every
    /// request running at the refusal was still live (must stay zero).
    var admittedWhileHeld = 0
    private var held: [CBv2CheckpointReservation] = []

    func exhaust(_ admission: AdmissionV2) {
        var bytes = 1 << 40
        while bytes > 0 {
            if let lease = try? admission.reserveTransient(bytes: bytes) {
                held.append(lease)
            } else {
                bytes /= 2
            }
        }
    }

    func release() { held.removeAll() }
}
