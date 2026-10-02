// Copyright © 2026 Eigen Labs.

/// Optional host policy for selecting a grouped-prefill profile. All amounts
/// are reservations inside the engine grant, never physical-memory estimates.
/// Admission still charges the complete selected workspace to every request.
public struct MiMoV26PrefillMemoryBudget: Sendable {
    public let minimumKVBytes: Int
    public let targetFixedBytesPerRequest: Int

    public init(minimumKVBytes: Int, targetFixedBytesPerRequest: Int) {
        self.minimumKVBytes = minimumKVBytes
        self.targetFixedBytesPerRequest = targetFixedBytesPerRequest
    }

    func admits(
        fixedBytesPerRequest: Int, concurrency: Int,
        capacityBytes: Int, watermarkFraction: Double
    ) -> Bool {
        guard minimumKVBytes >= 0, targetFixedBytesPerRequest >= 0,
            fixedBytesPerRequest >= 0, concurrency > 0, capacityBytes > 0,
            watermarkFraction.isFinite, watermarkFraction >= 0, watermarkFraction < 1
        else { return false }
        let (fixed, sumOverflow) = fixedBytesPerRequest.addingReportingOverflow(
            targetFixedBytesPerRequest)
        let (allRequests, multiplyOverflow) = fixed.multipliedReportingOverflow(by: concurrency)
        guard !sumOverflow, !multiplyOverflow else { return false }
        // Match AdmissionV2's watermark; guard the conversion at Int's limit.
        let watermark = Double(capacityBytes) * watermarkFraction
        guard watermark < Double(Int.max) else { return false }
        let usable = capacityBytes - Int(watermark)
        return minimumKVBytes <= usable && allRequests <= usable - minimumKVBytes
    }
}
