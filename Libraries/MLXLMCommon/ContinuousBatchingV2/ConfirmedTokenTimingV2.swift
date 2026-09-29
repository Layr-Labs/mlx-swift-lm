import Foundation

/// Committed tokens at the engine's existing readback boundary. MTP bursts
/// record accepted tokens, never draft proposals. Row ordinals are local to the
/// observation; the relative clock starts with its first confirmed token.
public struct CBv2ConfirmedTokenTiming: Codable, Sendable {
    public let rowOrdinal: Int
    public let tokenCount: Int
    public let relativeNanos: UInt64
}

/// Engine-queue confined; allocated only by the opt-in benchmark recorder.
final class CBv2ConfirmedTokenTimings {
    static let maximumReceipts = 65_536
    static let maximumRows = 256
    private var rowOrdinals: [ObjectIdentifier: Int] = [:]
    private var nextOrdinal = 0
    private var origin: UInt64?
    private var lastNanos: UInt64 = 0
    private(set) var receipts: [CBv2ConfirmedTokenTiming] = []
    private(set) var dropped: UInt64 = 0

    func record(row: ObjectIdentifier, firstToken: Bool, count: Int, nanos: UInt64) {
        // Object addresses can be reused by a later request. Retire the old
        // association even when this first receipt is invalid or all ordinals
        // are exhausted; subsequent tokens must never rejoin the old row.
        if firstToken { rowOrdinals.removeValue(forKey: row) }
        guard count > 0, count <= 8, nanos > 0, nanos >= lastNanos,
            receipts.count < Self.maximumReceipts
        else {
            dropped = CBv2ForwardShapeRecorder.add(dropped, 1)
            return
        }
        if rowOrdinals[row] == nil {
            // Reused object addresses start a fresh ordinal at their first
            // token, so serial requests cannot merge their histories.
            guard nextOrdinal < Self.maximumRows else {
                dropped = CBv2ForwardShapeRecorder.add(dropped, 1)
                return
            }
            rowOrdinals[row] = nextOrdinal
            nextOrdinal += 1
        }
        if origin == nil { origin = nanos }
        lastNanos = nanos
        receipts.append(
            .init(
                rowOrdinal: rowOrdinals[row]!, tokenCount: count,
                relativeNanos: nanos - origin!))
    }
}
