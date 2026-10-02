import Foundation

/// Package-only complete historical-assistant capability. Target KV alone is
/// not an assistant checkpoint, and a generic recurrent codec cannot opt in.
/// The request's existing bounded admission must cover the context's complete
/// retained host/native storage before installation; this is not a new lease.
package protocol CBv2HistoricalMTPPrefixCheckpointCoding: CBv2MTPPrefixCheckpointCoding {
    /// Install once on a fresh current-owner/generation state before observing
    /// any target input. Capture subsequently requires a settled interior prompt
    /// boundary, exact observed token evidence, and no pending/round/carry state.
    func installPrefixCaptureContext(
        requestState: any CBv2MTPRequestState, promptTokens: [Int]
    ) throws

    /// Retention only, not an admission grant. Authenticated import binds its
    /// actual charged backing before exposing the decoded checkpoint. Capture
    /// values must refuse this operation to avoid an owner/checkpoint cycle.
    func bindImportedPrefixCheckpointOwner(
        _ checkpoint: any CBv2MTPPrefixCheckpoint, owner: AnyObject
    ) throws
}
