import Foundation

/// Provider-owned encrypted I/O and single-use native stage tickets. The
/// engine consumes already-authenticated state; it never reads disk on its
/// execution queue. This is deliberately separate from the AR cache protocol.
public protocol CBv2NativeBlockPrefixCache: AnyObject, Sendable {
    var identity: CBv2CompleteCheckpointIdentity { get }
    func acceptsCheckpoint(position: Int, packedBytes: Int) -> Bool
    func takeNativeStaged(requestID: CBv2RequestID, tokens: [Int], cacheSalt: String?,
                          maximumSequenceLength: Int) -> CBv2NativeBlockCheckpoint?
    /// Always close source before completion, including refusal/cancellation.
    /// The engine's independent donor lease retires in that completion.
    func donate(_ source: CBv2CompleteCheckpointExport, requestID: CBv2RequestID?,
                tokens: [Int], cacheSalt: String?, completion: @escaping @Sendable ([Int]) -> Void)
    func close()
}

public typealias CBv2NativeBlockCheckpointPlanner = @Sendable (
    CBv2CompleteCheckpointManifest, CBv2Request, CBv2NativeBlockEngine
) throws -> CBv2NativeBlockCheckpointImportPlan
