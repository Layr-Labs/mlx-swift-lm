import Foundation

/// Native block-generation quantum. Only `committed` can reach an output
/// stream; prefill/refinement never masquerade as emitted tokens.
public enum CBv2NativeBlockStep: Sendable {
    case prefill(computedTokens: Int, complete: Bool)
    case progress
    case committed(tokens: [Int], stopToken: Int?, finishReason: CBv2FinishReason?)
}

/// Request state owned exclusively by a native engine queue. This deliberately
/// has no autoregressive logits/TokenIterator requirement.
public protocol CBv2NativeBlockSession: AnyObject {
    var retainedBytes: Int { get }
    /// Subset of retainedBytes physically owned by the shared paged backend.
    /// Engine gauges substitute its actual backing once, not once per row.
    var sharedStorageBytes: Int { get }
    var activeTokenCount: Int { get }
    var generatedTokenCount: Int { get }
    var prefixUsage: CBv2Usage { get }
    func advanceNative() throws -> CBv2NativeBlockStep
    func finish(reason: CBv2FinishReason)
    func cancel()
}

extension CBv2NativeBlockSession {
    public var sharedStorageBytes: Int { 0 }
    public var prefixUsage: CBv2Usage { .init(promptTokens: 0, completionTokens: 0) }
    public func finish(reason: CBv2FinishReason) {}
}

/// Engine-queue-only cache resources inside the existing slot grant. Closures
/// must not retain model weights after clear; this is not extra memory capacity.
public struct CBv2NativeBlockSharedResources: Sendable {
    public let maximumBytes: Int
    public let retainedBytes: @Sendable () -> Int
    public let trim: @Sendable (Int) -> Void
    public let clear: @Sendable () -> Void
    public init(
        maximumBytes: Int, retainedBytes: @escaping @Sendable () -> Int,
        trim: @escaping @Sendable (Int) -> Void, clear: @escaping @Sendable () -> Void
    ) {
        self.maximumBytes = maximumBytes
        self.retainedBytes = retainedBytes
        self.trim = trim
        self.clear = clear
    }
}

/// Cancellation is visible during a blocking device quantum without moving
/// the session or any MLX array onto another thread.
public final class CBv2NativeBlockCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    public var isCancelled: Bool { lock.withLock { cancelled } }
    func cancel() { lock.withLock { cancelled = true } }
}

public enum CBv2NativeBlockError: Error, Sendable, Equatable, LocalizedError {
    case invalidConfiguration
    case unsupportedRequest(String)
    case duplicateRequest
    case shuttingDown
    case tokenizerRewroteCommittedText

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration: "Invalid native block engine configuration."
        case .unsupportedRequest(let field): "Unsupported native block request: \(field)."
        case .duplicateRequest: "A request with this identifier is already active."
        case .shuttingDown: "The native block engine is draining."
        case .tokenizerRewroteCommittedText: "Tokenizer rewrote previously committed output."
        }
    }
}
