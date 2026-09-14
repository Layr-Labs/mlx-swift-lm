import Foundation
import MLX
import MLXLMCommon
import os

extension Qwen4ExpTextModel: CBv2Qwen4BatchCapabilityConfiguring {
    public var cbv2Qwen4BatchedAttentionEnabled: Bool {
        model.batchingPolicy.read { model.currentBatchedQSAPolicy }
    }

    /// Construction/load-time only, before this model is attached to a live
    /// engine. This is the same policy bit every QSA layer actually reads.
    public func cbv2ConfigureQwen4BatchedAttention(enabled: Bool) throws {
        try configureBatching(enabled: enabled, seal: false)
    }

    public func cbv2InstallQwen4BatchedAttention(enabled: Bool) throws {
        try configureBatching(enabled: enabled, seal: true)
    }

    private func configureBatching(enabled: Bool, seal: Bool) throws {
        try model.batchingPolicy.configure(enabled: enabled, seal: seal,
            current: { model.currentBatchedQSAPolicy }) {
            for layer in model.layers { layer.selfAttn?.batchedQSAEnabled = enabled }
        }
    }
}

extension Qwen4ExpModel: CBv2Qwen4BatchCapabilityConfiguring {
    public var cbv2Qwen4BatchedAttentionEnabled: Bool {
        languageModel.cbv2Qwen4BatchedAttentionEnabled
    }

    public func cbv2ConfigureQwen4BatchedAttention(enabled: Bool) throws {
        try languageModel.cbv2ConfigureQwen4BatchedAttention(enabled: enabled)
    }

    public func cbv2InstallQwen4BatchedAttention(enabled: Bool) throws {
        try languageModel.cbv2InstallQwen4BatchedAttention(enabled: enabled)
    }
}

/// The model owns the policy, not a particular provider preparation. First
/// installation OR first real forward seals it. Same-value reuse performs no
/// writes; a competing configuration fails before any live layer can change.
final class Qwen4ExpBatchedQSAPolicyState {
    private let lock = NSLock()
    private var sealed: Bool?

    func read(current: () -> Bool) -> Bool {
        lock.withLock { sealed ?? current() }
    }

    func sealOnForward(current: () -> Bool) {
        lock.withLock {
            if sealed == nil { sealed = current() }
        }
    }

    func configure(enabled: Bool, seal: Bool, current: () -> Bool, apply: () -> Void) throws {
        try lock.withLock {
            if let sealed {
                guard sealed == enabled else { throw CBv2Qwen4BatchPolicyError.alreadySealed }
                return
            }
            if current() != enabled { apply() }
            guard current() == enabled else { throw CBv2Qwen4BatchPolicyError.unavailable }
            if seal { sealed = enabled }
        }
    }
}

/// Experimental multirow QSA. Production admission uses this same exact
/// opt-in spelling; absence never enables an unqualified batching path.
enum Qwen4ExpBatchedQSA {
    static let environmentKey = "DARKBLOOM_QWEN4_BATCHED_QSA"

    static func isEnabled(environment: [String: String] = Qwen4ExpEnvironment.snapshot) -> Bool {
        environment[environmentKey] == "1"
    }

    /// Text positions are [B,L]; M-RoPE positions are [3,B,L]. Preserve the
    /// request's complete three-plane slice, including non-text history.
    static func positions(_ positions: MLXArray?, row: Int, batch: Int, length: Int) -> MLXArray? {
        guard let positions else { return nil }
        precondition(row >= 0 && row < batch)
        if positions.ndim == 2, positions.shape == [batch, length] {
            return positions[row ..< row + 1, 0...]
        }
        precondition(positions.ndim == 3 && positions.shape == [3, batch, length],
                     "Qwen4 batched positions must retain a distinct request axis")
        return positions[0..., row ..< row + 1, 0...]
    }
}

/// Counts actual multirow attention execution, not clients or scheduler
/// assignments. No tokens, request identifiers, arrays, or timers retained.
public enum Qwen4ExpBatchedQSAInvocation {
    private static let lock = NSLock()
    private static let logger = Logger(subsystem: "darkbloom", category: "Qwen4BatchedQSA")
    private static let diagnose = Qwen4ExpEnvironment.snapshot[
        "DARKBLOOM_QWEN4_BATCHED_QSA_DIAGNOSTICS"] == "1"
    nonisolated(unsafe) private static var calls = 0
    nonisolated(unsafe) private static var maximumBatch = 0

    static func record(batch: Int) {
        lock.lock()
        let newMaximum = batch > maximumBatch
        calls += 1
        maximumBatch = max(maximumBatch, batch)
        lock.unlock()
        // At most one line per increasing batch width, only when explicitly
        // requested. This records planned native work after a valid host
        // scope, not GPU completion; successful request output is separate
        // evidence. No request IDs, tokens, arrays or synchronization retained.
        if newMaximum && diagnose {
            logger.info("qwen4_batched_qsa maximum_planned_batch=\(batch, privacy: .public)")
        }
    }

    public static func snapshot() -> (calls: Int, maximumBatch: Int) {
        lock.lock()
        defer { lock.unlock() }
        return (calls, maximumBatch)
    }

    public static func reset() {
        lock.lock()
        calls = 0
        maximumBatch = 0
        lock.unlock()
    }
}
