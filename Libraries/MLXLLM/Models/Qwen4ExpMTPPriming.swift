import Foundation
import MLX

/// Bounded assistant-only catch-up. This can change draft proposals, never
/// target weights or the target verification/rollback contract. Explicit zero
/// retains the former whole-history catch-up for qualification/rollback.
enum Qwen4ExpMTPPriming {
    static let environmentFlag = "DARKBLOOM_QWEN4_MTP_PRIME_CHUNK_TOKENS"
    static let skipColdPromptReplayEnvironmentFlag =
        "DARKBLOOM_QWEN4_MTP_SKIP_COLD_PROMPT_REPLAY"
    static let maximumChunkTokens = 8192
    static let defaultChunkTokens = 2048

    /// Five accepted drafts plus the target seed can be normal round work.
    /// Initial history and a larger backlog after target-only steps are not.
    static func isPrefillCost(cacheTokens: Int, backlogTokens: Int) -> Bool {
        cacheTokens == 0 && backlogTokens > 0 || backlogTokens > 6
    }

    /// Qwen4-only rollbackable policy. Unset selects a carry-only first cold
    /// round; false and malformed values retain whole-history priming. The
    /// caller applies true only to a fresh request's first round; restored
    /// prefix and settled-state checkpoints keep their exact replay path.
    static func skipsColdPromptReplay(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        guard let raw = environment[skipColdPromptReplayEnvironmentFlag] else {
            return true
        }
        return ["1", "true", "yes", "on"].contains(
            raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
    }

    static func validatedChunkTokens(_ value: Int) -> Int {
        (1...maximumChunkTokens).contains(value) ? value : 0
    }

    static func environmentChunkTokens(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Int {
        guard let raw = environment[environmentFlag] else { return defaultChunkTokens }
        guard let value = Int(raw.trimmingCharacters(in: .whitespacesAndNewlines))
        else { return 0 }
        return validatedChunkTokens(value)
    }

    static func needsChunking(backlog: [MLXArray], chunkTokens: Int) -> Bool {
        guard validatedChunkTokens(chunkTokens) > 0 else { return false }
        // Reserve the final seed without summing potentially large shapes.
        var remaining = chunkTokens - 1
        for tokens in backlog {
            if tokens.dim(1) > remaining { return true }
            remaining -= tokens.dim(1)
        }
        return false
    }

    /// Join only enough fragments for one chunk, preserving every observation
    /// and the final seed in order. The caller keeps the original fragments for
    /// discard/retry. Materialization is mandatory before the next chunk starts.
    static func forward(
        hidden: [MLXArray], tokens: [MLXArray], chunkTokens: Int,
        forward: (MLXArray, MLXArray) -> (mixed: MLXArray, residual: MLXArray),
        materialize: ((mixed: MLXArray, residual: MLXArray)) -> Void
    ) -> (mixed: MLXArray, residual: MLXArray) {
        precondition(validatedChunkTokens(chunkTokens) > 0)
        precondition(!hidden.isEmpty && hidden.count == tokens.count)
        var fragment = 0
        var offset = 0
        var result: (mixed: MLXArray, residual: MLXArray)?
        while fragment < tokens.count {
            var hiddenParts: [MLXArray] = []
            var tokenParts: [MLXArray] = []
            var remaining = chunkTokens
            while remaining > 0 && fragment < tokens.count {
                let count = tokens[fragment].dim(1)
                precondition(count > 0 && hidden[fragment].dim(1) == count)
                let take = min(remaining, count - offset)
                hiddenParts.append(hidden[fragment][0..., offset..<offset + take, 0...])
                tokenParts.append(tokens[fragment][0..., offset..<offset + take])
                offset += take
                remaining -= take
                if offset == count {
                    fragment += 1
                    offset = 0
                }
            }
            // Release the previous output before allocating another chunk.
            result = nil
            let h = hiddenParts.count == 1 ? hiddenParts[0] : concatenated(hiddenParts, axis: 1)
            let t = tokenParts.count == 1 ? tokenParts[0] : concatenated(tokenParts, axis: 1)
            let output = forward(h, t)
            materialize(output)
            result = output
        }
        return result!
    }
}
