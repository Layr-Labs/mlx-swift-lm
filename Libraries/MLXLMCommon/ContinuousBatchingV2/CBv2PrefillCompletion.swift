import Foundation

/// A measurement retained after cancellation removed the scheduler record.
/// The in-flight step owns this value until successful device readback; no
/// request content, scheduler record or mutable KV state is retained here.
struct CBv2CancelledPrefillCompletion {
    let observe: @Sendable (CBv2Usage) -> Void
    let usage: CBv2Usage
    let timing: CBv2RequestTiming
    let enqueuedNanos: UInt64

    func publish(readbackDoneNanos: UInt64) {
        var observation = usage
        observation.completionTokens = 0
        var completed = timing
        completed.promptComputedNanos = max(1, readbackDoneNanos &- enqueuedNanos)
        // Confirmation here concerns prompt work only. The discarded sampled
        // token never becomes a first-token or decode-capacity observation.
        observation.timing = completed
        observe(observation)
    }
}
