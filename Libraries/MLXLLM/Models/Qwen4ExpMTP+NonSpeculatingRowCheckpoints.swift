import MLXLMCommon

extension Qwen4ExpInlineMTPAssistant {
    /// Requests that cannot speculate record prompt history too, so their
    /// checkpoints are the ones a speculating request saves.
    public var checkpointsNonSpeculatingRows: Bool { true }
}
