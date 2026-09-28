import MLX
import MLXLMCommon

/// Prepared image rows in prompt order. Pixels are evaluated before return and
/// must remain immutable while the native model consumes this request.
public struct DiffusionGemmaMediaInput: @unchecked Sendable {
    public struct Frame {
        public enum Kind: Sendable { case image, videoFrame }
        public let kind: Kind
        public let pixels: MLXArray
        public let span: CBv2ImageSpan
        public let timestampSeconds: Double?
    }
    public let tokens: [Int32]
    public let frames: [Frame]
}
