import MLXLMCommon
import MLXLLM

extension Qwen4Exp: CBv2Qwen4BatchCapabilityConfiguring {
    public var cbv2Qwen4BatchedAttentionEnabled: Bool {
        qwen4ExpTextTarget.cbv2Qwen4BatchedAttentionEnabled
    }

    public func cbv2ConfigureQwen4BatchedAttention(enabled: Bool) throws {
        try qwen4ExpTextTarget.cbv2ConfigureQwen4BatchedAttention(enabled: enabled)
    }

    public func cbv2InstallQwen4BatchedAttention(enabled: Bool) throws {
        try qwen4ExpTextTarget.cbv2InstallQwen4BatchedAttention(enabled: enabled)
    }
}
