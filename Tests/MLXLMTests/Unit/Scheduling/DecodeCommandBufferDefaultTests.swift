import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension UnitTests {

    /// The decode command-buffer default (`CBv2DecodeCommandBufferDefault`):
    /// on unless the process pins MLX's limits or turns it off, and never in
    /// place of a model's own limits.
    @Suite
    struct DecodeCommandBufferDefaultTests {
        @Test func defaultYieldsToTheProcessGuards() {
            let enabled = CBv2DecodeCommandBufferDefault.isEnabled
            #expect(enabled([:]))
            #expect(enabled(["DARKBLOOM_CBV2_DECODE_COMMAND_BUFFERS": "1"]))
            #expect(!enabled(["DARKBLOOM_CBV2_DECODE_COMMAND_BUFFERS": "0"]))
            // The cluster ranks pin 16 operations per buffer.
            #expect(!enabled(["MLX_MAX_OPS_PER_BUFFER": "16"]))
            #expect(!enabled(["MLX_MAX_MB_PER_BUFFER": "50"]))
        }

        @Test func declaredLimitsWinOverTheDefault() {
            let own = GPU.CommandBufferLimits(operations: 40, megabytes: 0)
            let pick = CBv2DecodeCommandBufferDefault.decodeLimits
            #expect(pick(own, true) == own)
            #expect(pick(own, false) == own)
            #expect(pick(nil, true) == CBv2DecodeCommandBufferDefault.limits)
            #expect(pick(nil, false) == nil)
            #expect(
                CBv2DecodeCommandBufferDefault.limits
                    == GPU.CommandBufferLimits(operations: 100, megabytes: 1 << 20))
        }
    }
}
