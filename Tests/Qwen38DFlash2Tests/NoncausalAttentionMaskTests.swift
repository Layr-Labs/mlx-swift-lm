import MLX
import Testing

@testable import Qwen38DFlash2

@Suite("DFlash2 non-causal proposal attention mask")
struct NoncausalAttentionMaskTests {
    @Test("context is windowed while every proposal row sees the whole proposal block")
    func windowAndProposalVisibility() {
        let mask = dflash2AttentionMask(
            blockLength: 2,
            queryOffset: 3,
            keyLength: 5,
            slidingWindow: 3)

        eval(mask)
        #expect(mask.shape == [2, 5])
        #expect(
            mask.asType(.int32).asArray(Int32.self) == [
                0, 1, 1, 1, 1,
                0, 0, 1, 1, 1,
            ])
    }
}
