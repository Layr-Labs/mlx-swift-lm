import MLX
import Testing

@testable import Qwen38DFlash2

@Suite("DFlash2 grouped dynamic causal convolution")
struct GroupedDynamicCausalConvTests {
    @Test("two taps use current and previous rows without leaking future values")
    func twoTapReference() {
        let hidden = MLXArray((1 ... 12).map(Float.init), [1, 3, 4])
        let dynamic = MLXArray(
            [
                1, 1, 2, 2,
                1, 1, 2, 2,
                1, 1, 2, 2,
            ].map(Float.init),
            [1, 3, 2, 2])
        let base = MLXArray.zeros([2, 4])

        let output = groupedDynamicCausalConvolution(
            hidden: hidden,
            dynamic: dynamic,
            base: base,
            groupSize: 2)

        eval(output)
        #expect(output.shape == [1, 3, 4])
        #expect(
            output.asArray(Float.self) == [
                1, 2, 3, 4,
                7, 10, 13, 16,
                19, 22, 25, 28,
            ])
    }
}
