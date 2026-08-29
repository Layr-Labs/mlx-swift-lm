import MLX
import Testing

@testable import Qwen38DFlash2

@Suite("DFlash2 grouped dynamic causal convolution")
struct GroupedDynamicCausalConvTests {
    @Test("matches Python's ordered fixed-then-dynamic accumulation")
    func orderedAccumulationReference() {
        let hidden = MLXArray(
            [
                0.3333, -1.234, 2.718, -3.141,
                4.125, -5.375, 6.625, -7.875,
                8.0625, -9.1875, 10.3125, -11.4375,
            ].map(Float.init),
            [1, 3, 4]
        ).asType(.bfloat16)
        let dynamic = MLXArray(
            [
                0.9375, -0.8125, 0.6875, -0.5625,
                0.4375, -0.3125, 0.1875, -0.0625,
                -0.03125, 0.15625, -0.28125, 0.40625,
            ].map(Float.init),
            [1, 3, 2, 2]
        ).asType(.bfloat16)
        let base = MLXArray(
            [
                0.84375, -0.71875, 0.59375, -0.46875,
                -0.34375, 0.21875, -0.09375, 0.03125,
            ].map(Float.init),
            [2, 4]
        ).asType(.bfloat16)

        let output = groupedDynamicCausalConvolution(
            hidden: hidden,
            dynamic: dynamic,
            base: base,
            groupSize: 2)

        let blocks = hidden.reshaped([1, 3, 2, 2])
        var reference = MLXArray.zeros(like: blocks)
        for offset in 0 ..< 2 {
            let values =
                offset == 0
                ? blocks
                : concatenated(
                    [
                        MLXArray.zeros([1, offset, 2, 2], dtype: hidden.dtype),
                        blocks[0..., 0 ..< (3 - offset), 0..., 0...],
                    ],
                    axis: 1)
            let fixed = base[offset, 0...].reshaped([1, 1, 2, 2])
            let generated = dynamic[0..., 0..., offset, 0...]
                .expandedDimensions(axis: -1)
            reference = reference + fixed * values
            reference = reference + generated * values
        }
        reference = reference.reshaped(hidden.shape)

        eval(output, reference)
        #expect(output.asArray(Float.self) == reference.asArray(Float.self))
    }

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
