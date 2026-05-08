// Copyright © 2026 Apple Inc.

import MLX
import MLXLLM
import MLXNN
import MLXRandom
import Testing

@Suite("DFlash verify linear")
struct DFlashVerifyLinearTests {
    @Test func optimizedQuantizedLinearMatchesFallbackForSixteenRows() {
        MLXRandom.seed(42)
        let linear = Linear(256, 64, bias: true)
        let quantized = QuantizedLinear(linear, groupSize: 64, bits: 4)
        let verify = DFlashVerifyQuantizedLinear(quantized)
        let x = MLXRandom.normal([1, 16, 256]).asType(.bfloat16)

        let expected = quantized(x)
        let actual = verify(x)
        eval(expected, actual)

        #expect(allClose(actual, expected, rtol: 1e-2, atol: 1e-2).item(Bool.self))
    }

    @Test func installReplacesOnlyEligibleQuantizedLinears() {
        let model = LinearPair()
        let replaced = DFlashVerifyLinear.install(on: model)

        #expect(replaced == 1)
        #expect(model.good is DFlashVerifyQuantizedLinear)
        #expect(model.bad is QuantizedLinear)
        #expect(!(model.bad is DFlashVerifyQuantizedLinear))
    }
}

private final class LinearPair: Module {
    @ModuleInfo var good: Linear
    @ModuleInfo var bad: Linear

    override init() {
        self._good.wrappedValue = QuantizedLinear(Linear(256, 64), groupSize: 64, bits: 4)
        self._bad.wrappedValue = QuantizedLinear(Linear(128, 64), groupSize: 64, bits: 4)
        super.init()
    }
}
