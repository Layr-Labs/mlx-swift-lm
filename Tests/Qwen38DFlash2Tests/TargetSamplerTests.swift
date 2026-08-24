import MLX
import Testing

@testable import Qwen38DFlash2

@Suite("Qwen 3.8 target posterior sampler")
struct Qwen38TargetSamplerTests {
    @Test("measured sampler contract is fixed")
    func fixedContract() {
        #expect(Qwen38TargetSampler.temperature == 1)
        #expect(Qwen38TargetSampler.topP == 0.95)
        #expect(Qwen38TargetSampler.topK == 20)
        #expect(Qwen38TargetSampler.seed == 42)
    }

    @Test("top-k ties are resolved by lower vocabulary id")
    func deterministicTieSupport() {
        let logits = MLXArray([Float(3), 2, 2, 2, 1]).reshaped([1, 5])
        let (ids, values) = qwen38OrderedTopKSupport(logits, count: 3)
        eval(ids, values)
        #expect(ids.asArray(Int32.self) == [0, 1, 2])
        #expect(values.asArray(Float.self) == [3, 2, 2])
    }

    @Test("identically seeded samplers reproduce their sequence")
    func seededSequence() {
        let logits = MLXArray([Float(4), 3, 2, 1]).reshaped([1, 4])
        let first = Qwen38TargetSampler(seed: 42)
        let second = Qwen38TargetSampler(seed: 42)
        let firstSequence = (0 ..< 4).map { _ in
            first.sample(logits: logits).item(Int32.self)
        }
        let secondSequence = (0 ..< 4).map { _ in
            second.sample(logits: logits).item(Int32.self)
        }
        #expect(firstSequence == secondSequence)
    }
}
