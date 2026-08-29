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

    @Test("seed 42 matches the MTPLX global MLX stream across verify widths")
    func sourceStreamSequence() {
        let widths = [1, 5, 5, 5, 4, 5, 4, 4, 5, 5, 5, 5, 5, 6, 6, 5, 5, 6]
        let source = [
            [20],
            [63, 47, 22, 13, 57],
            [22, 52, 19, 33, 30],
            [19, 57, 32, 13, 20],
            [9, 22, 5, 11],
            [5, 28, 3, 56, 45],
            [26, 62, 8, 21],
            [16, 61, 36, 19],
            [52, 35, 9, 8, 13],
            [16, 31, 53, 20, 12],
            [23, 37, 35, 3, 32],
            [4, 26, 9, 8, 45],
            [8, 39, 37, 43, 3],
            [61, 13, 27, 10, 39, 14],
            [11, 33, 62, 38, 4, 18],
            [63, 22, 4, 4, 34],
            [29, 49, 18, 9, 46],
            [2, 40, 14, 21, 42, 2],
        ]
        let sampler = Qwen38TargetSampler(seed: 42)

        for (call, rows) in widths.enumerated() {
            var values = [Float]()
            values.reserveCapacity(rows * 64)
            for row in 0 ..< rows {
                for column in 0 ..< 64 {
                    values.append(
                        Float(((call + 1) * 37 + row * 19 + column * 13) % 101) / 17
                            - Float(column) / 211)
                }
            }
            let logits = MLXArray(values).reshaped([rows, 64])
            let actual = sampler.sample(logits: logits)
            eval(actual)
            #expect(actual.asArray(Int32.self) == source[call].map(Int32.init))
        }
    }
}
