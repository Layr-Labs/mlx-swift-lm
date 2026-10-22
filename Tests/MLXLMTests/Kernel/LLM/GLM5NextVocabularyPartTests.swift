// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {
    /// A vocabulary part of a quantized head gives the whole head's bits for
    /// its rows at one row (the readout tensor-parallel greedy decode uses),
    /// and the parts' (max, index) pairs pick the whole head's argmax.
    @Suite struct GLM5NextVocabularyPartTests {
        @Test func partsAreTheWholeHeadsRowsOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                let vocabulary = 4096, hidden = 512
                let full = MLXRandom.normal([vocabulary, hidden], key: MLXRandom.key(11)) * 0.05
                let head = QuantizedLinear(weight: full, bias: nil, groupSize: 64, bits: 4)
                for seed in 0 ..< 4 {
                    let x = MLXRandom.normal([1, 1, hidden], key: MLXRandom.key(UInt64(20 + seed)))
                        .asType(.bfloat16)
                    let whole = head(x)
                    let parts = try (0 ..< 2).map { try GLM5NextTextStage.vocabularyPart(of: head, $0, of: 2)(x) }
                    eval([whole] + parts)
                    let joined = concatenated(parts, axis: -1)
                    #expect(
                        joined.asType(.float32).asArray(Float.self).map(\.bitPattern)
                            == whole.asType(.float32).asArray(Float.self).map(\.bitPattern))
                    let rows = parts.map { $0.reshaped(-1).asType(.float32) }
                    let best = rows.enumerated().map { part, row -> (Float, Int) in
                        let index = argMax(row).item(Int.self)
                        return (row[index].item(Float.self), index + part * (vocabulary / 2))
                    }
                    let chosen = best[1].0 > best[0].0 ? best[1].1 : best[0].1
                    #expect(chosen == argMax(whole.reshaped(-1).asType(.float32)).item(Int.self))
                }
                #expect(throws: (any Error).self) {
                    try GLM5NextTextStage.vocabularyPart(of: Linear(hidden, 8), 0, of: 2)
                }
            }
        }
    }
}
