// Copyright © 2026 Eigen Labs.

import MLX
import MLXRandom
import Testing

@testable import MLXLMCommon

extension KernelTests {
    /// The FP8 row-exact work layouts: every schedule gives the bytes of the
    /// standard one, and the standard one stays the default.
    struct ModelOptRowExactScheduleTests {
        private func layer(outputs: Int, inputs: Int, seed: UInt64) throws
            -> ModelOptCalibratedLinear
        {
            let raw = MLXRandom.randInt(
                Int32(0) ..< 256, [outputs, inputs], key: MLXRandom.key(seed)
            ).asType(.uint8)
            return try ModelOptCalibratedLinear(
                weights: ModelOptFP8Weights(
                    e4m3Bytes: which((raw & UInt8(0x7F)) .== UInt8(0x7F), raw - UInt8(1), raw),
                    tensorScale: 0.0011),
                inputScale: 0.009, weightDType: .bfloat16, activationOverflow: .saturating)
        }

        @Test func standardScheduleIsTheDefault() throws {
            let projection = try layer(outputs: 64, inputs: 128, seed: 1)
            #expect(projection.rowExactSchedule == .standard)
            #expect(!projection.rowExactBatches)
            #expect(
                ModelOptRowExact.Schedule.standard
                    == .init(
                        separateActivationCast: false, twoRowsPerThread: false))
        }

        /// Muse Glimmer's FP8 shapes (K/V, query and output projections).
        @Test(arguments: [1, 2, 5, 8, 16])
        func windowScheduleKeepsTheBytes(rows: Int) throws {
            let schedules: [ModelOptRowExact.Schedule] = [
                .windows,
                .init(separateActivationCast: true, twoRowsPerThread: false),
                .init(separateActivationCast: false, twoRowsPerThread: true),
            ]
            for (outputs, inputs) in [(256, 6656), (4096, 6656), (6656, 4096)] {
                let projection = try layer(
                    outputs: outputs, inputs: inputs, seed: UInt64(outputs + inputs))
                // Overflowing activations take the saturating path in every schedule.
                var x = MLXRandom.normal([1, rows, inputs], key: MLXRandom.key(UInt64(rows)))
                x[0..., 0..., 0 ..< 3] = MLXArray([Float(1e6), -1e6, 0.5])
                x = x.asType(.bfloat16)
                func forward() throws -> MLXArray {
                    rows == 1 ? try projection.graphForward(x) : try projection.rowExactForward(x)
                }
                projection.rowExactSchedule = .standard
                let reference = try forward()
                let oneRow = try (0 ..< rows).map {
                    try projection.graphForward(x[0..., $0 ..< $0 + 1])
                }
                for schedule in schedules {
                    projection.rowExactSchedule = schedule
                    let result = try forward()
                    #expect(
                        modelOptBitwiseEqual(result, reference),
                        "\(schedule) rows \(rows) [\(outputs), \(inputs)]")
                    for r in 0 ..< rows {
                        #expect(modelOptBitwiseEqual(result[0..., r ..< r + 1], oneRow[r]))
                    }
                }
            }
        }
    }
}
