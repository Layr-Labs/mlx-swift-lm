// Copyright © 2026 Eigen Labs.

import MLXLMCommon
import Testing

extension UnitTests {
    struct ModelOptWeightOnlyNVFP4LayoutTests {
        typealias Tensor = ModelOptProjectionLayout.Tensor
        typealias Failure = ModelOptProjectionLayout.Failure

        // Header dimensions of a Muse Glimmer 30B `mlp.gate_proj`.
        private let tensors: [String: Tensor] = [
            "weight": Tensor(dtype: "U8", shape: [19968, 3328], byteCount: 66_453_504),
            "weight_scale": Tensor(dtype: "F8_E4M3", shape: [19968, 416], byteCount: 8_306_688),
            "weight_scale_2": Tensor(dtype: "F32", shape: [], byteCount: 4),
        ]

        @Test func acceptsThePublishedLayoutAndNeedsEveryTensor() throws {
            let layout = try ModelOptWeightOnlyNVFP4Layout(
                tensors: tensors, inputDimensions: 6656, outputDimensions: 19968)
            #expect(layout.inputDimensions == 6656)
            #expect(layout.outputDimensions == 19968)
            for key in tensors.keys {
                var incomplete = tensors
                incomplete.removeValue(forKey: key)
                #expect(throws: Failure.missing(key)) {
                    try ModelOptWeightOnlyNVFP4Layout(
                        tensors: incomplete, inputDimensions: 6656, outputDimensions: 19968)
                }
            }
        }

        @Test func refusesActivationCalibrationAndUnknownSidecars() throws {
            for extra in ["input_scale", "scales", "bias", "k_scale"] {
                var extended = tensors
                extended[extra] = Tensor(dtype: "F32", shape: [], byteCount: 4)
                #expect(throws: Failure.unexpected(extra)) {
                    try ModelOptWeightOnlyNVFP4Layout(
                        tensors: extended, inputDimensions: 6656, outputDimensions: 19968)
                }
            }
        }

        @Test func refusesWrongShapesDtypesAndByteCounts() throws {
            let cases: [(String, Tensor)] = [
                ("weight", Tensor(dtype: "U8", shape: [3328, 19968], byteCount: 66_453_504)),
                ("weight", Tensor(dtype: "U8", shape: [19968, 3328], byteCount: 66_453_503)),
                ("weight", Tensor(dtype: "F8_E4M3", shape: [19968, 3328], byteCount: 66_453_504)),
                ("weight_scale", Tensor(dtype: "U8", shape: [19968, 416], byteCount: 8_306_688)),
                (
                    "weight_scale",
                    Tensor(dtype: "F8_E4M3", shape: [19968, 417], byteCount: 8_326_656)
                ),
                ("weight_scale_2", Tensor(dtype: "F32", shape: [2], byteCount: 8)),
                ("weight_scale_2", Tensor(dtype: "BF16", shape: [], byteCount: 2)),
            ]
            for (key, tensor) in cases {
                var changed = tensors
                changed[key] = tensor
                #expect(throws: Failure.invalidTensor(key)) {
                    try ModelOptWeightOnlyNVFP4Layout(
                        tensors: changed, inputDimensions: 6656, outputDimensions: 19968)
                }
            }
            // The architecture decides the dimensions, not the header.
            for (input, output) in [(6656, 4096), (6640, 19968), (0, 19968), (Int.max, 19968)] {
                #expect(throws: (any Error).self) {
                    try ModelOptWeightOnlyNVFP4Layout(
                        tensors: tensors, inputDimensions: input, outputDimensions: output)
                }
            }
        }
    }
}
