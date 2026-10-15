import Cmlx
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLMCommon

extension KernelTests {
    /// Graph-only ModelOpt paths against their eager references. NVFP4 rowwise
    /// results must be bit-identical; FP8 has the stated rounding tolerance.
    struct ModelOptGraphParityTests {
        private func input(
            _ fixtures: inout ModelOptGraphFixtures, _ shape: [Int], magnitude: Float, dtype: DType
        ) -> MLXArray {
            MLXArray(fixtures.values(shape.reduce(1, *), magnitude: magnitude)).reshaped(shape)
                .asType(dtype)
        }

        // 1024 and 1536 take MLX's fast matrix-vector kernel, 672 and 2688 the
        // general one; 64 the quad kernel for single-matrix eager rows.
        @Test(arguments: [
            (1024, 96, DType.bfloat16), (672, 64, .bfloat16), (2688, 32, .bfloat16),
            (1536, 40, .float16), (64, 24, .float32),
        ])
        func switchDecodeAndSortedPrefillAreBitwise(inputs: Int, outputs: Int, dtype: DType) throws
        {
            var fixtures = ModelOptGraphFixtures(seed: UInt64(inputs * 31 + outputs))
            let bank = try fixtures.expertBank(
                experts: 8, outputs: outputs, inputs: inputs, magnitude: 3, dtype: dtype)
            // Decode: one token broadcast to k unsorted, repeated assignments.
            let token = input(&fixtures, [1, 1, 1, 1, inputs], magnitude: 3, dtype: dtype)
            let ids = MLXArray([Int32(5), 1, 5, 0, 7, 2]).reshaped(1, 1, 6)
            let graph = try bank.graphForward(token, ids)
            let eager = try bank.checkedForward(token, ids)
            #expect(graph.value.shape == [1, 1, 6, 1, outputs])
            #expect(modelOptBitwiseEqual(graph.value, eager))
            #expect(graph.validIndices.item(Bool.self))
            // Prefill rows in expert order, as NemotronHSwitchMLP sorts them.
            let rows = input(&fixtures, [70, 1, inputs], magnitude: 3, dtype: dtype)
            let routed = MLXArray((0 ..< 70).map { Int32($0 * 8 / 70) })
            let sorted = try bank.graphForward(rows, routed, sortedIndices: true)
            #expect(
                modelOptBitwiseEqual(
                    sorted.value, try bank.checkedForward(rows, routed, sortedIndices: true)))
            // Matrix rows broadcast against one ID list.
            let matrix = input(&fixtures, [1, 3, inputs], magnitude: 3, dtype: dtype)
            let pair = MLXArray([Int32(3), 6])
            #expect(
                modelOptBitwiseEqual(
                    try bank.graphForward(matrix, pair).value, try bank.checkedForward(matrix, pair)
                ))
        }

        @Test func switchOverflowNaNAndInvalidExpertRows() throws {
            var fixtures = ModelOptGraphFixtures(seed: 7)
            let bank = try fixtures.expertBank(
                experts: 4, outputs: 32, inputs: 512, magnitude: 2, dtype: .bfloat16)
            var values = fixtures.values(3 * 512, magnitude: 2)
            values[600] = 4000  // row 1 exceeds the 464 E4M3 boundary
            values[1100] = .nan  // row 2 has NaN
            let rows = MLXArray(values).reshaped(3, 1, 512).asType(.bfloat16)
            let ids = MLXArray([Int32(0), 1, 3])
            let graph = try bank.graphForward(rows, ids)
            let eager = try bank.checkedForward(rows, ids)
            #expect(modelOptBitwiseEqual(graph.value, eager))
            #expect(isNaN(graph.value[1 ... 2]).all().item(Bool.self))
            #expect(!isNaN(graph.value[0]).any().item(Bool.self))

            let token = rows[0 ..< 1]
            for invalid in [MLXArray([Int32(2), -1, 4]), MLXArray([UInt32(2), .max, 4])] {
                #expect(throws: (any Error).self) { try bank.checkedForward(token, invalid) }
                let output = try bank.graphForward(token, invalid)
                #expect(!output.validIndices.item(Bool.self))
                #expect(isNaN(output.value[1 ... 2]).all().item(Bool.self))
                let reference = try bank.checkedForward(token, MLXArray([Int32(2)]))
                #expect(modelOptBitwiseEqual(output.value[0 ..< 1], reference))
                // The owner's fence check rejects the forward.
                let ledger = ModelOptGraphValidity()
                let value = ModelOptGraphValidity.$current.withValue(ledger) {
                    bank(token, invalid)
                }
                eval([value] + ledger.evaluationRoots)
                #expect(throws: ModelOptGraphValidity.Failure.invalidExpertIndices) {
                    try ledger.check()
                }
            }
            let ledger = ModelOptGraphValidity()
            _ = ModelOptGraphValidity.$current.withValue(ledger) {
                bank(token, MLXArray([Int32(3)]))
            }
            try ledger.check()
        }

        @Test(arguments: [
            (4096, 48, 1, DType.bfloat16), (5376, 40, 3, .bfloat16), (1024, 64, 5, .float16),
            (128, 16, 1, .float32), (64, 24, 4, .float32),
        ])
        func nvfp4ProjectionRowwiseIsBitwise(inputs: Int, outputs: Int, rows: Int, dtype: DType)
            throws
        {
            var fixtures = ModelOptGraphFixtures(seed: UInt64(inputs + rows))
            let layer = try fixtures.nvfp4Projection(
                outputs: outputs, inputs: inputs, magnitude: 3, dtype: dtype)
            let x = input(&fixtures, [1, rows, inputs], magnitude: 3, dtype: dtype)
            let graph = try layer.graphForward(x)
            #expect(graph.shape == [1, rows, outputs] && graph.dtype == dtype)
            #expect(modelOptBitwiseEqual(graph, try layer.checkedForward(x)))
        }

        /// The eager FP8 reference rounds every scaled weight to the weight dtype
        /// before its product; the packed path sums exact E4M3 weights in float32
        /// and scales once. Bound: |d| <= 2u * sum|x||w|s + 2u|y| + 1e-30, where u
        /// is the dtype unit roundoff (weight rounding plus both output roundings).
        @Test(arguments: [
            (4096, 64, 1, DType.bfloat16), (8192, 32, 1, .bfloat16), (4096, 48, 7, .bfloat16),
            (256, 40, 3, .float32),
        ])
        func fp8PackedProjectionWithinWeightRounding(
            inputs: Int, outputs: Int, rows: Int, dtype: DType
        ) throws {
            var fixtures = ModelOptGraphFixtures(seed: UInt64(inputs * 3 + rows))
            let layer = try fixtures.fp8Projection(
                outputs: outputs, inputs: inputs, magnitude: 3, dtype: dtype)
            layer.fp8Product = .packed
            let x = input(&fixtures, [rows, inputs], magnitude: 3, dtype: dtype)
            let graph = try layer.graphForward(x).asType(.float32)
            let eager = try layer.checkedForward(x).asType(.float32)
            let activation = ModelOptActivationKernels.fp8(
                x,
                multipliers: {
                    guard case .fp8(let m, _, _) = layer.graphConstants!.storage else {
                        fatalError()
                    }
                    return m
                }())
            let weights = try ModelOptFP8Weights(e4m3Bytes: layer.weight, tensorScale: 1)
                .decodedRows(0 ..< outputs, dtype: .float32)
            let scale = layer.weightScale.item(Float.self)
            let magnitudeSum = matmul(abs(activation), abs(weights).transposed()) * scale
            let u: Float = dtype == .bfloat16 ? 0x1p-8 : 0x1p-24
            let difference = abs(graph - eager)
            let bound = 2 * u * magnitudeSum + 2 * u * abs(eager) + 1e-30
            #expect((difference .<= bound).all().item(Bool.self))
            let relative = (difference / maximum(abs(eager), 1e-30)).max().item(Float.self)
            print(
                "fp8 parity K=\(inputs) N=\(outputs) rows=\(rows) \(dtype): max abs "
                    + "\(difference.max().item(Float.self)), max rel \(relative), "
                    + "max diff/bound \((difference / bound).max().item(Float.self))")
        }

        // 1100 outputs at K=4096 span three eager tiles of 511 rows.
        @Test(arguments: [(4096, 1100, 1), (4096, 64, 7), (8192, 40, 2)])
        func fp8ExactReferenceIsBitwise(inputs: Int, outputs: Int, rows: Int) throws {
            var fixtures = ModelOptGraphFixtures(seed: UInt64(inputs + outputs + rows))
            let layer = try fixtures.fp8Projection(
                outputs: outputs, inputs: inputs, magnitude: 3, dtype: .bfloat16)
            layer.fp8Product = .exactReference
            let x = input(&fixtures, [1, rows, inputs], magnitude: 3, dtype: .bfloat16)
            #expect(modelOptBitwiseEqual(try layer.graphForward(x), try layer.checkedForward(x)))
        }

        /// Super's FP8 shapes, including q_proj whose last eager tile splits K
        /// differently from the others, and the 131072-row output head.
        @Test(arguments: [(4096, 18560), (8192, 4096), (4096, 4096), (4096, 256), (4096, 131072)])
        func fp8ExactReferenceIsBitwiseAtSuperShapes(inputs: Int, outputs: Int) throws {
            let key = MLXRandom.key(UInt64(inputs ^ outputs))
            let raw = MLXRandom.randInt(Int32(0) ..< 256, [outputs, inputs], key: key)
                .asType(.uint8)
            let bytes = which((raw & UInt8(0x7F)) .== UInt8(0x7F), raw - UInt8(1), raw)
            let layer = try ModelOptCalibratedLinear(
                weights: ModelOptFP8Weights(e4m3Bytes: bytes, tensorScale: 0.00123),
                inputScale: 0.0071, weightDType: .bfloat16)
            #expect(layer.fp8Product == .exactReference)
            let x = (MLXRandom.normal([1, 1, inputs], key: MLXRandom.key(7)) * 0.8)
                .asType(.bfloat16)
            #expect(modelOptBitwiseEqual(try layer.graphForward(x), try layer.checkedForward(x)))
        }

        @Test func fp8ActivationKernelMatchesEagerForEveryBFloat16() throws {
            let bits = MLXArray((0 ..< 65536).map { UInt16($0) }).view(dtype: .bfloat16)
            for inputScale: Float in [0.01, 1, 3.5e-3] {
                let amax = inputScale * 448
                let multiplier = 448 / amax
                let inverse = 1 / multiplier
                let graph = ModelOptActivationKernels.fp8(
                    bits.reshaped(256, 256), multipliers: MLXArray([multiplier, inverse]))
                // The eager activation, as in ModelOptFP8Weights.calibratedMatmul.
                let scaled = bits.reshaped(256, 256).asType(.float32) * multiplier
                var handle = mlx_array_new()
                #expect(mlx_to_fp8(&handle, scaled.ctx, StreamOrDevice.default.ctx) == 0)
                let packed = MLXArray(handle)
                let decoded = try ModelOptFP8Weights(e4m3Bytes: packed, tensorScale: inverse)
                    .decodedRows(0 ..< 256, dtype: .bfloat16)
                let eager = which(
                    logicalOr(abs(scaled) .> 464, isNaN(scaled)),
                    MLXArray(Float.nan).asType(.bfloat16), decoded)
                #expect(modelOptBitwiseEqual(graph.asType(.bfloat16), eager))
                #expect(modelOptBitwiseEqual(graph, graph.asType(.bfloat16).asType(.float32)))
            }
        }

        @Test func calibrationInvalidUpdateRefusesGraphLikeEager() throws {
            var fixtures = ModelOptGraphFixtures(seed: 11)
            let fp8 = try fixtures.fp8Projection(
                outputs: 16, inputs: 64, magnitude: 2, dtype: .bfloat16)
            let nvfp4 = try fixtures.nvfp4Projection(
                outputs: 16, inputs: 64, magnitude: 2, dtype: .bfloat16)
            let x = input(&fixtures, [2, 64], magnitude: 2, dtype: .bfloat16)
            for layer in [fp8, nvfp4] {
                for invalid: Float in [0, -1, .nan, .infinity] {
                    try layer.update(
                        parameters: .unflattened([("input_scale", MLXArray(invalid))]),
                        verify: .noUnusedKeys)
                    #expect(throws: (any Error).self) { try layer.checkedForward(x) }
                    #expect(throws: (any Error).self) { try layer.graphForward(x) }
                    #expect(!layer.graphCalibrationValid)
                }
                // A valid update replaces the device constants, never stale ones.
                try layer.update(
                    parameters: .unflattened([("input_scale", MLXArray(Float(0.01)))]),
                    verify: .noUnusedKeys)
                #expect(layer.graphCalibrationValid)
                let graph = try layer.graphForward(x)
                let eager = try layer.checkedForward(x)
                #expect(!isNaN(eager).any().item(Bool.self))
                #expect(modelOptBitwiseEqual(graph, eager))
            }
            try nvfp4.update(
                parameters: .unflattened([("weight_scale_2", MLXArray(Float.nan))]),
                verify: .noUnusedKeys)
            #expect(throws: (any Error).self) { try nvfp4.checkedForward(x) }
            #expect(throws: (any Error).self) { try nvfp4.graphForward(x) }
            try fp8.update(
                parameters: .unflattened([("weight_scale", MLXArray(Float(0)))]),
                verify: .noUnusedKeys)
            #expect(throws: (any Error).self) { try fp8.checkedForward(x) }
            #expect(throws: (any Error).self) { try fp8.graphForward(x) }
        }

        @Test func fp8NaNCodeWeightsAndActivationOverflowMatchEager() throws {
            var fixtures = ModelOptGraphFixtures(seed: 5)
            var bytes = fixtures.fp8Weights(rows: 8, columns: 64).asArray(UInt8.self)
            bytes[3 * 64 + 9] = 0x7F
            bytes[6 * 64] = 0xFF
            let layer = try ModelOptCalibratedLinear(
                weights: ModelOptFP8Weights(
                    e4m3Bytes: MLXArray(bytes).reshaped(8, 64), tensorScale: 0.001),
                inputScale: 0.01, weightDType: .bfloat16)
            var values = fixtures.values(3 * 64, magnitude: 2)
            values[70] = 50  // 50 / 0.01 exceeds 464
            let x = MLXArray(values).reshaped(3, 64).asType(.bfloat16)
            let eager = try layer.checkedForward(x)
            for product in [ModelOptFP8Product.packed, .exactReference] {
                layer.fp8Product = product
                let graph = try layer.graphForward(x)
                #expect(arrayEqual(isNaN(graph), isNaN(eager)).item(Bool.self))
                #expect(isNaN(graph[0..., 3]).all().item(Bool.self))
                #expect(isNaN(graph[0..., 6]).all().item(Bool.self))
                #expect(isNaN(graph[1]).all().item(Bool.self))
            }
        }

        @Test func executionSelectsEagerReference() throws {
            var fixtures = ModelOptGraphFixtures(seed: 3)
            let bank = try fixtures.expertBank(
                experts: 2, outputs: 16, inputs: 512, magnitude: 1, dtype: .bfloat16)
            let token = input(&fixtures, [1, 1, 512], magnitude: 1, dtype: .bfloat16)
            let ids = MLXArray([Int32(1), 0])
            #expect(bank.execution == .graph)
            let graph = bank(token, ids)
            bank.execution = .eagerReference
            #expect(modelOptBitwiseEqual(graph, bank(token, ids)))
        }
    }
}
