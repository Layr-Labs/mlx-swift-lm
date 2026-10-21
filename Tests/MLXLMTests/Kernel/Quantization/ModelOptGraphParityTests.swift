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
            bank.accumulation = .rowwise
            let rows = input(&fixtures, [70, 1, inputs], magnitude: 3, dtype: dtype)
            let routedIDs: [Int32] = (0 ..< 70).map { (i: Int) -> Int32 in Int32(i * 8 / 70) }
            let routed = MLXArray(routedIDs)
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

        /// The tiled multi-row kernel at prefill scale, uneven expert runs
        /// (some shorter than a tile, one empty) and the narrow-output branch.
        @Test(arguments: [(1024, 256), (2688, 1024), (512, 4)])
        func sortedPrefillRowKernelIsBitwise(inputs: Int, outputs: Int) throws {
            var fixtures = ModelOptGraphFixtures(seed: UInt64(inputs * 7 + outputs))
            let bank = try fixtures.expertBank(
                experts: 16, outputs: outputs, inputs: inputs, magnitude: 3, dtype: .bfloat16)
            let ids: [Int32] = (0 ..< 600)
                .map { (i: Int) -> Int32 in Int32(min(15, (i * i) / 22_000)) }
                .filter { (id: Int32) -> Bool in id != 9 }
            let rows = input(&fixtures, [ids.count, 1, inputs], magnitude: 3, dtype: .bfloat16)
            let routed = MLXArray(ids)
            let eager = try bank.checkedForward(rows, routed, sortedIndices: true)
            bank.accumulation = .rowwise
            let graph = try bank.graphForward(rows, routed, sortedIndices: true)
            #expect(modelOptBitwiseEqual(graph.value, eager))
            let projection = try fixtures.nvfp4Projection(
                outputs: outputs, inputs: inputs, magnitude: 3, dtype: .bfloat16)
            projection.accumulation = .rowwise
            let dense = input(&fixtures, [1, 37, inputs], magnitude: 3, dtype: .bfloat16)
            let denseEager = try projection.checkedForward(dense)
            #expect(modelOptBitwiseEqual(try projection.graphForward(dense), denseEager))
            // Default tiled prefill: same producer bits and scales; only the
            // summation order and one rounding of the unscaled sums differ, so
            // each output stays within 2^-7 of its row's largest magnitude.
            bank.accumulation = .tiled
            projection.accumulation = .tiled
            for (name, tiled, reference) in [
                ("routed", try bank.graphForward(rows, routed, sortedIndices: true).value, eager),
                ("dense", try projection.graphForward(dense), denseEager),
            ] {
                let a = tiled.asType(.float32)
                let b = reference.asType(.float32)
                #expect(arrayEqual(isNaN(a), isNaN(b)).item(Bool.self))
                let rowMax = abs(b).max(axis: -1, keepDims: true)
                let excess = abs(a - b) - rowMax * 0x1p-7
                #expect(
                    !(excess .> 0).any().item(Bool.self),
                    "\(name) largest tiled deviation \((abs(a - b) / maximum(rowMax, 1e-30)).max().item(Float.self))"
                )
            }
        }

        /// The grouped NVFP4 GEMM over skewed, partly empty expert groups:
        /// fused relu2 equals relu2 of the plain product, and both stay within
        /// 2^-7 of each row's maximum of MLX's sorted gather product.
        @Test func groupedGEMMMatchesSortedGatherProduct() throws {
            var fixtures = ModelOptGraphFixtures(seed: 77)
            let bank = try fixtures.expertBank(
                experts: 24, outputs: 200, inputs: 1024, magnitude: 3, dtype: .bfloat16)
            let counts = [
                0, 1, 7, 33, 64, 0, 95, 2, 31, 32, 0, 140, 5, 0, 66, 17, 1, 0, 9, 40, 3, 0, 0, 12,
            ]
            let ids = MLXArray(
                counts.enumerated().flatMap {
                    Array(repeating: UInt32($0.offset), count: $0.element)
                })
            let rows = ids.dim(0)
            let x = input(&fixtures, [rows, 1024], magnitude: 3, dtype: .bfloat16)
            let quantized = ModelOptActivationKernels.nvfp4Routed(
                x, sources: MLXArray((0 ..< rows).map { Int32($0) }), experts: ids,
                globalScale: bank.routeTables.globalScale,
                overflowDivisor: bank.routeTables.overflowDivisor,
                inverseWeightScale: bank.routeTables.inverseWeightScale,
                activationDType: .bfloat16)
            func grouped(_ relu2: Bool) -> MLXArray {
                ModelOptNVFP4GroupedGEMM.product(
                    quantized.activation, weight: bank.weight, scales: bank.blockScales,
                    experts: quantized.safeExperts,
                    expertScale: bank.routeTables.inverseWeightScale, squaredReLU: relu2)
            }
            let plain = grouped(false)
            let r = maximum(plain, MLXArray(0))
            #expect(modelOptBitwiseEqual(grouped(true), r * r))
            let reference = ModelOptActivationKernels.scaledCast(
                gatherQuantizedMM(
                    quantized.activation.asType(.float32).reshaped(rows, 1, 1024), bank.weight,
                    scales: bank.blockScales, biases: nil, rhsIndices: quantized.safeExperts,
                    transpose: true, groupSize: 16, bits: 4, mode: .nvfp4
                ).reshaped(rows, -1), scale: quantized.rowScale, dtype: .bfloat16
            ).asType(.float32)
            let rowMax = abs(reference).max(axis: -1, keepDims: true)
            let deviation = (abs(plain.asType(.float32) - reference) / maximum(rowMax, 1e-30))
                .max().item(Float.self)
            print("grouped NVFP4 GEMM vs per-row product: max deviation / row max \(deviation)")
            #expect(deviation <= 0x1p-7)
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

        // 1100 outputs at K=4096 span three eager tiles of 511 rows.
        @Test(arguments: [(4096, 1100, 1), (4096, 64, 7), (8192, 40, 2)])
        func fp8ExactReferenceIsBitwise(inputs: Int, outputs: Int, rows: Int) throws {
            var fixtures = ModelOptGraphFixtures(seed: UInt64(inputs + outputs + rows))
            let layer = try fixtures.fp8Projection(
                outputs: outputs, inputs: inputs, magnitude: 3, dtype: .bfloat16)
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
            let x = (MLXRandom.normal([1, 1, inputs], key: MLXRandom.key(7)) * 0.8)
                .asType(.bfloat16)
            #expect(modelOptBitwiseEqual(try layer.graphForward(x), try layer.checkedForward(x)))
        }

        /// Batched decode: each row of a 2 to 16-row product equals the one-row
        /// product bit for bit (router, latent and FP8/NVFP4 projection shapes).
        @Test(arguments: [2, 7, 16])
        func rowExactProductsMatchOneRowCalls(rows: Int) throws {
            let key = MLXRandom.key(UInt64(rows))
            let x = (MLXRandom.normal([rows, 4096], key: key) * 0.5).asType(.bfloat16)
            for (outputs, inputs) in [(512, 4096), (1024, 4096), (4096, 1024)] {
                let weight = (MLXRandom.normal([outputs, inputs], key: MLXRandom.key(3)) * 0.02)
                    .asType(.bfloat16)
                let input = x[0..., 0 ..< inputs]
                let batched = ModelOptRowExact.matmul(input, weight)
                for r in 0 ..< rows {
                    let one = matmul(input[r ..< r + 1], weight.transposed())
                    #expect(modelOptBitwiseEqual(batched[r ..< r + 1], one))
                }
            }
            for (outputs, inputs) in [(18560, 4096), (4096, 8192), (4096, 4096), (256, 4096)] {
                let raw = MLXRandom.randInt(Int32(0) ..< 256, [outputs, inputs], key: key)
                    .asType(.uint8)
                let layer = try ModelOptCalibratedLinear(
                    weights: ModelOptFP8Weights(
                        e4m3Bytes: which((raw & UInt8(0x7F)) .== UInt8(0x7F), raw - UInt8(1), raw),
                        tensorScale: 0.0011),
                    inputScale: 0.009, weightDType: .bfloat16)
                let input = (MLXRandom.normal([1, rows, inputs], key: key) * 0.5).asType(.bfloat16)
                let batched = try layer.rowExactForward(input)
                for r in 0 ..< rows {
                    let one = try layer.graphForward(input[0..., r ..< r + 1])
                    #expect(modelOptBitwiseEqual(batched[0..., r ..< r + 1], one))
                }
            }
            var fixtures = ModelOptGraphFixtures(seed: UInt64(rows))
            for (outputs, inputs) in [(5376, 4096), (4096, 5376)] {
                let layer = try fixtures.nvfp4Projection(
                    outputs: outputs, inputs: inputs, magnitude: 3, dtype: .bfloat16)
                let input = input(&fixtures, [rows, inputs], magnitude: 3, dtype: .bfloat16)
                let batched = try layer.rowExactForward(input, squaredReLU: true)
                for r in 0 ..< rows {
                    let one = try layer.graphForward(input[r ..< r + 1], squaredReLU: true)
                    #expect(modelOptBitwiseEqual(batched[r ..< r + 1], one))
                }
            }
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
                #expect(graph.dtype == .bfloat16)
                #expect(modelOptBitwiseEqual(graph, eager))
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

        /// Construction evaluates nothing, so an MLX failure cannot reach the
        /// fatal default error handler there: the inverse weight scales stay
        /// lazy until the first forward, which matches the eager reference.
        @Test func constructionLeavesDeviceConstantsToTheFirstForward() throws {
            func evaluated(_ array: MLXArray) -> Bool {
                var result = false
                _ = _mlx_array_is_available(&result, array.ctx)
                return result
            }
            var fixtures = ModelOptGraphFixtures(seed: 19)
            let bank = try fixtures.expertBank(
                experts: 4, outputs: 32, inputs: 512, magnitude: 2, dtype: .bfloat16)
            let projection = try fixtures.nvfp4Projection(
                outputs: 32, inputs: 512, magnitude: 2, dtype: .bfloat16)
            let constants = [
                bank.routeTables.inverseWeightScale,
                try #require(projection.graphConstants).outputScale,
            ]
            #expect(constants.allSatisfy { !evaluated($0) })
            let x = input(&fixtures, [3, 1, 512], magnitude: 2, dtype: .bfloat16)
            let ids = MLXArray([Int32(3), 0, 2])
            let routed = try bank.graphForward(x, ids).value
            let dense = try projection.graphForward(x)
            eval(routed, dense)
            #expect(constants.allSatisfy(evaluated))
            #expect(modelOptBitwiseEqual(routed, try bank.checkedForward(x, ids)))
            #expect(modelOptBitwiseEqual(dense, try projection.checkedForward(x)))
        }

        @Test func fp8NaNCodeWeightsAndActivationOverflowAreBitwise() throws {
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
            let graph = try layer.graphForward(x)
            #expect(modelOptBitwiseEqual(graph, eager))
            #expect(isNaN(graph[0..., 3]).all().item(Bool.self))
            #expect(isNaN(graph[0..., 6]).all().item(Bool.self))
            #expect(isNaN(graph[1]).all().item(Bool.self))
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
