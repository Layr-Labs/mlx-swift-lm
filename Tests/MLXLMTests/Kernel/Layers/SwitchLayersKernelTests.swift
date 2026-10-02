import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// Deterministic values for the switch layer tests. A plain LCG keeps the
/// data the same on each run and does not touch the MLX random state.
struct SwitchKernelValues {
    private var state: UInt64

    init(seed: UInt64) {
        state = (seed &* 0x9E37_79B9_7F4A_7C15) | 1
    }

    /// The next value, uniform in [-1, 1).
    mutating func next() -> Float {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Float((state >> 40) & 0xFF_FFFF) / Float(1 << 23) - 1
    }

    /// A float32 array of `shape` with values uniform in [-scale, scale).
    mutating func array(_ shape: [Int], scale: Float = 1) -> MLXArray {
        let count = shape.reduce(1, *)
        var values = [Float]()
        values.reserveCapacity(count)
        for _ in 0 ..< count { values.append(next() * scale) }
        return MLXArray(values, shape)
    }
}

/// The float32 weights of a small set of GLU experts, kept apart from the
/// module so that the reference does not read the module under test.
struct SwitchKernelExperts {
    /// [E, H, D]
    let gate: MLXArray
    /// [E, H, D]
    let up: MLXArray
    /// [E, D, H]
    let down: MLXArray
    /// [E, H]
    let gateBias: MLXArray?
    /// [E, H]
    let upBias: MLXArray?
    /// [E, D]
    let downBias: MLXArray?

    init(
        gate: MLXArray, up: MLXArray, down: MLXArray,
        gateBias: MLXArray?, upBias: MLXArray?, downBias: MLXArray?
    ) {
        self.gate = gate
        self.up = up
        self.down = down
        self.gateBias = gateBias
        self.upBias = upBias
        self.downBias = downBias
    }

    init(inputDims: Int, hiddenDims: Int, numExperts: Int, bias: Bool, seed: UInt64) {
        var values = SwitchKernelValues(seed: seed)
        let inScale = 2 / Float(inputDims).squareRoot()
        let hiddenScale = 2 / Float(hiddenDims).squareRoot()
        gate = values.array([numExperts, hiddenDims, inputDims], scale: inScale)
        up = values.array([numExperts, hiddenDims, inputDims], scale: inScale)
        down = values.array([numExperts, inputDims, hiddenDims], scale: hiddenScale)
        if bias {
            gateBias = values.array([numExperts, hiddenDims], scale: 0.5)
            upBias = values.array([numExperts, hiddenDims], scale: 0.5)
            downBias = values.array([numExperts, inputDims], scale: 0.5)
        } else {
            gateBias = nil
            upBias = nil
            downBias = nil
        }
    }

    /// The module parameters for a split or a fused gate/up layout. The
    /// fused layout puts the gate rows first, then the up rows.
    func parameters(fused: Bool) -> ModuleParameters {
        var flat: [String: MLXArray] = ["down_proj.weight": down]
        if let downBias { flat["down_proj.bias"] = downBias }
        if fused {
            flat["gate_up_proj.weight"] = concatenated([gate, up], axis: 1)
            if let gateBias, let upBias {
                flat["gate_up_proj.bias"] = concatenated([gateBias, upBias], axis: -1)
            }
        } else {
            flat["gate_proj.weight"] = gate
            flat["up_proj.weight"] = up
            if let gateBias { flat["gate_proj.bias"] = gateBias }
            if let upBias { flat["up_proj.bias"] = upBias }
        }
        return ModuleParameters.unflattened(flat)
    }
}

/// A root module with one SwitchGLU child at the path `moe`.
final class SwitchKernelHost: Module {
    @ModuleInfo(key: "moe") var moe: SwitchGLU

    init(_ moe: SwitchGLU) {
        self._moe.wrappedValue = moe
        super.init()
    }
}

extension KernelTests {

    /// Tests of the expert layers in `SwitchLayers.swift`.
    ///
    /// Each forward test compares a layer with a per-expert loop reference
    /// built from the same weights: for each expert, a dense product over
    /// all rows, then a 0/1 mask keeps the rows routed to that expert. For
    /// a quantized layer the reference uses the dequantized weights.
    ///
    /// Serialized: SwitchGLU probes custom activations through compiled
    /// functions at init, and its forward runs compiled GLU kernels.
    @Suite(.serialized)
    struct SwitchLayersKernelTests {

        // Tolerance of the float32 comparisons: the gather matmul and the
        // dense reference matmul sum the same products in a different
        // order. The error is near 1e-6 on outputs of order 1.
        static let tolerance: Float = 1e-4

        /// Routing with `topK` distinct experts per token: slot `k` of token
        /// `n` goes to expert `(n * stride + k) % experts`.
        static func routing(tokens: Int, topK: Int, experts: Int, stride: Int = 3)
            -> (indices: MLXArray, rows: [[Int]])
        {
            let rows = (0 ..< tokens).map { n in
                (0 ..< topK).map { k in (n * stride + k) % experts }
            }
            let flat = rows.flatMap { row in row.map { UInt32($0) } }
            return (MLXArray(flat, [tokens, topK]), rows)
        }

        /// Per-expert loop reference of one expert projection.
        ///
        /// - Parameters:
        ///   - x: [R, in], one row per assignment.
        ///   - experts: the expert of each row.
        ///   - weight: [E, out, in].
        ///   - bias: [E, out] or nil.
        /// - Returns: [R, out].
        static func referenceLinear(
            _ x: MLXArray, experts: [Int], weight: MLXArray, bias: MLXArray?
        ) -> MLXArray {
            var output = MLXArray.zeros([x.dim(0), weight.dim(1)], dtype: .float32)
            for expert in 0 ..< weight.dim(0) {
                var y = matmul(x, weight[expert].swappedAxes(-1, -2))
                if let bias { y = y + bias[expert] }
                let mask = MLXArray(
                    experts.map { $0 == expert ? Float(1) : Float(0) }, [experts.count, 1])
                output = output + mask * y
            }
            return output
        }

        /// Per-expert loop reference of a SwitchGLU forward.
        /// Returns [N, K, D] for `x` of shape [N, D].
        static func referenceGLU(
            _ x: MLXArray, rows: [[Int]], experts: SwitchKernelExperts,
            activation: (MLXArray) -> MLXArray
        ) -> MLXArray {
            let tokens = x.dim(0)
            let topK = rows[0].count
            let flat = rows.flatMap { $0 }
            // Row n * K + k carries token n for slot k.
            let xRows = repeated(x, count: topK, axis: 0)
            let gate = referenceLinear(
                xRows, experts: flat, weight: experts.gate, bias: experts.gateBias)
            let up = referenceLinear(
                xRows, experts: flat, weight: experts.up, bias: experts.upBias)
            let down = referenceLinear(
                activation(gate) * up, experts: flat, weight: experts.down,
                bias: experts.downBias)
            return down.reshaped(tokens, topK, x.dim(1))
        }

        /// Weighted sum over the top-K slots: [N, K, D] and [N, K] to [N, D].
        static func referenceWeightedSum(_ outputs: MLXArray, _ weights: MLXArray) -> MLXArray {
            (outputs * expandedDimensions(weights, axis: -1)).sum(axis: 1)
        }

        /// The float32 weight of a quantized projection.
        static func dequantizedWeight(_ projection: SwitchLinear?) throws -> MLXArray {
            let layer = try #require(projection as? QuantizedSwitchLinear)
            return dequantized(
                layer.weight, scales: layer.scales, biases: layer.biases,
                groupSize: layer.groupSize, bits: layer.bits, mode: layer.mode,
                dtype: .float32)
        }

        /// The dequantized experts of a quantized split SwitchGLU.
        static func dequantizedExperts(_ glu: SwitchGLU) throws -> SwitchKernelExperts {
            try SwitchKernelExperts(
                gate: dequantizedWeight(glu.gateProj),
                up: dequantizedWeight(glu.upProj),
                down: dequantizedWeight(glu.downProj),
                gateBias: glu.gateProj?.bias, upBias: glu.upProj?.bias,
                downBias: glu.downProj.bias)
        }

        /// A tolerance relative to the size of the reference values.
        static func scaled(_ tolerance: Float, _ reference: MLXArray) -> Float {
            tolerance * Swift.max(1, SyntheticModel.maxAbs(reference))
        }

        // MARK: - SwitchLinear

        @Test func switchLinearMatchesPerExpertLoopWithBias() {
            var values = SwitchKernelValues(seed: 101)
            let weight = values.array([4, 8, 16], scale: 0.5)
            let bias = values.array([4, 8], scale: 0.5)
            let layer = SwitchLinear(
                inputDims: 16, outputDims: 8, numExperts: 4, weight: weight, bias: bias)

            // Unsorted: x [N, 1, 1, D] with indices [N, K] gives [N, K, 1, O].
            let x = values.array([3, 16])
            let routing = Self.routing(tokens: 3, topK: 2, experts: 4)
            let unsorted = layer(
                expandedDimensions(x, axes: [-2, -3]), routing.indices)
            let unsortedReference = Self.referenceLinear(
                repeated(x, count: 2, axis: 0), experts: routing.rows.flatMap { $0 },
                weight: weight, bias: bias
            ).reshaped(3, 2, 1, 8)
            eval(unsorted, unsortedReference)
            #expect(unsorted.shape == [3, 2, 1, 8])
            let unsortedDifference = SyntheticModel.maxAbsDifference(unsorted, unsortedReference)
            #expect(unsortedDifference <= Self.tolerance, "unsorted \(unsortedDifference)")

            // Sorted: one row per index, indices in order, with the hint.
            let sortedExperts = [0, 0, 1, 1, 1, 2, 3, 3]
            let rowsX = values.array([8, 16])
            let sorted = layer(
                expandedDimensions(rowsX, axis: 1),
                MLXArray(sortedExperts.map { UInt32($0) }), sortedIndices: true)
            let sortedReference = Self.referenceLinear(
                rowsX, experts: sortedExperts, weight: weight, bias: bias
            ).reshaped(8, 1, 8)
            eval(sorted, sortedReference)
            #expect(sorted.shape == [8, 1, 8])
            let sortedDifference = SyntheticModel.maxAbsDifference(sorted, sortedReference)
            #expect(sortedDifference <= Self.tolerance, "sorted \(sortedDifference)")
        }

        @Test func quantizedSwitchLinearMatchesDequantizedLoop() throws {
            var values = SwitchKernelValues(seed: 102)
            let weight = values.array([4, 16, 64], scale: 0.25)
            let bias = values.array([4, 16], scale: 0.5)
            let float = SwitchLinear(
                inputDims: 64, outputDims: 16, numExperts: 4, weight: weight, bias: bias)

            let affine = try #require(
                float.toQuantized(groupSize: 32, bits: 4, mode: .affine)
                    as? QuantizedSwitchLinear)
            let mxfp4 = QuantizedSwitchLinear(float, groupSize: 32, bits: 4, mode: .mxfp4)
            let affine8 = QuantizedSwitchLinear(float, groupSize: 64, bits: 8)

            #expect(affine.groupSize == 32 && affine.bits == 4 && affine.mode == .affine)
            #expect(affine8.groupSize == 64 && affine8.bits == 8 && affine8.mode == .affine)
            #expect(mxfp4.mode == .mxfp4)
            for layer in [affine, mxfp4, affine8] {
                // The float bias is carried as it is, and the layer is frozen.
                #expect(layer.bias === bias)
                #expect(layer.trainableParameters().flattened().isEmpty)
                #expect(layer.weight.dtype == .uint32)
            }
            #expect(affine.biases != nil)
            #expect(mxfp4.biases == nil)

            let x = values.array([3, 64])
            let routing = Self.routing(tokens: 3, topK: 2, experts: 4)
            let sortedExperts = [0, 1, 1, 2, 3, 3]
            let rowsX = values.array([6, 64])
            for (label, layer) in [("affine 4", affine), ("mxfp4", mxfp4), ("affine 8", affine8)] {
                let dense = try Self.dequantizedWeight(layer)
                let unsorted = layer(expandedDimensions(x, axes: [-2, -3]), routing.indices)
                let unsortedReference = Self.referenceLinear(
                    repeated(x, count: 2, axis: 0), experts: routing.rows.flatMap { $0 },
                    weight: dense, bias: bias
                ).reshaped(3, 2, 1, 16)
                let sorted = layer(
                    expandedDimensions(rowsX, axis: 1),
                    MLXArray(sortedExperts.map { UInt32($0) }), sortedIndices: true)
                let sortedReference = Self.referenceLinear(
                    rowsX, experts: sortedExperts, weight: dense, bias: bias
                ).reshaped(6, 1, 16)
                eval(unsorted, unsortedReference, sorted, sortedReference)
                #expect(unsorted.dtype == .float32)
                // Tolerance: the quantized kernel and the dense reference use
                // the same dequantized values in a different order of sums.
                let unsortedDifference = SyntheticModel.maxAbsDifference(
                    unsorted, unsortedReference)
                let sortedDifference = SyntheticModel.maxAbsDifference(sorted, sortedReference)
                #expect(
                    unsortedDifference <= Self.scaled(Self.tolerance, unsortedReference),
                    "\(label) unsorted \(unsortedDifference)")
                #expect(
                    sortedDifference <= Self.scaled(Self.tolerance, sortedReference),
                    "\(label) sorted \(sortedDifference)")
            }
        }

        // MARK: - gatherSort / scatterUnsort

        @Test func gatherSortOrdersAssignmentsAndScatterUnsortRestoresThem() {
            var values = SwitchKernelValues(seed: 103)
            let x = values.array([3, 1, 1, 4])
            let indices = MLXArray([UInt32(2), 0, 1, 2, 0, 1], [3, 2])
            let (sortedX, sortedIndices, inverse) = gatherSort(x: x, indices: indices)
            eval(sortedX, sortedIndices, inverse)
            #expect(sortedX.shape == [6, 1, 4])
            #expect(sortedIndices.asArray(UInt32.self) == [0, 0, 1, 1, 2, 2])

            // Exact: gathers and scatters only move values.
            let restored = scatterUnsort(x: sortedX, invOrder: inverse, shape: [3, 2])
            let expected = broadcast(x, to: [3, 2, 1, 4])
            #expect(restored.shape == [3, 2, 1, 4])
            #expect(SyntheticModel.maxAbsDifference(restored, expected) == 0)

            let flat = scatterUnsort(x: sortedX, invOrder: inverse)
            #expect(flat.shape == [6, 1, 4])
            #expect(SyntheticModel.maxAbsDifference(flat, expected.reshaped(6, 1, 4)) == 0)
        }

        // MARK: - SwitchGLU forward

        @Test func switchGLUMatchesPerExpertLoopForEveryLayout() throws {
            // 3 x 2 = 6 assignments run the unsorted gather. 16 x 4 = 64
            // assignments reach the sort threshold and run the sorted path.
            for (tokens, topK) in [(3, 2), (16, 4)] {
                for bias in [false, true] {
                    for fused in [false, true] {
                        let label = "N \(tokens) bias \(bias) fused \(fused)"
                        let experts = SwitchKernelExperts(
                            inputDims: 16, hiddenDims: 8, numExperts: 4, bias: bias,
                            seed: UInt64(200 + tokens))
                        let glu = SwitchGLU(
                            inputDims: 16, hiddenDims: 8, numExperts: 4, bias: bias,
                            fuseGateUp: fused)
                        try glu.update(parameters: experts.parameters(fused: fused), verify: [.all])
                        #expect(glu.hasFusedGateUp == fused)

                        var values = SwitchKernelValues(seed: UInt64(300 + tokens))
                        let x = values.array([tokens, 16])
                        let weights = softmax(values.array([tokens, topK]), axis: -1)
                        let routing = Self.routing(tokens: tokens, topK: topK, experts: 4)

                        let actual = glu(x, routing.indices)
                        let reduced = glu.callAndWeightedReduce(
                            x, routing.indices, weights: weights, fuseSortedReduction: true)
                        let expected = Self.referenceGLU(
                            x, rows: routing.rows, experts: experts, activation: { MLXNN.silu($0) })
                        let expectedReduced = Self.referenceWeightedSum(expected, weights)
                        eval(actual, reduced, expected, expectedReduced)

                        #expect(actual.shape == [tokens, topK, 16], "\(label) shape")
                        #expect(reduced.shape == [tokens, 16], "\(label) reduced shape")
                        let difference = SyntheticModel.maxAbsDifference(actual, expected)
                        let reducedDifference = SyntheticModel.maxAbsDifference(
                            reduced, expectedReduced)
                        #expect(difference <= Self.tolerance, "\(label) output \(difference)")
                        #expect(
                            reducedDifference <= Self.tolerance,
                            "\(label) reduced \(reducedDifference)")
                    }
                }
            }
        }

        @Test func customActivationsSelectTheirPathAndMatchReference() throws {
            let cases: [(String, (MLXArray) -> MLXArray, Bool, Bool)] = [
                ("silu", { MLXNN.silu($0) }, true, false),
                ("tanh gelu", safeGeluApproximate, false, true),
                ("other", { tanh($0) * 0.5 }, false, false),
            ]
            for (label, activation, isSilu, isGelu) in cases {
                let experts = SwitchKernelExperts(
                    inputDims: 16, hiddenDims: 8, numExperts: 4, bias: true, seed: 400)
                let glu = SwitchGLU(
                    inputDims: 16, hiddenDims: 8, numExperts: 4, activation: activation,
                    bias: true)
                try glu.update(parameters: experts.parameters(fused: false), verify: [.all])
                // The init probes the closure once at x = 1 to find a fused
                // SiLU or GELU product. Any other closure runs uncompiled.
                #expect(glu.isSiluActivation == isSilu, "\(label) SiLU flag")
                #expect(glu.isGeluActivation == isGelu, "\(label) GELU flag")
                let hasProduct = glu.activationProduct != nil
                #expect(!hasProduct, "\(label) has no default product")

                for (tokens, topK) in [(3, 2), (16, 4)] {
                    var values = SwitchKernelValues(seed: UInt64(500 + tokens))
                    let x = values.array([tokens, 16])
                    let routing = Self.routing(tokens: tokens, topK: topK, experts: 4)
                    let actual = glu(x, routing.indices)
                    let expected = Self.referenceGLU(
                        x, rows: routing.rows, experts: experts, activation: activation)
                    eval(actual, expected)
                    let difference = SyntheticModel.maxAbsDifference(actual, expected)
                    #expect(
                        difference <= Self.tolerance, "\(label) N \(tokens) output \(difference)")
                }
            }
        }

        @Test func safeGELUMatchesTheTanhFormula() {
            let inputs: [Float] = [-3, -1.5, -0.5, 0, 0.25, 1, 2, 3]
            let output = SafeGELU()(MLXArray(inputs))
            eval(output)
            let actual = output.asArray(Float.self)
            let c = (2 / Double.pi).squareRoot()
            for (x, y) in zip(inputs, actual) {
                let v = Double(x)
                let expected = 0.5 * v * (1 + tanh(c * (v + 0.044715 * v * v * v)))
                // Tolerance: float32 tanh against a Double formula on
                // values below 3. Metal may use a fast tanh, so the bound is 1e-4.
                #expect(Swift.abs(Double(y) - expected) <= 1e-4, "SafeGELU(\(x)) = \(y)")
            }
        }

        // MARK: - Weighted reduction profiles

        @Test func productionProfilesFallBackToLegacyReductionOffTheirGeometry() {
            // Each production profile names one exact checkpoint geometry.
            // A tiny layer is never that geometry, so every profile must
            // take the legacy scatter and weighted sum.
            let profiles: [SwitchGLUWeightedReductionProfile] = [
                .generic, .gemma4ProductionGeGLU, .qwen35ProductionSwiGLU, .qwen4ProductionSwiGLU,
            ]
            let experts = SwitchKernelExperts(
                inputDims: 16, hiddenDims: 8, numExperts: 4, bias: false, seed: 600)
            var values = SwitchKernelValues(seed: 601)
            let x = values.array([16, 16])
            let weights = softmax(values.array([16, 4]), axis: -1)
            let routing = Self.routing(tokens: 16, topK: 4, experts: 4)
            let expected = Self.referenceWeightedSum(
                Self.referenceGLU(
                    x, rows: routing.rows, experts: experts, activation: { MLXNN.silu($0) }),
                weights)
            for profile in profiles {
                let glu = SwitchGLU(
                    inputDims: 16, hiddenDims: 8, numExperts: 4,
                    weightedReductionProfile: profile)
                glu.update(parameters: experts.parameters(fused: false))
                let legacy = weightedExpertSum(glu(x, routing.indices), weights)
                let reduced = glu.callAndWeightedReduce(
                    x, routing.indices, weights: weights, fuseSortedReduction: true,
                    isProductionPrefill: true)
                eval(legacy, reduced)
                // Exact: the fallback runs the same ops on the same data.
                let legacyDifference = SyntheticModel.maxAbsDifference(reduced, legacy)
                #expect(legacyDifference == 0, "\(profile) legacy \(legacyDifference)")
                let difference = SyntheticModel.maxAbsDifference(reduced, expected)
                #expect(difference <= Self.tolerance, "\(profile) reference \(difference)")
            }
        }

        // MARK: - Quantized SwitchGLU

        /// Builds a generic owner and a Qwen4 owner that share the same
        /// quantized expert children.
        static func quantizedOwners(
            inputDims: Int, hiddenDims: Int, numExperts: Int, groupSize: Int,
            mode: QuantizationMode
        ) -> (generic: SwitchGLU, qwen4: SwitchGLU) {
            let generic = SwitchGLU(
                inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts)
            quantize(model: generic, groupSize: groupSize, bits: 4, mode: mode)
            let qwen4 = SwitchGLU(
                inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
                weightedReductionProfile: .qwen4ProductionSwiGLU)
            quantize(model: qwen4, groupSize: groupSize, bits: 4, mode: mode)
            qwen4.update(modules: ModuleChildren.unflattened(generic.leafModules().flattened()))
            eval(generic, qwen4)
            return (generic, qwen4)
        }

        @Test func quantizedSwitchGLUMatchesDequantizedLoopForBothOwners() throws {
            // Four experts: the Qwen4 owner takes its own projection call,
            // but its native kernel needs 512 experts, so the stock gather
            // runs and the output stays float32.
            let owners = Self.quantizedOwners(
                inputDims: 64, hiddenDims: 32, numExperts: 4, groupSize: 32, mode: .affine)
            let experts = try Self.dequantizedExperts(owners.generic)
            for (tokens, topK) in [(3, 2), (16, 4)] {
                var values = SwitchKernelValues(seed: UInt64(700 + tokens))
                let x = values.array([tokens, 64])
                let routing = Self.routing(tokens: tokens, topK: topK, experts: 4)
                let generic = owners.generic(x, routing.indices)
                let qwen4 = owners.qwen4(x, routing.indices)
                let expected = Self.referenceGLU(
                    x, rows: routing.rows, experts: experts, activation: { MLXNN.silu($0) })
                eval(generic, qwen4, expected)
                #expect(qwen4.dtype == .float32)
                // Exact: both owners run the same stock kernels on the same
                // quantized children.
                let ownerDifference = SyntheticModel.maxAbsDifference(generic, qwen4)
                #expect(ownerDifference == 0, "N \(tokens) owners differ by \(ownerDifference)")
                let difference = SyntheticModel.maxAbsDifference(generic, expected)
                #expect(
                    difference <= Self.scaled(Self.tolerance, expected),
                    "N \(tokens) quantized output \(difference)")
            }
        }

        @Test func qwen4OwnerRecordsFallbackAndPinsBFloat16OnUnsupportedQuantization() throws {
            // 512 experts and 512 x 4 = 2048 sorted assignments reach the
            // Qwen4 native kernel. MXFP4 is not a format it accepts, so each
            // projection records a fallback and runs the stock gather. The
            // Qwen4 activation pin then casts the float32 result to bfloat16.
            let owners = Self.quantizedOwners(
                inputDims: 32, hiddenDims: 32, numExperts: 512, groupSize: 32, mode: .mxfp4)
            var values = SwitchKernelValues(seed: 800)
            let x = values.array([512, 32])
            let routing = Self.routing(tokens: 512, topK: 4, experts: 512, stride: 5)

            let before = Qwen4ExpGatherQMMInvocation.snapshot()
            let qwen4 = owners.qwen4(x, routing.indices)
            eval(qwen4)
            let after = Qwen4ExpGatherQMMInvocation.snapshot()
            let generic = owners.generic(x, routing.indices)
            eval(generic)

            // At least three: gate, up and down. Other suites can only add.
            #expect(after.fallback - before.fallback >= 3)
            let pinned = Qwen4ExpActivation.isEnabled()
            #expect(qwen4.dtype == (pinned ? DType.bfloat16 : DType.float32))
            #expect(generic.dtype == .float32)
            #expect(qwen4.shape == [512, 4, 32])

            // Tolerance: the Qwen4 owner rounds gate, up and the output to
            // bfloat16 (8 significant bits). 5% of the largest output value
            // covers a few such roundings; a wrong expert is of order 100%.
            let difference = SyntheticModel.maxAbsDifference(qwen4, generic)
            let bound = 0.05 * Swift.max(SyntheticModel.maxAbs(generic), 1e-3)
            #expect(difference <= bound, "Qwen4 vs generic \(difference), bound \(bound)")
        }

        // MARK: - Gate/up topology

        @Test func twinsChangeTopologyAndKeepBiasActivationAndDownProjection() throws {
            let experts = SwitchKernelExperts(
                inputDims: 16, hiddenDims: 8, numExperts: 4, bias: true, seed: 900)
            let split = SwitchGLU(
                inputDims: 16, hiddenDims: 8, numExperts: 4, activation: safeGeluApproximate,
                bias: true)
            try split.update(parameters: experts.parameters(fused: false), verify: [.all])

            let fused = split.fusingGateUp()
            #expect(fused.hasFusedGateUp)
            let hasSplitHalves = fused.gateProj != nil || fused.upProj != nil
            #expect(!hasSplitHalves)
            let gateUp = try #require(fused.gateUpProj)
            #expect(gateUp.weight.shape == [4, 16, 16])
            #expect(gateUp.bias?.shape == [4, 16])
            #expect(fused.downProj === split.downProj)
            #expect(fused.isGeluActivation && !fused.isSiluActivation)
            let fusedHasProduct = fused.activationProduct != nil
            #expect(!fusedHasProduct)

            // The twin has fresh gate/up weights. Load the fused layout of
            // the same experts; the output must match the split module.
            try fused.update(parameters: experts.parameters(fused: true), verify: [.all])
            let back = fused.splittingGateUp()
            #expect(!back.hasFusedGateUp)
            #expect(back.gateProj?.bias != nil && back.upProj?.bias != nil)
            #expect(back.downProj === split.downProj)
            #expect(back.isGeluActivation)

            for (tokens, topK) in [(3, 2), (16, 4)] {
                var values = SwitchKernelValues(seed: UInt64(901 + tokens))
                let x = values.array([tokens, 16])
                let routing = Self.routing(tokens: tokens, topK: topK, experts: 4)
                let splitOutput = split(x, routing.indices)
                let fusedOutput = fused(x, routing.indices)
                let expected = Self.referenceGLU(
                    x, rows: routing.rows, experts: experts, activation: safeGeluApproximate)
                eval(splitOutput, fusedOutput, expected)
                let fusedDifference = SyntheticModel.maxAbsDifference(fusedOutput, splitOutput)
                let difference = SyntheticModel.maxAbsDifference(fusedOutput, expected)
                #expect(fusedDifference <= Self.tolerance, "N \(tokens) fused vs split")
                #expect(difference <= Self.tolerance, "N \(tokens) fused vs reference")
            }
        }

        @Test func setSwitchGLUGateUpFusedSwapsOnlyTheNamedModule() {
            let original = SwitchGLU(inputDims: 16, hiddenDims: 8, numExperts: 4, bias: true)
            let down = original.downProj
            let host = SwitchKernelHost(original)

            // A path that names no module changes nothing.
            setSwitchGLUGateUpFused(true, at: "missing.moe", in: host)
            #expect(host.moe === original)

            setSwitchGLUGateUpFused(true, at: "moe", in: host)
            let fused = host.moe
            #expect(fused !== original)
            #expect(fused.hasFusedGateUp)
            #expect(fused.downProj === down)
            #expect(fused.gateUpProj?.bias != nil)

            // The requested topology is already there: no new module.
            setSwitchGLUGateUpFused(true, at: "moe", in: host)
            #expect(host.moe === fused)

            // A checkpoint path resolves through an alias to the module path.
            setSwitchGLUGateUpFused(
                false, at: "language_model.moe", aliases: ["language_model.moe", "moe"],
                in: host)
            #expect(!host.moe.hasFusedGateUp)
            #expect(host.moe.downProj === down)
            #expect(host.moe.gateProj?.bias != nil)
        }

        // MARK: - fuseSwitchGLUGateUpWeights

        @Test func gateUpFusionConcatenatesBiasAndSkipsHalfPairsAndFilteredPaths() {
            let weight = MLXArray.zeros([4, 8, 16], dtype: .float32)
            let source: [String: MLXArray] = [
                // Only the gate half: left as it is for the strict update.
                "a.switch_mlp.gate_proj.weight": weight,
                // A full pair that the caller filters out.
                "b.switch_mlp.gate_proj.weight": weight,
                "b.switch_mlp.up_proj.weight": weight,
                // A full pair with biases.
                "c.switch_mlp.gate_proj.weight": weight,
                "c.switch_mlp.up_proj.weight": weight + 1,
                "c.switch_mlp.gate_proj.bias": MLXArray.zeros([4, 8]),
                "c.switch_mlp.up_proj.bias": MLXArray.ones([4, 8]),
            ]
            var reports: [String: Bool] = [:]
            let result = fuseSwitchGLUGateUpWeights(
                weights: source,
                shouldProcess: { $0 != "b.switch_mlp" },
                setFused: { reports[$0] = $1 })

            #expect(result["a.switch_mlp.gate_proj.weight"] != nil)
            #expect(result["a.switch_mlp.gate_up_proj.weight"] == nil)
            #expect(result["b.switch_mlp.gate_proj.weight"] != nil)
            #expect(result["b.switch_mlp.up_proj.weight"] != nil)
            #expect(result["c.switch_mlp.gate_proj.weight"] == nil)
            #expect(result["c.switch_mlp.up_proj.bias"] == nil)
            #expect(reports == ["c.switch_mlp": true])

            let fusedWeight = result["c.switch_mlp.gate_up_proj.weight"]
            let fusedBias = result["c.switch_mlp.gate_up_proj.bias"]
            #expect(fusedWeight?.shape == [4, 16, 16])
            #expect(fusedBias?.shape == [4, 16])
            if let fusedWeight, let fusedBias {
                // Gate rows first, then up rows; biases join on the last axis.
                #expect(fusedWeight[0..., ..<8].sum().item(Float.self) == 0)
                #expect(fusedWeight[0..., 8...].sum().item(Float.self) == Float(4 * 8 * 16))
                #expect(fusedBias[0..., ..<8].sum().item(Float.self) == 0)
                #expect(fusedBias[0..., 8...].sum().item(Float.self) == 32)
            }
        }

        @Test func gateUpFusionKeepsMismatchedPairsSplit() {
            let source: [String: MLXArray] = [
                // Different tensor sets: the gate has a bias, the up has none.
                "d.switch_mlp.gate_proj.weight": MLXArray.zeros([4, 8, 16]),
                "d.switch_mlp.gate_proj.bias": MLXArray.zeros([4, 8]),
                "d.switch_mlp.up_proj.weight": MLXArray.zeros([4, 8, 16]),
                // Different shapes.
                "e.switch_mlp.gate_proj.weight": MLXArray.zeros([4, 8, 16]),
                "e.switch_mlp.up_proj.weight": MLXArray.zeros([4, 6, 16]),
                // Different dtypes.
                "f.switch_mlp.gate_proj.weight": MLXArray.zeros([4, 8, 16], dtype: .float32),
                "f.switch_mlp.up_proj.weight": MLXArray.zeros([4, 8, 16], dtype: .float16),
            ]
            var reports: [String: Bool] = [:]
            let result = fuseSwitchGLUGateUpWeights(
                weights: source, setFused: { reports[$0] = $1 })
            #expect(result.keys.sorted() == source.keys.sorted())
            #expect(
                reports == ["d.switch_mlp": false, "e.switch_mlp": false, "f.switch_mlp": false])
        }

        @Test func gateUpFusionResolvesSkipPoliciesAndAliases() {
            let q4 = BaseConfiguration.Quantization(groupSize: 64, bits: 4, mode: .affine)
            let pair: (String) -> [String: MLXArray] = { base in
                [
                    "\(base).switch_mlp.gate_proj.weight": MLXArray.zeros([4, 8, 16]),
                    "\(base).switch_mlp.up_proj.weight": MLXArray.ones([4, 8, 16]),
                ]
            }
            func fused(_ result: [String: MLXArray], _ base: String) -> Bool {
                result["\(base).switch_mlp.gate_up_proj.weight"] != nil
            }

            // Both halves skip quantization: the same policy, so they fuse.
            // A fused entry that also skips agrees with them.
            let bothSkip = BaseConfiguration.PerLayerQuantization(
                quantization: q4,
                perLayerQuantization: [
                    "g.switch_mlp.gate_proj": .skip,
                    "g.switch_mlp.up_proj": .skip,
                    "g.switch_mlp.gate_up_proj": .skip,
                ])
            let skipped = fuseSwitchGLUGateUpWeights(
                weights: pair("g"), perLayerQuantization: bothSkip)
            #expect(fused(skipped, "g"))

            // Both halves skip, but the fused entry quantizes: stay split.
            let fusedQuantizes = BaseConfiguration.PerLayerQuantization(
                quantization: q4,
                perLayerQuantization: [
                    "h.switch_mlp.gate_proj": .skip,
                    "h.switch_mlp.up_proj": .skip,
                    "h.switch_mlp.gate_up_proj": .quantize(q4),
                ])
            var hReport: Bool?
            let quantizedFused = fuseSwitchGLUGateUpWeights(
                weights: pair("h"), perLayerQuantization: fusedQuantizes,
                setFused: { _, value in hReport = value })
            #expect(!fused(quantizedFused, "h"))
            #expect(hReport == false)

            // The gate path has no entry, but its alias skips. The up half
            // takes the default policy, so the two differ: stay split.
            let aliasTable = BaseConfiguration.PerLayerQuantization(
                quantization: q4,
                perLayerQuantization: ["raw.gate": .skip])
            var iReport: Bool?
            let aliased = fuseSwitchGLUGateUpWeights(
                weights: pair("i"), perLayerQuantization: aliasTable,
                quantizationAliases: { path in
                    path.hasSuffix(".gate_proj") ? [path, "raw.gate"] : []
                },
                setFused: { _, value in iReport = value })
            #expect(!fused(aliased, "i"))
            #expect(iReport == false)
        }
    }
}
