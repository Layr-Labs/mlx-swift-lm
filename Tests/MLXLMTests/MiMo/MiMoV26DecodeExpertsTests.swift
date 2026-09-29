import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

final class MiMoV26DecodeExpertEligibilityTests: XCTestCase {
    func testShortPackedShapeContract() {
        for rows in 1 ... 7 {
            XCTAssertTrue(
                MiMoV26DecodeExperts.supports(
                    rows: rows, hidden: 4096,
                    intermediate: 2048, experts: 256, topK: 8, dtype: .bfloat16))
        }
        for rows in [0, 8] {
            XCTAssertFalse(
                MiMoV26DecodeExperts.supports(
                    rows: rows, hidden: 4096,
                    intermediate: 2048, experts: 256, topK: 8, dtype: .bfloat16))
        }
        for hidden in [0, 4000, 8192] {
            XCTAssertFalse(
                MiMoV26DecodeExperts.supports(
                    rows: 1, hidden: hidden,
                    intermediate: 2048, experts: 256, topK: 8, dtype: .bfloat16))
        }
        XCTAssertFalse(
            MiMoV26DecodeExperts.supports(
                rows: 1, hidden: 4096,
                intermediate: 2048, experts: 256, topK: 6, dtype: .bfloat16))
        XCTAssertFalse(
            MiMoV26DecodeExperts.supports(
                rows: 1, hidden: 4096,
                intermediate: 2048, experts: 256, topK: 8, dtype: .float32))
    }
}

final class MiMoV26DecodeExpertsTests: XCTestCase {
    override func setUpWithError() throws {
        guard ProcessInfo.processInfo.environment["MLX_TEST_MIMO_DECODE_KERNELS"] == "1" else {
            throw XCTSkip("Requires explicitly owned native lane")
        }
        guard MiMoV26DecodeExperts.gpuStream(.default), MLXHardwareInfo.isCompiledDecodeSupported
        else {
            throw XCTSkip("Requires the actual GPU stream and existing compiled SiLU contract")
        }
    }

    private typealias Matrix = MiMoV26DecodeExperts.Matrix

    private func packed(_ experts: Int, _ output: Int, _ input: Int, salt: Int) -> Matrix {
        let count = experts * output * input / 8
        // Direct native codes/scales, not a requantized reference. Nibbles
        // exercise both signs, signed zeros and every E2M1 magnitude.
        let words = (0 ..< count).map { UInt32(truncatingIfNeeded: $0 * 2_654_435_761 + salt * 97) }
        let scales = (0 ..< experts * output * input / 32).map { UInt8(120 + ($0 + salt) % 7) }
        return Matrix(
            weight: MLXArray(words, [experts, output, input / 8]),
            scales: MLXArray(scales, [experts, output, input / 32]))
    }

    private func gather(_ x: MLXArray, _ matrix: Matrix, _ indices: MLXArray) -> MLXArray {
        MLX.gatherQuantizedMM(
            x, matrix.weight, scales: matrix.scales,
            biases: nil, rhsIndices: indices, transpose: true,
            groupSize: 32, bits: 4, mode: .mxfp4)
    }

    private func exact(_ actual: MLXArray, _ expected: MLXArray, _ label: String) {
        XCTAssertEqual(actual.shape, expected.shape, label)
        XCTAssertEqual(actual.dtype, expected.dtype, label)
        eval(actual, expected)
        let left = actual.view(dtype: .uint16).asArray(UInt16.self)
        let right = expected.view(dtype: .uint16).asArray(UInt16.self)
        func ordered(_ value: UInt16) -> UInt16 {
            value & 0x8000 == 0 ? value | 0x8000 : ~value
        }
        let maximum: UInt16 =
            zip(left, right).map { a, b in
                let x = ordered(a)
                let y = ordered(b)
                return x > y ? x - y : y - x
            }.max() ?? 0
        XCTAssertEqual(maximum, 0, "\(label) maxStorageULP=\(maximum)")
        XCTAssertEqual(left, right, label + " storage bits")
    }

    private func indices(rows: Int, pattern: Int) -> MLXArray {
        let ids: [UInt32] = (0 ..< rows * 8).map { p in
            let row = p / 8
            let slot = p % 8
            switch pattern {
            case 0: return UInt32((slot + row) % 12)  // cross-row sharing, unique per row
            case 1: return 3  // every slot shares one expert
            case 2: return UInt32((slot / 3 + row % 2) % 12)  // both duplicate kinds
            default: return UInt32((7 - slot + 2 * row) % 12)  // reversed slot order
            }
        }
        return MLXArray(ids, [1, rows, 8])
    }

    func testEveryShortRowCountAndDuplicatePatternMatchesStockMXFP4() throws {
        let gate = packed(12, 512, 1024, salt: 1)
        let up = packed(12, 512, 1024, salt: 7)
        let down = packed(12, 1024, 512, salt: 11)
        let fused = Matrix(
            weight: concatenated([gate.weight, up.weight], axis: 1),
            scales: concatenated([gate.scales, up.scales], axis: 1))
        for dtype: DType in [.bfloat16, .float16] {
            for rows in 1 ... 7 {
                let input = (0 ..< rows * 1024).map { sin(Float(($0 * 17 + 13) % 509)) * 0.2 }
                let x = MLXArray(input, [1, rows, 1024]).asType(dtype)
                for pattern in 0 ... 3 {
                    let ids = indices(rows: rows, pattern: pattern)
                    let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
                    let referenceActivation = compiledSiluProduct(
                        gather(expanded, gate, ids),
                        gather(expanded, up, ids))
                    let referenceOutput = gather(referenceActivation, down, ids).squeezed(axis: -2)
                    for fusion in [false, true] {
                        let actual = try XCTUnwrap(
                            MiMoV26DecodeExperts.project(
                                x, indices: ids, gate: fusion ? fused : gate,
                                up: fusion ? fused : up, down: down, upOffset: fusion ? 512 : 0))
                        exact(
                            actual.activation, referenceActivation.squeezed(axis: -2),
                            "activation \(dtype) rows\(rows) pattern\(pattern) fused\(fusion)")
                        exact(
                            actual.output, referenceOutput,
                            "down \(dtype) rows\(rows) pattern\(pattern) fused\(fusion)")
                    }
                }
            }
        }
    }

    func testActualFusedNonlinearityAcrossEveryFiniteLowPrecisionGate() {
        let kernel = MLXFast.metalKernel(
            name: "mimo_v26_decode_swiglu_contract_test", inputNames: ["g", "u"],
            outputNames: ["y"],
            source: "uint i = thread_position_in_grid.x; y[i] = mimo_swiglu(g[i], u[i]);",
            header: MiMoV26DecodeExpertMetal.header)
        for dtype: DType in [.bfloat16, .float16] {
            let mask: UInt16 = dtype == .bfloat16 ? 0x7f80 : 0x7c00
            let bits = (0 ... 65535).map(UInt16.init).filter { $0 & mask != mask }
            let gates = MLXArray(bits).view(dtype: dtype)
            let fractions: [Float] = [0.125, 0.75, 1.0625, -1.25, 3.140625]
            let ups = MLXArray(bits.indices.map { fractions[$0 % fractions.count] }).asType(dtype)
            let actual = kernel(
                [gates, ups], grid: (bits.count, 1, 1), threadGroup: (128, 1, 1),
                outputShapes: [[bits.count]], outputDTypes: [dtype])[0]
            exact(actual, compiledSiluProduct(gates, ups), "every finite \(dtype) gate")
        }
    }

    func testReloadAndNoncontiguousInputDoNotUseStaleMatrices() throws {
        let gate = packed(8, 512, 512, salt: 2)
        let up = packed(8, 512, 512, salt: 3)
        let down = packed(8, 512, 512, salt: 4)
        let newGate = packed(8, 512, 512, salt: 5)
        let ids = MLXArray((0 ..< 24).map { UInt32($0 % 8) }, [1, 3, 8])
        let x = MLXArray((0 ..< 1536).map { cos(Float($0)) * 0.1 }, [1, 512, 3])
            .asType(.bfloat16).transposed(0, 2, 1)
        for current in [gate, newGate] {
            let actual = try XCTUnwrap(
                MiMoV26DecodeExperts.project(
                    x, indices: ids, gate: current, up: up, down: down))
            let input = MLX.expandedDimensions(x, axes: [-2, -3])
            let act = compiledSiluProduct(gather(input, current, ids), gather(input, up, ids))
            exact(actual.output, gather(act, down, ids).squeezed(axis: -2), "fresh matrix")
        }
    }

    func testStockOwnedQuantizedModulesEngageAndKeepOriginalParameterObjects() throws {
        let glu = SwitchGLU(
            inputDims: 512, hiddenDims: 512, numExperts: 8,
            weightedReductionProfile: .mimoV26FP32)
        quantize(
            model: glu,
            filter: { _, module in
                module is SwitchLinear ? (groupSize: 32, bits: 4, mode: .mxfp4) : nil
            })
        let gate = packed(8, 512, 512, salt: 2)
        let up = packed(8, 512, 512, salt: 4)
        let down = packed(8, 512, 512, salt: 6)
        let leaves = ["gate_proj": gate, "up_proj": up, "down_proj": down]
        let parameters = glu.parameters().flattened().map { name, original in
            let parts = name.split(separator: ".")
            guard parts.count == 2, let matrix = leaves[String(parts[0])] else {
                return (name, original)
            }
            return (name, parts[1] == "weight" ? matrix.weight : matrix.scales)
        }
        try glu.update(parameters: .unflattened(parameters), verify: .all)
        let before = Dictionary(uniqueKeysWithValues: glu.parameters().flattened())
        let x = MLXArray((0 ..< 1536).map { sin(Float($0)) * 0.15 }, [1, 3, 512]).asType(.bfloat16)
        let ids = MLXArray((0 ..< 24).map { UInt32(($0 / 3) % 8) }, [1, 3, 8])
        let result = try XCTUnwrap(
            MiMoV26DecodeExperts.tryProject(x, indices: ids, glu: glu, enabled: true))
        let input = MLX.expandedDimensions(x, axes: [-2, -3])
        let act = compiledSiluProduct(gather(input, gate, ids), gather(input, up, ids))
        exact(result.activation, act.squeezed(axis: -2), "stock module activation")
        exact(result.output, gather(act, down, ids).squeezed(axis: -2), "stock module output")
        for (name, array) in glu.parameters().flattened() {
            XCTAssertTrue(before[name] === array, "parameter view/requantization change: \(name)")
        }
    }

    func testActualCPUStreamAndUnsupportedPackedLayoutsRefuse() throws {
        let matrix = packed(8, 512, 512, salt: 1)
        let x = MLXArray.ones([1, 1, 512], dtype: .bfloat16)
        let ids = MLXArray((0 ..< 8).map(UInt32.init), [1, 1, 8])
        Device.withDefaultDevice(.gpu) {
            Stream.withNewDefaultStream(device: .cpu) {
                XCTAssertEqual(Device.defaultDevice().deviceType, .gpu)
                XCTAssertFalse(MiMoV26DecodeExperts.gpuStream(.default))
                XCTAssertNil(
                    MiMoV26DecodeExperts.project(
                        x, indices: ids, gate: matrix, up: matrix, down: matrix))
            }
        }
        let wrongScales = Matrix(weight: matrix.weight, scales: matrix.scales.asType(.bfloat16))
        XCTAssertNil(
            MiMoV26DecodeExperts.project(
                x, indices: ids, gate: wrongScales, up: matrix, down: matrix))
        XCTAssertNil(
            MiMoV26DecodeExperts.project(
                x, indices: ids, gate: matrix, up: matrix, down: matrix, upOffset: 1))
        let ordinary = SwitchGLU(
            inputDims: 512, hiddenDims: 512, numExperts: 8,
            weightedReductionProfile: .mimoV26FP32)
        XCTAssertNil(
            MiMoV26DecodeExperts.tryProject(x, indices: ids, glu: ordinary, enabled: true),
            "dense leaves cannot enter MXFP4 kernels")
        XCTAssertNil(
            MiMoV26DecodeExperts.tryProject(x, indices: ids, glu: ordinary, enabled: false))
    }
}
