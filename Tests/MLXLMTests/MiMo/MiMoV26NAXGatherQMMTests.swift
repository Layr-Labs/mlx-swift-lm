// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0
// Oracle cases adapted from oMLX PR #3995; no run is claimed by this packet.

import Foundation
import MLX
import MLXRandom
import XCTest

@testable import MLXLMCommon

final class MiMoV26NAXGatherQMMTests: XCTestCase {
    func testProductionGeometryAndSchedules() throws {
        typealias Plan = MiMoV26NAXGatherQMM.Plan
        let chunk2048 = try XCTUnwrap(
            Plan.production(
                rows: 16384, experts: 256, input: 4096, output: 2048))
        let chunk4096 = try XCTUnwrap(
            Plan.production(
                rows: 32768, experts: 256, input: 2048, output: 4096))
        let chunk8192 = try XCTUnwrap(
            Plan.production(
                rows: 65536, experts: 256, input: 4096, output: 4096))
        XCTAssertEqual(chunk2048.tileRows, 64)
        XCTAssertTrue(chunk2048.doubleBuffer)
        XCTAssertEqual(chunk4096.tileRows, 128)
        XCTAssertTrue(chunk4096.doubleBuffer)
        XCTAssertEqual(chunk8192.tileRows, 64)
        XCTAssertFalse(chunk8192.doubleBuffer)
        for args in [
            (56, 256, 4096, 2048), (1023, 256, 4096, 2048),
            (32768, 512, 4096, 2048), (32768, 256, 4096, 192),
            (32768, 256, 96, 4096), (Int(Int32.max), 256, 4096, 2048),
        ] {
            XCTAssertNil(
                Plan.production(
                    rows: args.0, experts: args.1, input: args.2, output: args.3))
        }
    }

    func testOwnerOptInSurvivesTwinsWithoutChangingGenericDefault() {
        let generic = SwitchGLU(inputDims: 64, hiddenDims: 64, numExperts: 8)
        let mimo = SwitchGLU(
            inputDims: 64, hiddenDims: 64, numExperts: 8,
            mimoV26NAXGather: true)
        XCTAssertFalse(generic.mimoV26NAXGather)
        XCTAssertFalse(generic.fusingGateUp().splittingGateUp().mimoV26NAXGather)
        XCTAssertTrue(mimo.mimoV26NAXGather)
        XCTAssertTrue(mimo.fusingGateUp().splittingGateUp().mimoV26NAXGather)
    }

    private func requireGPU() throws {
        guard ProcessInfo.processInfo.environment["DARKBLOOM_TEST_MIMO_NAX_GATHER"] == "1"
        else { throw XCTSkip("Requires explicitly owned NAX GPU test lane") }
        guard MiMoV26NAXGatherQMM.gpuStream(.default),
            MiMoV26NAXGatherQMM.naxAvailable
        else { throw XCTSkip("Current stream/build/device cannot execute NAX") }
    }

    private func fixture(counts: [Int], input: Int, output: Int, dtype: DType)
        -> (x: MLXArray, indices: MLXArray, weight: MLXArray, scales: MLXArray)
    {
        let experts = counts.count
        let indices = counts.enumerated().flatMap {
            Array(repeating: UInt32($0.offset), count: $0.element)
        }
        let weights =
            MLXRandom.normal(
                [experts, output, input], key: MLXRandom.key(0x2267)
            ).asType(dtype) * 0.05
        let (packed, scales, biases) = MLX.quantized(
            weights, groupSize: 32, bits: 4, mode: .mxfp4)
        XCTAssertNil(biases)
        let x =
            (MLXRandom.normal(
                [indices.count, 1, input], key: MLXRandom.key(0x2268)) * 0.5).asType(dtype)
        return (x, MLXArray(indices), packed, scales)
    }

    private func stock(
        _ x: MLXArray, _ indices: MLXArray,
        _ weight: MLXArray, _ scales: MLXArray
    ) -> MLXArray {
        MLX.gatherQuantizedMM(
            x, weight, scales: scales, biases: nil, rhsIndices: indices,
            transpose: true, groupSize: 32, bits: 4, mode: .mxfp4,
            sortedIndices: true)
    }

    /// Native output ULPs, including signed zero: no absolute/relative tolerance.
    private func maxULP(_ actual: MLXArray, _ expected: MLXArray) -> UInt32 {
        let a = actual.asType(.float32).asArray(Float.self)
        let b = expected.asType(.float32).asArray(Float.self)
        XCTAssertEqual(a.count, b.count)
        var maximum: UInt32 = 0
        for (a, b) in zip(a, b) {
            XCTAssertTrue(a.isFinite && b.isFinite)
            let ar: UInt16 =
                actual.dtype == .bfloat16
                ? UInt16(truncatingIfNeeded: a.bitPattern >> 16) : Float16(a).bitPattern
            let br: UInt16 =
                expected.dtype == .bfloat16
                ? UInt16(truncatingIfNeeded: b.bitPattern >> 16) : Float16(b).bitPattern
            let ai = (ar & 0x8000) == 0 ? UInt32(ar) + 0x8000 : UInt32(~ar)
            let bi = (br & 0x8000) == 0 ? UInt32(br) + 0x8000 : UInt32(~br)
            maximum = max(maximum, ai > bi ? ai - bi : bi - ai)
        }
        return maximum
    }

    private func assertExact(
        _ actual: MLXArray, _ expected: MLXArray,
        label: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        eval(actual, expected)
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, file: file, line: line)
        let ulp = maxULP(actual, expected)
        print("MiMoNAX exact-oracle \(label) dtype=\(actual.dtype) maxULP=\(ulp)")
        XCTAssertEqual(ulp, 0, file: file, line: line)
        XCTAssertEqual(actual.asData().data, expected.asData().data, file: file, line: line)
    }

    func testAlignedNativeDtypesSchedulesAndTailsAreBitExact() throws {
        try requireGPU()
        // Empty experts; every 16-row boundary; runs crossing 64 and 128 rows.
        let counts = [70, 0, 5, 33, 64, 17, 1, 130, 0, 11, 15, 16, 31, 32, 63, 127]
        for dtype in [DType.bfloat16, .float16] {
            let f = fixture(counts: counts, input: 256, output: 128, dtype: dtype)
            let reference = stock(f.x, f.indices, f.weight, f.scales)
            let plan = MiMoV26NAXGatherQMM.Plan(
                rows: f.indices.size, experts: counts.count, input: 256, output: 128)
            for bm in [64, 128] {
                for db in [false, true] {
                    let actual = MiMoV26NAXGatherQMM.launch(
                        x: f.x, indices: f.indices, weight: f.weight, scales: f.scales,
                        plan: plan, tileRows: bm, doubleBuffer: db)
                    assertExact(actual, reference, label: "bm=\(bm) db=\(db)")
                }
            }
        }
    }

    func testMoreThan32768RowsAgainstIndependentStockSlices() throws {
        try requireGPU()
        let counts = Array(repeating: 2304, count: 15) + [4000]
        let f = fixture(counts: counts, input: 128, output: 64, dtype: .bfloat16)
        let half = f.indices.size / 2
        XCTAssertGreaterThan(f.indices.size, 32768)
        let reference = concatenated([
            stock(f.x[..<half], f.indices[..<half], f.weight, f.scales),
            stock(f.x[half...], f.indices[half...], f.weight, f.scales),
        ])
        let plan = MiMoV26NAXGatherQMM.Plan(
            rows: f.indices.size, experts: counts.count, input: 128, output: 64)
        for bm in [64, 128] {
            for db in [false, true] {
                let actual = MiMoV26NAXGatherQMM.launch(
                    x: f.x, indices: f.indices, weight: f.weight, scales: f.scales,
                    plan: plan, tileRows: bm, doubleBuffer: db)
                assertExact(actual, reference, label: "oversized bm=\(bm) db=\(db)")
            }
        }
    }

    func testSparseFirstAndLastExpertAndStridedActivationInput() throws {
        try requireGPU()
        for counts in [
            [129] + Array(repeating: 0, count: 31),
            Array(repeating: 0, count: 31) + [300],
        ] {
            let f = fixture(counts: counts, input: 128, output: 64, dtype: .bfloat16)
            let backing = concatenated([f.x, f.x * 0], axis: -1)
            let strided = backing[.ellipsis, ..<128]
            let reference = stock(f.x, f.indices, f.weight, f.scales)
            let plan = MiMoV26NAXGatherQMM.Plan(
                rows: f.indices.size, experts: counts.count, input: 128, output: 64)
            for bm in [64, 128] {
                let actual = MiMoV26NAXGatherQMM.launch(
                    x: strided, indices: f.indices, weight: f.weight, scales: f.scales,
                    plan: plan, tileRows: bm)
                assertExact(actual, reference, label: "sparse/strided bm=\(bm)")
            }
        }
    }

    func testCPUStreamDeclinesBeforeKernelCreation() {
        let before = MiMoV26NAXGatherDiagnostics.encodedCalls()
        Device.withDefaultDevice(.cpu) {
            XCTAssertFalse(MiMoV26NAXGatherQMM.gpuStream(.default))
        }
        XCTAssertEqual(MiMoV26NAXGatherDiagnostics.encodedCalls(), before)
    }
}
