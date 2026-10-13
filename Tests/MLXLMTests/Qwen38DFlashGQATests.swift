// Copyright 2026 bstnxbt
// SPDX-License-Identifier: Apache-2.0

import MLX
import XCTest

@testable import MLXLLM

final class Qwen38DFlashGQATests: XCTestCase {
    func testRetainedRouteIsOnlyWidthsSixThroughEightAt16KBand() {
        for width in 1 ... 16 {
            XCTAssertEqual(
                Qwen38DFlashGQARoute.usesPerHead(
                    queryLength: width, cachedLength: 16_384),
                (6 ... 8).contains(width))
        }
        XCTAssertFalse(
            Qwen38DFlashGQARoute.usesPerHead(
                queryLength: 6, cachedLength: 16_383))
        XCTAssertTrue(
            Qwen38DFlashGQARoute.usesPerHead(
                queryLength: 7, cachedLength: 32_767))
        XCTAssertFalse(
            Qwen38DFlashGQARoute.usesPerHead(
                queryLength: 8, cachedLength: 32_768))
    }

    func testPerHeadWidthsMatchNativeGQAAtRetainedContext() throws {
        let keyLength = 16_384
        let scale = pow(Float(256), -0.5)
        for width in 6 ... 8 {
            MLXRandom.seed(UInt64(0x6A00 + width))
            let queries = MLXRandom.normal([1, 24, width, 256]).asType(.bfloat16)
            let keys = MLXRandom.normal([1, 4, keyLength, 256]).asType(.bfloat16)
            let values = MLXRandom.normal([1, 4, keyLength, 256]).asType(.bfloat16)
            let queryPositions = arange(
                keyLength - width, keyLength
            ).expandedDimensions(axis: -1)
            let keyPositions = arange(keyLength).expandedDimensions(axis: 0)
            let tailCausalMask = keyPositions .<= queryPositions
            let tailCausalAdditive = MLX.where(
                tailCausalMask,
                MLXArray(0, dtype: .bfloat16),
                MLXArray(-Float.greatestFiniteMagnitude, dtype: .bfloat16))
            let expected = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .array(tailCausalAdditive))
            let actual = qwen38DFlashGroupedGQA(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale)
            eval(expected, actual)
            XCTAssertEqual(actual.shape, expected.shape)
            let maximumAbsoluteError = abs(
                actual.asType(.float32) - expected.asType(.float32)
            ).max().item(Float.self)
            // The source MLX 0.32.0 per-head route differs from native GQA by
            // at most two BF16 ULPs on this exact retained geometry.
            XCTAssertLessThanOrEqual(
                maximumAbsoluteError, 0.000_244_140_625,
                "width=\(width) max_abs=\(maximumAbsoluteError)")
        }
    }
}
