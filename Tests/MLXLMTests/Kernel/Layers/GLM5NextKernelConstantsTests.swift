// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// Cached kernel arguments must be indistinguishable from the per-call
    /// arrays they replace (shape, dtype, bytes), shared per value, and bounded.
    @Suite struct GLM5NextKernelConstantsTests {
        private static func same(_ cached: MLXArray, _ fresh: MLXArray) -> Bool {
            guard cached.shape == fresh.shape, cached.ndim == fresh.ndim,
                cached.dtype == fresh.dtype
            else { return false }
            switch fresh.dtype {
            case .float32:
                return cached.asArray(Float.self).map(\.bitPattern)
                    == fresh.asArray(Float.self).map(\.bitPattern)
            case .int32:
                return cached.asArray(Int32.self) == fresh.asArray(Int32.self)
            default:
                return false
            }
        }

        @Test func cachedArraysEqualPerCallArrays() {
            let constants = GLM5NextKernelConstants()
            let floats: [Float] = [-5, 1e-5, 1 / Float(128).squareRoot(), 1 / 128, -0.0, 1]
            #expect(Self.same(constants.floats(floats), MLXArray(floats)))
            #expect(Self.same(constants.float(0.125), MLXArray(Float(0.125))))
            #expect(Self.same(constants.int32s([4099, 2048]), MLXArray([Int32(4099), 2048])))
            #expect(Self.same(constants.int32(7), MLXArray(Int32(7))))
            #expect(constants.float(1).ndim == 0 && constants.int32(1).ndim == 0)
            #expect(constants.floats([1]).shape == [1] && constants.int32s([1]).shape == [1])
        }

        @Test func oneArrayPerValueShapeAndType() {
            let constants = GLM5NextKernelConstants()
            #expect(constants.floats([1, 2]) === constants.floats([1, 2]))
            #expect(constants.int32(3) === constants.int32(3))
            // A 0-D scalar is a different kernel argument from a [1] vector.
            #expect(constants.float(1) !== constants.floats([1]))
            #expect(constants.int32(1) !== constants.int32s([1]))
            // Same bits, different dtype.
            #expect(constants.int32(0) !== constants.float(0))
            // -0 and +0 are different bytes.
            #expect(constants.float(-0.0) !== constants.float(0))
            #expect(constants.count == 9)
        }

        @Test func theCacheStaysBounded() {
            let constants = GLM5NextKernelConstants()
            let fixed = constants.floats([0.5, 0.25])
            for position in 0 ..< 3 * GLM5NextKernelConstants.capacity {
                #expect(constants.int32(Int32(position)).item(Int32.self) == Int32(position))
                #expect(constants.count <= GLM5NextKernelConstants.capacity)
            }
            // An array handed out before a clear stays valid; the value is rebuilt.
            #expect(fixed.asArray(Float.self) == [0.5, 0.25])
            #expect(Self.same(constants.floats([0.5, 0.25]), fixed))
        }
    }
}
