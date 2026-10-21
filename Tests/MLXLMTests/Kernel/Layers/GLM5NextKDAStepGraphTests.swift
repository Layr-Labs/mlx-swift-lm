// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {
    /// What a KDA step builds: the fused step applies the low-rank f_b/g_b
    /// rows in-kernel, so it must not build them as quantized matmuls (dead
    /// graph nodes on every decode step). Paths that read them still do.
    @Suite(.serialized)
    struct GLM5NextKDAStepGraphTests {
        private typealias Counting = GLM5NextCountingQuantizedLinear

        private static func instrument(_ layer: GLM5NextKDASublayer) throws -> (Counting, Counting)
        {
            let forget = Counting(try #require(layer.fB as? QuantizedLinear))
            let output = Counting(try #require(layer.gB as? QuantizedLinear))
            layer.update(
                modules: ModuleChildren.unflattened([("f_b_proj", forget), ("g_b_proj", output)]))
            #expect(layer.fB === forget && layer.gB === output)
            return (forget, output)
        }

        private static func step(
            _ layer: GLM5NextKDASublayer, _ fixture: GLM5NextKDALayerFixture,
            cache: GLM5NextKDACache, tokens: Int, seed: UInt64
        ) throws -> MLXArray {
            let x = fixture.input(tokens: tokens, seed: seed)
            let values = try layer.forward(
                x, cache: cache, validity: MLXArray.ones([1, tokens], dtype: .bool))
            try withError { error in
                eval([values] + cache.innerState())
                try error.check()
            }
            return values
        }

        @Test func fusedStepsBuildNoGateMatmulsOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try GLM5NextKDALayerFixture()
                let layer = try fixture.sublayer(decodeStep: .fusedEveryStep)
                let (forget, output) = try Self.instrument(layer)
                let cache = GLM5NextKDACache()
                // A fresh cache's first step (no state), decode steps and
                // verification-sized steps: all fused, gates in-kernel.
                for (index, tokens) in [1, 1, 3, 8, 1].enumerated() {
                    let values = try Self.step(
                        layer, fixture, cache: cache, tokens: tokens, seed: 40 + UInt64(index))
                    #expect(isFinite(values).all().item(Bool.self), "T\(tokens) finite")
                    #expect(
                        forget.calls == 0 && output.calls == 0,
                        "T\(tokens): f_b \(forget.calls) g_b \(output.calls) matmuls built")
                }
                // A long step runs f_b/g_b as quantized GEMMs, once each.
                _ = try Self.step(layer, fixture, cache: cache, tokens: 20, seed: 60)
                #expect(forget.calls == 1 && output.calls == 1, "long step")
            }
        }

        @Test func composedStepsStillBuildTheGateMatmulsOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try GLM5NextKDALayerFixture()
                let layer = try fixture.sublayer(decodeStep: .composed)
                let (forget, output) = try Self.instrument(layer)
                let cache = GLM5NextKDACache()
                for (index, tokens) in [1, 3].enumerated() {
                    _ = try Self.step(
                        layer, fixture, cache: cache, tokens: tokens, seed: 70 + UInt64(index))
                    #expect(
                        forget.calls == index + 1 && output.calls == index + 1, "T\(tokens)")
                }
            }
        }

        /// The fused step's first call (no retained state) equals a call with
        /// an explicit zero state, bit for bit.
        @Test func aMissingStateIsTheZeroStateOnGPU() throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try GLM5NextKDALayerFixture()
                let layer = try fixture.sublayer(decodeStep: .fusedEveryStep)
                let linear = fixture.text.linearAttention
                let width = linear.numHeads * linear.headDim
                let x = fixture.input(tokens: 1, seed: 90)
                let valid = MLXArray.ones([1, 1], dtype: .bool)
                let fresh = try layer.forward(x, state: nil, validity: valid)
                let zero = try layer.forward(
                    x,
                    state: .init(
                        convolution: MLXArray.zeros(
                            [1, 3 * width, linear.shortConvKernelSize], dtype: .bfloat16),
                        recurrent: MLXArray.zeros(
                            [1, linear.numHeads, linear.headDim, linear.headDim])),
                    validity: valid)
                eval(
                    fresh.values, fresh.state.convolution, fresh.state.recurrent, zero.values,
                    zero.state.convolution, zero.state.recurrent)
                for (name, a, b) in [
                    ("values", fresh.values, zero.values),
                    ("convolution", fresh.state.convolution, zero.state.convolution),
                    ("recurrent", fresh.state.recurrent, zero.state.recurrent),
                ] {
                    #expect(
                        a.asType(.float32).asArray(Float.self).map(\.bitPattern)
                            == b.asType(.float32).asArray(Float.self).map(\.bitPattern), "\(name)")
                }
            }
        }
    }
}
