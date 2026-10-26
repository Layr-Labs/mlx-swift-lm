import Cmlx
import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Testing

@testable import MLXLLM

extension KernelTests {
    @Suite("Gemma 4 bounded command buffers", .serialized)
    struct Gemma4DecodeCommandBufferTests {
        private func config() throws -> Gemma4TextConfiguration {
            let json = """
                {
                    "hidden_size": 2816, "num_hidden_layers": 30,
                    "enable_moe_block": true, "num_experts": 128,
                    "top_k_experts": 8, "moe_intermediate_size": 704,
                    "use_bidirectional_attention": "vision",
                    "quantization": {"bits": 4, "group_size": 64}
                }
                """
            return try JSONDecoder().decode(
                Gemma4TextConfiguration.self, from: Data(json.utf8))
        }

        private func currentLimits() -> GPU.CommandBufferLimits {
            var operations: Int32 = 0
            var megabytes: Int32 = 0
            mlx_metal_set_thread_command_buffer_limits(0, 0, &operations, &megabytes)
            mlx_metal_set_thread_command_buffer_limits(operations, megabytes, nil, nil)
            return .init(operations: Int(operations), megabytes: Int(megabytes))
        }

        private func expectExact(_ actual: MLXArray, _ expected: MLXArray) {
            eval(actual, expected)
            #expect(actual.shape == expected.shape)
            #expect(actual.dtype == expected.dtype)
            #expect(actual.asData(access: .copy).data == expected.asData(access: .copy).data)
        }

        @Test func onlyTargetGeometryAndQuantizationAreEligible() throws {
            let target = try config()
            #expect(
                Gemma4DecodeCommandBuffers.resolve(target, device: .gpu)
                    == Gemma4DecodeCommandBuffers.limits)
            let changes: [(inout Gemma4TextConfiguration) -> Void] = [
                { $0.hiddenSize = 2048 }, { $0.numHiddenLayers = 29 },
                { $0.enableMoeBlock = false }, { $0.numExperts = 64 },
                { $0.topKExperts = 4 }, { $0.moeIntermediateSize = 768 },
                { $0.useBidirectionalAttention = nil },
                { $0.quantizationBits = nil }, { $0.quantizationBits = 8 },
                { $0.quantizationGroupSize = 32 }, { $0.quantizationMode = .mxfp4 },
            ]
            for change in changes {
                var other = target
                change(&other)
                #expect(Gemma4DecodeCommandBuffers.resolve(other, device: .gpu) == nil)
            }
            // Compile-time coverage that the canonical text tower exposes the
            // protocol the existing text/VLM serving adapter forwards.
            func provider<T: CBv2DecodeCommandBufferLimitsProviding>(_: T.Type) {}
            provider(Gemma4TextModel.self)
        }

        @Test func vlmRootQuantizationOverlayAndDenseOverridesRemainEligible() throws {
            var target = try config()
            target.quantizationBits = nil
            target.quantizationGroupSize = nil
            #expect(Gemma4DecodeCommandBuffers.resolve(target, device: .gpu) == nil)
            target.mergeQuantization(BaseConfiguration.Quantization(groupSize: 64, bits: 4))
            #expect(
                Gemma4DecodeCommandBuffers.resolve(target, device: .gpu)
                    == Gemma4DecodeCommandBuffers.limits)

            let encoded = try JSONEncoder().encode(target)
            var json = try #require(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
            json["quantization"] = [
                "bits": 4, "group_size": 64,
                "model.layers.0.mlp.gate_proj": ["bits": 8, "group_size": 64],
            ]
            let mixed = try JSONDecoder().decode(
                Gemma4TextConfiguration.self,
                from: JSONSerialization.data(withJSONObject: json))
            #expect(
                Gemma4DecodeCommandBuffers.resolve(mixed, device: .gpu)
                    == Gemma4DecodeCommandBuffers.limits)
        }

        @Test func cpuDefaultStreamsKeepDefaultPolicy() throws {
            let target = try config()
            #expect(Gemma4DecodeCommandBuffers.resolve(target, device: .cpu) == nil)
            Stream.withNewDefaultStream(device: .cpu) {
                #expect(Device.defaultDevice().deviceType == .cpu)
                #expect(Gemma4DecodeCommandBuffers.resolve(target) == nil)
            }
        }

        @Test func nestedScopesRestorePreviousLimits() {
            let before = currentLimits()
            let outer = GPU.CommandBufferLimits(operations: 17, megabytes: 23)
            GPU.withCommandBufferLimits(outer) {
                #expect(currentLimits() == outer)
                GPU.withCommandBufferLimits(Gemma4DecodeCommandBuffers.limits) {
                    #expect(currentLimits() == Gemma4DecodeCommandBuffers.limits)
                }
                #expect(currentLimits() == outer)
            }
            #expect(currentLimits() == before)
        }

        @Test func throwingOrEarlyReturnRestoresPreviousLimits() {
            struct Interrupted: Error {}
            let before = currentLimits()
            do {
                try GPU.withCommandBufferLimits(Gemma4DecodeCommandBuffers.limits) {
                    #expect(currentLimits() == Gemma4DecodeCommandBuffers.limits)
                    throw Interrupted()
                }
                Issue.record("The scoped interruption was not thrown")
            } catch is Interrupted {
            } catch {
                Issue.record("Unexpected error: \(error)")
            }
            #expect(currentLimits() == before)
            let result = GPU.withCommandBufferLimits(Gemma4DecodeCommandBuffers.limits) { false }
            #expect(!result)
            #expect(currentLimits() == before)
        }

        @Test func scopesDoNotLeakOntoOtherThreads() {
            final class Observation: @unchecked Sendable {
                var limits: GPU.CommandBufferLimits?
            }
            let observation = Observation()
            let done = DispatchSemaphore(value: 0)
            let before = currentLimits()
            GPU.withCommandBufferLimits(Gemma4DecodeCommandBuffers.limits) {
                Thread {
                    var operations: Int32 = 0
                    var megabytes: Int32 = 0
                    mlx_metal_set_thread_command_buffer_limits(0, 0, &operations, &megabytes)
                    mlx_metal_set_thread_command_buffer_limits(operations, megabytes, nil, nil)
                    observation.limits = .init(
                        operations: Int(operations), megabytes: Int(megabytes))
                    done.signal()
                }.start()
                let completed = done.wait(timeout: .now() + 5)
                #expect(completed == .success)
                if completed == .success {
                    #expect(observation.limits == .init(operations: 0, megabytes: 0))
                }
                #expect(currentLimits() == Gemma4DecodeCommandBuffers.limits)
            }
            #expect(currentLimits() == before)
        }

        private struct QuantizedFixture {
            let input: MLXArray
            let weight: MLXArray
            let scales: MLXArray
            let biases: MLXArray?

            init(rows: Int, dtype: DType) {
                input = MLXRandom.normal([rows, 128], key: MLXRandom.key(9401)).asType(dtype)
                let source =
                    (MLXRandom.normal([128, 128], key: MLXRandom.key(9402)) / 16).asType(dtype)
                let quantization = quantized(source, groupSize: 64, bits: 4)
                weight = quantization.wq
                scales = quantization.scales
                biases = quantization.biases
                eval(input, weight, scales)
                if let biases { eval(biases) }
            }

            // Repeated quantized projection, activation, product and residual
            // nodes cross the candidate's operation commit threshold. This is
            // an arithmetic/lifetime fixture, not a miniature model benchmark.
            func graph() -> MLXArray {
                var value = input
                for _ in 0 ..< 24 {
                    let projected = quantizedMM(
                        value, weight, scales: scales, biases: biases,
                        groupSize: 64, bits: 4)
                    value = tanh(projected) * 0.125 + value * 0.875
                }
                return value
            }
        }

        @Test func boundedWidthsPreserveNativeBytesAcrossPartitions() {
            for dtype in [DType.float16, .bfloat16, .float32] {
                // Width one also covers a short plain prompt tail. Widths
                // two through eight can occur in mixed speculative rounds.
                for rows in [1, 2, 4, 8] {
                    let fixture = QuantizedFixture(rows: rows, dtype: dtype)
                    let reference = GPU.withCommandBufferLimits(
                        .init(operations: 0, megabytes: 0)
                    ) {
                        let output = fixture.graph()
                        eval(output)
                        return output
                    }
                    for limits in [
                        GPU.CommandBufferLimits(operations: 1, megabytes: 1),
                        Gemma4DecodeCommandBuffers.limits,
                        GPU.CommandBufferLimits(operations: 200, megabytes: 512),
                    ] {
                        let actual = GPU.withCommandBufferLimits(limits) {
                            let output = fixture.graph()
                            eval(output)
                            return output
                        }
                        expectExact(actual, reference)
                    }
                }
            }
        }

        @Test func asynchronousMixedWidthsSurviveRestorationAndInputRelease() {
            let before = currentLimits()
            var fixtures = [1, 3, 8, 2].map {
                QuantizedFixture(rows: $0, dtype: .bfloat16)
            }
            let expected = fixtures.map { fixture in
                let output = fixture.graph()
                eval(output)
                return output
            }
            // Submit inside the scope, release local fixtures, then wait and
            // inspect output only after the limits have been restored.
            let actual = GPU.withCommandBufferLimits(Gemma4DecodeCommandBuffers.limits) {
                let outputs = fixtures.map { fixture in
                    let output = fixture.graph()
                    asyncEval(output)
                    return output
                }
                fixtures.removeAll()
                return outputs
            }
            #expect(currentLimits() == before)
            for (result, reference) in zip(actual, expected) {
                expectExact(result, reference)
            }
        }
    }
}
