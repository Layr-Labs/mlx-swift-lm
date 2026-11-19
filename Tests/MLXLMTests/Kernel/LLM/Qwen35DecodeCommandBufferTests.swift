// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    @Suite("Qwen 3.5 decode command buffers", .serialized)
    struct Qwen35DecodeCommandBufferTests {
        /// The dense variant of `Qwen35MoEForwardPassTests`' tiny hybrid: two
        /// Gated DeltaNet layers and two full-attention layers.
        private static func dense(seed: UInt64 = 3) throws -> Qwen35Model {
            var base = Qwen35MoEForwardPassTests.base
            var text = base["text_config"] as! [String: Any]
            for key in [
                "num_experts", "num_experts_per_tok", "moe_intermediate_size",
                "shared_expert_intermediate_size", "norm_topk_prob",
            ] {
                text.removeValue(forKey: key)
            }
            text["model_type"] = "qwen3_5_text"
            base["model_type"] = "qwen3_5"
            base["text_config"] = text
            let model = Qwen35Model(
                try SyntheticModel.configuration(Qwen35Configuration.self, base))
            SyntheticModel.randomize(model, seed: seed)
            return model
        }

        /// The tiny MoE hybrid, its projections and expert banks packed in 4 bits
        /// (groups of 32), as the served 35B-A3B builds' banks are packed.
        private static func quantizedMoE() throws -> Qwen35MoEModel {
            let model = try Qwen35MoEForwardPassTests.makeModel(seed: 5)
            quantize(model: model, groupSize: 32, bits: 4)
            eval(model)
            return model
        }

        @Test func declaresDecodeLimitsThroughTheSteppableAdapter() throws {
            defer { Qwen35DecodeCommandBuffers.overrideForTesting = nil }
            Qwen35DecodeCommandBuffers.overrideForTesting = true
            for model in [try Self.dense() as Qwen35Model, try Self.quantizedMoE()] {
                let adapter = CBv2SteppableLanguageModelAdapter(model)
                #expect(adapter.decodeCommandBufferLimits == Qwen35DecodeCommandBuffers.limits)
            }
            #expect(Qwen35DecodeCommandBuffers.limits.operations == 100)
            #expect(Qwen35DecodeCommandBuffers.limits.megabytes == 1 << 20)
        }

        @Test func declaresMTPRoundLimitsThroughTheSteppableAdapter() throws {
            defer { Qwen35DecodeCommandBuffers.speculativeOverrideForTesting = nil }
            Qwen35DecodeCommandBuffers.speculativeOverrideForTesting = true
            #expect(EngineLoopV2.speculativeCommandBufferRowTokens == 8)
            for model in [try Self.dense() as Qwen35Model, try Self.quantizedMoE()] {
                let adapter = CBv2SteppableLanguageModelAdapter(model)
                #expect(adapter.speculativeCommandBufferLimits == Qwen35DecodeCommandBuffers.limits)
            }
            Qwen35DecodeCommandBuffers.speculativeOverrideForTesting = false
            let adapter = CBv2SteppableLanguageModelAdapter(try Self.dense())
            #expect(adapter.speculativeCommandBufferLimits == nil)
            #expect(
                Qwen35DecodeCommandBuffers.speculativeEnvironmentKey
                    == "DARKBLOOM_QWEN35_MTP_COMMAND_BUFFERS")
        }

        /// MTP verification rows (a few rows of 2 to 8 tokens, the shapes the round's
        /// target forward takes) under the limits against a commit every two kernel
        /// dispatches: the same bytes.
        @Test func mtpRoundSwitchIsOptIn() {
            #expect(!Qwen35DecodeCommandBuffers.speculativeEnabled(nil))
            #expect(!Qwen35DecodeCommandBuffers.speculativeEnabled(""))
            #expect(!Qwen35DecodeCommandBuffers.speculativeEnabled("0"))
            #expect(!Qwen35DecodeCommandBuffers.speculativeEnabled("off"))
            #expect(Qwen35DecodeCommandBuffers.speculativeEnabled("1"))
            #expect(Qwen35DecodeCommandBuffers.speculativeEnabled(" On "))
            #expect(Qwen35DecodeCommandBuffers.speculativeEnabled("YES"))
        }

        @Test func verificationRowLimitsKeepExactBytes() throws {
            let everyOperation = GPU.CommandBufferLimits(operations: 1, megabytes: 1)
            for model in [try Self.dense() as Qwen35Model, try Self.quantizedMoE()] {
                for (rows, width) in [(1, 2), (1, 4), (2, 4), (1, 8)] {
                    let tokens = MLXArray((0 ..< rows * width).map { Int32(($0 * 7 + 3) % 64) })
                        .reshaped(rows, width)
                    func logits(_ limits: GPU.CommandBufferLimits) -> Data {
                        GPU.withCommandBufferLimits(limits) {
                            let out = model(tokens, cache: nil)
                            eval(out)
                            return out.asData(access: .copy).data
                        }
                    }
                    #expect(
                        logits(everyOperation) == logits(Qwen35DecodeCommandBuffers.limits),
                        "rows \(rows) width \(width)")
                }
            }
        }

        @Test func theSwitchRestoresMLXDefaults() throws {
            #expect(Qwen35DecodeCommandBuffers.enabled(nil))
            for on in ["", " ", "1", "on", "true", "YES", "anything"] {
                #expect(Qwen35DecodeCommandBuffers.enabled(on), "\(on)")
            }
            for off in ["0", "off", "false", "no", " OFF ", "No"] {
                #expect(!Qwen35DecodeCommandBuffers.enabled(off), "\(off)")
            }
            defer { Qwen35DecodeCommandBuffers.overrideForTesting = nil }
            Qwen35DecodeCommandBuffers.overrideForTesting = false
            let adapter = CBv2SteppableLanguageModelAdapter(try Self.dense())
            #expect(adapter.decodeCommandBufferLimits == nil)
            Qwen35DecodeCommandBuffers.overrideForTesting = true
            #expect(adapter.decodeCommandBufferLimits == Qwen35DecodeCommandBuffers.limits)
        }

        /// One prompt and four one-token steps through a fresh cache under `limits`;
        /// every step's logits, as bytes.
        private static func decodeBytes(
            _ model: Qwen35Model, limits: GPU.CommandBufferLimits
        ) throws -> [Data] {
            let cache = try model.newCache(parameters: nil)
            let prompt = MLXArray((0 ..< 7).map { Int32(($0 * 5 + 3) % 64) }).reshaped(1, -1)
            var steps: [Data] = []
            var logits = GPU.withCommandBufferLimits(limits) {
                let out = model(prompt, cache: cache)
                eval(out)
                return out
            }
            steps.append(logits.asData(access: .copy).data)
            for _ in 0 ..< 4 {
                let next = argMax(logits[0..., -1, 0...], axis: -1).reshaped(1, 1)
                logits = GPU.withCommandBufferLimits(limits) {
                    let out = model(next, cache: cache)
                    eval(out)
                    return out
                }
                steps.append(logits.asData(access: .copy).data)
            }
            return steps
        }

        /// The decode limits against a commit every two kernel dispatches (the finest
        /// split MLX allows: it commits once a buffer holds more than `operations`
        /// dispatches): the same bytes at every step.
        @Test func decodeLimitsKeepExactBytes() throws {
            let everyOperation = GPU.CommandBufferLimits(operations: 1, megabytes: 1)
            for model in [try Self.dense() as Qwen35Model, try Self.quantizedMoE()] {
                let expected = try Self.decodeBytes(model, limits: everyOperation)
                let actual = try Self.decodeBytes(model, limits: Qwen35DecodeCommandBuffers.limits)
                #expect(expected.count == 5)
                #expect(expected == actual)
            }
        }
    }
}
