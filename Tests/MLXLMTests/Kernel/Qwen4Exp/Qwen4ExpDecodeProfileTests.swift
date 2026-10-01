import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {

    /// Tests of the Qwen4 decode profiler in `Qwen4ExpDecodeProfile.swift`.
    ///
    /// The profiler reads its flags from the process environment snapshot.
    /// The tests set the flags with `setenv`, read the snapshot again, and
    /// restore the old values at the end. The suite is serialized and CI
    /// runs the tests with `--no-parallel`, so no other test sees the flags.
    ///
    /// The dump hooks (`DARKBLOOM_QWEN4_DUMP_DIR`) read their flags once, at
    /// the first use, in DEBUG builds only. The tests check only that they
    /// do nothing when no dump directory is set.
    @Suite(.serialized)
    struct Qwen4ExpDecodeProfileTests {
        typealias Profile = Qwen4ExpDecodeProfile
        typealias Support = Qwen4ExpKernelSupport

        static let fine = "DARKBLOOM_QWEN4_DECODE_PROFILE"
        static let coarse = "DARKBLOOM_QWEN4_DECODE_COARSE_PROFILE"
        static let names = [
            fine, coarse,
            "DARKBLOOM_QWEN4_DECODE_PROFILE_WARMUP",
            "DARKBLOOM_QWEN4_DECODE_PROFILE_INTERVAL",
            "DARKBLOOM_QWEN4_DECODE_COARSE_PROFILE_WARMUP",
            "DARKBLOOM_QWEN4_DECODE_COARSE_PROFILE_INTERVAL",
            "DARKBLOOM_QWEN4_DECODE_PROFILE_MAX_WIDTH",
        ]

        /// The given flags, with every other profiler flag unset.
        static func flags(_ values: [String: String]) -> [String: String?] {
            var all: [String: String?] = [:]
            for name in names {
                all.updateValue(values[name], forKey: name)
            }
            return all
        }

        static var token: MLXArray { MLXArray([Int32(5)], [1, 1]) }

        @Test func modesAndSamplingRules() {
            #expect(Support.isNil(Profile.fineMode(environment: [:])))
            #expect(Profile.fineMode(environment: [Self.fine: "1"]) == .synchronized)
            #expect(Profile.fineMode(environment: [Self.fine: "build"]) == .buildOnly)
            #expect(Support.isNil(Profile.fineMode(environment: [Self.fine: "yes"])))
            #expect(Profile.fineEnabled(environment: [Self.fine: "1"]))
            #expect(Profile.coarseEnabled(environment: [Self.coarse: "1"]))
            #expect(
                !Profile.coarseEnabled(environment: [Self.coarse: "1", Self.fine: "1"]),
                "the fine profile wins")
            #expect(!Profile.coarseEnabled(environment: [:]))

            #expect(Profile.profileInt("X", default: 8, minimum: 0, environment: [:]) == 8)
            #expect(Profile.profileInt("X", default: 8, minimum: 0, environment: ["X": "a"]) == 8)
            #expect(Profile.profileInt("X", default: 8, minimum: 1, environment: ["X": "-3"]) == 1)
            #expect(Profile.profileInt("X", default: 8, minimum: 0, environment: ["X": "12"]) == 12)

            // Warmup 2, interval 3: calls 3, 6, 9 are sampled.
            let sampled = (1 ... 10).filter {
                Profile.shouldSample(callIndex: $0, warmup: 2, interval: 3)
            }
            #expect(sampled == [3, 6, 9])
            #expect(Profile.shouldSample(callIndex: 1, warmup: 0, interval: 0))

            #expect(Profile.maxWidth(environment: [:]) == 1)
            #expect(
                Profile.maxWidth(environment: ["DARKBLOOM_QWEN4_DECODE_PROFILE_MAX_WIDTH": "6"])
                    == 6)
            #expect(
                Profile.maxWidth(environment: ["DARKBLOOM_QWEN4_DECODE_PROFILE_MAX_WIDTH": "0"])
                    == 1)

            let wide = MLXArray([Int32(1), 2, 3], [1, 3])
            #expect(
                Profile.isOrdinaryDecode(
                    inputs: Self.token, inputEmbeddings: nil, contextTokens: 4, maxWidth: 1))
            #expect(
                !Profile.isOrdinaryDecode(
                    inputs: Self.token, inputEmbeddings: nil, contextTokens: 0, maxWidth: 1))
            #expect(
                !Profile.isOrdinaryDecode(
                    inputs: Self.token, inputEmbeddings: MLXArray.zeros([1, 1, 4]),
                    contextTokens: 4, maxWidth: 1))
            #expect(
                !Profile.isOrdinaryDecode(
                    inputs: wide, inputEmbeddings: nil, contextTokens: 4, maxWidth: 1))
            #expect(
                Profile.isOrdinaryDecode(
                    inputs: wide, inputEmbeddings: nil, contextTokens: 4, maxWidth: 3))
            #expect(
                !Profile.isOrdinaryDecode(
                    inputs: MLXArray([Int32(1), 2], [2, 1]), inputEmbeddings: nil,
                    contextTokens: 4, maxWidth: 3))
            #expect(
                !Profile.isOrdinaryDecode(
                    inputs: MLXArray([Int32(1)], [1]), inputEmbeddings: nil, contextTokens: 4,
                    maxWidth: 3))

            let sample = Profile.Sample(callIndex: 1, contextTokens: 4, mode: .synchronized)
            sample.add("gdn", 1_500_000)
            sample.add("gdn", 500_000)
            #expect(sample.width == 1)
            #expect(sample.milliseconds("gdn") == 2)
            #expect(sample.milliseconds("moe") == 0)
        }

        /// Without flags the hooks only run their bodies.
        @Test func hooksAreNoOpsWithoutFlags() {
            Support.withEnvironment(Self.flags([:])) {
                #expect(
                    Support.isNil(
                        Profile.beginFine(
                            inputs: Self.token, inputEmbeddings: nil, contextTokens: 4)))
                #expect(
                    Support.isNil(
                        Profile.beginCoarse(
                            inputs: Self.token, inputEmbeddings: nil, contextTokens: 4)))
                #expect(Support.isNil(Profile.current))
                let x = MLXArray([Float(1), 2], [2])
                #expect(Support.isEqual(Profile.stage("gdn") { x + 1 }, x + 1))
                let kv = Profile.stageKV("qsa.kv_read") { [(keys: x, values: x * 2)] }
                #expect(kv.count == 1)
                let triple = Profile.stage3("attn_hc") { (x, x, x) }
                #expect(Support.isEqual(triple.2, x))

                // No dump directory: the dump hooks return at once.
                Profile.beginDump(inputs: Self.token, inputEmbeddings: nil, contextTokens: 4)
                Profile.dumpStage("attn", layer: 0, x.reshaped([1, 2]))
                Profile.dumpTensor("ple.history", layer: 0, x)
                Profile.endDump(logits: x.reshaped([1, 1, 2]))
                Profile.endDumpIfPending()
            }
        }

        /// `DARKBLOOM_QWEN4_DECODE_PROFILE=1`: each stage is evaluated and
        /// timed, and `endFine` writes the log line and clears the sample.
        @Test func synchronizedSampleRecordsEveryStage() throws {
            try Support.withEnvironment(
                Self.flags([
                    Self.fine: "1",
                    "DARKBLOOM_QWEN4_DECODE_PROFILE_WARMUP": "0",
                    "DARKBLOOM_QWEN4_DECODE_PROFILE_INTERVAL": "1",
                ])
            ) {
                let started = Profile.now()
                let sample = try #require(
                    Profile.beginFine(inputs: Self.token, inputEmbeddings: nil, contextTokens: 7))
                #expect(sample.mode == .synchronized)
                #expect(sample.contextTokens == 7)
                #expect(Profile.current === sample)

                let x = MLXRandom.normal([4, 8], key: MLXRandom.key(8100))
                let staged = Profile.stage("gdn") { x * 2 }
                let kv = Profile.stageKV("qsa.kv_read") { [(keys: x, values: x + 1)] }
                let triple = Profile.stage3("attn_hc") { (x, x * 3, x - 1) }
                let pooled = Profile.stage("qsa.pool") { x.sum(axis: 0) }
                #expect(Support.isEqual(staged, x * 2))
                #expect(Support.isEqual(kv[0].values, x + 1))
                #expect(Support.isEqual(triple.1, x * 3))
                #expect(pooled.shape == [8])
                for stage in ["gdn", "qsa.kv_read", "attn_hc", "qsa.pool"] {
                    #expect(sample.stageNs[stage] != nil, "stage \(stage)")
                }
                sample.modelNs = Profile.now() &- started
                Profile.endFine(sample, logits: x, totalStart: started)
                #expect(Support.isNil(Profile.current))

                // A width above the maximum is not sampled.
                #expect(
                    Support.isNil(
                        Profile.beginFine(
                            inputs: MLXArray([Int32(1), 2], [1, 2]), inputEmbeddings: nil,
                            contextTokens: 7)))
            }
        }

        /// `DARKBLOOM_QWEN4_DECODE_PROFILE=build` with a wider maximum width:
        /// the stages are timed without a forced evaluation.
        @Test func buildOnlySampleTimesConstruction() throws {
            try Support.withEnvironment(
                Self.flags([
                    Self.fine: "build",
                    "DARKBLOOM_QWEN4_DECODE_PROFILE_WARMUP": "0",
                    "DARKBLOOM_QWEN4_DECODE_PROFILE_INTERVAL": "1",
                    "DARKBLOOM_QWEN4_DECODE_PROFILE_MAX_WIDTH": "4",
                ])
            ) {
                let inputs = MLXArray([Int32(1), 2, 3], [1, 3])
                let started = Profile.now()
                let sample = try #require(
                    Profile.beginFine(inputs: inputs, inputEmbeddings: nil, contextTokens: 9))
                #expect(sample.mode == .buildOnly)
                #expect(sample.width == 3)
                let x = MLXRandom.normal([2, 4], key: MLXRandom.key(8110))
                _ = Profile.stage("moe") { x * 2 }
                _ = Profile.stageKV("qsa.kv_read") { [(keys: x, values: x)] }
                _ = Profile.stage3("mlp_hc") { (x, x, x) }
                #expect(sample.stageNs["moe"] != nil)
                #expect(sample.stageNs["mlp_hc"] != nil)
                Profile.endFine(sample, logits: x, totalStart: started)
                #expect(Support.isNil(Profile.current))
            }
        }

        /// The warmup skips the first calls of the fine profile.
        @Test func warmupSkipsTheFirstCalls() {
            Support.withEnvironment(
                Self.flags([
                    Self.fine: "1",
                    "DARKBLOOM_QWEN4_DECODE_PROFILE_WARMUP": "1000000",
                ])
            ) {
                #expect(
                    Support.isNil(
                        Profile.beginFine(
                            inputs: Self.token, inputEmbeddings: nil, contextTokens: 3)))
                #expect(Support.isNil(Profile.current))
            }
        }

        /// `DARKBLOOM_QWEN4_DECODE_COARSE_PROFILE=1`: one sample for each
        /// call after the warmup; `endCoarse` evaluates the logits and
        /// writes the log line.
        @Test func coarseSampleTimesBuildAndEval() throws {
            try Support.withEnvironment(
                Self.flags([
                    Self.coarse: "1",
                    "DARKBLOOM_QWEN4_DECODE_COARSE_PROFILE_WARMUP": "0",
                    "DARKBLOOM_QWEN4_DECODE_COARSE_PROFILE_INTERVAL": "1",
                ])
            ) {
                #expect(
                    Support.isNil(
                        Profile.beginFine(
                            inputs: Self.token, inputEmbeddings: nil, contextTokens: 3)))
                let started = Profile.now()
                let first = try #require(
                    Profile.beginCoarse(
                        inputs: Self.token, inputEmbeddings: nil, contextTokens: 3))
                let second = try #require(
                    Profile.beginCoarse(
                        inputs: Self.token, inputEmbeddings: nil, contextTokens: 4))
                #expect(second.callIndex == first.callIndex + 1)
                #expect(second.contextTokens == 4)
                let logits = MLXRandom.normal([1, 1, 8], key: MLXRandom.key(8120))
                Profile.endCoarse(first, logits: logits, buildStart: started)
                #expect(
                    Support.isNil(
                        Profile.beginCoarse(
                            inputs: Self.token, inputEmbeddings: logits, contextTokens: 3)))
            }
        }
    }
}
