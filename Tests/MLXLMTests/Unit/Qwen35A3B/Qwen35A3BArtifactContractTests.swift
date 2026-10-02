import Foundation
import Testing

@testable import MLXLLM

extension UnitTests {

    /// Tests of the configuration inspection and the construction
    /// installation in `Qwen35A3BOptimization.swift`. The tests write a
    /// `config.json` to a temporary file and read it with
    /// `Qwen35A3BArtifactContract.inspect(configurationURL:)`. They use no
    /// MLX arrays.
    @Suite
    struct Qwen35A3BArtifactContractTests {

        /// The configuration of the EigenLabs Qwen3.6 35B-A3B artifact.
        static func root() -> [String: Any] {
            var quantization: [String: Any] = ["mode": "affine", "bits": 4, "group_size": 64]
            for layer in 0 ..< 40 {
                quantization["language_model.model.layers.\(layer).mlp.gate"] = ["bits": 8]
                quantization["language_model.model.layers.\(layer).mlp.shared_expert_gate"] = [
                    "bits": 8
                ]
            }
            let layerTypes = (0 ..< 40).map { $0 % 4 == 3 ? "full_attention" : "linear_attention" }
            return [
                "model_type": "qwen3_5_moe",
                "text_config": [
                    "model_type": "qwen3_5_moe_text", "hidden_size": 2_048,
                    "num_hidden_layers": 40, "num_experts": 256,
                    "num_experts_per_tok": 8, "moe_intermediate_size": 512,
                    "shared_expert_intermediate_size": 512, "mtp_num_hidden_layers": 1,
                    "layer_types": layerTypes,
                ] as [String: Any],
                "quantization": quantization,
                "mtplx_mtp": ["included": true],
                "mtplx_mtp_quantization": ["mode": "mxfp8", "bits": 8, "group_size": 32],
            ]
        }

        static func inspect(_ root: [String: Any]) throws -> Qwen35A3BArtifactContract {
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("qwen35a3b-\(UUID().uuidString).json")
            try JSONSerialization.data(withJSONObject: root).write(to: url)
            defer { try? FileManager.default.removeItem(at: url) }
            return try Qwen35A3BArtifactContract.inspect(configurationURL: url)
        }

        /// The error of `inspect` for `root`, or nil when it does not throw.
        static func error(_ root: [String: Any]) -> Qwen35A3BArtifactError? {
            do {
                _ = try inspect(root)
                return nil
            } catch {
                return error as? Qwen35A3BArtifactError
            }
        }

        static func editing(
            _ table: String, _ edit: (inout [String: Any]) -> Void
        ) -> [String: Any] {
            var copy = Self.root()
            var value = copy[table] as? [String: Any] ?? [:]
            edit(&value)
            copy[table] = value
            return copy
        }

        @Test func validConfigurationGivesTheContract() throws {
            let contract = try Self.inspect(Self.root())
            #expect(contract.geometry.layers == 40)
            #expect(contract.geometry.recurrentLayers == 30)
            #expect(contract.geometry.fullAttentionLayers == 10)
            #expect(
                contract.summary
                    == "H2048/E256/K8/I512/L40; target=affine-w4-g64; mtp=mxfp8-g32")
        }

        @Test func missingTablesAndFieldsAreMalformed() {
            var noText = Self.root()
            noText.removeValue(forKey: "text_config")
            #expect(Self.error(noText) == .malformed(field: "config.json"))

            var noMTP = Self.root()
            noMTP.removeValue(forKey: "mtplx_mtp_quantization")
            #expect(Self.error(noMTP) == .malformed(field: "config.json"))

            let noLayers = Self.editing("text_config") {
                $0.removeValue(forKey: "num_hidden_layers")
            }
            #expect(Self.error(noLayers) == .malformed(field: "num_hidden_layers"))

            let noMode = Self.editing("quantization") { $0.removeValue(forKey: "mode") }
            #expect(Self.error(noMode) == .malformed(field: "mode"))

            let textHidden = Self.editing("text_config") { $0["hidden_size"] = "2048" }
            #expect(Self.error(textHidden) == .malformed(field: "hidden_size"))

            let noMTPMode = Self.editing("mtplx_mtp_quantization") {
                $0.removeValue(forKey: "mode")
            }
            #expect(Self.error(noMTPMode) == .malformed(field: "mode"))

            var noRootType = Self.root()
            noRootType.removeValue(forKey: "model_type")
            #expect(Self.error(noRootType) == .malformed(field: "model_type"))
        }

        @Test func projectionOverridesMustKeepTheTargetPacking() {
            let notATable = Self.editing("quantization") { $0["lm_head"] = true }
            #expect(Self.error(notATable) == .malformed(field: "quantization.lm_head"))

            let wrongMode = Self.editing("quantization") { $0["lm_head"] = ["mode": "mxfp4"] }
            #expect(
                Self.error(wrongMode)
                    == .mismatch(
                        field: "quantization.lm_head.mode", expected: "affine", actual: "mxfp4"))

            let path = "language_model.model.layers.3.self_attn.o_proj"
            let wrongGroup = Self.editing("quantization") { $0[path] = ["group_size": 32] }
            #expect(
                Self.error(wrongGroup)
                    == .mismatch(
                        field: "quantization.\(path).group_size", expected: "64", actual: "32"))

            let sameAsTarget = Self.editing("quantization") { $0[path] = ["bits": 4] }
            #expect(Self.error(sameAsTarget) == nil)
        }

        @Test func everyLayerNeedsAnEightBitRouterAndSharedGate() {
            let gate = "language_model.model.layers.0.mlp.gate"
            let shared = "language_model.model.layers.7.mlp.shared_expert_gate"

            let noGate = Self.editing("quantization") { $0.removeValue(forKey: gate) }
            #expect(Self.error(noGate) == .malformed(field: "quantization.\(gate)"))

            let targetBitsGate = Self.editing("quantization") { $0[gate] = [String: Any]() }
            #expect(
                Self.error(targetBitsGate)
                    == .mismatch(field: "quantization.\(gate).bits", expected: "8", actual: "4"))

            let sharedGroup = Self.editing("quantization") {
                $0[shared] = ["bits": 8, "group_size": 128]
            }
            #expect(
                Self.error(sharedGroup)
                    == .mismatch(
                        field: "quantization.\(shared).group_size", expected: "64",
                        actual: "128"))
        }

        @Test func contractFieldsAreCheckedAfterParsing() {
            let notIncluded = Self.editing("mtplx_mtp") { $0["included"] = false }
            #expect(
                Self.error(notIncluded)
                    == .mismatch(field: "mtplx_mtp.included", expected: "true", actual: "false"))

            let fewerExperts = Self.editing("text_config") { $0["num_experts"] = 128 }
            #expect(
                Self.error(fewerExperts)
                    == .mismatch(field: "text_config.num_experts", expected: "256", actual: "128"))

            let mtpBits = Self.editing("mtplx_mtp_quantization") { $0["bits"] = 4 }
            #expect(
                Self.error(mtpBits) == .mismatch(field: "mtp.bits", expected: "8", actual: "4"))
        }

        @Test func errorDescriptionsNameTheField() {
            let mismatch = Qwen35A3BArtifactError.mismatch(
                field: "router.bits", expected: "8", actual: "4")
            #expect(
                mismatch.description
                    == "Qwen35 A3B artifact mismatch at router.bits: expected 8, got 4")
            #expect(
                Qwen35A3BArtifactError.malformed(field: "mode").description
                    == "Qwen35 A3B artifact is missing or malformed at mode")
        }

        /// Exact M=1 verify arithmetic needs a decode-capable profile.
        @Test func installationChecksTheVerifyArithmetic() throws {
            let contract = try Qwen35A3BArtifactContract.inspect(fixture: .eigenLabsRouter8)
            #expect(throws: Qwen35A3BArtifactError.self) {
                try Qwen35A3BConstructionInstallation.install(
                    contract: contract, profile: .prefill, targetVerifyArithmetic: .exactM1)
            }
            #expect(throws: Qwen35A3BArtifactError.self) {
                try Qwen35A3BConstructionInstallation.install(
                    contract: contract, profile: .stock, targetVerifyArithmetic: .exactM1)
            }
            let installation = try Qwen35A3BConstructionInstallation.install(
                contract: contract, profile: .decode, targetVerifyArithmetic: .exactM1)
            #expect(installation.contract == contract)

            #expect(Qwen35A3BConstructionContext.profile == .stock)
            #expect(Qwen35A3BConstructionContext.targetVerifyArithmetic == .rectangular)
            let inside = Qwen35A3BConstructionContext.withInstallation(installation) {
                (
                    Qwen35A3BConstructionContext.profile,
                    Qwen35A3BConstructionContext.targetVerifyArithmetic
                )
            }
            #expect(inside.0 == .decode)
            #expect(inside.1 == .exactM1)
            #expect(Qwen35A3BConstructionContext.profile == .stock)

            let decode = try Qwen35A3BRouteTable.install(contract: contract, profile: .decode)
            let full = try Qwen35A3BRouteTable.install(contract: contract, profile: .full)
            #expect(decode.prefill == .stock)
            #expect(decode.targetDecode == .rightShaped)
            #expect(decode.mtpDecode == .rightShaped)
            #expect(full.prefill == .rightShaped)
            #expect(full.targetDecode == .rightShaped)
            #expect(full.mtpDecode == .rightShaped)
            #expect(Qwen35A3BRouteLane.rightShaped.rawValue == "right-shaped")
        }

        @Test func asyncInstallationScopesTheProfile() async throws {
            let contract = try Qwen35A3BArtifactContract.inspect(fixture: .eigenLabsRouter8)
            let installation = try Qwen35A3BConstructionInstallation.install(
                contract: contract, profile: .full)
            let profile = await Qwen35A3BConstructionContext.withInstallation(installation) {
                () async -> Qwen35A3BOptimizationProfile in
                await Task.yield()
                return Qwen35A3BConstructionContext.profile
            }
            #expect(profile == .full)
            #expect(Qwen35A3BConstructionContext.profile == .stock)
        }
    }
}
