import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXVLM

final class MiMoV26MLXVLMWeightsTests: XCTestCase {
    func testSelectedPublishedMetadataInventory() throws {
        guard let location = ProcessInfo.processInfo.environment["MIMO_V26_MLXVLM_METADATA"] else {
            throw XCTSkip(
                "Set the approved metadata/header receipt directory; no payload or credential access"
            )
        }
        let root = URL(fileURLWithPath: location)
        let data = try Data(contentsOf: root.appendingPathComponent("all-root-headers.json"))
        let receipt = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let headers = try XCTUnwrap(receipt["headers"] as? [[String: Any]])
        var descriptors: [String: MiMoV26ConvertedTensorDescriptor] = [:]
        for header in headers {
            let file = try XCTUnwrap(header["file"] as? String)
            for (key, raw) in try XCTUnwrap(header["tensors"] as? [String: Any])
            where key != "__metadata__" {
                let tensor = try XCTUnwrap(raw as? [String: Any])
                let shape = try XCTUnwrap(tensor["shape"] as? [Int])
                let dtype = try XCTUnwrap(
                    MiMoV26ConvertedScalarType(rawValue: XCTUnwrap(tensor["dtype"] as? String)))
                XCTAssertNil(
                    descriptors.updateValue(
                        .init(shape: shape, dtype: dtype, file: file), forKey: key))
            }
        }
        let provenanceData = try Data(
            contentsOf: root.appendingPathComponent("metadata/artifact-provenance.json"))
        let provenance = try XCTUnwrap(
            JSONSerialization.jsonObject(with: provenanceData) as? [String: Any])
        let plan = try MiMoV26ConvertedLoadPlan.make(
            configurationData: Data(
                contentsOf: root.appendingPathComponent("metadata/config.json")),
            indexData: Data(
                contentsOf: root.appendingPathComponent("metadata/model.safetensors.index.json")),
            descriptors: descriptors,
            provenance: .init(
                artifactID: "selected-published-header-check",
                sourceRepository: XCTUnwrap(provenance["original_model"] as? String),
                sourceRevision: XCTUnwrap(provenance["original_revision"] as? String),
                conversionManifestSHA256: convertedHash(provenanceData), layout: .mlxVLM))
        XCTAssertEqual(plan.descriptors.count, 1732)
        XCTAssertEqual(plan.rootFiles.count, 37)
        XCTAssertEqual(plan.tensorBytes, 170_974_427_392)
        XCTAssertEqual(plan.components[.target]?.count, 1103)
        XCTAssertEqual(plan.components[.vision]?.count, 364)
        XCTAssertEqual(plan.components[.audioPatch]?.count, 223)
        XCTAssertEqual(plan.components[.mtp]?.count, 42)
        XCTAssertGreaterThan(plan.tensorBytes(for: .mtp), 0)
        XCTAssertEqual(
            MiMoV26ConvertedComponent.allCases.reduce(0) { $0 + plan.tensorBytes(for: $1) },
            plan.tensorBytes)
        XCTAssertEqual(plan.targetExpertModulePaths.count, 141)
        XCTAssertEqual(
            plan.targetQuantizationPolicies.count + plan.audioQuantizationPolicies.count, 402)
        XCTAssertTrue(plan.floatingMTP)
        XCTAssertEqual(plan.configuration.eosTokenIDs, Set([151643, 151645, 151672]))
        XCTAssertNil(plan.provenance.payloadVerificationReceiptSHA256)
    }

    private struct Fixture {
        var config: Data
        var index: Data
        var descriptors: [String: MiMoV26ConvertedTensorDescriptor]
        let tensors: [String: MLXArray]
        let provenance = MiMoV26ConvertedProvenance(
            artifactID: "synthetic-mlx-vlm-fixture",
            sourceRepository: "XiaomiMiMo/MiMo-V2.6-Flash-MOPD",
            sourceRevision: String(repeating: "1", count: 40),
            conversionManifestSHA256: String(repeating: "2", count: 64), layout: .mlxVLM)
        func plan() throws -> MiMoV26ConvertedLoadPlan {
            try .make(
                configurationData: config, indexData: index, descriptors: descriptors,
                provenance: provenance)
        }
    }

    private func fixtureValues(key: String, count: Int) -> [Float] {
        // Full-key FNV seed + xorshift stream: repeated shapes do not produce
        // identical gate/up/layer/head bytes and hide an incorrect name mapping.
        var state = key.utf8.reduce(UInt64(14_695_981_039_346_656_037)) {
            ($0 ^ UInt64($1)) &* 1_099_511_628_211
        }
        if state == 0 { state = 1 }
        return (0 ..< count).map { _ in
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            return Float(Int(state & 0xffff) - 32768) / 262144
        }
    }

    private func fixture() throws -> Fixture {
        let env = ProcessInfo.processInfo.environment
        guard env["MIMO_V26_MLXVLM_NATIVE_TESTS"] == "1",
            let root = env["MIMO_V26_SERIAL_LOAD_FIXTURES"]
        else {
            throw XCTSkip("Requires exclusive native lane and the existing synthetic fixture root")
        }
        let original = try Data(
            contentsOf: URL(fileURLWithPath: root).appendingPathComponent("tiny-bf16/config.json"))
        var raw = try XCTUnwrap(JSONSerialization.jsonObject(with: original) as? [String: Any])
        raw.removeValue(forKey: "quantization")
        raw.removeValue(forKey: "quantization_config")
        raw.removeValue(forKey: "omlx_mimo_mtp")
        var audio = try XCTUnwrap(raw["audio_config"] as? [String: Any])
        audio["input_local_dim"] = 64
        audio["input_local_head_dim"] = 32
        audio["input_local_intermediate_size"] = 128
        raw["audio_config"] = audio
        let config = try JSONDecoder().decode(
            MiMoV26Configuration.self, from: JSONSerialization.data(withJSONObject: raw))
        // Build the independent float modules, then quantize SMALL deterministic
        // arrays with MLX's stock quantizer. Never convert the published weights.
        let target = try MiMoV26TextModel(config)
        let patch = try MiMoV26AudioPatchEncoder(configuration: XCTUnwrap(config.audio))
        let mtp = try MiMoV26MTP(target: target)
        var q: [String: Any] = ["bits": 4, "group_size": 64, "mode": "affine"]
        var tensors: [String: MLXArray] = [:]
        var descriptors: [String: MiMoV26ConvertedTensorDescriptor] = [:]
        var index: [String: String] = [:]
        var bytes = 0
        func record(_ name: String, _ array: MLXArray, _ component: String) {
            let file = component + ".safetensors"
            let dtype: MiMoV26ConvertedScalarType =
                array.dtype == .uint32
                ? .uint32
                : (array.dtype == .uint8
                    ? .uint8 : (array.dtype == .float32 ? .float32 : .bfloat16))
            tensors[name] = array
            descriptors[name] = .init(shape: array.shape, dtype: dtype, file: file)
            index[name] = file
            bytes += array.size * dtype.bytes
        }
        for (component, module) in [("target", target as Module), ("audio", patch), ("mtp", mtp)] {
            let leaves = Dictionary(uniqueKeysWithValues: module.leafModules().flattened())
            for (path, param) in module.parameters().flattened() {
                let name: String
                switch component {
                case "target": name = "language_model." + path
                case "mtp": name = "language_model.model.mtp." + path
                default: name = path
                }
                let float = MLXArray(fixtureValues(key: name, count: param.size)).reshaped(
                    param.shape
                )
                .asType(
                    path.hasSuffix("e_score_correction_bias") || path.hasSuffix("mlp.gate.weight")
                        ? .float32 : .bfloat16)
                let modulePath = String(path.dropLast(".weight".count))
                let leaf = path.hasSuffix(".weight") ? leaves[modulePath] : nil
                let packed =
                    ["target", "audio"].contains(component)
                    && (leaf is Linear || leaf is Embedding || leaf is SwitchLinear)
                if packed {
                    let expert = leaf is SwitchLinear
                    let group = expert ? 32 : 64
                    let bits = expert ? 4 : 8
                    let prefix = String(name.dropLast(".weight".count))
                    q[prefix] = [
                        "mode": expert ? "mxfp4" : "affine", "bits": bits, "group_size": group,
                    ]
                    let (weight, scales, biases) = MLX.quantized(
                        float, groupSize: group, bits: bits, mode: expert ? .mxfp4 : .affine)
                    record(name, weight, component)
                    record(prefix + ".scales", scales, component)
                    if let biases { record(prefix + ".biases", biases, component) }
                } else {
                    record(name, float, component)
                }
            }
        }
        for (key, shape) in try MiMoV26VisionTower.expectedTensorShapes(
            configuration: XCTUnwrap(config.vision))
        {
            let stored =
                key == "visual.patch_embed.proj.weight"
                ? [shape[0], shape.dropFirst().reduce(1, *)] : shape
            let name = "vision_tower." + key.dropFirst("visual.".count)
            let array = MLXArray(fixtureValues(key: name, count: shape.reduce(1, *))).reshaped(
                stored
            ).asType(.bfloat16)
            record(name, array, "vision")
        }
        eval(Array(tensors.values))
        for (a, b) in [
            (
                "language_model.model.layers.0.mlp.gate_proj.weight",
                "language_model.model.layers.0.mlp.up_proj.weight"
            ),
            (
                "language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight",
                "language_model.model.layers.1.mlp.switch_mlp.up_proj.weight"
            ),
            (
                "language_model.model.mtp.layers.0.mlp.gate_proj.weight",
                "language_model.model.mtp.layers.0.mlp.up_proj.weight"
            ),
            (
                "language_model.model.mtp.layers.1.enorm.weight",
                "language_model.model.mtp.layers.2.enorm.weight"
            ),
            (
                "audio_encoder.input_local_transformer.layers.0.self_attn.q_proj.weight",
                "audio_encoder.input_local_transformer.layers.0.self_attn.k_proj.weight"
            ),
            ("vision_tower.blocks.0.norm1.weight", "vision_tower.blocks.0.norm2.weight"),
        ] {
            let first = try XCTUnwrap(tensors[a])
            let second = try XCTUnwrap(tensors[b])
            XCTAssertEqual(first.shape, second.shape)
            XCTAssertNotEqual(
                first.asData(access: .copy).data, second.asData(access: .copy).data, a + " / " + b)
        }
        raw["quantization"] = q
        raw["quantization_config"] = q
        return .init(
            config: try JSONSerialization.data(withJSONObject: raw, options: [.sortedKeys]),
            index: try JSONSerialization.data(
                withJSONObject: ["metadata": ["total_size": bytes], "weight_map": index],
                options: [.sortedKeys]),
            descriptors: descriptors, tensors: tensors)
    }

    func testPackedTargetAudioAndFloatingHeadsLoadAndExecute() throws {
        let f = try fixture()
        let plan = try f.plan()
        XCTAssertEqual(plan.components[.mtp]?.count, 42)
        XCTAssertEqual(plan.externalRequiredComponents, ["audio_tokenizer"])
        XCTAssertEqual(Set(plan.parameterNames.values).count, plan.descriptors.count)
        // Freeze independent copies BEFORE loading: a mutation of a borrowed
        // source handle must not change both sides of the equality oracle.
        let sourceBytes = f.tensors.mapValues { $0.asData(access: .copy).data }
        let loaded = try MiMoV26ConvertedWeights.load(plan: plan, tensors: f.tensors)
        XCTAssertTrue(loaded.target.hasLoadedEmbeddingPrecision)
        XCTAssertTrue(loaded.target.hasLoadedReadoutPrecision)
        XCTAssertTrue(loaded.mtp.isLoaded)
        XCTAssertEqual(loaded.mtp.headCount, 3)
        XCTAssertFalse(loaded.mtp.leafModules().flattened().contains { $0.1 is Quantized })
        let output = try loaded.target.forward(inputIDs: MLXArray([Int32(1), 2, 3]).reshaped(1, 3))
        let audio = try loaded.audioPatch.forward(
            clips: [.init(codes: [Int32](repeating: 1, count: 20), frameCount: 1)],
            limits: .init(
                maximumClips: 1, maximumFrames: 1, maximumPatches: 1,
                maximumWorkingElements: 100_000))
        let image = try loaded.vision.forward(
            patches: MLXArray.zeros([4, 24]),
            grids: [.init(temporal: 1, height: 2, width: 2)],
            limits: .init(maximumPatches: 4, maximumAttentionScoreElements: 1000))
        eval(output.logits, audio.features, image)
        XCTAssertEqual(output.logits.shape, [1, 3, 128])
        XCTAssertEqual(audio.features.shape, [1, 64])
        XCTAssertTrue(output.logits.asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertTrue(audio.features.asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertEqual(image.shape, [1, 64])
        XCTAssertTrue(image.asArray(Float.self).allSatisfy(\.isFinite))
        _ = try MiMoV26CBv2Adapter(target: loaded.target)
        let features = try MiMoV26MTPFeatures(targetOutput: output, target: loaded.target)
        let cache = try loaded.mtp.newCache()
        for depth in 0 ..< 3 {
            let result = try loaded.mtp.forward(
                depth: depth, features: features,
                inputIDs: MLXArray([Int32(2), 3, 4]).reshaped(1, 3), target: loaded.target,
                cache: cache)
            eval(result.logits)
            XCTAssertEqual(result.logits.shape, [1, 3, 128])
            XCTAssertTrue(result.logits.asArray(Float.self).allSatisfy(\.isFinite))
        }
        for (component, module) in [
            (MiMoV26ConvertedComponent.target, loaded.target as Module),
            (.vision, loaded.vision), (.audioPatch, loaded.audioPatch), (.mtp, loaded.mtp),
        ] {
            let parameters = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
            for source in plan.components[component]! {
                let canonical = plan.parameterNames[source]!
                let path =
                    component == .vision
                    ? String(canonical.dropFirst("visual.".count))
                    : (component == .mtp ? String(canonical.dropFirst("mtp.".count)) : canonical)
                let actual = try XCTUnwrap(parameters[path])
                let expected = f.tensors[source]!
                XCTAssertEqual(actual.shape, expected.shape, source)
                XCTAssertEqual(actual.dtype, expected.dtype, source)
                XCTAssertEqual(actual.asData(access: .copy).data, sourceBytes[source], source)
            }
        }
    }

    func testMissingHeadWrongStorageAndConflictingPolicyRefuse() throws {
        let f = try fixture()
        for change in [
            "missingHead", "wrongDtype", "duplicateNamespace", "implicitDense", "quantizedHead",
        ] {
            var bad = f
            let key = "language_model.model.mtp.layers.2.eh_proj.weight"
            if change == "missingHead" {
                bad.descriptors.removeValue(forKey: key)
            } else if change == "wrongDtype" {
                let d = try XCTUnwrap(bad.descriptors[key])
                bad.descriptors[key] = .init(shape: d.shape, dtype: .float16, file: d.file)
            } else if change == "duplicateNamespace" {
                bad.descriptors["mtp.layers.2.eh_proj.weight"] = bad.descriptors[key]
            } else {
                var raw = try XCTUnwrap(
                    JSONSerialization.jsonObject(with: bad.config) as? [String: Any])
                var q = try XCTUnwrap(raw["quantization"] as? [String: Any])
                if change == "implicitDense" {
                    q.removeValue(forKey: "language_model.model.embed_tokens")
                } else {
                    q["language_model.model.mtp.layers.2.eh_proj"] = [
                        "mode": "affine", "bits": 8, "group_size": 64,
                    ]
                }
                raw["quantization"] = q
                raw["quantization_config"] = q
                bad.config = try JSONSerialization.data(withJSONObject: raw)
            }
            XCTAssertThrowsError(try bad.plan(), change)
        }
    }
}
