import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Metadata plans and in-memory loads of the tiny synthetic checkpoint, in the
/// native converted layout and in the published MLX-VLM layout. The expected
/// counts and byte totals are computed by hand from the tiny geometry in
/// MiMoV26TinyCheckpoint.
final class MiMoV26TinyConvertedPlanTests: XCTestCase {
    private typealias Fixture = MiMoV26TinyCheckpoint

    // MARK: - Native layout

    func testNativePlanClosesEveryComponentWithHandCountedBytes() throws {
        let inputs = try Fixture.nativeInputs()
        let plan = try inputs.plan()
        XCTAssertEqual(plan.descriptors.count, 207)
        XCTAssertEqual(plan.components[.target]?.count, 27)
        XCTAssertEqual(plan.components[.vision]?.count, 55)
        XCTAssertEqual(plan.components[.audioPatch]?.count, 35)
        XCTAssertEqual(plan.components[.mtp]?.count, 90)
        XCTAssertEqual(
            plan.targetExpertModulePaths,
            [
                "model.layers.1.mlp.switch_mlp.gate_proj", "model.layers.1.mlp.switch_mlp.up_proj",
                "model.layers.1.mlp.switch_mlp.down_proj",
            ])
        XCTAssertEqual(plan.rootFiles, Set(Fixture.nativeFiles))
        // Hand totals: target 139160, vision 13468, audio 755968, MTP 67608.
        XCTAssertEqual(plan.tensorBytes(for: .target), 139_160)
        XCTAssertEqual(plan.tensorBytes(for: .vision), 13_468)
        XCTAssertEqual(plan.tensorBytes(for: .audioPatch), 755_968)
        XCTAssertEqual(plan.tensorBytes(for: .mtp), 67_608)
        XCTAssertEqual(plan.tensorBytes, 976_204)
        XCTAssertEqual(plan.externalRequiredComponents, ["audio_tokenizer", "dflash"])
        XCTAssertEqual(plan.parameterNames.count, 207)
        XCTAssertTrue(plan.parameterNames.allSatisfy { $0.key == $0.value })
        XCTAssertFalse(plan.floatingMTP)
        XCTAssertEqual(plan.floatingType, .bfloat16)
        XCTAssertTrue(plan.audioQuantizationPolicies.isEmpty)
        XCTAssertEqual(Set(plan.targetQuantizationPolicies.keys), plan.targetExpertModulePaths)
        XCTAssertTrue(
            plan.targetQuantizationPolicies.values.allSatisfy {
                $0.mode == "mxfp4" && $0.bits == 4 && $0.groupSize == 32
            })
        XCTAssertEqual(
            plan.descriptors["model.layers.1.mlp.switch_mlp.down_proj.weight"]?.shape, [4, 64, 4])
        XCTAssertEqual(
            plan.descriptors["model.layers.1.mlp.switch_mlp.down_proj.scales"]?.dtype, .uint8)
        XCTAssertEqual(plan.descriptors["mtp.layers.2.eh_proj.weight"]?.shape, [64, 16])
        XCTAssertEqual(plan.configSHA256.count, 64)
        XCTAssertEqual(plan.descriptorSHA256, try inputs.plan().descriptorSHA256)
        XCTAssertNil(plan.provenance.payloadVerificationReceiptSHA256)
    }

    func testNativePlanRejectsMissingExtraShapeTypeFileAndTotal() throws {
        let inputs = try Fixture.nativeInputs()
        let key = "model.layers.1.mlp.switch_mlp.gate_proj.weight"
        let original = try XCTUnwrap(inputs.descriptors[key])
        var missing = inputs
        missing.descriptors.removeValue(forKey: "visual.blocks.1.attn.sinks")
        XCTAssertThrowsError(try missing.plan())
        var extra = inputs
        extra.descriptors["audio_tokenizer.encoder.weight"] = original
        XCTAssertThrowsError(try extra.plan())
        var wrongShape = inputs
        wrongShape.descriptors[key] = .init(shape: [4, 32, 9], dtype: .uint32, file: original.file)
        XCTAssertThrowsError(try wrongShape.plan()) {
            XCTAssertEqual($0 as? MiMoV26ConvertedLoadError, .invalidInventory(key))
        }
        var wrongType = inputs
        wrongType.descriptors[key] = .init(
            shape: original.shape, dtype: .uint8, file: original.file)
        XCTAssertThrowsError(try wrongType.plan()) {
            XCTAssertEqual($0 as? MiMoV26ConvertedLoadError, .invalidInventory(key))
        }
        var fused = inputs
        fused.descriptors.removeValue(forKey: "model.layers.0.self_attn.q_proj.weight")
        fused.descriptors["model.layers.0.self_attn.qkv_proj.weight"] = original
        XCTAssertThrowsError(try fused.plan())
        var badFile = inputs
        badFile.descriptors[key] = .init(
            shape: original.shape, dtype: original.dtype, file: "../target.safetensors")
        var badIndex = try Fixture.object(badFile.index)
        var badMap = try XCTUnwrap(badIndex["weight_map"] as? [String: String])
        badMap[key] = "../target.safetensors"
        badIndex["weight_map"] = badMap
        badFile.index = try Fixture.data(badIndex)
        XCTAssertThrowsError(try badFile.plan()) {
            XCTAssertEqual($0 as? MiMoV26ConvertedLoadError, .invalidInventory(key))
        }
        // An MTP tensor must live in the declared embedded MTP file.
        var movedHead = inputs
        let headKey = "mtp.layers.0.enorm.weight"
        let head = try XCTUnwrap(inputs.descriptors[headKey])
        movedHead.descriptors[headKey] = .init(
            shape: head.shape, dtype: head.dtype, file: "target.safetensors")
        var movedIndex = try Fixture.object(movedHead.index)
        var movedMap = try XCTUnwrap(movedIndex["weight_map"] as? [String: String])
        movedMap[headKey] = "target.safetensors"
        movedIndex["weight_map"] = movedMap
        movedHead.index = try Fixture.data(movedIndex)
        XCTAssertThrowsError(try movedHead.plan()) {
            XCTAssertEqual($0 as? MiMoV26ConvertedLoadError, .invalidInventory(headKey))
        }
        var wrongTotal = inputs
        var index = try Fixture.object(wrongTotal.index)
        index["metadata"] = ["total_size": 1]
        wrongTotal.index = try Fixture.data(index)
        XCTAssertThrowsError(try wrongTotal.plan()) {
            XCTAssertEqual(
                $0 as? MiMoV26ConvertedLoadError, .invalidInventory("index tensor byte total"))
        }
        var indexExtra = inputs
        var extraIndex = try Fixture.object(indexExtra.index)
        var map = try XCTUnwrap(extraIndex["weight_map"] as? [String: String])
        map["unexpected.weight"] = "target.safetensors"
        extraIndex["weight_map"] = map
        indexExtra.index = try Fixture.data(extraIndex)
        XCTAssertThrowsError(try indexExtra.plan()) {
            XCTAssertEqual(
                $0 as? MiMoV26ConvertedLoadError,
                .invalidInventory("missing/unmapped root tensors"))
        }
    }

    func testNativePlanRejectsUnmappedConflictingAndMissingPolicies() throws {
        let inputs = try Fixture.nativeInputs()
        let key = "mtp.layers.0.self_attn.q_proj"
        let alias = "language_model." + key
        for change in ["missing", "conflict", "dense", "typo", "unknownMetadata", "headCount"] {
            var test = inputs
            var c = try Fixture.object(test.config)
            var q = try XCTUnwrap(c["quantization"] as? [String: Any])
            switch change {
            case "missing": q.removeValue(forKey: key)
            case "conflict": q[alias] = ["mode": "mxfp4", "bits": 4, "group_size": 32]
            case "dense":
                q["model.layers.0.self_attn.q_proj"] = [
                    "mode": "mxfp4", "bits": 4, "group_size": 32,
                ]
            case "typo":
                q["language_model.mtp.layers.7.self_attn.q_proj"] = [
                    "mode": "affine", "bits": 4, "group_size": 64,
                ]
            case "unknownMetadata": q["unparsed"] = "ignore me"
            default: c["num_nextn_predict_layers"] = 2
            }
            c["quantization"] = q
            test.config = try Fixture.data(c)
            XCTAssertThrowsError(try test.plan(), change)
        }
        // The wrapped alias alone names the same explicit projection policy.
        var single = inputs
        var c = try Fixture.object(single.config)
        var q = try XCTUnwrap(c["quantization"] as? [String: Any])
        let policy = q.removeValue(forKey: key)
        q[alias] = policy
        c["quantization"] = q
        single.config = try Fixture.data(c)
        XCTAssertEqual(try single.plan().descriptors.count, 207)
    }

    func testNativePlanRejectsUnsupportedGeometryAndMalformedProvenance() throws {
        for change in ["tie", "noMTPFile", "noVision", "noQuantization", "mode"] {
            var test = try Fixture.nativeInputs()
            var c = try Fixture.object(test.config)
            switch change {
            case "tie": c["tie_word_embeddings"] = true
            case "noMTPFile": c.removeValue(forKey: "omlx_mimo_mtp")
            case "noVision": c.removeValue(forKey: "vision_config")
            case "noQuantization": c.removeValue(forKey: "quantization")
            default:
                var q = try XCTUnwrap(c["quantization"] as? [String: Any])
                q["mode"] = "affine"
                q["group_size"] = 64
                c["quantization"] = q
            }
            test.config = try Fixture.data(c)
            XCTAssertThrowsError(try test.plan(), change) {
                XCTAssertEqual(
                    $0 as? MiMoV26ConvertedLoadError,
                    .invalidConfiguration(
                        "requires complete converted target/vision/audio/MTP geometry"),
                    change)
            }
        }
        var inputs = try Fixture.nativeInputs()
        inputs.provenance = .init(
            artifactID: "", sourceRepository: "wrong/model", sourceRevision: "main",
            conversionManifestSHA256: "unknown")
        XCTAssertThrowsError(try inputs.plan()) {
            XCTAssertEqual(
                $0 as? MiMoV26ConvertedLoadError,
                .invalidProvenance("artifact/source/receipt identity"))
        }
        inputs.provenance = .init(
            artifactID: "tiny", sourceRepository: Fixture.provenance.sourceRepository,
            sourceRevision: Fixture.provenance.sourceRevision,
            conversionManifestSHA256: Fixture.provenance.conversionManifestSHA256,
            payloadVerificationReceiptSHA256: "NOT-HEX")
        XCTAssertThrowsError(try inputs.plan())
        inputs.provenance = .init(
            artifactID: "tiny", sourceRepository: Fixture.provenance.sourceRepository,
            sourceRevision: Fixture.provenance.sourceRevision,
            conversionManifestSHA256: Fixture.provenance.conversionManifestSHA256,
            payloadVerificationReceiptSHA256: String(repeating: "e", count: 64))
        let plan = try inputs.plan()
        XCTAssertEqual(
            plan.provenance.payloadVerificationReceiptSHA256, String(repeating: "e", count: 64))
        // A supplied receipt reference never changes the inventory.
        XCTAssertEqual(plan.tensorBytes, 976_204)
    }

    func testNativeInMemoryLoadBuildsAllComponentsAndRunsTinyForwards() throws {
        let specs = try Fixture.nativeSpecs()
        let plan = try Fixture.nativeInputs().plan()
        let weights = Fixture.arrays(specs)
        let bundle = try MiMoV26ConvertedWeights.load(plan: plan, tensors: weights)
        XCTAssertEqual(bundle.plan.configSHA256, plan.configSHA256)
        XCTAssertTrue(bundle.mtp.isLoaded)
        XCTAssertEqual(bundle.mtp.headCount, 3)
        let leaves = bundle.target.leafModules().flattened()
        XCTAssertEqual(
            Set(leaves.compactMap { path, module in module is Quantized ? path : nil }),
            plan.targetExpertModulePaths)
        for (path, module) in leaves where plan.targetExpertModulePaths.contains(path) {
            let packed = try XCTUnwrap(module as? QuantizedSwitchLinear)
            XCTAssertEqual(packed.groupSize, 32)
            XCTAssertEqual(packed.bits, 4)
            XCTAssertEqual(packed.mode, .mxfp4)
        }
        let targetWeights = Dictionary(uniqueKeysWithValues: bundle.target.parameters().flattened())
        XCTAssertEqual(Set(targetWeights.keys), try XCTUnwrap(plan.components[.target]))
        for (key, value) in targetWeights {
            XCTAssertEqual(value.shape, weights[key]?.shape, key)
            XCTAssertEqual(value.dtype, weights[key]?.dtype, key)
        }
        let result = try bundle.target.forward(inputIDs: MLXArray([Int32(1), 2, 3]).reshaped(1, 3))
        let audio = try bundle.audioPatch.forward(
            clips: [.init(codes: [Int32](repeating: 1, count: 20), frameCount: 1)],
            limits: .init(
                maximumClips: 1, maximumFrames: 1, maximumPatches: 1,
                maximumWorkingElements: 1_000_000))
        let vision = try bundle.vision.forward(
            patches: MLXArray.zeros([4, 24]), grids: [.init(temporal: 1, height: 2, width: 2)],
            limits: .init(maximumPatches: 4, maximumAttentionScoreElements: 1000))
        eval(result.logits, audio.features, vision)
        XCTAssertEqual(result.logits.shape, [1, 3, 128])
        XCTAssertEqual(audio.features.shape, [1, 64])
        XCTAssertEqual(vision.shape, [1, 64])
        XCTAssertTrue(result.logits.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertTrue(audio.features.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertTrue(vision.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testNativeInMemoryLoadRejectsHandleClosureTypeAndLateConstruction() throws {
        let specs = try Fixture.nativeSpecs()
        let plan = try Fixture.nativeInputs().plan()
        let weights = Fixture.arrays(specs)
        var missing = weights
        missing.removeValue(forKey: "mtp.layers.2.eh_proj.biases")
        XCTAssertThrowsError(try MiMoV26ConvertedWeights.load(plan: plan, tensors: missing)) {
            XCTAssertEqual(
                $0 as? MiMoV26ConvertedLoadError,
                .invalidTensor("missing/unmapped root tensor handles"))
        }
        var badType = weights
        badType["model.norm.weight"] = weights["model.norm.weight"]!.asType(.int32)
        XCTAssertThrowsError(try MiMoV26ConvertedWeights.load(plan: plan, tensors: badType)) {
            XCTAssertEqual($0 as? MiMoV26ConvertedLoadError, .invalidTensor("model.norm.weight"))
        }
        var badShape = weights
        badShape["model.norm.weight"] = MLXArray.zeros([63]).asType(.bfloat16)
        XCTAssertThrowsError(try MiMoV26ConvertedWeights.load(plan: plan, tensors: badShape))
        var extra = weights
        extra["dflash.fc.weight"] = MLXArray.zeros([1])
        XCTAssertThrowsError(try MiMoV26ConvertedWeights.load(plan: plan, tensors: extra))
        // The plan accepts unknown metadata, but target construction refuses
        // an unsupported execution semantic after the inventory passed.
        var late = try Fixture.nativeInputs()
        var c = try Fixture.object(late.config)
        c["rope_scaling"] = ["type": "linear", "factor": 2]
        late.config = try Fixture.data(c)
        let latePlan = try late.plan()
        XCTAssertThrowsError(try MiMoV26ConvertedWeights.load(plan: latePlan, tensors: weights)) {
            XCTAssertEqual($0 as? MiMoV26ExecutionError, .unsupportedConfiguration("rope_scaling"))
        }
    }

    // MARK: - MLX-VLM layout

    func testMLXVLMPlanMapsSourceNamesAndExplicitPackedPolicies() throws {
        let (inputs, _, specs) = try Fixture.mlxVLMInputs()
        let plan = try inputs.plan()
        XCTAssertEqual(plan.descriptors.count, 243)
        XCTAssertEqual(plan.components[.target]?.count, 53)
        XCTAssertEqual(plan.components[.vision]?.count, 55)
        XCTAssertEqual(plan.components[.audioPatch]?.count, 93)
        XCTAssertEqual(plan.components[.mtp]?.count, 42)
        XCTAssertEqual(plan.tensorBytes, specs.reduce(0) { $0 + $1.byteCount })
        XCTAssertEqual(plan.externalRequiredComponents, ["audio_tokenizer"])
        XCTAssertTrue(plan.floatingMTP)
        XCTAssertEqual(plan.floatingType, .bfloat16)
        // One source name and one native name per trained parameter.
        XCTAssertEqual(Set(plan.parameterNames.values).count, 243)
        XCTAssertEqual(
            plan.parameterNames["vision_tower.patch_embed.proj.weight"],
            "visual.patch_embed.proj.weight")
        XCTAssertEqual(
            plan.parameterNames["language_model.model.mtp.layers.1.eh_proj.weight"],
            "mtp.layers.1.eh_proj.weight")
        XCTAssertEqual(
            plan.parameterNames["language_model.lm_head.scales"], "lm_head.scales")
        XCTAssertEqual(
            plan.parameterNames["speech_embeddings.3.biases"], "speech_embeddings.3.biases")
        XCTAssertEqual(plan.targetExpertModulePaths.count, 3)
        // Target: embed, lm_head, 8 attention, 3 dense, 3 expert matrices.
        XCTAssertEqual(plan.targetQuantizationPolicies.count, 16)
        // Audio: 20 speech embeddings, 4 attention, 3 MLP, 2 projection matrices.
        XCTAssertEqual(plan.audioQuantizationPolicies.count, 29)
        XCTAssertTrue(
            plan.audioQuantizationPolicies.values.allSatisfy {
                $0.mode == "affine" && $0.bits == 8 && $0.groupSize == 64
            })
        XCTAssertEqual(
            plan.descriptors["vision_tower.patch_embed.proj.weight"]?.shape, [8, 24])
    }

    func testMLXVLMPlanRejectsIdentityGeometryPolicyAndInventoryDrift() throws {
        let (inputs, fields, _) = try Fixture.mlxVLMInputs()
        var wrongSource = inputs
        wrongSource.provenance = .init(
            artifactID: "tiny", sourceRepository: "XiaomiMiMo/Other",
            sourceRevision: String(repeating: "c", count: 40),
            conversionManifestSHA256: String(repeating: "d", count: 64), layout: .mlxVLM)
        XCTAssertThrowsError(try wrongSource.plan()) {
            XCTAssertEqual(
                $0 as? MiMoV26ConvertedLoadError,
                .invalidProvenance("MLX-VLM artifact/source identity"))
        }
        func changed(_ mutate: (inout [String: Any]) throws -> Void) throws -> Fixture.Inputs {
            var copy = fields
            try mutate(&copy)
            var test = inputs
            test.config = try Fixture.data(copy)
            return test
        }
        let float32 = try changed { $0["dtype"] = "float32" }
        XCTAssertThrowsError(try float32.plan()) {
            XCTAssertEqual(
                $0 as? MiMoV26ConvertedLoadError,
                .invalidConfiguration(
                    "MLX-VLM packed target/audio and floating vision/MTP geometry"))
        }
        let fp8Source = try changed {
            $0["quantization_config"] = [
                "quant_method": "fp8", "store_dtype": "mxfp4", "fmt": "e4m3",
                "activation_scheme": "dynamic", "mxfp4_block_size": 32,
                "weight_block_size": [128, 128], "ignored_layers": [String](),
            ]
        }
        XCTAssertThrowsError(try fp8Source.plan()) {
            XCTAssertEqual(
                $0 as? MiMoV26ConvertedLoadError,
                .invalidConfiguration(
                    "MLX-VLM packed target/audio and floating vision/MTP geometry"))
        }
        let implicitDense = try changed { fields in
            for field in ["quantization", "quantization_config"] {
                var q = try XCTUnwrap(fields[field] as? [String: Any])
                q.removeValue(forKey: "language_model.model.embed_tokens")
                fields[field] = q
            }
        }
        XCTAssertThrowsError(try implicitDense.plan()) {
            XCTAssertEqual(
                $0 as? MiMoV26ConvertedLoadError,
                .invalidQuantization(
                    "missing/incompatible explicit MLX-VLM policy: language_model.model.embed_tokens"
                ))
        }
        let quantizedHead = try changed { fields in
            for field in ["quantization", "quantization_config"] {
                var q = try XCTUnwrap(fields[field] as? [String: Any])
                q["language_model.model.mtp.layers.2.eh_proj"] = [
                    "mode": "affine", "bits": 8, "group_size": 64,
                ]
                fields[field] = q
            }
        }
        XCTAssertThrowsError(try quantizedHead.plan()) {
            XCTAssertEqual(
                $0 as? MiMoV26ConvertedLoadError,
                .invalidQuantization(
                    "unmapped MLX-VLM policy; floating MTP/vision must not inherit root default"))
        }
        let unknownMetadata = try changed { fields in
            for field in ["quantization", "quantization_config"] {
                var q = try XCTUnwrap(fields[field] as? [String: Any])
                q["unparsed"] = "kept"
                fields[field] = q
            }
        }
        XCTAssertThrowsError(try unknownMetadata.plan()) {
            XCTAssertEqual(
                $0 as? MiMoV26ConvertedLoadError,
                .invalidQuantization("unmapped native metadata: quantization"))
        }
        let embeddedFile = try changed {
            $0["omlx_mimo_mtp"] = [
                "architecture": "mimo_v2_nextn", "num_layers": 3, "storage": "embedded",
                "file": "language_model.safetensors",
            ]
        }
        XCTAssertThrowsError(try embeddedFile.plan()) {
            XCTAssertEqual(
                $0 as? MiMoV26ConvertedLoadError,
                .invalidInventory("embedded MTP file declaration"))
        }
        let matchingFile = try changed {
            $0["omlx_mimo_mtp"] = [
                "architecture": "mimo_v2_nextn", "num_layers": 3, "storage": "embedded",
                "file": "mtp.safetensors",
            ]
        }
        XCTAssertEqual(try matchingFile.plan().descriptors.count, 243)
        let headKey = "language_model.model.mtp.layers.2.eh_proj.weight"
        var missingHead = inputs
        missingHead.descriptors.removeValue(forKey: headKey)
        XCTAssertThrowsError(try missingHead.plan()) {
            XCTAssertEqual(
                $0 as? MiMoV26ConvertedLoadError,
                .invalidInventory("missing/unmapped MLX-VLM root tensors"))
        }
        var wrongType = inputs
        let head = try XCTUnwrap(inputs.descriptors[headKey])
        wrongType.descriptors[headKey] = .init(shape: head.shape, dtype: .float16, file: head.file)
        XCTAssertThrowsError(try wrongType.plan()) {
            XCTAssertEqual($0 as? MiMoV26ConvertedLoadError, .invalidInventory(headKey))
        }
        var wrongTotal = inputs
        var index = try Fixture.object(wrongTotal.index)
        index["metadata"] = ["total_size": 7]
        wrongTotal.index = try Fixture.data(index)
        XCTAssertThrowsError(try wrongTotal.plan()) {
            XCTAssertEqual(
                $0 as? MiMoV26ConvertedLoadError, .invalidInventory("index tensor byte total"))
        }
    }

    func testMLXVLMInMemoryLoadInstallsPackedTargetAudioAndFloatingHeads() throws {
        let (inputs, _, specs) = try Fixture.mlxVLMInputs()
        let plan = try inputs.plan()
        let weights = Fixture.arrays(specs)
        let bundle = try MiMoV26ConvertedWeights.load(plan: plan, tensors: weights)
        XCTAssertTrue(bundle.target.hasLoadedEmbeddingPrecision)
        XCTAssertTrue(bundle.target.hasLoadedReadoutPrecision)
        XCTAssertTrue(bundle.mtp.isLoaded)
        XCTAssertEqual(bundle.mtp.headCount, 3)
        XCTAssertFalse(bundle.mtp.leafModules().flattened().contains { $0.1 is Quantized })
        XCTAssertEqual(
            bundle.audioPatch.leafModules().flattened().filter { $0.1 is Quantized }.count, 29)
        let parameters = Dictionary(
            uniqueKeysWithValues: bundle.audioPatch.parameters().flattened())
        for source in try XCTUnwrap(plan.components[.audioPatch]) {
            let native = try XCTUnwrap(plan.parameterNames[source])
            XCTAssertEqual(parameters[native]?.shape, weights[source]?.shape, source)
            XCTAssertEqual(parameters[native]?.dtype, weights[source]?.dtype, source)
        }
        let output = try bundle.target.forward(inputIDs: MLXArray([Int32(1), 2, 3]).reshaped(1, 3))
        let audio = try bundle.audioPatch.forward(
            clips: [.init(codes: [Int32](repeating: 2, count: 40), frameCount: 2)],
            limits: .init(
                maximumClips: 1, maximumFrames: 2, maximumPatches: 1,
                maximumWorkingElements: 1_000_000))
        eval(output.logits, audio.features)
        XCTAssertEqual(output.logits.shape, [1, 3, 128])
        XCTAssertEqual(audio.features.shape, [1, 64])
        XCTAssertEqual(audio.clipPatchRanges, [0 ..< 1])
        XCTAssertTrue(output.logits.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertTrue(audio.features.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
    }
}
