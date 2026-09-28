import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM
import XCTest

final class MiMoV26ConvertedWeightsTests: XCTestCase {
    func testActualMetadataPlanCoversCompleteBundle() throws {
        let inputs = try fixture("actual")
        let plan = try inputs.plan()
        XCTAssertEqual(plan.descriptors.count, 1258)
        XCTAssertEqual(plan.rootFiles.count, 52)
        XCTAssertEqual(plan.tensorBytes, 175362324736)
        XCTAssertEqual(plan.components[.target]?.count, 709)
        XCTAssertEqual(plan.components[.vision]?.count, 364)
        XCTAssertEqual(plan.components[.audioPatch]?.count, 95)
        XCTAssertEqual(plan.components[.mtp]?.count, 90)
        XCTAssertEqual(plan.targetExpertModulePaths.count, 141)
        XCTAssertEqual(plan.configuration.attentionProjectionLayout, "fused_qkv")
        XCTAssertFalse(plan.descriptors.keys.contains { $0.contains("qkv_proj") })
        XCTAssertEqual(plan.descriptors["model.layers.0.self_attn.q_proj.weight"]?.shape, [12288,4096])
        XCTAssertEqual(plan.descriptors["model.layers.0.self_attn.v_proj.weight"]?.shape, [512,4096])
        XCTAssertEqual(plan.descriptors["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.dtype, .uint32)
        XCTAssertEqual(plan.descriptors["model.layers.1.mlp.switch_mlp.gate_proj.scales"]?.dtype, .uint8)
        XCTAssertEqual(plan.descriptors["model.layers.1.mlp.gate.e_score_correction_bias"]?.dtype, .float32)
        XCTAssertEqual(plan.configSHA256, "6f1895d782b0b872961a9e03dfef52144c5983d1c7373194290ea854889cacc6")
        XCTAssertEqual(plan.indexSHA256, "2bcc3ba76e2c5502299858e2e8d20b4cb11ae26e7b18609a196c06317e9ac959")
        XCTAssertEqual(plan.externalRequiredComponents, ["audio_tokenizer", "dflash"])
        XCTAssertNil(plan.provenance.payloadVerificationReceiptSHA256)
        XCTAssertEqual(plan.configuration.rawFields, try JSONDecoder().decode(MiMoV26Configuration.self, from: inputs.config).rawFields)
        XCTAssertEqual(plan.descriptorSHA256, try inputs.plan().descriptorSHA256)
    }

    func testMetadataRejectsMissingExtraShapesPrecisionAndFiles() throws {
        let inputs = try fixture("tiny")
        let key = "model.layers.1.mlp.switch_mlp.gate_proj.weight", original = try XCTUnwrap(inputs.descriptors[key])
        var missing = inputs; missing.descriptors.removeValue(forKey: "visual.blocks.1.attn.sinks")
        XCTAssertThrowsError(try missing.plan())
        var extra = inputs; extra.descriptors["audio_tokenizer.encoder.weight"] = original
        XCTAssertThrowsError(try extra.plan())
        var wrongShape = inputs
        wrongShape.descriptors[key] = .init(shape: [2,64,9], dtype: .uint32, file: original.file)
        XCTAssertThrowsError(try wrongShape.plan())
        var wrongDType = inputs
        wrongDType.descriptors[key] = .init(shape: original.shape, dtype: .uint8, file: original.file)
        XCTAssertThrowsError(try wrongDType.plan())
        var fused = inputs
        fused.descriptors.removeValue(forKey: "model.layers.0.self_attn.q_proj.weight")
        fused.descriptors["model.layers.0.self_attn.qkv_proj.weight"] = original
        XCTAssertThrowsError(try fused.plan())
        var badFile = inputs
        badFile.descriptors[key] = .init(shape: original.shape, dtype: original.dtype, file: "../target.safetensors")
        var badIndex = try object(badFile.index), badMap = try XCTUnwrap(badIndex["weight_map"] as? [String:String])
        badMap[key] = "../target.safetensors"; badIndex["weight_map"] = badMap; badFile.index = try data(badIndex)
        XCTAssertThrowsError(try badFile.plan()) {
            XCTAssertEqual($0 as? MiMoV26ConvertedLoadError, .invalidInventory(key))
        }
        var wrongTotal = inputs
        var index = try object(wrongTotal.index); index["metadata"] = ["total_size": 1]
        wrongTotal.index = try data(index)
        XCTAssertThrowsError(try wrongTotal.plan())
        var indexExtra = inputs
        var idx = try object(indexExtra.index), map = try XCTUnwrap(idx["weight_map"] as? [String:String])
        map["unexpected.weight"] = "target.safetensors"; idx["weight_map"] = map; indexExtra.index = try data(idx)
        XCTAssertThrowsError(try indexExtra.plan())
    }

    func testMetadataRejectsUnmappedAndConflictingPolicies() throws {
        let inputs = try fixture("tiny")
        let key = "mtp.layers.0.self_attn.q_proj", alias = "language_model." + key
        for change in ["missing", "conflict", "dense", "typo", "unknownMetadata", "headCount"] {
            var test = inputs, c = try object(test.config), q = try XCTUnwrap(c["quantization"] as? [String:Any])
            switch change {
            case "missing": q.removeValue(forKey: key); q.removeValue(forKey: alias)
            case "conflict": q[alias] = ["mode":"mxfp4","bits":4,"group_size":32]
            case "dense": q["model.layers.0.self_attn.q_proj"] = ["mode":"mxfp4","bits":4,"group_size":32]
            case "typo": q["language_model.mtp.layers.7.self_attn.q_proj"] = ["mode":"affine","bits":4,"group_size":64]
            case "unknownMetadata": q["unparsed"] = "ignore me"
            default: c["num_nextn_predict_layers"] = 2
            }
            c["quantization"] = q; test.config = try data(c)
            XCTAssertThrowsError(try test.plan(), change)
        }
        // One valid alias may name the same explicit projection policy.
        var single = inputs, c = try object(single.config), q = try XCTUnwrap(c["quantization"] as? [String:Any])
        q.removeValue(forKey: alias); c["quantization"] = q; single.config = try data(c)
        XCTAssertNoThrow(try single.plan())
    }

    func testProvenanceRejectsMalformedIdentityWithoutClaimingPayloadVerification() throws {
        var inputs = try fixture("tiny")
        inputs.provenance = .init(artifactID: "", sourceRepository: "wrong/model", sourceRevision: "main", conversionManifestSHA256: "unknown")
        XCTAssertThrowsError(try inputs.plan())
        inputs = try fixture("tiny")
        inputs.provenance = .init(artifactID: "tiny", sourceRepository: inputs.provenance.sourceRepository,
            sourceRevision: inputs.provenance.sourceRevision, conversionManifestSHA256: inputs.provenance.conversionManifestSHA256,
            payloadVerificationReceiptSHA256: String(repeating:"a", count:64))
        let plan = try inputs.plan()
        XCTAssertEqual(plan.provenance.payloadVerificationReceiptSHA256, String(repeating:"a", count:64))
        // This is only a supplied receipt reference, never a verified-byte flag.
        XCTAssertEqual(plan.tensorBytes, 370672)
    }

    func testNativeTinyBundleLoadsAllComponentsAndOnlyDeclaredQuantization() throws {
        try nativeLane()
        let inputs = try fixture("tiny"), plan = try inputs.plan(), weights = arrays(plan)
        let bundle = try MiMoV26ConvertedWeights.load(plan: plan, tensors: weights)
        XCTAssertEqual(bundle.plan.configSHA256, plan.configSHA256)
        XCTAssertTrue(bundle.mtp.isLoaded); XCTAssertEqual(bundle.mtp.headCount, 3)
        let leaves = bundle.target.leafModules().flattened()
        XCTAssertEqual(Set(leaves.compactMap { path,module in module is Quantized ? path : nil }), plan.targetExpertModulePaths)
        for (path, module) in leaves where plan.targetExpertModulePaths.contains(path) {
            let packed = try XCTUnwrap(module as? QuantizedSwitchLinear)
            XCTAssertEqual(packed.groupSize,32); XCTAssertEqual(packed.bits,4); XCTAssertEqual(packed.mode,.mxfp4)
        }
        let targetWeights = Dictionary(uniqueKeysWithValues: bundle.target.parameters().flattened())
        XCTAssertEqual(Set(targetWeights.keys), try XCTUnwrap(plan.components[.target]))
        for (key, value) in targetWeights { XCTAssertEqual(value.shape,weights[key]?.shape); XCTAssertEqual(value.dtype,weights[key]?.dtype) }
        let result = try bundle.target.forward(inputIDs: MLXArray([Int32(1),2,3]).reshaped(1,3))
        eval(result.logits)
        XCTAssertEqual(result.logits.shape,[1,3,128])
        XCTAssertTrue(result.logits.asArray(Float.self).allSatisfy(\.isFinite))
        let audio = try bundle.audioPatch.forward(clips:[.init(codes:[Int32](repeating:1,count:20),frameCount:1)],
            limits:.init(maximumClips:1,maximumFrames:1,maximumPatches:1,maximumWorkingElements:100_000))
        let vision = try bundle.vision.forward(patches:MLXArray.zeros([4,24]),grids:[.init(temporal:1,height:2,width:2)],
            limits:.init(maximumPatches:4,maximumAttentionScoreElements:1000))
        eval(audio.features,vision)
        XCTAssertEqual(audio.features.shape,[1,64]); XCTAssertEqual(vision.shape,[1,64])
        XCTAssertTrue(audio.features.asArray(Float.self).allSatisfy(\.isFinite)); XCTAssertTrue(vision.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testNativeInvalidSecondBundleDoesNotModifyPublishedFirstBundle() throws {
        try nativeLane()
        let plan = try fixture("tiny").plan(), weights = arrays(plan)
        let first = try MiMoV26ConvertedWeights.load(plan:plan,tensors:weights)
        let ids = MLXArray([Int32(1),2]).reshaped(1,2), before = try first.target.forward(inputIDs:ids).logits
        eval(before)
        var missing = weights; missing.removeValue(forKey:"mtp.layers.2.eh_proj.biases")
        XCTAssertThrowsError(try MiMoV26ConvertedWeights.load(plan:plan,tensors:missing))
        var badType = weights; badType["model.norm.weight"] = weights["model.norm.weight"]!.asType(.int32)
        XCTAssertThrowsError(try MiMoV26ConvertedWeights.load(plan:plan,tensors:badType))
        var extra = weights; extra["dflash.fc.weight"] = MLXArray.zeros([1])
        XCTAssertThrowsError(try MiMoV26ConvertedWeights.load(plan:plan,tensors:extra))
        // This metadata plan and tensor map pass complete inventory preflight,
        // but target construction rejects an unsupported execution semantic.
        // Exercise a late construction failure, not just missing-map failures.
        var late = try fixture("tiny"), configuration = try object(late.config)
        configuration["rope_scaling"] = ["type": "linear", "factor": 2]
        late.config = try data(configuration)
        let latePlan = try late.plan()
        XCTAssertThrowsError(try MiMoV26ConvertedWeights.load(plan: latePlan, tensors: weights)) {
            XCTAssertEqual($0 as? MiMoV26ExecutionError, .unsupportedConfiguration("rope_scaling"))
        }
        let after = try first.target.forward(inputIDs:ids).logits; eval(after)
        XCTAssertEqual(before.asArray(Float.self),after.asArray(Float.self))
        XCTAssertTrue(first.mtp.isLoaded)
    }

    private struct Inputs {
        var config, index: Data
        var descriptors: [String:MiMoV26ConvertedTensorDescriptor]
        var provenance: MiMoV26ConvertedProvenance
        func plan() throws -> MiMoV26ConvertedLoadPlan {
            try .make(configurationData:config,indexData:index,descriptors:descriptors,provenance:provenance)
        }
    }
    private func fixture(_ prefix: String) throws -> Inputs {
        guard let root = ProcessInfo.processInfo.environment["MIMO_V26_CONVERTED_LOAD_FIXTURES"] else {
            throw XCTSkip("Set MIMO_V26_CONVERTED_LOAD_FIXTURES to metadata-only fixture directory")
        }
        let url = URL(fileURLWithPath:root)
        let config = try Data(contentsOf:url.appendingPathComponent(prefix+"-config.json"))
        let index = try Data(contentsOf:url.appendingPathComponent(prefix+"-index.json"))
        let descriptors = try JSONDecoder().decode([String:MiMoV26ConvertedTensorDescriptor].self,
            from:Data(contentsOf:url.appendingPathComponent(prefix+"-descriptors.json")))
        let p = try object(Data(contentsOf:url.appendingPathComponent("provenance.json")))
        let provenance = try MiMoV26ConvertedProvenance(artifactID:XCTUnwrap(p["artifactID"] as? String),
            sourceRepository:XCTUnwrap(p["sourceRepository"] as? String),sourceRevision:XCTUnwrap(p["sourceRevision"] as? String),
            conversionManifestSHA256:XCTUnwrap(p["conversionManifestSHA256"] as? String))
        return .init(config:config,index:index,descriptors:descriptors,provenance:provenance)
    }
    private func nativeLane() throws {
        guard ProcessInfo.processInfo.environment["MIMO_V26_CONVERTED_LOAD_NATIVE_TESTS"] == "1" else {
            throw XCTSkip("Requires an exclusive native execution lane and MIMO_V26_CONVERTED_LOAD_NATIVE_TESTS=1")
        }
    }
    private func arrays(_ plan: MiMoV26ConvertedLoadPlan) -> [String:MLXArray] {
        plan.descriptors.mapValues { d in
            let count = d.shape.reduce(1,*)
            switch d.dtype {
            case .uint32: return MLXArray([UInt32](repeating:0x22222222,count:count)).reshaped(d.shape)
            case .uint8: return MLXArray([UInt8](repeating:127,count:count)).reshaped(d.shape)
            default:
                return MLXArray((0..<count).map { Float(($0 % 13)+1)/100 }).reshaped(d.shape)
                    .asType(d.dtype == .bfloat16 ? .bfloat16 : (d.dtype == .float16 ? .float16 : .float32))
            }
        }
    }
    private func object(_ d:Data) throws -> [String:Any] { try XCTUnwrap(JSONSerialization.jsonObject(with:d) as? [String:Any]) }
    private func data(_ object:[String:Any]) throws -> Data { try JSONSerialization.data(withJSONObject:object,options:[.sortedKeys]) }
}
