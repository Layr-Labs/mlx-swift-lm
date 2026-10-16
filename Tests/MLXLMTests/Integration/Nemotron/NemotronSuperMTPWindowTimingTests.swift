import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Wall-clock comparison, at the pinned Super dimensions with synthetic
/// weights, of one captured `[1, W]` verify window against W one-token target
/// steps over the same layer stack (ModelOpt FP8/NVFP4 graph path, BF16 KV),
/// whose logits and hidden must also agree bit for bit. Opt-in: set
/// `NEMOTRON_SUPER_MTP_WINDOW_BENCH`. Timings are reports, not assertions;
/// short stacks only, never the checkpoint.
@Suite(
    .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["NEMOTRON_SUPER_MTP_WINDOW_BENCH"] != nil))
struct NemotronSuperMTPWindowTimingTests {
    static let hidden = 4096

    static func configuration(pattern: String) -> NemotronHConfiguration {
        NemotronHConfiguration(
            vocabSize: 131072, hiddenSize: hidden, numHiddenLayers: pattern.count,
            numAttentionHeads: 32, numKeyValueHeads: 2, mambaNumHeads: 128, mambaHeadDim: 64,
            ssmStateSize: 128, convKernel: 4, nGroups: 8, intermediateSize: 2688,
            moeIntermediateSize: 2688, moeSharedExpertIntermediateSize: 5376,
            nRoutedExperts: 512, numExpertsPerTok: 22, hybridOverridePattern: pattern,
            headDim: 128, nSharedExperts: 1, routedScalingFactor: 5, moeLatentSize: 1024)
    }

    /// Seeded E4M3 codes without NaN encodings; rows repeat every 4096.
    static func codes(_ outputs: Int, _ inputs: Int, seed: UInt64) -> MLXArray {
        let block = MLXRandom.randInt(
            UInt8(0) ..< UInt8(127), [min(outputs, 4096), inputs], key: MLXRandom.key(seed))
        let signed =
            block
            | (MLXRandom.randInt(
                UInt8(0) ..< UInt8(2), block.shape, key: MLXRandom.key(seed &+ 1)) * UInt8(128))
        return tiled(signed, repetitions: [(outputs + 4095) / 4096, 1])[..<outputs]
    }

    static func nvfp4(_ inputs: Int, _ outputs: Int, seed: UInt64, scale: Float) throws
        -> ModelOptNVFP4Weights
    {
        try ModelOptNVFP4Weights(
            packedBytes: MLXRandom.randInt(
                UInt8(0) ..< UInt8(255), [outputs, inputs / 2], key: MLXRandom.key(seed)),
            e4m3ScaleBytes: MLXRandom.randInt(
                UInt8(40) ..< UInt8(60), [outputs, inputs / 16], key: MLXRandom.key(seed &+ 1)),
            tensorScale: scale)
    }

    /// Installs the checkpoint's precision mix, then seeds only the remaining
    /// dense parameters so replaced placeholders are never evaluated.
    static func model(pattern: String) throws -> NemotronSuperModel {
        let c = configuration(pattern: pattern)
        let language = NemotronHModel(c)
        // Lazy narrowing; the replaced placeholders are never evaluated.
        let narrowed = language.parameters().flattened().map { name, value -> (String, MLXArray) in
            let keep = name.hasSuffix("A_log") || name.hasSuffix("e_score_correction_bias")
            return (name, keep ? value : value.asType(.bfloat16))
        }
        try language.update(parameters: .unflattened(narrowed), verify: [.noUnusedKeys])
        var projections: [String: ModelOptCalibratedLinear] = [:]
        var experts: [String: ModelOptNVFP4SwitchLinear] = [:]
        var seed: UInt64 = 1
        func fp8(_ inputs: Int, _ outputs: Int, scale: Float) throws -> ModelOptCalibratedLinear {
            seed &+= 2
            return try ModelOptCalibratedLinear(
                weights: ModelOptFP8Weights(
                    e4m3Bytes: codes(outputs, inputs, seed: seed), tensorScale: scale),
                inputScale: 0.02, weightDType: .bfloat16)
        }
        for (layer, kind) in pattern.enumerated() {
            let prefix = "backbone.layers.\(layer).mixer."
            switch kind {
            case "M":
                projections[prefix + "in_proj"] = try fp8(hidden, 18560, scale: 1.0 / 4096)
                projections[prefix + "out_proj"] = try fp8(8192, hidden, scale: 1.0 / 8192)
            case "*":
                projections[prefix + "q_proj"] = try fp8(hidden, 4096, scale: 1.0 / 4096)
                projections[prefix + "k_proj"] = try fp8(hidden, 256, scale: 1.0 / 4096)
                projections[prefix + "v_proj"] = try fp8(hidden, 256, scale: 1.0 / 4096)
                projections[prefix + "o_proj"] = try fp8(4096, hidden, scale: 1.0 / 8192)
            case "E":
                seed &+= 8
                projections[prefix + "shared_experts.up_proj"] = try ModelOptCalibratedLinear(
                    weights: nvfp4(hidden, 5376, seed: seed, scale: 0.01), inputScale: 0.01,
                    weightDType: .bfloat16)
                projections[prefix + "shared_experts.down_proj"] = try ModelOptCalibratedLinear(
                    weights: nvfp4(5376, hidden, seed: seed &+ 2, scale: 0.01), inputScale: 0.05,
                    weightDType: .bfloat16)
                let up = try nvfp4(1024, 2688, seed: seed &+ 4, scale: 0.01)
                let down = try nvfp4(2688, 1024, seed: seed &+ 6, scale: 0.01)
                experts[prefix + "switch_mlp.fc1"] = try ModelOptNVFP4SwitchLinear(
                    experts: Array(repeating: up, count: 512),
                    inputScales: Array(repeating: 0.01, count: 512), weightDType: .bfloat16)
                experts[prefix + "switch_mlp.fc2"] = try ModelOptNVFP4SwitchLinear(
                    experts: Array(repeating: down, count: 512),
                    inputScales: Array(repeating: 0.05, count: 512), weightDType: .bfloat16)
            default: break
            }
        }
        projections["lm_head"] = try fp8(hidden, 131072, scale: 1.0 / 4096)
        try language.installModelOptProjections(projections)
        try language.installModelOptExperts(experts)
        let installed = Array(projections.keys) + Array(experts.keys)
        var updates: [(String, MLXArray)] = []
        for (index, (name, value)) in language.parameters().flattened().enumerated()
        where [.float32, .bfloat16, .float16].contains(value.dtype)
            && !installed.contains(where: { name.hasPrefix($0 + ".") })
        {
            let leaf = name.split(separator: ".").last.map(String.init) ?? name
            if ["A_log", "D", "dt_bias"].contains(leaf) { continue }
            let keep = leaf == "e_score_correction_bias"
            let key = MLXRandom.key(UInt64(1000 + index))
            let next: MLXArray
            if name.hasSuffix("embeddings.weight") {
                let block = MLXRandom.normal([4096, value.dim(1)], key: key)
                next = tiled(block, repetitions: [value.dim(0) / 4096, 1])
            } else if value.ndim == 1 && name.contains("norm") {
                next = 1 + 0.1 * MLXRandom.normal(value.shape, key: key)
            } else if value.ndim == 1 {
                next = 0.1 * MLXRandom.normal(value.shape, key: key)
            } else {
                next =
                    MLXRandom.normal(value.shape, key: key)
                    * (1 / Float(value.dim(-1)).squareRoot())
            }
            updates.append((name, keep ? next.asType(.float32) : next.asType(.bfloat16)))
        }
        try language.update(parameters: .unflattened(updates), verify: [.noUnusedKeys])
        eval(language.parameters())
        let images = NemotronSuperImageComposition(
            tower: NemotronRadio(
                NemotronRadioDimensions(
                    width: 2, layers: 1, heads: 1,
                    channels: 1, patch: 2, positionSide: 2, prefix: 1, mlpWidth: 3)),
            projector: NemotronSuperVisionProjector(
                visionWidth: 2, hiddenWidth: 3, outputWidth: hidden),
            width: hidden, imageTokenID: 18)
        return try NemotronSuperModel(languageModel: language, imageComposition: images)
    }

    @Test(arguments: ["M*E", "MM**EE", "MM*EE", "MM**E"])
    func windowVersusSerialSteps(pattern: String) throws {
        let model = try Self.model(pattern: pattern)
        let target = try NemotronSuperMTPCaptureTarget(capturing: model)
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 64 << 20, kvDType: .bfloat16))
        let bank = CBv2LayerCacheBank(
            caches: target.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
        let rows = try backend.makeSequenceState(
            layerKinds: model.cbv2LayerKinds, promptLength: 64, maxLength: 128)
        let state = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
        defer {
            bank.releaseBoundRows()
            backend.release(rows)
            try? state.release()
        }
        func caches() throws -> [KVCache] {
            try bank.layerCaches(rowStates: [rows]).map { try #require($0 as? KVCache) }
        }
        let prompt = MLXArray((0 ..< 64).map { Int32(($0 * 37 + 5) % 131000 + 20) })
            .reshaped(1, 64)
        let prefill = try state.bind()
        let first = target.cbv2ForwardWithHiddenForPrefill(
            prompt, caches: try caches(), recurrentState: [prefill], positionIds: nil,
            requirement: .lastPositionLogits)
        eval([first.logits] + (try prefill.evaluate()))
        try prefill.commit()

        let inputs = [Int32(101), 2024, 77, 31337]
        /// One measured pass, then rolled back so every pass starts equal.
        /// Returns the wall time and every position's logits and hidden.
        func pass(width: Int, captured: Bool) throws -> (UInt64, [MLXArray]) {
            for row in rows.compactMap({ $0 }) { row.beginSpeculativeWrite() }
            let attending = try bank.layerCaches(rowStates: [rows])
            let serializing = attending.compactMap { $0 as? any CBv2MTPRectangularSerializing }
            var transactions: [CBv2RecurrentStateEvaluation] = []
            var outputs: [MLXArray] = []
            let start = DispatchTime.now().uptimeNanoseconds
            if captured {
                for cache in serializing { cache.mtpSerializesRectangularAttention = true }
                let transaction = try state.bind()
                transactions.append(transaction)
                let output = target.cbv2ForwardWithHiddenCaptured(
                    MLXArray(Array(inputs.prefix(width))).reshaped(1, width),
                    caches: try caches(), recurrentState: [transaction], positionIds: nil)
                eval([output.logits, output.lastHidden] + (try transaction.evaluate()))
                for cache in serializing { cache.mtpSerializesRectangularAttention = false }
                outputs = [output.logits, output.lastHidden]
            } else {
                for token in inputs.prefix(width) {
                    let transaction = try state.bind()
                    transactions.append(transaction)
                    let output = target.cbv2ForwardWithHidden(
                        MLXArray([token]).reshaped(1, 1), caches: try caches(),
                        recurrentState: [transaction], positionIds: nil)
                    eval([output.logits, output.lastHidden] + (try transaction.evaluate()))
                    outputs.append(contentsOf: [output.logits, output.lastHidden])
                }
                outputs = [
                    concatenated(
                        stride(from: 0, to: outputs.count, by: 2).map { outputs[$0] }, axis: 1),
                    concatenated(
                        stride(from: 1, to: outputs.count, by: 2).map { outputs[$0] }, axis: 1),
                ]
            }
            let elapsed = DispatchTime.now().uptimeNanoseconds - start
            for transaction in transactions.reversed() { try transaction.rollback() }
            for row in rows.compactMap({ $0 }) {
                row.rollback(width)
                row.commitSpeculativeWrite()
            }
            bank.invalidateBoundComposition()
            return (elapsed, outputs)
        }
        func median(width: Int, captured: Bool) throws -> (UInt64, [MLXArray]) {
            let (_, outputs) = try pass(width: width, captured: captured)
            var samples: [UInt64] = []
            for _ in 0 ..< 9 { samples.append(try pass(width: width, captured: captured).0) }
            return (samples.sorted()[4], outputs)
        }
        var line = "SYNTHETIC-WEIGHTS super-dims pattern=\(pattern) median ns:"
        for width in 1 ... 4 {
            let window = try median(width: width, captured: true)
            let serial = try median(width: width, captured: false)
            line += " W=\(width) window=\(window.0) serial=\(serial.0)"
            // The same window is bit-identical to the serial steps here too.
            for (a, b) in zip(window.1, serial.1) {
                #expect(
                    a.shape == b.shape && a.dtype == .bfloat16 && b.dtype == .bfloat16
                        && arrayEqual(a.view(dtype: .uint16), b.view(dtype: .uint16))
                            .item(Bool.self),
                    "\(pattern) W=\(width)")
            }
        }
        print(line)
    }
}
