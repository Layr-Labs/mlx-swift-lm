// Copyright 2026 bstnxbt
// SPDX-License-Identifier: Apache-2.0
// Swift port of dflash-mlx DFlash2 at the revision recorded in NOTICE.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public final class DFlash2DraftModel: Module, @unchecked Sendable {
    public static let runtimeQuantization = (
        groupSize: 64,
        bits: 4,
        mode: QuantizationMode.affine
    )

    public let configuration: DFlash2Configuration

    @ModuleInfo(key: "fc") public var contextProjection: Linear
    @ModuleInfo(key: "hidden_norm") public var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [DFlash2DecoderLayer]
    @ModuleInfo public var norm: RMSNorm
    @ModuleInfo(key: "candidate_selector") var candidateSelector: DFlash2CandidateSelector

    public init(configuration: DFlash2Configuration) throws {
        try configuration.validatePinnedContract()
        self.configuration = configuration
        _contextProjection.wrappedValue = Linear(
            configuration.hiddenSize * configuration.targetLayerIDs.count,
            configuration.hiddenSize,
            bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: Float(configuration.rmsNormEpsilon))
        _layers.wrappedValue = (0 ..< configuration.hiddenLayers).map { _ in
            DFlash2DecoderLayer(configuration: configuration)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: Float(configuration.rmsNormEpsilon))
        _candidateSelector.wrappedValue = DFlash2CandidateSelector(
            configuration: configuration)
        super.init()
    }

    public static func loadConfiguration(from directory: URL) throws -> DFlash2Configuration {
        let configURL = directory.appending(component: "config.json")
        let configuration = try JSONDecoder().decode(
            DFlash2Configuration.self,
            from: Data(contentsOf: configURL))
        try configuration.validatePinnedContract()
        return configuration
    }

    public static func remappedWeightName(_ name: String) throws -> String {
        switch name {
        case "candidate_selector.predecessor_codebook":
            "candidate_selector.predecessor_codebook.weight"
        case "candidate_selector.successor_codebook":
            "candidate_selector.successor_codebook.weight"
        default:
            name
        }
    }

    func makeCache() -> [DFlash2ContextKVCache] {
        layers.map { _ in
            DFlash2ContextKVCache(
                sinkSize: 64,
                windowSize: configuration.slidingWindow)
        }
    }

    public func projectTargetFeatures(_ targetFeatures: MLXArray) -> MLXArray {
        hiddenNorm(contextProjection(targetFeatures))
    }

    func forwardProjectedContext(
        noiseEmbedding: MLXArray,
        draftContext: MLXArray,
        cache: [DFlash2ContextKVCache]
    ) -> MLXArray {
        var hidden = noiseEmbedding
        for index in layers.indices {
            hidden = layers[index](
                hidden,
                targetContext: draftContext,
                cache: cache[index])
        }
        return norm(hidden)
    }

    func advanceProjectedContextCache(
        draftContext: MLXArray,
        cache: [DFlash2ContextKVCache]
    ) {
        for index in layers.indices {
            layers[index].advanceProjectedContextCache(
                draftContext,
                cache: cache[index])
        }
    }

    func callAsFunction(
        noiseEmbedding: MLXArray,
        targetFeatures: MLXArray,
        cache: [DFlash2ContextKVCache]
    ) -> MLXArray {
        forwardProjectedContext(
            noiseEmbedding: noiseEmbedding,
            draftContext: projectTargetFeatures(targetFeatures),
            cache: cache)
    }

    public func selectProposal(
        draftHidden: MLXArray,
        logits: MLXArray,
        anchorIDs: MLXArray,
        temperature: Float,
        captureQ: Bool = false
    ) -> DFlash2DraftProposal {
        candidateSelector.select(
            hidden: draftHidden,
            logits: logits,
            anchorIDs: anchorIDs,
            temperature: temperature,
            captureQ: captureQ)
    }

    public static func load(from directory: URL) async throws -> DFlash2DraftModel {
        let configuration = try loadConfiguration(from: directory)
        let model = try DFlash2DraftModel(configuration: configuration)
        var weights = [String: MLXArray]()
        let enumerator = FileManager.default.enumerator(
            at: directory, includingPropertiesForKeys: nil)
        let shards = (enumerator?.allObjects as? [URL] ?? [])
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.path < $1.path }

        for shard in shards {
            let (arrays, _) = try loadArraysAndMetadata(url: shard)
            for (rawName, value) in arrays {
                let name = try remappedWeightName(rawName)
                if weights[name] != nil {
                    throw DFlash2ConfigurationError.invariant(
                        "duplicate DFlash2 weight: \(name)")
                }
                weights[name] = value
            }
        }
        if shards.isEmpty {
            throw DFlash2ConfigurationError.invariant(
                "DFlash2 checkpoint has no safetensors files")
        }

        try model.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: [.all])
        let runtimeQuantization = Self.runtimeQuantization
        quantize(
            model: model,
            groupSize: runtimeQuantization.groupSize,
            bits: runtimeQuantization.bits,
            mode: runtimeQuantization.mode)
        eval(model)
        return model
    }
}
