import Foundation
import MLX
import MLXNN

@testable import MLXLLM
@testable import MLXLMCommon

/// Generated miniature checkpoint, with real packed mmap PLE tensors. No
/// external artifact, environment override or production numerical baseline.
enum Qwen4FactoryFixture {
    static func write(to directory: URL) throws {
        var c = Qwen4ExpTextConfiguration()
        c.hiddenSize = 64; c.hiddenLayers = 2; c.attentionHeads = 2; c.kvHeads = 1; c.headDim = 64
        c.linearNumValueHeads = 2; c.linearNumKeyHeads = 1
        c.linearKeyHeadDim = 64; c.linearValueHeadDim = 64
        c.vocabularySize = 64; c.maxPositionEmbeddings = 512
        c.fullAttentionInterval = 2; c.layerTypes = ["linear_attention", "qwen_sparse_attention"]
        c.hcCount = 2; c.hcLowrank = 8; c.pleLayerIds = [1]; c.pleEmbedDim = 128
        c.pleConvKernelSize = 3; c.ngramSize = 3; c.headsPerNgram = 2
        c.ngramVocabSizeBase = 17; c.makeNgramVocabSizeDivisibleBy = 8; c.splitNgramParts = 1
        c.indexerNHeads = 2; c.indexerKVHeads = 1; c.indexerHeadDim = 32
        c.indexerBudget = 16; c.indexerCompressRatio = 4
        c.numExperts = 1; c.numExpertsPerTok = 1
        c.sharedExpertIntermediateSize = 32; c.moeIntermediateSize = 32
        c.mropeSection = [2, 1, 1]; c.partialRotaryFactor = 0.25; c.eosTokenId = [0]
        MLXRandom.seed(8319)
        let model = Qwen4ExpModel(Qwen4ExpConfiguration(textConfig: c))
        var weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map {
            ($0.0, $0.1.asType(.bfloat16))
        })
        let tables = Qwen4ExpNGramTables(c, pleIndex: 0)
        for (index, rows) in tables.shardSizes.enumerated() {
            let values = MLXRandom.normal([rows, tables.headEmbedDim]).asType(.bfloat16)
            let (weight, scales, biases) = quantized(values, groupSize: 32, bits: 4)
            let prefix = "language_model.model.layers.0.ple.ple_embedding.ngram_embedding.shards.\(index)"
            weights[prefix + ".weight"] = weight
            weights[prefix + ".scales"] = scales
            weights[prefix + ".biases"] = biases
        }
        try JSONEncoder().encode(Qwen4ExpConfiguration(textConfig: c))
            .write(to: directory.appendingPathComponent("config.json"))
        let shard = "model.safetensors"
        try save(arrays: weights, url: directory.appendingPathComponent(shard))
        try JSONSerialization.data(withJSONObject: ["weight_map":
            Dictionary(uniqueKeysWithValues: weights.keys.map { ($0, shard) })])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        let processor = #"{"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"processor_class":"Qwen3VLProcessor","image_processor_type":"Qwen2VLImageProcessorFast"}"#
        try Data(processor.utf8).write(to: directory.appendingPathComponent("preprocessor_config.json"))
    }

    struct Loader: TokenizerLoader {
        var fail = false
        enum Failure: Error { case tokenizer }
        func load(from directory: URL) async throws -> any Tokenizer {
            if fail { throw Failure.tokenizer }
            return TestTokenizer(vocabularySize: 64)
        }
    }
}
