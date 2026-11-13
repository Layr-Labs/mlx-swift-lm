// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

@testable import MLXLLM

/// The published drafter and the two target tensors it borrows, loaded once
/// per test process from the directories the environment names. Real-weight
/// tests are explicitly skipped without them.
final class MuseGlimmerDFlashRealFixture: @unchecked Sendable {
    static let environment = ProcessInfo.processInfo.environment
    static let targetDirectory = environment["MUSE_GLIMMER_NVFP4_DIR"]
    static let drafterDirectory = environment["MUSE_GLIMMER_ASSISTANT_DIR"]
    /// The independent oracle's `fixtures` directory.
    static let oracleDirectory = environment["MUSE_GLIMMER_ORACLE_DIR"]
    /// Traces written by the oracle with every committed position's context.
    static let traceDirectory = environment["MUSE_GLIMMER_DFLASH_TRACE_DIR"]
    /// Where study results (JSON) are written.
    static let resultsDirectory = environment["MUSE_GLIMMER_DFLASH_RESULTS_DIR"]
    /// Where the quantized artifact is written.
    static let artifactDirectory = environment["MUSE_GLIMMER_DFLASH_ARTIFACT_DIR"]

    static var hasWeights: Bool { targetDirectory != nil && drafterDirectory != nil }

    /// Shards of `nvidia/Muse-Glimmer-30B-NVFP4` at revision `47818374` that
    /// hold the embedding table and the output head.
    static let embeddingShard = (
        "model-00001-of-00003.safetensors",
        "0e260a8f2c5e16bcc2ad8e90a8a59f1e889431b0de82f4c490c31d9ea9114ce4"
    )
    static let headShard = (
        "model-00003-of-00003.safetensors",
        "017293c10a76b131f36116c597f6a4eb9a4dd52d8773318b8df4b6afcc625132"
    )

    static let shared = Result { try MuseGlimmerDFlashRealFixture() }

    let drafter: MuseGlimmerDFlashDraftModel
    /// `embed_tokens`, `[vocabulary, hidden]`, BF16. The drafter takes rows
    /// of it without the target's embedding norm.
    let embedding: MLXArray
    private let headWeights: ModelOptNVFP4Weights
    private let headPacked: MLXArray
    private let headScales: MLXArray
    private let headScale: Float
    let loadSeconds: Double

    private init() throws {
        guard let target = Self.targetDirectory, let assistant = Self.drafterDirectory else {
            throw MuseGlimmerDFlashError.artifact("environment")
        }
        let start = Date()
        drafter = try MuseGlimmerDFlashDraftModel.load(
            directory: URL(fileURLWithPath: assistant), pin: .museGlimmer30BBF16,
            maximumBytes: 5_111_970_304)
        let root = URL(fileURLWithPath: target)
        let embeddingFile = try ModelOptTensorFile(
            url: root.appendingPathComponent(Self.embeddingShard.0), sha256: Self.embeddingShard.1)
        embedding = try ModelOptTensorSource(
            file: embeddingFile, name: "model.language_model.embed_tokens.weight"
        ).array(maximumBytes: 3 << 30)
        let headFile = try ModelOptTensorFile(
            url: root.appendingPathComponent(Self.headShard.0), sha256: Self.headShard.1)
        let packed = try ModelOptTensorSource(file: headFile, name: "lm_head.weight").array(
            maximumBytes: 1 << 30)
        let scales = try ModelOptTensorSource(file: headFile, name: "lm_head.weight_scale").array(
            maximumBytes: 1 << 30)
        headScale = try ModelOptTensorSource(file: headFile, name: "lm_head.weight_scale_2")
            .scalar()
        // Weight-only NVFP4 as ModelOpt stores it: value = e2m1 nibble x E4M3
        // block scale x `weight_scale_2` (Model-Optimizer
        // modelopt/torch/quantization/qtensor/nvfp4_tensor.py). The SDK type
        // checks the layout; the product below is MLX's NVFP4 kernel.
        headWeights = try ModelOptNVFP4Weights(
            packedBytes: packed, e4m3ScaleBytes: scales, tensorScale: headScale)
        headPacked = headWeights.packedWeights
        headScales = headWeights.blockScales
        eval(embedding, headPacked, headScales)
        loadSeconds = Date().timeIntervalSince(start)
    }

    var configuration: MuseGlimmerDFlashConfiguration { drafter.configuration }

    /// Raw `lm_head` rows in float32: no output multiplier, no softcap.
    func head(_ hidden: MLXArray) -> MLXArray {
        quantizedMatmul(
            hidden.asType(.float32), headPacked, scales: headScales, biases: nil,
            transpose: true, groupSize: 16, bits: 4, mode: .nvfp4) * headScale
    }

    /// Dense rows of the head, for checking the kernel against the formula.
    func denseHeadRows(_ rows: Range<Int>) throws -> MLXArray {
        try ModelOptNVFP4Weights(
            packedBytes: headPacked.view(dtype: .uint8)[rows, 0...],
            e4m3ScaleBytes: headScales[rows, 0...], tensorScale: headScale
        ).dequantized(dtype: .float32)
    }

    /// Block embeddings for an anchor token.
    func blockEmbeddings(anchor: Int, model: MuseGlimmerDFlashDraftModel) -> MLXArray {
        model.blockEmbeddings(
            anchor: embedding[anchor], mask: embedding[configuration.maskTokenID])
    }

    /// A copy of the drafter sharing its dense arrays, optionally in float32.
    func drafterCopy(dtype: DType = .bfloat16) throws -> MuseGlimmerDFlashDraftModel {
        let copy = try MuseGlimmerDFlashDraftModel(configuration, dtype: dtype)
        let parameters = drafter.parameters().flattened().map { ($0.0, $0.1.asType(dtype)) }
        try copy.update(parameters: .unflattened(parameters), verify: .all)
        eval(copy)
        return copy
    }

    /// Draft tokens of rows `1...count` of a block: ids and float32 logits.
    func drafts(
        model: MuseGlimmerDFlashDraftModel, state: MuseGlimmerDFlashContextState, anchor: Int,
        count: Int? = nil
    ) -> (ids: MLXArray, logits: MLXArray) {
        let hidden = model.draftBlock(
            embeddings: blockEmbeddings(anchor: anchor, model: model), state: state)
        let rows = count ?? (configuration.blockSize - 1)
        let logits = head(hidden[1 ..< (1 + rows), 0...])
        return (argMax(logits, axis: -1).asType(.int32), logits)
    }

    static func writeResult(_ name: String, _ object: [String: Any]) throws {
        let data = try JSONSerialization.data(
            withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        print("MUSE_DFLASH_RESULT \(name) \(String(decoding: data, as: UTF8.self))")
        guard let directory = resultsDirectory else { return }
        try data.write(to: URL(fileURLWithPath: directory).appendingPathComponent(name + ".json"))
    }
}

/// One oracle trace: a prompt, its greedy continuation and the target's
/// five-layer context features for every committed position.
struct MuseGlimmerDFlashTrace {
    let name: String
    let tokens: [Int]
    let promptLength: Int
    /// Committed positions; `tokens` has one more entry, the last anchor.
    let committed: Int
    let anchorPositions: [Int]
    let oracleDrafts: [[Int]]
    let oracleTopIDs: [[Int]]
    let oracleTopRaw: [[Float]]
    let oracleAccepted: [Int]
    private let file: MuseGlimmerDFlashNPZ

    init(_ url: URL) throws {
        name = url.deletingPathExtension().lastPathComponent
        file = try MuseGlimmerDFlashNPZ(url)
        tokens = try file.ints("tokens")
        promptLength = try file.ints("prompt_length")[0]
        committed = try file.shape("features")[0]
        anchorPositions = try file.ints("round_anchor_position")
        let rounds = anchorPositions.count
        func split<T>(_ values: [T], _ width: Int) -> [[T]] {
            (0 ..< rounds).map { Array(values[($0 * width) ..< (($0 + 1) * width)]) }
        }
        oracleDrafts = split(try file.ints("round_drafts"), 15)
        oracleTopIDs = split(try file.ints("round_top5_ids"), 75)
        oracleTopRaw = split(try file.floats("round_top5_raw"), 75)
        oracleAccepted = try file.ints("round_n_matches")
    }

    /// Context feature rows `[rows, 33280]`, float32.
    func features(_ rows: Range<Int>) throws -> MLXArray { try file.rows("features", rows) }

    static func all() throws -> [MuseGlimmerDFlashTrace] {
        guard let directory = MuseGlimmerDFlashRealFixture.traceDirectory else { return [] }
        return try FileManager.default.contentsOfDirectory(atPath: directory)
            .filter { $0.hasSuffix(".npz") && !$0.contains(".tmp") }.sorted()
            .map { try MuseGlimmerDFlashTrace(URL(fileURLWithPath: directory + "/" + $0)) }
    }

    /// Drafts a target that picks `tokens` accepts after the anchor at
    /// `position`: the longest prefix equal to the continuation, capped at
    /// `limit`; nil when the trace ends inside the window.
    func accepted(_ drafts: [Int], after position: Int, limit: Int) -> Int? {
        guard position + limit < tokens.count else { return nil }
        var count = 0
        while count < limit, drafts[count] == tokens[position + 1 + count] { count += 1 }
        return count
    }
}
