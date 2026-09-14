import MLX
import MLXNN
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

/// Identical trusted prefix and forced inputs, with no sampler or assistant.
/// Keep exact comparisons: a first mismatch identifies the target arithmetic
/// or state seam, rather than hiding it behind free-running token divergence.
final class Qwen4RectangularStateParityTests: XCTestCase {
    private func model() -> Qwen4ExpTextModel {
        var c = Qwen4ExpTextConfiguration()
        c.hiddenSize = 64; c.hiddenLayers = 2; c.attentionHeads = 2; c.kvHeads = 1; c.headDim = 64
        c.linearNumValueHeads = 2; c.linearNumKeyHeads = 1
        c.linearKeyHeadDim = 64; c.linearValueHeadDim = 64
        c.vocabularySize = 64; c.maxPositionEmbeddings = 512
        c.fullAttentionInterval = 2; c.layerTypes = ["linear_attention", "qwen_sparse_attention"]
        c.hcCount = 2; c.hcLowrank = 8; c.pleLayerIds = []; c.pleEmbedDim = 64
        c.indexerNHeads = 2; c.indexerKVHeads = 1; c.indexerHeadDim = 32
        c.indexerBudget = 16; c.indexerCompressRatio = 4
        c.numExperts = 1; c.numExpertsPerTok = 1
        c.sharedExpertIntermediateSize = 32; c.moeIntermediateSize = 32
        c.mropeSection = [2, 1, 1]; c.partialRotaryFactor = 0.25
        MLXRandom.seed(8301)
        let model = Qwen4ExpTextModel(c)
        model.update(parameters: ModuleParameters.unflattened(
            model.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        eval(model)
        return model
    }

    private struct State {
        let recurrent: CBv2RecurrentRequestState
        let cache: CBv2LayerCache
        let row: CBv2FullSequenceKV
    }

    private func empty(_ model: Qwen4ExpTextModel) throws -> State {
        let kind = model.cbv2LayerKinds[0]
        let row = CBv2FullSequenceKV(promptLength: 256, maxLength: 512, kvHeads: 1, headDim: 64)
        return try State(recurrent: CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec),
            cache: CBv2LayerCache(layerIndex: 0, kind: kind, rows: [row]), row: row)
    }

    private func clone(_ source: State, model: Qwen4ExpTextModel) throws -> State {
        let result = try empty(model)
        let kv = try XCTUnwrap(source.row.snapshot())
        _ = result.row.update(keys: kv.keys, values: kv.values)
        result.cache.setRows([result.row])
        try result.row.restoreQwen4Indexer(source.row.snapshotQwen4Indexer())
        _ = CBv2Qwen4IndexerBind.restore(result.cache, from: result.row)
        let recurrent = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec,
            adoptedCommitted: XCTUnwrap(source.recurrent.confirmedStateSnapshot()))
        return State(recurrent: recurrent, cache: result.cache, row: result.row)
    }

    private struct Trace {
        let stages: [String: MLXArray]
        let conv: MLXArray
        let ssm: MLXArray
    }

    private func trace(_ tokens: MLXArray, state: State, model: Qwen4ExpTextModel, captured: Bool) throws -> Trace {
        let transaction = try state.recurrent.bind()
        var hidden = model.model.embedAndTile(tokens)
        var stages = ["embedding": hidden]
        for (index, layer) in model.model.layers.enumerated() {
            let attentionInput = layer.attnHyperConnection(hidden).0
            stages["layer\(index).attn_input"] = attentionInput
            if let attention = layer.selfAttn {
                let indexer = attention.indexer
                let projected = Qwen4ExpAffineQMM.apply(indexer.indexQKProj, attentionInput)
                    .reshaped(1, tokens.dim(1), indexer.nHeads + indexer.kvHeads, indexer.headDim)
                let queries = indexer.qLayerNorm(projected[0..., 0..., 0..<indexer.nHeads, 0...])
                    .transposed(0, 2, 1, 3)
                let positions = Qwen4ExpTextPositions.synthesized(offset: state.row.absoluteOffset, length: tokens.dim(1))
                stages["layer\(index).index_queries"] = attention.mrope.apply(
                    queries: queries, keys: queries, positionIds: positions).0.transposed(0, 2, 1, 3)
            }
            hidden = layer.cbv2Forward(hidden, inputIds: tokens, modelLayerIndex: index,
                attentionCache: index == 1 ? state.cache : nil,
                recurrentState: [transaction], positionIds: nil, captureRecurrentWindow: captured)
            stages["layer\(index).output"] = hidden
        }
        let mixed = model.model.hyperConnectionMixer.mix(hidden)
        stages["head_input"] = mixed
        stages["logits"] = model.headLogits(mixed)
        let recurrentArrays = try transaction.evaluate()
        eval(Array(stages.values) + recurrentArrays + state.cache.innerState())
        XCTAssertEqual(recurrentArrays.count, 2)
        if captured { try transaction.commit(keepPositions: tokens.dim(1)) }
        else { try transaction.commit() }
        return Trace(stages: stages, conv: recurrentArrays[0], ssm: recurrentArrays[1])
    }

    private func compare(_ actual: MLXArray, _ expected: MLXArray, stage: String, width: Int, column: Int) {
        XCTAssertEqual(actual.shape, expected.shape)
        let error = abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
        print("[qwen4-rectangular-state] width=\(width) column=\(column) stage=\(stage) max_abs=\(error)")
        XCTAssertEqual(error, 0, "width=\(width) column=\(column) stage=\(stage)")
    }

    func testIdenticalPrefixLocatesFirstRectangularTargetDifference() throws {
        let model = model()
        let prefix = try empty(model)
        let prompt = (0..<149).map { 1 + ($0 * 9 + 11) % 61 }
        let beforePrefill = Qwen4ExpQSAInvocation.snapshot()
        for start in stride(from: 0, to: prompt.count, by: 128) {
            let stop = min(start + 128, prompt.count)
            _ = try trace(MLXArray(prompt[start..<stop].map(Int32.init), [1, stop - start]),
                state: prefix, model: model, captured: false)
        }
        let afterPrefill = Qwen4ExpQSAInvocation.snapshot()
        XCTAssertEqual(afterPrefill.portable - beforePrefill.portable, 5,
                       "ordinary prefill retains four 32-query tiles plus its final 21-query tile")
        XCTAssertEqual(afterPrefill.decode, beforePrefill.decode,
                       "canonical verification must not serialize ordinary prefill")
        for width in [2, 3, 4, 5, 6] {
            let serial = try clone(prefix, model: model)
            let rectangular = try clone(prefix, model: model)
            rectangular.cache.mtpSerializesRectangularAttention = true
            let inputs = (0..<width).map { Int32(3 + $0 * 11) }
            var singles: [Trace] = []
            for token in inputs {
                singles.append(try trace(MLXArray([token], [1, 1]), state: serial, model: model, captured: false))
            }
            let priorGDN = Qwen4ExpGDNStackedVerify.snapshot()
            let priorQSA = Qwen4ExpQSAInvocation.snapshot()
            let batch = try trace(MLXArray(inputs, [1, width]), state: rectangular, model: model, captured: true)
            let gdn = Qwen4ExpGDNStackedVerify.snapshot()
            let qsa = Qwen4ExpQSAInvocation.snapshot()
            XCTAssertEqual(qsa.decode - priorQSA.decode, width)
            XCTAssertEqual(qsa.portable, priorQSA.portable,
                           "unsupported-geometry verification must use canonical sparse query calls")
            print("[qwen4-rectangular-dispatch] width=\(width) gdn_stacked=\(gdn.stacked - priorGDN.stacked) gdn_chained=\(gdn.chained - priorGDN.chained) qsa_native=\(qsa.native - priorQSA.native) qsa_portable=\(qsa.portable - priorQSA.portable) qsa_verify=\(qsa.verify - priorQSA.verify)")
            let stages = ["embedding", "layer0.attn_input", "layer0.output", "layer1.attn_input", "layer1.output", "head_input", "logits"]
            for column in 0..<width {
                for stage in stages {
                    compare(batch.stages[stage]![0..., column..<column + 1, 0...], singles[column].stages[stage]!,
                        stage: stage, width: width, column: column)
                }
                compare(batch.conv[column..<column + 1], singles[column].conv, stage: "gdn.conv", width: width, column: column)
                compare(batch.ssm[column..<column + 1], singles[column].ssm, stage: "gdn.ssm", width: width, column: column)
            }
            let actual = try rectangular.row.snapshotQwen4Indexer()
            let expected = try serial.row.snapshotQwen4Indexer()
            let pooled = try XCTUnwrap(actual.pooledIndexKeys)
            let selection = Qwen4ExpGatheredQSA.verifySelectedBlocks(
                indexQueries: batch.stages["layer1.index_queries"]!, pooledIndexKeys: pooled,
                keyTokens: actual.tokenCount, indexerHeadDim: 32, maxBlocks: actual.tokenCount / 4,
                blockBudget: 4, ratio: 4)
            for column in 0..<width {
                let visible = 149 + column + 1
                let oracle = Qwen4ExpGatheredQSA.decodeSelectedBlocks(
                    indexQueries: singles[column].stages["layer1.index_queries"]!,
                    pooledIndexKeys: pooled[0..., 0..<visible / 4, 0...], keyTokens: visible,
                    indexerHeadDim: 32, maxBlocks: visible / 4, blockBudget: 4)
                compare(selection[0..., column..<column + 1, 0...], oracle,
                    stage: "qsa.selected_blocks", width: width, column: column)
            }
            XCTAssertEqual(actual.tokenCount, expected.tokenCount)
            XCTAssertEqual(rectangular.row.qwen4IndexTokenCount, serial.row.qwen4IndexTokenCount)
            compare(actual.indexKeys, expected.indexKeys, stage: "qsa.index_keys", width: width, column: -1)
            compare(actual.positionIds, expected.positionIds, stage: "qsa.positions", width: width, column: -1)
            if let a = actual.pooledIndexKeys, let b = expected.pooledIndexKeys {
                compare(a, b, stage: "qsa.pooled_keys", width: width, column: -1)
            }
            let akv = try XCTUnwrap(rectangular.row.snapshot())
            let bkv = try XCTUnwrap(serial.row.snapshot())
            compare(akv.keys, bkv.keys, stage: "qsa.keys", width: width, column: -1)
            compare(akv.values, bkv.values, stage: "qsa.values", width: width, column: -1)
        }
    }
}
