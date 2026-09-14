import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

/// A resident miniature table tests request history/conv ownership. Real SSD
/// deferred-fill and full-model concurrency remain separate physical gates.
final class Qwen4BatchedPLETests: XCTestCase {
    func testPLEBatchRowsKeepTheirOwnHistoryAcrossReorderAndLeave() throws {
        var c = Qwen4ExpTextConfiguration()
        c.hiddenSize = 16; c.hcCount = 2; c.hiddenLayers = 1
        c.layerTypes = ["qwen_sparse_attention"]; c.pleLayerIds = [1]
        c.pleEmbedDim = 8; c.ngramSize = 3; c.headsPerNgram = 2
        c.ngramVocabSizeBase = 17; c.makeNgramVocabSizeDivisibleBy = 8
        c.splitNgramParts = 1; c.vocabularySize = 64; c.eosTokenId = [0]
        c.pleConvKernelSize = 3
        MLXRandom.seed(8307)
        let layer = Qwen4ExpPLELayer(c, layerIndex: 0, pleIndex: 0, mmap: false)
        layer.update(parameters: ModuleParameters.unflattened(
            layer.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
        eval(layer)
        let spec = c.cbv2RecurrentStateSpec()
        let rows = try (0..<2).map { _ in try CBv2RecurrentRequestState(spec: spec) }
        let solos = try (0..<2).map { _ in try CBv2RecurrentRequestState(spec: spec) }
        let pleIndex = Qwen4ExpNGramGeometry.recurrentLayerIndex(0)
        let orders = [[0], [1], [0, 1], [1, 0], [0], [0, 1]]
        var positions = [0, 0]
        for (step, order) in orders.enumerated() {
            let width = step < 2 ? (step == 0 ? 3 : 7) : 1
            let inputRows = order.map { row in
                MLXRandom.normal([1, width, 32], key: MLXRandom.key(UInt64(9000 + row * 100 + positions[row])))
                    .asType(.bfloat16)
            }
            let tokenRows = order.map { row in
                MLXArray((0..<width).map { Int32((row * 17 + positions[row] + $0) % 61) }, [1, width])
            }
            var expected: [MLXArray] = []
            for (index, row) in order.enumerated() {
                let transaction = try solos[row].bind()
                let output = layer.cbv2Forward(inputRows[index], inputIds: tokenRows[index], recurrentState: [transaction])
                eval([output] + (try transaction.evaluate()))
                try transaction.commit()
                expected.append(output)
            }
            let transactions = try order.map { try rows[$0].bind() }
            let output = layer.cbv2Forward(
                concatenated(inputRows, axis: 0), inputIds: concatenated(tokenRows, axis: 0),
                recurrentState: transactions)
            eval([output] + (try transactions.flatMap { try $0.evaluate() }))
            for transaction in transactions { try transaction.commit() }
            for (index, row) in order.enumerated() {
                XCTAssertTrue(all(output[index ..< index + 1] .== expected[index]).item(Bool.self),
                              "PLE output step=\(step) row=\(row)")
                let state = try XCTUnwrap(rows[row].confirmedStateSnapshot()?[pleIndex])
                let reference = try XCTUnwrap(solos[row].confirmedStateSnapshot()?[pleIndex])
                let actualConv = try XCTUnwrap(state.conv)
                let expectedConv = try XCTUnwrap(reference.conv)
                let actualHistory = try XCTUnwrap(state.ssm)
                let expectedHistory = try XCTUnwrap(reference.ssm)
                XCTAssertTrue(all(actualConv .== expectedConv).item(Bool.self))
                XCTAssertTrue(all(actualHistory .== expectedHistory).item(Bool.self))
                XCTAssertEqual(rows[row].materializedByteCount, rows[row].byteCount)
                positions[row] += width
            }
        }
    }
}
