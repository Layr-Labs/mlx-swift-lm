// Qwen4ExpNGramStoredConstantsTests.swift
//
// The checkpoint stores the n-gram hash constants the original model was
// trained with. The hash uses the constants the configuration derives, so a
// load whose stored copies differ must refuse by name.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen4ExpNGramStoredConstantsTests: XCTestCase {

    private func embedding() throws -> Qwen4ExpNGramEmbedding {
        Qwen4ExpNGramEmbedding(
            try Qwen4ExpFixture.configuration(), embedDimensions: 8, pleLayerIndex: 0)
    }

    /// Load `multipliers` and `sizes` into the stored buffers, the way the
    /// checkpoint load does.
    private func store(
        _ embedding: Qwen4ExpNGramEmbedding, multipliers: [Int64], sizes: [Int64]
    ) throws {
        try embedding.update(
            parameters: ModuleParameters.unflattened([
                "layer_multipliers": MLXArray(multipliers),
                "ngram_heads_vocab_sizes": MLXArray(sizes),
            ]),
            verify: [])
    }

    private func derived(_ embedding: Qwen4ExpNGramEmbedding) -> (
        multipliers: [Int64], sizes: [Int64]
    ) {
        (
            embedding.constants.multipliers.asArray(Int64.self),
            embedding.constants.headVocabSizes.map(Int64.init)
        )
    }

    private func assertRefuses(
        _ embedding: Qwen4ExpNGramEmbedding, field: String, line: UInt = #line
    ) {
        XCTAssertThrowsError(try embedding.validateStoredHashConstants(), line: line) { error in
            guard let mismatch = error as? Qwen4ExpNGramHashConstantsMismatch else {
                return XCTFail("refused with \(error)", line: line)
            }
            XCTAssertEqual(mismatch.field, field, line: line)
            XCTAssertTrue(
                mismatch.description.hasPrefix("QWEN38-NGRAM-HASH-CONSTANTS-MISMATCH"),
                line: line)
        }
    }

    func testMatchingStoredConstantsAreAccepted() throws {
        let embedding = try embedding()
        let (multipliers, sizes) = derived(embedding)
        try store(embedding, multipliers: multipliers, sizes: sizes)
        XCTAssertNoThrow(try embedding.validateStoredHashConstants())
    }

    /// A module built without a load holds zeros: there is no stored copy.
    func testUnloadedBuffersAreNotCompared() throws {
        XCTAssertNoThrow(try embedding().validateStoredHashConstants())
    }

    func testOtherMultipliersAreRefused() throws {
        let embedding = try embedding()
        let (multipliers, sizes) = derived(embedding)
        try store(embedding, multipliers: multipliers.map { $0 + 2 }, sizes: sizes)
        assertRefuses(embedding, field: "layer_multipliers")
    }

    func testOtherHeadVocabularySizesAreRefused() throws {
        let embedding = try embedding()
        let (multipliers, sizes) = derived(embedding)
        try store(embedding, multipliers: multipliers, sizes: sizes.reversed())
        assertRefuses(embedding, field: "ngram_heads_vocab_sizes")
    }
}
