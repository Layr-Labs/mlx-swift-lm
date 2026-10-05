// Qwen4ExpNGramHashSeedTests.swift
//
// The n-gram hash seed. The pinned checkpoints carry no `seed` key, so the
// configuration default decides the hash multipliers, and it must be the
// original model's (`transformers` `Qwen4ExpConfig.seed`, 1234). The original
// checkpoint stores the seed-1234 multipliers at
// `layers.1.ple.ple_embedding.layer_multipliers`. A seed of 0 derives another
// triple, and every token then reads the wrong n-gram rows.
//
// The pinned triples come from an independent splitmix64 computation outside
// this codebase.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class Qwen4ExpNGramHashSeedTests: XCTestCase {

    static let seed1234Multipliers: [Int64] = [
        23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071,
    ]
    static let seed0Multipliers: [Int64] = [
        4_788_054_244_585, 5_075_510_189_727, 24_189_832_309_785,
    ]

    /// The multipliers that the configuration `json` derives for the first
    /// PLE layer. An empty configuration is the production geometry
    /// (vocabulary 248,320, n-gram size 3).
    private func multipliers(_ json: String) throws -> [Int64] {
        let configuration = try JSONDecoder().decode(
            Qwen4ExpTextConfiguration.self, from: Data(json.utf8))
        return Qwen4ExpNGramConstants(configuration, pleLayerIndex: 0)
            .multipliers.asArray(Int64.self)
    }

    func testTheDefaultSeedIsTheOriginalModels() throws {
        let configuration = try JSONDecoder().decode(
            Qwen4ExpTextConfiguration.self, from: Data("{}".utf8))
        XCTAssertEqual(configuration.seed, 1234)
    }

    func testTheDefaultSeedDerivesTheCheckpointsMultipliers() throws {
        XCTAssertEqual(try multipliers("{}"), Self.seed1234Multipliers)
    }

    func testAZeroSeedDerivesAnotherTriple() throws {
        XCTAssertEqual(try multipliers(#"{"seed": 0}"#), Self.seed0Multipliers)
    }
}
