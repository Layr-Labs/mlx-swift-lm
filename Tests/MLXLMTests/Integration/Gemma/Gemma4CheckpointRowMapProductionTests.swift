#if canImport(Metal)
import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Actual original QAT tensors, through the real SwitchGLU route. Inputs and
/// routing are synthetic; this is not full-model/KV or endpoint qualification.
@Suite(
    "Gemma4 checkpoint production row-map route", .tags(.integration), .serialized,
    .enabled(if: ProcessInfo.processInfo.environment["GEMMA4_ROW_MAP_PRODUCTION_CHECKPOINT"] != nil)
)
struct Gemma4CheckpointRowMapProductionTests {
    typealias O = Gemma4RowMapProductionOracle

    private func loadBanks(
        directory: URL, layer: Int, weightMap: [String: String],
        quantization: [String: Any]
    ) throws -> [O.Bank] {
        let prefix = "language_model.model.layers.\(layer).experts.switch_glu."
        let projections = ["gate_proj", "up_proj", "down_proj"]
        let wanted = Set(
            projections.flatMap { projection in
                ["weight", "scales", "biases"].map { prefix + projection + "." + $0 }
            })
        try #require(wanted.count == 9 && wanted.allSatisfy { weightMap[$0] != nil })
        let shards = Set(wanted.compactMap { weightMap[$0] }).sorted()
        try #require(shards.allSatisfy { URL(fileURLWithPath: $0).lastPathComponent == $0 })
        let checkpoint = try loadModelCheckpoint(
            urls: shards.map { directory.appendingPathComponent($0) },
            isRead: { wanted.contains($0) })
        try #require(Set(checkpoint.weights.keys) == wanted)
        return try projections.map { projection in
            let name = prefix + projection
            let policy = quantization[name] as? [String: Any] ?? quantization
            try #require(policy["bits"] as? Int == 4 && policy["group_size"] as? Int == 64)
            try #require((policy["mode"] as? String ?? "affine") == "affine")
            let w = try #require(checkpoint.weights[name + ".weight"])
            let s = try #require(checkpoint.weights[name + ".scales"])
            let b = try #require(checkpoint.weights[name + ".biases"])
            let k = projection == "down_proj" ? 704 : 2816
            let n = projection == "down_proj" ? 2816 : 704
            try #require(w.dtype == .uint32 && w.shape == [128, n, k / 8])
            try #require(s.dtype == .bfloat16 && s.shape == [128, n, k / 64])
            try #require(b.dtype == .bfloat16 && b.shape == s.shape)
            eval(w, s, b)
            return O.Bank(weight: w, scales: s, biases: b)
        }
    }

    @Test func actualPackedBanksPreserveCompleteSwitchGLUGraph() throws {
        try O.lane()
        let report = try O.Report("actual-checkpoint-production")
        defer { try? report.flush() }
        let directory = URL(
            fileURLWithPath: try #require(
                ProcessInfo.processInfo.environment["GEMMA4_ROW_MAP_PRODUCTION_CHECKPOINT"]))
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let indexData = try Data(
            contentsOf: directory.appendingPathComponent("model.safetensors.index.json"))
        try #require(
            O.hash(configData) == "29910322dd085f45c8f95c6c0f1611b20f722d6f6c8394321b34817e98a972fa"
        )
        try #require(
            O.hash(indexData) == "5455e83705bbdd4e3702c7d4f9d49d4900e84533036628f74500538075dd5c80")
        let config = try #require(
            try JSONSerialization.jsonObject(with: configData) as? [String: Any])
        let quantization = try #require(config["quantization"] as? [String: Any])
        let index = try #require(
            try JSONSerialization.jsonObject(with: indexData) as? [String: Any])
        let weights = try #require(index["weight_map"] as? [String: String])
        var cases = 0
        for layer in [0, 15, 29] {
            let banks = try loadBanks(
                directory: directory, layer: layer, weightMap: weights, quantization: quantization)
            let before = banks.flatMap { [$0.weight, $0.scales, $0.biases] }.map {
                O.hash(O.bytes($0))
            }
            let a = try O.model(banks)
            let b = try O.model(banks, original: true)
            for (rows, hot) in [(64, false), (2048, false), (2047, false), (2048, true)] {
                let input = O.inputs(rows, hot: hot)
                for fused in [false, true] {
                    try O.compare(
                        a, b, input: input, report: report,
                        label: "layer\(layer) T\(rows) hot\(hot) fused\(fused)", eligible: true,
                        fused: fused)
                    cases += 1
                }
            }
            #expect(
                before
                    == banks.flatMap { [$0.weight, $0.scales, $0.biases] }.map {
                        O.hash(O.bytes($0))
                    })
        }
        try #require(cases == 24 && report.rows.count == 24)
        print(
            "GEMMA4_ROW_MAP_PRODUCTION_CHECKPOINT cases=24 immutable_banks=9 requested=\(Gemma4RowMappedNAX.requested)"
        )
        report.completed = true
    }
}
#endif
