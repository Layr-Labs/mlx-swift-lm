import Foundation
import MLX
import Testing

@testable import MLXLLM

/// Opt-in physical GPU capture of actual checkpoint operators. Captures only
/// layers 0–2 and the output projection. Metal may still snapshot all loaded
/// checkpoint buffers (about 21 GiB); reserve at least 25 GiB temporary disk.
/// This is dispatch evidence, not a whole-model performance or quality test.
@Suite(.serialized, .tags(.integration))
struct FlintCaptureTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["FLINT_METAL_CAPTURE"] != nil))
    func captureOfficialDenseMoEAttentionAndHead() throws {
        let env = ProcessInfo.processInfo.environment
        let checkpoint = URL(fileURLWithPath: try #require(env["FLINT_CHECKPOINT_DIRECTORY"]))
        let output = URL(fileURLWithPath: try #require(env["FLINT_METAL_CAPTURE"]))
        try #require(env["MTL_CAPTURE_ENABLED"] == "1" && env["MLX_ENABLE_TF32"] == "0")
        try #require(!FileManager.default.fileExists(atPath: output.path))
        let config = try JSONDecoder().decode(
            FlintConfiguration.self,
            from: Data(contentsOf: checkpoint.appendingPathComponent("config.json")))
        try #require(config.hiddenLayers == 49 && config.mlxExport != nil)
        let model = FlintModel(config)
        try model.loadFlintWeights(from: checkpoint)
        let tokens = ((MLXArray(0 ..< 512) * 7 + 3) % 150000).reshaped(1, -1)
        func operators() -> MLXArray {
            let caches = model.newCache(parameters: nil)
            var h = model.model.embedding(tokens)
            for index in 0 ..< 3 { h = model.model.layers[index](h, cache: caches[index]) }
            // The reference keeps this projection's input FP32. With TF32=0
            // its portable fallback must remain visible in the capture.
            return model.lmHead!(model.model.norm(h[0..., (-1)..., 0...]).asType(.float32))
        }
        let warm = operators()
        eval(warm)
        try withError { GPU.startCapture(url: output) }
        defer { GPU.stopCapture(url: output) }
        let captured = operators()
        eval(captured)
        #expect(all(isFinite(captured)).item(Bool.self))
        #expect(arrayEqual(warm, captured).item(Bool.self))
    }
}
