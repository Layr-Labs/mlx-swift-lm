import Foundation
import MLX
@_spi(DiffusionGemmaDiagnostics) import MLXLMCommon
import Testing

@Suite("DiffusionGemma native sampler contract", .serialized)
struct DiffusionGemmaNativeSamplerTests {
    private final class Keys {
        var key = MLXRandom.key(341)
        var count = 0
        func next() -> MLXArray {
            let parts = MLXRandom.split(key: key)
            key = parts.0
            count += 1
            return parts.1
        }
    }
    private func same(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && a.dtype == b.dtype
            && a.asData(access: .copy).data == b.asData(access: .copy).data
    }

    @Test func fallbackPreservesNativeKeysAndState() throws {
        for temperature: Float in [0, 1, 0.7] {
            let configuration = try DiffusionGemmaGenerationConfiguration(
                maxDenoisingSteps: 3, sampler: .init(entropyBound: 0.2),
                stabilityThreshold: 2, confidenceThreshold: 0.01)
            let canvas = MLXArray.zeros([2, 4], dtype: .int32)
            let logits = MLXArray(0..<56).asType(.float32).reshaped(2, 4, 7) / Float(10)
            let original = try DiffusionGemmaDenoisingState(initialCanvas: canvas,
                vocabularySize: 7, embeddingDType: .float32, configuration: configuration)
            let native = try DiffusionGemmaDenoisingState(initialCanvas: canvas,
                vocabularySize: 7, embeddingDType: .float32, configuration: configuration)
            let a = Keys(), b = Keys()
            for _ in 0..<3 {
                let accepted = try original.step(rawLogits: logits, sample: { processed in
                    if temperature == 0 { return processed.argMax(axis: -1).asType(.int32) }
                    return MLXRandom.categorical(processed / temperature, key: a.next()).asType(.int32)
                }, noise: { MLXRandom.randInt(Int32(0)..<Int32(7), canvas.shape, key: a.next()) })
                let actual = try native.stepNative(rawLogits: logits, samplingTemperature: temperature, nextKey: b.next)
                #expect(same(accepted, actual))
                #expect(same(original.currentCanvas, native.currentCanvas))
                #expect(same(original.argmaxCanvas, native.argmaxCanvas))
                #expect(same(original.finishedRows, native.finishedRows))
                #expect(same(original.stepsUsed, native.stepsUsed))
                #expect(same(try #require(original.selfConditioningLogits), try #require(native.selfConditioningLogits)))
                #expect(original.remainingSteps == native.remainingSteps)
                #expect(a.count == b.count && same(a.key, b.key))
            }
            #expect(b.count == (temperature == 0 ? 3 : 6))
        }
    }

    @Test func invalidDistributionDoesNotAdvanceNativeKeys() throws {
        for bad in [Float.nan, Float.infinity, -Float.infinity] {
            let canvas = MLXArray.zeros([1, 4], dtype: .int32)
            let state = try DiffusionGemmaDenoisingState(initialCanvas: canvas,
                vocabularySize: 7, embeddingDType: .float32,
                configuration: DiffusionGemmaGenerationConfiguration())
            let keys = Keys()
            #expect(throws: DiffusionGemmaSamplingError.invalidDistribution) {
                try state.stepNative(rawLogits: MLXArray.full([1, 4, 7], values: MLXArray(bad)),
                    samplingTemperature: 1, nextKey: keys.next)
            }
            #expect(keys.count == 0 && state.remainingSteps == 48)
            #expect(same(state.currentCanvas, canvas))
            #expect(state.selfConditioningLogits == nil)
        }
    }

    @Test(.enabled(if:
        ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_SAMPLER_NUMERICAL_EDGE_LIVE"] == "1"
        && ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_COMPILED_SAMPLER"] == "1"))
    func compiledNativeSamplingPreservesExtremeTemperatureDraws() throws {
        let config = try DiffusionGemmaGenerationConfiguration(maxDenoisingSteps: 2)
        let canvas = MLXArray.zeros([1, 256], dtype: .int32)
        let row = which(MLXArray(0..<262144) .== 3, MLXArray(Float(40)), MLXArray(Float(-40)))
        let logits = broadcast(row, to: [1, 256, 262144])
        let original = try DiffusionGemmaDenoisingState(initialCanvas: canvas,
            vocabularySize: 262144, embeddingDType: .float32, configuration: config)
        let native = try DiffusionGemmaDenoisingState(initialCanvas: canvas,
            vocabularySize: 262144, embeddingDType: .float32, configuration: config)
        let a = Keys(), b = Keys()
        a.key = MLXRandom.key(419)
        b.key = MLXRandom.key(419)
        DiffusionGemmaCompiledSamplerDiagnostics.clearAndArm()
        for _ in 0..<2 {
            let accepted = try original.step(rawLogits: logits,
                sample: { MLXRandom.categorical($0 / Float.greatestFiniteMagnitude, key: a.next()).asType(.int32) },
                noise: { MLXRandom.randInt(Int32(0)..<Int32(262144), canvas.shape, key: a.next()) })
            let actual = try native.stepNative(rawLogits: logits,
                samplingTemperature: Float.greatestFiniteMagnitude, nextKey: b.next)
            let canvasExact = same(original.currentCanvas, native.currentCanvas)
            let argmaxExact = same(original.argmaxCanvas, native.argmaxCanvas)
            let acceptanceExact = same(accepted, actual)
            #expect(canvasExact && argmaxExact && acceptanceExact)
            #expect(a.count == b.count && same(a.key, b.key))
        }
        #expect(DiffusionGemmaCompiledSamplerDiagnostics.snapshotAndDisarm() == 2)
    }
}
