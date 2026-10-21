import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import Testing

@testable import MLXLLM

/// Super's chunked prefill SSD (``NemotronSuperSSDChunked``) against the
/// `ssmAttn` oracle, and the whole chunked Mamba prefill against the current
/// path at Super's shapes. NAX GPUs run the native fragment layout; other
/// GPUs check the same kernel through the generic layout. Both need a Metal
/// compiler that builds them (`runtimeCompilerSupported`, macOS 27).
@Suite(
    .tags(.integration), .serialized,
    .enabled(
        if: ModelOptNVFP4NAXExperts.runtimeCompilerSupported,
        "needs the macOS 27 Metal compiler"))
struct NemotronSuperMambaPrefillTests {
    @Test(arguments: [
        (37, 64, false), (300, 64, true), (300, 32, true), (2048, 64, true), (2048, 32, true),
        (2000, 64, false), (2000, 32, false),
    ])
    func chunkedScanStaysWithinRoundingOfTheOracle(length: Int, chunk: Int, withState: Bool)
        throws
    {
        let (heads, headDim, groups, stateDim) = (16, 64, 2, 128)
        let keys = MLXRandom.split(key: MLXRandom.key(UInt64(length + chunk)), into: 6)
        let x = MLXRandom.normal([1, length, heads, headDim], key: keys[0]).asType(.bfloat16)
        let B = (MLXRandom.normal([1, length, groups, stateDim], key: keys[1]) * 0.5)
            .asType(.bfloat16)
        let C = (MLXRandom.normal([1, length, groups, stateDim], key: keys[2]) * 0.5)
            .asType(.bfloat16)
        let dt = (MLXRandom.normal([1, length, heads], key: keys[3]) - 3).asType(.bfloat16)
        let ALog = MLX.log(MLXRandom.uniform(low: 1, high: 16, [heads], key: keys[4]))
        let D = MLXArray.ones([heads])
        let dtBias = MLXArray.zeros([heads])
        let state =
            withState
            ? MLXRandom.normal([1, heads, headDim, stateDim], key: keys[5]) * 0.1
            : MLXArray.zeros([1, heads, headDim, stateDim])
        let reference = ssmAttn(
            x: x, ALog: ALog, B: B, C: C, D: D, dt: dt, dtBias: dtBias, state: state,
            timeStepLimit: (0, .infinity))
        let xbc = concatenated(
            [x.reshaped(1, length, -1), B.reshaped(1, length, -1), C.reshaped(1, length, -1)],
            axis: -1)
        let chunked = NemotronSuperSSDChunked.scan(
            xbc: xbc, dt: dt, dtOffset: 0, heads: heads, headDim: headDim, groups: groups,
            stateSize: stateDim, aLog: ALog, dtBias: dtBias, D: D, state: state,
            timeStepLimit: (0, .infinity), outputDType: reference.0.dtype, chunkSize: chunk)
        for (name, ours, oracle) in [
            ("output", chunked.y, reference.0), ("state", chunked.state, reference.1),
        ] {
            #expect(ours.shape == oracle.shape && ours.dtype == oracle.dtype)
            let a = ours.asType(.float32)
            let b = oracle.asType(.float32)
            let relative =
                (abs(a - b).max() / maximum(abs(b).max(), 1e-30)).item(Float.self)
            print("chunked SSD \(name) L=\(length) Q=\(chunk): max deviation / max \(relative)")
            #expect(!isNaN(a).any().item(Bool.self))
            #expect(relative <= 0x1p-6)
        }
    }

    /// The hook: a mixer selected for the chunked path (here by its own
    /// override, leaving the global switch alone) runs exactly the chunked
    /// computation in the CBv2 mixer, and the legacy cache path stays within
    /// rounding of the current one with the same cached states.
    @Test func selectedMixerTakesTheChunkedPath() throws {
        let (mixer, hidden, convState, ssmState) =
            try NemotronSuperMambaPrefill.comparisonLayer(tokens: 256, seed: 1)
        let chunked = mixer.recurrentMix(
            hidden, convState: convState, ssmState: ssmState, chunked: true)
        let current = mixer.recurrentMix(
            hidden, convState: convState, ssmState: ssmState, chunked: false)
        for (selected, expected) in [(true, chunked), (false, current)] {
            mixer.chunkedSuperPrefill = selected
            let routed = mixer.recurrentMix(hidden, convState: convState, ssmState: ssmState)
            #expect(modelOptBitwiseEqual(routed.0, expected.0))
            #expect(modelOptBitwiseEqual(routed.2, expected.2))
        }

        var caches: [MambaCache] = []
        var outputs: [MLXArray] = []
        for selected in [false, true] {
            mixer.chunkedSuperPrefill = selected
            let cache = MambaCache()
            let output = mixer(hidden, attentionMask: .none, ssmMask: nil, cache: cache)
            eval(output, cache.state)
            caches.append(cache)
            outputs.append(output)
        }
        mixer.chunkedSuperPrefill = nil
        #expect(modelOptBitwiseEqual(caches[0][0]!, caches[1][0]!))
        let state = (abs(caches[1][1]! - caches[0][1]!).max() / abs(caches[0][1]!).max())
            .item(Float.self)
        #expect(caches[1][1]!.dtype == .float32 && state <= 0x1p-10)
        let output = (abs(outputs[1] - outputs[0]).mean() / abs(outputs[0]).mean())
            .item(Float.self)
        #expect(output.isFinite && output <= 0x1p-5)
    }

    @Test func superLayerAgainstTheCurrentPath() throws {
        let comparison = try NemotronSuperMambaPrefill.compareWithReference(
            tokens: 2048, iterations: 5)
        print(comparison)
        for value in [
            comparison.outputMaxRel, comparison.ssdOutputMaxRel, comparison.ssmStateMaxRel,
        ] {
            #expect(value.isFinite && value <= 0x1p-5)
        }
        // The convolution reproduces the op chain's roundings.
        #expect(comparison.convStateMaxAbs == 0)
    }
}
