import MLX
import MLXRandom
import Testing

@testable import MLXLLM

/// Super's fused SSD prefill windows against the shared `ssmAttn` oracle:
/// within 2^-7 of each row maximum (deviations are printed) and, as the window
/// kernel reproduces the oracle's segment sums, exponential and mask, bit for
/// bit.
@Suite(.tags(.integration))
struct NemotronSuperSSDTests {
    @Test(arguments: [
        (37, 16, 16, 4, 32, false), (300, 16, 16, 4, 32, true), (600, 8, 32, 2, 16, true),
        (512, 128, 64, 8, 128, true), (2304, 16, 16, 4, 32, true), (2048, 8, 32, 2, 16, false),
    ])
    func fusedWindowsStayWithinRoundingOfTheOracle(
        length: Int, heads: Int, headDim: Int, groups: Int, stateDim: Int, withState: Bool
    ) throws {
        let key = MLXRandom.key(UInt64(length * heads + headDim))
        let keys = MLXRandom.split(key: key, into: 6)
        let x = (MLXRandom.normal([1, length, heads, headDim], key: keys[0])).asType(.bfloat16)
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
            ? MLXRandom.normal([1, heads, headDim, stateDim], key: keys[5]) * 0.1 : nil
        let reference = ssmAttn(
            x: x, ALog: ALog, B: B, C: C, D: D, dt: dt, dtBias: dtBias, state: state,
            timeStepLimit: (0, .infinity))
        let fused = NemotronSuperSSD.scan(
            x: x, ALog: ALog, B: B, C: C, D: D, dt: dt, dtBias: dtBias, state: state,
            timeStepLimit: (0, .infinity))
        for (name, a, b) in [("output", fused.0, reference.0), ("state", fused.1, reference.1)] {
            #expect(a.shape == b.shape && a.dtype == b.dtype)
            let ours = a.asType(.float32)
            let oracle = b.asType(.float32)
            let rowMax = abs(oracle).max(axis: -1, keepDims: true)
            let relative = (abs(ours - oracle) / maximum(rowMax, 1e-30)).max().item(Float.self)
            print("fused SSD \(name) L=\(length) H=\(heads): max deviation / row max \(relative)")
            #expect(!isNaN(ours).any().item(Bool.self))
            #expect(relative <= 0x1p-7)
            // The window kernel reproduces the oracle's ops, so the results
            // are the oracle's bits.
            #expect(modelOptBitwiseEqual(a, b))
        }
    }
}
