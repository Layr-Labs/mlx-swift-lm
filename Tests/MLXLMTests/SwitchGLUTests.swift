import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

// Serialized: SwitchGLU's init probes custom activations through compiled
// functions (`silu(probe)`), and its forward runs compiled GLU kernels.
// Running these tests in parallel can deadlock MLX-Swift's per-
// CompiledFunction locks (one test tracing inside a compiled call while
// another blocks on the same function's lock from init).
@Suite(.serialized)
struct SwitchGLUTests {
    @Test func fusedGateUpMatchesSeparateProjections() {
        let inputDims = 8
        let hiddenDims = 4
        let numExperts = 3

        let separate = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            activation: { $0 }, bias: false)
        let fused = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            activation: { $0 }, bias: false, fuseGateUp: true)

        let gate = values(numExperts * hiddenDims * inputDims)
            .reshaped(numExperts, hiddenDims, inputDims)
            .asType(.float32)
        let up = values(numExperts * hiddenDims * inputDims, offset: 1000)
            .reshaped(numExperts, hiddenDims, inputDims)
            .asType(.float32)
        let down = values(numExperts * inputDims * hiddenDims, offset: 2000)
            .reshaped(numExperts, inputDims, hiddenDims)
            .asType(.float32)

        separate.update(parameters: ModuleParameters.unflattened([
            "gate_proj.weight": gate,
            "up_proj.weight": up,
            "down_proj.weight": down,
        ]))
        fused.update(parameters: ModuleParameters.unflattened([
            "gate_up_proj.weight": concatenated([gate, up], axis: -2),
            "down_proj.weight": down,
        ]))

        let x = values(2 * inputDims, offset: 3000)
            .reshaped(2, inputDims)
            .asType(.float32)
        let indices = MLXArray([Int32(0), 2, 1, 0]).reshaped(2, 2)

        let separateOut = separate(x, indices)
        let fusedOut = fused(x, indices)
        eval(separateOut, fusedOut)

        let maxDiff = max(abs(separateOut - fusedOut)).item(Float.self)
        #expect(maxDiff == 0)
    }

    /// N1 fast-follow: same parity check for the *quantized* fused path. Affine
    /// quantization is independent per output row, so quantizing the fused
    /// `[gate; up]` weight yields gate/up rows bit-identical to quantizing the
    /// two projections separately -- the single gathered quantized matmul + split
    /// must therefore match two separate quantized matmuls exactly.
    @Test func fusedGateUpQuantizedMatchesSeparateProjections() {
        let inputDims = 64
        let hiddenDims = 64
        let numExperts = 3
        let groupSize = 64
        let bits = 4

        let separate = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            activation: { $0 }, bias: false)
        let fused = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts,
            activation: { $0 }, bias: false, fuseGateUp: true)

        let gate = values(numExperts * hiddenDims * inputDims)
            .reshaped(numExperts, hiddenDims, inputDims)
            .asType(.float32)
        let up = values(numExperts * hiddenDims * inputDims, offset: 20000)
            .reshaped(numExperts, hiddenDims, inputDims)
            .asType(.float32)
        let down = values(numExperts * inputDims * hiddenDims, offset: 40000)
            .reshaped(numExperts, inputDims, hiddenDims)
            .asType(.float32)

        separate.update(parameters: ModuleParameters.unflattened([
            "gate_proj.weight": gate,
            "up_proj.weight": up,
            "down_proj.weight": down,
        ]))
        fused.update(parameters: ModuleParameters.unflattened([
            "gate_up_proj.weight": concatenated([gate, up], axis: -2),
            "down_proj.weight": down,
        ]))

        quantize(model: separate, groupSize: groupSize, bits: bits)
        quantize(model: fused, groupSize: groupSize, bits: bits)

        let x = values(2 * inputDims, offset: 60000)
            .reshaped(2, inputDims)
            .asType(.float32)
        let indices = MLXArray([Int32(0), 2, 1, 0]).reshaped(2, 2)

        let separateOut = separate(x, indices)
        let fusedOut = fused(x, indices)
        eval(separateOut, fusedOut)

        let maxDiff = max(abs(separateOut - fusedOut)).item(Float.self)
        #expect(maxDiff == 0)
    }

    /// Regression for the removed runtime fused gate+up cache: SwitchGLU must
    /// retain NO weight copies beyond its parameters. The removed cache
    /// concatenated a full second copy of the quantized gate+up expert weights
    /// on the module at the first decode-shaped forward (`indices.size <= 8`)
    /// — ~8 GiB (qat-4bit) / ~15 GiB (8bit) model-wide on Gemma 4 26B, and the
    /// root cause of the v0.7.3 production incident. This pins both views:
    ///
    ///   * parameter identity — every parameter array is the SAME instance
    ///     after N decode-shaped forwards (nothing re-pointed, nothing added);
    ///   * MLX active memory — after a warm-up forward (compiled-kernel
    ///     constants etc. already materialized), N more forwards through the
    ///     old fused trigger shape leave active memory flat to well under the
    ///     bytes a fused gate+up copy of this module would occupy.
    @Test func decodeShapedForwardsRetainNoExtraWeightCopies() {
        // Sized so a retained gate+up copy (~36 MB quantized here) dwarfs the
        // process-global GPU-counter noise other concurrently-running suites
        // add (measured single-digit MB) — the assertion threshold sits well
        // above the noise and an order of magnitude below the signal — while
        // keeping the transient fp32 instantiation (~384 MB pre-quantization)
        // inside what low-memory CI runners tolerate.
        let inputDims = 2048
        let hiddenDims = 2048
        let numExperts = 8

        let glu = SwitchGLU(
            inputDims: inputDims, hiddenDims: hiddenDims, numExperts: numExperts)
        quantize(model: glu, groupSize: 64, bits: 4)

        // Decode shape: B=1, top_k=8 → indices.size == 8, the exact per-call
        // gate the removed fused dispatch keyed on.
        let x = values(inputDims, offset: 100).reshaped(1, inputDims).asType(.float32)
        let indices = MLXArray((0 ..< 8).map(Int32.init)).reshaped(1, 8)

        // Warm-up: materialize lazy parameters and one-time kernel state
        // through a shape the old lazy cache could NOT trigger on
        // (indices.size == 16 > 8). If the cache is ever reintroduced with
        // its first-decode-forward build, it stays unbuilt until the
        // measured loop below and lands inside the growth check — warming up
        // through the trigger shape would have built and retained it before
        // the snapshot, silently passing the test.
        let warmX = values(2 * inputDims, offset: 200)
            .reshaped(2, inputDims).asType(.float32)
        let warmIndices = MLXArray((0 ..< 16).map { Int32($0 % 8) }).reshaped(2, 8)
        eval(glu(warmX, warmIndices))
        MLX.eval(glu.parameters().flattened().map(\.1))

        let paramsBefore = Dictionary(
            uniqueKeysWithValues: glu.parameters().flattened())
        let weightBytes = paramsBefore.reduce(0) { $0 + $1.value.nbytes }
        Memory.clearCache()
        let activeBefore = GPU.activeMemory

        for _ in 0 ..< 8 {
            eval(glu(x, indices))
        }

        Memory.clearCache()
        let growth = GPU.activeMemory - activeBefore
        // The gate+up projections are ~2/3 of the module's parameter bytes;
        // a retained fused copy would grow active memory by that amount.
        // Anything transient the forwards allocate is released with the
        // outputs, so flat-to-noise is the pass bar.
        let fusedCopyBytes = weightBytes * 2 / 3
        #expect(
            growth < fusedCopyBytes / 4,
            Comment(
                rawValue: "8 decode-shaped forwards retained \(growth) bytes of module state "
                    + "(a fused gate+up copy would be ~\(fusedCopyBytes) bytes) — "
                    + "SwitchGLU is caching weights again"))

        let paramsAfter = Dictionary(
            uniqueKeysWithValues: glu.parameters().flattened())
        #expect(paramsAfter.count == paramsBefore.count)
        for (key, before) in paramsBefore {
            #expect(
                paramsAfter[key] === before,
                Comment(rawValue: "parameter \(key) was replaced during forwards"))
        }
    }

    private func values(_ count: Int, offset: Int = 0) -> MLXArray {
        MLXArray((offset ..< offset + count).map { Float($0) / 1000 })
    }
}
