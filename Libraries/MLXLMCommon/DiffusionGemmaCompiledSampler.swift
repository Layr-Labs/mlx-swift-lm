// Native DiffusionGemma sampling operations, with unchanged arithmetic and RNG.
import Cmlx
import Foundation
import MLX

enum DiffusionGemmaCompiledSampler {
    // Private qualification candidate. Unset/0 retains the original sampler.
    static let enabled = ProcessInfo.processInfo.environment[
        "DARKBLOOM_DIFFUSION_COMPILED_SAMPLER"] == "1"

    static func eligible(
        logits: MLXArray, canvas: MLXArray, embeddingDType: DType,
        configuration: DiffusionGemmaGenerationConfiguration
    ) -> Bool {
        guard enabled, Device.defaultDevice().deviceType == .gpu,
            StreamOrDevice.default.stream == Stream.gpu,
            logits.shape == [1, 256, 262144], logits.dtype == .float32,
            canvas.shape == [1, 256], canvas.dtype == .int32, embeddingDType == .float32,
            configuration.stabilityThreshold == 1,
            configuration.sampler.entropyBound == Float(0.1),
            configuration.confidenceThreshold == Float(0.005)
        else { return false }
        return [logits, canvas].allSatisfy { array in
            var identity: UInt = 0
            var ordinary = false
            return _mlx_array_constant_cache_identity(&identity, &ordinary, array.ctx) == 0
                && ordinary
        }
    }

    // Bounded immutable functions capture flags only. All request arrays and
    // randomness are explicit inputs. MLX owns compilation/evaluation locking;
    // never hold a separate lock while entering one of these functions.
    private static let functions: [@Sendable ([MLXArray]) -> [MLXArray]] =
        (0..<16).map { flags in
            compile { arrays in
                graph(arrays, greedy: flags & 1 != 0, hasPrevious: flags & 2 != 0,
                    hasHistory: flags & 4 != 0, lastStep: flags & 8 != 0)
            }
        }

    static func call(
        _ arrays: [MLXArray], greedy: Bool, hasPrevious: Bool, hasHistory: Bool,
        lastStep: Bool
    ) -> [MLXArray] {
        let flags = (greedy ? 1 : 0) | (hasPrevious ? 2 : 0)
            | (hasHistory ? 4 : 0) | (lastStep ? 8 : 0)
        let result = functions[flags](arrays)
        if result.count == 10 { DiffusionGemmaCompiledSamplerDiagnostics.recordDispatch() }
        return result
    }

    // Use the same integer primitive as native MLX uniform sampling. Keep its
    // floating constants as graph inputs: the pinned literal printer rounds them.
    static func randomBits(shape: [Int], key: MLXArray) -> MLXArray {
        var result = mlx_array_new()
        let dimensions = shape.map(Int32.init)
        dimensions.withUnsafeBufferPointer { pointer in
            _ = mlx_random_bits(
                &result, pointer.baseAddress, pointer.count, 4, key.ctx,
                StreamOrDevice.default.ctx)
        }
        return MLXArray(result)
    }

    // Inputs: processed logits, random bits, native noise, canvas, argmax,
    // finished, counts, conditioning, history, scaled logits, denominator,
    // exclusive upper bound and the exact finite-log clamp.
    private static func graph(
        _ a: [MLXArray], greedy: Bool, hasPrevious: Bool, hasHistory: Bool,
        lastStep: Bool
    ) -> [MLXArray] {
        let processed = a[0], finished = a[5]
        let sampled: MLXArray
        if greedy {
            sampled = processed.argMax(axis: -1).asType(.int32)
        } else {
            // Same promotion/divide/clamp/affine/Gumbel construction as
            // mlx/random.cpp, with no lossy printed RNG constants.
            let uniform = Float(1) * minimum(a[1] / a[10], a[11]).asType(.float32) + Float(0)
            let gumbel = -log(-log(uniform))
            sampled = (gumbel + a[9]).argMax(axis: -1).asType(.int32)
        }
        let logProbabilities = processed - processed.logSumExp(axis: -1, keepDims: true)
        let finiteLog = maximum(logProbabilities, a[12])
        let entropy = -(exp(logProbabilities) * finiteLog).sum(axis: -1)
        let order = argSort(entropy, axis: -1)
        let sorted = takeAlong(entropy, order, axis: -1)
        let accepted = sorted.cumsum(axis: -1) - sorted .<= Float(0.1)
        let acceptance = putAlong(MLXArray.zeros(like: accepted), order, values: accepted, axis: -1)
        let proposed = which(acceptance, sampled, a[2])
        let argmax = processed.argMax(axis: -1).asType(.int32)
        let nextCanvas = which(finished[0..., .newAxis], a[3], proposed)
        let nextArgmax = which(finished[0..., .newAxis], a[4], argmax)
        let stable = hasHistory ? (a[8] .== nextArgmax).all(axis: -1)
            : MLXArray.zeros([processed.dim(0)], dtype: .bool)
        let completed = logicalOr(finished, logicalAnd(stable, entropy.mean(axis: -1) .< Float(0.005)))
        let conditioning = hasPrevious
            ? which(finished[0..., .newAxis, .newAxis], a[7], processed).asType(.float32)
            : processed.asType(.float32)
        let steps = a[6] + logicalNot(finished).asType(.int32)
        let nextFinished = lastStep ? MLXArray.ones(like: completed) : completed
        let validSample = logicalAnd(sampled .>= 0, sampled .< processed.dim(-1)).all()
        let validNoise = logicalAnd(a[2] .>= 0, a[2] .< processed.dim(-1)).all()
        return [nextCanvas, nextArgmax, nextFinished, conditioning, steps, acceptance,
            sampled, a[2], validSample, validNoise]
    }
}

/// Explicitly armed diagnostics only; ordinary serving does not collect counts.
@_spi(DiffusionGemmaDiagnostics)
public enum DiffusionGemmaCompiledSamplerDiagnostics {
    public struct Snapshot: Sendable, Equatable {
        public let armed: Bool
        public let calls: Int
    }
    private final class State: @unchecked Sendable {
        let lock = NSLock()
        var armed = false
        var calls = 0
    }
    private static let state = State()
    public static func clearAndArm() {
        state.lock.withLock { state.calls = 0; state.armed = true }
    }
    public static func snapshot() -> Snapshot {
        state.lock.withLock { Snapshot(armed: state.armed, calls: state.calls) }
    }
    public static func snapshotAndDisarm() -> Int {
        state.lock.withLock { state.armed = false; return state.calls }
    }
    static func recordDispatch() {
        state.lock.withLock { if state.armed { state.calls += 1 } }
    }
}
