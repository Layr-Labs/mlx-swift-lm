// Native entropy-bound block diffusion. Algorithm reference:
// Transformers c587bc884db2c2e31fc2b8102314656b17aa07b1, generation_diffusion_gemma.py.
// See docs/diffusiongemma/implementation-references.md for provenance and scope.

import Foundation
import MLX

/// Numerical operations shared by the native runner and fixed-noise tests.
/// This sampler accepts/refines a whole canvas; it is not an AR token sampler.
public enum DiffusionGemmaSampling {
    public static func tokenEntropy(_ processedLogits: MLXArray) throws -> MLXArray {
        guard processedLogits.ndim == 3, processedLogits.shape.allSatisfy({ $0 > 0 }),
            processedLogits.dtype.isFloatingPoint
        else { throw DiffusionGemmaSamplingError.invalidShape("logits") }
        let logits = processedLogits.asType(.float32)
        let logProbabilities = logits - logits.logSumExp(axis: -1, keepDims: true)
        // As in torch.distributions.Categorical, zero-probability masked entries
        // contribute zero rather than 0 * -infinity. All-masked rows remain
        // invalid and are rejected by the native request/logit validation gate.
        let finiteLog = maximum(logProbabilities, -Float.greatestFiniteMagnitude)
        return -(exp(logProbabilities) * finiteLog).sum(axis: -1)
    }

    /// Select the largest lowest-entropy prefix whose sum excluding its maximum
    /// is <= the configured bound. Selection is recomputed on every step.
    public static func acceptanceMask(entropy: MLXArray, bound: Float) throws -> MLXArray {
        guard entropy.ndim == 2, entropy.shape.allSatisfy({ $0 > 0 }),
            entropy.dtype.isFloatingPoint, bound.isFinite, bound > 0
        else { throw DiffusionGemmaSamplingError.invalidShape("entropy/bound") }
        let order = argSort(entropy, axis: -1)
        let sorted = takeAlong(entropy, order, axis: -1)
        let accepted = sorted.cumsum(axis: -1) - sorted .<= bound
        return putAlong(MLXArray.zeros(like: accepted), order, values: accepted, axis: -1)
    }
}

/// Request-owned denoising state. The model's shared weights never own this object.
/// A caller may observe intermediate tensors for diagnostics, but may publish only
/// `finalizedCanvas()`. Prefix-cache mutation belongs after that commit boundary.
public final class DiffusionGemmaDenoisingState {
    public let configuration: DiffusionGemmaGenerationConfiguration
    public let vocabularySize: Int
    public let embeddingDType: DType
    public private(set) var currentCanvas: MLXArray
    public private(set) var argmaxCanvas: MLXArray
    public private(set) var selfConditioningLogits: MLXArray?
    public private(set) var finishedRows: MLXArray
    public private(set) var stepsUsed: MLXArray
    public private(set) var remainingSteps: Int
    private var argmaxHistory: [MLXArray] = []

    /// Conservative live payload accounting (aliases may be counted twice),
    /// separate from allocator/workspace and measured process footprint.
    public var retainedPayloadBytes: Int {
        ([currentCanvas, argmaxCanvas, finishedRows, stepsUsed] + argmaxHistory
            + [selfConditioningLogits].compactMap { $0 }).reduce(0) { $0 + $1.nbytes }
    }

    public init(
        initialCanvas: MLXArray,
        vocabularySize: Int,
        embeddingDType: DType,
        configuration: DiffusionGemmaGenerationConfiguration
    ) throws {
        guard initialCanvas.ndim == 2, initialCanvas.shape.allSatisfy({ $0 > 0 }),
            initialCanvas.dtype == .int32 || initialCanvas.dtype == .uint32,
            vocabularySize > 0, vocabularySize <= Int(Int32.max), embeddingDType.isFloatingPoint
        else { throw DiffusionGemmaSamplingError.invalidShape("initial canvas") }
        guard logicalAnd(initialCanvas .>= 0, initialCanvas .< vocabularySize).all().item(Bool.self)
        else {
            throw DiffusionGemmaSamplingError.invalidToken
        }
        self.configuration = configuration
        self.vocabularySize = vocabularySize
        self.embeddingDType = embeddingDType
        self.currentCanvas = initialCanvas
        self.argmaxCanvas = initialCanvas
        self.finishedRows = MLXArray.zeros([initialCanvas.dim(0)], dtype: .bool)
        self.stepsUsed = MLXArray.zeros([initialCanvas.dim(0)], dtype: .int32)
        self.remainingSteps = configuration.maxDenoisingSteps
    }

    /// Sampling/noise callbacks provide a deterministic injection seam. The native
    /// runner supplies request-local RNG; independent fixtures provide fixed draws.
    /// `sample` receives the already temperature-processed logits exactly once.
    @discardableResult
    public func step(
        rawLogits: MLXArray,
        sample: (MLXArray) throws -> MLXArray,
        noise: () throws -> MLXArray
    ) throws -> MLXArray {
        guard remainingSteps > 0 else { throw DiffusionGemmaSamplingError.exhaustedCanvas }
        guard rawLogits.shape == currentCanvas.shape + [vocabularySize],
            rawLogits.dtype.isFloatingPoint
        else { throw DiffusionGemmaSamplingError.invalidShape("decoder logits") }
        let temperature = try configuration.temperature(remainingStep: remainingSteps)
        let processed = rawLogits.asType(.float32) / temperature
        let finite = isFinite(processed)
        let validDistribution = logicalAnd(
            logicalOr(finite, processed .== -Float.infinity).all(),
            finite.any(axis: -1).all())
        guard validDistribution.item(Bool.self) else {
            throw DiffusionGemmaSamplingError.invalidDistribution
        }
        let sampled = try sample(processed)
        let randomCanvas = try noise()
        for (name, value) in [("sampled canvas", sampled), ("noise canvas", randomCanvas)] {
            guard value.shape == currentCanvas.shape, value.dtype == currentCanvas.dtype else {
                throw DiffusionGemmaSamplingError.invalidShape(name)
            }
            guard logicalAnd(value .>= 0, value .< vocabularySize).all().item(Bool.self) else {
                throw DiffusionGemmaSamplingError.invalidToken
            }
        }
        let entropy = try DiffusionGemmaSampling.tokenEntropy(processed)
        let acceptance = try DiffusionGemmaSampling.acceptanceMask(
            entropy: entropy, bound: configuration.sampler.entropyBound)
        let proposed = which(acceptance, sampled, randomCanvas)
        let proposedArgmax = processed.argMax(axis: -1).asType(currentCanvas.dtype)
        let rowMask = finishedRows[0..., .newAxis]
        let nextCanvas = which(rowMask, currentCanvas, proposed)
        let nextArgmax = which(rowMask, argmaxCanvas, proposedArgmax)

        var stable = MLXArray.ones([currentCanvas.dim(0)], dtype: .bool)
        if configuration.stabilityThreshold > 0 {
            if argmaxHistory.count < configuration.stabilityThreshold {
                stable = MLXArray.zeros(like: stable)
            } else {
                for previous in argmaxHistory {
                    stable = logicalAnd(stable, (previous .== nextArgmax).all(axis: -1))
                }
            }
        }
        let confident = entropy.mean(axis: -1) .< configuration.confidenceThreshold
        let completed = logicalOr(finishedRows, logicalAnd(stable, confident))
        // Preserve a row's last self-conditioning exactly after it finishes, even
        // while peers continue. Cast as the official generation contract does.
        let nextConditioning: MLXArray
        if let previous = selfConditioningLogits {
            nextConditioning = which(finishedRows[0..., .newAxis, .newAxis], previous, processed)
                .asType(embeddingDType)
        } else {
            nextConditioning = processed.asType(embeddingDType)
        }

        // Mutate only after every throwable input/callback check has succeeded.
        stepsUsed = stepsUsed + logicalNot(finishedRows).asType(.int32)
        currentCanvas = nextCanvas
        argmaxCanvas = nextArgmax
        selfConditioningLogits = nextConditioning
        remainingSteps -= 1
        finishedRows = remainingSteps == 0 ? MLXArray.ones(like: completed) : completed
        if configuration.stabilityThreshold > 0 {
            argmaxHistory.append(nextArgmax)
            if argmaxHistory.count > configuration.stabilityThreshold {
                argmaxHistory.removeFirst()
            }
        }
        return acceptance
    }

    /// Native request-local sampling. The arbitrary-callback `step` contract is
    /// unchanged; this entrypoint may compile only the known native operations.
    /// `nextKey` is invoked after distribution validation, in the original order.
    @discardableResult
    public func stepNative(
        rawLogits: MLXArray, samplingTemperature: Float,
        nextKey: () -> MLXArray
    ) throws -> MLXArray {
        guard samplingTemperature.isFinite, samplingTemperature >= 0 else {
            throw DiffusionGemmaSamplingError.invalidShape("sampling temperature")
        }
        guard DiffusionGemmaCompiledSampler.eligible(
            logits: rawLogits, canvas: currentCanvas, embeddingDType: embeddingDType,
            configuration: configuration)
        else {
            return try step(
                rawLogits: rawLogits,
                sample: { processed in
                    if samplingTemperature == 0 { return processed.argMax(axis: -1).asType(.int32) }
                    return MLXRandom.categorical(processed / samplingTemperature, key: nextKey()).asType(.int32)
                }, noise: {
                    MLXRandom.randInt(Int32(0)..<Int32(vocabularySize), currentCanvas.shape, key: nextKey())
                })
        }
        guard remainingSteps > 0 else { throw DiffusionGemmaSamplingError.exhaustedCanvas }
        guard rawLogits.shape == currentCanvas.shape + [vocabularySize] else {
            throw DiffusionGemmaSamplingError.invalidShape("decoder logits")
        }
        let temperature = try configuration.temperature(remainingStep: remainingSteps)
        let processed = rawLogits.asType(.float32) / temperature
        return try MLX.withError { errors in
            let finite = isFinite(processed)
            let valid = logicalAnd(logicalOr(finite, processed .== -Float.infinity).all(),
                finite.any(axis: -1).all()).item(Bool.self)
            try errors.check()
            guard valid else { throw DiffusionGemmaSamplingError.invalidDistribution }
            // A greedy draw consumes no key, exactly as the original session.
            let sampleBits = samplingTemperature == 0 ? MLXArray(UInt32(0))
                : DiffusionGemmaCompiledSampler.randomBits(shape: processed.shape, key: nextKey())
            let randomCanvas = MLXRandom.randInt(
                Int32(0)..<Int32(vocabularySize), currentCanvas.shape, key: nextKey())
            try errors.check()
            let samplingLogits = samplingTemperature == 0 ? processed : processed / samplingTemperature
            let result = DiffusionGemmaCompiledSampler.call(
                [processed, sampleBits, randomCanvas, currentCanvas, argmaxCanvas,
                 finishedRows, stepsUsed, selfConditioningLogits ?? processed,
                 argmaxHistory.last ?? argmaxCanvas, samplingLogits,
                 MLXArray(Float(UInt32.max)), MLXArray(Float(1).nextDown),
                 MLXArray(-Float.greatestFiniteMagnitude)],
                greedy: samplingTemperature == 0, hasPrevious: selfConditioningLogits != nil,
                hasHistory: !argmaxHistory.isEmpty, lastStep: remainingSteps == 1)
            try errors.check()
            guard result.count == 10 else {
                throw DiffusionGemmaSamplingError.invalidShape("compiled sampler result")
            }
            for (name, index, validIndex) in [("sampled canvas", 6, 8), ("noise canvas", 7, 9)] {
                guard result[index].shape == currentCanvas.shape, result[index].dtype == currentCanvas.dtype else {
                    throw DiffusionGemmaSamplingError.invalidShape(name)
                }
                let valid = result[validIndex].item(Bool.self)
                try errors.check()
                guard valid else { throw DiffusionGemmaSamplingError.invalidToken }
            }
            // Do not commit a failed device graph or provisional request state.
            eval(result)
            try errors.check()
            currentCanvas = result[0]
            argmaxCanvas = result[1]
            finishedRows = result[2]
            selfConditioningLogits = result[3]
            stepsUsed = result[4]
            remainingSteps -= 1
            argmaxHistory.append(result[1])
            if argmaxHistory.count > 1 { argmaxHistory.removeFirst() }
            return result[5]
        }
    }

    public func finalizedCanvas() throws -> MLXArray {
        guard finishedRows.all().item(Bool.self) else {
            throw DiffusionGemmaSamplingError.canvasNotFinal
        }
        return argmaxCanvas
    }
}

public enum DiffusionGemmaSamplingError: Error, Sendable, Equatable {
    case invalidShape(String)
    case exhaustedCanvas
    case canvasNotFinal
    case invalidDistribution
    case invalidToken
}
