import Foundation
import MLX
import MLXLLM
import MLXLMCommon

public struct DiffusionGemmaGenerationResult: Sendable {
    /// Actual output only: excludes EOS and unused canvas slots.
    public let tokenIds: [Int32]
    /// Native accounting includes a terminal EOS when one was generated.
    public let generatedTokenCount: Int
    public let finishReason: String
    public let promptTokenCount: Int
    public let reusedPromptTokenCount: Int
    public let committedCanvasCount: Int
    public let denoisingSteps: Int
    public let workTokenCount: Int
    public let prefillSeconds: Double
    public let firstCommittedOutputSeconds: Double?
    public internal(set) var totalSeconds: Double
}

extension DiffusionGemma {
    /// Native single-request runner. State, noise and cache are local to the
    /// invocation; the weights are shared and unchanged. Continuous batching is
    /// a separate scheduler integration, not a claim made by this convenience API.
    /// Only finalized visible tokens reach onCommittedBlock, never draft canvases.
    public func generateNative(
        promptTokenIds: MLXArray,
        generation: DiffusionGemmaGenerationConfiguration,
        seed: UInt64,
        samplingTemperature: Float = 1,
        prefillChunkSize: Int? = nil,
        pagedBackend: PagedKVBackend? = nil,
        pixelValues: MLXArray? = nil,
        visualOutputLengths: [Int]? = nil,
        visualBlockIds: MLXArray? = nil,
        multimodal: CBv2MultimodalInput? = nil,
        prefixCheckpoint: DiffusionGemmaPrefixCheckpoint? = nil,
        prefixIdentity: DiffusionGemmaPrefixIdentity? = nil,
        onPromptCheckpoint: ((DiffusionGemmaPrefixCheckpoint) throws -> Void)? = nil,
        isCancelled: @escaping () -> Bool = { Task.isCancelled },
        onCommittedBlock: ([Int32]) throws -> Void = { _ in }
    ) throws -> DiffusionGemmaGenerationResult {
        let started = ProcessInfo.processInfo.systemUptime
        let session = try DiffusionGemmaGenerationSession(
            model: self, promptTokenIds: promptTokenIds, generation: generation, seed: seed,
            samplingTemperature: samplingTemperature, prefillChunkSize: prefillChunkSize,
            pagedBackend: pagedBackend,
            pixelValues: pixelValues, visualOutputLengths: visualOutputLengths,
            visualBlockIds: visualBlockIds, multimodal: multimodal, prefixCheckpoint: prefixCheckpoint,
            prefixIdentity: prefixIdentity, onPromptCheckpoint: onPromptCheckpoint,
            isCancelled: isCancelled)
        while session.phase != .finished {
            if case .committed(let tokens, _) = try session.advance(), !tokens.isEmpty {
                try onCommittedBlock(tokens)
            }
        }
        guard var result = session.result else {
            throw DiffusionGemmaModelError.invalidInput("missing terminal generation result")
        }
        // The synchronous convenience call includes consumer callback time;
        // the session's intrinsic timing alone must not inflate this rate.
        result.totalSeconds = ProcessInfo.processInfo.systemUptime - started
        return result
    }
}
