import MLX

@testable import MLXLMCommon

class CompleteCheckpointFixtureModel:
    CBv2RecurrentSteppableModel, CBv2CompleteCheckpointKVTypeProviding
{
    var cbv2Capabilities: CBv2ModelCapabilities {
        var result = CBv2ModelCapabilities.initialRecurrentTarget
        result.supportsRecurrentCheckpointReuse = true
        return result
    }
    let cbv2CompleteCheckpointKVDTypes: [DType]? = [.float32]
    private let spec: CBv2RecurrentStateSpec = .init(layers: [
        .init(
            modelLayerIndex: 0, convShape: [1, 1, 1], convDType: .float32,
            ssmShape: [1, 1, 1, 1], ssmDType: .float32)
    ])
    private(set) var recurrentSpecReads = 0
    var recurrentStateSpec: CBv2RecurrentStateSpec? {
        recurrentSpecReads += 1
        return spec
    }

    func forward(tokens: MLXArray, caches: [CBv2AttendingLayerCache]) -> MLXArray {
        preconditionFailure("explicit recurrent state is required")
    }
    /// One recurrent row per batch row: each row's state is the running sum
    /// of its own tokens, so a restored checkpoint reproduces the exact
    /// greedy continuation and company rows cannot perturb a donor.
    func forward(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        let rows = tokens.dim(0)
        let length = tokens.dim(1)
        precondition(recurrentState.count == rows, "one recurrent evaluation per batch row")
        let qkv = MLXArray.zeros([rows, 1, length, 1])
        for cache in caches {
            _ = cache.updateAndAttend(queries: qkv, keys: qkv, values: qkv, scale: 1, sinks: nil)
        }
        var logitsRows: [MLXArray] = []
        for (row, evaluation) in recurrentState.enumerated() {
            let previous =
                evaluation.inputState(modelLayerIndex: 0)?.ssm
                ?? MLXArray.zeros([1, 1, 1, 1])
            let value = previous.reshaped([]) + sum(tokens[row].asType(.float32))
            try! evaluation.stage(
                modelLayerIndex: 0, conv: value.reshaped([1, 1, 1]),
                ssm: value.reshaped([1, 1, 1, 1]))
            let target = value.asType(.int32) % 16
            let logits = MLX.where(MLXArray(Int32(0) ..< Int32(16)) .== target, 10, -10)
            logitsRows.append(broadcast(logits.reshaped([1, 1, 16]), to: [1, length, 16]))
        }
        return concatenated(logitsRows, axis: 0)
    }
}

/// The same fixture claiming rectangular packed prefill, as Qwen3.5 does.
/// A recurrent row prefills through `targetForward` whether packed or solo.
final class PackableCompleteCheckpointFixtureModel: CompleteCheckpointFixtureModel,
    CBv2PackedPrefillSteppableModel
{
    let supportsPackedPrefill = true
    override var cbv2Capabilities: CBv2ModelCapabilities {
        var result = super.cbv2Capabilities
        result.supportsPackedPrefill = true
        return result
    }

    func prefill(
        tokens: MLXArray, inputEmbeddings: MLXArray?,
        caches: [CBv2AttendingLayerCache], requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        preconditionFailure("recurrent rows prefill through targetForward, packed or solo")
    }
}
