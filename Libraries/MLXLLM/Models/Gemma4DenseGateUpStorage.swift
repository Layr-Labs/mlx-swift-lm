// Copyright © 2026 Eigen Labs.
// Adapted from David Tai's Gemma MLXFast port, SDK 0cb4c68d, with descriptor-
// owned invalidation and a disabled-until-qualified candidate policy.
import Cmlx
import MLX
import MLXNN

/// Model-owned concatenated affine rows. Split parameters remain contiguous
/// views of precisely the same weight/scale/bias values. No checkpoint file or
/// quantization policy changes, and no process-global memoization.
final class Gemma4DenseGateUpStorage {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let splitParameters: [MLXArray]
    private let sourceSnapshots: [MLXArray]
    private let sourceIdentities: [UInt]
    private let stream: StreamOrDevice

    init?(_ gateWeight: MLXArray, _ gateScales: MLXArray, _ gateBiases: MLXArray,
          _ upWeight: MLXArray, _ upScales: MLXArray, _ upBiases: MLXArray) {
        let inputs = [gateWeight, gateScales, gateBiases, upWeight, upScales, upBiases]
        let shapes = [[2112, 704], [2112, 44], [2112, 44],
                      [2112, 704], [2112, 44], [2112, 44]]
        let dtypes: [DType] = [.uint32, .bfloat16, .bfloat16, .uint32, .bfloat16, .bfloat16]
        guard zip(inputs, zip(shapes, dtypes)).allSatisfy({ array, expected in
            array.shape == expected.0 && array.dtype == expected.1
        }) else { return nil }

        let stream = StreamOrDevice.default
        let weight = concatenated([gateWeight, upWeight], axis: 0, stream: stream)
        let scales = concatenated([gateScales, upScales], axis: 0, stream: stream)
        let biases = concatenated([gateBiases, upBiases], axis: 0, stream: stream)
        let split = [weight[0..<2112], scales[0..<2112], biases[0..<2112],
                     weight[2112..<4224], scales[2112..<4224], biases[2112..<4224]]
        // A Module update can replace a descriptor without replacing its Swift
        // wrapper. Pin independent C contexts, not ObjectIdentifier/shape alone.
        let snapshots = split.compactMap(Self.snapshot)
        guard snapshots.count == split.count else { return nil }
        let identities = snapshots.compactMap(Self.constantIdentity)
        guard identities.count == snapshots.count else { return nil }
        self.stream = stream
        self.weight = weight
        self.scales = scales
        self.biases = biases
        splitParameters = split
        sourceSnapshots = snapshots
        sourceIdentities = identities
    }

    private static func snapshot(_ array: MLXArray) -> MLXArray? {
        var context = mlx_array_new()
        guard mlx_array_set(&context, array.ctx) == 0 else {
            mlx_array_free(context)
            return nil
        }
        return MLXArray(context)
    }

    private static func constantIdentity(_ array: MLXArray) -> UInt? {
        var identity: UInt = 0
        var allowed = false
        guard _mlx_array_constant_cache_identity(&identity, &allowed, array.ctx) == 0,
              allowed else { return nil }
        return identity
    }

    /// False during tracing or after any parameter descriptor changes. The
    /// retained snapshots keep old descriptors alive and prevent address reuse.
    func matches(gate: QuantizedLinear, up: QuantizedLinear) -> Bool {
        guard let gateBias = gate.biases, let upBias = up.biases else { return false }
        let live = [gate.weight, gate.scales, gateBias, up.weight, up.scales, upBias]
        for index in live.indices {
            guard Self.constantIdentity(live[index]) == sourceIdentities[index] else { return false }
        }
        return true
    }

    func project(_ x: MLXArray) -> MLXArray? {
        guard StreamOrDevice.default == stream else { return nil }
        return quantizedMM(x, weight, scales: scales, biases: biases,
                           transpose: true, groupSize: 64, bits: 8, mode: .affine,
                           stream: stream)
    }
}
