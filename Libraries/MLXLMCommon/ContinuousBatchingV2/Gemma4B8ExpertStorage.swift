// Copyright © 2026 Eigen Labs.
import Cmlx
import MLX

/// Model-owned paired-16 repack. Original parameters remain unchanged.
/// This adds 272.25 MiB per admitted layer; it is not zero-copy storage.
final class Gemma4B8ExpertStorage {
    let gateUp: [MLXArray]
    let down: [MLXArray]
    private let snapshots: [MLXArray]
    private let identities: [UInt]
    private let stream: StreamOrDevice

    init?(_ parameters: [MLXArray]) {
        let shapes = [[128, 704, 352], [128, 704, 44], [128, 704, 44],
                      [128, 704, 352], [128, 704, 44], [128, 704, 44],
                      [128, 2816, 88], [128, 2816, 11], [128, 2816, 11]]
        let types: [DType] = [.uint32, .bfloat16, .bfloat16, .uint32, .bfloat16,
                             .bfloat16, .uint32, .bfloat16, .bfloat16]
        guard StreamOrDevice.default == .gpu, parameters.count == shapes.count,
            zip(parameters, zip(shapes, types)).allSatisfy({ $0.shape == $1.0 && $0.dtype == $1.1 })
        else { return nil }
        let snapshots = parameters.compactMap(Self.snapshot)
        let identities = snapshots.compactMap(Self.identity)
        guard snapshots.count == parameters.count, identities.count == parameters.count else { return nil }
        let stream = StreamOrDevice.default
        self.stream = stream
        self.snapshots = snapshots
        self.identities = identities
        gateUp = (0..<3).map { index in
            let tail = index == 0 ? 352 : 44
            let gate = snapshots[index].reshaped(128, 44, 16, tail)
            let up = snapshots[index + 3].reshaped(128, 44, 16, tail)
            return stacked([gate, up], axis: 2, stream: stream).reshaped(128, 1408, tail)
        }
        down = Array(snapshots[6..<9])
    }

    private static func snapshot(_ x: MLXArray) -> MLXArray? {
        var context = mlx_array_new()
        guard mlx_array_set(&context, x.ctx) == 0 else { mlx_array_free(context); return nil }
        return MLXArray(context)
    }

    private static func identity(_ x: MLXArray) -> UInt? {
        var identity: UInt = 0
        var allowed = false
        guard _mlx_array_constant_cache_identity(&identity, &allowed, x.ctx) == 0, allowed else { return nil }
        return identity
    }

    func matches(_ parameters: [MLXArray]) -> Bool {
        guard stream == StreamOrDevice.default, parameters.count == identities.count else { return false }
        return zip(parameters, identities).allSatisfy { Self.identity($0) == $1 }
    }
}
