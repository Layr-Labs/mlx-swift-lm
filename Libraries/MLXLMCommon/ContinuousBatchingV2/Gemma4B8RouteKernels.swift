// Copyright © 2026 Eigen Labs.
import MLX
import MLXFast

enum Gemma4B8RouteKernels {
    private static let rankRaw = makeRank(prefix: false)
    private static let rankPrefix = makeRank(prefix: true)
    private static let foldNativeRaw = makeFold(native: true, prefix: false)
    private static let foldNativePrefix = makeFold(native: true, prefix: true)
    private static let foldBitonicRaw = makeFold(native: false, prefix: false)
    private static let foldBitonicPrefix = makeFold(native: false, prefix: true)

    private static func makeRank(prefix: Bool) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(name: "db_gemma4_b8_route_rank_prefix\(prefix ? 1 : 0)_v1",
            inputNames: ["indices"], outputNames: ["row_order", "sorted_keys", "inverse_order"],
            source: prefix ? Gemma4B8RouteSources.prefix : Gemma4B8RouteSources.raw,
            ensureRowContiguous: true)
    }

    private static func makeFold(native: Bool, prefix: Bool) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(name: "db_gemma4_b8_route_fold_native\(native ? 1 : 0)_prefix\(prefix ? 1 : 0)_stock_softmax_v1",
            inputNames: ["scores", "pes"], outputNames: ["indices", "weights", "row_order", "sorted_keys", "inverse_order"],
            source: Gemma4B8RouteFoldSources.make(native: native, prefix: prefix),
            header: Gemma4RouterFinalistsSources.weightsHeader(orderKeysEnabled: native),
            ensureRowContiguous: true)
    }

    static func rank(_ indices: MLXArray, policy: Gemma4B8RoutePolicy, stream: StreamOrDevice) -> [MLXArray] {
        let input = policy.directInput ? indices : indices.flattened()
        return (policy.prefixBounds ? rankPrefix : rankRaw)([input], grid: (64, 1, 1),
            threadGroup: (64, 1, 1), outputShapes: [[64], [64], [64]],
            outputDTypes: [.uint32, .uint32, .uint32], stream: stream)
    }

    static func fold(scores: MLXArray, scale: MLXArray, policy: Gemma4B8RoutePolicy,
                     stream: StreamOrDevice) -> [MLXArray] {
        let kernel = policy.nativeOrderKeys
            ? (policy.prefixBounds ? foldNativePrefix : foldNativeRaw)
            : (policy.prefixBounds ? foldBitonicPrefix : foldBitonicRaw)
        let threads = policy.nativeOrderKeys ? 256 : 1024
        return kernel([scores, scale], template: [("T", DType.bfloat16)], grid: (threads, 1, 1),
            threadGroup: (threads, 1, 1), outputShapes: [[8, 1, 8], [8, 1, 8], [64], [64], [64]],
            outputDTypes: [.uint32, .bfloat16, .uint32, .uint32, .uint32], stream: stream)
    }
}
