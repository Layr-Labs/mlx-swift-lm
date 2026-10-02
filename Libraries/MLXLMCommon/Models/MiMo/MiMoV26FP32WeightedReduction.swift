// Copyright © 2026 Eigen Labs.
// Isolated, OFF-by-default native MiMo experiment. Not a losslessness claim.
import Foundation
import MLX

public enum MiMoV26FP32WeightedReduction {
    public static let envFlag = "DARKBLOOM_MIMO_FP32_WEIGHTED_REDUCE"
    public static func isEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        let value = environment[envFlag]?.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return value == "1" || value == "true" || value == "on"
    }
}

public enum MiMoV26FP32WeightedReductionRoute: String, Sendable, Equatable {
    case fused, disabled, unsupportedProfile, unsupportedActivation, notPrefill
    case unsupportedShape, unsupportedTopK, unsupportedDType, notSorted, unsupportedStream
}

/// Native-local graph result. Scalar route is selection evidence only, never
/// completion authority, physical residency credit or a numerical qualification.
public struct MiMoV26FP32WeightedReductionResult {
    public let output: MLXArray
    public let route: MiMoV26FP32WeightedReductionRoute
    package init(output: MLXArray, route: MiMoV26FP32WeightedReductionRoute) {
        self.output = output
        self.route = route
    }
}

/// Metadata-only gate. inverseOrder must be the genuine SwitchGLU argsort
/// inverse, not a caller-provided unchecked index buffer. This internal helper
/// does not perform a synchronizing index readback or create another projection.
func mimoV26FP32WeightedUnsortRefusal(
    sortedOutputs: MLXArray, inverseOrder: MLXArray, weights: MLXArray
) -> MiMoV26FP32WeightedReductionRoute? {
    guard sortedOutputs.ndim == 2, weights.ndim == 3, inverseOrder.ndim == 1,
        sortedOutputs.dim(1) > 0, sortedOutputs.dim(1).isMultiple(of: 64),
        weights.dim(0) > 0, weights.dim(1) > 1,
        sortedOutputs.dim(0) == weights.size, inverseOrder.size == weights.size,
        sortedOutputs.size <= Int(Int32.max), weights.size <= Int(Int32.max)
    else {
        return .unsupportedShape
    }
    guard weights.dim(2) == 6 || weights.dim(2) == 8 else { return .unsupportedTopK }
    guard sortedOutputs.dtype == .bfloat16 || sortedOutputs.dtype == .float16,
        weights.dtype == .float32, inverseOrder.dtype == .uint32
    else { return .unsupportedDType }
    guard weights.size >= 64 else { return .notSorted }
    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        // Initial profile deliberately excludes CPU and private/nondefault streams.
        guard StreamOrDevice.default.stream == MLX.Stream.gpu else { return .unsupportedStream }
        return nil
    #else
        return .unsupportedStream
    #endif
}

private let mimoFP32WeightedUnsortKernel = MLXFast.metalKernel(
    name: "mimo_v26_fp32_weighted_unsort",
    inputNames: ["sorted_outputs", "inverse_order", "weights"],
    outputNames: ["output"],
    source: """
        #pragma clang fp contract(off)
        uint feature = thread_position_in_grid.x;
        uint token = thread_position_in_grid.y;
        const uint base = token * (uint)K;
        float accumulator = 0.0f;
        for (uint slot = 0; slot < (uint)K; ++slot) {
            const uint row = inverse_order[base + slot];
            const float product = float(sorted_outputs[row * (uint)D + feature]) * weights[base + slot];
            // Keep both the product and reduction FP32. The extra +0 mirrors
            // the pinned small-column reducer's per-slot initial accumulation.
            const float partial = product + 0.0f;
            accumulator = partial + accumulator;
        }
        output[token * (uint)D + feature] = accumulator;
        """,
    ensureRowContiguous: true
)

/// Called only after the gate and the real sort. FP32 output leaves the original
/// MiMo final cast at its original caller. No BF16 per-product/accumulator cast.
/// This is a candidate: compiler contraction, dispatch/reduction order and final
/// bytes still require exact comparison with the original unfused expression.
func mimoV26FP32WeightedUnsort(
    sortedOutputs: MLXArray, inverseOrder: MLXArray, weights: MLXArray
) -> MLXArray {
    precondition(
        mimoV26FP32WeightedUnsortRefusal(
            sortedOutputs: sortedOutputs, inverseOrder: inverseOrder, weights: weights) == nil)
    let hidden = sortedOutputs.dim(1)
    let topK = weights.dim(2)
    return mimoFP32WeightedUnsortKernel(
        [sortedOutputs, inverseOrder, weights], template: [("K", topK), ("D", hidden)],
        grid: (hidden, weights.size / topK, 1), threadGroup: (64, 1, 1),
        outputShapes: [[weights.dim(0), weights.dim(1), hidden]], outputDTypes: [.float32],
        stream: .gpu
    )[0]
}
