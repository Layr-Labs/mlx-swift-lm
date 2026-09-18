// Copyright © 2026 Eigen Labs.
import MLX
import MLXFast

/// Gemma-only scheduled-prefill GeGLU candidate. No KV mirrors, global tensor
/// registry, eager cross-check or changed compiled-activation policy.
public enum Gemma4PromptGlueV1 {
    private static let vector = make(vectorized: true)
    private static let scalar = make(vectorized: false)

    private static func make(vectorized: Bool) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(name: "db_gemma4_prompt_geglu_v1_\(vectorized ? "vec4" : "scalar")",
            inputNames: ["gate", "up"], outputNames: ["out"],
            source: vectorized ? Gemma4PromptGlueSources.vector : Gemma4PromptGlueSources.scalar,
            ensureRowContiguous: true)
    }

    public static func geluProduct(gate: MLXArray, up: MLXArray,
        context: Gemma4PrefillGluePolicy.Context, compiledBaseline: Bool,
        stream: StreamOrDevice = .default) -> MLXArray? {
        guard gate.shape == up.shape, up.dtype == .bfloat16,
            let plan = context.gegluPlan(shape: gate.shape, inputBF16: gate.dtype == .bfloat16,
                compiledBaseline: compiledBaseline),
            Gemma4PrefillGlueV1.gpuStream(stream) else { return nil }
        return dispatch(gate: gate, up: up, plan: plan, stream: stream)
    }

    /// Only a producer with a real gate|up layout should call this entry point.
    /// The caller retains the original split views for fallback behavior.
    public static func geluProductFusedPlane(_ plane: MLXArray, hidden: Int,
        context: Gemma4PrefillGluePolicy.Context, compiledBaseline: Bool,
        stream: StreamOrDevice = .default) -> MLXArray? {
        guard let plan = context.gegluPlan(shape: plane.shape, inputBF16: plane.dtype == .bfloat16,
            compiledBaseline: compiledBaseline, fusedHidden: hidden),
            Gemma4PrefillGlueV1.gpuStream(stream) else { return nil }
        return dispatch(gate: plane, up: plane, plan: plan, stream: stream)
    }

    private static func dispatch(gate: MLXArray, up: MLXArray,
        plan: Gemma4PrefillGluePolicy.Context.GeGLUPlan, stream: StreamOrDevice) -> MLXArray {
        let kernel = Gemma4PrefillGlueV1.alignedActivations([gate, up]) ? vector : scalar
        return kernel([gate, up], template: [("T", DType.bfloat16), ("N", plan.columns),
            ("ROWS", plan.rows), ("PITCH", plan.pitch), ("UP_OFF", plan.upOffset)],
            grid: (plan.threads, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [plan.outputShape], outputDTypes: [.bfloat16], stream: stream)[0]
    }
}
