// Copyright © 2026 Eigen Labs.
import MLX
import MLXFast

public enum Gemma4QKVNormV1 {
    public struct RopeParameters {
        public let log2Base: Float?
        public let frequencies: MLXArray?
        public init(log2Base: Float) { self.log2Base = log2Base; frequencies = nil }
        public init(frequencies: MLXArray) { self.frequencies = frequencies; log2Base = nil }
    }
    public struct Output {
        public let q: MLXArray
        public let k: MLXArray
        public let v: MLXArray
        public let appliedRope: Bool
    }
    // Stock RoPE's shader tape differs from the current custom-kernel safe
    // math defaults. Keep norm-only kernels unchanged; only the separately
    // opted-in fused RoPE variant selects the matching relaxed compiler mode.
    static let relaxedRopeMathAvailable: Bool = {
        if #available(macOS 15, iOS 18, tvOS 18, visionOS 2, *) { return true }
        return false
    }()
    private static func make(_ name: String, inputs: [String], source: String,
                             mathMode: MLXFast.KernelMathMode = .safe) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(name: "db_gemma4_qkv_\(name)_v1", inputNames: inputs,
            outputNames: ["q_out", "k_out", "v_out"], source: source, ensureRowContiguous: true,
            mathMode: mathMode)
    }
    private static let decodeVector = make("decode_vec4", inputs: ["q", "k", "v", "q_weight", "k_weight", "position_offsets", "rope_log2_base", "rope_freqs"], source: Gemma4QKVNormSources.decodeVector)
    private static let decodeScalar = make("decode_scalar", inputs: ["q", "k", "v", "q_weight", "k_weight", "position_offsets", "rope_log2_base", "rope_freqs"], source: Gemma4QKVNormSources.decodeScalar)
    private static let full = make("prefill_full_masked", inputs: ["q", "k", "q_weight", "k_weight", "position_offsets", "rope_freqs"], source: Gemma4QKVNormSources.fullPrefill)
    private static let sliding = make("prefill_sliding_masked", inputs: ["q", "k", "v", "q_weight", "k_weight", "position_offsets", "rope_log2_base"], source: Gemma4QKVNormSources.slidingPrefill)
    private static let decodeVectorRope = make("decode_vec4_rope_relaxed", inputs: ["q", "k", "v", "q_weight", "k_weight", "position_offsets", "rope_log2_base", "rope_freqs"], source: Gemma4QKVNormSources.decodeVector, mathMode: .relaxed)
    private static let decodeScalarRope = make("decode_scalar_rope_relaxed", inputs: ["q", "k", "v", "q_weight", "k_weight", "position_offsets", "rope_log2_base", "rope_freqs"], source: Gemma4QKVNormSources.decodeScalar, mathMode: .relaxed)
    private static let fullRope = make("prefill_full_masked_rope_relaxed", inputs: ["q", "k", "q_weight", "k_weight", "position_offsets", "rope_freqs"], source: Gemma4QKVNormSources.fullPrefill, mathMode: .relaxed)
    private static let slidingRope = make("prefill_sliding_masked_rope_relaxed", inputs: ["q", "k", "v", "q_weight", "k_weight", "position_offsets", "rope_log2_base"], source: Gemma4QKVNormSources.slidingPrefill, mathMode: .relaxed)

    public static func apply(q: MLXArray, k: MLXArray, v: MLXArray, qWeight: MLXArray, kWeight: MLXArray,
        eps: Float, keyValueShared: Bool, positionOffsets: MLXArray, rope: RopeParameters?,
        equalQueryKeyOffsets: Bool, targetEligible: Bool, scheduledPrefill: Bool,
        policy: Gemma4QKVNormPolicy, stream: StreamOrDevice = .default) -> Output? {
        guard !keyValueShared || v === k,
            let plan = policy.plan(targetEligible: targetEligible, q: q.shape, k: k.shape, v: v.shape,
                bf16: q.dtype == .bfloat16 && k.dtype == .bfloat16 && v.dtype == .bfloat16,
                weightsMatch: q.ndim == 4 && qWeight.shape == [q.dim(3)] && kWeight.shape == qWeight.shape
                    && qWeight.dtype == .bfloat16 && kWeight.dtype == .bfloat16,
                eps: eps, keyValueShared: keyValueShared, scheduledPrefill: scheduledPrefill),
            positionOffsets.dtype == .int32, positionOffsets.shape == [plan.batch],
            Gemma4PrefillGlueV1.gpuStream(stream) else { return nil }
        let frequencies = rope?.frequencies
        let hasFrequencies = frequencies?.dtype == .float32 && frequencies?.size == plan.dimension / 2
        let hasBase = rope?.log2Base?.isFinite == true
        let supportedRope = plan.kind == .fullPrefill ? hasFrequencies
            : (plan.kind == .slidingPrefill ? hasBase : hasFrequencies || hasBase)
        let applyRope = policy.rope && equalQueryKeyOffsets && supportedRope && relaxedRopeMathAvailable
        let log2Base = MLXArray([rope?.log2Base ?? 0])
        let frequencyInput = frequencies ?? MLXArray([Float.infinity])
        if plan.kind == .decode {
            let vector = Gemma4PrefillGlueV1.alignedActivations([q, k, v, qWeight, kWeight])
            let kernel = vector ? (applyRope ? decodeVectorRope : decodeVector)
                : (applyRope ? decodeScalarRope : decodeScalar)
            let values = kernel([q, k, v, qWeight, kWeight, positionOffsets, log2Base, frequencyInput],
                template: [("T", q.dtype), ("D", plan.dimension), ("Q_ROWS", plan.queryRows), ("K_ROWS", plan.keyRows),
                    ("Q_HEADS", 16), ("K_HEADS", plan.keyHeads), ("KEY_VALUE_SHARED", keyValueShared),
                    ("APPLY_ROPE", applyRope), ("USE_FREQS", hasFrequencies)],
                grid: (plan.grid, 1, 1), threadGroup: (plan.threads, 1, 1),
                outputShapes: applyRope ? [[plan.batch, 16, 1, plan.dimension], [plan.batch, plan.keyHeads, 1, plan.dimension], v.shape] : [q.shape, k.shape, v.shape],
                outputDTypes: [.bfloat16, .bfloat16, .bfloat16], stream: stream)
            return Output(q: applyRope ? values[0] : values[0].transposed(0, 2, 1, 3),
                k: applyRope ? values[1] : values[1].transposed(0, 2, 1, 3),
                v: values[2].transposed(0, 2, 1, 3), appliedRope: applyRope)
        }
        var template: [(String, any KernelTemplateArg)] = [("T", q.dtype), ("D", plan.dimension),
            ("Q_ROWS", plan.queryRows), ("TOTAL_ROWS", plan.totalRows), ("RPT", plan.rowsPerGroup),
            ("LQ", plan.queryLength), ("HQ", 16), ("LK", plan.keyLength), ("HK", plan.keyHeads), ("APPLY_ROPE", applyRope)]
        let inputs: [MLXArray]
        let kernel: MLXFast.MLXFastKernel
        if plan.kind == .fullPrefill {
            inputs = [q, k, qWeight, kWeight, positionOffsets, frequencyInput]
            kernel = applyRope ? fullRope : full
        } else {
            inputs = [q, k, v, qWeight, kWeight, positionOffsets, log2Base]
            template.append(("K_ROWS", plan.keyRows))
            kernel = applyRope ? slidingRope : sliding
        }
        let values = kernel(inputs, template: template, grid: (plan.grid, 1, 1), threadGroup: (plan.threads, 1, 1),
            outputShapes: [[plan.batch, 16, plan.queryLength, plan.dimension],
                [plan.batch, plan.keyHeads, plan.keyLength, plan.dimension], [plan.batch, plan.keyHeads, plan.keyLength, plan.dimension]],
            outputDTypes: [.bfloat16, .bfloat16, .bfloat16], stream: stream)
        return Output(q: values[0], k: values[1], v: values[2], appliedRope: applyRope)
    }
}
