import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom

/// Super's opt-in Mamba prefill: the projection's convolution columns go
/// through one fused kernel (`convolve(projected:columnOffset:channels:state:weight:bias:)`:
/// the conv-state concatenation, depthwise conv1d, bias, `silu` and the
/// retained conv state, reading the in_proj output in place), the SSD runs as
/// ``NemotronSuperSSDChunked`` on the convolution output and the projected
/// `dt` in place (no split copies, `computeDt` inside), and the gated norm runs
/// as the fused `NemotronSuperMambaStep.gatedNorm` over every row. The
/// convolution reproduces the op chain's roundings (the one-token step
/// kernel's expressions); the SSD does not (see ``NemotronSuperSSDChunked``).
public enum NemotronSuperMambaPrefill {
    /// Routes Super graph prefill chunks of at least ``minimumTokens`` tokens
    /// through the chunked path; on by default on NAX GPUs (elsewhere only
    /// with ``allowsGenericLayout``). Not bit-identical: on an M5 Ultra it
    /// takes 12K-token prefill from 2,471 to 2,881 tokens/s with teacher-forced
    /// top-1 accuracy 78.09% and perplexity 2.7275 over 46,070 positions
    /// (grouped GEMM baseline 78.12% / 2.7294, NAX experts 78.16% / 2.7238).
    nonisolated(unsafe) public static var enabled =
        NemotronSuperSSDChunked.naxAvailable && ModelOptNVFP4NAXExperts.runtimeCompilerSupported

    /// Lets the path run on GPUs without Neural Accelerators with the slow
    /// `.generic` fragment layout (for checks; never faster there).
    nonisolated(unsafe) public static var allowsGenericLayout = false

    /// Shorter chunks (decode steps, verify windows) keep the existing path.
    public static let minimumTokens = 32

    static var active: Bool {
        enabled && ModelOptNVFP4NAXExperts.runtimeCompilerSupported
            && (NemotronSuperSSDChunked.naxAvailable || allowsGenericLayout)
    }

    // MARK: - Convolution

    private static let convolveKernel = MLXFast.metalKernel(
        name: "nemotron_super_mamba_conv_prefill",
        inputNames: ["projected", "state", "weight", "bias", "token_count"],
        outputNames: ["out", "next_state"],
        source: """
            const uint c = thread_position_in_grid.x;
            const int t0 = int(thread_position_in_grid.y) * TBK;
            const uint b = thread_position_in_grid.z;
            const int tokens = token_count;
            if (c >= C || t0 >= tokens) {
                return;
            }
            const device T* in = projected + size_t(b) * tokens * XS + XO + c;
            const device T* rows = state + size_t(b) * (KS - 1) * C + c;
            float w[KS];
            for (int k = 0; k < KS; ++k) {
                w[k] = static_cast<float>(weight[size_t(c) * KS + k]);
            }
            const T bias_c = bias[c];
            // Padded row i is state row i (i < KS - 1), else input row i - (KS - 1).
            T window[KS - 1];
            for (int k = 0; k < KS - 1; ++k) {
                const int t = t0 - (KS - 1) + k;
                window[k] = t < 0 ? rows[size_t(t + KS - 1) * C] : in[size_t(t) * XS];
            }
            device T* o = out + size_t(b) * tokens * C + c;
            const int t1 = min(t0 + TBK, tokens);
            for (int t = t0; t < t1; ++t) {
                const T current = in[size_t(t) * XS];
                float acc = 0.0;
                for (int k = 0; k < KS - 1; ++k) {
                    acc += static_cast<float>(window[k]) * w[k];
                }
                acc += static_cast<float>(current) * w[KS - 1];
                for (int k = 0; k < KS - 2; ++k) {
                    window[k] = window[k + 1];
                }
                window[KS - 2] = current;
                const T convolved = static_cast<T>(acc);
                const T biased = convolved + bias_c;
                o[size_t(t) * C] = nemotron_super_silu(biased);
            }
            if (t0 == 0) {
                const T one = static_cast<T>(1);
                device T* next = next_state + size_t(b) * (KS - 1) * C + c;
                for (int k = 0; k < KS - 1; ++k) {
                    const int t = tokens - (KS - 1) + k;
                    const T kept = t < 0 ? rows[size_t(t + KS - 1) * C] : in[size_t(t) * XS];
                    next[size_t(k) * C] = kept * one;
                }
            }
            """,
        header: NemotronSuperMambaStep.header)

    /// Whether ``convolve(projected:columnOffset:channels:state:weight:bias:)``
    /// applies: depthwise `weight` `[C, K, 1]`, `bias` `[C]`, `state`
    /// `[B, K - 1, C]` and `projected` `[B, L, >= columnOffset + C]`, one dtype.
    static func supportsConvolve(
        projected: MLXArray, columnOffset: Int, channels: Int, state: MLXArray,
        weight: MLXArray, bias: MLXArray?
    ) -> Bool {
        guard let bias, projected.ndim == 3, projected.dim(2) >= columnOffset + channels,
            weight.shape.count == 3, weight.dim(0) == channels, weight.dim(2) == 1,
            weight.dim(1) >= 2,
            state.shape == [projected.dim(0), weight.dim(1) - 1, channels],
            bias.shape == [channels]
        else { return false }
        let dtype = projected.dtype
        return [.bfloat16, .float16, .float32].contains(dtype) && state.dtype == dtype
            && weight.dtype == dtype && bias.dtype == dtype
    }

    /// `silu(conv1d(concat(state, x)) + bias)` for the `channels` columns of
    /// `projected` from `columnOffset` (`[B, L, C]`), and the retained state
    /// (the last K - 1 rows of the concatenation, a new buffer).
    static func convolve(
        projected: MLXArray, columnOffset: Int, channels: Int, state: MLXArray,
        weight: MLXArray, bias: MLXArray
    ) -> (MLXArray, MLXArray) {
        let (batch, tokens) = (projected.dim(0), projected.dim(1))
        let taps = weight.dim(1)
        let perThread = 16
        let threads = 256
        let outputs = convolveKernel(
            [projected, state, weight, bias, MLXArray(Int32(tokens))],
            template: [
                ("T", projected.dtype), ("C", channels), ("KS", taps), ("TBK", perThread),
                ("XS", projected.dim(2)), ("XO", columnOffset),
            ],
            grid: (
                (channels + threads - 1) / threads * threads,
                (tokens + perThread - 1) / perThread, batch
            ),
            threadGroup: (threads, 1, 1),
            outputShapes: [[batch, tokens, channels], [batch, taps - 1, channels]],
            outputDTypes: [projected.dtype, projected.dtype])
        return (outputs[0], outputs[1])
    }

    // MARK: - Reference comparison

    /// One run of ``compareWithReference(tokens:iterations:seed:layout:operandType:chunkSize:)``.
    /// Relative errors are the largest absolute difference over the largest
    /// reference magnitude. Times are medians in milliseconds of evaluating
    /// the stage alone on inputs that are already evaluated.
    public struct Comparison: Sendable, CustomStringConvertible {
        public let tokens: Int
        public let layout: String
        public let operandType: String
        public let chunkSize: Int
        /// The mixer output (after out_proj). Its FP8 projection quantizes
        /// its input to E4M3, so one-ulp input differences move whole FP8
        /// steps: compare the gated-norm output for the scan itself.
        public let outputMaxAbs: Float
        public let outputMaxRel: Float
        public let outputMeanAbs: Float
        /// The gated-norm output (the out_proj input) of both scans, both
        /// normalized by the same op chain.
        public let normedMaxAbs: Float
        public let normedMaxRel: Float
        /// The SSD output alone (before the gated norm), both scans on the
        /// same convolution output.
        public let ssdOutputMaxAbs: Float
        public let ssdOutputMaxRel: Float
        public let ssmStateMaxAbs: Float
        public let ssmStateMaxRel: Float
        public let convStateMaxAbs: Float
        public let convStateMaxRel: Float
        /// Whole mixer, in_proj to out_proj.
        public let referenceLayerMs: Double
        public let candidateLayerMs: Double
        /// Convolution (concat, conv1d, bias, silu) alone.
        public let referenceConvMs: Double
        public let candidateConvMs: Double
        /// SSD scan alone (reference: `NemotronSuperSSD.scan`, the current
        /// fused-window path, on split views as the mixer passes them).
        public let referenceSSDMs: Double
        public let candidateSSDMs: Double

        public var description: String {
            String(
                format: """
                    Nemotron Super Mamba prefill, %d tokens, layout %@, operands %@, chunk %d
                      layer output   max abs %.3e  max rel %.3e  mean abs %.3e
                      gated norm     max abs %.3e  max rel %.3e
                      SSD output     max abs %.3e  max rel %.3e
                      SSM state      max abs %.3e  max rel %.3e
                      conv state     max abs %.3e  max rel %.3e
                      layer  reference %.3f ms  chunked %.3f ms
                      conv   reference %.3f ms  fused   %.3f ms
                      SSD    reference %.3f ms  chunked %.3f ms
                    """, tokens, layout, operandType, chunkSize, outputMaxAbs, outputMaxRel,
                outputMeanAbs, normedMaxAbs, normedMaxRel, ssdOutputMaxAbs, ssdOutputMaxRel,
                ssmStateMaxAbs, ssmStateMaxRel, convStateMaxAbs,
                convStateMaxRel, referenceLayerMs, candidateLayerMs, referenceConvMs,
                candidateConvMs, referenceSSDMs, candidateSSDMs)
        }
    }

    /// Builds one Super Mamba layer at the real shapes (d_model 4096, 128
    /// heads x 64, state 128, 8 groups, conv 4; FP8 ModelOpt in/out
    /// projections on the graph path, random weights) and a `tokens`-token
    /// chunk entering with random conv and SSM states, then runs the current
    /// prefill path and the chunked one on the GPU. `layout` defaults to
    /// `.native` on NAX GPUs and `.generic` elsewhere; `operandType` and
    /// `chunkSize` default to ``NemotronSuperSSDChunked``'s settings. Global
    /// switches are left as they were.
    public static func compareWithReference(
        tokens: Int = 2048, iterations: Int = 10, seed: UInt64 = 0,
        layout: NemotronSuperSSDChunked.Layout? = nil, operandType: DType? = nil,
        chunkSize: Int? = nil
    ) throws -> Comparison {
        precondition(tokens >= minimumTokens && iterations > 0, "comparison geometry")
        let resolvedLayout = layout ?? (NemotronSuperSSDChunked.naxAvailable ? .native : .generic)
        let operand = operandType ?? NemotronSuperSSDChunked.operandType
        let chunk = chunkSize ?? NemotronSuperSSDChunked.chunkSize
        let (mixer, hidden, convState, ssmState) = try comparisonLayer(tokens: tokens, seed: seed)
        let heads = mixer.numHeads
        func run(_ chunked: Bool) -> (MLXArray, MLXArray, MLXArray) {
            mixer.recurrentMix(
                hidden, convState: convState, ssmState: ssmState, chunked: chunked,
                layout: resolvedLayout, operandType: operand, chunkSize: chunk)
        }
        let reference = run(false)
        let candidate = run(true)
        eval(reference.0, reference.1, reference.2, candidate.0, candidate.1, candidate.2)

        // The scans alone on the same convolution output.
        let projected = mixer.inProj(hidden)
        let (convOutput, _) = convolve(
            projected: projected, columnOffset: mixer.intermediateSize,
            channels: mixer.convDim, state: convState, weight: mixer.conv1d.weight,
            bias: mixer.conv1d.bias!)
        eval(projected, convOutput)
        let width = mixer.intermediateSize
        let groupWidth = mixer.numGroups * mixer.ssmStateSize
        let parts = split(convOutput, indices: [width, width + groupWidth], axis: -1)
        let x = parts[0].reshaped(1, tokens, heads, mixer.headDim)
        let B = parts[1].reshaped(1, tokens, mixer.numGroups, mixer.ssmStateSize)
        let C = parts[2].reshaped(1, tokens, mixer.numGroups, mixer.ssmStateSize)
        let dt = projected[0..., 0..., (width + mixer.convDim)...]
        func referenceSSD() -> (MLXArray, MLXArray) {
            NemotronSuperSSD.scan(
                x: x, ALog: mixer.aLog, B: B, C: C, D: mixer.D, dt: dt, dtBias: mixer.dtBias,
                state: ssmState, timeStepLimit: mixer.timeStepLimit)
        }
        func candidateSSD(_ outputDType: DType) -> (MLXArray, MLXArray) {
            NemotronSuperSSDChunked.scan(
                xbc: convOutput, dt: projected, dtOffset: width + mixer.convDim, heads: heads,
                headDim: mixer.headDim, groups: mixer.numGroups, stateSize: mixer.ssmStateSize,
                aLog: mixer.aLog, dtBias: mixer.dtBias, D: mixer.D, state: ssmState,
                timeStepLimit: mixer.timeStepLimit, outputDType: outputDType,
                layout: resolvedLayout, operandType: operand, chunkSize: chunk)
        }
        let ssdReference = referenceSSD()
        let ssdCandidate = candidateSSD(ssdReference.0.dtype)
        eval(ssdReference.0, ssdCandidate.0)

        func errors(_ a: MLXArray, _ b: MLXArray) -> (Float, Float) {
            let difference = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
            let scale = abs(b.asType(.float32)).max().item(Float.self)
            return (difference, scale > 0 ? difference / scale : difference)
        }
        func median(_ body: () -> [MLXArray]) -> Double {
            for _ in 0 ..< 2 { eval(body()) }
            var times: [Double] = []
            for _ in 0 ..< iterations {
                let start = DispatchTime.now().uptimeNanoseconds
                eval(body())
                times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
            }
            times.sort()
            return times[times.count / 2]
        }
        let output = errors(candidate.0, reference.0)
        let outputMeanAbs = abs(candidate.0.asType(.float32) - reference.0.asType(.float32))
            .mean().item(Float.self)
        let gate = projected[0..., 0..., ..<width]
        let normed = errors(
            mixer.norm(ssdCandidate.0.asType(hidden.dtype).flattened(start: 2), gate: gate),
            mixer.norm(ssdReference.0.asType(hidden.dtype).flattened(start: 2), gate: gate))
        let ssd = errors(ssdCandidate.0, ssdReference.0)
        let ssm = errors(candidate.2, reference.2)
        let conv = errors(candidate.1, reference.1)
        let convInput = projected[0..., 0..., width ..< (width + mixer.convDim)]
        return Comparison(
            tokens: tokens, layout: resolvedLayout == .native ? "native" : "generic",
            operandType: "\(operand)", chunkSize: chunk,
            outputMaxAbs: output.0, outputMaxRel: output.1, outputMeanAbs: outputMeanAbs,
            normedMaxAbs: normed.0, normedMaxRel: normed.1, ssdOutputMaxAbs: ssd.0,
            ssdOutputMaxRel: ssd.1, ssmStateMaxAbs: ssm.0, ssmStateMaxRel: ssm.1,
            convStateMaxAbs: conv.0, convStateMaxRel: conv.1,
            referenceLayerMs: median {
                let r = run(false)
                return [r.0, r.1, r.2]
            },
            candidateLayerMs: median {
                let r = run(true)
                return [r.0, r.1, r.2]
            },
            referenceConvMs: median {
                [silu(mixer.conv1d(concatenated([convState, convInput], axis: 1)))]
            },
            candidateConvMs: median {
                let r = convolve(
                    projected: projected, columnOffset: width, channels: mixer.convDim,
                    state: convState, weight: mixer.conv1d.weight, bias: mixer.conv1d.bias!)
                return [r.0, r.1]
            },
            referenceSSDMs: median {
                let r = referenceSSD()
                return [r.0, r.1]
            },
            candidateSSDMs: median {
                let r = candidateSSD(ssdReference.0.dtype)
                return [r.0, r.1]
            })
    }

    /// One Super Mamba layer with random weights (FP8 ModelOpt projections on
    /// the graph path; every parameter but `A_log` in bfloat16, as
    /// `castPredicate` loads them), a bfloat16 `tokens`-token input scaled so the
    /// projections are about unit size, and random entering conv and float32
    /// SSM states; everything evaluated.
    static func comparisonLayer(tokens: Int, seed: UInt64) throws
        -> (NemotronHMamba2Mixer, MLXArray, MLXArray, MLXArray)
    {
        let config = NemotronHConfiguration(
            vocabSize: 1024, hiddenSize: 4096, numHiddenLayers: 1, numAttentionHeads: 32,
            numKeyValueHeads: 2, mambaNumHeads: 128, mambaHeadDim: 64, ssmStateSize: 128,
            convKernel: 4, nGroups: 8, intermediateSize: 2688, moeIntermediateSize: 2688,
            moeSharedExpertIntermediateSize: 5376, nRoutedExperts: 8, numExpertsPerTok: 2,
            hybridOverridePattern: "M")
        let mixer = NemotronHMamba2Mixer(config)
        let key = MLXRandom.key(seed)
        let keys = MLXRandom.split(key: key, into: 12)
        let dtype = DType.bfloat16
        let heads = mixer.numHeads
        // Mamba-2's dt initialization: dt log-uniform in [1e-3, 1e-1] and
        // dt_bias = dt + log(1 - exp(-dt)), the inverse softplus.
        let initialDt = MLX.exp(
            MLXRandom.uniform(low: log(Float(1e-3)), high: log(Float(1e-1)), [heads], key: keys[11])
        )
        let dtBias = initialDt + MLX.log(1 - MLX.exp(-initialDt))
        var values: [String: MLXArray] = [:]
        for (name, value) in mixer.parameters().flattened() {
            switch name {
            case "A_log":
                values[name] = MLX.log(MLXRandom.uniform(low: 1, high: 16, [heads], key: keys[0]))
            case "dt_bias":
                values[name] = dtBias.asType(dtype)
            case "D":
                values[name] = (1 + 0.1 * MLXRandom.normal([heads], key: keys[1])).asType(dtype)
            case "norm.weight":
                values[name] = (1 + 0.1 * MLXRandom.normal(value.shape, key: keys[2])).asType(dtype)
            case "conv1d.weight":
                values[name] = (0.5 * MLXRandom.normal(value.shape, key: keys[3])).asType(dtype)
            case "conv1d.bias":
                values[name] = (0.1 * MLXRandom.normal(value.shape, key: keys[4])).asType(dtype)
            default:
                values[name] = (0.02 * MLXRandom.normal(value.shape, key: keys[5])).asType(dtype)
            }
        }
        try mixer.update(parameters: .unflattened(values), verify: .all)
        func fp8(_ outputs: Int, _ inputs: Int, _ key: MLXArray) throws -> ModelOptCalibratedLinear
        {
            let raw = MLXRandom.randInt(Int32(0) ..< 256, [outputs, inputs], key: key)
                .asType(.uint8)
            // 0x7F / 0xFF are E4M3 NaN.
            let bytes = which((raw & UInt8(0x7F)) .== UInt8(0x7F), raw - UInt8(1), raw)
            return try ModelOptCalibratedLinear(
                weights: ModelOptFP8Weights(e4m3Bytes: bytes, tensorScale: 1e-3),
                inputScale: 64.0 / 448, weightDType: dtype)
        }
        let projection = mixer.intermediateSize + mixer.convDim + heads
        try mixer.update(
            modules: .unflattened([
                ("in_proj", fp8(projection, config.hiddenSize, keys[6])),
                ("out_proj", fp8(config.hiddenSize, mixer.intermediateSize, keys[7])),
            ]))
        eval(mixer.parameters())

        // Hidden states scaled so the projections are about unit size.
        var hidden = MLXRandom.normal([1, tokens, config.hiddenSize], key: keys[8]).asType(dtype)
        let probe = mixer.inProj(hidden).asType(.float32)
        let spread = sqrt((probe * probe).mean()).item(Float.self)
        hidden = (hidden.asType(.float32) / max(spread, 1e-3)).asType(dtype)
        let convState = MLXRandom.normal(
            [1, mixer.convKernelSize - 1, mixer.convDim], key: keys[9]
        )
        .asType(dtype)
        let ssmState =
            0.1 * MLXRandom.normal([1, heads, mixer.headDim, mixer.ssmStateSize], key: keys[10])
        eval(hidden, convState, ssmState)
        return (mixer, hidden, convState, ssmState)
    }

}
