// Copyright 2026 the HuggingFace Team. All rights reserved.
// Copyright 2026 Eigen Labs.
// Modified in 2026: FP32 Metal KDA recurrence with functional state and
// bounded prefix capture. The reference linear-attention path is unchanged.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Secondary execution reference: MLX-VLM's vector/capture kernel (MIT).
// Copyright © 2025 Prince Canuma
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import MLX
import MLXLMCommon

/// Whole-step KDA recurrence of the `LinearAttention.nativeRecurrence*` policy
/// members; the reference policy never selects it.
///
/// Primary equations: `recurrent_kimi_delta_attention` in
/// https://github.com/huggingface/transformers/blob/f0705f991a71f05dad9569336b181119f5546f93/src/transformers/models/glm5_next/modeling_glm5_next.py
/// Secondary vector/capture execution reference (NOT its transposed state ABI):
/// https://github.com/Blaizzy/mlx-vlm/blob/4623cf9d2633c5ee2b1ac125eeae51423447c41f/mlx_vlm/models/gated_delta.py
///
/// For every t, h, key channel k and value channel v:
///   D[k,v] = S[k,v] * exp(logDecay[k])
///   memory[v] = sum_k D[k,v] * key[k]
///   delta[v] = (value[v] - memory[v]) * beta
///   S'[k,v] = D[k,v] + key[k] * delta[v]
///   output[v] = sum_k S'[k,v] * query[k]
/// State is row-major [B,H,K,V], offset ((b*H+h)*K+k)*V+v.
/// One SIMD group owns one value channel; lanes reduce over key channels.
/// Every operation and buffer is FP32, with separate multiply/add operations.
/// The reduction tree differs from the ops baseline; bit parity is not assumed.
enum GLM5NextLinearKernel {
    enum Failure: Error, Equatable {
        case invalidGeometry(String)
        case invalidTensor(String)
        case invalidCapture(String)
        case metalUnavailable
    }

    /// Kernel safety limits, not serving admission or a model capability claim.
    static let maximumHeadDimension = 128
    static let maximumCapturedSteps = 32
    static let maximumCaptureBytes = 64 * 1024 * 1024

    struct Output {
        /// FP32 [B,T,H,V], before the caller's activation-dtype cast and output norm.
        let values: MLXArray
        /// Newly allocated FP32 [B,H,K,V]; the input state is never written.
        let state: MLXArray
        /// Optional FP32 [B,C,H,K,V], entry i is the state after token i.
        let capturedPrefixStates: MLXArray?
        private let initialState: MLXArray
        private let tokens: Int

        fileprivate init(
            values: MLXArray, state: MLXArray, capturedPrefixStates: MLXArray?,
            initialState: MLXArray, tokens: Int
        ) {
            self.values = values
            self.state = state
            self.capturedPrefixStates = capturedPrefixStates
            self.initialState = initialState
            self.tokens = tokens
        }

        /// Accept zero tokens, a captured prefix, or the complete window.
        /// Returns a lazy read-only view; no host readback or mutation is needed.
        /// A partial view retains the full bounded capture buffer until released.
        func state(acceptingPrefix count: Int) throws -> MLXArray {
            guard (0 ... tokens).contains(count) else {
                throw Failure.invalidCapture("accepted prefix is outside the window")
            }
            if count == 0 { return initialState }
            if count == tokens { return state }
            guard let capturedPrefixStates, count <= capturedPrefixStates.dim(1) else {
                throw Failure.invalidCapture("accepted prefix was not captured")
            }
            return capturedPrefixStates[0..., count - 1, 0..., 0..., 0...]
        }
    }

    /// Pure metadata validation, also usable without constructing large arrays.
    struct Layout {
        let batch: Int
        let tokens: Int
        let heads: Int
        let keyDimension: Int
        let valueDimension: Int
        let capturePrefixCount: Int
        let stateShape: [Int]
        let outputShape: [Int]
        let captureShape: [Int]?

        /// Each physically distinct output gets its own allocator bound. The
        /// admitted capture API reserves these before constructing any output.
        var outputAllocation: [CBv2MTPFixedBufferSpec] {
            ([outputShape, stateShape] + [captureShape].compactMap { $0 }).map {
                CBv2MTPFixedBufferSpec(logicalBytes: $0.reduce(4, *))
            }
        }

        init(queryShape: [Int], valueShape: [Int], capturePrefixCount: Int) throws {
            guard queryShape.count == 4, valueShape.count == 4 else {
                throw Failure.invalidGeometry("query/value must have rank four")
            }
            _ = try Self.elementCount(queryShape)
            _ = try Self.elementCount(valueShape)
            guard queryShape.prefix(3) == valueShape.prefix(3) else {
                throw Failure.invalidGeometry("query/value must share B,T,H")
            }
            batch = queryShape[0]
            tokens = queryShape[1]
            heads = queryShape[2]
            keyDimension = queryShape[3]
            valueDimension = valueShape[3]
            guard keyDimension <= maximumHeadDimension,
                valueDimension <= maximumHeadDimension
            else { throw Failure.invalidGeometry("K and V must be in 1...128") }
            guard capturePrefixCount >= 0, capturePrefixCount <= tokens,
                capturePrefixCount <= maximumCapturedSteps
            else { throw Failure.invalidCapture("capture must be in 0...min(T,32)") }
            self.capturePrefixCount = capturePrefixCount
            stateShape = [batch, heads, keyDimension, valueDimension]
            outputShape = valueShape
            _ = try Self.elementCount(stateShape)
            if capturePrefixCount > 0 {
                let shape = [batch, capturePrefixCount, heads, keyDimension, valueDimension]
                let elements = try Self.elementCount(shape)
                guard elements <= maximumCaptureBytes / MemoryLayout<Float>.size else {
                    throw Failure.invalidCapture("captured states exceed 64 MiB")
                }
                captureShape = shape
            } else {
                captureShape = nil
            }
        }

        private static func elementCount(_ shape: [Int]) throws -> Int {
            var count = 1
            for dimension in shape {
                guard dimension > 0, dimension <= Int(Int32.max) else {
                    throw Failure.invalidGeometry("dimensions must be positive Int32 values")
                }
                let (next, overflow) = count.multipliedReportingOverflow(by: dimension)
                guard !overflow, next <= Int(Int32.max) else {
                    throw Failure.invalidGeometry("tensor exceeds Int32 indexing")
                }
                count = next
            }
            return count
        }
    }

    /// An actual lease from the existing request's prepaid MTP allocation, not
    /// an extra capacity declaration. Retain this owner through all readers and
    /// the evaluation fence; partial states still retain the entire stack.
    struct AdmittedCapture {
        private let output: Output
        private let lease: CBv2MTPAllocationLease

        fileprivate init(output: Output, lease: CBv2MTPAllocationLease) {
            self.output = output
            self.lease = lease
        }

        func withEvaluationLease<R>(_ body: (Output) throws -> R) rethrows -> R {
            try withExtendedLifetime(lease) { try body(output) }
        }
    }

    /// For an admitted caller, reserve every returned buffer from the existing
    /// resident pool. A missing/insufficient declaration fails at reserve; this
    /// never creates or enlarges admission. `apply` remains the unadmitted math
    /// primitive for isolated qualification and is not an MTP serving adapter.
    /// Inputs and contiguous-copy scratch retain their existing activation charge.
    static func applyAdmittedCapture(
        query: MLXArray, key: MLXArray, value: MLXArray,
        logDecay: MLXArray, beta: MLXArray, state: MLXArray,
        capturePrefixCount: Int, admission: CBv2AdmittedMTPContext
    ) throws -> AdmittedCapture {
        guard capturePrefixCount > 0 else {
            throw Failure.invalidCapture("admitted capture requires a nonempty prefix")
        }
        let layout = try validateInputs(
            query: query, key: key, value: value, logDecay: logDecay, beta: beta,
            state: state, capturePrefixCount: capturePrefixCount)
        guard layout.batch == 1, layout.tokens <= admission.maximumInputs else {
            throw Failure.invalidCapture("capture exceeds the admitted single-request envelope")
        }
        try requireDefaultGPUStream()
        let lease = try admission.reserve(.resident, buffers: layout.outputAllocation)
        let output = try apply(
            query: query, key: key, value: value, logDecay: logDecay, beta: beta,
            state: state, capturePrefixCount: capturePrefixCount)
        return AdmittedCapture(output: output, lease: lease)
    }

    /// No fallback: CPU, foreign streams and non-Metal platforms fail clearly.
    static func requireDefaultGPUStream() throws {
        guard glm5NextOnDefaultGPUStream() else { throw Failure.metalUnavailable }
    }

    /// Inputs are PRECOMPUTED FP32: q is already L2-normalized and scaled by
    /// 1/sqrt(K); k is already L2-normalized; v already crossed the convolution
    /// activation-dtype boundary. logDecay is per key, NOT an exponentiated gate.
    /// beta [B,T,H] must already be rounded to the projection dtype then widened
    /// to FP32. This function does not normalize, cast, sigmoid or convolve.
    ///
    /// Validation precedes allocation/dispatch. MLXFast owns the new outputs,
    /// retains the lazy input graph, and makes contiguous copies when necessary.
    /// No raw Metal buffers, mutable inputs, host reads, or eager eval are used.
    /// Supports only the default GPU stream, K,V in 1...128,
    /// and capture counts in 0...min(T,32) whose FP32 buffers fit 64 MiB.
    static func apply(
        query: MLXArray, key: MLXArray, value: MLXArray,
        logDecay: MLXArray, beta: MLXArray, state: MLXArray? = nil,
        capturePrefixCount: Int = 0
    ) throws -> Output {
        let layout = try validateInputs(
            query: query, key: key, value: value, logDecay: logDecay, beta: beta,
            state: state, capturePrefixCount: capturePrefixCount)
        try requireDefaultGPUStream()
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            let initial = state ?? MLXArray.zeros(layout.stateShape, dtype: .float32)
            let capturing = layout.captureShape != nil
            let kernel = capturing ? capturedKernel : plainKernel
            var shapes = [layout.outputShape, layout.stateShape]
            if let captureShape = layout.captureShape { shapes.append(captureShape) }
            let outputs = kernel(
                [query, key, value, logDecay, beta, initial, MLXArray(Int32(layout.tokens))],
                template: [
                    ("H", layout.heads), ("K", layout.keyDimension),
                    ("V", layout.valueDimension), ("CaptureT", layout.capturePrefixCount),
                ],
                grid: (32, layout.valueDimension, layout.batch * layout.heads),
                threadGroup: (32, 4, 1),
                outputShapes: shapes, outputDTypes: shapes.map { _ in .float32 })
            return Output(
                values: outputs[0], state: outputs[1],
                capturedPrefixStates: capturing ? outputs[2] : nil,
                initialState: initial, tokens: layout.tokens)
        #else
            throw Failure.metalUnavailable
        #endif
    }

    private static func validateInputs(
        query: MLXArray, key: MLXArray, value: MLXArray,
        logDecay: MLXArray, beta: MLXArray, state: MLXArray?, capturePrefixCount: Int
    ) throws -> Layout {
        let layout = try Layout(
            queryShape: query.shape, valueShape: value.shape,
            capturePrefixCount: capturePrefixCount)
        for (name, array, shape) in [
            ("query", query, query.shape), ("key", key, query.shape),
            ("value", value, layout.outputShape), ("logDecay", logDecay, query.shape),
            ("beta", beta, [layout.batch, layout.tokens, layout.heads]),
        ] {
            try validate(array, shape: shape, name: name)
        }
        if let state { try validate(state, shape: layout.stateShape, name: "state") }
        return layout
    }

    private static func validate(_ array: MLXArray, shape: [Int], name: String) throws {
        guard array.shape == shape, array.dtype == .float32 else {
            throw Failure.invalidTensor("\(name) must be FP32 \(shape)")
        }
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        private static let plainKernel = makeKernel(capturing: false)
        private static let capturedKernel = makeKernel(capturing: true)

        private static func makeKernel(capturing: Bool) -> MLXFast.MLXFastKernel {
            let captureStore: String
            if capturing {
                captureStore = """
                    if (t < CaptureT) {
                        const size_t prefix = ((b * CaptureT + t) * H + h) * K * V;
                        captured[prefix + keyIndex * V + dv] = s[i];
                    }
                    """
            } else {
                captureStore = ""
            }
            let source = """
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                const size_t n = thread_position_in_grid.z;
                const size_t b = n / H;
                const size_t h = n % H;
                const uint dv = thread_position_in_grid.y;
                const uint lane = thread_index_in_simdgroup;
                constexpr int entries = (K + 31) / 32;
                float s[entries];
                for (int i = 0; i < entries; ++i) {
                    const uint keyIndex = lane + 32 * i;
                    s[i] = keyIndex < K ? state_in[(n * K + keyIndex) * V + dv] : 0.0f;
                }
                for (int t = 0; t < T; ++t) {
                    const size_t tokenHead = (b * T + t) * H + h;
                    const size_t keyBase = tokenHead * K;
                    float memory = 0.0f;
                    for (int i = 0; i < entries; ++i) {
                        const uint keyIndex = lane + 32 * i;
                        if (keyIndex < K) {
                            // Same FP32 exponential as MLX's unary Exp, not fast::exp.
                            s[i] = s[i] * metal::precise::exp(log_decay[keyBase + keyIndex]);
                            memory += s[i] * key[keyBase + keyIndex];
                        }
                    }
                    memory = simd_sum(memory);
                    const float delta = (value[tokenHead * V + dv] - memory) * beta[tokenHead];
                    float out = 0.0f;
                    for (int i = 0; i < entries; ++i) {
                        const uint keyIndex = lane + 32 * i;
                        if (keyIndex < K) {
                            s[i] = s[i] + key[keyBase + keyIndex] * delta;
                            out += s[i] * query[keyBase + keyIndex];
                            \(captureStore)
                        }
                    }
                    out = simd_sum(out);
                    if (lane == 0) { y[tokenHead * V + dv] = out; }
                }
                for (int i = 0; i < entries; ++i) {
                    const uint keyIndex = lane + 32 * i;
                    if (keyIndex < K) { state_out[(n * K + keyIndex) * V + dv] = s[i]; }
                }
                """
            return MLXFast.metalKernel(
                name: capturing ? "glm5_next_kda_fp32_capture" : "glm5_next_kda_fp32",
                inputNames: ["query", "key", "value", "log_decay", "beta", "state_in", "T"],
                outputNames: capturing ? ["y", "state_out", "captured"] : ["y", "state_out"],
                source: source, ensureRowContiguous: true)
        }
    #endif
}
