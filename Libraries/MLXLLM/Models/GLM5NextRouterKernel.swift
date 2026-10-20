// Copyright 2026 the HuggingFace Team. All rights reserved.
// Copyright 2026 Eigen Labs.
// Modified in 2026: opt-in two-launch sigmoid/noaux_tc router for short steps
// and, on Neural-Accelerator GPUs, for prefill.
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

import MLX

/// `GLM5NextRouter` for steps of at most `maximumTokens` rows, selected only by
/// `GLM5NextExecutionPolicy.Routing.nativeKernel`, in two launches: FP32 logits
/// from the stored router weight (widened in registers, never copied), then one
/// SIMD group per row for sigmoid, correction bias, top-K and normalization.
/// Same equations as the reference; the FP32 dot products and the K-weight sum
/// use their own order, so not bitwise, and a near-tie expert can change.
/// Only ungrouped routing (`topk_group == n_group`) is supported.
///
/// `routePrefill` (`Routing.nativeKernelPrefill`, longer steps) computes the
/// logits with `GLM5NextMoEPrefillKernel.routerLogits` (tensor ops, exact
/// BF16/FP16 products, FP32 accumulation) and selects with the same per-row
/// launch, in place of the reference's FP32 copies of `x` and the weight, its
/// FP32 matmul and its seven selection launches.
enum GLM5NextRouterKernel {
    enum Failure: Error, Equatable {
        case invalidGeometry(String)
        case unsupportedDeviceOrStream
    }

    static let maximumTokens = 16
    static let maximumExperts = 32 * 16
    static let maximumTopK = 16
    /// Experts per logits threadgroup, one SIMD group each.
    static let expertsPerGroup = 2

    static func supports(tokens: Int) -> Bool { (1 ... maximumTokens).contains(tokens) }

    /// Steps `routePrefill` serves: longer than `maximumTokens`, with an
    /// activation and router weight `routerLogits` reads.
    static func supportsPrefill(tokens: Int, x: MLXArray, weight: MLXArray, topK: Int) -> Bool {
        tokens > maximumTokens && weight.ndim == 2
            && (1 ... maximumExperts).contains(weight.dim(0))
            && (1 ... maximumTopK).contains(topK) && topK <= weight.dim(0)
            && GLM5NextMoEPrefillKernel.admitsRouterLogits(
                tokens: tokens, hidden: weight.dim(1), experts: weight.dim(0),
                activation: x.dtype, weight: weight.dtype)
    }

    /// `x` `[N,D]`, weight `[E,D]`, bias `[E]`. Returns FP32 logits `[N,E]`,
    /// FP32 scaled weights `[N,K]` and UInt32 expert indices `[N,K]`.
    static func route(
        _ x: MLXArray, weight: MLXArray, bias: MLXArray, topK: Int, normTopkProb: Bool,
        routedScalingFactor: Float
    ) throws -> (logits: MLXArray, weights: MLXArray, indices: MLXArray) {
        guard x.ndim == 2, supports(tokens: x.dim(0)), weight.ndim == 2,
            weight.dim(1) == x.dim(1), x.dim(1) > 0, bias.shape == [weight.dim(0)],
            (1 ... maximumExperts).contains(weight.dim(0)), (1 ... maximumTopK).contains(topK),
            topK <= weight.dim(0), x.dtype.isFloatingPoint, weight.dtype.isFloatingPoint,
            bias.dtype.isFloatingPoint, routedScalingFactor.isFinite
        else { throw Failure.invalidGeometry("router geometry") }
        guard glm5NextOnDefaultGPUStream() else { throw Failure.unsupportedDeviceOrStream }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            let tokens = x.dim(0)
            let experts = weight.dim(0)
            let groups = (experts + expertsPerGroup - 1) / expertsPerGroup
            let logits = logitsKernel(
                [x, weight],
                template: [
                    ("Tokens", tokens), ("Width", x.dim(1)), ("Experts", experts),
                    ("GroupExperts", expertsPerGroup),
                ],
                grid: (32 * expertsPerGroup * groups, 1, 1),
                threadGroup: (32 * expertsPerGroup, 1, 1),
                outputShapes: [[tokens, experts]], outputDTypes: [.float32])[0]
            return select(
                logits, bias: bias, topK: topK, normTopkProb: normTopkProb,
                routedScalingFactor: routedScalingFactor)
        #else
            throw Failure.unsupportedDeviceOrStream
        #endif
    }

    /// `route` for steps longer than `maximumTokens` (`supportsPrefill`):
    /// `x` `[N,D]`, weight `[E,D]` of x's dtype, bias `[E]`. Same outputs.
    static func routePrefill(
        _ x: MLXArray, weight: MLXArray, bias: MLXArray, topK: Int, normTopkProb: Bool,
        routedScalingFactor: Float
    ) throws -> (logits: MLXArray, weights: MLXArray, indices: MLXArray) {
        guard x.ndim == 2, weight.ndim == 2, weight.dim(1) == x.dim(1),
            supportsPrefill(tokens: x.dim(0), x: x, weight: weight, topK: topK),
            bias.shape == [weight.dim(0)], bias.dtype.isFloatingPoint,
            routedScalingFactor.isFinite
        else { throw Failure.invalidGeometry("prefill router geometry") }
        guard glm5NextOnDefaultGPUStream() else { throw Failure.unsupportedDeviceOrStream }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            let logits = try GLM5NextMoEPrefillKernel.routerLogits(x, weight: weight)
            return select(
                logits, bias: bias, topK: topK, normTopkProb: normTopkProb,
                routedScalingFactor: routedScalingFactor)
        #else
            throw Failure.unsupportedDeviceOrStream
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        /// One SIMD group per row of FP32 `logits` `[N,E]`.
        private static func select(
            _ logits: MLXArray, bias: MLXArray, topK: Int, normTopkProb: Bool,
            routedScalingFactor: Float
        ) -> (logits: MLXArray, weights: MLXArray, indices: MLXArray) {
            let (tokens, experts) = (logits.dim(0), logits.dim(1))
            let selected = selectKernel(
                [logits, bias, MLXArray(routedScalingFactor)],
                template: [
                    ("Experts", experts), ("TopK", topK), ("Normalize", normTopkProb),
                ],
                grid: (32, tokens, 1), threadGroup: (32, 1, 1),
                outputShapes: [[tokens, topK], [tokens, topK]],
                outputDTypes: [.float32, .uint32])
            return (logits, selected[0], selected[1])
        }

        // SIMD group g of threadgroup b owns expert b*2+g; lane l sums columns
        // l, l+32, ... for every row, then one SIMD sum per row.
        private static let logitsKernel = MLXFast.metalKernel(
            name: "glm5_next_router_logits",
            inputNames: ["x", "weight"], outputNames: ["logits"],
            source: """
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                const uint expert = threadgroup_position_in_grid.x * GroupExperts
                    + simdgroup_index_in_threadgroup;
                if (expert >= uint(Experts)) return;
                const uint lane = thread_index_in_simdgroup;
                const size_t row = size_t(expert) * Width;
                float totals[Tokens];
                for (int t = 0; t < Tokens; ++t) { totals[t] = 0.0f; }
                for (uint column = lane; column < uint(Width); column += 32) {
                    const float w = static_cast<float>(weight[row + column]);
                    for (int t = 0; t < Tokens; ++t) {
                        const float product = static_cast<float>(x[size_t(t) * Width + column]) * w;
                        totals[t] += product;
                    }
                }
                for (int t = 0; t < Tokens; ++t) {
                    const float total = simd_sum(totals[t]);
                    if (lane == 0) { logits[size_t(t) * Experts + expert] = total; }
                }
                """,
            ensureRowContiguous: true)

        // One SIMD group per row. Lane l owns experts l, l+32, ...; each round
        // takes the largest bias-corrected score (lowest ID on an exact tie).
        // Sigmoid is MLX's form; the mixture weight is the uncorrected score.
        private static let selectKernel = MLXFast.metalKernel(
            name: "glm5_next_router_select",
            inputNames: ["logits", "bias", "scaling"], outputNames: ["weights", "indices"],
            source: """
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                constexpr int PerLane = (Experts + 31) / 32;
                const uint row = threadgroup_position_in_grid.y;
                const uint lane = thread_index_in_simdgroup;
                float score[PerLane];
                float corrected[PerLane];
                bool taken[PerLane];
                for (int i = 0; i < PerLane; ++i) {
                    const uint expert = lane + 32 * i;
                    taken[i] = expert >= uint(Experts);
                    if (!taken[i]) {
                        const float logit = logits[size_t(row) * Experts + expert];
                        const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(logit)));
                        score[i] = logit < 0.0f ? y : 1.0f - y;
                        corrected[i] = score[i] + static_cast<float>(bias[expert]);
                    } else {
                        score[i] = 0.0f;
                        corrected[i] = -INFINITY;
                    }
                }
                float chosen[TopK];
                uint chosenIndex[TopK];
                for (int k = 0; k < TopK; ++k) {
                    float best = -INFINITY;
                    uint bestIndex = 0xFFFFFFFFu;
                    float bestScore = 0.0f;
                    for (int i = 0; i < PerLane; ++i) {
                        const uint expert = lane + 32 * i;
                        if (!taken[i] && (corrected[i] > best || bestIndex == 0xFFFFFFFFu)) {
                            best = corrected[i];
                            bestIndex = expert;
                            bestScore = score[i];
                        }
                    }
                    const float top = simd_max(bestIndex == 0xFFFFFFFFu ? -INFINITY : best);
                    const uint winner = simd_min(
                        bestIndex != 0xFFFFFFFFu && best == top ? bestIndex : 0xFFFFFFFFu);
                    chosen[k] = simd_broadcast(bestScore, ushort(winner % 32));
                    chosenIndex[k] = winner;
                    if (winner % 32 == lane) { taken[winner / 32] = true; }
                }
                if (lane == 0) {
                    float total = 0.0f;
                    if (Normalize) {
                        for (int k = 0; k < TopK; ++k) { total += chosen[k]; }
                        total = total + 1e-20f;
                    }
                    for (int k = 0; k < TopK; ++k) {
                        const float normalized = Normalize ? chosen[k] / total : chosen[k];
                        weights[size_t(row) * TopK + k] = normalized * scaling;
                        indices[size_t(row) * TopK + k] = chosenIndex[k];
                    }
                }
                """,
            ensureRowContiguous: true)
    #endif
}
