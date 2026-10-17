// Copyright 2026 the HuggingFace Team. All rights reserved.
// Copyright 2026 Eigen Labs.
// Modified in 2026: opt-in fused routed-expert combine Metal path.
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

/// One-launch replacement for the routed-expert combine in `GLM5NextExperts`
/// (slot argSort, two takeAlongs, FP32 weighting, casts, zeros and K adds).
///
/// For each token, contributions are `T(float(projected) * weight)` with FP32
/// router weights, then added in the activation dtype starting from zero, in
/// ascending expert-ID order (`Glm5NextTextExperts` iterates experts by ID and
/// `index_add_`s each cast contribution). This is the reference's rounding
/// sequence exactly. Expert IDs must be distinct per token, as the router
/// guarantees; equal IDs keep slot order.
enum GLM5NextMoEKernel {
    enum Failure: Error, Equatable {
        case invalidTensor(String)
        case unsupportedDeviceOrStream
    }

    static let maximumTopK = 16

    /// `projected` is [A, D] with A = tokens * K. Without `rowMap`, row
    /// token*K + slot holds that slot's output; with it, `rowMap[token*K+slot]`
    /// names the row (the inverse permutation of an expert-sorted gather).
    /// `addend` [tokens, D] (the shared experts' output) is then added with
    /// the reference's single activation-dtype rounding of `routed + shared`.
    static func ascendingCombine(
        projected: MLXArray, rowMap: MLXArray? = nil, indices: MLXArray, weights: MLXArray,
        addend: MLXArray? = nil
    ) throws -> MLXArray {
        guard indices.ndim == 2, indices.dim(0) > 0,
            (1 ... maximumTopK).contains(indices.dim(1)),
            indices.dtype == .uint32 || indices.dtype == .int32
        else { throw Failure.invalidTensor("indices must be integer [tokens, 1...16]") }
        guard weights.shape == indices.shape, weights.dtype == .float32 else {
            throw Failure.invalidTensor("weights must be FP32 and match indices")
        }
        let tokens = indices.dim(0)
        let topK = indices.dim(1)
        let assignments = tokens * topK
        guard projected.ndim == 2, projected.dim(0) == assignments, projected.dim(1) > 0,
            [.float32, .float16, .bfloat16].contains(projected.dtype),
            projected.size <= Int(Int32.max)
        else { throw Failure.invalidTensor("projected must be floating [tokens*K, D]") }
        if let rowMap {
            guard rowMap.shape == [assignments], rowMap.dtype == .uint32 else {
                throw Failure.invalidTensor("rowMap must be uint32 [tokens*K]")
            }
        }
        if let addend {
            guard addend.shape == [tokens, projected.dim(1)], addend.dtype == projected.dtype
            else { throw Failure.invalidTensor("addend must match [tokens, D] and dtype") }
        }
        guard glm5NextOnDefaultGPUStream() else { throw Failure.unsupportedDeviceOrStream }
        #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
            let width = projected.dim(1)
            let inputs =
                [projected, indices, weights] + (rowMap.map { [$0] } ?? [])
                + (addend.map { [$0] } ?? [])
            let kernel =
                switch (rowMap != nil, addend != nil) {
                case (false, false): directKernel
                case (true, false): mappedKernel
                case (false, true): directAddKernel
                case (true, true): mappedAddKernel
                }
            // Four adjacent columns per thread when they divide the width.
            let lanes = width % 4 == 0 ? 4 : 1
            return kernel(
                inputs,
                template: [
                    ("T", projected.dtype), ("TopK", topK), ("Width", width), ("Lanes", lanes),
                ],
                grid: (width / lanes, tokens, 1), threadGroup: (256, 1, 1),
                outputShapes: [[tokens, width]], outputDTypes: [projected.dtype])[0]
        #else
            throw Failure.unsupportedDeviceOrStream
        #endif
    }

    #if os(macOS) || os(iOS) || os(tvOS) || os(visionOS)
        private static let directKernel = makeKernel(mapped: false, adds: false)
        private static let mappedKernel = makeKernel(mapped: true, adds: false)
        private static let directAddKernel = makeKernel(mapped: false, adds: true)
        private static let mappedAddKernel = makeKernel(mapped: true, adds: true)

        private static func makeKernel(mapped: Bool, adds: Bool) -> MLXFast.MLXFastKernel {
            let row = mapped ? "size_t(row_map[first + slot])" : "first + slot"
            let store =
                adds
                ? "static_cast<T>(total[c] + addend[size_t(token) * Width + column + c])"
                : "total[c]"
            return MLXFast.metalKernel(
                name: "glm5_next_moe_ascending_combine" + (mapped ? "_mapped" : "")
                    + (adds ? "_add" : ""),
                inputNames: ["projected", "indices", "weights"] + (mapped ? ["row_map"] : [])
                    + (adds ? ["addend"] : []),
                outputNames: ["combined"],
                source: """
                    #pragma clang fp reassociate(off)
                    #pragma clang fp contract(off)
                    const uint column = thread_position_in_grid.x * Lanes;
                    const uint token = thread_position_in_grid.y;
                    const size_t first = size_t(token) * TopK;
                    // Stable insertion sort of this token's slots by expert ID.
                    uint ids[TopK];
                    uint order[TopK];
                    for (int slot = 0; slot < TopK; ++slot) {
                        const uint id = static_cast<uint>(indices[first + slot]);
                        int position = slot;
                        while (position > 0 && ids[position - 1] > id) {
                            ids[position] = ids[position - 1];
                            order[position] = order[position - 1];
                            --position;
                        }
                        ids[position] = id;
                        order[position] = uint(slot);
                    }
                    T total[Lanes];
                    for (int c = 0; c < Lanes; ++c) { total[c] = static_cast<T>(0.0f); }
                    for (int rank = 0; rank < TopK; ++rank) {
                        const uint slot = order[rank];
                        const size_t row = \(row);
                        const float weight = weights[first + slot];
                        for (int c = 0; c < Lanes; ++c) {
                            const float weighted =
                                static_cast<float>(projected[row * Width + column + c]) * weight;
                            const T contribution = static_cast<T>(weighted);
                            total[c] = static_cast<T>(total[c] + contribution);
                        }
                    }
                    for (int c = 0; c < Lanes; ++c) {
                        combined[size_t(token) * Width + column + c] = \(store);
                    }
                    """,
                ensureRowContiguous: true)
        }
    #endif
}
