// Video frame sampling, timestamps, tubelet labels and the frame grid rule translated
// from NVIDIA's processing_nemotron_h_omni.py and image_processing_nemotron_h_omni.py
// and from vLLM's uniform (opencv) video loader. EVS token selection follows vLLM's
// multimodal/video_prune/evs.py.
// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// Copyright contributors to the vLLM project.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import Foundation
import MLX

/// CPU-only video planning: frame sampling, timestamps, tubelet labels and the
/// per-frame grid. Frames are resized and normalized exactly like images
/// (`NemotronSuperImagePreprocessing.normalizeRGB`). Does not decode files.
public enum NemotronSuperVideoPreprocessing {
    public enum Failure: Error, Equatable { case emptyVideo, invalidFrameRate, invalidGeometry }
    public static let targetFramesPerSecond: Double = 2
    public static let maximumFrames = 256
    public static let framesPerTubelet = 2
    public static let defaultPruningRate: Double = 0.5
    /// `video_target_num_patches`, before the 2x2 pixel shuffle.
    static let targetPatches = 1024
    static let patchSize = 16
    static let downsampleFactor = 2
    /// The tower's per-image (and per-tubelet) patch limit.
    static let maximumPatches = 13312

    /// `min(maximumFrames, totalFrames, floor(totalFrames / sourceFPS * 2))`
    /// frames, at least one, spread with `np.linspace(0, totalFrames - 1, n,
    /// dtype=int)`, which truncates.
    public static func sampledFrameIndices(totalFrames: Int, sourceFPS: Double) throws -> [Int] {
        guard totalFrames > 0 else { throw Failure.emptyVideo }
        guard sourceFPS.isFinite, sourceFPS > 0 else { throw Failure.invalidFrameRate }
        let byRate = (Double(totalFrames) / sourceFPS * targetFramesPerSecond).rounded(.down)
        let limit = min(maximumFrames, totalFrames)
        let count = max(1, byRate < Double(limit) ? Int(byRate) : limit)
        if count == totalFrames { return Array(0 ..< totalFrames) }
        if count == 1 { return [0] }
        let step = Double(totalFrames - 1) / Double(count - 1)
        return (0 ..< count).map {
            $0 == count - 1 ? totalFrames - 1 : Int((Double($0) * step).rounded(.down))
        }
    }

    /// Upstream truncates the frame duration to whole milliseconds before
    /// scaling (`int(1000 / fps)`: 33 ms at 30 fps), then prints `%.2f`.
    /// A frame rate that `sampledFrameIndices` rejects prints "0.00".
    public static func timestamp(frameIndex: Int, sourceFPS: Double) -> String {
        let milliseconds = 1000 / sourceFPS
        let duration =
            milliseconds.isFinite && milliseconds >= 0 && milliseconds < 1e15
            ? Int(milliseconds) : 0
        let (product, overflow) = frameIndex.multipliedReportingOverflow(by: duration)
        return String(format: "%.2f", overflow ? 0 : Double(product) / 1000)
    }

    /// One label per tubelet, exact upstream text up to and including ": " (no image tokens).
    /// An odd last tubelet is labelled with its single frame. The processor joins
    /// the tubelet groups with "\n".
    public static func tubeletLabels(frameIndices: [Int], sourceFPS: Double) -> [String] {
        stride(from: 0, to: frameIndices.count, by: framesPerTubelet).map { first in
            (first ..< min(first + framesPerTubelet, frameIndices.count)).map { frame in
                let prefix = frame == first ? "Frame" : "frame"
                let seconds = timestamp(frameIndex: frameIndices[frame], sourceFPS: sourceFPS)
                return "\(prefix) \(frame + 1) sampled at \(seconds) seconds"
            }
            .joined(separator: " and ") + ": "
        }
    }

    /// 1024-patch even-snapped grid; `tokens` = widthPatches * heightPatches / 4 (per tubelet).
    /// Mirrors `_compute_target_patches_video` with Python's ties-to-even round:
    /// both sides snap up together when the product still fits the target,
    /// otherwise both snap down. Two sides that round up to even values can still
    /// exceed the target (1000 x 650 gives 40 x 26 patches), as upstream.
    public static func frameGrid(width: Int, height: Int) throws
        -> NemotronSuperImagePreprocessing.Grid
    {
        guard width > 0, height > 0 else { throw Failure.invalidGeometry }
        let aspect = Double(width) / Double(height)
        func side(_ area: Double) -> Int {
            max(Int(area.squareRoot().rounded(.toNearestOrEven)), 1)
        }
        let divisor = downsampleFactor
        var rows = side(Double(targetPatches) / aspect)
        var columns = side(Double(targetPatches) * aspect)
        let upRows = rows % divisor == 0 ? rows : rows + divisor - rows % divisor
        let upColumns = columns % divisor == 0 ? columns : columns + divisor - columns % divisor
        let (upPatches, upOverflow) = upRows.multipliedReportingOverflow(by: upColumns)
        if !upOverflow && upPatches <= targetPatches {
            (rows, columns) = (upRows, upColumns)
        } else {
            rows = max(divisor, rows - rows % divisor)
            columns = max(divisor, columns - columns % divisor)
        }
        let (patches, overflow) = rows.multipliedReportingOverflow(by: columns)
        guard !overflow, patches <= maximumPatches else { throw Failure.invalidGeometry }
        return .init(
            widthPatches: columns, heightPatches: rows, width: columns * patchSize,
            height: rows * patchSize, tokens: patches / (divisor * divisor))
    }
}

/// Efficient Video Sampling: weight-free pruning of projected video tokens.
public enum NemotronSuperEVS {
    /// `features`: [T, N, D]. Returns per tubelet the kept token positions, ascending.
    /// Each token scores `1 - cos` against the same position of the previous
    /// tubelet; the first tubelet is always kept, then the highest scores (ties to
    /// the earlier tubelet and position) up to `max(N, Int(T * N * (1 - q)))`
    /// tokens overall. A non-positive or non-finite `pruningRate` keeps every token.
    /// Evaluates the similarity graph synchronously when anything is pruned; the
    /// caller owns MLX fault handling. Returns `[]` unless `features` is a
    /// non-empty rank-3 array.
    public static func keptPositions(features: MLXArray, pruningRate: Double) -> [[Int]] {
        guard features.ndim == 3, features.dim(0) > 0, features.dim(1) > 0, features.dim(2) > 0
        else { return [] }
        let tubelets = features.dim(0)
        let tokens = features.dim(1)
        guard
            retainedCount(tubelets: tubelets, tokens: tokens, pruningRate: pruningRate)
                < tubelets * tokens
        else { return Array(repeating: Array(0 ..< tokens), count: tubelets) }
        return keptPositions(
            dissimilarity: dissimilarity(features).asArray(Float.self),
            tubelets: tubelets, tokens: tokens, pruningRate: pruningRate)
    }

    /// vLLM `compute_retained_tokens_count`.
    static func retainedCount(tubelets: Int, tokens: Int, pruningRate: Double) -> Int {
        let total = tubelets * tokens
        guard pruningRate.isFinite, pruningRate > 0 else { return total }
        return max(tokens, Int(Double(total) * (1 - min(pruningRate, 1))))
    }

    /// `1 - cos` of tubelets 1..<T against their predecessors: a lazy [T - 1, N] FP32
    /// graph, for hosts that retain arrays before evaluating them. Like torch
    /// `cosine_similarity` in the feature dtype: each row is divided by its norm
    /// clamped to 1e-8, and the products are summed in FP32. Each row depends only on
    /// a tubelet and its predecessor, so a host may score overlapping windows of
    /// consecutive tubelets ([k + 1, N, D] sharing one tubelet) and concatenate them.
    public static func dissimilarity(_ features: MLXArray) -> MLXArray {
        let promoted = features.asType(.float32)
        let norms = maximum(sum(promoted * promoted, axis: -1, keepDims: true).sqrt(), Float(1e-8))
        let unit = features / norms.asType(features.dtype)
        let tubelets = features.dim(0)
        let products = unit[1..., 0..., 0...] * unit[0 ..< (tubelets - 1), 0..., 0...]
        let similarity = sum(products.asType(.float32), axis: -1).asType(features.dtype)
        return (1 - similarity).asType(.float32)
    }

    /// Selection over the evaluated row-major [T - 1, N] `dissimilarity` scores.
    /// Upstream scores the first tubelet 255, above any `1 - cos`; it is kept here
    /// unconditionally. The rest follow torch's stable descending argsort, which
    /// ranks NaN first. Returns `[]` when the score count does not match.
    public static func keptPositions(
        dissimilarity scores: [Float], tubelets: Int, tokens: Int, pruningRate: Double
    ) -> [[Int]] {
        guard tubelets > 0, tokens > 0, scores.count == (tubelets - 1) * tokens else { return [] }
        let extra =
            retainedCount(tubelets: tubelets, tokens: tokens, pruningRate: pruningRate) - tokens
        let ranked = scores.indices.sorted { a, b in
            let (x, y) = (scores[a], scores[b])
            if x.isNaN != y.isNaN { return x.isNaN }
            if !x.isNaN && x != y { return x > y }
            return a < b
        }
        var kept = [Array(0 ..< tokens)] + Array(repeating: [Int](), count: tubelets - 1)
        for index in ranked.prefix(extra).sorted() {
            kept[1 + index / tokens].append(index % tokens)
        }
        return kept
    }
}
