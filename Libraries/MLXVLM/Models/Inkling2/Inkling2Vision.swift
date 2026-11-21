// Copyright © 2026 Eigen Labs.
//
// Inkling 2 vision tower: a ViT over 20 px patches with one class token per
// image, a learned 32x32 position grid resized bilinearly to each image, an
// axial 2D rotary over interleaved channel pairs, layer scale, and a 2x2
// merge projected to the text hidden size.
//
// Reference: transformers PR 51050 `modeling_inkling2.py`
// (`Inkling2VisionEmbeddings`, `Inkling2VisionRotaryEmbedding`,
// `Inkling2VisionAttention`, `Inkling2VisionEncoderLayer`,
// `Inkling2VisionMerger`, `Inkling2VisionModel`), cross-checked with vLLM PR
// 69028 `nvidia/vision.py` + `ops/vision.py`. Module keys follow the
// checkpoint (`model.visual.*`), so no renames are needed.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class Inkling2VisionAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "out_proj") var outputProjection: Linear

    let heads: Int
    let headDim: Int
    let scale: Float

    init(_ config: Inkling2VisionConfiguration) {
        heads = config.attentionHeads
        headDim = config.headDim
        scale = pow(Float(config.headDim), -0.5)
        let width = config.hiddenSize
        _queryProjection.wrappedValue = Linear(width, width, bias: true)
        _keyProjection.wrappedValue = Linear(width, width, bias: true)
        _valueProjection.wrappedValue = Linear(width, width, bias: true)
        _outputProjection.wrappedValue = Linear(width, width, bias: true)
        super.init()
    }

    /// `x`: one image's tokens `[S, width]`. `cos`/`sin`: `[S, 1, headDim]`
    /// float32. Every token attends every token of the same image.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let length = x.dim(0)
        var q = queryProjection(x).reshaped(length, heads, headDim)
        var k = keyProjection(x).reshaped(length, heads, headDim)
        let v = valueProjection(x).reshaped(length, heads, headDim)
        q = inkling2ApplyRotary(q, cos: cos, sin: sin)
        k = inkling2ApplyRotary(k, cos: cos, sin: sin)
        return outputProjection(attend(q, k, v))
    }

    /// Softmax attention of rotated `[S, heads, headDim]` queries and keys
    /// over the values, as `[S, heads * headDim]` before the output projection.
    func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        let output = MLXFast.scaledDotProductAttention(
            queries: q.transposed(1, 0, 2).expandedDimensions(axis: 0),
            keys: k.transposed(1, 0, 2).expandedDimensions(axis: 0),
            values: v.transposed(1, 0, 2).expandedDimensions(axis: 0),
            scale: scale, mask: .none)
        return output[0].transposed(1, 0, 2).reshaped(q.dim(0), -1)
    }
}

/// Rotates each interleaved pair `(x1, x2)` by its angle, in float32, and
/// returns the input dtype (transformers `apply_rotary_pos_emb_vision`).
func inkling2ApplyRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let dtype = x.dtype
    let values = x.asType(.float32)
    let pairs = values.reshaped(Array(x.shape.dropLast()) + [x.dim(-1) / 2, 2])
    let even = pairs[.ellipsis, 0 ..< 1]
    let odd = pairs[.ellipsis, 1 ..< 2]
    let rotated = concatenated([-odd, even], axis: -1).reshaped(x.shape)
    return (values * cos + rotated * sin).asType(dtype)
}

/// Exact (erf) GELU evaluated in float32 and rounded once to the input dtype,
/// as torch computes `F.gelu` on bf16 tensors.
func inkling2Gelu(_ x: MLXArray) -> MLXArray {
    gelu(x.asType(.float32)).asType(x.dtype)
}

/// LayerNorm evaluated in float32 and rounded once to the input dtype, as
/// torch computes `nn.LayerNorm` on bf16 tensors. MLX's bf16 layer norm rounds
/// three times: the normalized value, the weight product and the bias sum.
/// That made the norms 1.5x noisier than the reference (2x on the worst row),
/// and they were the only tower ops that were. One fused kernel reads the
/// low-precision row, keeps mean, variance and `(x - mean) * rstd * w + b` in
/// float32 and writes once; other dtypes and shapes take the float32 MLX path.
final class Inkling2VisionLayerNorm: LayerNorm {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let weight, let bias, x.dtype == .bfloat16 || x.dtype == .float16,
            x.ndim >= 1, x.dim(-1) == dimensions, dimensions <= 8192, x.size > 0
        {
            return Self.fused(x, weight: weight, bias: bias, eps: eps)
        }
        return MLXFast.layerNorm(
            x.asType(.float32), weight: weight?.asType(.float32), bias: bias?.asType(.float32),
            eps: eps
        ).asType(x.dtype)
    }

    static let threads = 128

    static func fused(_ x: MLXArray, weight: MLXArray, bias: MLXArray, eps: Float) -> MLXArray {
        let width = x.dim(-1)
        let rows = x.size / width
        return kernel(
            [x.reshaped(rows, width), weight, bias, ResidualRMSNorm.Constants.shared.epsilon(eps)],
            template: [("T", x.dtype), ("D", width), ("TG", threads)],
            grid: (threads * rows, 1, 1), threadGroup: (threads, 1, 1),
            outputShapes: [[rows, width]], outputDTypes: [x.dtype]
        )[0].reshaped(x.shape)
    }

    // One threadgroup per row, eight adjacent values per thread held in
    // registers: two-pass mean and variance in float32, then the affine map
    // in float32 and a single rounding to T.
    private static let kernel = MLXFast.metalKernel(
        name: "inkling2_vision_layer_norm_f32", inputNames: ["x", "w", "b", "eps"],
        outputNames: ["out"], source: source)

    private static let source = """
        constexpr uint R = 8;
        constexpr uint STEP = TG * R;
        constexpr uint CHUNKS = (uint(D) + STEP - 1) / STEP;
        constexpr uint GROUPS = TG / 32;
        const uint row = threadgroup_position_in_grid.x;
        const uint tid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint group = simdgroup_index_in_threadgroup;
        threadgroup float partial[GROUPS];
        const device T* src = x + size_t(row) * D;
        device T* dst = out + size_t(row) * D;

        float values[CHUNKS * R];
        float sum = 0;
        for (uint c = 0; c < CHUNKS; c++) {
          for (uint r = 0; r < R; r++) {
            const uint i = c * STEP + tid * R + r;
            values[c * R + r] = i < uint(D) ? static_cast<float>(src[i]) : 0;
            sum += values[c * R + r];
          }
        }
        sum = simd_sum(sum);
        if (lane == 0) { partial[group] = sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0;
        for (uint g = 0; g < GROUPS; g++) { total += partial[g]; }
        const float mean = total / D;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float squares = 0;
        for (uint c = 0; c < CHUNKS; c++) {
          for (uint r = 0; r < R; r++) {
            const uint i = c * STEP + tid * R + r;
            const float d = i < uint(D) ? values[c * R + r] - mean : 0;
            squares += d * d;
          }
        }
        squares = simd_sum(squares);
        if (lane == 0) { partial[group] = squares; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float variance = 0;
        for (uint g = 0; g < GROUPS; g++) { variance += partial[g]; }
        const float rstd = metal::precise::rsqrt(variance / D + eps[0]);

        for (uint c = 0; c < CHUNKS; c++) {
          for (uint r = 0; r < R; r++) {
            const uint i = c * STEP + tid * R + r;
            if (i < uint(D)) {
              const float y = (values[c * R + r] - mean) * rstd;
              dst[i] = static_cast<T>(y * static_cast<float>(w[i]) + static_cast<float>(b[i]));
            }
          }
        }
        """
}

final class Inkling2VisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "c_fc") var fc: Linear
    @ModuleInfo(key: "c_proj") var projection: Linear

    init(_ config: Inkling2VisionConfiguration) {
        _fc.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        _projection.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projection(inkling2Gelu(fc(x)))
    }
}

final class Inkling2VisionLayerScale: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(_ width: Int) {
        _gamma.wrappedValue = MLXArray.ones([width])
        super.init()
    }
}

final class Inkling2VisionLayer: Module {
    @ModuleInfo(key: "ln_1") var norm1: Inkling2VisionLayerNorm
    @ModuleInfo(key: "self_attn") var attention: Inkling2VisionAttention
    @ModuleInfo(key: "ln_2") var norm2: Inkling2VisionLayerNorm
    @ModuleInfo(key: "mlp") var mlp: Inkling2VisionMLP
    @ModuleInfo(key: "ls_1") var scale1: Inkling2VisionLayerScale
    @ModuleInfo(key: "ls_2") var scale2: Inkling2VisionLayerScale

    init(_ config: Inkling2VisionConfiguration) {
        _norm1.wrappedValue = Inkling2VisionLayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        _attention.wrappedValue = Inkling2VisionAttention(config)
        _norm2.wrappedValue = Inkling2VisionLayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        _mlp.wrappedValue = Inkling2VisionMLP(config)
        _scale1.wrappedValue = Inkling2VisionLayerScale(config.hiddenSize)
        _scale2.wrappedValue = Inkling2VisionLayerScale(config.hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + scale1.gamma * attention(norm1(x), cos: cos, sin: sin)
        return h + scale2.gamma * mlp(norm2(h))
    }
}

final class Inkling2VisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    /// The learned `[grid * grid, width]` position grid, row-major.
    @ParameterInfo(key: "position_embedding") var positionEmbedding: MLXArray
    @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
    @ParameterInfo(key: "cls_position_embedding") var classPositionEmbedding: MLXArray

    init(_ config: Inkling2VisionConfiguration) {
        let width = config.hiddenSize
        _patchEmbedding.wrappedValue = Linear(config.patchFeatures, width, bias: false)
        _positionEmbedding.wrappedValue = MLXArray.zeros(
            [config.positionGridSize * config.positionGridSize, width])
        _classEmbedding.wrappedValue = MLXArray.zeros([width])
        _classPositionEmbedding.wrappedValue = MLXArray.zeros([width])
        super.init()
    }
}

final class Inkling2VisionBackbone: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Inkling2VisionEmbeddings
    @ModuleInfo(key: "ln_pre") var preNorm: Inkling2VisionLayerNorm
    @ModuleInfo(key: "layers") var layers: [Inkling2VisionLayer]

    init(_ config: Inkling2VisionConfiguration) {
        _embeddings.wrappedValue = Inkling2VisionEmbeddings(config)
        _preNorm.wrappedValue = Inkling2VisionLayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in Inkling2VisionLayer(config) }
        super.init()
    }
}

/// Host-side geometry of one image in the tower: native patch coordinates
/// in sequence order, the bilinear position-grid taps and their weights.
struct Inkling2VisionGeometry: Equatable {
    /// Native patch rows and columns (`2 * token rows`, `2 * token columns`).
    let patchRows: Int
    let patchColumns: Int
    /// `(y, x)` per patch, in the processor's merge order: token-major raster,
    /// then the `merge x merge` block of that token in raster order.
    let coordinates: [(y: Int, x: Int)]

    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.patchRows == rhs.patchRows && lhs.patchColumns == rhs.patchColumns
            && lhs.coordinates.map { [$0.y, $0.x] } == rhs.coordinates.map { [$0.y, $0.x] }
    }

    init(tokenRows: Int, tokenColumns: Int, mergeSize: Int) {
        patchRows = tokenRows * mergeSize
        patchColumns = tokenColumns * mergeSize
        var coordinates: [(y: Int, x: Int)] = []
        coordinates.reserveCapacity(patchRows * patchColumns)
        for row in 0 ..< tokenRows {
            for column in 0 ..< tokenColumns {
                for dy in 0 ..< mergeSize {
                    for dx in 0 ..< mergeSize {
                        coordinates.append((row * mergeSize + dy, column * mergeSize + dx))
                    }
                }
            }
        }
        self.coordinates = coordinates
    }

    /// Bilinear taps for output index `index` of `target` on a `source`-long
    /// axis, as transformers `_interpolation_axis_taps_weights` (bilinear,
    /// `align_corners=False`, border padding): float32 half-pixel source
    /// coordinate, hat weights, out-of-range taps clamped to the edge.
    static func bilinearTaps(index: Int, target: Int, source: Int) -> (
        Int, Int, Float, Float
    ) {
        let coordinate = (Float(index) + 0.5) * Float(source) / Float(target) - 0.5
        let floor = coordinate.rounded(.down)
        let base = Int(floor)
        let lower = min(max(base, 0), source - 1)
        let upper = min(max(base + 1, 0), source - 1)
        let lowerWeight = max(1 - abs(coordinate - floor), 0)
        let upperWeight = max(1 - abs(coordinate - floor - 1), 0)
        return (lower, upper, lowerWeight, upperWeight)
    }

    /// Four `(grid index, weight)` taps per patch, flattened `[patches * 4]`,
    /// as the outer product of the row and column taps.
    func positionTaps(gridSize: Int) -> (indices: [Int32], weights: [Float]) {
        let rowTaps = (0 ..< patchRows).map {
            Self.bilinearTaps(index: $0, target: patchRows, source: gridSize)
        }
        let columnTaps = (0 ..< patchColumns).map {
            Self.bilinearTaps(index: $0, target: patchColumns, source: gridSize)
        }
        var indices: [Int32] = []
        var weights: [Float] = []
        indices.reserveCapacity(coordinates.count * 4)
        weights.reserveCapacity(coordinates.count * 4)
        for (y, x) in coordinates {
            let (y0, y1, wy0, wy1) = rowTaps[y]
            let (x0, x1, wx0, wx1) = columnTaps[x]
            indices += [
                Int32(y0 * gridSize + x0), Int32(y0 * gridSize + x1),
                Int32(y1 * gridSize + x0), Int32(y1 * gridSize + x1),
            ]
            weights += [wy0 * wx0, wy0 * wx1, wy1 * wx0, wy1 * wx1]
        }
        return (indices, weights)
    }

    /// Rotary positions `(y + 1, x + 1)` per sequence row; the class token
    /// in front sits at `(0, 0)`.
    func rotaryPositions() -> [Int32] {
        var positions: [Int32] = [0, 0]
        positions.reserveCapacity(2 * (coordinates.count + 1))
        for (y, x) in coordinates {
            positions += [Int32(y + 1), Int32(x + 1)]
        }
        return positions
    }
}

/// The Inkling 2 vision encoder plus its merge projector
/// (`model.visual.{backbone, proj_in, proj_out}`).
public final class Inkling2VisionModel: Module {
    @ModuleInfo(key: "backbone") var backbone: Inkling2VisionBackbone
    @ModuleInfo(key: "proj_in") var projectionIn: Linear
    @ModuleInfo(key: "proj_out") var projectionOut: Linear

    public let configuration: Inkling2VisionConfiguration
    /// `[headDim / 4]` axial inverse frequencies, shared by both axes.
    private let inverseFrequencies: [Float]

    public init(_ configuration: Inkling2VisionConfiguration) {
        self.configuration = configuration
        let merged = configuration.mergeSize * configuration.mergeSize * configuration.hiddenSize
        _backbone.wrappedValue = Inkling2VisionBackbone(configuration)
        _projectionIn.wrappedValue = Linear(
            merged, configuration.projectionHiddenSize, bias: configuration.projectorBias)
        _projectionOut.wrappedValue = Linear(
            configuration.projectionHiddenSize, configuration.textHiddenSize,
            bias: configuration.projectorBias)
        let spatial = configuration.headDim / 2
        inverseFrequencies = stride(from: 0, to: spatial, by: 2).map {
            1 / pow(configuration.ropeTheta, Float($0) / Float(spatial))
        }
        super.init()
    }

    /// The dtype the tower computes in (its patch-embedding weight dtype).
    public var computeDType: DType { backbone.embeddings.patchEmbedding.weight.dtype }

    /// Validates the processor output before any graph is built: patches
    /// `[sum(rows * cols), side, side, 3]` with one temporal step per grid.
    public func validate(pixels: MLXArray, grids: [THW]) throws {
        let side = configuration.tokenSide
        guard !grids.isEmpty, pixels.ndim == 4, pixels.dim(1) == side, pixels.dim(2) == side,
            pixels.dim(3) == 3
        else {
            throw VLMError.processing("Inkling 2 image patches must be [N, \(side), \(side), 3]")
        }
        var total = 0
        for grid in grids {
            guard grid.t == 1, grid.h > 0, grid.w > 0 else {
                throw VLMError.processing("Inkling 2 vision accepts still images only (t = 1)")
            }
            let (count, overflow) = grid.h.multipliedReportingOverflow(by: grid.w)
            guard !overflow else { throw VLMError.processing("Inkling 2 image grid overflow") }
            total += count
        }
        guard total == pixels.dim(0) else {
            throw VLMError.processing(
                "Inkling 2 image grids cover \(total) tokens but \(pixels.dim(0)) patches arrived")
        }
    }

    /// One `[rows * cols, textHidden]` feature array per image, in input order.
    /// Each image runs through the tower on its own (attention never crosses
    /// images), which also bounds the attention workspace to one image.
    public func imageFeatures(pixels: MLXArray, grids: [THW]) throws -> [MLXArray] {
        try validate(pixels: pixels, grids: grids)
        var start = 0
        return grids.map { grid in
            let count = grid.h * grid.w
            defer { start += count }
            return encode(
                pixels[start ..< (start + count)], tokenRows: grid.h, tokenColumns: grid.w)
        }
    }

    /// Projected features of one image whose `[rows * cols, side, side, 3]`
    /// patches arrive in raster token order.
    func encode(_ pixels: MLXArray, tokenRows: Int, tokenColumns: Int) -> MLXArray {
        let merge = configuration.mergeSize
        let width = configuration.hiddenSize
        let hidden = backboneStates(pixels, tokenRows: tokenRows, tokenColumns: tokenColumns)
        let merged = hidden[1...].reshaped(tokenRows * tokenColumns, merge * merge * width)
        return projectionOut(inkling2Gelu(projectionIn(merged)))
    }

    /// Last-layer states `[1 + patches, width]` of one image: the class token,
    /// then the patches in merge order (transformers `last_hidden_state`).
    func backboneStates(_ pixels: MLXArray, tokenRows: Int, tokenColumns: Int) -> MLXArray {
        let geometry = Inkling2VisionGeometry(
            tokenRows: tokenRows, tokenColumns: tokenColumns, mergeSize: configuration.mergeSize)
        var hidden = backbone.embeddings.patchEmbedding(patchFeatures(pixels, geometry))
        hidden = hidden + positionTerm(geometry, dtype: hidden.dtype)
        hidden = backbone.preNorm(
            concatenated([classToken(dtype: hidden.dtype), hidden], axis: 0))

        let (cos, sin) = rotaryTables(geometry)
        for layer in backbone.layers {
            hidden = layer(hidden, cos: cos, sin: sin)
        }
        return hidden
    }

    /// `[patches, patchFeatures]` in the compute dtype: the `[tokens, m*p, m*p,
    /// 3]` pixels as each token's merge block in raster order, each patch
    /// flattened as (py, px, channel).
    func patchFeatures(_ pixels: MLXArray, _ geometry: Inkling2VisionGeometry) -> MLXArray {
        let merge = configuration.mergeSize
        let patch = configuration.patchSize
        return
            pixels.asType(computeDType)
            .reshaped(-1, merge, patch, merge, patch, 3)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(geometry.coordinates.count, configuration.patchFeatures)
    }

    /// The position grid resized to the image: four bilinear taps per patch
    /// summed in float32, rounded once to `dtype`.
    func positionTerm(_ geometry: Inkling2VisionGeometry, dtype: DType) -> MLXArray {
        let patches = geometry.coordinates.count
        let taps = geometry.positionTaps(gridSize: configuration.positionGridSize)
        let table = backbone.embeddings.positionEmbedding[MLXArray(taps.indices)]
            .reshaped(patches, 4, configuration.hiddenSize).asType(.float32)
        let tapWeights = MLXArray(taps.weights).reshaped(patches, 4, 1)
        return (table * tapWeights).sum(axis: 1).asType(dtype)
    }

    /// The class token row `[1, width]` in `dtype`.
    func classToken(dtype: DType) -> MLXArray {
        let embeddings = backbone.embeddings
        return (embeddings.classEmbedding + embeddings.classPositionEmbedding)
            .asType(dtype).reshaped(1, configuration.hiddenSize)
    }

    /// `[S, 1, headDim]` cos/sin over interleaved pairs: the first half of the
    /// pairs turns with the column (x + 1), the second half with the row.
    func rotaryTables(_ geometry: Inkling2VisionGeometry) -> (MLXArray, MLXArray) {
        let rows = geometry.coordinates.count + 1
        let positions = MLXArray(geometry.rotaryPositions()).reshaped(rows, 2).asType(.float32)
        let frequencies = MLXArray(inverseFrequencies).reshaped(1, 1, -1)
        let angles = positions.expandedDimensions(axis: -1) * frequencies
        // [S, 2(y, x), F] -> [S, 2(x, y), F] -> [S, 2F] -> pairs repeated twice.
        let ordered = concatenated([angles[0..., 1, 0...], angles[0..., 0, 0...]], axis: -1)
        let interleaved = repeated(ordered.expandedDimensions(axis: -1), count: 2, axis: -1)
            .reshaped(rows, 1, configuration.headDim)
        return (MLX.cos(interleaved), MLX.sin(interleaved))
    }
}

/// Replaces the placeholder rows of `textEmbeddings` (`[1, L, hidden]`, the
/// normed text embeddings) with the image features, in prompt order
/// (`Inkling2Model.forward`'s `masked_scatter`). Image rows are not normed.
/// Throws unless the placeholder count equals the total feature rows.
public func inkling2SpliceImageFeatures(
    textEmbeddings: MLXArray, tokens: [Int], imageTokenId: Int, features: [MLXArray]
) throws -> MLXArray {
    guard textEmbeddings.ndim == 3, textEmbeddings.dim(0) == 1,
        textEmbeddings.dim(1) == tokens.count
    else {
        throw VLMError.processing("Inkling 2 text embeddings must be [1, prompt, hidden]")
    }
    let rows = features.reduce(0) { $0 + $1.dim(0) }
    let placeholders = tokens.reduce(0) { $0 + ($1 == imageTokenId ? 1 : 0) }
    guard placeholders == rows,
        features.allSatisfy({ $0.ndim == 2 && $0.dim(1) == textEmbeddings.dim(2) })
    else {
        throw VLMError.processing(
            "Image features and image tokens do not match: tokens \(placeholders), features \(rows)"
        )
    }
    guard rows > 0 else { return textEmbeddings }
    let visual = concatenated(features, axis: 0).asType(textEmbeddings.dtype)
    var pieces: [MLXArray] = []
    var cursor = 0
    var start = 0
    while start < tokens.count {
        let isImage = tokens[start] == imageTokenId
        var end = start + 1
        while end < tokens.count && (tokens[end] == imageTokenId) == isImage { end += 1 }
        if isImage {
            pieces.append(visual[cursor ..< (cursor + end - start)].expandedDimensions(axis: 0))
            cursor += end - start
        } else {
            pieces.append(textEmbeddings[0..., start ..< end, 0...])
        }
        start = end
    }
    return concatenated(pieces, axis: 1)
}
