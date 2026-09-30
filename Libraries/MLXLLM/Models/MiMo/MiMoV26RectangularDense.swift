// Copyright © 2026 Eigen Labs.
// Opt-in scalar-shape candidate; exact full-state qualification is pending.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum MiMoV26RectangularDense {
    static var enabledByEnvironment: Bool {
        enabled(environment: ProcessInfo.processInfo.environment)
    }

    static func enabled(environment: [String: String]) -> Bool {
        environment["DARKBLOOM_MIMO_RECTANGULAR_SCALAR_DENSE"] == "1"
    }

    /// Scalar-dense rows use the row-exact multi-row affine kernel where it
    /// applies; exact `0` / `false` / `no` / `off` keeps one matmul per row.
    static let rowExactProjectionEnabled = MiMoV26DecodeDefaults.isEnabled(
        "DARKBLOOM_MIMO_ROW_EXACT_PROJECTION")

    static func eligible(
        shape: [Int], rectangularCacheFlags: [Bool],
        requested: Bool, fusedNorms: Bool
    ) -> Bool {
        requested && !fusedNorms && shape.count == 2 && shape[0] == 1
            && (2 ... 4).contains(shape[1]) && !rectangularCacheFlags.isEmpty
            && rectangularCacheFlags.allSatisfy { $0 }
    }

    static func eligible(
        tokens: MLXArray, caches: [any CBv2AttendingLayerCache],
        requested: Bool, fusedNorms: Bool
    ) -> Bool {
        eligible(
            shape: tokens.shape,
            rectangularCacheFlags: caches.map {
                guard let cache = $0 as? any CBv2MTPRectangularSerializing else { return false }
                return cache.mtpSerializesRectangularAttention
                    || cache.mtpBatchesRectangularAttention
            }, requested: requested, fusedNorms: fusedNorms)
    }

    static func supports(_ x: MLXArray) -> Bool {
        x.ndim == 3 && x.dim(0) == 1 && (2 ... 4).contains(x.dim(1))
            && x.dim(2) > 0 && x.dim(2) <= 16384
            && (x.dtype == .bfloat16 || x.dtype == .float16)
    }

    private static func rows(_ x: MLXArray, _ body: (MLXArray) -> MLXArray) -> MLXArray {
        // Preserve scalar [1,1,D] rank and request row-contiguous operands.
        // Already contiguous slices may alias their whole input backing; no
        // compact-allocation or early-release claim follows from contiguous().
        // No weight conversion or host readback.
        let values = (0 ..< x.dim(1)).map { position in
            body(x[0..., position ..< (position + 1), 0...].contiguous())
        }
        return concatenated(values, axis: 1)
    }

    static func projection(_ layer: Linear, _ x: MLXArray, enabled: Bool) -> MLXArray {
        guard enabled && supports(x) else { return layer(x) }
        if let output = rowExact(layer, x) { return output }
        return rows(x) { layer($0) }
    }

    private static func rowExact(_ layer: Linear, _ x: MLXArray) -> MLXArray? {
        guard rowExactProjectionEnabled, let quantized = layer as? QuantizedLinear else {
            return nil
        }
        return MiMoV26RowExactProjection.apply(quantized, x)
    }

    static func readout(_ target: MiMoV26TextModel, _ x: MLXArray, enabled: Bool) -> MLXArray {
        func project(_ value: MLXArray) -> MLXArray {
            target.lmHead.map { $0(value) } ?? target.model.embedTokens.asLinear(value)
        }
        guard enabled && supports(x) else { return project(x) }
        if let head = target.lmHead, let output = rowExact(head, x) { return output }
        return rows(x, project)
    }

    static func mlp(_ layer: any UnaryLayer, _ x: MLXArray, enabled: Bool) -> MLXArray {
        guard enabled && supports(x) else { return layer(x) }
        if let dense = layer as? MiMoV26DenseMLP {
            // Keep the three dense projections and their activation rounding
            // on the scalar path. This is not applied to expert projections.
            if let gate = rowExact(dense.gateProj, x), let up = rowExact(dense.upProj, x),
                let output = rowExact(dense.downProj, silu(gate) * up)
            {
                return output
            }
            return rows(x) { dense($0) }
        }
        if let moe = layer as? MiMoV26MoE {
            return moe.forwardWithWeightedReductionRoute(x, rowLocalRouter: true).output
        }
        return layer(x)
    }
}

extension MiMoV26RectangularDense {
    /// Additive source-visible graph envelope, not an MLX physical-peak claim.
    /// Bound BOTH a predecessor and successor, all layers and every independent
    /// row allocation. Do not subtract the ordinary target/assistant/OS reserve.
    static func scratchSpec(_ target: MiMoV26TextModel)
        -> MiMoV26RectangularDenseScratchSpec?
    {
        let c = target.configuration
        guard target.activationDType == .bfloat16 || target.activationDType == .float16,
            (1 ... 48).contains(c.numHiddenLayers),
            target.model.layers.count == c.numHiddenLayers,
            (1 ... 16384).contains(c.hiddenSize),
            (1 ... 65536).contains(c.intermediateSize),
            (1 ... 262144).contains(c.vocabularySize),
            (1 ... 1024).contains(c.routedExpertCount)
        else { return nil }
        var buffers: [CBv2MTPFixedBufferSpec] = []
        func append(_ elements: Int, _ count: Int) -> Bool {
            let bytes = elements.multipliedReportingOverflow(by: 4)
            let copies = count.multipliedReportingOverflow(by: 2)  // two live forwards
            guard elements > 0, count > 0, !bytes.overflow, !copies.overflow else { return false }
            buffers.append(
                .init(logicalBytes: bytes.partialValue, allocationCount: copies.partialValue))
            return true
        }
        func projection(input: Int, output: Int) -> Bool {
            guard (1 ... 16384).contains(input), (1 ... 262144).contains(output) else {
                return false
            }
            // Worst L=4: compact inputs, independent outputs, final concat.
            // Four-byte pricing conservatively includes native two-byte arrays.
            return append(input, 4) && append(output, 4) && append(4 * output, 1)
        }
        for layer in target.model.layers {
            let a = layer.selfAttention
            let g = a.geometry
            let q = g.queryHeads.multipliedReportingOverflow(by: g.headDim)
            let k = g.keyValueHeads.multipliedReportingOverflow(by: g.headDim)
            let v = g.keyValueHeads.multipliedReportingOverflow(by: g.valueHeadDim)
            let o = g.queryHeads.multipliedReportingOverflow(by: g.valueHeadDim)
            guard !q.overflow, !k.overflow, !v.overflow, !o.overflow,
                a.qProj.shape == (q.partialValue, c.hiddenSize),
                a.kProj.shape == (k.partialValue, c.hiddenSize),
                a.vProj.shape == (v.partialValue, c.hiddenSize),
                a.oProj.shape == (c.hiddenSize, o.partialValue),
                projection(input: c.hiddenSize, output: q.partialValue),
                projection(input: c.hiddenSize, output: k.partialValue),
                projection(input: c.hiddenSize, output: v.partialValue),
                projection(input: o.partialValue, output: c.hiddenSize)
            else { return nil }
            if let dense = layer.mlp as? MiMoV26DenseMLP {
                guard dense.gateProj.shape == (c.intermediateSize, c.hiddenSize),
                    dense.upProj.shape == (c.intermediateSize, c.hiddenSize),
                    dense.downProj.shape == (c.hiddenSize, c.intermediateSize),
                    // Row compact; gate, SiLU, up, product; down + concat.
                    append(c.hiddenSize, 4), append(c.intermediateSize, 16),
                    append(c.hiddenSize, 4), append(4 * c.hiddenSize, 1)
                else { return nil }
            } else if let moe = layer.mlp as? MiMoV26MoE {
                guard moe.gate.weight.shape == [c.routedExpertCount, c.hiddenSize],
                    moe.gate.correctionBias.shape == [c.routedExpertCount]
                else { return nil }
                let matrix = c.routedExpertCount.multipliedReportingOverflow(by: c.hiddenSize)
                guard !matrix.overflow,
                    // ONE shared matrix per forward; price both possible casts,
                    // even when the first is an identity for this checkpoint.
                    append(matrix.partialValue, 2),
                    append(c.hiddenSize, 12),  // compact + two casts per row
                    append(c.routedExpertCount, 4 * 32),  // per-row logits, selection/group/top-k roots
                    append(4 * c.routedExpertCount, 2),  // indices + weights concat, topK <= experts
                    append(4, 4 * 16)
                else { return nil }
                // Expert gather, SiLU/product and weighted reduction remain
                // bulk, unchanged. Their baseline reserve is never discounted.
            } else {
                return nil
            }
        }
        let readout = target.lmHead?.shape ?? target.model.embedTokens.shape
        guard readout == (c.vocabularySize, c.hiddenSize),
            projection(input: c.hiddenSize, output: c.vocabularySize)
        else { return nil }
        // Bounded Swift lists/graph handles/metadata; not a native allocator cap.
        let host = 2 * (128 * 1024 + c.numHiddenLayers * 4 * 2048)
        return .init(buffers: buffers, hostBytes: host)
    }
}
