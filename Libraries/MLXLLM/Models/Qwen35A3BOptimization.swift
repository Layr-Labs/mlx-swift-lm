// Exact-artifact construction contract for the EigenLabs Qwen3.6 35B-A3B
// campaign. This file owns only load-time inspection and immutable route
// selection. Model forwards never parse configuration, read the environment,
// or recover from an ineligible optimized dispatch.

import Foundation
import MLX
import MLXLMCommon

public enum Qwen35A3BTargetPacking: Equatable, Sendable {
    case affine(bits: Int, groupSize: Int, routerBits: Int, routerGroupSize: Int)
}

public enum Qwen35A3BMTPPacking: Equatable, Sendable {
    case mxfp8(bits: Int, groupSize: Int)
}

public struct Qwen35A3BGeometry: Equatable, Sendable {
    public let hidden: Int
    public let experts: Int
    public let topK: Int
    public let expertIntermediate: Int
    public let sharedIntermediate: Int
    public let layers: Int
    public let recurrentLayers: Int
    public let fullAttentionLayers: Int

    public init(
        hidden: Int, experts: Int, topK: Int, expertIntermediate: Int,
        sharedIntermediate: Int, layers: Int, recurrentLayers: Int,
        fullAttentionLayers: Int
    ) {
        self.hidden = hidden
        self.experts = experts
        self.topK = topK
        self.expertIntermediate = expertIntermediate
        self.sharedIntermediate = sharedIntermediate
        self.layers = layers
        self.recurrentLayers = recurrentLayers
        self.fullAttentionLayers = fullAttentionLayers
    }
}

public enum Qwen35A3BArtifactError: Error, Equatable, CustomStringConvertible {
    case mismatch(field: String, expected: String, actual: String)
    case malformed(field: String)

    public var description: String {
        switch self {
        case .mismatch(let field, let expected, let actual):
            return "Qwen35 A3B artifact mismatch at \(field): expected \(expected), got \(actual)"
        case .malformed(let field):
            return "Qwen35 A3B artifact is missing or malformed at \(field)"
        }
    }
}

/// Model-free fixture used by construction tests. Production inspection builds
/// the same value from config.json before any optimized route can be installed.
struct Qwen35A3BArtifactFixture: Equatable, Sendable {
    var rootModelType: String
    var textModelType: String
    var hidden: Int
    var experts: Int
    var topK: Int
    var expertIntermediate: Int
    var sharedIntermediate: Int
    var layers: Int
    var recurrentLayers: Int
    var fullAttentionLayers: Int
    var targetMode: String
    var targetBits: Int
    var targetGroupSize: Int
    var routerBits: Int
    var routerGroupSize: Int
    var routerLayerEntries: Int
    var sharedGateLayerEntries: Int
    var mtpIncluded: Bool
    var mtpLayers: Int
    var mtpMode: String
    var mtpBits: Int
    var mtpGroupSize: Int

    static let eigenLabsRouter8 = Qwen35A3BArtifactFixture(
        rootModelType: "qwen3_5_moe", textModelType: "qwen3_5_moe_text",
        hidden: 2_048, experts: 256, topK: 8, expertIntermediate: 512,
        sharedIntermediate: 512, layers: 40, recurrentLayers: 30,
        fullAttentionLayers: 10, targetMode: "affine", targetBits: 4,
        targetGroupSize: 64, routerBits: 8, routerGroupSize: 64,
        routerLayerEntries: 40, sharedGateLayerEntries: 40,
        mtpIncluded: true, mtpLayers: 1, mtpMode: "mxfp8", mtpBits: 8,
        mtpGroupSize: 32)
}

public struct Qwen35A3BArtifactContract: Equatable, Sendable {
    public let target: Qwen35A3BTargetPacking
    public let mtp: Qwen35A3BMTPPacking
    public let geometry: Qwen35A3BGeometry
    public let targetReaderIdentity: String
    public let mtpReaderIdentity: String

    public var summary: String {
        "H\(geometry.hidden)/E\(geometry.experts)/K\(geometry.topK)/I"
            + "\(geometry.expertIntermediate)/L\(geometry.layers); target="
            + targetReaderIdentity + "; mtp=" + mtpReaderIdentity
    }

    static func inspect(fixture: Qwen35A3BArtifactFixture) throws -> Self {
        let expected = Qwen35A3BArtifactFixture.eigenLabsRouter8
        func require<T: Equatable>(
            _ field: String, _ actual: T, _ wanted: T
        ) throws {
            guard actual == wanted else {
                throw Qwen35A3BArtifactError.mismatch(
                    field: field, expected: String(describing: wanted),
                    actual: String(describing: actual))
            }
        }

        try require("model_type", fixture.rootModelType, expected.rootModelType)
        try require("text_config.model_type", fixture.textModelType, expected.textModelType)
        try require("text_config.hidden_size", fixture.hidden, expected.hidden)
        try require("text_config.num_experts", fixture.experts, expected.experts)
        try require("text_config.num_experts_per_tok", fixture.topK, expected.topK)
        try require(
            "text_config.moe_intermediate_size", fixture.expertIntermediate,
            expected.expertIntermediate)
        try require(
            "text_config.shared_expert_intermediate_size", fixture.sharedIntermediate,
            expected.sharedIntermediate)
        try require("text_config.num_hidden_layers", fixture.layers, expected.layers)
        try require("layer_types.recurrent", fixture.recurrentLayers, expected.recurrentLayers)
        try require(
            "layer_types.full_attention", fixture.fullAttentionLayers,
            expected.fullAttentionLayers)
        try require("quantization.mode", fixture.targetMode, expected.targetMode)
        try require("quantization.bits", fixture.targetBits, expected.targetBits)
        try require(
            "quantization.group_size", fixture.targetGroupSize, expected.targetGroupSize)
        try require("router.bits", fixture.routerBits, expected.routerBits)
        try require("router.group_size", fixture.routerGroupSize, expected.routerGroupSize)
        try require(
            "router.layer_entries", fixture.routerLayerEntries, expected.routerLayerEntries)
        try require(
            "shared_gate.layer_entries", fixture.sharedGateLayerEntries,
            expected.sharedGateLayerEntries)
        try require("mtplx_mtp.included", fixture.mtpIncluded, expected.mtpIncluded)
        try require("text_config.mtp_num_hidden_layers", fixture.mtpLayers, expected.mtpLayers)
        try require("mtp.mode", fixture.mtpMode, expected.mtpMode)
        try require("mtp.bits", fixture.mtpBits, expected.mtpBits)
        try require("mtp.group_size", fixture.mtpGroupSize, expected.mtpGroupSize)

        return Self(
            target: .affine(
                bits: fixture.targetBits, groupSize: fixture.targetGroupSize,
                routerBits: fixture.routerBits, routerGroupSize: fixture.routerGroupSize),
            mtp: .mxfp8(bits: fixture.mtpBits, groupSize: fixture.mtpGroupSize),
            geometry: Qwen35A3BGeometry(
                hidden: fixture.hidden, experts: fixture.experts, topK: fixture.topK,
                expertIntermediate: fixture.expertIntermediate,
                sharedIntermediate: fixture.sharedIntermediate, layers: fixture.layers,
                recurrentLayers: fixture.recurrentLayers,
                fullAttentionLayers: fixture.fullAttentionLayers),
            targetReaderIdentity: "affine-w4-g64",
            mtpReaderIdentity: "mxfp8-g32")
    }

    public static func inspect(configurationURL: URL) throws -> Self {
        let data = try Data(contentsOf: configurationURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let text = root["text_config"] as? [String: Any],
            let target = root["quantization"] as? [String: Any],
            let inline = root["mtplx_mtp"] as? [String: Any],
            let mtp = root["mtplx_mtp_quantization"] as? [String: Any],
            let layerTypes = text["layer_types"] as? [String]
        else {
            throw Qwen35A3BArtifactError.malformed(field: "config.json")
        }

        func string(_ table: [String: Any], _ key: String) throws -> String {
            guard let value = table[key] as? String else {
                throw Qwen35A3BArtifactError.malformed(field: key)
            }
            return value
        }
        func int(_ table: [String: Any], _ key: String) throws -> Int {
            guard let value = table[key] as? Int else {
                throw Qwen35A3BArtifactError.malformed(field: key)
            }
            return value
        }

        let layers = try int(text, "num_hidden_layers")
        var routerEntries = 0
        var sharedGateEntries = 0
        var routerBits: Int?
        var routerGroupSize: Int?
        for layer in 0 ..< layers {
            let gateKey = "language_model.model.layers.\(layer).mlp.gate"
            let sharedKey = "language_model.model.layers.\(layer).mlp.shared_expert_gate"
            if let gate = target[gateKey] as? [String: Any],
                let bits = gate["bits"] as? Int
            {
                routerEntries += 1
                routerBits = routerBits ?? bits
                routerGroupSize = routerGroupSize ?? (gate["group_size"] as? Int)
            }
            if target[sharedKey] is [String: Any] { sharedGateEntries += 1 }
        }

        return try inspect(fixture: Qwen35A3BArtifactFixture(
            rootModelType: try string(root, "model_type"),
            textModelType: try string(text, "model_type"),
            hidden: try int(text, "hidden_size"), experts: try int(text, "num_experts"),
            topK: try int(text, "num_experts_per_tok"),
            expertIntermediate: try int(text, "moe_intermediate_size"),
            sharedIntermediate: try int(text, "shared_expert_intermediate_size"),
            layers: layers,
            recurrentLayers: layerTypes.filter { $0 == "linear_attention" }.count,
            fullAttentionLayers: layerTypes.filter { $0 == "full_attention" }.count,
            targetMode: try string(target, "mode"), targetBits: try int(target, "bits"),
            targetGroupSize: try int(target, "group_size"),
            routerBits: routerBits ?? -1, routerGroupSize: routerGroupSize ?? 64,
            routerLayerEntries: routerEntries, sharedGateLayerEntries: sharedGateEntries,
            mtpIncluded: (inline["included"] as? Bool) ?? false,
            mtpLayers: try int(text, "mtp_num_hidden_layers"),
            mtpMode: try string(mtp, "mode"), mtpBits: try int(mtp, "bits"),
            mtpGroupSize: try int(mtp, "group_size")))
    }
}

public enum Qwen35A3BOptimizationProfile: String, Equatable, Sendable {
    case stock
    case prefill
    case decode
    case full
}

/// Selected by an integrator before constructing the exact checkpoint. Sparse
/// blocks bind their route closures during init; forwards never read this
/// value. The benchmark process constructs one model, so profile publication
/// is complete before any concurrent engine work begins.
public nonisolated(unsafe) var qwen35A3BConstructionProfile:
    Qwen35A3BOptimizationProfile = .stock

public enum Qwen35A3BTargetVerifyArithmetic: Equatable, Sendable {
    case rectangular
    case exactM1
}

/// Published before target construction. The model captures this choice in
/// immutable stored state; no verify layer reads process-global policy.
public nonisolated(unsafe) var qwen35A3BTargetVerifyArithmetic:
    Qwen35A3BTargetVerifyArithmetic = .rectangular

enum Qwen35A3BRouteLane: String, Equatable, Sendable {
    case disabled
    case stock
    case rightShaped = "right-shaped"
}

/// Construction result. Later kernel tasks replace each `rightShaped` lane's
/// implementation atomically before this table is attached to the model; the
/// table itself never mutates once published.
struct Qwen35A3BRouteTable: Equatable, Sendable {
    let profile: Qwen35A3BOptimizationProfile
    let contract: Qwen35A3BArtifactContract
    let prefill: Qwen35A3BRouteLane
    let targetDecode: Qwen35A3BRouteLane
    let mtpDecode: Qwen35A3BRouteLane

    static func install(
        contract: Qwen35A3BArtifactContract,
        profile: Qwen35A3BOptimizationProfile
    ) throws -> Self {
        switch profile {
        case .stock:
            return Self(
                profile: profile, contract: contract, prefill: .stock,
                targetDecode: .stock, mtpDecode: .disabled)
        case .prefill:
            return Self(
                profile: profile, contract: contract, prefill: .rightShaped,
                targetDecode: .stock, mtpDecode: .disabled)
        case .decode:
            return Self(
                profile: profile, contract: contract, prefill: .stock,
                targetDecode: .rightShaped, mtpDecode: .rightShaped)
        case .full:
            return Self(
                profile: profile, contract: contract, prefill: .rightShaped,
                targetDecode: .rightShaped, mtpDecode: .rightShaped)
        }
    }
}

// MARK: - Exact E256/K8 row-owned router finalizer

/// One threadgroup owns one complete BF16 probability row. The two-stage
/// tournament preserves MLX argPartition's ascending selected order and its
/// BF16 denominator accumulation. This primitive is unchecked by design: the
/// artifact contract and model construction route prove E256/K8/BF16 once.
private let qwen35A3BRowOwnedRouterKernel = MLXFast.metalKernel(
    name: "qwen35_a3b_row_owned_top8_bf16_exact",
    inputNames: ["probabilities"],
    outputNames: ["expert_ids", "route_scores"],
    source: """
        constexpr int N = 256;
        constexpr int TOPK = 8;
        constexpr int SIMD_GROUPS = 8;
        constexpr int LOCAL_CANDIDATES = SIMD_GROUPS * TOPK;

        uint row = threadgroup_position_in_grid.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;

        threadgroup float local_probabilities[LOCAL_CANDIDATES];
        threadgroup int local_indices[LOCAL_CANDIDATES];
        threadgroup float merged_probabilities[TOPK];
        threadgroup int merged_indices[TOPK];

        int expert = int(simd_gid) * 32 + int(lane);
        float candidate_probability = float(probabilities[row * N + expert]);
        int candidate_index = expert;

        _Pragma("unroll")
        for (int rank = 0; rank < TOPK; ++rank) {
            float winner_probability = simd_max(candidate_probability);
            float winner_index_value = simd_max(
                candidate_probability == winner_probability
                    ? float(candidate_index) : -1.0f);
            int winner_index = int(winner_index_value);
            if (lane == 0) {
                int destination = int(simd_gid) * TOPK + rank;
                local_probabilities[destination] = winner_probability;
                local_indices[destination] = winner_index;
            }
            if (candidate_index == winner_index) {
                candidate_probability = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_gid == 0) {
            int slot0 = int(lane);
            int slot1 = int(lane) + 32;
            float probability0 = local_probabilities[slot0];
            float probability1 = local_probabilities[slot1];
            int index0 = local_indices[slot0];
            int index1 = local_indices[slot1];

            _Pragma("unroll")
            for (int rank = 0; rank < TOPK; ++rank) {
                bool take1 = probability1 > probability0
                    || (probability1 == probability0 && index1 > index0);
                float lane_probability = take1 ? probability1 : probability0;
                int lane_index = take1 ? index1 : index0;
                float winner_probability = simd_max(lane_probability);
                float winner_index_value = simd_max(
                    lane_probability == winner_probability
                        ? float(lane_index) : -1.0f);
                int winner_index = int(winner_index_value);
                if (lane == 0) {
                    merged_probabilities[rank] = winner_probability;
                    merged_indices[rank] = winner_index;
                }
                if (lane_index == winner_index) {
                    if (take1) { probability1 = -INFINITY; }
                    else { probability0 = -INFINITY; }
                }
            }

            if (lane == 0) {
                bfloat rounded_denominator = bfloat(0.0f);
                _Pragma("unroll")
                for (int output = 0; output < TOPK; ++output) {
                    rounded_denominator = bfloat(
                        float(rounded_denominator)
                            + merged_probabilities[TOPK - 1 - output]);
                }
                _Pragma("unroll")
                for (int output = 0; output < TOPK; ++output) {
                    int source = TOPK - 1 - output;
                    int destination = int(row) * TOPK + output;
                    expert_ids[destination] = uint(merged_indices[source]);
                    route_scores[destination] = bfloat(
                        merged_probabilities[source]
                            / float(rounded_denominator));
                }
            }
        }
    """,
    ensureRowContiguous: true)

func qwen35A3BRowOwnedRoute(
    _ probabilities: MLXArray, rows: Int
) -> (expertIDs: MLXArray, routeScores: MLXArray) {
    let outputs = qwen35A3BRowOwnedRouterKernel(
        [probabilities.reshaped([rows, 256])],
        grid: (rows * 256, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [[rows, 8], [rows, 8]],
        outputDTypes: [.uint32, .bfloat16])
    let outputShape = Array(probabilities.shape.dropLast()) + [8]
    return (
        outputs[0].reshaped(outputShape),
        outputs[1].reshaped(outputShape))
}

typealias Qwen35A3BRouterFinalizer = (MLXArray) -> (
    expertIDs: MLXArray, routeScores: MLXArray
)

private func qwen35A3BStockRoute(
    _ probabilities: MLXArray, topK: Int, normalize: Bool
) -> (expertIDs: MLXArray, routeScores: MLXArray) {
    let kth = probabilities.dim(-1) - topK
    let ids = argPartition(probabilities, kth: kth, axis: -1)[.ellipsis, kth...]
    var scores = takeAlong(probabilities, ids, axis: -1)
    if normalize { scores = scores / scores.sum(axis: -1, keepDims: true) }
    return (ids, scores)
}

/// Bind either the unchanged stock finalizer or the exact row-owned decode
/// route. The only execution-time decision in the optimized closure is logical
/// M: rows 1...16 are the installed Metal lane; all wider values are the
/// explicit prefill route selected by construction.
func qwen35A3BRouterFinalizer(
    hidden: Int, experts: Int, topK: Int, normalize: Bool
) -> Qwen35A3BRouterFinalizer {
    let stock: Qwen35A3BRouterFinalizer = { probabilities in
        qwen35A3BStockRoute(
            probabilities, topK: topK, normalize: normalize)
    }
    guard (qwen35A3BConstructionProfile == .decode
        || qwen35A3BConstructionProfile == .full),
        hidden == 2_048, experts == 256, topK == 8, normalize
    else { return stock }

    return { probabilities in
        let rows = probabilities.size / 256
        if rows >= 1 && rows <= 16 {
            return qwen35A3BRowOwnedRoute(probabilities, rows: rows)
        }
        return stock(probabilities)
    }
}

private let qwen35A3BCombineKernel = MLXFast.metalKernel(
    name: "qwen35_a3b_combine_bf16_exact",
    inputNames: ["routed", "scores"],
    outputNames: ["combined"],
    source: """
        constexpr int TOPK = 8;
        constexpr int HIDDEN = 2048;
        uint output_index = thread_position_in_grid.x;
        uint row = output_index / HIDDEN;
        uint column = output_index - row * HIDDEN;
        bfloat accumulator = bfloat(0.0f);
        _Pragma("unroll")
        for (int expert = 0; expert < TOPK; ++expert) {
            uint routed_index = (row * TOPK + uint(expert)) * HIDDEN + column;
            uint score_index = row * TOPK + uint(expert);
            bfloat product = bfloat(
                float(routed[routed_index]) * float(scores[score_index]));
            accumulator = bfloat(float(accumulator) + float(product));
        }
        combined[output_index] = accumulator;
    """,
    ensureRowContiguous: true)

typealias Qwen35A3BExpertCombiner = (MLXArray, MLXArray) -> MLXArray

func qwen35A3BExpertCombiner(
    hidden: Int, topK: Int
) -> Qwen35A3BExpertCombiner {
    let stock: Qwen35A3BExpertCombiner = { routed, scores in
        weightedExpertSum(routed, scores.asType(routed.dtype))
    }
    guard (qwen35A3BConstructionProfile == .decode
        || qwen35A3BConstructionProfile == .full),
        hidden == 2_048, topK == 8
    else { return stock }

    return { routed, scores in
        let rows = routed.size / (8 * 2_048)
        guard rows == 1 || rows == 2 else { return stock(routed, scores) }
        let output = qwen35A3BCombineKernel(
            [routed, scores],
            grid: (rows * 2_048, 1, 1),
            threadGroup: (rows == 1 ? 128 : 64, 1, 1),
            outputShapes: [[rows, 2_048]],
            outputDTypes: [.bfloat16])[0]
        let shape = Array(routed.shape.dropLast(2)) + [2_048]
        return output.reshaped(shape)
    }
}
