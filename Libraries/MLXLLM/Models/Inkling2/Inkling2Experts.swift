// Inkling 2 routed experts with the checkpoint's native NVFP4 / MXFP8 banks.
// Reference: transformers PR 51050 `Inkling2LoopSparseMoeBlock`,
// `Inkling2LoopDequantizeExperts` (Apache-2.0).
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Storage format of one expert bank, as found in the checkpoint.
public enum Inkling2ExpertFormat: String, Sendable, Equatable {
    /// E2M1 pairs (even element in the low nibble), E4M3 scales per 16 values
    /// and a per-expert FP32 factor `amax / denominator`.
    case nvfp4
    /// E4M3 values with E8M0 scales per 32 values.
    case mxfp8
    /// Dense `[experts, out, in]` weights (tests and dequantized exports).
    case dense
}

/// One projection over all experts: `y[..., e] = x @ W[e].T` for the gathered `e`.
class Inkling2ExpertBank: Module {
    let experts: Int
    let inputDimensions: Int
    let outputDimensions: Int
    var format: Inkling2ExpertFormat { .dense }

    init(experts: Int, inputDimensions: Int, outputDimensions: Int) {
        self.experts = experts
        self.inputDimensions = inputDimensions
        self.outputDimensions = outputDimensions
    }

    /// `x`: `[..., rows, in]`, `indices` broadcast against `x`'s batch dims.
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sorted: Bool) -> MLXArray {
        fatalError("abstract")
    }
}

final class Inkling2DenseExpertBank: Inkling2ExpertBank {
    let weight: MLXArray

    override init(experts: Int, inputDimensions: Int, outputDimensions: Int) {
        weight = MLXArray.zeros([experts, outputDimensions, inputDimensions])
        super.init(
            experts: experts, inputDimensions: inputDimensions, outputDimensions: outputDimensions)
    }

    override func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sorted: Bool) -> MLXArray {
        gatherMM(x, weight.swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: sorted)
    }
}

/// Packed banks kept exactly as stored: `weight` is the checkpoint bytes viewed
/// as `uint32` (`[E, out, in * bits / 32]`), `scales` the FP8 block scales
/// (`[E, out, in / group]` uint8). NVFP4 adds `global_scale` `[E]` FP32 in
/// MLX's convention (`gather_qmm` multiplies by `global_scale / (448 * 6)`).
final class Inkling2QuantizedExpertBank: Inkling2ExpertBank {
    let weight: MLXArray
    let scales: MLXArray
    @ModuleInfo(key: "global_scale") var globalScale: MLXArray?
    let mode: QuantizationMode
    let groupSize: Int
    let bits: Int

    init(format: Inkling2ExpertFormat, experts: Int, inputDimensions: Int, outputDimensions: Int) {
        precondition(format != .dense, "dense banks use Inkling2DenseExpertBank")
        mode = format == .nvfp4 ? .nvfp4 : .mxfp8
        groupSize = format == .nvfp4 ? 16 : 32
        bits = format == .nvfp4 ? 4 : 8
        weight = MLXArray.zeros(
            [experts, outputDimensions, inputDimensions * bits / 32], dtype: .uint32)
        scales = MLXArray.zeros(
            [experts, outputDimensions, inputDimensions / groupSize], dtype: .uint8)
        _globalScale.wrappedValue =
            format == .nvfp4 ? MLXArray.ones([experts], dtype: .float32) : nil
        super.init(
            experts: experts, inputDimensions: inputDimensions, outputDimensions: outputDimensions)
    }

    override var format: Inkling2ExpertFormat { mode == .nvfp4 ? .nvfp4 : .mxfp8 }

    override func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sorted: Bool) -> MLXArray {
        gatherQuantizedMM(
            x, weight, scales: scales, biases: nil, rhsIndices: indices, transpose: true,
            groupSize: groupSize, bits: bits, mode: mode, globalScale: globalScale,
            sortedIndices: sorted)
    }
}

/// Diagnostics for oracle checks.
public enum Inkling2Diagnostics {
    /// Receives each router call's selected expert indices `[N, k]` (unsorted).
    @TaskLocal public static var routerIndices: (@Sendable (MLXArray) -> Void)? = nil
    /// Receives each router call's selection scores `[N, experts]` (sigmoid + bias).
    @TaskLocal public static var routerSelection: (@Sendable (MLXArray) -> Void)? = nil
    /// Profiling: called at section boundaries of each layer application with
    /// the arrays that end the section (a profiler evaluates them and times it).
    @TaskLocal public static var section: (@Sendable (String, [MLXArray]) -> Void)? = nil
    /// Profiling of engine-driven steps (the engine runs the model off the
    /// caller's task, so this is process-wide): called with the step kind
    /// ("prefill" / "forward") and its token count before the graph is built.
    nonisolated(unsafe) public static var stepHook: ((String, Int) -> Void)? = nil

    /// Like `stepHook`, for one model's steps only, so tests running side by
    /// side each see their own model's steps.
    public static func setStepHook(for model: AnyObject, _ hook: ((String, Int) -> Void)?) {
        modelHooksLock.withLock { modelHooks[ObjectIdentifier(model)] = hook }
    }

    nonisolated(unsafe) private static var modelHooks: [ObjectIdentifier: (String, Int) -> Void] =
        [:]
    private static let modelHooksLock = NSLock()

    static func step(_ model: AnyObject, _ kind: String, _ length: Int) {
        stepHook?(kind, length)
        modelHooksLock.withLock { modelHooks[ObjectIdentifier(model)] }?(kind, length)
    }
}

/// Sigmoid scores, `gate.bias`-corrected top-k selection, renormalized weights
/// times `route_scale`; all FP32 (`Inkling2LoopTopKRouter`).
final class Inkling2Router: Module {
    let weight: MLXArray
    let bias: MLXArray
    let topK: Int
    let routeScale: Float

    init(hidden: Int, experts: Int, topK: Int, routeScale: Float) {
        weight = MLXArray.zeros([experts, hidden], dtype: .float32)
        bias = MLXArray.zeros([experts], dtype: .float32)
        self.topK = topK
        self.routeScale = routeScale
    }

    /// `x`: `[N, hidden]`. Returns expert indices `[N, k]` (uint32) and weights `[N, k]` FP32.
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = Inkling2RowExact.fp32Rows(x, weight)
        let scores = sigmoid(logits)
        let selection = scores + bias.asType(.float32)
        let k = topK
        let indices = argPartition(-selection, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        let selected = takeAlong(scores, indices, axis: -1)
        let weights = selected / (selected.sum(axis: -1, keepDims: true) + 1e-20) * routeScale
        Inkling2Diagnostics.routerIndices?(indices)
        Inkling2Diagnostics.routerSelection?(selection)
        return (indices.asType(.uint32), weights)
    }
}

/// Gate/up bank `w13` in the checkpoint's row order (row `2i` = gate `i`,
/// row `2i + 1` = up `i`) and down bank `w2`.
final class Inkling2Experts: Module {
    @ModuleInfo(key: "w13") var gateUp: Inkling2ExpertBank
    @ModuleInfo(key: "w2") var down: Inkling2ExpertBank
    let intermediate: Int
    let hidden: Int
    let experts: Int

    init(hidden: Int, intermediate: Int, experts: Int, format: Inkling2ExpertFormat) {
        self.intermediate = intermediate
        self.hidden = hidden
        self.experts = experts
        _gateUp.wrappedValue = Self.bank(format, experts, hidden, 2 * intermediate)
        _down.wrappedValue = Self.bank(format, experts, intermediate, hidden)
    }

    static func bank(_ format: Inkling2ExpertFormat, _ experts: Int, _ input: Int, _ output: Int)
        -> Inkling2ExpertBank
    {
        switch format {
        case .dense:
            return Inkling2DenseExpertBank(
                experts: experts, inputDimensions: input, outputDimensions: output)
        case .nvfp4, .mxfp8:
            return Inkling2QuantizedExpertBank(
                format: format, experts: experts, inputDimensions: input, outputDimensions: output)
        }
    }

    var format: Inkling2ExpertFormat { gateUp.format }

    /// Replace both banks with empty banks of `format` before a checkpoint update.
    func install(format: Inkling2ExpertFormat) {
        guard format != self.format else { return }
        update(
            modules: ModuleChildren.unflattened([
                ("w13", Self.bank(format, experts, hidden, 2 * intermediate)),
                ("w2", Self.bank(format, experts, intermediate, hidden)),
            ]))
    }
}

final class Inkling2SparseMoE: Module {
    @ModuleInfo(key: "gate") var router: Inkling2Router
    @ModuleInfo(key: "experts") var experts: Inkling2Experts
    let gateSoftcap: Float
    let upSoftcap: Float

    init(_ c: Inkling2TextConfiguration, format: Inkling2ExpertFormat) {
        _router.wrappedValue = Inkling2Router(
            hidden: c.hiddenSize, experts: c.experts, topK: c.expertsPerToken,
            routeScale: c.routeScale)
        _experts.wrappedValue = Inkling2Experts(
            hidden: c.hiddenSize, intermediate: c.moeIntermediateSize, experts: c.experts,
            format: format)
        gateSoftcap = c.gateSoftcap
        upSoftcap = c.upSoftcap
    }

    /// `x`: `[B, L, hidden]` in the model dtype. Returns the model dtype; expert
    /// outputs are weighted and summed in FP32.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let flat = x.reshaped(-1, shape.last!)
        let (indices, weights) = router(flat)
        Inkling2Diagnostics.section?("router", [indices, weights])
        let rows = expandedDimensions(flat, axes: [-2, -3])
        // Sorting groups rows by expert for the tiled kernel. Decode steps
        // (one token per row) keep the per-row kernel so rows stay batch-invariant.
        let doSort =
            !Inkling2RowExact.active && shape.count >= 2 && shape[shape.count - 2] > 1
            && indices.size >= 64
        var gathered = rows
        var routeIndices = indices
        var inverse: MLXArray? = nil
        if doSort {
            let (sortedRows, sortedIndices, inverseOrder): (MLXArray, MLXArray, MLXArray)
            if Inkling2ExpertSort.supported(indices, experts: experts.experts) {
                let (order, ids, inv) = Inkling2ExpertSort.sort(indices, experts: experts.experts)
                sortedRows = rows.flattened(start: 0, end: -3)[order.floorDivide(indices.dim(-1))]
                (sortedIndices, inverseOrder) = (ids, inv)
            } else {
                (sortedRows, sortedIndices, inverseOrder) = gatherSort(x: rows, indices: indices)
            }
            gathered = sortedRows
            routeIndices = sortedIndices
            inverse = inverseOrder
            Inkling2Diagnostics.section?("moe-sort", [sortedRows, sortedIndices, inverseOrder])
        }
        let activated: MLXArray
        if Inkling2Fused.active {
            activated = Inkling2Fused.expertActivation(
                experts.gateUp(gathered, routeIndices, sorted: doSort), gateCap: gateSoftcap,
                upCap: upSoftcap, dtype: x.dtype)
        } else {
            let gateUp = experts.gateUp(gathered, routeIndices, sorted: doSort).asType(.float32)
            let gate = gateUp[.ellipsis, .stride(from: 0, by: 2)]
            let up = gateUp[.ellipsis, .stride(from: 1, by: 2)]
            activated =
                (Inkling2Math.softcap(gate, gateSoftcap) * sigmoid(gate)
                * Inkling2Math.softcap(up, upSoftcap)).asType(x.dtype)
        }
        Inkling2Diagnostics.section?("moe-gate-up+act", [activated])
        var projected = experts.down(activated, routeIndices, sorted: doSort)
        Inkling2Diagnostics.section?("moe-down", [projected])
        if let inverse, Inkling2Fused.active,
            Inkling2ExpertCombine.supported(
                sortedOutputs: projected, inverseOrder: inverse, weights: weights)
        {
            return Inkling2ExpertCombine.combine(
                sortedOutputs: projected, inverseOrder: inverse, weights: weights
            ).reshaped(shape)
        }
        if let inverse {
            projected = scatterUnsort(x: projected, invOrder: inverse, shape: indices.shape)
        }
        if Inkling2Fused.active {
            return Inkling2Fused.expertCombine(
                projected.squeezed(axis: -2), weights: weights, dtype: x.dtype
            ).reshaped(shape)
        }
        let perExpert = projected.squeezed(axis: -2).asType(.float32)
        let combined = (perExpert * weights.expandedDimensions(axis: -1)).sum(axis: -2)
        return combined.asType(x.dtype).reshaped(shape)
    }
}
