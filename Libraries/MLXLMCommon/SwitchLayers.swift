import Foundation
import MLX
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

private let switchGLUSortMinSize: Int = {
    let raw = ProcessInfo.processInfo.environment["MLX_SWITCH_GLU_SORT_MIN_SIZE"]
    guard let raw, let value = Int(raw) else { return 32 }
    return max(0, value)
}()

private let switchGLUFuseGateUp: Bool = {
    switch ProcessInfo.processInfo.environment["MLX_SWITCH_GLU_FUSE_GATE_UP"]?.lowercased() {
    case "1", "true", "yes", "on":
        true
    default:
        false
    }
}()

private let switchGLUDebugFuseGateUp: Bool = {
    switch ProcessInfo.processInfo.environment["MLX_SWITCH_GLU_DEBUG_FUSE_GATE_UP"]?.lowercased() {
    case "1", "true", "yes", "on":
        true
    default:
        false
    }
}()

private let switchGLUGemma4FuseGateUp: Bool = {
    if let raw = ProcessInfo.processInfo.environment["MLX_SWITCH_GLU_GEMMA4_FUSE_GATE_UP"] {
        switch raw.lowercased() {
        case "0", "false", "no", "off":
            return false
        default:
            return true
        }
    }
    if let raw = ProcessInfo.processInfo.environment["MLX_SWITCH_GLU_FUSE_GATE_UP"] {
        switch raw.lowercased() {
        case "0", "false", "no", "off":
            return false
        default:
            return true
        }
    }
    return true
}()

private let switchGLUGemma4WeightedFuseGateUp: Bool = {
    if let raw = ProcessInfo.processInfo.environment["MLX_SWITCH_GLU_GEMMA4_WEIGHTED_FUSE_GATE_UP"] {
        switch raw.lowercased() {
        case "0", "false", "no", "off":
            return false
        default:
            return true
        }
    }
    if let raw = ProcessInfo.processInfo.environment["MLX_SWITCH_GLU_GEMMA4_FUSE_GATE_UP"] {
        switch raw.lowercased() {
        case "0", "false", "no", "off":
            return false
        default:
            return true
        }
    }
    return false
}()

private let switchGLUGemma4WeightedTimings: Bool = {
    switch ProcessInfo.processInfo.environment["MLX_SWITCH_GLU_GEMMA4_WEIGHTED_TIMINGS"]?
        .lowercased()
    {
    case "1", "true", "yes", "on":
        true
    default:
        false
    }
}()

private let switchGLUGemma4WeightedMaxRows: Int = {
    guard let raw = ProcessInfo.processInfo.environment["MLX_SWITCH_GLU_GEMMA4_WEIGHTED_MAX_ROWS"],
        let value = Int(raw),
        value > 0
    else {
        return 16
    }
    return value
}()

private let switchGLUGemma4RouteSortKernel: Bool = {
    if let raw = ProcessInfo.processInfo.environment["MLX_SWITCH_GLU_GEMMA4_ROUTE_SORT_KERNEL"] {
        switch raw.lowercased() {
        case "0", "false", "no", "off":
            return false
        default:
            return true
        }
    }
    return true
}()

nonisolated(unsafe) private var switchGLUDebugFuseGateUpReports = 0
nonisolated(unsafe) private var switchGLUGemma4WeightedTimingCalls = 0
nonisolated(unsafe) private var switchGLUGemma4WeightedTimingSort = 0.0
nonisolated(unsafe) private var switchGLUGemma4WeightedTimingGateUp = 0.0
nonisolated(unsafe) private var switchGLUGemma4WeightedTimingActivation = 0.0
nonisolated(unsafe) private var switchGLUGemma4WeightedTimingDown = 0.0
nonisolated(unsafe) private var switchGLUGemma4WeightedTimingReduce = 0.0

private let switchGLUCompiledGELUGateUp: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { gate, up in
    geluApproximate(gate) * up
}

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}

private func recordSwitchGLUGemma4WeightedTiming(
    sortSeconds: Double,
    gateUpSeconds: Double,
    activationSeconds: Double,
    downSeconds: Double,
    reduceSeconds: Double
) {
    switchGLUGemma4WeightedTimingCalls += 1
    switchGLUGemma4WeightedTimingSort += sortSeconds
    switchGLUGemma4WeightedTimingGateUp += gateUpSeconds
    switchGLUGemma4WeightedTimingActivation += activationSeconds
    switchGLUGemma4WeightedTimingDown += downSeconds
    switchGLUGemma4WeightedTimingReduce += reduceSeconds

    guard switchGLUGemma4WeightedTimingCalls % 30 == 0 else { return }
    let calls = Double(switchGLUGemma4WeightedTimingCalls)
    let ms: (Double) -> String = { String(format: "%.3f", $0 * 1000 / calls) }
    print(
        "gemma4_weighted_expert ms/layer: "
            + "sort=\(ms(switchGLUGemma4WeightedTimingSort)) "
            + "gate_up=\(ms(switchGLUGemma4WeightedTimingGateUp)) "
            + "activation=\(ms(switchGLUGemma4WeightedTimingActivation)) "
            + "down=\(ms(switchGLUGemma4WeightedTimingDown)) "
            + "reduce=\(ms(switchGLUGemma4WeightedTimingReduce)) "
            + "calls=\(switchGLUGemma4WeightedTimingCalls)"
    )
}

// MARK: - SwitchGLU

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    private var fusedGateUpProj: QuantizedSwitchLinear?
    private var fusedGateUpUnavailable = false

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray = MLXNN.silu,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        routed(x, indices, fuseGateUp: switchGLUFuseGateUp)
    }

    public func gemma4Routed(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        routed(x, indices, fuseGateUp: switchGLUGemma4FuseGateUp)
    }

    private func routed(_ input: MLXArray, _ indices: MLXArray, fuseGateUp: Bool) -> MLXArray {
        var x = MLX.expandedDimensions(input, axes: [-2, -3])

        let doSort = indices.size >= switchGLUSortMinSize

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let xUp: MLXArray
        let xGate: MLXArray
        if fuseGateUp, let fused = fusedGateUp() {
            let gateUp = fused(x, idx, sortedIndices: doSort)
            let parts = MLX.split(gateUp, parts: 2, axis: -1)
            xGate = parts[0].contiguous()
            xUp = parts[1].contiguous()
            if switchGLUDebugFuseGateUp, switchGLUDebugFuseGateUpReports < 6 {
                switchGLUDebugFuseGateUpReports += 1
                let separateUp = upProj(x, idx, sortedIndices: doSort)
                let separateGate = gateProj(x, idx, sortedIndices: doSort)
                let gateDiff = MLX.abs(parts[0].asType(.float32) - separateGate.asType(.float32))
                    .max()
                let upDiff = MLX.abs(parts[1].asType(.float32) - separateUp.asType(.float32))
                    .max()
                eval(gateDiff, upDiff)
                print(
                    "switch glu fused gate/up shapes fused=\(gateUp.shape) "
                        + "gate=\(parts[0].shape) up=\(parts[1].shape) "
                        + "separate_gate=\(separateGate.shape) separate_up=\(separateUp.shape) "
                        + "gate_diff=\(gateDiff.item(Float.self)) "
                        + "up_diff=\(upDiff.item(Float.self)) sorted=\(doSort)"
                )
            }
        } else {
            xUp = upProj(x, idx, sortedIndices: doSort)
            xGate = gateProj(x, idx, sortedIndices: doSort)
        }
        x = downProj(
            activation(xGate) * xUp,
            idx,
            sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }

    private func fusedGateUp() -> QuantizedSwitchLinear? {
        if let fusedGateUpProj {
            return fusedGateUpProj
        }
        guard !fusedGateUpUnavailable,
            let gate = gateProj as? QuantizedSwitchLinear,
            let up = upProj as? QuantizedSwitchLinear,
            gate.bias == nil,
            up.bias == nil,
            gate.bits == up.bits,
            gate.groupSize == up.groupSize,
            gate.mode == up.mode,
            gate.weight.shape == up.weight.shape,
            gate.scales.shape == up.scales.shape,
            gate.biases?.shape == up.biases?.shape,
            let gateBiases = gate.biases,
            let upBiases = up.biases
        else {
            fusedGateUpUnavailable = true
            return nil
        }

        let fused = QuantizedSwitchLinear(
            inputDims: inputDims,
            outputDims: hiddenDims * 2,
            numExperts: numExperts,
            weight: concatenated([gate.weight, up.weight], axis: 1),
            bias: nil,
            scales: concatenated([gate.scales, up.scales], axis: 1),
            biases: concatenated([gateBiases, upBiases], axis: 1),
            groupSize: gate.groupSize,
            bits: gate.bits,
            mode: gate.mode
        )
        fusedGateUpProj = fused
        return fused
    }

    public func gemma4Weighted(
        _ x: MLXArray, indices: MLXArray, weights: MLXArray
    ) -> MLXArray? {
        guard x.dim(-1) == 2816, indices.dim(-1) == 8, hiddenDims == 704,
            inputDims == 2816, numExperts == 128
        else { return nil }

        let rows = x.size / inputDims
        guard rows > 0, rows <= switchGLUGemma4WeightedMaxRows, rows * 8 == indices.size
        else { return nil }
        guard indices.size >= switchGLUSortMinSize else { return nil }

        var sortSeconds = 0.0
        var gateUpSeconds = 0.0
        var activationSeconds = 0.0
        var downSeconds = 0.0
        var reduceSeconds = 0.0

        let sortStart = switchGLUGemma4WeightedTimings ? Date() : nil
        let expanded: MLXArray
        let idx: MLXArray
        let inverseOrder: MLXArray
        if switchGLUGemma4RouteSortKernel {
            let order: MLXArray
            (order, idx, inverseOrder) = Self.gemma4RouteSortMetadata(
                indices: indices.contiguous(),
                rows: rows,
                topK: 8
            )
            var sortedExpanded = MLX.expandedDimensions(x, axes: [-2, -3])
            sortedExpanded = sortedExpanded.flattened(start: 0, end: -3)[order.floorDivide(8)]
            expanded = sortedExpanded
        } else {
            var sortedExpanded = MLX.expandedDimensions(x, axes: [-2, -3])
            let flatIndices = indices.flattened()
            let order = argSort(flatIndices)
            inverseOrder = Self.gemma4InversePermutation(
                order: order.contiguous(), count: rows * 8)
            sortedExpanded = sortedExpanded.flattened(start: 0, end: -3)[order.floorDivide(8)]
            expanded = sortedExpanded
            idx = flatIndices[order]
        }
        if let sortStart {
            eval(expanded, idx, inverseOrder)
            sortSeconds = Date().timeIntervalSince(sortStart)
        }

        let xUp: MLXArray
        let xGate: MLXArray
        let gateUpStart = switchGLUGemma4WeightedTimings ? Date() : nil
        if switchGLUGemma4WeightedFuseGateUp, let fused = fusedGateUp() {
            let gateUp = fused(expanded, idx, sortedIndices: true)
            let parts = MLX.split(gateUp, parts: 2, axis: -1)
            xGate = parts[0].contiguous()
            xUp = parts[1].contiguous()
        } else {
            xUp = upProj(expanded, idx, sortedIndices: true)
            xGate = gateProj(expanded, idx, sortedIndices: true)
        }
        if let gateUpStart {
            eval(xUp, xGate)
            gateUpSeconds = Date().timeIntervalSince(gateUpStart)
        }

        let activationStart = switchGLUGemma4WeightedTimings ? Date() : nil
        let activated = switchGLUCompiledGELUGateUp(xGate, xUp)
        if let activationStart {
            eval(activated)
            activationSeconds = Date().timeIntervalSince(activationStart)
        }

        let downStart = switchGLUGemma4WeightedTimings ? Date() : nil
        let sorted = downProj(
            activated,
            idx,
            sortedIndices: true)
        let sortedFlat = MLX.squeezed(sorted, axis: -2).contiguous()
        if let downStart {
            eval(sortedFlat)
            downSeconds = Date().timeIntervalSince(downStart)
        }

        let reduceStart = switchGLUGemma4WeightedTimings ? Date() : nil
        let reduced = Self.gemma4UnsortWeightedSumKernel(
            sorted: sortedFlat,
            inverseOrder: inverseOrder.contiguous(),
            weights: weights.asType(sortedFlat.dtype).flattened().contiguous(),
            rows: rows,
            topK: 8,
            hidden: inputDims,
            dtype: sortedFlat.dtype
        )
        if let reduceStart {
            eval(reduced)
            reduceSeconds = Date().timeIntervalSince(reduceStart)
            recordSwitchGLUGemma4WeightedTiming(
                sortSeconds: sortSeconds,
                gateUpSeconds: gateUpSeconds,
                activationSeconds: activationSeconds,
                downSeconds: downSeconds,
                reduceSeconds: reduceSeconds
            )
        }
        return reduced
    }

    private static func gemma4UnsortWeightedSumKernel(
        sorted: MLXArray,
        inverseOrder: MLXArray,
        weights: MLXArray,
        rows: Int,
        topK: Int,
        hidden: Int,
        dtype: DType
    ) -> MLXArray {
        let kernel = Gemma4UnsortWeightedSumKernelManager.shared.kernel
        return kernel(
            [sorted, inverseOrder, weights, rows, topK, hidden],
            template: [("T", dtype)],
            grid: (rows * hidden, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[rows, hidden]],
            outputDTypes: [dtype]
        )[0]
    }

    private static func gemma4InversePermutation(order: MLXArray, count: Int) -> MLXArray {
        let kernel = Gemma4InversePermutationKernelManager.shared.kernel
        return kernel(
            [order, count],
            grid: (count, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[count]],
            outputDTypes: [.uint32]
        )[0]
    }

    private static func gemma4RouteSortMetadata(
        indices: MLXArray,
        rows: Int,
        topK: Int
    ) -> (order: MLXArray, indices: MLXArray, inverseOrder: MLXArray) {
        let routeCount = rows * topK
        let outputs = Gemma4RouteSortKernelManager.shared.kernel(
            [indices, rows, topK],
            grid: (routeCount, 1, 1),
            threadGroup: (128, 1, 1),
            outputShapes: [[routeCount], [routeCount], [routeCount]],
            outputDTypes: [.uint32, .uint32, .uint32]
        )
        return (outputs[0], outputs[1], outputs[2])
    }
}

private final class Gemma4InversePermutationKernelManager: Sendable {
    static let shared = Gemma4InversePermutationKernelManager()

    let kernel: MLXFast.MLXFastKernel

    private init() {
        kernel = MLXFast.metalKernel(
            name: "gemma4_inverse_permutation_u32",
            inputNames: ["order", "count"],
            outputNames: ["inverse_order"],
            source: """
                uint gid = thread_position_in_grid.x;
                if (gid >= uint(count)) {
                    return;
                }

                inverse_order[uint(order[gid])] = gid;
                """
        )
    }
}

private final class Gemma4RouteSortKernelManager: Sendable {
    static let shared = Gemma4RouteSortKernelManager()

    let kernel: MLXFast.MLXFastKernel

    private init() {
        kernel = MLXFast.metalKernel(
            name: "gemma4_route_sort_top8_metadata",
            inputNames: ["indices", "rows", "top_k"],
            outputNames: ["order", "sorted_indices", "inverse_order"],
            source: """
                uint route = thread_position_in_grid.x;
                uint route_count = uint(rows) * uint(top_k);
                if (route >= route_count) {
                    return;
                }

                uint expert = uint(indices[route]);
                uint sorted_pos = 0;
                for (uint other = 0; other < route_count; ++other) {
                    uint other_expert = uint(indices[other]);
                    if (other_expert < expert || (other_expert == expert && other < route)) {
                        sorted_pos += 1;
                    }
                }

                sorted_indices[sorted_pos] = expert;
                inverse_order[route] = sorted_pos;
                order[sorted_pos] = route;
                """
        )
    }
}

private final class Gemma4UnsortWeightedSumKernelManager: Sendable {
    static let shared = Gemma4UnsortWeightedSumKernelManager()

    let kernel: MLXFast.MLXFastKernel

    private init() {
        kernel = MLXFast.metalKernel(
            name: "gemma4_unsort_weighted_sum_top8",
            inputNames: ["sorted", "inverse_order", "weights", "rows", "top_k", "hidden_dims"],
            outputNames: ["out"],
            source: """
                uint gid = thread_position_in_grid.x;
                uint total = uint(rows) * uint(hidden_dims);
                if (gid >= total) {
                    return;
                }

                uint h = gid % uint(hidden_dims);
                uint row = gid / uint(hidden_dims);
                uint route_base = row * uint(top_k);

                float acc = 0.0f;
                for (uint route = 0; route < uint(top_k); ++route) {
                    uint original_route = route_base + route;
                    uint sorted_route = uint(inverse_order[original_route]);
                    acc += float(weights[original_route])
                        * float(sorted[sorted_route * uint(hidden_dims) + h]);
                }

                out[gid] = T(acc);
                """
        )
    }
}

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        weight: MLXArray,
        bias: MLXArray? = nil,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int = 64,
        bits: Int = 4,
        mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: inputDims,
            outputDims: outputDims,
            numExperts: numExperts,
            weight: weight,
            bias: bias
        )

        self.freeze()
    }

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        var result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }
}
