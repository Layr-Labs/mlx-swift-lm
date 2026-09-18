// Copyright © 2026 Eigen Labs.
import MLX
import MLXFast

public enum Gemma4B8ExpertExecution {
    public static let policy = Gemma4B8ExpertPolicy()

    public static var available: Bool {
        #if os(macOS)
        policy.enabled && MLXHardwareInfo.isCompiledDecodeSupported && StreamOrDevice.default == .gpu
        #else
        false
        #endif
    }

    private static let gateUpKernel = makeGateUp(tagged: false)
    private static let gateUpTagged = makeGateUp(tagged: true)
    private static let downKernel = makeDown(tagged: false)
    private static let downTagged = makeDown(tagged: true)

    private static func makeGateUp(tagged: Bool) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
        name: "db_gemma4_b8_expert_gu_cap\(policy.runCap)_prefix\(tagged ? 1 : 0)_v1",
        inputNames: ["w", "scales", "biases", "x", "lhs", "rhs"], outputNames: ["y"],
        source: Gemma4B8ExpertGUSources.source,
        header: "#define GU_RUN_CAP \(policy.runCap)\n#define GU_TAGGED_ROUTE \(tagged ? 1 : 0)\n"
            + Gemma4B8ExpertGUSources.header, ensureRowContiguous: true)
    }

    private static func makeDown(tagged: Bool) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
        name: "db_gemma4_b8_expert_down_word\(policy.packedWordLoads ? 1 : 0)_prefix\(tagged ? 1 : 0)_v1",
        inputNames: ["w", "scales", "biases", "x", "lhs_indices", "rhs_indices"], outputNames: ["y"],
        source: Gemma4B8ExpertDownSources.source,
        header: "#define DOWN_PACKED_WORD_LOAD \(policy.packedWordLoads ? 1 : 0)\n#define DOWN_TAGGED_ROUTE \(tagged ? 1 : 0)\n"
            + Gemma4B8ExpertDownSources.header, ensureRowContiguous: true)
    }

    static func gateUp(_ inputs: [MLXArray], tagged: Bool = false) -> MLXArray {
        (tagged ? gateUpTagged : gateUpKernel)(inputs, grid: (32, 352, 64), threadGroup: (32, 2, 1),
            outputShapes: [[64, 1, 704]], outputDTypes: [.bfloat16], stream: .gpu)[0]
    }

    static func down(_ inputs: [MLXArray], tagged: Bool = false) -> MLXArray {
        (tagged ? downTagged : downKernel)(inputs, template: [("T", DType.bfloat16), ("SPAN", policy.tileSpan)],
            grid: (32, (352 / policy.tileSpan) * 2, 64), threadGroup: (32, 2, 1),
            outputShapes: [[64, 1, 2816]], outputDTypes: [.bfloat16], stream: .gpu)[0]
    }

    private static let compiledBody = makeCompiled(tagged: false)
    private static let compiledTagged = makeCompiled(tagged: true)

    private static func makeCompiled(tagged: Bool) -> @Sendable ([MLXArray]) -> [MLXArray] {
        MLX.compile(shapeless: false) { inputs in
            let activated = gateUp([inputs[0], inputs[1], inputs[2], inputs[6], inputs[7], inputs[8]], tagged: tagged)
            return [down([inputs[3], inputs[4], inputs[5], activated, inputs[9], inputs[8]], tagged: tagged)]
        }
    }

    static func compiledProject(storage: Gemma4B8ExpertStorage, x: MLXArray,
                                routing: Gemma4B8ExpertRouting, identity: MLXArray) -> MLXArray {
        // Every tensor is substituted; no layer weight or route is captured.
        (routing.usesPrefixBounds ? compiledTagged : compiledBody)(
            storage.gateUp + storage.down + [x, routing.rowOrder, routing.executionKeys, identity])[0]
    }
}
