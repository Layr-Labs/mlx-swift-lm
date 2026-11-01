#if canImport(Metal)
import Cmlx
import CryptoKit
import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing

@testable import MLXLMCommon

// An empty subclass preserves the complete previous SwitchGLU implementation,
// including its chosen weighted reduction. Only the new exact-class route is
// refused. This avoids changing reduction policy to construct a reference.
private final class RowMapOriginalSwitch: SwitchGLU {}
private final class RowMapCustomBank: QuantizedSwitchLinear {
    override func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray,
        sortedIndices: Bool = false
    ) -> MLXArray {
        super.callAsFunction(x, indices, sortedIndices: sortedIndices) + Float(0.125)
    }
}

enum Gemma4RowMapProductionOracle {
    struct Bank {
        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray
    }

    static func alias(_ x: MLXArray) -> MLXArray {
        var handle = mlx_array_new()
        mlx_array_set(&handle, x.ctx)
        return MLXArray(handle)
    }

    static func bytes(_ x: MLXArray) -> Data { x.asData(access: .copy).data }
    static func hash(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    final class Report {
        let directory: URL
        var rows: [[String: Any]] = []
        var completed = false
        var failureSaved = false

        init(_ name: String) throws {
            let root = try #require(
                ProcessInfo.processInfo.environment["GEMMA4_ROW_MAP_PRODUCTION_REPORT"])
            directory = URL(fileURLWithPath: root).appendingPathComponent(name)
            try #require(!FileManager.default.fileExists(atPath: directory.path))
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: false,
                attributes: [.posixPermissions: 0o700])
            try flush()
        }

        func exact(
            _ actual: MLXArray, _ expected: MLXArray, _ label: String,
            evidence: [String: Any] = [:]
        ) throws {
            let a = bytes(actual)
            let b = bytes(expected)
            let equal = actual.shape == expected.shape && actual.dtype == expected.dtype && a == b
            var row = evidence
            row.merge([
                "label": label, "equal": equal, "shape": actual.shape,
                "dtype": String(describing: actual.dtype), "bytes": a.count,
                "candidate_sha256": hash(a), "reference_sha256": hash(b),
            ]) { _, new in new }
            rows.append(row)
            if !equal && !failureSaved {
                failureSaved = true
                try a.write(to: directory.appendingPathComponent("first-candidate.bin"))
                try b.write(to: directory.appendingPathComponent("first-reference.bin"))
            }
            try flush()
            #expect(equal, "\(label): native bytes, shape or dtype differ")
        }

        func flush() throws {
            let object: [String: Any] = [
                "completed": completed,
                "requested": Gemma4RowMappedNAX.requested,
                "observations": rows,
                "failed": rows.filter { $0["equal"] as? Bool == false }.count,
                "scope":
                    "Whole SwitchGLU graph and fallback contracts; not full-model or endpoint qualification",
            ]
            try JSONSerialization.data(
                withJSONObject: object, options: [.prettyPrinted, .sortedKeys]
            )
            .write(to: directory.appendingPathComponent("report.json"), options: .atomic)
        }
    }

    static func lane() throws {
        try #require(Device.defaultDevice().deviceType == .gpu)
        try #require(GPU.gemma4ExpertQMMDiagnostics().naxAvailable)
        try #require(ProcessInfo.processInfo.environment["MLX_GEMMA4_NAX_GATHER_STAGED"] == nil)
        try #require(ProcessInfo.processInfo.environment["MLX_GEMMA4_NAX_GATHER_EXPERIMENT"] == nil)
        let flag = ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_NAX_ROW_MAP"]
        try #require(flag == nil || flag == "0" || flag == "1")
        try #require(Gemma4RowMappedNAX.requested == (flag == nil || flag == "1"))
    }

    static func banks() -> [Bank] {
        (0 ..< 3).map { index in
            let k = index == 2 ? 704 : 2816
            let n = index == 2 ? 2816 : 704
            let seed = UInt64(91000 + index * 10)
            let w = MLXRandom.randInt(
                UInt32.min ..< UInt32.max, [128, n, k / 8], key: MLXRandom.key(seed))
            let s =
                (abs(MLXRandom.normal([128, n, k / 64], key: MLXRandom.key(seed + 1)))
                * Float(0.003) + Float(0.001)).asType(.bfloat16)
            let b = (MLXRandom.normal(s.shape, key: MLXRandom.key(seed + 2)) * Float(0.02)).asType(
                .bfloat16)
            eval(w, s, b)
            return Bank(weight: w, scales: s, biases: b)
        }
    }

    static func child(_ bank: Bank, index: Int, custom: Bool = false) throws
        -> QuantizedSwitchLinear
    {
        let k = index == 2 ? 704 : 2816
        let n = index == 2 ? 2816 : 704
        if custom {
            return try RowMapCustomBank(
                inputDims: k, outputDims: n, numExperts: 128,
                weight: alias(bank.weight), scales: alias(bank.scales), biases: alias(bank.biases))
        }
        return try QuantizedSwitchLinear(
            inputDims: k, outputDims: n, numExperts: 128,
            weight: alias(bank.weight), scales: alias(bank.scales), biases: alias(bank.biases))
    }

    static func model(
        _ banks: [Bank], original: Bool = false, customIndex: Int? = nil,
        generic: Bool = false, gelu: Bool = true
    ) throws -> SwitchGLU {
        let profile: SwitchGLUWeightedReductionProfile = generic ? .generic : .gemma4ProductionGeGLU
        let activation: (MLXArray) -> MLXArray
        if gelu {
            activation = { safeGeluApproximate($0) }
        } else {
            activation = { $0 + Float(0.25) }
        }
        let result: SwitchGLU
        if original {
            result = RowMapOriginalSwitch(
                inputDims: 2816, hiddenDims: 704, numExperts: 128,
                activation: activation, weightedReductionProfile: profile)
        } else {
            result = SwitchGLU(
                inputDims: 2816, hiddenDims: 704, numExperts: 128,
                activation: activation, weightedReductionProfile: profile)
        }
        var children: [(String, Module)] = []
        for (index, name) in ["gate_proj", "up_proj", "down_proj"].enumerated() {
            children.append(
                (name, try child(banks[index], index: index, custom: customIndex == index)))
        }
        result.update(modules: .unflattened(children))
        result.train(false)
        return result
    }

    static func inputs(_ count: Int = 64, topK: Int = 8, hot: Bool = false)
        -> (MLXArray, MLXArray, MLXArray)
    {
        let x =
            (MLXRandom.normal([count, 2816], key: MLXRandom.key(UInt64(92000 + count)))
            * Float(0.25)).asType(.bfloat16)
        let ids = (0 ..< count * topK).map { UInt32(hot ? 127 : ($0 * 37 + 11) % 128) }
        let indices = MLXArray(ids, [count, topK])
        let weights = softmax(MLXRandom.normal([count, topK], key: MLXRandom.key(93000)), axis: -1)
            .asType(.bfloat16)
        eval(x, indices, weights)
        return (x, indices, weights)
    }

    static func call(
        _ model: SwitchGLU, _ input: (MLXArray, MLXArray, MLXArray),
        fused: Bool = false, prefill: Bool = true
    ) -> MLXArray {
        model.callAndWeightedReduce(
            input.0, input.1, weights: input.2,
            fuseSortedReduction: fused, isProductionPrefill: prefill)
    }

    static func customNodes(_ graph: String) -> Int {
        graph.split(separator: "\n").filter { $0.hasPrefix("CustomKernel ") }.count
    }

    @discardableResult
    static func compare(
        _ actual: SwitchGLU, _ reference: SwitchGLU,
        input: (MLXArray, MLXArray, MLXArray), report: Report, label: String,
        eligible: Bool, fused: Bool = false, prefill: Bool = true,
        requireProjectionGraphCounts: Bool = true
    ) throws -> Data {
        CBv2StepProfiler.armEvents()
        defer { _ = CBv2StepProfiler.snapshotAndDisarmEvents() }
        let a = call(actual, input, fused: fused, prefill: prefill)
        let graphA = graphDescription([a])
        eval(a)
        let callsA = CBv2StepProfiler.snapshotAndDisarmEvents()["v2.gemma4.nax.row_map", default: 0]
        CBv2StepProfiler.armEvents()
        let b = call(reference, input, fused: fused, prefill: prefill)
        let graphB = graphDescription([b])
        eval(b)
        let callsB = CBv2StepProfiler.snapshotAndDisarmEvents()["v2.gemma4.nax.row_map", default: 0]
        let countA = graphA.split(separator: "\n").filter { $0.hasPrefix("GatherQMM ") }.count
        let countB = graphB.split(separator: "\n").filter { $0.hasPrefix("GatherQMM ") }.count
        let selected = eligible && Gemma4RowMappedNAX.requested
        #expect(
            callsA == (selected ? 1 : 0) && callsB == 0,
            "\(label): whole-module pair construction count")
        #expect(
            customNodes(graphA) - customNodes(graphB) == (selected ? 3 : 0), "\(label): route graph"
        )
        if requireProjectionGraphCounts {
            #expect(countB == 3, "\(label): original must contain all three projection nodes")
            #expect(
                countA == (selected ? 1 : 3),
                "\(label): row-map graph must retain only down GatherQMM")
        }
        if selected {
            let sourceShape = "(\(input.0.dim(0)),1,2816)"
            func stagesTokenRows(_ graph: String) -> Bool {
                graph.split(separator: "\n").contains { line in
                    guard line.hasPrefix("Gather "), let arrow = line.range(of: " -> ") else {
                        return false
                    }
                    return line[..<arrow.lowerBound].contains(sourceShape)
                }
            }
            // The down output legitimately has the same dimensions as expanded
            // input rows. Identify the input Gather edge, not any matching shape.
            #expect(stagesTokenRows(graphB), "\(label): original input-staging Gather witness")
            #expect(
                !stagesTokenRows(graphA), "\(label): candidate still gathers expanded token rows")
        }
        try report.exact(
            a, b, label,
            evidence: [
                "selected": selected,
                "candidate_pair_calls": callsA, "reference_pair_calls": callsB,
                "candidate_gather_qmm_graph_nodes": countA,
                "reference_gather_qmm_graph_nodes": countB,
                "candidate_custom_nodes": customNodes(graphA),
                "reference_custom_nodes": customNodes(graphB),
            ])
        return bytes(a)
    }

    enum Outcome {
        case arrays([MLXArray])
        case error(String)
    }

    static func outcome(_ body: () throws -> [MLXArray]) -> Outcome {
        do {
            return try MLX.withError { errors in
                let values = try body()
                try errors.check()
                eval(values)
                try errors.check()
                return .arrays(values)
            }
        } catch { return .error(String(describing: error)) }
    }

    static func compareOutcomes(
        _ actual: Outcome, _ expected: Outcome,
        report: Report, label: String
    ) throws {
        switch (actual, expected) {
        case (.arrays(let a), .arrays(let b)):
            try #require(a.count == b.count)
            for index in a.indices { try report.exact(a[index], b[index], "\(label)[\(index)]") }
        case (.error(let a), .error(let b)):
            report.rows.append([
                "label": label, "equal": a == b, "candidate_error": a,
                "reference_error": b, "original_error_contract": true,
            ])
            try report.flush()
            #expect(a == b, "\(label): original error contract changed")
        default:
            report.rows.append(["label": label, "equal": false, "result_kind_mismatch": true])
            try report.flush()
            Issue.record("\(label): success/error behavior differs")
        }
    }
}

extension KernelTests {
    @Suite(
        "Gemma4 production row-map route", .serialized,
        .enabled(if: ProcessInfo.processInfo.environment["GEMMA4_ROW_MAP_PRODUCTION_TEST"] == "1"))
    struct Gemma4RowMappedNAXProductionTests {
        typealias O = Gemma4RowMapProductionOracle

        @Test func completeGraphBothReductionPoliciesAndStaging() throws {
            try O.lane()
            let report = try O.Report("whole-graph")
            defer { try? report.flush() }
            let banks = O.banks()
            let a = try O.model(banks)
            let b = try O.model(banks, original: true)
            let before = banks.flatMap { [$0.weight, $0.scales, $0.biases] }.map {
                O.hash(O.bytes($0))
            }
            for rows in [64, 65, 512, 2048] {
                for hot in [false, true] {
                    for fused in [false, true] {
                        try O.compare(
                            a, b, input: O.inputs(rows, hot: hot), report: report,
                            label: "T\(rows) hot\(hot) fused\(fused)", eligible: true, fused: fused)
                    }
                }
            }
            let layoutInput = O.inputs()
            let wide = O.inputs(128).0.reshaped([64, 5632])
            let offset = O.inputs(65).0[1...]
            for (name, x) in [("strided", wide[.ellipsis, .stride(by: 2)]), ("offset", offset)] {
                try O.compare(
                    a, b, input: (x, layoutInput.1, layoutInput.2), report: report,
                    label: "whole graph \(name)", eligible: true)
            }
            #expect(
                before
                    == banks.flatMap { [$0.weight, $0.scales, $0.biases] }.map {
                        O.hash(O.bytes($0))
                    })
            report.completed = true
        }

        @Test func trainingCustomChildrenProfilesAndExplicitDecodeKeepOriginalGraph() throws {
            try O.lane()
            let report = try O.Report("module-fallbacks")
            defer { try? report.flush() }
            let banks = O.banks()
            let input = O.inputs()
            for custom in [0, 1, 2] {
                let a = try O.model(banks, customIndex: custom)
                let b = try O.model(banks, original: true, customIndex: custom)
                try O.compare(
                    a, b, input: input, report: report, label: "custom child\(custom)",
                    eligible: false)
            }
            for (generic, gelu) in [(true, true), (false, false)] {
                try O.compare(
                    O.model(banks, generic: generic, gelu: gelu),
                    O.model(banks, original: true, generic: generic, gelu: gelu), input: input,
                    report: report, label: "generic\(generic) gelu\(gelu)", eligible: false)
            }
            let biasedA = try O.model(banks)
            let biasedB = try O.model(banks, original: true)
            for model in [biasedA, biasedB] {
                let down = try QuantizedSwitchLinear(
                    inputDims: 704, outputDims: 2816, numExperts: 128,
                    weight: O.alias(banks[2].weight), scales: O.alias(banks[2].scales),
                    biases: O.alias(banks[2].biases),
                    bias: full([128, 2816], values: MLXArray(Float(0.125)), dtype: .bfloat16))
                model.update(modules: .unflattened([("down_proj", down)]))
                model.train(false)
            }
            try O.compare(
                biasedA, biasedB, input: input, report: report, label: "linear-biased down",
                eligible: false)
            let a = try O.model(banks)
            let b = try O.model(banks, original: true)
            a.train(true)
            b.train(true)
            try O.compare(a, b, input: input, report: report, label: "training", eligible: false)
            a.train(false)
            b.train(false)
            try O.compare(
                a, b, input: input, report: report, label: "explicit nonprefill", eligible: false,
                prefill: false)
            try O.compare(
                O.model(banks, original: true), b, input: input, report: report,
                label: "subclass whole graph", eligible: false)
            report.completed = true
        }

        @Test func nearShapesCPUAndCustomGPUStreamRetainOriginalGraph() throws {
            try O.lane()
            let report = try O.Report("shape-device-fallbacks")
            defer { try? report.flush() }
            let banks = O.banks()
            let a = try O.model(banks)
            let b = try O.model(banks, original: true)
            for rows in [63, 2049] {
                try O.compare(
                    a, b, input: O.inputs(rows), report: report, label: "T\(rows)", eligible: false,
                    requireProjectionGraphCounts: rows >= 64)
            }
            try O.compare(
                a, b, input: O.inputs(128, topK: 4), report: report,
                label: "topK4", eligible: false)
            let halfBanks = banks.map {
                O.Bank(
                    weight: $0.weight,
                    scales: $0.scales.asType(.float16), biases: $0.biases.asType(.float16))
            }
            let halfInput = O.inputs()
            try O.compare(
                O.model(halfBanks), O.model(halfBanks, original: true),
                input: (halfInput.0.asType(.float16), halfInput.1, halfInput.2.asType(.float16)),
                report: report, label: "FP16 original graph", eligible: false)
            let input = O.inputs()
            try Stream.withNewDefaultStream(device: .gpu) {
                #expect(!MLXProcessStreams.isOnDefaultGPUStream)
                try O.compare(
                    a, b, input: input, report: report, label: "custom GPU stream", eligible: false)
            }
            try Stream.withNewDefaultStream(device: .cpu) {
                // CPU refusal is checked at eligible dimensions, and the full
                // CPU graph is bounded to one token to avoid a large CPU workload.
                #expect(
                    Gemma4RowMappedNAX.tryPair(
                        input.0, indices: input.1,
                        gate: try #require(a.gateProj), up: try #require(a.upProj), enabled: true)
                        == nil)
                try O.compare(
                    a, b, input: O.inputs(1), report: report, label: "CPU full graph",
                    eligible: false, requireProjectionGraphCounts: false)
            }
            report.completed = true
        }

        @Test func liveParameterRebindingAndChildReplacementStayLive() throws {
            try O.lane()
            let report = try O.Report("live-parameters")
            defer { try? report.flush() }
            let banks = O.banks()
            let a = try O.model(banks)
            let b = try O.model(banks, original: true)
            let input = O.inputs()
            let initial = try O.compare(
                a, b, input: input, report: report, label: "initial", eligible: true)
            let pendingA = O.call(a, input)
            let pendingB = O.call(b, input)
            let replacement = (banks[0].scales * Float(1.25)).asType(.bfloat16)
            eval(replacement)
            for model in [a, b] {
                model.update(parameters: .unflattened([("gate_proj.scales", O.alias(replacement))]))
            }
            try report.exact(pendingA, pendingB, "graphs created before parameter rebind")
            let changed = try O.compare(
                a, b, input: input, report: report, label: "rebound scales", eligible: true)
            #expect(changed != initial, "Fixture must observe the changed live parameter")
            for model in [a, b] {
                let gate = try #require(model.gateProj as? QuantizedSwitchLinear)
                gate.scales[0, 0, 0] = MLXArray(Float(0.25)).asType(.bfloat16)
            }
            try O.compare(
                a, b, input: input, report: report, label: "indexed scale wrapper update",
                eligible: true)
            for model in [a, b] {
                model.update(
                    modules: .unflattened([
                        ("down_proj", try O.child(banks[2], index: 2, custom: true))
                    ]))
                model.train(false)
            }
            try O.compare(
                a, b, input: input, report: report, label: "custom down after warm route",
                eligible: false)
            report.completed = true
        }

        @Test func pendingAsyncGraphsOutliveModulesAndMutableInputWrappers() throws {
            try O.lane()
            let report = try O.Report("lifetime")
            defer { try? report.flush() }
            let banks = O.banks()
            let input = O.inputs()
            weak var weakModel: SwitchGLU?
            var pending: MLXArray?
            let reference = O.call(try O.model(banks, original: true), input)
            do {
                let model = try O.model(banks)
                weakModel = model
                // A temporary input and output wrapper may be donated/reused by
                // MLX. This checks lifetime correctness, not allocator donation counts.
                let temporary = input.0 + MLXArray(Float(0)).asType(input.0.dtype)
                pending = O.call(model, (temporary, input.1, input.2))
                asyncEval(try #require(pending))
            }
            #expect(weakModel == nil)
            try report.exact(
                try #require(pending), reference, "async graph after module and temporary release")
            pending = nil
            let x = O.alias(input.0)
            let candidate = O.call(try O.model(banks), (x, input.1, input.2))
            let original = O.call(
                try O.model(banks, original: true), (O.alias(x), input.1, input.2))
            x[0] = MLXArray(Float(7)).asType(.bfloat16)
            eval(x)
            try report.exact(candidate, original, "indexed wrapper update after graph construction")
            report.completed = true
        }

        @Test func unusedExceptionalProjectionAndSubsetOutputStayIndependent() throws {
            try O.lane()
            let report = try O.Report("subset-output")
            defer { try? report.flush() }
            var banks = O.banks()
            banks[1] = O.Bank(
                weight: banks[1].weight, scales: banks[1].scales,
                biases: full(banks[1].biases.shape, values: MLXArray(Float.nan), dtype: .bfloat16))
            let model = try O.model(banks)
            let input = O.inputs()
            let gate = try #require(model.gateProj)
            let up = try #require(model.upProj)
            let pair = try #require(
                Gemma4RowMappedNAX.tryPair(
                    input.0, indices: input.1,
                    gate: gate, up: up, enabled: true))
            let order = argSort(input.1.flattened())
            let rows = input.0.reshaped([-1, 1, 2816])[order.floorDivide(8)]
            let reference = gate(rows, input.1.flattened()[order], sortedIndices: true)
            try report.exact(pair.gate, reference, "gate ignores unused NaN up bank")
            report.completed = true
        }

        @Test func gradJVPAndVmapKeepOriginalTransformContracts() throws {
            try O.lane()
            let report = try O.Report("transforms")
            defer { try? report.flush() }
            let banks = O.banks()
            let a = try O.model(banks)
            let b = try O.model(banks, original: true)
            let input = O.inputs()
            func body(_ model: SwitchGLU) -> ([MLXArray]) -> [MLXArray] {
                { values in
                    // Even captured constant activations must refuse raw Metal
                    // kernels while the function is traced for a transform.
                    #expect(
                        Gemma4RowMappedNAX.tryPair(
                            input.0, indices: input.1,
                            gate: model.gateProj!, up: model.upProj!, enabled: true) == nil)
                    return [
                        model.callAndWeightedReduce(
                            input.0, input.1, weights: values[0],
                            fuseSortedReduction: false, isProductionPrefill: true)
                    ]
                }
            }
            // Differentiation is with respect to post-routing weights. Broadcast
            // activation VJP uses pre-existing atomic reductions and is not an
            // exact deterministic oracle; no stop_gradient or tolerance is added.
            try O.compareOutcomes(
                O.outcome { [grad { sum(body(a)([$0])[0]) }(input.2)] },
                O.outcome { [grad { sum(body(b)([$0])[0]) }(input.2)] }, report: report,
                label: "routing-weight grad")
            let tangent = ones(like: input.2)
            try O.compareOutcomes(
                O.outcome {
                    let result = jvp(body(a), primals: [input.2], tangents: [tangent])
                    return result.0 + result.1
                },
                O.outcome {
                    let result = jvp(body(b), primals: [input.2], tangents: [tangent])
                    return result.0 + result.1
                }, report: report, label: "routing-weight JVP")
            func inputBody(_ model: SwitchGLU) -> ([MLXArray]) -> [MLXArray] {
                { v in [O.call(model, (v[0], input.1, input.2))] }
            }
            try O.compareOutcomes(
                O.outcome {
                    let r = jvp(inputBody(a), primals: [input.0], tangents: [ones(like: input.0)])
                    return r.0 + r.1
                },
                O.outcome {
                    let r = jvp(inputBody(b), primals: [input.0], tangents: [ones(like: input.0)])
                    return r.0 + r.1
                }, report: report, label: "activation JVP")
            let batch = stacked([input.2, input.2 * Float(0.5)])
            try O.compareOutcomes(
                O.outcome { vmap(body(a))([batch]) },
                O.outcome { vmap(body(b))([batch]) }, report: report, label: "weight vmap")
            report.completed = true
        }

        @Test func exportAndImportPreserveOriginalGraphOrOriginalError() throws {
            try O.lane()
            let report = try O.Report("export")
            defer { try? report.flush() }
            let banks = O.banks()
            let input = O.inputs()
            let names = ["gate_proj", "up_proj", "down_proj"].flatMap { path in
                ["weight", "scales", "biases"].map { path + "." + $0 }
            }
            let args =
                [input.0, input.1, input.2] + banks.flatMap { [$0.weight, $0.scales, $0.biases] }
            func roundTrip(_ original: Bool) throws -> [MLXArray] {
                let model = try O.model(banks, original: original)
                let file = report.directory.appendingPathComponent(
                    original ? "original.mlxfn" : "candidate.mlxfn")
                defer { try? FileManager.default.removeItem(at: file) }
                let body: ([MLXArray]) -> [MLXArray] = { values in
                    // Packed banks are explicit exporter inputs. The file does
                    // not embed a second copy of the checkpoint's tensor banks.
                    model.update(parameters: .unflattened(Array(zip(names, values.dropFirst(3)))))
                    return [O.call(model, (values[0], values[1], values[2]))]
                }
                try exportFunction(to: file, body)(
                    args[0], args[1], args[2],
                    args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10],
                    args[11])
                let restored = try importFunction(from: file)
                return try restored(
                    args[0], args[1], args[2], args[3], args[4], args[5],
                    args[6], args[7], args[8], args[9], args[10], args[11])
            }
            try O.compareOutcomes(
                O.outcome { try roundTrip(false) }, O.outcome { try roundTrip(true) },
                report: report, label: "export/import complete original graph")
            report.completed = true
        }

        @Test func compiledNestedAndPrebuiltGraphsKeepOriginalContracts() throws {
            try O.lane()
            let report = try O.Report("compiled-prebuilt")
            defer { try? report.flush() }
            let banks = O.banks()
            let a = try O.model(banks)
            let b = try O.model(banks, original: true)
            func body(_ model: SwitchGLU) -> ([MLXArray]) -> [MLXArray] {
                { v in [O.call(model, (v[0], v[1], v[2]))] }
            }
            for shapeless in [false, true] {
                let ca = compile(inputs: [a], shapeless: shapeless, body(a))
                let cb = compile(inputs: [b], shapeless: shapeless, body(b))
                for rows in [64, 65, 64] {
                    let x = O.inputs(rows)
                    let args = [x.0, x.1, x.2]
                    try O.compareOutcomes(
                        O.outcome { ca(args) }, O.outcome { cb(args) },
                        report: report, label: "compile shapeless\(shapeless) T\(rows)")
                }
                let input = O.inputs()
                let args = [input.0, input.1, input.2]
                let batchWeights = stacked([input.2, input.2 * Float(0.5)])
                try O.compareOutcomes(
                    O.outcome {
                        vmap(ca, inAxes: [nil, nil, 0])([input.0, input.1, batchWeights])
                    },
                    O.outcome {
                        vmap(cb, inAxes: [nil, nil, 0])([input.0, input.1, batchWeights])
                    }, report: report, label: "vmap warmed compile shapeless\(shapeless)")
                let na = compile(inputs: [a], { ca($0) })
                let nb = compile(inputs: [b], { cb($0) })
                try O.compareOutcomes(
                    O.outcome { na(args) }, O.outcome { nb(args) },
                    report: report, label: "nested compile shapeless\(shapeless)")
                try O.compareOutcomes(
                    O.outcome {
                        [grad { w in sum(ca([input.0, input.1, w])[0]) }(input.2)]
                    },
                    O.outcome {
                        [grad { w in sum(cb([input.0, input.1, w])[0]) }(input.2)]
                    }, report: report, label: "grad warmed compile shapeless\(shapeless)")
            }
            let input = O.inputs()
            let pa = O.call(a, input)
            let pb = O.call(b, input)
            // Prebuilt outputs are captured constants in these public transforms;
            // preserve that contract and any original error rather than asserting
            // a newly supported derivative through a graph built outside tracing.
            try O.compareOutcomes(
                O.outcome { [grad { w in sum(pa * w) }(ones(like: pa))] },
                O.outcome { [grad { w in sum(pb * w) }(ones(like: pb))] },
                report: report, label: "prebuilt output as captured constant")
            report.completed = true
        }
    }
}
#endif
