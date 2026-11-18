import Foundation
import MLX
@_spi(MaterializedModule) import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

private final class Gemma4OffsetQuantizedLinear: QuantizedLinear {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        super.callAsFunction(x) + Float(0.25)
    }
}

@Suite("Gemma 4 compiled feed-forward", .tags(.integration), .serialized)
struct Gemma4CompiledFeedforwardTests {
    private func layer(expertGroupSize: Int = 64) throws -> Gemma4DecoderLayer {
        let configuration = try JSONDecoder().decode(
            Gemma4TextConfiguration.self,
            from: Data(
                """
                {
                    "model_type": "gemma4_text", "vocab_size": 128,
                    "hidden_size": 128, "intermediate_size": 256,
                    "num_hidden_layers": 1, "num_attention_heads": 2,
                    "num_key_value_heads": 1, "head_dim": 64, "global_head_dim": 64,
                    "layer_types": ["full_attention"], "num_kv_shared_layers": 0,
                    "hidden_size_per_layer_input": 0, "vocab_size_per_layer_input": 0,
                    "enable_moe_block": true, "num_experts": 128,
                    "top_k_experts": 8, "moe_intermediate_size": 64
                }
                """.utf8))
        let layer = Gemma4DecoderLayer(configuration, layerIdx: 0)
        let parameters = layer.parameters().flattened().sorted { $0.0 < $1.0 }
        var weights = [String: MLXArray]()
        for (index, parameter) in parameters.enumerated() {
            if parameter.1.ndim < 2 {
                weights[parameter.0] = parameter.1.asType(.bfloat16)
            } else {
                weights[parameter.0] =
                    (MLXRandom.normal(parameter.1.shape, key: MLXRandom.key(UInt64(9400 + index)))
                    / 16).asType(.bfloat16)
            }
        }
        try layer.update(parameters: .unflattened(weights), verify: .all)
        quantize(model: layer) { path, _ in
            path.hasPrefix("experts.")
                ? (groupSize: expertGroupSize, bits: 4, mode: .affine)
                : (groupSize: 64, bits: 8, mode: .affine)
        }
        layer.train(false)
        eval(layer)
        return layer
    }

    private func input(
        rows: Int = 1, sequenceWidth: Int = 1, dtype: DType = .bfloat16
    ) -> MLXArray {
        let x = MLXRandom.normal([rows, sequenceWidth, 128], key: MLXRandom.key(9500)).asType(dtype)
        eval(x)
        return x
    }

    private func expectExact(_ actual: MLXArray, _ expected: MLXArray) {
        eval(actual, expected)
        #expect(actual.shape == expected.shape)
        #expect(actual.dtype == expected.dtype)
        #expect(actual.asData(access: .copy).data == expected.asData(access: .copy).data)
    }

    /// Whether the trace is on, and whether it covers sorted sizes, in this process.
    private var compiledEnabled: Bool {
        MLXHardwareInfo.isCompiledDecodeSupported
            && (ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_COMPILED_FEEDFORWARD"]
                .map({ ["1", "true", "yes", "on"].contains($0.lowercased()) }) ?? true)
    }

    private func expectedCompiled(rows: Int, width: Int, eligible: Bool) -> Bool {
        let sorted = rows * width * 8 >= 64
        return eligible && compiledEnabled
            && (!sorted || Gemma4CompiledFeedforwardCache.sortedEnabled)
    }

    private func compare(_ layer: Gemma4DecoderLayer, _ x: MLXArray, prefill: Bool = false) {
        expectExact(
            layer.feedforwardWithCompilation(x, isExpertPrefill: prefill),
            layer.feedforward(x, isExpertPrefill: prefill))
    }

    @Test func shapeBucketsAndTiedRoutingPreserveBytes() throws {
        let layer = try layer()
        #expect(layer.feedforwardCompilationState() != nil)
        let names = layer.parameters().flattened().map(\.0).sorted()
        for rows in [1, 2, 4, 7, 8, 12, 16, 8, 4, 2, 1] {
            compare(layer, input(rows: rows))
        }
        let experts = try #require(layer.modules().compactMap { $0 as? SwitchGLU }.first)
        for enabled in [false, true, false] {
            experts.suppliesHostRowIndices = enabled
            compare(layer, input(rows: 2))
        }
        let requested = ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_COMPILED_FEEDFORWARD"]
        if MLXHardwareInfo.isCompiledDecodeSupported,
            requested.map({ ["1", "true", "yes", "on"].contains($0.lowercased()) }) ?? true
        {
            #expect(layer.feedforwardTraceIsCompiled)
        }
        let parameters = Dictionary(uniqueKeysWithValues: layer.parameters().flattened())
        let scales = try #require(parameters["router.proj.scales"])
        let biases = try #require(parameters["router.proj.biases"])
        layer.update(
            parameters: .unflattened([
                "router.proj.scales": MLXArray.zeros(scales.shape, dtype: .bfloat16),
                "router.proj.biases": MLXArray.zeros(biases.shape, dtype: .bfloat16),
            ]))
        // All expert scores tie. Keep argPartition's selected order and
        // the subsequent weighted reduction exactly as the eager graph.
        for rows in [1, 2, 4, 7, 8, 16] { compare(layer, input(rows: rows)) }
        for rows in [1, 2, 3, 4, 8] { compare(layer, input(rows: rows, sequenceWidth: 2)) }
        #expect(layer.parameters().flattened().map(\.0).sorted() == names)
    }

    @Test func verificationSequenceWidthsPreserveBytesAndRoutingBound() throws {
        let layer = try layer()
        let requested = ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_COMPILED_FEEDFORWARD"]
        let enabled =
            MLXHardwareInfo.isCompiledDecodeSupported
            && (requested.map({ ["1", "true", "yes", "on"].contains($0.lowercased()) }) ?? true)
        print("[Gemma4 FFN sequence2] requested=\(requested ?? "unset") enabled=\(enabled)")
        for (rows, width, eligible) in [
            (1, 1, true), (1, 2, true), (2, 2, true), (3, 2, true),
            (4, 2, true), (8, 2, true), (8, 1, true), (16, 1, true),
            (1, 3, false), (9, 2, false), (17, 1, false),
        ] {
            layer.invalidateCompiledTraces()
            let x = input(rows: rows, sequenceWidth: width)
            compare(layer, x)
            #expect(
                layer.feedforwardTraceIsCompiled
                    == expectedCompiled(rows: rows, width: width, eligible: eligible))
            let changed = x + Float(0.125)
            eval(changed)
            compare(layer, changed)
        }
        // Reuse shape-specialized traces across ordinary decode and the
        // original two-column verification layout, without flattening/padding.
        for (rows, width) in [(1, 2), (2, 2), (4, 2), (8, 1), (3, 2), (1, 1), (16, 1), (1, 2)] {
            compare(layer, input(rows: rows, sequenceWidth: width))
        }
        layer.invalidateCompiledTraces()
        compare(layer, input(rows: 2, sequenceWidth: 2), prefill: true)
        #expect(!layer.feedforwardTraceIsCompiled)
    }

    /// QAT-lattice expert banks (4 bits in groups of 32, with 8-bit dense
    /// projections) take the compiled trace with the same bytes as eager.
    @Test func latticeExpertBanksCompileWithTheSameBytes() throws {
        let layer = try layer(expertGroupSize: 32)
        let experts = try #require(layer.modules().compactMap { $0 as? SwitchGLU }.first)
        let banks = try [experts.gateProj, experts.upProj, experts.downProj].map {
            try #require($0 as? QuantizedSwitchLinear)
        }
        #expect(banks.allSatisfy { $0.bits == 4 && $0.groupSize == 32 })
        #expect(layer.feedforwardCompilationState() != nil)
        let requested = ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_COMPILED_FEEDFORWARD"]
        let enabled =
            MLXHardwareInfo.isCompiledDecodeSupported
            && (requested.map({ ["1", "true", "yes", "on"].contains($0.lowercased()) }) ?? true)
        print("[Gemma4 FFN lattice] requested=\(requested ?? "unset") enabled=\(enabled)")
        for (rows, width, eligible) in [
            (1, 1, true), (4, 1, true), (7, 1, true), (1, 2, true), (3, 2, true),
            (8, 1, true), (4, 2, true), (12, 1, true), (16, 1, true), (8, 2, true),
            (17, 1, false), (9, 2, false),
        ] {
            layer.invalidateCompiledTraces()
            let x = input(rows: rows, sequenceWidth: width)
            compare(layer, x)
            #expect(
                layer.feedforwardTraceIsCompiled
                    == expectedCompiled(rows: rows, width: width, eligible: eligible))
            let changed = x + Float(0.125)
            eval(changed)
            compare(layer, changed)
        }
        layer.invalidateCompiledTraces()
        compare(layer, input(rows: 2, sequenceWidth: 2), prefill: true)
        #expect(!layer.feedforwardTraceIsCompiled)
    }

    /// 8 to 16 rows sort their expert assignments. The switch off, or a scoped
    /// batch-decode frame, keeps those rows on the eager graph with the same bytes.
    @Test func sortedRowsFollowTheirSwitchAndScopes() throws {
        let layer = try layer(expertGroupSize: 32)
        #expect(Gemma4CompiledFeedforwardCache.sortedEnabled(nil))
        for raw in ["1", "true", "YES", "on"] {
            #expect(Gemma4CompiledFeedforwardCache.sortedEnabled(raw))
        }
        for raw in ["0", "false", "no", "off", ""] {
            #expect(!Gemma4CompiledFeedforwardCache.sortedEnabled(raw))
        }
        let enabled = compiledEnabled
        for sorted in [false, true] {
            Gemma4CompiledFeedforwardCache.$sortedOverrideForTesting.withValue(sorted) {
                for (rows, width) in [(8, 1), (4, 2), (16, 1)] {
                    layer.invalidateCompiledTraces()
                    compare(layer, input(rows: rows, sequenceWidth: width))
                    #expect(layer.feedforwardTraceIsCompiled == (sorted && enabled))
                }
                layer.invalidateCompiledTraces()
                compare(layer, input(rows: 7))
                #expect(layer.feedforwardTraceIsCompiled == enabled)
            }
        }
        Gemma4CompiledFeedforwardCache.$sortedOverrideForTesting.withValue(true) {
            for scoped in [false, true] {
                layer.invalidateCompiledTraces()
                let x = input(rows: 8)
                Gemma4BatchDecodeProjection.$active.withValue(scoped) {
                    compare(layer, x)
                }
                #expect(layer.feedforwardTraceIsCompiled == (!scoped && enabled))
            }
            // An unscoped [8, 1] trace is never reused by a scoped call of that shape.
            layer.invalidateCompiledTraces()
            let x = input(rows: 8)
            compare(layer, x)
            #expect(layer.feedforwardTraceIsCompiled == enabled)
            Gemma4BatchDecodeProjection.$active.withValue(true) {
                compare(layer, x)
            }
        }
    }

    @Test func parameterUpdatesAndModuleReplacementsRemainLive() throws {
        let layer = try layer()
        let x = input(rows: 2)
        compare(layer, x)
        let parameters = Dictionary(uniqueKeysWithValues: layer.parameters().flattened())
        let weight = try #require(parameters["post_feedforward_layernorm.weight"])
        layer.update(
            parameters: .unflattened([
                "post_feedforward_layernorm.weight": weight + Float(0.125)
            ]))
        compare(layer, x)
        weight._updateInternal(weight - Float(0.25))
        compare(layer, x)

        let replacement = try self.layer()
        let up = try #require(replacement.namedModules().first { $0.0 == "mlp.up_proj" }?.1)
        let upScales = try #require((up as? QuantizedLinear)?.scales)
        up.update(parameters: .unflattened(["scales": upScales * Float(1.25)]))
        // Replacing the child directly bypasses a decoder-level update
        // override. The retained topology signature must still retrace.
        let mlp = try #require(layer.namedModules().first { $0.0 == "mlp" }?.1)
        mlp.update(modules: .unflattened([("up_proj", up)]))
        compare(layer, x)

        weak var oldNorm = layer.preFeedforwardLayernorm
        let norm = RMSNorm(dimensions: 128, eps: 0.25)
        norm.update(
            parameters: .unflattened([
                "weight": MLXArray.ones([128], dtype: .bfloat16)
            ]))
        norm.train(false)
        layer.update(modules: .unflattened([("pre_feedforward_layernorm", norm)]))
        compare(layer, x, prefill: true)
        #expect(oldNorm == nil)
        #expect(!layer.feedforwardTraceIsCompiled)
        compare(layer, x)
        layer.invalidateCompiledTraces()
        #expect(!layer.feedforwardTraceIsCompiled)
        compare(layer, x)
    }

    @Test func unsupportedInputsAndTrainingKeepEagerExecution() throws {
        let layer = try layer()
        compare(layer, input(rows: 17))
        compare(layer, input(rows: 8, dtype: .float32))
        compare(layer, input(rows: 2, dtype: .float32))
        compare(layer, input(rows: 2, dtype: .float16))
        compare(layer, input(rows: 2), prefill: true)
        let prompt = MLXRandom.normal([1, 4, 128], key: MLXRandom.key(9600)).asType(.bfloat16)
        eval(prompt)
        compare(layer, prompt)
        #expect(!layer.feedforwardTraceIsCompiled)
        layer.train(true)
        compare(layer, input())
        #expect(layer.feedforwardCompilationState() == nil)
        #expect(!layer.feedforwardTraceIsCompiled)
        layer.train(false)
        Stream.withNewDefaultStream(device: .cpu) {
            compare(layer, input())
            #expect(!layer.feedforwardTraceIsCompiled)
        }
    }

    @Test func unsupportedReplacementDoesNotReuseAnOldTrace() throws {
        let layer = try layer()
        let x = input()
        compare(layer, x)
        weak var oldUp = layer.namedModules().first { $0.0 == "mlp.up_proj" }?.1
        let replacement = Linear(128, 256, bias: false)
        replacement.update(
            parameters: .unflattened([
                "weight": MLXRandom.normal([256, 128], key: MLXRandom.key(9700)).asType(.bfloat16)
            ]))
        replacement.train(false)
        layer.update(modules: .unflattened([("mlp.up_proj", replacement)]))
        #expect(layer.feedforwardCompilationState() == nil)
        compare(layer, input(rows: 8))
        #expect(oldUp == nil)
        #expect(!layer.feedforwardTraceIsCompiled)
        compare(layer, x)

        let custom = Gemma4OffsetQuantizedLinear(
            weight: MLXRandom.normal([256, 128], key: MLXRandom.key(9800)).asType(.bfloat16),
            bias: nil, groupSize: 64, bits: 8)
        custom.train(false)
        layer.update(modules: .unflattened([("mlp.up_proj", custom)]))
        #expect(layer.feedforwardCompilationState() == nil)
        compare(layer, x)
    }

    @Test func diagnosticsKeepPerInvocationComponentReceipts() throws {
        let layer = try layer()
        let x = input(rows: 2)
        let recorder = CBv2ForwardShapeRecorder()
        try recorder.reset()
        let step = recorder.beginStep()
        let output = CBv2ForwardShapeObservation.dispatch(step: step, phase: .decode) {
            let call = CBv2ForwardShapeObservation.beginTarget(liveBatchRows: 2, sequenceWidth: 1)
            defer { call?.end() }
            return layer.feedforwardWithCompilation(x, isExpertPrefill: false)
        }
        step.attach()
        eval(output)
        step.complete()
        #expect(!layer.feedforwardTraceIsCompiled)
        #expect(recorder.snapshot().entries.contains { $0.axes.kind == .compiledComponent })
        expectExact(output, layer.feedforward(x, isExpertPrefill: false))
    }

    private func expectUnsupportedRoutingGradient(_ body: () throws -> Void) {
        do {
            try body()
            Issue.record("Routing-index differentiation unexpectedly succeeded")
        } catch let MLXError.caught(message) {
            #expect(message.contains("[gather] Cannot calculate VJP with respect to indices."))
        } catch {
            Issue.record("Unexpected differentiation error: \(error)")
        }
    }

    @Test func inputGradientsRetainTheExistingRoutingLimitation() throws {
        for compiled in [false, true] {
            let layer = try layer()
            let x = input()
            compare(layer, x)
            // The original router differentiates through argPartition indices.
            // Preserve that existing failure rather than change routing math.
            expectUnsupportedRoutingGradient {
                try MLX.withError { errors in
                    let gradient = grad { (inputs: [MLXArray]) -> [MLXArray] in
                        let output =
                            compiled
                            ? layer.feedforwardWithCompilation(inputs[0], isExpertPrefill: false)
                            : layer.feedforward(inputs[0], isExpertPrefill: false)
                        return [sum(output)]
                    }([x])
                    try errors.check()
                    eval(gradient)
                }
            }
        }
    }

    @Test func routerScaleGradientsRetainTheExistingRoutingLimitation() throws {
        for compiled in [false, true] {
            let layer = try layer()
            let x = input()
            compare(layer, x)
            layer.freeze()
            let router = try #require(layer.namedModules().first { $0.0 == "router" }?.1)
            router.unfreeze(recursive: false, keys: ["scale"])
            #expect(layer.trainableParameters().flattened().map(\.0) == ["router.scale"])
            let derivative = valueAndGrad(model: layer) {
                (layer: Gemma4DecoderLayer, inputs: [MLXArray]) -> [MLXArray] in
                let output =
                    compiled
                    ? layer.feedforwardWithCompilation(inputs[0], isExpertPrefill: false)
                    : layer.feedforward(inputs[0], isExpertPrefill: false)
                return [sum(output)]
            }
            expectUnsupportedRoutingGradient {
                try MLX.withError { errors in
                    let result = derivative(layer, [x])
                    try errors.check()
                    eval(result.0, result.1)
                }
            }
        }
    }

    @Test func postRoutingParameterGradientsPreserveBytesAfterWarmup() throws {
        func differentiableLayer() throws -> Gemma4DecoderLayer {
            let result = try layer()
            result.freeze()
            result.postFeedforwardLayernorm.unfreeze(recursive: false, keys: ["weight"])
            try #require(result.postFeedforwardLayernorm1).unfreeze(
                recursive: false, keys: ["weight"])
            let router = try #require(result.namedModules().first { $0.0 == "router" }?.1)
            router.unfreeze(recursive: false, keys: ["per_expert_scale"])
            return result
        }
        for width in [1, 2] {
            let eagerLayer = try differentiableLayer()
            let compiledLayer = try differentiableLayer()
            let x = input(sequenceWidth: width)
            compare(compiledLayer, x)
            let keys = [
                "post_feedforward_layernorm.weight", "post_feedforward_layernorm_1.weight",
                "router.per_expert_scale",
            ]
            #expect(eagerLayer.trainableParameters().flattened().map(\.0).sorted() == keys)
            #expect(compiledLayer.trainableParameters().flattened().map(\.0).sorted() == keys)
            // These parameters affect values after discrete expert selection;
            // this does not assert support for differentiating routing indices.
            let eager = valueAndGrad(model: eagerLayer) {
                (layer: Gemma4DecoderLayer, inputs: [MLXArray]) -> [MLXArray] in
                [sum(layer.feedforward(inputs[0], isExpertPrefill: false))]
            }
            let compiled = valueAndGrad(model: compiledLayer) {
                (layer: Gemma4DecoderLayer, inputs: [MLXArray]) -> [MLXArray] in
                [sum(layer.feedforwardWithCompilation(inputs[0], isExpertPrefill: false))]
            }
            let expected = try MLX.withError { errors in
                let result = eager(eagerLayer, [x])
                try errors.check()
                eval(result.0, result.1)
                return Dictionary(uniqueKeysWithValues: result.1.flattened())
            }
            let actual = try MLX.withError { errors in
                let result = compiled(compiledLayer, [x])
                try errors.check()
                eval(result.0, result.1)
                return Dictionary(uniqueKeysWithValues: result.1.flattened())
            }
            for key in keys {
                let reference = try #require(expected[key])
                #expect(abs(reference).max().item(Float.self) > 0)
                expectExact(try #require(actual[key]), reference)
            }
        }
    }

    @Test func materializingAfterWarmupRebindsSealedWeights() throws {
        func warmedLayer() throws -> Gemma4DecoderLayer {
            let result = try layer()
            compare(result, input())
            return result
        }
        // Consume the returned decoder without retaining an unsealed alias.
        let sealed = MaterializedModule(try warmedLayer())
        compare(sealed._base, input())
        compare(sealed._base, input(rows: 2))
        #expect(sealed._base.feedforwardCompilationState() != nil)
    }

    @Test func cacheDoesNotKeepItsDecoderAlive() throws {
        weak var retained: Gemma4DecoderLayer?
        do {
            let layer = try layer()
            retained = layer
            eval(layer.feedforwardWithCompilation(input(), isExpertPrefill: false))
        }
        #expect(retained == nil)
    }
}
