import Cmlx
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// The head's LOAD construction allowance counts only source payload copies.
/// That is exact only if the loader never evaluates the dense constructor or
/// cast placeholders: a second handle on each placeholder, taken when they are
/// bound, must still be unevaluated after the loader's final evaluation, while
/// every installed parameter is evaluated.
@Suite(.tags(.integration))
struct NemotronSuperMTPLoadBoundTests {
    static func evaluated(_ array: MLXArray) -> Bool {
        var result = false
        _ = _mlx_array_is_available(&result, array.ctx)
        return result
    }

    /// A second handle on the same graph node, unaffected by later in-place
    /// parameter updates of the original wrapper.
    static func probe(_ array: MLXArray) -> MLXArray {
        var handle = mlx_array_new()
        _ = mlx_array_set(&handle, array.ctx)
        return MLXArray(handle)
    }

    @Test(arguments: [DType.bfloat16, .float32])
    func constructorPlaceholdersAreNeverEvaluated(dtype: DType) throws {
        let c = NemotronSuperMTPFixture.configuration()
        let placeholders = try NemotronSuperMTPModule(
            configuration: c, calibration: .init(keyScale: 1, valueScale: 1), dtype: dtype
        ).parameters().flattened().count
        try NemotronSuperMTPFixture.withSource(
            NemotronSuperMTPFixture.headTensors(.random, seed: 3, dtype: dtype)
        ) { source in
            let quote = try NemotronSuperMTPModule.loadQuote(configuration: c, source: source)
            #expect(quote.constructionBytes == 4 * quote.payloadBytes)
            let scope = NativeConstructionScope()
            var probes: [MLXArray] = []
            scope.testingBoundary = { name, scope in
                if name == "mtpHead.placeholdersBound" {
                    probes = scope.retainedArraysForTesting.map(Self.probe)
                }
            }
            try scope.withPhase(.filesystemLoad) {
                try scope.capture(StreamOrDevice.default.stream)
                try scope.capture(StreamOrDevice.cpu.stream)
                try scope.willSubmit()
                // An allowance one byte short is refused before any payload read.
                #expect(throws: NemotronSuperMTPModule.Failure.self) {
                    try NemotronSuperMTPModule.load(
                        configuration: c, source: source, maximumBytes: 1 << 24,
                        constructionAllowanceBytes: quote.constructionBytes - 1,
                        construction: scope, cachePolicy: .fp8, dtype: dtype)
                }
                #expect(probes.isEmpty)
                let head = try NemotronSuperMTPModule.load(
                    configuration: c, source: source, maximumBytes: 1 << 24,
                    constructionAllowanceBytes: quote.constructionBytes,
                    construction: scope, cachePolicy: .fp8, dtype: dtype)
                try scope.retainOwner(head)
                // Only the placeholders are retained when the hook runs.
                #expect(probes.count == placeholders)
                #expect(probes.allSatisfy { !Self.evaluated($0) })
                #expect(head.parameters().flattened().allSatisfy { Self.evaluated($0.1) })
                // Control: the probe does observe an evaluation.
                let control = try #require(probes.last)
                eval(control)
                #expect(Self.evaluated(control))
            }
        }
    }
}
