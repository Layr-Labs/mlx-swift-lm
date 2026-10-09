import MLX
import Testing

@testable import MLXLMCommon

@Suite("Resolved packed kernel preflight", .serialized)
struct PagedQuantizedKernelSmokeTests {
    @Test(arguments: [DType.bfloat16, .float32])
    func seventeenSegmentBF16ModelGeometryUsesEvaluatedLayout(queryDType: DType) throws {
        let shape = PagedQuantizedKernelSmokeShape(
            headDim: 256, kvHeads: 2,
            queryHeads: 16, hasSinks: false, dtype: .bfloat16, queryDType: queryDType)
        let coverage = try PagedQuantizedKernelSmoke.runtimeSmoke(
            shapes: [shape], quantization: .init())
        let completed = try #require(coverage[shape])
        #expect(completed.contains("part-n17-q1-mask0-cap0"))
        #expect(completed.contains("part-n17-q8-mask1-cap1"))
        #expect(completed.contains("selected-n17"))
    }

    @Test func serializedShapePreservesNativeAndQueryDTypes() throws {
        let shape = PagedQuantizedKernelSmokeShape(
            headDim: 256, kvHeads: 2,
            queryHeads: 8, hasSinks: false, dtype: .bfloat16, queryDType: .float32)
        #expect(shape.argumentValue == "256:2:8:0:0:0:bf16:f32")
        let restored = try PagedQuantizedKernelSmokeShape(argumentValue: shape.argumentValue)
        #expect(restored == shape)
        #expect(throws: PagedAttentionKernelSmokeError.self) {
            try PagedQuantizedKernelSmokeShape(argumentValue: "256:2:8:0:0:0:bf16")
        }
        #expect(throws: PagedAttentionKernelSmokeError.self) {
            try PagedQuantizedKernelSmokeShape(argumentValue: "256:2:8:0:0:0:int8:f32")
        }
    }

    @Test func selectedNativeOwnersAndSmallWindowsNeverBecomePackedProbes() throws {
        let kinds = [
            CBv2LayerKind(attention: .full, headDim: 256, kvHeads: 2, queryHeads: 8),
            CBv2LayerKind(
                attention: .slidingWindow(128), hasSinks: true,
                headDim: 64, kvHeads: 8, queryHeads: 64),
            CBv2LayerKind(attention: .full, headDim: 512, kvHeads: 2, queryHeads: 16),
            CBv2LayerKind(
                attention: .full, sharesKVWithLayer: 2,
                headDim: 512, kvHeads: 2, queryHeads: 32),
        ]
        let selected = try PagedQuantizedKernelSmoke.smokeShapes(
            layerKinds: kinds,
            quantization: .init(), nativeLayerIndices: [2])
        #expect(selected.count == 9)
        let selectedDimensionMatches = selected.allSatisfy { $0.headDim == 256 }
        #expect(selectedDimensionMatches)
        #expect(Set(selected.map(\.dtype)) == [.float16, .bfloat16, .float32])
        #expect(Set(selected.map(\.queryDType)) == [.float16, .bfloat16, .float32])
        let all = try PagedQuantizedKernelSmoke.smokeShapes(
            layerKinds: kinds, quantization: .init())
        #expect(all.count == 27)
        let borrowedQueryGeometryCovered = all.contains { $0.queryHeads == 32 }
        #expect(borrowedQueryGeometryCovered)
        #expect(throws: PagedAttentionKernelSmokeError.self) {
            try PagedQuantizedKernelSmoke.smokeShapes(
                layerKinds: kinds,
                quantization: .init(), nativeLayerIndices: [3])
        }
    }

    @Test(arguments: [
        PagedKVQuantizationConfig(), .init(keyBits: 8, valueBits: 4),
        .init(keyBits: 8, valueBits: 8),
    ])
    func actualPackedOperationsCompileAndDispatchEveryBindingMaskAndQueryBlock(
        quantization: PagedKVQuantizationConfig
    ) throws {
        let shape = PagedQuantizedKernelSmokeShape(
            headDim: 64, kvHeads: 1,
            queryHeads: 2, hasSinks: false, dtype: .float32)
        let coverage = try PagedQuantizedKernelSmoke.runtimeSmoke(
            shapes: [shape], quantization: quantization)
        let completed = try #require(coverage[shape])
        for stage in ["packed-write", "native-gather", "packed-gather", "update-q1", "update-q8"] {
            #expect(completed.contains(stage))
        }
        for bindings in [1, 4, 8, 17] {
            #expect(completed.contains("selected-n\(bindings)"))
            for queries in [1, 8] {
                for masked in [0, 1] {
                    for softcap in [0, 1] {
                        #expect(
                            completed.contains(
                                "part-n\(bindings)-q\(queries)-mask\(masked)-cap\(softcap)"))
                    }
                }
            }
        }
    }
}
