import MLXLMCommon
import Testing

extension UnitTests {
    struct ModelOptProjectionLayoutTests {
        typealias Tensor = ModelOptProjectionLayout.Tensor
        @Test func officialExpertLayoutAndScalarVariants() throws {
            // Actual header dimensions: pinned shard88, layer87 expert0 up_proj.
            let tensors: [String: Tensor] = [
                "weight": Tensor(dtype: "U8", shape: [2688, 512], byteCount: 1376256),
                "weight_scale": Tensor(dtype: "F8_E4M3", shape: [2688, 64], byteCount: 172032),
                "weight_scale_2": Tensor(dtype: "F32", shape: [], byteCount: 4),
                "input_scale": Tensor(dtype: "F32", shape: [1], byteCount: 4)]
            let result = try ModelOptProjectionLayout(tensors: tensors, inputDimensions: 1024, outputDimensions: 2688)
            #expect(result.storage == .nvfp4)
            for key in tensors.keys {
                var incomplete = tensors; incomplete.removeValue(forKey: key)
                #expect(throws: (any Error).self) {
                    try ModelOptProjectionLayout(tensors: incomplete, inputDimensions: 1024, outputDimensions: 2688)
                }
            }
        }
        @Test func fp8AndDenseRemainDistinct() throws {
            let scalar = Tensor(dtype: "F32", shape: [], byteCount: 4)
            let fp8 = try ModelOptProjectionLayout(tensors: [
                "weight": Tensor(dtype: "F8_E4M3", shape: [4, 16], byteCount: 64),
                "weight_scale": scalar, "input_scale": scalar], inputDimensions: 16, outputDimensions: 4)
            #expect(fp8.storage == .fp8E4M3)
            let dense = try ModelOptProjectionLayout(tensors: [
                "weight": Tensor(dtype: "BF16", shape: [4, 16], byteCount: 128)],
                inputDimensions: 16, outputDimensions: 4)
            #expect(dense.storage == .bfloat16)
        }
        @Test func rejectsMislabelledShapesBytesAndUnknownSidecars() throws {
            let valid = Tensor(dtype: "BF16", shape: [4, 16], byteCount: 128)
            for tensor in [Tensor(dtype: "BF16", shape: [16, 4], byteCount: 128),
                Tensor(dtype: "BF16", shape: [4, 16], byteCount: 64),
                Tensor(dtype: "U32", shape: [4, 2], byteCount: 32)] {
                #expect(throws: (any Error).self) {
                    try ModelOptProjectionLayout(tensors: ["weight": tensor], inputDimensions: 16, outputDimensions: 4)
                }
            }
            #expect(throws: (any Error).self) {
                try ModelOptProjectionLayout(tensors: ["weight": valid, "scales": valid], inputDimensions: 16, outputDimensions: 4)
            }
            #expect(throws: (any Error).self) {
                try ModelOptProjectionLayout(tensors: ["weight": valid], inputDimensions: Int.max, outputDimensions: 4)
            }
        }
    }
}
