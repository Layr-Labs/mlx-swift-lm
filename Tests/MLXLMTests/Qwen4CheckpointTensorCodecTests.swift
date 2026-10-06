import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class Qwen4CheckpointTensorCodecTests: XCTestCase {
    private func geometry(_ dtype: DType = .bfloat16) throws -> CBv2Qwen4CheckpointGeometry {
        try .init(layer: 3, headDim: 8, compressRatio: 4, keyDType: dtype, pooledDType: dtype)
    }

    func testExactSideStateRoundTripAcrossNativeFloatingAndPositionTypes() throws {
        for dtype: DType in [.float16, .bfloat16, .float32] {
            for planes in [1, 3] {
                let position = 9
                let shape = planes == 1 ? [1, position] : [3, 1, position]
                // Values beyond int32 catch accidental narrowing on disk.
                let positions = MLXArray((0..<(planes * position)).map {
                    Int64(Int32.max) + Int64(11 + $0)
                }, shape)
                let snapshot = CBv2Qwen4IndexerSnapshot(tokenCount: position,
                    indexKeys: MLXArray((0..<72).map { Float($0) / 13 }, [1, position, 8]).asType(dtype),
                    positionIds: positions,
                    pooledIndexKeys: MLXArray((0..<16).map { -Float($0) / 7 }, [1, 2, 8]).asType(dtype),
                    pooledIndexBlocks: 2)
                let spec = try geometry(dtype)
                let descriptors = try CBv2Qwen4CheckpointTensorCodec.descriptors(snapshot, geometry: spec)
                let encoded = try JSONEncoder().encode(descriptors)
                let decoded = try JSONDecoder().decode([CBv2CheckpointTensorDescriptor].self, from: encoded)
                XCTAssertEqual(decoded, descriptors)
                let payload = snapshot.arrays.map { ($0.asData().data, $0.shape, $0.dtype) }
                let restoredArrays = payload.map { MLXArray($0.0, $0.1, dtype: $0.2) }
                let restored = try CBv2Qwen4CheckpointTensorCodec.decode(restoredArrays,
                    descriptors: decoded, position: position, geometry: spec)
                XCTAssertEqual(restored.pooledIndexBlocks, 2)
                for (old, new) in zip(snapshot.arrays, restored.arrays) {
                    XCTAssertEqual(old.dtype, new.dtype)
                    XCTAssertEqual(old.asData().data, new.asData().data)
                }
                XCTAssertEqual(descriptors[1].byteCount, planes * position * 8)
            }
        }
    }

    func testLazyPoolRemainsAbsentAndNoEvaluationIsNeededForMetadata() throws {
        let snapshot = CBv2Qwen4IndexerSnapshot(tokenCount: 8,
            indexKeys: MLXArray.zeros([1, 8, 8], dtype: .bfloat16),
            positionIds: MLXArray.arange(8, dtype: .int32).reshaped([1, 8]),
            pooledIndexKeys: nil, pooledIndexBlocks: 0)
        let spec = try geometry()
        let descriptors = try CBv2Qwen4CheckpointTensorCodec.descriptors(snapshot, geometry: spec)
        XCTAssertEqual(descriptors.count, 2)
        XCTAssertNil(try snapshot.indexKeys.evaluatedBufferInfo())
        let decoded = try CBv2Qwen4CheckpointTensorCodec.decode(snapshot.arrays,
            descriptors: descriptors, position: 8, geometry: spec)
        XCTAssertNil(decoded.pooledIndexKeys)
        XCTAssertNil(try snapshot.indexKeys.evaluatedBufferInfo())
        XCTAssertEqual(decoded.pooledIndexBlocks, 0)
    }

    func testMalformedGeometryDtypeRoleAndFuturePoolAreRejected() throws {
        let spec = try geometry()
        let keys = try CBv2CheckpointTensorDescriptor(role: .indexKeys, layer: 3,
            shape: [1, 8, 8], dtype: .bfloat16)
        let positions = try CBv2CheckpointTensorDescriptor(role: .indexPositions, layer: 3,
            shape: [1, 8], dtype: .int32)
        let invalid: [[CBv2CheckpointTensorDescriptor]] = [
            [positions, keys], [keys],
            [keys, try .init(role: .indexPositions, layer: 3, shape: [2, 1, 8], dtype: .int32)],
            [keys, try .init(role: .indexPositions, layer: 3, shape: [1, 8], dtype: .float32)],
            [keys, try .init(role: .indexPositions, layer: 7, shape: [1, 8], dtype: .int32)],
            [try .init(role: .indexKeys, layer: 3, shape: [1, 8, 8], dtype: .float32), positions],
            [keys, positions, try .init(role: .pooledIndexKeys, layer: 3, shape: [1, 3, 8], dtype: .bfloat16)],
        ]
        for descriptors in invalid {
            XCTAssertThrowsError(try CBv2Qwen4CheckpointTensorCodec.validate(
                descriptors, position: 8, geometry: spec))
        }
        XCTAssertThrowsError(try CBv2Qwen4CheckpointGeometry(layer: 3, headDim: 8,
            compressRatio: 0, keyDType: .bfloat16, pooledDType: .bfloat16))
        XCTAssertThrowsError(try geometry(.int64))
        XCTAssertFalse(CBv2CheckpointDType.int64.isFloatingPoint)
        XCTAssertFalse(CBv2CheckpointDType.int32.isFloatingPoint)
        XCTAssertThrowsError(try CBv2Qwen4CheckpointTensorCodec.decode(
            [MLXArray.zeros([1, 8, 8], dtype: .float32), MLXArray.zeros([1, 8], dtype: .int32)],
            descriptors: [keys, positions], position: 8, geometry: spec))
    }

    func testSnapshotCounterMustMatchActualPooledTensor() throws {
        let spec = try geometry()
        for (pool, count): (MLXArray?, Int) in [
            (nil, 1), (MLXArray.zeros([1, 1, 8], dtype: .bfloat16), 0),
            (MLXArray.zeros([1, 2, 8], dtype: .bfloat16), 1)
        ] {
            let snapshot = CBv2Qwen4IndexerSnapshot(tokenCount: 8,
                indexKeys: MLXArray.zeros([1, 8, 8], dtype: .bfloat16),
                positionIds: MLXArray.zeros([1, 8], dtype: .int32),
                pooledIndexKeys: pool, pooledIndexBlocks: count)
            XCTAssertThrowsError(try CBv2Qwen4CheckpointTensorCodec.descriptors(snapshot, geometry: spec))
        }
    }
}
