import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class NativeQuantizedContiguousCheckpointTests: XCTestCase {
    private let position = 64
    private let chunkSize = 32
    private let quantization = PagedKVQuantizationConfig(
        keyBits: 4, valueBits: 8, groupSize: 64,
        rotationBlockSize: 128, recentTokenCount: 8)

    private func kinds(window: Int, valueWidth: Int = 64) -> [CBv2LayerKind] {
        [
            .init(
                attention: .full, headDim: 128, valueHeadDim: valueWidth,
                kvHeads: 2, queryHeads: 4, modelLayerIndex: 3),
            .init(
                attention: .slidingWindow(window), headDim: 128, valueHeadDim: valueWidth,
                kvHeads: 1, queryHeads: 2, modelLayerIndex: 7),
            .init(
                attention: .full, sharesKVWithLayer: 0, headDim: 128,
                valueHeadDim: valueWidth, kvHeads: 2, queryHeads: 4, modelLayerIndex: 9),
            .init(
                attention: .slidingWindow(window), sharesKVWithLayer: 1, headDim: 128,
                valueHeadDim: valueWidth, kvHeads: 1, queryHeads: 2, modelLayerIndex: 11),
        ]
    }

    private func codec(
        kinds: [CBv2LayerKind], dtype: DType, profile: PagedKVQuantizationConfig?
    ) -> CBv2CompleteCheckpointCodec {
        .init(
            identity: .init(
                modelAggregateHash: "contiguous-checkpoint", promptContractID: "causal",
                buildID: "test", numericsFingerprint: "\(dtype)-\(profile?.identity ?? "native")"),
            layerKinds: kinds, recurrentSpec: nil,
            kvDTypes: Array(repeating: dtype, count: kinds.count), assistant: nil,
            admission: .init(
                layerKinds: kinds, bytesCapacity: 128 << 20,
                config: .init(watermarkFraction: 0, elementBytes: dtype.size)),
            checkpointQuantization: profile)
    }

    private func tensor(
        kind: CBv2LayerKind, values: Bool, start: Int, count: Int, dtype: DType
    ) -> MLXArray {
        let width = values ? kind.valueHeadDim : kind.headDim
        let data: [Float] = (0 ..< kind.kvHeads * count * width).map { index in
            let head = index / (count * width)
            let token = start + (index / width) % count
            let feature = index % width
            if feature == 0, token % 3 == 0 { return -Float.zero }
            let phase = Double(head * 187 + token * 13 + feature * 3) * 0.037
            return Float(sin(phase)) * (values ? 0.75 : 1) + Float(head) * 0.125
        }
        return MLXArray(data, [1, kind.kvHeads, count, width]).asType(dtype)
    }

    private func rows(kinds: [CBv2LayerKind], dtype: DType) -> [CBv2SequenceKV?] {
        kinds.map { kind in
            guard kind.sharesKVWithLayer == nil else { return nil }
            let row: CBv2SequenceKV
            switch kind.attention {
            case .full:
                row = CBv2FullSequenceKV(
                    promptLength: position, maxLength: position + 32,
                    kvHeads: kind.kvHeads, headDim: kind.headDim,
                    valueHeadDim: kind.valueHeadDim)
            case .slidingWindow(let window):
                row = CBv2WindowedSequenceKV(
                    window: window, kvHeads: kind.kvHeads, headDim: kind.headDim,
                    valueHeadDim: kind.valueHeadDim)
            }
            let updated = row.update(
                keys: tensor(kind: kind, values: false, start: 0, count: position, dtype: dtype),
                values: tensor(kind: kind, values: true, start: 0, count: position, dtype: dtype))
            eval(updated.0, updated.1)
            return row
        }
    }

    private var request: CBv2Request {
        .init(
            id: .init(1), promptTokens: Array(repeating: 1, count: position + 5),
            maxTokens: 7, cacheSalt: "tenant")
    }

    private func export(
        codec: CBv2CompleteCheckpointCodec, rows: [CBv2SequenceKV?]
    ) throws -> CBv2CompleteCheckpointExport {
        try codec.export(
            checkpoint: .init(
                position: position, chunkSize: chunkSize, layers: [:], byteCount: 0),
            state: rows, tokens: request.promptTokens, cacheSalt: request.cacheSalt)
    }

    private func transfer(
        _ source: CBv2CompleteCheckpointExport, to sink: CBv2CompleteCheckpointImport
    ) throws {
        let fragments = [1, 3, 17, 257, 1021]
        for (index, descriptor) in source.manifest.tensors.enumerated() {
            var offset = 0
            var fragment = 0
            while offset < descriptor.byteCount {
                let maximum = max(
                    descriptor.dtype.mlxDType.size, fragments[fragment % fragments.count])
                let data = try source.readSegment(
                    tensorIndex: index, byteOffset: offset, maximumBytes: maximum)
                guard !data.isEmpty else { throw CBv2CompleteCheckpointError.incompleteTransfer }
                XCTAssertLessThanOrEqual(data.count, maximum)
                try sink.appendSegment(tensorIndex: index, byteOffset: offset, data: data)
                offset += data.count
                fragment += 1
            }
        }
    }

    private func assertReference(
        original: MLXArray, restored: MLXArray, isKey: Bool, dtype: DType
    ) throws {
        XCTAssertEqual(restored.shape, original.shape)
        XCTAssertEqual(restored.dtype, dtype)
        guard restored.shape == original.shape else { return }
        let native = original.asType(.float32).asArray(Float.self)
        let actual = restored.asType(.float32).asArray(Float.self)
        let heads = original.dim(1)
        let count = original.dim(2)
        let width = original.dim(3)
        let recent = min(count, quantization.recentTokenCount)
        var reference = native
        for head in 0 ..< heads {
            for token in 0 ..< count - recent {
                let start = (head * count + token) * width
                let range = start ..< start + width
                reference.replaceSubrange(
                    range,
                    with: try PagedKVQuantizationReference.roundTrip(
                        Array(native[range]), config: quantization, isKey: isKey))
            }
            let recentStart = (head * count + count - recent) * width
            let recentRange = recentStart ..< (head + 1) * count * width
            XCTAssertEqual(
                actual[recentRange].map(\.bitPattern), native[recentRange].map(\.bitPattern),
                "The recent native band must preserve exact values, including signed zero")
        }
        // The oracle is rounded to the destination dtype, not compared with
        // the original lossy history or an unrounded FP32 reconstruction.
        let expected = MLXArray(reference, original.shape).asType(dtype).asType(.float32)
            .asArray(Float.self)
        let maximumError = zip(actual, expected).reduce(Float.zero) {
            max($0, abs($1.0 - $1.1))
        }
        let tolerance: Float =
            dtype == .float32 ? 0.00001 : (dtype == .float16 ? 1.0 / 512 : 1.0 / 64)
        XCTAssertLessThanOrEqual(maximumError, tolerance)
    }

    private func checkRoundTrip(dtype: DType, window: Int) throws {
        let kinds = kinds(window: window)
        let codec = codec(kinds: kinds, dtype: dtype, profile: quantization)
        let nativeDType = try XCTUnwrap(CBv2CheckpointDType(dtype))
        XCTAssertEqual(codec.checkpointQuantization, quantization)
        XCTAssertNil(codec.pagedConfig)
        let original = rows(kinds: kinds, dtype: dtype)
        let source = try export(codec: codec, rows: original)
        defer { source.close() }
        XCTAssertEqual(
            source.manifest.backendLayout,
            CBv2CompleteCheckpointManifest.nativeQuantizedContiguousHistoricalLayout)
        XCTAssertEqual(source.manifest.checkpointQuantization, quantization)
        XCTAssertEqual(
            source.manifest.checkpointNativeDTypes,
            Array(repeating: nativeDType, count: kinds.count))
        XCTAssertEqual(source.manifest.attentionLayers?.map(\.owner), [0, 1, 0, 1])
        XCTAssertEqual(source.manifest.tensors.map(\.layer), [3, 3, 7, 7])
        XCTAssertEqual(source.manifest.tensors.prefix(2).map(\.dtype), [.uint8, .uint8])
        XCTAssertEqual(
            source.manifest.tensors.suffix(2).map(\.dtype),
            window <= quantization.recentTokenCount
                ? Array(repeating: nativeDType, count: 2)
                : [.uint8, .uint8])
        let wireBytes = try source.manifest.validateStructure()
        let nativeBytes = try codec.nativeTargetDescriptors(position: position).reduce(0) {
            $0 + $1.byteCount
        }
        XCTAssertLessThan(wireBytes, nativeBytes)
        let manifest = try JSONDecoder().decode(
            CBv2CompleteCheckpointManifest.self, from: JSONEncoder().encode(source.manifest))
        XCTAssertEqual(manifest, source.manifest)
        let plan = try codec.plan(manifest: manifest, request: request)
        let capacity = request.promptTokens.count + request.maxTokens
        XCTAssertEqual(
            plan.destinationShapes,
            [
                [1, 2, capacity, 128], [1, 2, capacity, 64],
                [1, 1, window, 128], [1, 1, window, 64],
            ])
        XCTAssertGreaterThan(plan.nativeTargetBytes, wireBytes)
        let sink = try plan.allocate(onRelease: {})
        defer { sink.close() }
        try transfer(source, to: sink)
        let staged = try sink.finish()
        defer { staged.close() }
        let restored = try staged.consumePreparedState { prepared in
            XCTAssertNil(prepared.pagedFrame)
            return prepared.state
        }
        XCTAssertEqual(restored.count, kinds.count)
        guard restored.count == kinds.count else { return }
        XCTAssertTrue(restored[0] is CBv2FullSequenceKV)
        XCTAssertTrue(restored[1] is CBv2WindowedSequenceKV)
        XCTAssertNil(restored[2])
        XCTAssertNil(restored[3])
        for index in [0, 1] {
            let row = try XCTUnwrap(restored[index])
            XCTAssertEqual(row.absoluteOffset, position)
            XCTAssertEqual(row.retainedCount, index == 0 ? position : window)
            let before = try XCTUnwrap(original[index]).snapshot()
            let after = row.snapshot()
            try assertReference(
                original: before.keys, restored: after.keys, isKey: true, dtype: dtype)
            try assertReference(
                original: before.values, restored: after.values, isKey: false, dtype: dtype)
        }
    }

    func testFragmentedAsymmetricRestoreUsesNativeRowsAndExactRecentBand() throws {
        for dtype: DType in [.float16, .bfloat16, .float32] {
            // A short window is intentionally native even with compression on.
            for window in [37, quantization.recentTokenCount] {
                try checkRoundTrip(dtype: dtype, window: window)
            }
        }
    }

    func testAbsentOrUnsupportedProfilePreservesNativeAsymmetricExport() throws {
        let profiles: [(Int, PagedKVQuantizationConfig?)] = [(64, nil), (96, quantization)]
        for (width, profile) in profiles {
            let kinds = kinds(window: 37, valueWidth: width)
            let codec = codec(kinds: kinds, dtype: .float32, profile: profile)
            XCTAssertNil(codec.checkpointQuantization)
            let source = try export(codec: codec, rows: rows(kinds: kinds, dtype: .float32))
            defer { source.close() }
            XCTAssertNil(source.manifest.checkpointQuantization)
            XCTAssertNil(source.manifest.checkpointNativeDTypes)
            XCTAssertEqual(
                source.manifest.backendLayout,
                CBv2CompleteCheckpointManifest.contiguousAsymmetricLayout)
            XCTAssertEqual(
                source.manifest.tensors, try codec.nativeTargetDescriptors(position: position))
            XCTAssertNoThrow(try source.manifest.validateStructure())
        }
    }
}
