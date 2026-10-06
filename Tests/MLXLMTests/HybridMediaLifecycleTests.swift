import Foundation
import MLX
import XCTest
@testable import MLXLMCommon

/// Extends the existing exact resident-cache lifecycle fixture with the prior
/// media identity/QSA-state contract. These are native mechanics, not images.
final class HybridMediaLifecycleTests: XCTestCase {
    func testIdentityBoundMediaStateSurvivesPinAndCloseThenReleasesExactly() async throws {
        let media = try CBv2HybridPrefixIdentity(digest: Data(repeating: 0x61, count: 32))
        let changed = try CBv2HybridPrefixIdentity(digest: Data(repeating: 0x62, count: 32))
        let request = CBv2Request(id: .init(31), promptTokens: Array(0..<12), maxTokens: 1,
            cacheSalt: "tenant", hybridPrefixIdentity: media)
        var other = request
        other.hybridPrefixIdentity = changed
        let spec = CBv2RecurrentStateSpec(layers: [.init(modelLayerIndex: 0,
            convShape: [1, 2], convDType: .float32, ssmShape: [1, 2], ssmDType: .float32)])
        let cache = CBv2HybridPrefixCache(config: .init(maximumBytes: 16384, modelID: "qwen4",
            promptContractID: "test", buildID: "test"))
        let keys = MLXArray([Float(0), -0.0, 1, -1, 2, -2, 3, -3], [1, 4, 2]).asType(.bfloat16)
        let positions = MLXArray((0..<12).map { Int64(Int32.max) + Int64($0) }, [3, 1, 4])
        let side = CBv2Qwen4IndexerSnapshot(tokenCount: 4, indexKeys: keys, positionIds: positions,
            pooledIndexKeys: MLXArray.ones([1, 1, 2], dtype: .bfloat16), pooledIndexBlocks: 1)
        let layers: [Int: CBv2RecurrentLayerState] = [0: .init(
            conv: MLXArray([Float(2), 3], [1, 2]), ssm: MLXArray([Float(5), 7], [1, 2]))]
        let roots = cache.capture(requestID: request.id, position: 4, chunkSize: 4,
            spec: spec, layers: layers, qwen4: [3: side], mediaIdentity: media, mediaTargetOnly: true)
        XCTAssertFalse(roots.isEmpty)
        eval(roots)
        let retained = cache.stats.retainedBytes
        XCTAssertEqual(retained, 16 + side.arrays.reduce(0) { $0 + $1.nbytes })
        let row = MLXArray.zeros([1, 1, 12, 2])
        asyncEval(row)
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            cache.publish(requestID: request.id, tokens: request.promptTokens,
                cacheSalt: request.checkpointCacheSalt,
                kv: [(keys: row, values: row, offset: 12)], backingBytes: row.nbytes * 2) { done.resume() }
        }
        XCTAssertNil(cache.lookup(tokens: request.promptTokens, cacheSalt: other.checkpointCacheSalt, maximumChunkSize: 4))
        let hit = try XCTUnwrap(cache.lookup(tokens: request.promptTokens,
            cacheSalt: request.checkpointCacheSalt, maximumChunkSize: 4))
        XCTAssertEqual(hit.checkpoint.mediaIdentity, media)
        XCTAssertTrue(hit.checkpoint.mediaTargetOnly)
        XCTAssertNil(hit.checkpoint.assistant)
        let restored = try XCTUnwrap(hit.checkpoint.qwen4[3])
        for (expected, actual) in zip(side.arrays, restored.arrays) {
            XCTAssertEqual(expected.dtype, actual.dtype)
            XCTAssertEqual(expected.asData().data, actual.asData().data)
        }
        cache.close()
        XCTAssertGreaterThan(cache.stats.retainedBytes, 0, "an adoption pin cannot be refunded by close")
        cache.endAdoption(pin: hit.pin)
        XCTAssertEqual(cache.stats.retainedBytes, 0)
    }

    func testUnboundOrAbsentQSARefusesMediaCaptureAndCancelledCaptureDrains() async throws {
        let media = try CBv2HybridPrefixIdentity(digest: Data(repeating: 0x63, count: 32))
        let spec = CBv2RecurrentStateSpec(layers: [.init(modelLayerIndex: 0,
            convShape: [1, 2], convDType: .float32, ssmShape: [1, 2], ssmDType: .float32)])
        let layers: [Int: CBv2RecurrentLayerState] = [0: .init(
            conv: MLXArray.ones([1, 2]), ssm: MLXArray.ones([1, 2]))]
        let side = CBv2Qwen4IndexerSnapshot(tokenCount: 4,
            indexKeys: MLXArray.ones([1, 4, 2], dtype: .bfloat16),
            positionIds: MLXArray((0..<12).map(Int32.init), [3, 1, 4]),
            pooledIndexKeys: nil, pooledIndexBlocks: 0)
        let cache = CBv2HybridPrefixCache(config: .init(maximumBytes: 4096, modelID: "qwen4",
            promptContractID: "test", buildID: "test"))
        XCTAssertTrue(cache.capture(requestID: .init(1), position: 4, chunkSize: 4,
            spec: spec, layers: layers, qwen4: [3: side], mediaTargetOnly: true).isEmpty)
        XCTAssertTrue(cache.capture(requestID: .init(1), position: 4, chunkSize: 4,
            spec: spec, layers: layers, mediaIdentity: media, mediaTargetOnly: true).isEmpty)
        XCTAssertEqual(cache.stats.retainedBytes, 0)
        let roots = cache.capture(requestID: .init(1), position: 4, chunkSize: 4,
            spec: spec, layers: layers, qwen4: [3: side], mediaIdentity: media, mediaTargetOnly: true)
        XCTAssertFalse(roots.isEmpty)
        asyncEval(roots)
        let done = expectation(description: "cancelled media capture retires once")
        XCTAssertTrue(cache.dropStaged(requestID: .init(1)) { done.fulfill() })
        await fulfillment(of: [done], timeout: 10)
        XCTAssertEqual(cache.stats.retainedBytes, 0)
        XCTAssertEqual(cache.stats.entries, 0)
        cache.close()
    }
}
