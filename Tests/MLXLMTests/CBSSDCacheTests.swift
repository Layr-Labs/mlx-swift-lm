// Tests for SSDCacheManager — §13 of ContinuousBatchingTestPlan.md

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class CBSSDCacheManagerTests: XCTestCase {

    private var tempDir: URL!

    override func setUp() {
        super.setUp()
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ssd-cache-test-\(UUID().uuidString)")
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: tempDir)
        super.tearDown()
    }

    private func makeManager(maxBytes: Int = 100 * 1024 * 1024) -> SSDCacheManager {
        SSDCacheManager(config: SSDCacheConfig(cacheDir: tempDir, maxSizeBytes: maxBytes))
    }

    private func makeHash(seed: UInt8 = 1) -> Data {
        Data(repeating: seed, count: 32)
    }

    private func makeLayerCache(value: Float = 1.0, dim: Int = 4) -> KVCacheSimple {
        let c = KVCacheSimple()
        let t = MLXArray(Array(repeating: value, count: dim)).reshaped([1, 1, dim, 1])
        c.state = [t, t * 2]
        return c
    }

    /// Synchronously wait for background writes to flush.
    private func waitForWrites() {
        Thread.sleep(forTimeInterval: 0.3)
    }

    // MARK: - Round-trip

    func testSSDCacheManagerRoundTrip() {
        let mgr = makeManager()
        let hash = makeHash(seed: 42)
        let layer = makeLayerCache(value: 3.14)

        mgr.saveBlock(hash: hash, layerCaches: [layer], tokenCount: 4)
        waitForWrites()

        let loaded = mgr.loadBlock(hash: hash)
        XCTAssertNotNil(loaded, "loadBlock must return non-nil after save")
        XCTAssertEqual(loaded?.count, 1)

        if let l = loaded?.first, l.state.count >= 2 {
            let keys = l.state[0].asArray(Float.self)
            let expected = layer.state[0].asArray(Float.self)
            XCTAssertEqual(keys.count, expected.count)
            for (a, b) in zip(keys, expected) {
                XCTAssertEqual(a, b, accuracy: 1e-5)
            }
        }
    }

    // MARK: - hasBlock

    func testSSDCacheManagerHasBlockAfterSave() {
        let mgr = makeManager()
        let hash = makeHash(seed: 7)
        XCTAssertFalse(mgr.hasBlock(hash: hash))

        mgr.saveBlock(hash: hash, layerCaches: [makeLayerCache()], tokenCount: 4)
        waitForWrites()

        XCTAssertTrue(mgr.hasBlock(hash: hash))
    }

    // MARK: - LRU eviction

    func testSSDCacheManagerLRUEviction() {
        // maxSizeBytes smaller than two blocks to force eviction.
        // Each block ~ 4 * 4 bytes * 2 arrays = 32 bytes; set limit to 40.
        let mgr = makeManager(maxBytes: 40)

        let hashA = makeHash(seed: 1)
        let hashB = makeHash(seed: 2)

        mgr.saveBlock(hash: hashA, layerCaches: [makeLayerCache()], tokenCount: 4)
        waitForWrites()

        mgr.saveBlock(hash: hashB, layerCaches: [makeLayerCache()], tokenCount: 4)
        waitForWrites()

        // At least one eviction should have happened.
        XCTAssertGreaterThanOrEqual(mgr.evictions, 1, "LRU eviction must occur when budget exceeded")
    }

    // MARK: - Scan existing on init

    func testSSDCacheManagerScanExistingOnInit() {
        let mgr1 = makeManager()
        let hash = makeHash(seed: 99)

        mgr1.saveBlock(hash: hash, layerCaches: [makeLayerCache()], tokenCount: 4)
        waitForWrites()

        // New manager on the same directory must rediscover the block.
        let mgr2 = makeManager()
        XCTAssertTrue(mgr2.hasBlock(hash: hash),
                      "new SSDCacheManager must discover blocks from previous session")
    }

    // MARK: - File layout

    func testSSDCacheManagerFileLayout() {
        let mgr = makeManager()
        let hash = makeHash(seed: 0xAB)   // first hex char == 'a'

        mgr.saveBlock(hash: hash, layerCaches: [makeLayerCache()], tokenCount: 4)
        waitForWrites()

        let hexStr = hash.map { String(format: "%02x", $0) }.joined()
        let firstChar = String(hexStr.prefix(1))
        let expectedFile = tempDir
            .appendingPathComponent(firstChar)
            .appendingPathComponent("\(hexStr).safetensors")

        XCTAssertTrue(
            FileManager.default.fileExists(atPath: expectedFile.path),
            "cache file must be at cacheDir/<firstHexChar>/<64hex>.safetensors"
        )
    }
}
