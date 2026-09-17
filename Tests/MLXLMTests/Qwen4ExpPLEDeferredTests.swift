// Copyright © 2026 Eigen Labs Inc.
//
// Qwen4 PLE "graph before the n-gram lookup": the deferred slot path must be
// bit-identical to the eager mmap gather, the host n-gram ids must match the
// GPU ids the eager path uses, and the engine-side scope must enforce its
// open / close / run discipline.

import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen4ExpPLEDeferredTests: XCTestCase {
    /// Real-data tests require the product's active resource binding or an
    /// explicit owned artifact. Never silently select a developer's third-party
    /// checkpoint; absent input skips those cells instead of qualifying it.
    private static var modelDirectory: URL {
        if let resolved = Qwen4ExpPLEResidency.resolvedModelDirectory {
            return resolved
        }
        if let path = ProcessInfo.processInfo.environment["DARKBLOOM_QWEN4_REAL_MODEL"]?
            .trimmingCharacters(in: .whitespacesAndNewlines), !path.isEmpty
        {
            return URL(fileURLWithPath: path, isDirectory: true)
        }
        return URL(fileURLWithPath: "/nonexistent/qwen4-exp-qualification", isDirectory: true)
    }

    private func flashNextTextConfig() throws -> Qwen4ExpTextConfiguration {
        let url = Self.modelDirectory.appendingPathComponent("config.json")
        let data: Data
        if FileManager.default.fileExists(atPath: url.path) {
            data = try Data(contentsOf: url)
        } else {
            data = Data(Qwen4ExpConfigurationTests.embeddedFlashNextConfig.utf8)
        }
        return try JSONDecoder().decode(Qwen4ExpConfiguration.self, from: data).textConfig
    }

    // MARK: Scope discipline

    func testScopeLifecycle() {
        XCTAssertNil(CBv2DeferredHostFill.current)
        let scope = CBv2DeferredHostFill.open()
        XCTAssertTrue(CBv2DeferredHostFill.current === scope)
        var order: [Int] = []
        scope.register { order.append(1) }
        scope.register { order.append(2) }
        XCTAssertEqual(scope.registeredCount, 2)
        XCTAssertEqual(order, [], "fills must not run during graph construction")
        scope.close()
        XCTAssertNil(CBv2DeferredHostFill.current, "close detaches the scope from the thread")
        scope.run()
        XCTAssertEqual(order, [1, 2])
        scope.run()
        XCTAssertEqual(order, [1, 2], "run is idempotent")
    }

    func testScopeIsThreadConfined() {
        let scope = CBv2DeferredHostFill.open()
        defer {
            scope.close()
            scope.run()
        }
        let seenOnOtherThread = expectation(description: "other thread")
        var other: CBv2DeferredHostFill?
        Thread {
            other = CBv2DeferredHostFill.current
            seenOnOtherThread.fulfill()
        }.start()
        wait(for: [seenOnOtherThread], timeout: 5)
        XCTAssertNil(other)
    }

    func testEagerBoundaryResolvesOnlyPendingInputsAndKeepsScopeOpen() {
        let scope = CBv2DeferredHostFill.open()
        defer { scope.close(); scope.run() }
        var order: [Int] = []
        scope.register {
            XCTAssertNil(CBv2DeferredHostFill.current, "fill callbacks are not model graph builders")
            order.append(1)
        }
        CBv2DeferredHostFill.resolveBeforeEvaluation()
        XCTAssertEqual(order, [1])
        XCTAssertTrue(CBv2DeferredHostFill.current === scope)
        CBv2DeferredHostFill.resolveBeforeEvaluation()
        XCTAssertEqual(order, [1], "empty flush is idempotent")
        scope.register { order.append(2) }
        XCTAssertEqual(scope.registeredCount, 2)
        XCTAssertEqual(order, [1], "later inputs stay deferred until needed")
        scope.close()
        scope.run()
        XCTAssertEqual(order, [1, 2])
        CBv2DeferredHostFill.resolveBeforeEvaluation()
        scope.run()
        XCTAssertEqual(order, [1, 2])
    }

    // MARK: Host ids == GPU ids

    func testHostNGramIdsMatchGPUIdsForDecodeWindow() throws {
        let tables = Qwen4ExpNGramTables(try flashNextTextConfig(), pleIndex: 0)
        let eos = tables.eosTokenId
        var generator = SystemRandomNumberGenerator()
        var histories: [[Int]] = [
            [eos, eos, 17],
            [eos, 17, 42],
            [17, 42, eos],
            [42, eos, 7],
            [248_319, 0, 1],
        ]
        for _ in 0 ..< 32 {
            histories.append(
                (0 ..< tables.contextLen + 1).map { _ in
                    Int.random(in: 0 ..< 248_320, using: &generator)
                })
        }
        for history in histories {
            let host = Qwen4ExpNGramIDs.ids(history: history, inputWidth: 1, tables: tables)
            XCTAssertEqual(host.count, 1)
            let gpu = Qwen4ExpNGramIDs.gpuIds(
                history: MLXArray(history.map { Int64($0) }).reshaped([1, history.count]),
                inputWidth: 1, tables: tables)
            XCTAssertEqual(gpu.shape, [1, 1, tables.ngramHeads])
            XCTAssertEqual(
                gpu.reshaped(-1).asArray(Int64.self).map { Int($0) }, host[0],
                "history \(history)")
        }
    }

    /// Lightning verify window: the captured PLE path fills `batch * width`
    /// deferred rows from host ids over `context + window`. Every position
    /// must equal the GPU ids the eager path computed for the same history.
    func testHostNGramIdsMatchGPUIdsForVerifyWindow() throws {
        let tables = Qwen4ExpNGramTables(try flashNextTextConfig(), pleIndex: 0)
        let eos = tables.eosTokenId
        var generator = SystemRandomNumberGenerator()
        for width in [2, 5, 6] {
            var histories: [[Int]] = [
                [eos, eos] + (0 ..< width).map { 17 + $0 },
                [eos, 17] + Array(repeating: eos, count: width),
                [42, eos] + [7, eos, 9, eos, 11, 13].prefix(width),
            ]
            for _ in 0 ..< 16 {
                histories.append(
                    (0 ..< tables.contextLen + width).map { _ in
                        Int.random(in: 0 ..< 248_320, using: &generator)
                    })
            }
            for history in histories {
                let host = Qwen4ExpNGramIDs.ids(
                    history: history, inputWidth: width, tables: tables)
                XCTAssertEqual(host.count, width)
                let gpu = Qwen4ExpNGramIDs.gpuIds(
                    history: MLXArray(history.map { Int64($0) }).reshaped([1, history.count]),
                    inputWidth: width, tables: tables)
                XCTAssertEqual(gpu.shape, [1, width, tables.ngramHeads])
                XCTAssertEqual(
                    gpu.reshaped(-1).asArray(Int64.self).map { Int($0) },
                    host.flatMap { $0 },
                    "width \(width) history \(history)")
            }
        }
    }

    func testHostContextMatchesSSMTruncationAndEOSFallback() throws {
        let tables = Qwen4ExpNGramTables(try flashNextTextConfig(), pleIndex: 0)
        XCTAssertEqual(
            Qwen4ExpPLELayer.hostContext(nil, tables: tables),
            [Int](repeating: tables.eosTokenId, count: tables.contextLen))
        let ssm = MLXArray([Float32(123), Float32(248_044)]).reshaped([1, 1, 1, tables.contextLen])
        XCTAssertEqual(Qwen4ExpPLELayer.hostContext(ssm, tables: tables), [123, 248_044])
    }

    func testDeferredFlagDefaultsOnAndHonorsKillSwitch() {
        XCTAssertTrue(Qwen4ExpPLEDeferred.isEnabled(environment: [:]))
        XCTAssertTrue(Qwen4ExpPLEDeferred.isEnabled(environment: [Qwen4ExpPLEDeferred.envFlag: "1"]))
        XCTAssertFalse(Qwen4ExpPLEDeferred.isEnabled(environment: [Qwen4ExpPLEDeferred.envFlag: "0"]))
        XCTAssertFalse(
            Qwen4ExpPLEDeferred.isEnabled(environment: [Qwen4ExpPLEDeferred.envFlag: "off"]))
    }

    func testDeferredRowBufferPolicyIsByteBounded() {
        XCTAssertEqual(
            Qwen4ExpPLEDeferredBufferPolicy.byteBudget(environment: [:]),
            256 * 1_024 * 1_024)
        XCTAssertEqual(
            Qwen4ExpPLEDeferredBufferPolicy.byteBudget(
                environment: [Qwen4ExpPLEDeferredBufferPolicy.byteBudgetFlag: "0"]),
            0)
        XCTAssertEqual(
            Qwen4ExpPLEDeferredBufferPolicy.byteBudget(
                environment: [Qwen4ExpPLEDeferredBufferPolicy.byteBudgetFlag: "7"]),
            7 * 1_024 * 1_024)
        // 4 U32 packed columns + 2 BF16 scales + 2 BF16 biases = 24 B/row.
        XCTAssertEqual(
            Qwen4ExpPLEDeferredBufferPolicy.slotBytes(
                rows: 10, packedCols: 4, scaleCols: 2),
            240)
        XCTAssertEqual(
            Qwen4ExpPLEDeferredBufferPolicy.slotBytes(
                rows: Int.max, packedCols: Int.max, scaleCols: Int.max),
            Int.max)
    }

    // MARK: Deferred slot == eager mmap gather (needs the local checkpoint)

    /// Run alone in a fresh test process: the native indexer's first-use
    /// stream proof must not evaluate an unfilled SSD-backed placeholder.
    func testFirstNativeIndexerProofResolvesDeferredPLEInput() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["DARKBLOOM_QWEN4_FIRST_USE_TEST"] == "1",
            "Requires a fresh process selected with --filter testFirstNativeIndexerProofResolvesDeferredPLEInput")
        try XCTSkipUnless(FileManager.default.fileExists(atPath:
            Self.modelDirectory.appendingPathComponent("model.safetensors.index.json").path),
            "Flash-Next Q4 checkpoint index not present (set DARKBLOOM_QWEN4_REAL_MODEL)")
        let config = try flashNextTextConfig()
        Qwen4ExpPLEResidency.reset()
        Qwen4ExpPLEResidency.modelDirectory = Self.modelDirectory
        defer { Qwen4ExpPLEResidency.reset() }
        let embedding = Qwen4ExpNGramEmbedding(
            config, layerIndex: config.pleLayerIds[0] - 1, pleIndex: 0, mmap: true)
        let ids = Qwen4ExpNGramIDs.ids(
            history: [embedding.tables.eosTokenId, 17, 42], inputWidth: 1,
            tables: embedding.tables)[0]
        let deferred = try XCTUnwrap(embedding.deferredGather(tokenRows: 1))
        let pooled = MLXRandom.normal([1, 513, 128], key: MLXRandom.key(91)).asType(.bfloat16)
        eval(pooled)
        let scope = CBv2DeferredHostFill.open()
        defer { scope.close(); scope.run() }
        var filled = false
        scope.register { deferred.fill(ids); filled = true }
        let lazyQuery = deferred.values[0..., 0..<512].reshaped([1, 1, 4, 128])
        let observed = try XCTUnwrap(Qwen4ExpNativeIndexer.scores(
            queries: lazyQuery, pooledKeys: pooled, maskQOffset: 2051))
        XCTAssertTrue(filled, "First-use proof evaluated before the deferred PLE slot was filled")
        scope.close()
        scope.run()
        let eagerQuery = embedding.gather(ids: [ids])[0..., 0..<512].reshaped([1, 1, 4, 128])
        let expected = try XCTUnwrap(Qwen4ExpNativeIndexer.scores(
            queries: eagerQuery, pooledKeys: pooled, maskQOffset: 2051))
        eval(observed, expected)
        XCTAssertTrue(MLX.all(observed .== expected).item(Bool.self),
            "A late fill cannot repair already-evaluated scores from placeholder bytes")

        let keys = MLXRandom.normal([1, 2, 2055, 256], key: MLXRandom.key(92)).asType(.bfloat16)
        let values = MLXRandom.normal([1, 2, 2055, 256], key: MLXRandom.key(93)).asType(.bfloat16)
        let blocks = MLXArray(0..<512).asType(.int32).reshaped([1, 1, 512])
        eval(keys, values, blocks)
        for boundary in ["topk", "attention"] {
            let next = try XCTUnwrap(embedding.deferredGather(tokenRows: 1))
            let pending = CBv2DeferredHostFill.open()
            defer { pending.close(); pending.run() }
            var resolved = false
            pending.register { next.fill(ids); resolved = true }
            func operation(_ input: MLXArray) -> MLXArray? {
                if boundary == "topk" {
                    guard let scores = Qwen4ExpNativeIndexer.scores(
                        queries: input[0..., 0..<512].reshaped([1, 1, 4, 128]),
                        pooledKeys: pooled, maskQOffset: 2051) else { return nil }
                    return Qwen4ExpNativeIndexer.topKIndices(scores)
                }
                let query = concatenated([input, input, input], axis: -1)[0..., 0..<6144]
                    .asType(.bfloat16).reshaped([1, 24, 1, 256])
                return Qwen4ExpNativeSparseGQA.attend(
                    queries: query, keys: keys, values: values,
                    selectedBlocks: blocks, qOffset: 2054)
            }
            let actual = try XCTUnwrap(operation(next.values))
            XCTAssertTrue(resolved, "\(boundary) proof read a pending PLE input")
            pending.close()
            pending.run()
            let reference = try XCTUnwrap(operation(embedding.gather(ids: [ids])))
            eval(actual, reference)
            XCTAssertTrue(MLX.all(actual .== reference).item(Bool.self),
                "\(boundary) proof changed values relative to eager SSD lookup")
        }
    }

    func testDeferredSlotIsBitIdenticalToEagerGather() throws {
        let indexURL = Self.modelDirectory.appendingPathComponent("model.safetensors.index.json")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: indexURL.path),
            "Flash-Next Q4 checkpoint index not present (set DARKBLOOM_QWEN4_REAL_MODEL)")
        let config = try flashNextTextConfig()
        let layerIndex = config.pleLayerIds[0] - 1
        Qwen4ExpPLEResidency.reset()
        Qwen4ExpPLEResidency.modelDirectory = Self.modelDirectory
        defer { Qwen4ExpPLEResidency.reset() }

        let embedding = Qwen4ExpNGramEmbedding(
            config, layerIndex: layerIndex, pleIndex: 0, mmap: true)
        let tables = embedding.tables
        var generator = SystemRandomNumberGenerator()
        for round in 0 ..< 4 {
            let history = round == 0
                ? [tables.eosTokenId, tables.eosTokenId, 17]
                : (0 ..< tables.contextLen + 1).map { _ in
                    Int.random(in: 0 ..< 248_320, using: &generator)
                }
            let ids = Qwen4ExpNGramIDs.ids(history: history, inputWidth: 1, tables: tables)
            // Force a duplicate id in round 1 so the no-`inverse` slot path is
            // exercised against the unique+inverse eager path.
            var flat = ids[0]
            if round == 1 { flat[3] = flat[9] }

            guard let deferred = embedding.deferredGather(tokenRows: 1) else {
                return XCTFail("deferred gather unavailable with mmap catalog present")
            }
            // Graph first, bytes later — the production order.
            let lazy = deferred.values
            XCTAssertEqual(lazy.shape, [1, tables.embedDim])
            deferred.fill(flat)
            let eager = embedding.gather(ids: [flat])
            eval(lazy, eager)
            XCTAssertEqual(eager.shape, lazy.shape)
            XCTAssertEqual(lazy.dtype, eager.dtype)
            XCTAssertTrue(
                MLX.all(lazy .== eager).item(Bool.self),
                "round \(round): deferred slot values differ from eager gather")
            XCTAssertTrue(
                MLX.any(lazy .!= MLXArray(Float32(0)).asType(lazy.dtype)).item(Bool.self),
                "round \(round): slot decoded to all zeros")
        }

        // Verify window: one `width`-row slot, rows in position order.
        let width = 5
        let window = (0 ..< tables.contextLen + width).map { _ in
            Int.random(in: 0 ..< 248_320, using: &generator)
        }
        let rows = Qwen4ExpNGramIDs.ids(history: window, inputWidth: width, tables: tables)
        guard let deferredWindow = embedding.deferredGather(tokenRows: width) else {
            return XCTFail("deferred gather unavailable for the verify window")
        }
        let lazyWindow = deferredWindow.values
        XCTAssertEqual(lazyWindow.shape, [width, tables.embedDim])
        deferredWindow.fill(rows.flatMap { $0 })
        let eagerWindow = embedding.gather(ids: rows)
        eval(lazyWindow, eagerWindow)
        XCTAssertEqual(eagerWindow.shape, lazyWindow.shape)
        XCTAssertTrue(
            MLX.all(lazyWindow .== eagerWindow).item(Bool.self),
            "verify window: deferred slot values differ from eager gather")
    }

    func testDeferredSlotCacheEvictsWidthsUnderByteBudgetAndReleases() throws {
        let indexURL = Self.modelDirectory.appendingPathComponent("model.safetensors.index.json")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: indexURL.path),
            "Flash-Next Q4 checkpoint index not present (set DARKBLOOM_QWEN4_REAL_MODEL)")
        let config = try flashNextTextConfig()
        let layerIndex = config.pleLayerIds[0] - 1
        Qwen4ExpPLEResidency.reset()
        Qwen4ExpPLEResidency.modelDirectory = Self.modelDirectory
        defer { Qwen4ExpPLEResidency.reset() }

        // Size the budget for exactly the width-2 pair. Inserting width 2
        // after width 1 must evict width 1; width 3 is larger than the whole
        // budget and is therefore request-owned, never cached.
        let heads = Qwen4ExpNGramTables(config, pleIndex: 0).ngramHeads
        let probe = Qwen4ExpNGramEmbedding(
            config, layerIndex: layerIndex, pleIndex: 0, mmap: true,
            deferredByteBudget: Int.max)
        _ = try XCTUnwrap(probe.deferredGather(tokenRows: 1))
        let one = probe.deferredBufferCacheSnapshot
        XCTAssertEqual(one.rowCounts, [heads])
        let onePairBytes = one.bytes
        probe.releaseExternalResources()
        XCTAssertEqual(probe.deferredBufferCacheSnapshot.bytes, 0)

        let embedding = Qwen4ExpNGramEmbedding(
            config, layerIndex: layerIndex, pleIndex: 0, mmap: true,
            deferredByteBudget: onePairBytes * 2)
        _ = try XCTUnwrap(embedding.deferredGather(tokenRows: 1))
        XCTAssertLessThanOrEqual(
            embedding.deferredBufferCacheSnapshot.bytes,
            embedding.deferredBufferCacheSnapshot.budget)
        _ = try XCTUnwrap(embedding.deferredGather(tokenRows: 2))
        let two = embedding.deferredBufferCacheSnapshot
        XCTAssertEqual(two.rowCounts, [heads * 2])
        XCTAssertLessThanOrEqual(two.bytes, two.budget)
        _ = try XCTUnwrap(embedding.deferredGather(tokenRows: 3))
        let three = embedding.deferredBufferCacheSnapshot
        XCTAssertEqual(three.rowCounts, [heads * 2], "oversized width must not enter cache")
        XCTAssertLessThanOrEqual(three.bytes, three.budget)

        embedding.releaseExternalResources()
        let released = embedding.deferredBufferCacheSnapshot
        XCTAssertEqual(released.bytes, 0)
        XCTAssertEqual(released.rowCounts, [])
    }

    func testRepeatedMmapOpenAndExplicitReleaseReturnsEveryGaugeToZero() throws {
        let indexURL = Self.modelDirectory.appendingPathComponent("model.safetensors.index.json")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: indexURL.path),
            "Flash-Next Q4 checkpoint index not present (set DARKBLOOM_QWEN4_REAL_MODEL)")
        let config = try flashNextTextConfig()
        let layerIndex = config.pleLayerIds[0] - 1
        Qwen4ExpPLEResidency.reset()
        Qwen4ExpPLEResourceMetrics.resetForTesting()
        Qwen4ExpPLEResidency.modelDirectory = Self.modelDirectory
        defer {
            Qwen4ExpPLEResidency.reset()
            Qwen4ExpPLEResourceMetrics.resetForTesting()
        }

        for cycle in 0 ..< 5 {
            let embedding = Qwen4ExpNGramEmbedding(
                config, layerIndex: layerIndex, pleIndex: 0, mmap: true)
            _ = try XCTUnwrap(embedding.deferredGather(tokenRows: 1))
            let opened = Qwen4ExpPLEResourceMetrics.snapshot()
            XCTAssertGreaterThan(opened.mappedFiles, 0, "cycle \(cycle)")
            XCTAssertGreaterThan(opened.activeRowBufferBytes, 0, "cycle \(cycle)")
            XCTAssertGreaterThan(opened.cachedRowBufferBytes, 0, "cycle \(cycle)")

            embedding.releaseExternalResources()
            XCTAssertEqual(
                Qwen4ExpPLEResourceMetrics.snapshot(),
                .init(
                    mappedFiles: 0,
                    activeRowBufferBytes: 0,
                    cachedRowBufferBytes: 0),
                "cycle \(cycle)")
        }
    }

    func testLoadTimeValidationThrowsForIncompleteCatalog() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4-ple-invalid-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try Data(#"{"weight_map":{}}"#.utf8).write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))
        addTeardownBlock {
            try? FileManager.default.removeItem(at: directory)
        }

        let config = try flashNextTextConfig()
        let layerIndex = config.pleLayerIds[0] - 1
        Qwen4ExpPLEResidency.reset()
        Qwen4ExpPLEResidency.modelDirectory = directory
        defer { Qwen4ExpPLEResidency.reset() }
        let embedding = Qwen4ExpNGramEmbedding(
            config, layerIndex: layerIndex, pleIndex: 0, mmap: true)

        XCTAssertThrowsError(try embedding.validateExternalResources()) { error in
            XCTAssertTrue(
                String(describing: error).contains("PLE shard 0 missing"),
                "unexpected validation error: \(error)")
        }
    }
}
