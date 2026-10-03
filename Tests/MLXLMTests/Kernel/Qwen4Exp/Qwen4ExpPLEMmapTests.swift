import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {

    /// Tests of the memory-mapped n-gram PLE table in `Qwen4ExpPLE.swift`
    /// with a tiny synthetic checkpoint: the batched and serial gathers, the
    /// deferred (graph before lookup) slots, the catalog checks, and the
    /// PLE layer on the CBv2 decode and capture-verify paths.
    ///
    /// The checkpoint has two shards of 44 rows. Each row has 32 values,
    /// packed with 4 bits and group size 32. The expected rows are the
    /// same packed rows dequantized with the same MLX operation, so every
    /// comparison is exact.
    ///
    /// The PLE table reads its directory from `Qwen4ExpPLEResidency`, a
    /// process-wide binding. The suite is serialized, and each test binds
    /// the directory only for its own body.
    @Suite(.serialized)
    struct Qwen4ExpPLEMmapTests {
        typealias Support = Qwen4ExpKernelSupport

        static func configuration() -> Qwen4ExpTextConfiguration {
            var c = Qwen4ExpTextConfiguration()
            c.hiddenSize = 16
            c.hcCount = 2
            c.hiddenLayers = 1
            c.layerTypes = ["qwen_sparse_attention"]
            c.pleLayerIds = [1]
            c.pleEmbedDim = 128
            c.ngramSize = 3
            c.headsPerNgram = 2
            c.ngramVocabSizeBase = 17
            c.makeNgramVocabSizeDivisibleBy = 8
            c.splitNgramParts = 2
            c.vocabularySize = 64
            c.eosTokenId = [0]
            c.pleConvKernelSize = 3
            return c
        }

        enum Layout {
            case standard
            case underscoreNames
            case modelPrefix
            case missingShard
            case splitFiles
            case float32Scales
            case mixedBits
        }

        struct Checkpoint {
            let directory: URL
            /// All rows of all shards, dequantized, `[paddedVocab, headEmbedDim]`.
            let table: MLXArray
        }

        static func write(
            _ c: Qwen4ExpTextConfiguration, layout: Layout = .standard
        ) throws -> Checkpoint {
            let directory = FileManager.default.temporaryDirectory
                .appendingPathComponent("qwen4-ple-kernel-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let tables = Qwen4ExpNGramTables(c, pleIndex: 0)
            var main: [String: MLXArray] = [:]
            var side: [String: MLXArray] = [:]
            var rows: [MLXArray] = []
            for (index, count) in tables.shardSizes.enumerated() {
                let values = MLXRandom.normal(
                    [count, tables.headEmbedDim], key: MLXRandom.key(UInt64(9100 + index))
                ).asType(.bfloat16)
                let bits = layout == .mixedBits && index == 1 ? 8 : 4
                let (weight, scales, biases) = MLX.quantized(
                    values, groupSize: 32, bits: bits, mode: .affine)
                let offsets = biases ?? MLXArray.zeros(like: scales)
                rows.append(
                    dequantized(
                        weight, scales: scales, biases: offsets, groupSize: 32, bits: bits,
                        mode: .affine
                    ).asType(.bfloat16))
                let base: String
                switch layout {
                case .underscoreNames:
                    base =
                        "language_model.model.layers.0.ple.ple_embedding.ngram_embedding.shard_\(index)"
                case .modelPrefix:
                    base =
                        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shards.\(index)"
                default:
                    base =
                        "language_model.model.layers.0.ple.ple_embedding.ngram_embedding.shards.\(index)"
                }
                main[base + ".weight"] = weight
                let storedScales =
                    layout == .float32Scales && index == 0 ? scales.asType(.float32) : scales
                if layout == .splitFiles && index == 0 {
                    side[base + ".scales"] = storedScales
                } else {
                    main[base + ".scales"] = storedScales
                }
                main[base + ".biases"] = offsets
            }
            try save(arrays: main, url: directory.appendingPathComponent("model.safetensors"))
            var map: [String: String] = [:]
            for key in main.keys { map[key] = "model.safetensors" }
            if !side.isEmpty {
                try save(arrays: side, url: directory.appendingPathComponent("side.safetensors"))
                for key in side.keys { map[key] = "side.safetensors" }
            }
            if layout == .missingShard {
                map = map.filter { !$0.key.contains("shards.1.") }
            }
            try JSONSerialization.data(withJSONObject: ["weight_map": map])
                .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
            let table = concatenated(rows, axis: 0)
            eval(table)
            return Checkpoint(directory: directory, table: table)
        }

        /// Runs `body` with the PLE directory bound to `directory`, then
        /// restores the old binding and deletes the directory.
        static func bound<T>(_ directory: URL?, _ body: () throws -> T) rethrows -> T {
            let previous = Qwen4ExpPLEResidency.modelDirectory
            Qwen4ExpPLEResidency.modelDirectory = directory
            defer {
                Qwen4ExpPLEResidency.modelDirectory = previous
                if let directory {
                    try? FileManager.default.removeItem(at: directory)
                }
            }
            return try body()
        }

        static func expected(
            _ ids: [[Int]], table: MLXArray, embedding: Qwen4ExpNGramEmbedding
        ) -> MLXArray {
            let flat = MLXArray(ids.flatMap { $0 }.map { Int32($0) })
            let heads = embedding.tables.ngramHeads
            return table[flat].reshaped([ids.count, heads * embedding.tables.headEmbedDim])
                * embedding.ngramEmbedding.weightScale
        }

        @Test func tableGeometry() {
            let tables = Qwen4ExpNGramTables(Self.configuration(), pleIndex: 0)
            #expect(tables.ngramHeads == 4)
            #expect(tables.headEmbedDim == 32)
            #expect(tables.headVocabSizes == [17, 19, 23, 29])
            #expect(tables.paddedVocabSize == 88)
            #expect(tables.shardSizes == [44, 44])
            #expect(tables.shardIndex(for: 43) == 0)
            #expect(tables.shardIndex(for: 44) == 1)
        }

        /// The batched gather (unique rows, concurrent shard copies, one
        /// dequantization) gives the table rows in the request order, also
        /// for duplicate ids and ids from both shards.
        @Test func batchedGatherReturnsTheTableRows() throws {
            let c = Self.configuration()
            let checkpoint = try Self.write(c)
            try Self.bound(checkpoint.directory) {
                let embedding = Qwen4ExpNGramEmbedding(c, layerIndex: 0, pleIndex: 0, mmap: true)
                try embedding.validateExternalResources()
                let ids = [[0, 43, 44, 87], [5, 5, 60, 43]]
                let expected = Self.expected(ids, table: checkpoint.table, embedding: embedding)
                let actual = embedding.gather(ids: ids)
                #expect(actual.dtype == .bfloat16)
                #expect(Support.isEqual(actual, expected))

                // The GPU id path evaluates `[..., heads]` int64 ids first.
                let gpuIds = MLXArray(ids.flatMap { $0 }.map { Int64($0) }, [1, 2, 4])
                #expect(Support.isEqual(embedding.gather(gpuIds), expected))
                embedding.releaseExternalResources()
            }
        }

        /// `DARKBLOOM_QWEN4_PLE_GATHER=0` selects the serial gather: one
        /// dequantization for each shard and one scatter.
        @Test func serialGatherReturnsTheSameRows() throws {
            let c = Self.configuration()
            let checkpoint = try Self.write(c)
            try Self.bound(checkpoint.directory) {
                try Support.withEnvironment(["DARKBLOOM_QWEN4_PLE_GATHER": "0"]) {
                    #expect(!Qwen4ExpPLEGather.isEnabled())
                    let embedding = Qwen4ExpNGramEmbedding(
                        c, layerIndex: 0, pleIndex: 0, mmap: true)
                    let ids = [[87, 1, 44, 1], [2, 50, 3, 70]]
                    let actual = embedding.gather(ids: ids)
                    let expected = Self.expected(
                        ids, table: checkpoint.table, embedding: embedding)
                    #expect(Support.isEqual(actual, expected))
                    try embedding.validateExternalResources()
                }
            }
        }

        /// Both alternative key spellings of the shards are found.
        @Test(arguments: [Layout.underscoreNames, Layout.modelPrefix])
        func alternativeShardNamesAreFound(_ layout: Layout) throws {
            let c = Self.configuration()
            let checkpoint = try Self.write(c, layout: layout)
            try Self.bound(checkpoint.directory) {
                let embedding = Qwen4ExpNGramEmbedding(c, layerIndex: 0, pleIndex: 0, mmap: true)
                try embedding.validateExternalResources()
                let ids = [[10, 20, 50, 80]]
                #expect(
                    Support.isEqual(
                        embedding.gather(ids: ids),
                        Self.expected(ids, table: checkpoint.table, embedding: embedding)))
            }
        }

        /// A catalog that does not match the exact affine layout throws at
        /// validation, with a message that names the problem, and has no
        /// deferred slots.
        @Test(arguments: [
            (Layout.missingShard, "PLE shard 1 missing from weight_map"),
            (Layout.splitFiles, "span multiple files"),
            (Layout.float32Scales, "does not match exact affine geometry"),
            (Layout.mixedBits, "PLE shard 1 geometry differs from shard 0"),
        ])
        func invalidCatalogsThrow(_ layout: Layout, _ message: String) throws {
            let c = Self.configuration()
            let checkpoint = try Self.write(c, layout: layout)
            Self.bound(checkpoint.directory) {
                let embedding = Qwen4ExpNGramEmbedding(c, layerIndex: 0, pleIndex: 0, mmap: true)
                do {
                    try embedding.validateExternalResources()
                    Issue.record("validation did not throw for \(layout)")
                } catch {
                    #expect(String(describing: error).contains(message), "\(error)")
                }
                #expect(Support.isNil(embedding.deferredGather(tokenRows: 1)))
            }
        }

        /// Without a bound directory the mmap table cannot open; a resident
        /// table has no external resources and no deferred slots.
        @Test func unboundAndResidentTables() throws {
            let c = Self.configuration()
            try Self.bound(nil) {
                let mapped = Qwen4ExpNGramEmbedding(c, layerIndex: 0, pleIndex: 0, mmap: true)
                do {
                    try mapped.validateExternalResources()
                    Issue.record("validation did not throw without a directory")
                } catch {
                    #expect(String(describing: error).contains("no model directory"), "\(error)")
                }
                #expect(Support.isNil(mapped.deferredGather(tokenRows: 1)))

                let resident = Qwen4ExpNGramEmbedding(c, layerIndex: 0, pleIndex: 0, mmap: false)
                try resident.validateExternalResources()
                #expect(Support.isNil(resident.deferredGather(tokenRows: 1)))
                #expect(resident.ngramEmbedding.shards.count == 2)
            }
        }

        /// The deferred slot is built before its bytes exist. After the
        /// fill it holds the same rows as the eager gather. Each width keeps
        /// two slots (the third width-2 fill reuses the first slot), and the
        /// slot cache counts its bytes.
        ///
        /// Known issue: a one-token slot of this tiny table has 4 rows, so
        /// its scales and biases buffers are 8 bytes each. The fill writes
        /// them through `Data(bytesNoCopy:)` (`Qwen4ExpPLE.swift`,
        /// `DeferredSlot.withMutableBytes`). For 14 bytes or less, Foundation
        /// keeps such a `Data` value inline, as a copy, so the writes do not
        /// reach the array. The slot then reads zero scales and gives zero
        /// rows. Slots of 8 or more rows (16 bytes) are correct.
        @Test func deferredSlotsMatchTheEagerGather() throws {
            let c = Self.configuration()
            let checkpoint = try Self.write(c)
            try Self.bound(checkpoint.directory) {
                let metricsBefore = Qwen4ExpPLEResourceMetrics.snapshot()
                let embedding = Qwen4ExpNGramEmbedding(
                    c, layerIndex: 0, pleIndex: 0, mmap: true, deferredByteBudget: Int.max)

                let single = try #require(embedding.deferredGather(tokenRows: 1))
                #expect(single.values.shape == [1, 128])
                single.fill([3, 50, 50, 87])
                let eagerSingle = embedding.gather(ids: [[3, 50, 50, 87]])
                eval(single.values, eagerSingle)
                withKnownIssue(
                    "A deferred PLE slot of 14 bytes or less is filled through an inline Data copy"
                ) {
                    #expect(
                        Support.isEqual(single.values, eagerSingle), "one-token deferred slot")
                } matching: {
                    $0.isFailedExpectation(["one-token deferred slot"])
                }

                for rows in [
                    [[3, 50, 50, 87], [44, 0, 1, 2]],
                    [[9, 9, 9, 9], [87, 86, 85, 0]],
                    [[1, 2, 3, 4], [60, 61, 62, 63]],
                ] {
                    let deferred = try #require(embedding.deferredGather(tokenRows: 2))
                    #expect(deferred.values.shape == [2, 128])
                    deferred.fill(rows.flatMap { $0 })
                    let eager = embedding.gather(ids: rows)
                    eval(deferred.values, eager)
                    #expect(Support.isEqual(deferred.values, eager), "rows \(rows)")
                }
                let rows = [[1, 2, 3, 4], [60, 61, 62, 63], [44, 45, 46, 47]]
                let window = try #require(embedding.deferredGather(tokenRows: 3))
                window.fill(rows.flatMap { $0 })
                let eagerWindow = embedding.gather(ids: rows)
                eval(window.values, eagerWindow)
                #expect(Support.isEqual(window.values, eagerWindow))

                let geometry = (packedCols: 4, scaleCols: 1)
                let pairs = [4, 8, 12].map {
                    2
                        * Qwen4ExpPLEDeferredBufferPolicy.slotBytes(
                            rows: $0, packedCols: geometry.packedCols,
                            scaleCols: geometry.scaleCols)
                }.reduce(0, +)
                let snapshot = embedding.deferredBufferCacheSnapshot
                #expect(snapshot.rowCounts == [4, 8, 12])
                #expect(snapshot.bytes == pairs)
                #expect(snapshot.budget == Int.max)
                let opened = Qwen4ExpPLEResourceMetrics.snapshot()
                #expect(opened.mappedFiles == metricsBefore.mappedFiles + 1)
                #expect(
                    opened.cachedRowBufferBytes
                        == metricsBefore.cachedRowBufferBytes + pairs)

                embedding.releaseExternalResources()
                let released = embedding.deferredBufferCacheSnapshot
                #expect(released.bytes == 0)
                #expect(released.rowCounts.isEmpty)
                let closed = Qwen4ExpPLEResourceMetrics.snapshot()
                #expect(closed.mappedFiles == metricsBefore.mappedFiles)
                #expect(closed.cachedRowBufferBytes == metricsBefore.cachedRowBufferBytes)
            }
        }

        /// A byte budget for one width-2 pair: width 2 evicts width 1, and
        /// width 3 is larger than the budget, so its slot is not cached.
        @Test func deferredSlotCacheEvictsUnderItsBudget() throws {
            let c = Self.configuration()
            let checkpoint = try Self.write(c)
            try Self.bound(checkpoint.directory) {
                let pairForWidth1 =
                    2
                    * Qwen4ExpPLEDeferredBufferPolicy.slotBytes(
                        rows: 4, packedCols: 4, scaleCols: 1)
                let embedding = Qwen4ExpNGramEmbedding(
                    c, layerIndex: 0, pleIndex: 0, mmap: true,
                    deferredByteBudget: 2 * pairForWidth1)
                _ = try #require(embedding.deferredGather(tokenRows: 1))
                #expect(embedding.deferredBufferCacheSnapshot.rowCounts == [4])
                #expect(embedding.deferredBufferCacheSnapshot.bytes == pairForWidth1)
                _ = try #require(embedding.deferredGather(tokenRows: 2))
                #expect(embedding.deferredBufferCacheSnapshot.rowCounts == [8])
                #expect(embedding.deferredBufferCacheSnapshot.bytes == 2 * pairForWidth1)

                let rows = [[7, 8, 9, 10], [11, 12, 13, 14], [80, 81, 82, 83]]
                let oneShot = try #require(embedding.deferredGather(tokenRows: 3))
                #expect(embedding.deferredBufferCacheSnapshot.rowCounts == [8])
                oneShot.fill(rows.flatMap { $0 })
                let eager = embedding.gather(ids: rows)
                eval(oneShot.values, eager)
                #expect(Support.isEqual(oneShot.values, eager))
            }
        }

        /// Hidden states for 2 rows: `[2, width, hcCount * hiddenSize]`.
        static func hidden(width: Int, seed: UInt64) -> MLXArray {
            let x = MLXRandom.normal([2, width, 32], key: MLXRandom.key(seed)).asType(.bfloat16)
            eval(x)
            return x
        }

        static func layer(_ c: Qwen4ExpTextConfiguration) throws -> Qwen4ExpPLELayer {
            MLXRandom.seed(9200)
            let layer = Qwen4ExpPLELayer(c, layerIndex: 0, pleIndex: 0, mmap: true)
            layer.update(
                parameters: ModuleParameters.unflattened(
                    layer.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
            eval(layer)
            try layer.validateExternalResources()
            return layer
        }

        /// Runs one CBv2 step for the rows of `states` and commits it.
        static func step(
            _ layer: Qwen4ExpPLELayer, _ states: [CBv2RecurrentRequestState], hidden: MLXArray,
            ids: MLXArray, deferred: Bool, capture keep: Int? = nil
        ) throws -> MLXArray {
            let scope = deferred ? CBv2DeferredHostFill.open() : nil
            defer {
                scope?.close()
                scope?.run()
            }
            let transactions = try states.map { try $0.bind() }
            let output = layer.cbv2Forward(
                hidden, inputIds: ids, recurrentState: transactions,
                captureRecurrentWindow: keep != nil)
            if let scope {
                #expect(scope.registeredCount == 1)
                scope.close()
                scope.run()
            }
            eval([output] + (try transactions.flatMap { try $0.evaluate() }))
            for transaction in transactions {
                if let keep {
                    try transaction.commit(keepPositions: keep)
                } else {
                    try transaction.commit()
                }
            }
            return output
        }

        static func expectSameState(
            _ a: CBv2RecurrentRequestState, _ b: CBv2RecurrentRequestState
        ) throws {
            let index = Qwen4ExpNGramGeometry.recurrentLayerIndex(0)
            let left = try #require(a.confirmedStateSnapshot()?[index])
            let right = try #require(b.confirmedStateSnapshot()?[index])
            let leftConv = try #require(left.conv)
            let rightConv = try #require(right.conv)
            let leftHistory = try #require(left.ssm)
            let rightHistory = try #require(right.ssm)
            #expect(Support.isEqual(leftConv, rightConv))
            #expect(Support.isEqual(leftHistory, rightHistory))
        }

        /// The PLE layer on the CBv2 path, with 2 rows. The deferred decode
        /// step (T=1) and the deferred capture-verify window (T=3) give the
        /// same output and the same committed state as the eager path, bit
        /// for bit. The legacy `ArraysCache` path gives the same output as
        /// the CBv2 path. With 2 rows a decode slot has 8 table rows, which
        /// avoids the one-token known issue of `deferredSlotsMatchTheEagerGather`.
        @Test func layerDeferredPathsMatchTheEagerPath() throws {
            let c = Self.configuration()
            let checkpoint = try Self.write(c)
            try Self.bound(checkpoint.directory) {
                let layer = try Self.layer(c)
                let spec = c.cbv2RecurrentStateSpec()
                let eager = try (0 ..< 2).map { _ in try CBv2RecurrentRequestState(spec: spec) }
                let deferred = try (0 ..< 2).map { _ in
                    try CBv2RecurrentRequestState(spec: spec)
                }

                let prompt = MLXArray([Int32(7), 11, 13, 5, 0, 17], [2, 3])
                let promptHidden = Self.hidden(width: 3, seed: 9201)
                let eagerPrompt = try Self.step(
                    layer, eager, hidden: promptHidden, ids: prompt, deferred: false)
                _ = try Self.step(
                    layer, deferred, hidden: promptHidden, ids: prompt, deferred: false)

                let token = MLXArray([Int32(19), 23], [2, 1])
                let tokenHidden = Self.hidden(width: 1, seed: 9202)
                let before = Qwen4ExpPLEDeferredInvocation.snapshot()
                let eagerStep = try Self.step(
                    layer, eager, hidden: tokenHidden, ids: token, deferred: false)
                let deferredStep = try Self.step(
                    layer, deferred, hidden: tokenHidden, ids: token, deferred: true)
                let after = Qwen4ExpPLEDeferredInvocation.snapshot()
                #expect(after.deferred == before.deferred + 1)
                #expect(after.eager == before.eager + 1)
                #expect(after.line.hasPrefix("pleGather deferred="))
                #expect(deferredStep.shape == [2, 1, 32])
                #expect(Support.isEqual(deferredStep, eagerStep))
                for row in 0 ..< 2 {
                    try Self.expectSameState(eager[row], deferred[row])
                }

                let window = MLXArray([Int32(23), 29, 31, 37, 41, 43], [2, 3])
                let windowHidden = Self.hidden(width: 3, seed: 9203)
                let eagerWindow = try Self.step(
                    layer, eager, hidden: windowHidden, ids: window, deferred: false, capture: 2)
                let deferredWindow = try Self.step(
                    layer, deferred, hidden: windowHidden, ids: window, deferred: true,
                    capture: 2)
                #expect(deferredWindow.shape == [2, 3, 32])
                #expect(Support.isEqual(deferredWindow, eagerWindow))
                for row in 0 ..< 2 {
                    try Self.expectSameState(eager[row], deferred[row])
                }

                // Legacy path: the history and the conv state live in an
                // `ArraysCache`. EOS is token 0, the same as the zero state
                // of a new CBv2 request.
                let cache = ArraysCache(size: 2)
                let legacyPrompt = layer(promptHidden, inputIds: prompt, cache: cache)
                let legacyStep = layer(tokenHidden, inputIds: token, cache: cache)
                eval(legacyPrompt, legacyStep)
                #expect(Support.isEqual(legacyPrompt, eagerPrompt))
                #expect(Support.isEqual(legacyStep, eagerStep))
                #expect(cache.offset == 4)
                layer.releaseExternalResources()
            }
        }
    }
}
