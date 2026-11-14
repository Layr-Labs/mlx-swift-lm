// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// Exact verify windows on tiny towers with the production attention shape:
/// one `[B, W]` forward on the paged caches gives every row the logits and
/// K/V bytes of the one-row decode steps it replaces, and prompt chunks keep
/// MLX's arithmetic.
@Suite(.serialized, .tags(.integration))
struct MuseGlimmerExactWindowTests {
    /// Large enough that MLX's kernels give a window other bytes than steps.
    static let hidden = 512
    static let intermediate = 1024
    static let window = 16
    static let chunk = 256

    /// Dense projections only.
    static func dense(_ dtype: DType = .bfloat16) throws -> MuseGlimmerTextModel {
        try MuseGlimmerFixture.model(
            MuseGlimmerFixture.pagedText(
                window: window, layers: 4, hidden: hidden, intermediate: intermediate),
            dtype: dtype)
    }

    /// FP8, weight-only NVFP4 and BF16 projections side by side, staged by
    /// the installer.
    static func mixed(
        order: [MuseGlimmerInstallation.ProjectionFormat] = [.fp8, .nvfp4, .dense],
        head: MuseGlimmerInstallation.ProjectionFormat = .nvfp4, layers: Int = 4,
        dtype: DType = .bfloat16
    ) throws -> MuseGlimmerTextModel {
        let checkpoint = try MuseGlimmerTinyCheckpoint(
            layers: layers, hidden: hidden, intermediate: intermediate, window: window,
            order: order, head: head, dtype: dtype)
        let written = try checkpoint.write()
        defer { try? FileManager.default.removeItem(at: written.directory) }
        return try MuseGlimmerInstallation.stage(
            configuration: checkpoint.configuration, source: try written.source(),
            dtype: dtype, activationOverflow: .museGlimmer
        ).model
    }

    static func exact(_ m: MuseGlimmerTextModel) throws -> MuseGlimmerTextModel {
        try m.setRowExactWindows(true)
        return m
    }

    /// Paged rows on one backend, each prefilled alone.
    final class Paged {
        let model: MuseGlimmerTextModel
        let backend: PagedKVBackend
        let caches: [PagedLayerCache]
        private(set) var states: [[CBv2SequenceKV?]] = []

        let chunk: Int

        init(
            _ m: MuseGlimmerTextModel, capacityBytes: Int = 192 << 20,
            chunk: Int = MuseGlimmerExactWindowTests.chunk, segmentBytes: Int = 1 << 20,
            maximumLength: Int = 2048
        ) throws {
            model = m
            self.chunk = chunk
            backend = try PagedKVBackend(
                layerKinds: m.cbv2LayerKinds,
                config: .init(
                    capacityBytes: capacityBytes, dtype: m.activationDType,
                    maxPrefillChunk: chunk, nominalMaxSequenceLength: maximumLength,
                    segmentSizeBytes: segmentBytes, layerDTypes: m.kvStorageDTypes))
            caches = backend.makeLayerCaches()
        }

        deinit {
            eval(caches.flatMap { $0.innerState() })
            for cache in caches { cache.setRows([]) }
            for state in states { backend.release(state) }
        }

        func add(_ prompt: [Int32], extra: Int) throws {
            let state = try backend.makeSequenceState(
                layerKinds: model.cbv2LayerKinds, promptLength: prompt.count,
                maxLength: prompt.count + extra)
            states.append(state)
            select([states.count - 1])
            for start in stride(from: 0, to: prompt.count, by: chunk) {
                let piece = Array(prompt[start ..< min(prompt.count, start + chunk)])
                _ = forward([piece], rectangular: false)
            }
        }

        /// A new row prefilled through a prompt entry point, as the engine
        /// prefills: `cbv2Prefill`, or `prefillWithMTPHidden` with `mtp`.
        /// Returns each chunk's last-row logits, then its drafter hidden.
        func addPrompt(_ prompt: [Int32], extra: Int, mtp: Bool = false) throws -> [MLXArray] {
            let state = try backend.makeSequenceState(
                layerKinds: model.cbv2LayerKinds, promptLength: prompt.count,
                maxLength: prompt.count + extra)
            states.append(state)
            select([states.count - 1])
            return stride(from: 0, to: prompt.count, by: chunk).flatMap { start -> [MLXArray] in
                let piece = Array(prompt[start ..< min(prompt.count, start + chunk)])
                let input = MLXArray(piece).reshaped(1, -1)
                let outputs: [MLXArray]
                if mtp {
                    let result = model.prefillWithMTPHidden(
                        input, cache: caches, requirement: .lastPositionLogits)
                    outputs = [result.logits, result.hidden]
                } else {
                    outputs = [
                        model.cbv2Prefill(
                            input, inputEmbedding: nil, cache: caches,
                            requirement: .lastPositionLogits)
                    ]
                }
                eval(outputs + caches.flatMap { $0.innerState() })
                return outputs
            }
        }

        func select(_ rows: [Int]) {
            for (layer, cache) in caches.enumerated() {
                cache.setRows(rows.map { states[$0][layer]! })
            }
        }

        /// Logits `[B, W, vocab]` of one forward over the selected rows.
        func forward(_ tokens: [[Int32]], rectangular: Bool, columnsAsRows: Bool = false)
            -> MLXArray
        {
            let input = MLXArray(tokens.flatMap { $0 }).reshaped(tokens.count, -1)
            let logits =
                rectangular
                ? CBv2LayerCacheBank(caches: caches).withMTPRectangularVerification(
                    columnsAsRows: columnsAsRows
                ) {
                    model(input, cache: caches)
                } : model(input, cache: caches)
            eval([logits] + caches.flatMap { $0.innerState() })
            return logits
        }

        /// Layer calls attended as decode rows.
        var columnsAsRowsCalls: Int { caches.map(\.mtpColumnsAsRowsCalls).reduce(0, +) }

        func kv(_ row: Int) -> [Data] {
            states[row].flatMap { state -> [Data] in
                let snapshot = state!.snapshot()
                eval(snapshot.keys, snapshot.values)
                return [
                    snapshot.keys.asData(access: .copy).data,
                    snapshot.values.asData(access: .copy).data,
                ]
            }
        }
    }

    /// Whether one `[B, W]` window equals each row's `W` one-row steps, in
    /// logits and in the K/V written.
    static func windowEqualsSteps(
        _ m: MuseGlimmerTextModel, prompts: [[Int32]], width: Int, columnsAsRows: Bool = false,
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws -> Bool {
        let drafts = prompts.indices.map {
            MuseGlimmerFixture.tokens(width, vocabulary: 96, salt: 40 + $0)
        }
        let windowed = try Paged(m)
        for prompt in prompts { try windowed.add(prompt, extra: width) }
        windowed.select(Array(prompts.indices))
        let window = windowed.forward(drafts, rectangular: true, columnsAsRows: columnsAsRows)
        // Every layer attends a window of two or more columns as decode rows.
        #expect(
            windowed.columnsAsRowsCalls
                == (columnsAsRows && width > 1 ? windowed.caches.count : 0),
            sourceLocation: sourceLocation)
        let stepped = try Paged(m)
        var same = true
        for (row, prompt) in prompts.enumerated() {
            try stepped.add(prompt, extra: width)
            stepped.select([row])
            for column in 0 ..< width {
                let step = stepped.forward([[drafts[row][column]]], rectangular: false)
                let scored = window[row ..< row + 1, column ..< column + 1]
                same = same && MuseGlimmerFixture.sameBits(scored, step)
            }
            same = same && windowed.kv(row) == stepped.kv(row)
        }
        #expect(!windowed.backend.pool.writeValidation.isFaulted, sourceLocation: sourceLocation)
        return same
    }

    // MARK: One sequence

    /// Window widths 1 to 8, starting before, at and after the sliding window
    /// edge and after the ring has wrapped. Float32 towers show the sums
    /// unrounded.
    @Test(arguments: [false, true], [DType.bfloat16, .float32])
    func windowsEqualSequentialSteps(mixedFormats: Bool, dtype: DType) throws {
        let m = try Self.exact(mixedFormats ? Self.mixed(dtype: dtype) : Self.dense(dtype))
        for length in [3, 10, 15, 16, 31, 70] {
            let prompt = MuseGlimmerFixture.tokens(length, vocabulary: 96, salt: length)
            for width in 1 ... 8 {
                #expect(
                    try Self.windowEqualsSteps(m, prompts: [prompt], width: width),
                    "prompt \(length) width \(width)")
            }
        }
    }

    // MARK: Columns as decode rows

    /// Attention as decode rows, one dispatch per layer: the same logits and
    /// K/V. The prompts cross the window edge and the ring's wrap (256
    /// tokens here).
    @Test(arguments: [DType.bfloat16, .float32])
    func windowsAttendedAsDecodeRowsEqualSteps(dtype: DType) throws {
        let m = try Self.exact(Self.mixed(dtype: dtype))
        for length in [3, 15, 16, 31, 70, 250, 260, 520] {
            let prompt = MuseGlimmerFixture.tokens(length, vocabulary: 96, salt: length)
            for width in 1 ... 8 {
                #expect(
                    try Self.windowEqualsSteps(
                        m, prompts: [prompt], width: width, columnsAsRows: true),
                    "prompt \(length) width \(width)")
            }
        }
        let lengths = [250, 3, 15, 260, 16, 31, 70, 520, 9, 47, 100, 255, 2, 33, 129, 64]
        for (sequences, width) in [(2, 8), (5, 7), (8, 8), (16, 2), (16, 4)] {
            let prompts = lengths.prefix(sequences).enumerated().map {
                MuseGlimmerFixture.tokens($0.element, vocabulary: 96, salt: $0.offset)
            }
            #expect(
                try Self.windowEqualsSteps(
                    m, prompts: prompts, width: width, columnsAsRows: true),
                "\(sequences) sequences width \(width)")
        }
    }

    // MARK: Several sequences

    /// 2 to 64 rows in one forward (the pieces of 16 and the reductions from
    /// 32 rows on), each row equal to its sequence's own steps.
    @Test(arguments: [(2, 8), (4, 8), (5, 7), (8, 8), (8, 1)], [DType.bfloat16, .float32])
    func batchedWindowsEqualEachSequencesSteps(shape: (sequences: Int, width: Int), dtype: DType)
        throws
    {
        let (sequences, width) = shape
        let m = try Self.exact(Self.mixed(dtype: dtype))
        let lengths = [3, 15, 16, 31, 70, 9, 47, 100]
        let prompts = lengths.prefix(sequences).enumerated().map {
            MuseGlimmerFixture.tokens($0.element, vocabulary: 96, salt: $0.offset)
        }
        #expect(try Self.windowEqualsSteps(m, prompts: prompts, width: width))
    }

    /// The block norms at the production width: 32 rows and more reduce like
    /// one row (MLX takes another row reduction from 32 rows on).
    @Test(arguments: [DType.bfloat16, .float32], [32, 47, 64, 65, 99, 200])
    func blockNormsAtTheProductionWidthAreRowExact(dtype: DType, rows: Int) throws {
        let numerics = MuseGlimmerNumerics()
        numerics.exactWindows = true
        let norm = MuseGlimmerCenteredRMSNorm(dimensions: 6656, eps: 1e-5, numerics: numerics)
        try norm.update(
            parameters: .unflattened([
                "weight": (MuseGlimmerFixture.hostTensor([6656], seed: 3) * 0.1).asType(dtype)
            ]), verify: [.all])
        let x = (MuseGlimmerFixture.hostTensor([1, rows, 6656], seed: 4) * 3).asType(dtype)
        let residual = MuseGlimmerFixture.hostTensor([1, rows, 6656], seed: 5).asType(dtype)
        let (all, added) = (norm(x), norm.addedTo(residual, x))
        for r in 0 ..< rows {
            let one = x[0..., r ..< r + 1]
            #expect(MuseGlimmerFixture.sameBits(all[0..., r ..< r + 1], norm(one)))
            #expect(
                MuseGlimmerFixture.sameBits(
                    added[0..., r ..< r + 1], norm.addedTo(residual[0..., r ..< r + 1], one)))
        }
    }

    // MARK: Above 64 rows

    /// More rows than a verify window: 65 decode rows, and windows of 9 and 16
    /// sequences (72 and 128 rows), each equal to its sequence's steps.
    @Test(.timeLimit(.minutes(6)), arguments: [DType.bfloat16, .float32])
    func callsAbove64RowsEqualSteps(dtype: DType) throws {
        let m = try Self.exact(Self.mixed(dtype: dtype))
        func prompts(_ count: Int) -> [[Int32]] {
            (0 ..< count).map {
                MuseGlimmerFixture.tokens(3 + $0 % 29, vocabulary: 96, salt: 200 + $0)
            }
        }
        #expect(try Self.windowEqualsSteps(m, prompts: prompts(65), width: 1), "65 x 1")
        for (sequences, width) in [(9, 8), (16, 8)] {
            #expect(
                try Self.windowEqualsSteps(
                    m, prompts: prompts(sequences), width: width, columnsAsRows: true),
                "\(sequences) x \(width)")
        }
    }

    /// Decode and verify calls above 64 rows take the pieces too: every
    /// projection of the tower gives each of 65, 99 and 200 rows the bytes of
    /// its one-row call.
    @Test(arguments: [DType.bfloat16, .float32])
    func everyProjectionIsRowExactAbove64Rows(dtype: DType) throws {
        let m = try Self.exact(Self.mixed(dtype: dtype))
        for (index, (path, layer)) in m.projectionSlots.enumerated() {
            let projection = try #require(layer as? any UnaryLayer, "\(path)")
            let inputs: Int
            switch layer {
            case let nvfp4 as ModelOptWeightOnlyNVFP4Linear: inputs = nvfp4.inputDimensions
            case let fp8 as ModelOptCalibratedLinear: inputs = fp8.inputDimensions
            case let dense as Linear: inputs = dense.weight.dim(1)
            default: Issue.record("\(path)"); continue
            }
            for rows in [65, 99, 200] {
                let x = MuseGlimmerFixture.hostTensor(
                    [1, rows, inputs], seed: UInt64(index * 1000 + rows)
                ).asType(dtype)
                let all = m.numerics.project(projection, x)
                let one = (0 ..< rows).map { m.numerics.project(projection, x[0..., $0 ..< $0 + 1]) }
                #expect(
                    MuseGlimmerFixture.sameBits(all, concatenated(one, axis: 1)),
                    "\(path) rows \(rows)")
            }
        }
    }

    // MARK: Prompt chunks

    /// Prompt chunks keep MLX's arithmetic with exact windows on: through
    /// either prompt entry point, at every chunk size, the K/V, last-row
    /// logits and drafter hidden equal the plain tower's. The same chunks as
    /// decode or verify rows (`callAsFunction`) take the exact pieces, which
    /// in float32 write other K/V bytes.
    @Test(arguments: [DType.bfloat16, .float32])
    func promptChunksKeepMLXArithmetic(dtype: DType) throws {
        let m = try Self.mixed(dtype: dtype)
        m.mtpHiddenLayerIDs = [1, 3]
        let prompt = MuseGlimmerFixture.tokens(200, vocabulary: 96, salt: 12)
        for chunk in [1, 7, 16, 33, 64, 65, 200] {
            try m.setRowExactWindows(false)
            let plain = try Paged(m, chunk: chunk)
            let expected = try plain.addPrompt(prompt, extra: 1, mtp: true)
            try m.setRowExactWindows(true)
            let exact = try Paged(m, chunk: chunk)
            let logits = try exact.addPrompt(prompt, extra: 1)
            let withHidden = try exact.addPrompt(prompt, extra: 1, mtp: true)
            #expect(exact.kv(0) == plain.kv(0) && exact.kv(1) == plain.kv(0), "chunk \(chunk)")
            let expectedLogits = stride(from: 0, to: expected.count, by: 2).map { expected[$0] }
            #expect(
                zip(logits, expectedLogits).allSatisfy { MuseGlimmerFixture.sameBits($0, $1) }
                    && logits.count == expectedLogits.count, "chunk \(chunk)")
            #expect(
                zip(withHidden, expected).allSatisfy { MuseGlimmerFixture.sameBits($0, $1) }
                    && withHidden.count == expected.count, "chunk \(chunk)")
            if dtype == .float32 && chunk > 16 {
                let asRows = try Paged(m, chunk: chunk)
                try asRows.add(prompt, extra: 1)
                #expect(asRows.kv(0) != plain.kv(0), "chunk \(chunk)")
            }
            #expect(m.verifiesWindowsExactly && !m.numerics.buildingPrompt)
        }
    }

    // MARK: The switch

    /// Off by default; MLX's kernels then give a float32 window other bytes
    /// (BF16 rounding hides most of the difference at these sizes).
    @Test(arguments: [false, true])
    func exactWindowsAreOffByDefaultAndTheyMatter(mixedFormats: Bool) throws {
        let m = try mixedFormats ? Self.mixed(dtype: .float32) : Self.dense(.float32)
        #expect(!m.verifiesWindowsExactly && m.numericsProfile == "mlx")
        #expect(!m.dflashVerifiesWindowsExactly)
        let prompt = MuseGlimmerFixture.tokens(15, vocabulary: 96, salt: 1)
        #expect(try !Self.windowEqualsSteps(m, prompts: [prompt], width: 8))
        try m.setRowExactWindows(true)
        #expect(m.verifiesWindowsExactly && m.dflashVerifiesWindowsExactly)
        #expect(m.numericsProfile == "muse-glimmer-exact-windows-v3")
        #expect(try Self.windowEqualsSteps(m, prompts: [prompt], width: 8))
        try m.setRowExactWindows(false)
        #expect(!m.verifiesWindowsExactly)
        for case (_, let layer as ModelOptCalibratedLinear) in m.projectionSlots {
            #expect(layer.rowExactSchedule == .standard)
        }
    }

    /// BF16 and FP8 decode steps keep their bytes with exact windows on: a
    /// tower prefilled and stepped with the switch on equals the plain tower,
    /// in the prompt chunk's K/V and logits and in each one-row step.
    @Test(arguments: [false, true])
    func decodeBytesDoNotMoveForBF16AndFP8(fp8: Bool) throws {
        func run(exact: Bool) throws -> (prefill: [MLXArray], kv: [Data], steps: [MLXArray]) {
            let m = fp8 ? try Self.mixed(order: [.fp8, .dense], head: .dense) : try Self.dense()
            if exact {
                _ = try Self.exact(m)
                #expect(m.verifiesWindowsExactly)
            }
            let paged = try Paged(m)
            let prompt = MuseGlimmerFixture.tokens(100, vocabulary: 96, salt: 5)
            let prefill = try paged.addPrompt(prompt, extra: 140)
            let kv = paged.kv(0)
            let steps = MuseGlimmerFixture.tokens(24, vocabulary: 96, salt: 6).map {
                paged.forward([[$0]], rectangular: false)
            }
            return (prefill, kv, steps)
        }
        let plain = try run(exact: false)
        let exact = try run(exact: true)
        #expect(plain.prefill.count == 1 && exact.prefill.count == 1)
        #expect(MuseGlimmerFixture.sameBits(plain.prefill[0], exact.prefill[0]))
        #expect(plain.kv == exact.kv)
        #expect(plain.steps.count == 24 && exact.steps.count == 24)
        #expect(zip(plain.steps, exact.steps).allSatisfy { MuseGlimmerFixture.sameBits($0, $1) })
    }

    @Test func aProjectionWithoutARowExactProductIsRefused() throws {
        // Hidden 64: the query projection reads 64 inputs, below the FP8 and
        // BF16 row-exact products' range.
        let m = try MuseGlimmerFixture.model(
            MuseGlimmerFixture.pagedText(window: Self.window, layers: 2), dtype: .bfloat16)
        #expect(throws: MuseGlimmerRowExactError.self) { try m.setRowExactWindows(true) }
        #expect(!m.verifiesWindowsExactly)
    }
}
