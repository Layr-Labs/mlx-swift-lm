import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Measured residency of recurrent rows after requests leave a batch. These
/// read process-global MLX memory: run this suite in its own `swift test`
/// invocation. Rows are large (16 MiB) and thresholds are half the expected
/// bytes so small concurrent allocations cannot decide the outcome.
@Suite(.tags(.integration), .serialized)
struct NemotronSuperBatchResidencyTests {
    /// A plain hybrid trunk with 16 MiB of SSM state per request.
    static func wideStateModel() throws -> NemotronSuperModel {
        let configuration = NemotronHConfiguration(
            vocabSize: 64, hiddenSize: 128, numHiddenLayers: 5,
            numAttentionHeads: 2, numKeyValueHeads: 1, mambaNumHeads: 128, mambaHeadDim: 64,
            ssmStateSize: 128, convKernel: 4, nGroups: 1, intermediateSize: 128,
            moeIntermediateSize: 64, moeSharedExpertIntermediateSize: 64,
            nRoutedExperts: 4, numExpertsPerTok: 2, hybridOverridePattern: "MM*MM", headDim: 64)
        let language = NemotronHModel(configuration)
        eval(language.parameters())
        let images = NemotronSuperImageComposition(
            tower: NemotronRadio(
                NemotronRadioDimensions(
                    width: 2, layers: 1, heads: 1,
                    channels: 1, patch: 2, positionSide: 2, prefix: 1, mlpWidth: 3)),
            projector: NemotronSuperVisionProjector(
                visionWidth: 2, hiddenWidth: 3, outputWidth: 128),
            width: 128, imageTokenID: 18)
        return try NemotronSuperModel(languageModel: language, imageComposition: images)
    }

    /// Resident bytes after command-buffer temporaries have retired.
    static func resident() -> Int {
        Stream.defaultStream(.gpu).synchronize()
        usleep(100_000)
        Memory.clearCache()
        return Memory.activeMemory
    }

    /// MLX row slices share their parent allocation: dropping the parent and
    /// seven of eight row views frees nothing; only a copy is independent.
    @Test func rowViewsShareTheirBatchAllocation() throws {
        let before = Self.resident()
        var views: [MLXArray] = []
        do {
            let batch = MLXRandom.normal([8, 128, 64, 128 * 8])
            eval(batch)
            views = (0 ..< 8).map { batch[$0 ..< ($0 + 1)] }
            eval(views)
        }
        let batchBytes = 8 * 128 * 64 * 128 * 8 * 4
        #expect(Self.resident() - before >= batchBytes / 2)
        views.removeLast(7)
        let afterSeven = Self.resident() - before
        let copy = views[0] * 1
        eval(copy)
        views.removeAll()
        let afterCopy = Self.resident() - before
        print(
            "BATCH residency: 8-row buffer \(batchBytes) B; after dropping 7 views \(afterSeven) B, after copying the last row \(afterCopy) B"
        )
        #expect(afterSeven >= batchBytes / 2)
        #expect(afterCopy < batchBytes / 4)
    }

    struct Residency {
        let perRow: Int
        let freedAtRelease: Int
        let freedAtSurvivorCommit: Int
        let freedAtEnd: Int
    }

    /// Eight requests decode one batched step, seven are released, the
    /// survivor decodes once more alone and is released; resident bytes are
    /// sampled after each phase.
    static func departure(_ products: NemotronSuperRecurrentAdapter.DecodeProducts) throws
        -> Residency
    {
        let model = try Self.wideStateModel()
        let rows = 8
        let runner = try NemotronSuperBatchRunner(
            model: model, maximumRows: rows, kvDType: .float32, products: products)
        let lanes = try (0 ..< rows).map {
            try runner.lane(prompt: [Int32(1 + $0), 3, 5], inputs: [7, 9, 11])
        }
        for lane in lanes { try runner.prefill(lane) }
        func resident() -> Int {
            for lane in lanes { lane.logits.removeAll() }
            return Self.resident()
        }
        try runner.decode(lanes)
        let batched = resident()
        for lane in lanes.dropFirst() { try lane.state.release() }
        let departed = resident()
        try runner.decode([lanes[0]])
        let survivorCommitted = resident()
        try lanes[0].state.release()
        let released = resident()
        for lane in lanes { runner.backend.release(lane.rows) }
        let result = Residency(
            perRow: try model.cbv2RecurrentStateSpec.fixedBytesPerRequest(),
            freedAtRelease: batched - departed,
            freedAtSurvivorCommit: batched - survivorCommitted, freedAtEnd: batched - released)
        print(
            "BATCH residency \(products): per-request recurrent generation \(result.perRow) B; "
                + "freed after releasing 7 of 8: \(result.freedAtRelease) B, after the "
                + "survivor's next commit: \(result.freedAtSurvivorCommit) B, after releasing "
                + "the survivor: \(result.freedAtEnd) B")
        return result
    }

    /// Shared products stage every request's recurrent rows as views of one
    /// buffer per layer: releasing departed requests frees none of their
    /// bytes while a survivor holds that buffer, and the survivor's next
    /// commit frees them. The declared alias reserve covers what stays.
    @Test func sharedProductRowsStayResidentUntilTheSurvivorCommits() throws {
        let measured = try Self.departure(.shared)
        let departedBytes = 7 * measured.perRow
        #expect(measured.freedAtRelease < departedBytes / 2)
        #expect(measured.freedAtSurvivorCommit >= departedBytes / 2)
        #expect(measured.freedAtEnd >= departedBytes / 2 + measured.perRow)
        // This trunk has no ModelOpt modules; only the recurrent terms matter.
        let charge = NemotronSuperDecodeCharge(
            recurrentBytesPerGeneration: measured.perRow,
            recurrentGenerations: CBv2RecurrentStateSpec.maximumLiveGenerations,
            attentionKVBytesPerToken: 0,
            rowWorkspace: .init(
                modelOptLayerBytes: 0, outputHeadBytes: 0, logitsBytes: 0, trunkBytes: 0,
                backendScratchBytes: 0, totalBytes: 0))
        let lingering = measured.freedAtEnd - measured.freedAtRelease - measured.perRow
        #expect(lingering <= (try charge.batchAliasBytes(maximumRows: 8)))
    }

    /// Row-exact batched decode stages every row in its own buffer: a
    /// departed request's recurrent bytes are freed when it is released.
    @Test func rowExactRowsAreFreedWithTheirRequest() throws {
        let measured = try Self.departure(.rowExact)
        let departedBytes = 7 * measured.perRow
        #expect(measured.freedAtRelease >= departedBytes / 2)
        #expect(measured.freedAtEnd >= departedBytes / 2 + measured.perRow)
    }
}
