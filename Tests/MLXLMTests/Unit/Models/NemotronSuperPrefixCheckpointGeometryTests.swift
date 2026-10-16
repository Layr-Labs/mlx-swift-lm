import Foundation
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

extension UnitTests {
    /// The chunk-grid checkpoint rule of models whose prefill output depends
    /// on the chunk partition, and the resident bank bytes of the pinned
    /// Super configuration. Swift logic only; no arrays are created.
    struct NemotronSuperPrefixCheckpointGeometryTests {
        typealias Geometry = CBv2RecurrentCheckpointGeometry
        static let grid: Int = 2048

        /// Records one computed range under the grid rule; true at a boundary.
        static func record(
            _ geometry: inout Geometry, _ range: Range<Int>, prompt: Int, packed: Bool = false
        ) -> Bool {
            geometry.record(
                range: range, cap: grid, promptLength: prompt, packed: packed, chunkGrid: grid)
        }

        @Test func wholeGridChunksAreTheOnlyBoundaries() {
            let grid = Self.grid
            let prompt: Int = 3 * grid + 512
            var geometry = Geometry()
            let first = Self.record(&geometry, 0 ..< grid, prompt: prompt)
            let second = Self.record(&geometry, grid ..< (2 * grid), prompt: prompt)
            let third = Self.record(&geometry, (2 * grid) ..< (3 * grid), prompt: prompt)
            #expect(first && second && third)
            // The prompt's last range is shorter; its 256-aligned end is no boundary.
            let tail = Self.record(&geometry, (3 * grid) ..< prompt, prompt: prompt)
            #expect(!tail)
            #expect(geometry.isArmed)
            // The chunk-agnostic stride rule alone would have taken that end.
            #expect(Geometry.isRecurrentBoundary(prompt, chunkSize: grid))
        }

        @Test func aShortenedOffGridOrPackedChunkDisarmsTheRequest() {
            let grid = Self.grid
            let prompt: Int = 4 * grid
            let half: Int = grid / 2
            var shortened = Geometry()
            let short = Self.record(&shortened, 0 ..< half, prompt: prompt)
            let afterShort = Self.record(&shortened, half ..< (half + grid), prompt: prompt)
            #expect(!short && !afterShort)
            #expect(!shortened.isArmed)
            #expect(shortened.disarmReason == .geometry)

            // A restore off the grid continues off the grid: never captured.
            var offGrid = Geometry(position: 256, chunkSize: grid)
            let offGridEnd = Self.record(&offGrid, 256 ..< (256 + grid), prompt: prompt)
            #expect(!offGridEnd)
            #expect(offGrid.disarmReason == .geometry)

            var packed = Geometry()
            let packedEnd = Self.record(&packed, 0 ..< grid, prompt: prompt, packed: true)
            #expect(!packedEnd)
            #expect(packed.disarmReason == .packed)

            // A solo stripe longer than the grid that ends the prompt on a grid
            // multiple: the cold partition would split it, so no boundary.
            let striped: Int = 2 * grid
            var stripe = Geometry()
            let stripeEnd = Self.record(&stripe, 0 ..< striped, prompt: striped)
            #expect(!stripeEnd)
            #expect(stripe.disarmReason == .geometry)
        }

        @Test func aRestoreOnTheGridResumesCapture() {
            let grid = Self.grid
            let restored: Int = 2 * grid
            let prompt: Int = restored + grid + 7
            var geometry = Geometry(position: restored, chunkSize: grid)
            let next = Self.record(&geometry, restored ..< (restored + grid), prompt: prompt)
            #expect(next)
            let tail = Self.record(&geometry, (restored + grid) ..< prompt, prompt: prompt)
            #expect(!tail)
            // The decode range after the prompt disarms, as without a grid.
            let decode = Self.record(&geometry, prompt ..< (prompt + 1), prompt: prompt)
            #expect(!decode)
            #expect(!geometry.isArmed)
        }

        /// One pinned checkpoint is the 40 Mamba layers' conv/SSM state,
        /// 170,229,760 bytes (162.3 MiB), plus 8,192 KV bytes per token. The
        /// default bank of a 4 GiB grant (KV / 8 = 512 MiB) holds one
        /// two-checkpoint entry at the 16,384-token context limit; a 1 GiB
        /// bank holds two.
        @Test func pinnedBankHoldsWholeEntriesAtTheContextLimit() throws {
            let language = try NemotronSuperModel.validatedLanguageConfiguration(
                NemotronSuperDecodeChargeArithmeticTests.pinnedLanguage)
            let checkpoint = try NemotronSuperDecodeCharge.recurrentBytesPerGeneration(
                configuration: language, convolutionDType: .bfloat16)
            let perToken = try NemotronSuperDecodeCharge.attentionKVBytesPerToken(
                configuration: language, elementBytes: 2)
            #expect(checkpoint == 170_229_760)
            #expect(perToken == 8_192)
            let context: Int = 16_384
            let entry: Int = 2 * checkpoint + context * perToken
            let defaultBank: Int = min(1 << 30, (4 << 30) / 8)
            let largeBank: Int = 1 << 30
            #expect(defaultBank / entry == 1)
            #expect(largeBank / entry == 2)
        }
    }
}
