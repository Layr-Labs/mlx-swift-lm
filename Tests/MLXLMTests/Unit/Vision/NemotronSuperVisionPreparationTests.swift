import Testing
@testable import MLXVLM

extension UnitTests {
    struct NemotronSuperVisionPreparationTests {
        @Test func separateAdjacentImagesKeepCausalRanges() throws {
            let ranges = try NemotronSuperVisionPreparation.ranges(
                prompt: [1, 18, 18, 18, 2, 18, 18, 3], featureCounts: [2, 1, 2])
            #expect(ranges == [1..<3, 3..<4, 5..<7])
        }

        @Test func unmatchedOrInterruptedPlaceholdersAreRejected() {
            for (prompt, counts) in [([Int32(1), 18, 2, 18], [2]),
                                     ([Int32(18), 18], [1]),
                                     ([Int32(18)], [2]),
                                     ([Int32(18)], [0]),
                                     ([Int32(18)], [Int.max])] {
                #expect(throws: (any Error).self) {
                    try NemotronSuperVisionPreparation.ranges(prompt: prompt, featureCounts: counts)
                }
            }
        }
    }
}
