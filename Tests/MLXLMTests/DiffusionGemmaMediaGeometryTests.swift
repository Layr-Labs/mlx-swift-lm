import Foundation
import MLXVLM
import Testing

@Suite("DiffusionGemma reference media geometry")
struct DiffusionGemmaMediaGeometryTests {
    private struct Fixture: Decodable {
        struct Row: Decodable {
            let width: Int, height: Int, budget: Int
            let resizedWidth: Int, resizedHeight: Int, softTokens: Int
        }
        let source: String
        let cases: [Row]
    }

    @Test func actualReferenceProcessorDimensionsMatch() throws {
        let url = try #require(Bundle.module.url(forResource: "diffusiongemma-media-geometry", withExtension: "json"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        #expect(fixture.source == "e79b0e041677ec4ca5333ba750376bb4e8c434cb")
        try #require(fixture.cases.count == 40)
        for item in fixture.cases {
            let actual = try DiffusionGemmaMediaGeometry.resized(width: item.width, height: item.height,
                patchSize: 16, poolingSize: 3, maxSoftTokens: item.budget)
            #expect(actual.width == item.resizedWidth)
            #expect(actual.height == item.resizedHeight)
            #expect(actual.softTokens == item.softTokens)
            #expect(actual.softTokens <= item.budget)
        }
    }

    @Test func budgetIsNotAFixedPlaceholderCount() throws {
        let image = try DiffusionGemmaMediaGeometry.resized(width: 512, height: 512,
            patchSize: 16, poolingSize: 3, maxSoftTokens: 280)
        #expect(image.softTokens == 256 && image.width == 768 && image.height == 768)
        let frame = try DiffusionGemmaMediaGeometry.resized(width: 512, height: 512,
            patchSize: 16, poolingSize: 3, maxSoftTokens: 70)
        #expect(frame.softTokens == 64)
    }

    @Test func malformedGeometryRejectsInsteadOfTrapping() {
        for (w, h, patch, pool, budget) in [
            (0, 1, 16, 3, 280), (1, -1, 16, 3, 280), (1, 1, 0, 3, 280),
            (1, 1, 16, 0, 280), (1, 1, 16, 3, 281), (1, 1, Int.max, 3, 280),
            (1, 1, Int.max / 2, 1, 280),
        ] {
            #expect(throws: (any Error).self) {
                try DiffusionGemmaMediaGeometry.resized(width: w, height: h,
                    patchSize: patch, poolingSize: pool, maxSoftTokens: budget)
            }
        }
    }
}
