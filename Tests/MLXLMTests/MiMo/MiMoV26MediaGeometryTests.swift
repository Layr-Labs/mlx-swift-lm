import Foundation

#if !MIMO_MEDIA_CPU_PROBE
import MLXVLM
import XCTest
#endif

private enum MiMoV26MediaChecks {
    typealias Geometry = MiMoV26MediaGeometry
    struct Failure: Error, CustomStringConvertible { let description: String }
    static func expect(_ value: Bool, _ message: String) throws {
        if !value { throw Failure(description: message) }
    }
    static func native() throws -> Geometry.Settings {
        try .init(patchSize: 16, mergeSize: 2, temporalPatchSize: 2, temporalCompressionRatio: 1,
                  imageMinPixels: 8192, imageMaxPixels: 8388608,
                  videoMinPixels: 8192, videoMaxPixels: 8388608, videoTotalMaxPixels: 268435456)
    }
    static func rejects(_ body: () throws -> Void) throws {
        do { try body(); throw Failure(description: "invalid geometry accepted") }
        catch is Geometry.Failure {}
    }
    static func pureCases() -> [(String, () throws -> Void)] {
        [
            ("native still-image repetition and patch accounting", {
                let p = try Geometry.image(height: 64, width: 128, settings: native())
                try expect(p.height == 64 && p.width == 128 && p.gridT == 1 && p.gridH == 4 && p.gridW == 8,
                           "native image grid changed")
                try expect(p.alignedFrames == 2 && p.duplicatedFrames == 1 && p.patchCount == 32
                           && p.patchVectorSize == 1536 && p.patchElementCount == 49152
                           && p.mediaTokens == 8 && p.promptTokensExcludingTimestamps == 10,
                           "image patch/token accounting changed")
            }),
            ("native ties round to even rather than away from zero", {
                let s = try Geometry.Settings(patchSize: 16, mergeSize: 2, temporalPatchSize: 2,
                    temporalCompressionRatio: 1, imageMinPixels: 1, imageMaxPixels: 1000000,
                    videoMinPixels: 1, videoMaxPixels: 1000000, videoTotalMaxPixels: 1000000)
                let p = try Geometry.image(height: 48, width: 80, settings: s)
                try expect(p.height == 64 && p.width == 64, "Python round(1.5)/round(2.5) parity")
            }),
            ("native tiny-axis branch precedes aspect refusal", {
                let p = try Geometry.image(height: 1, width: 201, settings: native())
                try expect(p.height == 32 && p.width == 6432, "tiny aspect branch incorrectly rejected or clamped")
                try rejects { _ = try Geometry.image(height: 32, width: 6432, settings: native()) }
            }),
            ("video duplicates last odd frame and separates timestamp tokens", {
                let p = try Geometry.video(height: 64, width: 128, sampledFrames: 3, settings: native())
                try expect(p.alignedFrames == 4 && p.duplicatedFrames == 1 && p.gridT == 2
                           && p.mediaTokens == 16 && p.timestampCount == 2
                           && p.fixedWrapperTokens == 6 && p.promptTokensExcludingTimestamps == 22,
                           "video padding/token accounting")
            }),
            ("full sampled clip determines segment pixel cap", {
                let p = try Geometry.video(height: 64, width: 128, sampledFrames: 1000,
                                           segmentFrames: 3, settings: native())
                try expect(p.effectiveMaxPixelsPerFrame == 536870 && p.alignedFrames == 4,
                           "segment count replaced full clip divisor")
            }),
            ("sampling maximum is not silently a geometry limit", {
                _ = try Geometry.video(height: 64, width: 128, sampledFrames: 3601, settings: native())
                try rejects {
                    _ = try Geometry.video(height: 64, width: 128, sampledFrames: 3601,
                                           sampledFrameLimit: 3600, settings: native())
                }
            }),
            ("native minimum floor reports soft aggregate overrun", {
                let s = try Geometry.Settings(patchSize: 16, mergeSize: 2, temporalPatchSize: 2,
                    temporalCompressionRatio: 1, imageMinPixels: 8192, imageMaxPixels: 8388608,
                    videoMinPixels: 8192, videoMaxPixels: 8388608, videoTotalMaxPixels: 1)
                let p = try Geometry.video(height: 64, width: 128, sampledFrames: 1, settings: s)
                try expect(p.effectiveMaxPixelsPerFrame == 8192 && p.aggregateBudgetPixels == 8192
                           && p.exceedsAggregateBudget, "minimum floor was clamped or hidden")
            }),
            ("temporal duplication alone can exceed aggregate budget", {
                let s = try Geometry.Settings(patchSize: 1, mergeSize: 1, temporalPatchSize: 2,
                    temporalCompressionRatio: 1, imageMinPixels: 1, imageMaxPixels: 100,
                    videoMinPixels: 1, videoMaxPixels: 100, videoTotalMaxPixels: 10)
                let p = try Geometry.video(height: 2, width: 3, sampledFrames: 3, settings: s)
                try expect(p.effectiveMaxPixelsPerFrame == 6 && p.aggregateBudgetPixels == 12
                           && p.exceedsAggregateBudget && !p.exceedsPerFrameMaximum,
                           "odd padding aggregate cost omitted")
            }),
            ("native and generic HF sidecar bounds remain distinct", {
                let generic = try Geometry.Settings(patchSize: 16, mergeSize: 2, temporalPatchSize: 2,
                    temporalCompressionRatio: 1, imageMinPixels: 3136, imageMaxPixels: 12845056,
                    videoMinPixels: 8192, videoMaxPixels: 8388608, videoTotalMaxPixels: 268435456)
                let n = try Geometry.image(height: 64, width: 64, settings: native())
                let g = try Geometry.image(height: 64, width: 64, settings: generic)
                try expect(n.height == 96 && n.width == 96 && g.height == 64 && g.width == 64,
                           "generic metadata replaced native bounds")
            }),
            ("invalid and unrepresentable inputs reject recoverably", {
                for (h, w) in [(0, 64), (-1, 64), (64, 0), (Int.max, 64), (Int(Int32.max), Int(Int32.max))] {
                    try rejects { _ = try Geometry.image(height: h, width: w, settings: native()) }
                }
                for frames in [0, -1, Int.max] {
                    try rejects { _ = try Geometry.video(height: 64, width: 64, sampledFrames: frames, settings: native()) }
                }
                try rejects { _ = try Geometry.video(height: 64, width: 64, sampledFrames: 4, segmentFrames: 5, settings: native()) }
                try rejects { _ = try Geometry.video(height: 64, width: 64, sampledFrames: 4, sampledFrameLimit: 0, settings: native()) }
            }),
            ("decoded settings cannot bypass operational validation", {
                var raw = try JSONSerialization.jsonObject(with: JSONEncoder().encode(native())) as! [String: Any]
                raw["patchSize"] = 0
                let decoded = try JSONDecoder().decode(Geometry.Settings.self, from: JSONSerialization.data(withJSONObject: raw))
                try rejects { _ = try Geometry.image(height: 64, width: 64, settings: decoded) }
                raw["patchSize"] = Int.max
                let large = try JSONDecoder().decode(Geometry.Settings.self, from: JSONSerialization.data(withJSONObject: raw))
                try rejects { _ = try Geometry.image(height: 64, width: 64, settings: large) }
            }),
            ("patch count overflow and native soft spatial bounds", {
                try rejects {
                    _ = try Geometry.video(height: 64, width: 128, sampledFrames: Int(Int32.max) - 1,
                                           settings: native())
                }
                let s = try Geometry.Settings(patchSize: 16, mergeSize: 2, temporalPatchSize: 2,
                    temporalCompressionRatio: 1, imageMinPixels: 5000, imageMaxPixels: 5000,
                    videoMinPixels: 8192, videoMaxPixels: 8388608, videoTotalMaxPixels: 268435456)
                let up = try Geometry.image(height: 32, width: 32, settings: s)
                let down = try Geometry.image(height: 100, width: 100, settings: s)
                try expect(up.height == 96 && up.width == 96 && up.exceedsPerFrameMaximum,
                           "native ceil overshoot was silently clamped")
                try expect(down.height == 64 && down.width == 64 && down.belowPerFrameMinimum,
                           "native floor undershoot was silently clamped")
            }),
        ]
    }

    struct Vector: Decodable {
        let id, kind: String
        let height, width: Int
        let sampledFrames, segmentFrames, sampledFrameLimit: Int?
        let settings: Geometry.Settings
        let expected: Geometry.Plan?
        let expectsFailure: Bool
    }
    struct Corpus: Decodable { let schemaVersion: Int; let vectors: [Vector] }
    static func referenceCases(_ path: String) throws -> [(String, () throws -> Void)] {
        let corpus = try JSONDecoder().decode(Corpus.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        try expect(corpus.schemaVersion == 1 && !corpus.vectors.isEmpty, "empty reference corpus")
        try expect(Set(corpus.vectors.map(\.id)).count == corpus.vectors.count, "duplicate reference IDs")
        return corpus.vectors.map { v in
            (v.id, {
                func run() throws -> Geometry.Plan {
                    if v.kind == "image" { return try Geometry.image(height: v.height, width: v.width, settings: v.settings) }
                    try expect(v.kind == "video" && v.sampledFrames != nil, "invalid fixture kind")
                    return try Geometry.video(height: v.height, width: v.width, sampledFrames: v.sampledFrames!,
                        segmentFrames: v.segmentFrames, sampledFrameLimit: v.sampledFrameLimit, settings: v.settings)
                }
                if v.expectsFailure { try rejects { _ = try run() } }
                else { try expect(try run() == v.expected, "scalar reference divergence: \(v.id)") }
            })
        }
    }
}

#if MIMO_MEDIA_CPU_PROBE
@main private struct MiMoV26MediaCPUProbe {
    static func main() throws {
        guard CommandLine.arguments.count == 2 else {
            throw MiMoV26MediaChecks.Failure(description: "expected scalar vector path")
        }
        let tests = MiMoV26MediaChecks.pureCases() + (try MiMoV26MediaChecks.referenceCases(CommandLine.arguments[1]))
        var failures = 0
        for (name, test) in tests {
            do { try test(); print("PASS \(name)") }
            catch { failures += 1; print("FAIL \(name): \(error)") }
        }
        print("RESULT discovered=\(tests.count) passed=\(tests.count - failures) failed=\(failures) skipped=0")
        if failures > 0 { throw MiMoV26MediaChecks.Failure(description: "geometry probe failed") }
    }
}
#else
final class MiMoV26MediaGeometryTests: XCTestCase {
    func testNativeGeometryContracts() throws {
        for (name, test) in MiMoV26MediaChecks.pureCases() {
            do { try test() } catch { XCTFail("\(name): \(error)") }
        }
    }
    func testIndependentScalarVectors() throws {
        guard let path = ProcessInfo.processInfo.environment["MIMO_MEDIA_REFERENCE_VECTORS"] else {
            throw XCTSkip("Set MIMO_MEDIA_REFERENCE_VECTORS for the independent scalar reference corpus")
        }
        for (name, test) in try MiMoV26MediaChecks.referenceCases(path) {
            do { try test() } catch { XCTFail("\(name): \(error)") }
        }
    }
}
#endif
