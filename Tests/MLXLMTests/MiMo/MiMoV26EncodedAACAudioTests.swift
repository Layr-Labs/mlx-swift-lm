// Copyright © 2026 Eigen Labs.
import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class MiMoV26EncodedAACAudioTests: XCTestCase {
    private let videoLimits = MiMoV26EncodedVisualDecoder.Limits(
        maximumPixels: 10000,
        maximumWorkingBytes: 64 << 20, maximumSourceFrames: 1000, maximumSampledFrames: 16)
    private let audioLimits = MiMoV26EncodedAudiovisualDecoder.Limits(
        maximumFrames: 48000,
        maximumWorkingBytes: 64 << 20)
    private func video() async throws -> MiMoV26EncodedVisualDecoder.VideoPlan {
        let data = try XCTUnwrap(Data(base64Encoded: Self.fixture))
        return try await MiMoV26EncodedVisualDecoder.inspectVideo(
            MemoryBackedVideoAsset(videoData: data),
            sampling: .init(fps: 1, minimumFrames: 8, maximumFrames: 16), limits: videoLimits)
    }
    func testActualAACRetainsStereoRateAndCompletePresentedTimeline() async throws {
        let plan = try await MiMoV26EncodedAudiovisualDecoder.inspect(video(), limits: audioLimits)
        XCTAssertEqual(plan.channels, 2)
        XCTAssertEqual(plan.sampleRate, 32000)
        XCTAssertEqual(plan.frameCount, 32000)
        XCTAssertEqual(plan.sampleCount, 64000)
        let decoded = try await MiMoV26EncodedAudiovisualDecoder.decode(
            plan, videoLimits: videoLimits, audioLimits: audioLimits)
        XCTAssertEqual(decoded.wholeAudio.descriptor.channels, 2)
        XCTAssertEqual(decoded.wholeAudio.descriptor.sampleRate, 32000)
        XCTAssertEqual(decoded.wholeAudio.descriptor.frameCount, 32000)
        XCTAssertEqual(decoded.wholeAudio.samples.count, 64000)
        XCTAssertTrue(decoded.wholeAudio.samples.allSatisfy(\.isFinite))
        XCTAssertTrue(decoded.wholeAudio.samples.contains { abs($0) > 0.01 })
        XCTAssertTrue(
            decoded.wholeAudio.descriptor.sourceIdentity.hasPrefix("mimo-av-aac-32000-2:"))
        XCTAssertGreaterThanOrEqual(decoded.frames.count, 2)
    }
    func testAACMetadataBoundsRefuseBeforeDecompressionAndOnPlanReuse() async throws {
        let video = try await video()
        for limits in [
            MiMoV26EncodedAudiovisualDecoder.Limits(
                maximumFrames: 31999, maximumWorkingBytes: 64 << 20),
            .init(maximumFrames: 48000, maximumWorkingBytes: 1 << 20),
            .init(maximumFrames: 48000, maximumWorkingBytes: 64 << 20, maximumChannels: 1),
            .init(maximumFrames: 48000, maximumWorkingBytes: 64 << 20, maximumSampleRate: 24000),
        ] {
            do {
                _ = try await MiMoV26EncodedAudiovisualDecoder.inspect(video, limits: limits)
                XCTFail("AAC bound ignored")
            } catch { XCTAssertTrue(error is MiMoV26EncodedAudiovisualDecoder.Failure) }
        }
        let plan = try await MiMoV26EncodedAudiovisualDecoder.inspect(video, limits: audioLimits)
        do {
            _ = try await MiMoV26EncodedAudiovisualDecoder.decode(
                plan, videoLimits: videoLimits,
                audioLimits: .init(
                    maximumFrames: 48000, maximumWorkingBytes: 64 << 20, maximumChannels: 1))
            XCTFail("reused AAC plan bypassed channel cap")
        } catch { XCTAssertEqual(error as? MiMoV26EncodedAudiovisualDecoder.Failure, .limit) }
    }
    // Original synthetic black video + 440 Hz tone, 1 s, H.264 + stereo AAC.
    // AVAssetReader performs real decompression, including container priming.
    private static let fixture = [
        "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAZTbW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD6AAAA+gAAQAAAQAAAAAAAAAAAAAAAAEAAAAA",
        "AAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwAAAqB0cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAAB",
        "AAAAAAAAA+gAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAACAAAAAgAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAA",
        "AAEAAAPoAAAgAAABAAAAAAIYbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAABAAAAAUABVxAAAAAAALWhkbHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABWaWRl",
        "b0hhbmRsZXIAAAABw21pbmYAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAYNzdGJsAAAAv3N0c2QA",
        "AAAAAAAAAQAAAK9hdmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAACAAIABIAAAASAAAAAAAAAABFUxhdmM2Mi4yOC4xMDAgbGlieDI2NAAAAAAAAAAAAAAA",
        "GP//AAAANWF2Y0MBZAAK/+EAGGdkAAqs2UlsBEAAAAMAQAAAAwIDxIllgAEABmjr48siwP34+AAAAAAQcGFzcAAAAAEAAAABAAAAFGJ0cnQAAAAAAAAXeAAA",
        "AAAAAAAYc3R0cwAAAAAAAAABAAAABAAAEAAAAAAUc3RzcwAAAAAAAAABAAAAAQAAAChjdHRzAAAAAAAAAAMAAAABAAAgAAAAAAEAAEAAAAAAAgAAEAAAAAAo",
        "c3RzYwAAAAAAAAACAAAAAQAAAAIAAAABAAAAAgAAAAEAAAABAAAAJHN0c3oAAAAAAAAAAAAAAAQAAALKAAAADQAAAAwAAAAMAAAAHHN0Y28AAAAAAAAAAwAA",
        "BoMAAAnJAAANKwAAAt10cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAACAAAAAAAAA+gAAAAAAAAAAAAAAAEBAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAA",
        "AAAAAAAAAABAAAAAAAAAAAAAAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAAPoAAAEAAABAAAAAAJVbWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAAB9AAAA",
        "gQBVxAAAAAAALWhkbHIAAAAAAAAAAHNvdW4AAAAAAAAAAAAAAABTb3VuZEhhbmRsZXIAAAACAG1pbmYAAAAQc21oZAAAAAAAAAAAAAAAJGRpbmYAAAAcZHJl",
        "ZgAAAAAAAAABAAAADHVybCAAAAABAAABxHN0YmwAAAB+c3RzZAAAAAAAAAABAAAAbm1wNGEAAAAAAAAAAQAAAAAAAAAAAAIAEAAAAAB9AAAAAAAANmVzZHMA",
        "AAAAA4CAgCUAAgAEgICAF0AVAAAAAAB9AAAAdm0FgICABRKQVuUABoCAgAECAAAAFGJ0cnQAAAAAAAB9AAAAdm0AAAAgc3R0cwAAAAAAAAACAAAAIAAABAAA",
        "AAABAAABAAAAADRzdHNjAAAAAAAAAAMAAAABAAAAAQAAAAEAAAACAAAACAAAAAEAAAADAAAAGAAAAAEAAACYc3RzegAAAAAAAAAAAAAAIQAAAG8AAACsAAAA",
        "XQAAAF0AAABdAAAAXwAAAGIAAABnAAAAawAAAG8AAAByAAAAdwAAAHcAAABxAAAAjQAAAH8AAAB8AAAAggAAAHoAAAB5AAAAhAAAAIUAAACXAAAAfgAAAHwA",
        "AACCAAAAggAAAIUAAACLAAAAewAAAIkAAACMAAAABwAAABxzdGNvAAAAAAAAAAMAAAlaAAAJ1QAADTcAAAAac2dwZAEAAAByb2xsAAAAAgAAAAH//wAAABxz",
        "YmdwAAAAAHJvbGwAAAABAAAAIQAAAAEAAABidWR0YQAAAFptZXRhAAAAAAAAACFoZGxyAAAAAAAAAABtZGlyYXBwbAAAAAAAAAAAAAAAAC1pbHN0AAAAJal0",
        "b28AAAAdZGF0YQAAAAEAAAAATGF2ZjYyLjEyLjEwMAAAAAhmcmVlAAASPm1kYXQAAAKtBgX//6ncRem95tlIt5Ys2CDZI+7veDI2NCAtIGNvcmUgMTY1IHIz",
        "MjIyIGIzNTYwNWEgLSBILjI2NC9NUEVHLTQgQVZDIGNvZGVjIC0gQ29weWxlZnQgMjAwMy0yMDI1IC0gaHR0cDovL3d3dy52aWRlb2xhbi5vcmcveDI2NC5o",
        "dG1sIC0gb3B0aW9uczogY2FiYWM9MSByZWY9MyBkZWJsb2NrPTE6MDowIGFuYWx5c2U9MHgzOjB4MTEzIG1lPWhleCBzdWJtZT03IHBzeT0xIHBzeV9yZD0x",
        "LjAwOjAuMDAgbWl4ZWRfcmVmPTEgbWVfcmFuZ2U9MTYgY2hyb21hX21lPTEgdHJlbGxpcz0xIDh4OGRjdD0xIGNxbT0wIGRlYWR6b25lPTIxLDExIGZhc3Rf",
        "cHNraXA9MSBjaHJvbWFfcXBfb2Zmc2V0PS0yIHRocmVhZHM9MSBsb29rYWhlYWRfdGhyZWFkcz0xIHNsaWNlZF90aHJlYWRzPTAgbnI9MCBkZWNpbWF0ZT0x",
        "IGludGVybGFjZWQ9MCBibHVyYXlfY29tcGF0PTAgY29uc3RyYWluZWRfaW50cmE9MCBiZnJhbWVzPTMgYl9weXJhbWlkPTIgYl9hZGFwdD0xIGJfYmlhcz0w",
        "IGRpcmVjdD0xIHdlaWdodGI9MSBvcGVuX2dvcD0wIHdlaWdodHA9MiBrZXlpbnQ9MjUwIGtleWludF9taW49NCBzY2VuZWN1dD00MCBpbnRyYV9yZWZyZXNo",
        "PTAgcmNfbG9va2FoZWFkPTQwIHJjPWNyZiBtYnRyZWU9MSBjcmY9MjMuMCBxY29tcD0wLjYwIHFwbWluPTAgcXBtYXg9NjkgcXBzdGVwPTQgaXBfcmF0aW89",
        "MS40MCBhcT0xOjEuMDAAgAAAABVliIQAEv/+963fgU3DKzVrt923T8EAAAAJQZojbEEP/quA3gIATGF2YzYyLjI4LjEwMABCTKnqSkGXhuuK4SWuXIuUChQg",
        "TRkYkZbBaLnt48He6nD/2/gwMDAwMDGzYMDIkQMbNmzYMDIgY2bBgZEilllNm5ZUSKWWWWWWWWeHLLw3XFcJLXLkXKAAAAAAAAAHAAAACEGeQXiCPy+hIUds",
        "/////N2U1ktZTWSz1/jri+Of/7n/X241x7//xf9fPGuN//3P+vxxx5CLvhoqVKlcgnLZBLCYS6Sk4XH9jkcDlyBihNFzURfKiIfLmoEQ+XyMPdiDNPchKVhv",
        "RScDQrUppTISoGCVAwSVU4MDBIS7gwNaYSOL40Y1FFBqKJseB4HgeHr/HXF8c//3P+vtxrj3//i/6+eNcb//uf9fjjjyAAAAAAAAAAAAOCF2lLIsnuTPHz7/",
        "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",
        "L8zjXX/Tn/qHz6/0EAAAAAAAAAAA4CFA2kYIwcA=",
    ].joined()
}
