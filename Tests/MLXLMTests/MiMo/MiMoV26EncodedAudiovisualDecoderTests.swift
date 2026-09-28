import Foundation
import MLXLMCommon
import XCTest
@testable import MLXVLM

/// Prepared/unrun CPU transport tests. No MLX/model/codec-weight work.
final class MiMoV26EncodedAudiovisualDecoderTests: XCTestCase {
    private let videoLimits = MiMoV26EncodedVisualDecoder.Limits(maximumPixels:10000,
        maximumWorkingBytes:16 << 20,maximumSourceFrames:10000,maximumSampledFrames:64)
    private let audioLimits = MiMoV26EncodedAudiovisualDecoder.Limits(maximumFrames:48000,
        maximumWorkingBytes:16 << 20)
    private func inspect(_ data: Data) async throws -> MiMoV26EncodedAudiovisualDecoder.Plan {
        let owner = try MemoryBackedVideoAsset(videoData:data)
        let video = try await MiMoV26EncodedVisualDecoder.inspectVideo(owner,
            sampling:.init(fps:1,minimumFrames:8,maximumFrames:3600),limits:videoLimits)
        XCTAssertTrue(video.hasAudioTrack)
        XCTAssertTrue(video.sourceOwner === owner)
        XCTAssertEqual(video.sampledIndices,[0,2])
        let audio = try await MiMoV26EncodedAudiovisualDecoder.inspect(video,limits:audioLimits)
        XCTAssertTrue(audio.video.sourceOwner === owner)
        return audio
    }
    func testActualPCM16MOVBecomesOneWholeAudioClipWithNativeFrameTimestamps() async throws {
        let plan = try await inspect(MiMoAVFixture.movie())
        XCTAssertEqual(plan.frameCount,24000)
        XCTAssertGreaterThan(plan.audioWorkingByteBound,24000 * 4)
        let result = try await MiMoV26EncodedAudiovisualDecoder.decode(
            plan,videoLimits:videoLimits,audioLimits:audioLimits)
        XCTAssertEqual(result.frames.count,2)
        let silentOwner = try MemoryBackedVideoAsset(videoData:
            XCTUnwrap(Data(base64Encoded:MiMoAVFixture.videoBase64)))
        let silentPlan = try await MiMoV26EncodedVisualDecoder.inspectVideo(silentOwner,
            sampling:.init(fps:1,minimumFrames:8,maximumFrames:3600),limits:videoLimits)
        let silent = try await MiMoV26EncodedVisualDecoder.silentVideo(silentPlan,limits:videoLimits)
        for (actual,original) in zip(result.frames,silent.frames) {
            XCTAssertEqual(actual.width,original.width); XCTAssertEqual(actual.height,original.height)
            XCTAssertEqual(actual.planarRGB,original.planarRGB)
        }
        XCTAssertEqual(result.timestamps.map(\.bitPattern),[Float(0).bitPattern,(Float(2)/Float(3)).bitPattern])
        XCTAssertEqual(result.wholeAudio.descriptor.sampleRate,24000)
        XCTAssertEqual(result.wholeAudio.descriptor.channels,1)
        let expected = (0..<24000).map { Float(Int16(bitPattern:
            [UInt16(0x8000),0xffff,0,1,0x7fff][$0 % 5])) / 32768 }
        XCTAssertEqual(result.wholeAudio.samples.map(\.bitPattern),expected.map(\.bitPattern))
        let fps: Float = 1 / (Float(2) / Float(3))
        let end: Float = Float(2) / Float(3) + 1 / fps
        XCTAssertEqual(result.segmentEnd,.float32(end))
        let layout = try MiMoV26AudiovisualLayout.make(timestamps:result.timestamps,
            segmentEnd:result.segmentEnd,temporalPatchSize:2,wholeAudioPatches:7,maximumUnits:1)
        XCTAssertEqual(layout.units.map(\.audioRange),[0..<7])
        XCTAssertEqual(layout.usedAudioPatches,7)
        // Existing silent entrypoint MUST NOT become a sound-dropping bypass.
        do { _ = try await MiMoV26EncodedVisualDecoder.silentVideo(plan.video,limits:videoLimits)
            XCTFail("audio silently dropped") }
        catch { XCTAssertEqual(error as? MiMoV26EncodedVisualDecoder.Failure,
                               .audioTrackRequiresAudiovisualProfile) }
    }
    func testActualFloat32LPCMPreservesSignedZeroSubnormalsAndFractionalBits() async throws {
        let bits: [UInt32] = [0x80000000,1,0x3f800001,0xbfa00000,0x3eaaaaab]
        let plan = try await inspect(MiMoAVFixture.movie(floatBits:bits))
        let value = try await MiMoV26EncodedAudiovisualDecoder.decode(
            plan,videoLimits:videoLimits,audioLimits:audioLimits)
        XCTAssertEqual(value.wholeAudio.samples.map(\.bitPattern),(0..<24000).map { bits[$0 % bits.count] })
    }
    func testActualNonfiniteTrackFailsWithoutNormalization() async throws {
        let plan = try await inspect(MiMoAVFixture.movie(floatBits:[0x7fc00001]))
        do { _ = try await MiMoV26EncodedAudiovisualDecoder.decode(
            plan,videoLimits:videoLimits,audioLimits:audioLimits); XCTFail("NaN accepted") }
        catch { XCTAssertEqual(error as? MiMoV26EncodedAudiovisualDecoder.Failure,.nonfiniteSamples) }
    }
    func testActualRateChannelsMultipleTracksAndEditsRemainTypedRefusals() async throws {
        let cases: [(Data,MiMoV26EncodedAudiovisualDecoder.Failure)] = [
            (try MiMoAVFixture.movie(rate:22050),.unsupportedSampleRate),
            (try MiMoAVFixture.movie(channels:2),.unsupportedChannels),
            (try MiMoAVFixture.movie(edited:true),.unsupportedTimeline),
            (try MiMoAVFixture.movie(duplicateTrack:true),.unsupportedEncoding)]
        for (data,expected) in cases {
            do { _ = try await inspect(data); XCTFail("unsupported transport accepted") }
            catch { XCTAssertEqual(error as? MiMoV26EncodedAudiovisualDecoder.Failure,expected) }
        }
    }
    func testRealCompressedTrackMetadataIsNotARequestForImplicitDecode() async throws {
        // The u-law sample entry is an actual unsupported audio codec declaration,
        // not a mock Plan. Payload is deliberately not decoded on this route.
        do { _ = try await inspect(MiMoAVFixture.movie(codec:"ulaw")); XCTFail("codec accepted") }
        catch { XCTAssertEqual(error as? MiMoV26EncodedAudiovisualDecoder.Failure,.unsupportedEncoding) }
    }
    func testBoundAndCancellationBeforePCMExpansion() async throws {
        let plan = try await inspect(MiMoAVFixture.movie())
        let small = MiMoV26EncodedAudiovisualDecoder.Limits(maximumFrames:23999,maximumWorkingBytes:16 << 20)
        do { _ = try await MiMoV26EncodedAudiovisualDecoder.inspect(plan.video,limits:small); XCTFail("frame cap") }
        catch { XCTAssertEqual(error as? MiMoV26EncodedAudiovisualDecoder.Failure,.limit) }
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await MiMoV26EncodedAudiovisualDecoder.decode(
                plan,videoLimits:videoLimits,audioLimits:audioLimits)
        }
        do { _ = try await task.value; XCTFail("cancelled transport returned content") }
        catch { XCTAssertTrue(error is CancellationError) }
    }
    func testIndividualEndKeepsFloat32ReciprocalOrderNotContainerDuration() throws {
        for gap: Float in [0.16,0.33333334,0.7,1.234567] {
            let t: [Float] = [0,gap,17.3]
            let fps: Float = 1 / gap
            let step: Float = 1 / fps
            XCTAssertEqual(try MiMoV26EncodedAudiovisualDecoder.individualSegmentEnd(t).bitPattern,
                (t[2] + step).bitPattern)
        }
        XCTAssertThrowsError(try MiMoV26EncodedAudiovisualDecoder.individualSegmentEnd([0,0]))
    }

    /// Separate regression for the root-owned encoded-byte pre-audio guard.
    /// Uses the actual PCM MOV plan, never a fake decoder or completion flag.
    func testEncodedSourceCeilingPrecedesCancellationAndAcceptsExactBytes() async throws {
        let data = try MiMoAVFixture.movie()
        let plan = try await inspect(data)
        XCTAssertEqual(plan.video.sourceOwner.byteCount, data.count)
        let audio = audioLimits
        let tight = MiMoV26EncodedVisualDecoder.Limits(maximumPixels:10000,
            maximumWorkingBytes:16 << 20,maximumSourceFrames:10000,maximumSampledFrames:64,
            maximumEncodedBytes:data.count - 1)
        do {
            _ = try await MiMoV26EncodedAudiovisualDecoder.decode(plan,videoLimits:tight,audioLimits:audio)
            XCTFail("encoded source exceeded the reused plan's current ceiling")
        } catch {
            XCTAssertEqual(error as? MiMoV26EncodedAudiovisualDecoder.Failure,.limit)
        }
        let cancelled = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            XCTAssertTrue(Task.isCancelled)
            return try await MiMoV26EncodedAudiovisualDecoder.decode(plan,videoLimits:tight,audioLimits:audio)
        }
        do {
            _ = try await cancelled.value
            XCTFail("encoded ceiling did not refuse cancelled AV transport")
        } catch {
            // Removing the pre-audio byte guard exposes CancellationError here.
            XCTAssertEqual(error as? MiMoV26EncodedAudiovisualDecoder.Failure,.limit)
        }
        let exact = MiMoV26EncodedVisualDecoder.Limits(maximumPixels:10000,
            maximumWorkingBytes:16 << 20,maximumSourceFrames:10000,maximumSampledFrames:64,
            maximumEncodedBytes:data.count)
        let decoded = try await MiMoV26EncodedAudiovisualDecoder.decode(plan,videoLimits:exact,audioLimits:audio)
        XCTAssertEqual(decoded.frames.count,2)
        XCTAssertEqual(decoded.timestamps.map(\.bitPattern),[Float(0).bitPattern,(Float(2)/Float(3)).bitPattern])
        XCTAssertEqual(decoded.wholeAudio.descriptor.sampleRate,24000)
        XCTAssertEqual(decoded.wholeAudio.descriptor.channels,1)
        let words: [UInt16] = [0x8000,0xffff,0,1,0x7fff]
        let expected: [Float] = (0..<24000).map { Float(Int16(bitPattern:words[$0 % 5])) / 32768 }
        XCTAssertEqual(decoded.wholeAudio.samples.map(\.bitPattern),expected.map(\.bitPattern))
    }
}

/// A real ISO-BMFF fixture: retain the pre-existing three-frame H.264 mdat and
/// video track byte-for-byte, append one bounded uncompressed audio mdat/track.
/// Pure Swift byte construction only; AVAssetReader remains the actual decoder.
/// The two test targets deliberately carry identical private fixture builders.
private enum MiMoAVFixture {
    static let videoBase64 = "AAAAHGZ0eXBtcDQyAAAAAWlzb21tcDQxbXA0MgAAAAFtZGF0AAAAAAAAAK4AAAA7BgUyR1ZK3FxMQz+U78URPNFDqAEAAAMAAQMAAAMAAQIAAeYACwAAAwAAAwAAAwAUDAOJJAEN/////4AAAAAxJbggH4AuSqwRNmYXSACJwyG5akafRwrPDoFqVCtjHBP+QvRWhyAAGk1PzfAEsEedgAAAABEh4QhfAoAvQrFXFN4ACQ7CtgAAABEBqIGK/1jQw/VufW+ACvdnuAAAAvFtb292AAAAbG12aGQAAAAA5lOws+ZTsLMAAAJYAAACWAABAAABAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAACfXRyYWsAAABcdGtoZAAAAAHmU7Cz5lOwswAAAAEAAAAAAAACWAAAAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAQAAAAEAAAAAAACRlZHRzAAAAHGVsc3QAAAAAAAAAAQAAAlgAAADIAAEAAAAAAfVtZGlhAAAAIG1kaGQAAAAA5lOws+ZTsLMAAAJYAAACWFXEAAAAAAAxaGRscgAAAAAAAAAAdmlkZQAAAAAAAAAAAAAAAENvcmUgTWVkaWEgVmlkZW8AAAABnG1pbmYAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAVxzdGJsAAAAoXN0c2QAAAAAAAAAAQAAAJFhdmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAEAAQABIAAAASAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGP//AAAAJ2F2Y0MBZAAL/+EADCdkAAusVlDDeBBhFAEABCjuPLD9+PgAAAAACmZpZWwBAAAAAApjaHJtAAAAAAAYc3R0cwAAAAAAAAABAAAAAwAAAMgAAAAoY3R0cwAAAAAAAAADAAAAAQAAAMgAAAABAAABkAAAAAEAAAAAAAAAFHN0c3MAAAAAAAAAAQAAAAEAAAAPc2R0cAAAAAAgEBgAAAAcc3RzYwAAAAAAAAABAAAAAQAAAAMAAAABAAAAIHN0c3oAAAAAAAAAAAAAAAMAAAB0AAAAFQAAABUAAAAUc3RjbwAAAAAAAAABAAAALA=="
    static func be(_ value: UInt64, _ width: Int = 4) -> Data {
        Data((0..<width).reversed().map { UInt8(truncatingIfNeeded: value >> ($0 * 8)) })
    }
    static func atom(_ name: String, _ body: Data) -> Data {
        be(UInt64(body.count + 8)) + Data(name.utf8) + body
    }
    static let matrix = [UInt64(0x10000),0,0,0,0x10000,0,0,0,0x40000000]
        .reduce(into: Data()) { $0.append(be($1)) }
    static let frames = 24000
    static func movie(floatBits: [UInt32]? = nil, rate: Int = 24000, channels: Int = 1,
                      codec: String? = nil, edited: Bool = false, duplicateTrack: Bool = false) throws -> Data {
        let original = try XCTUnwrap(Data(base64Encoded: videoBase64))
        let step = floatBits == nil ? 2 : 4
        var pcm = Data()
        for i in 0..<frames {
            let bits = floatBits.map { $0[i % $0.count] }
                ?? UInt32([UInt16(0x8000),0xffff,0,1,0x7fff][i % 5])
            for _ in 0..<channels {
                for byte in 0..<step { pcm.append(UInt8(truncatingIfNeeded: bits >> (byte * 8))) }
            }
        }
        func track(offset: Int, id: Int = 2) -> Data {
            let duration = UInt64(frames * 600 / rate)
            let tkhdFields: [Data] = [be(7), be(0), be(0), be(UInt64(id)), be(0), be(duration),
                Data(repeating: 0, count: 12), be(0x100,2), be(0,2), matrix, be(0), be(0)]
            let tkhd = atom("tkhd", tkhdFields.reduce(into: Data()) { $0.append($1) })
            let mdhd = atom("mdhd", Data(repeating: 0, count: 12) + be(UInt64(rate)) + be(UInt64(frames))
                + be(0x55c4,2) + be(0,2))
            let hdlr = atom("hdlr", Data(repeating: 0, count: 8) + Data("soun".utf8)
                + Data(repeating: 0, count: 12) + Data("Sound\0".utf8))
            let sample: Data
            if floatBits != nil {
                // QuickTime SoundDescription V2, LPCM float32 little-endian,
                // packed, 1 PCM frame/packet. No platform conversion request.
                let common = Data(repeating: 0, count: 6) + be(1,2) + be(2,2) + be(0,2) + be(0)
                    + be(3,2) + be(16,2) + be(0xfffe,2) + be(0,2) + be(0x10000)
                let extended = be(72) + be(Double(rate).bitPattern,8) + be(UInt64(channels))
                    + be(0x7f000000) + be(32) + be(9) + be(UInt64(step * channels)) + be(1)
                sample = atom(codec ?? "lpcm", common + extended)
            } else {
                let body = Data(repeating: 0, count: 6) + be(1,2) + Data(repeating: 0, count: 8)
                    + be(UInt64(channels),2) + be(16,2) + be(0,2) + be(0,2) + be(UInt64(rate) << 16)
                sample = atom(codec ?? "sowt", body)
            }
            let stsd = atom("stsd", be(0) + be(1) + sample)
            let stts = atom("stts", be(0) + be(1) + be(UInt64(frames)) + be(1))
            let stsc = atom("stsc", be(0) + be(1) + be(1) + be(UInt64(frames)) + be(1))
            let stsz = atom("stsz", be(0) + be(UInt64(step * channels)) + be(UInt64(frames)))
            let stco = atom("stco", be(0) + be(1) + be(UInt64(offset)))
            let stbl = atom("stbl", stsd + stts + stsc + stsz + stco)
            let dinf = atom("dinf", atom("dref", be(0) + be(1) + atom("url ", be(1))))
            let minf = atom("minf", atom("smhd", Data(repeating: 0, count: 8)) + dinf + stbl)
            let edit = edited ? atom("edts", atom("elst",
                be(0) + be(1) + be(duration) + be(1) + be(0x10000))) : Data()
            return atom("trak", tkhd + edit + atom("mdia", mdhd + hdlr + minf))
        }
        // This bound fixture's last atom is moov; existing stco offsets remain
        // unchanged because only the tail moov grows, then a new mdat is added.
        var moovStart = 0
        while moovStart < original.count {
            let n = original[moovStart..<moovStart+4].reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
            let type = String(data: original[moovStart+4..<moovStart+8], encoding: .ascii)
            if type == "moov" { break }
            let actual = n == 1 ? original[moovStart+8..<moovStart+16]
                .reduce(UInt64(0)) { ($0 << 8) | UInt64($1) } : n
            guard actual >= 8, actual <= UInt64(original.count - moovStart) else {
                throw NSError(domain: "MiMoAVFixture", code: 1)
            }
            moovStart += Int(actual)
        }
        let addition = track(offset: 0).count * (duplicateTrack ? 2 : 1)
        let offset = original.count + addition + 8
        var body = Data(original[(moovStart + 8)...])
        // mvhd's final next_track_ID; movie/video payload otherwise unchanged.
        let mvhdSize = body.prefix(4).reduce(0) { ($0 << 8) | Int($1) }
        body.replaceSubrange((mvhdSize - 4)..<mvhdSize, with: be(duplicateTrack ? 4 : 3))
        body += track(offset: offset)
        if duplicateTrack { body += track(offset: offset, id: 3) }
        var prefix = Data(original.prefix(moovStart))
        // LPCM sound descriptions use the QuickTime container contract.
        // Retain byte offsets/lengths; video track and video mdat are unchanged.
        prefix.replaceSubrange(8..<12,with:Data("qt  ".utf8))
        prefix.replaceSubrange(16..<28,with:Data("qt  qt  qt  ".utf8))
        return prefix + atom("moov", body) + atom("mdat", pcm)
    }
}
