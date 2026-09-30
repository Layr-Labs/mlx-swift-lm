import Foundation
import MLXLLM
import XCTest

@testable import MLXVLM

/// Pure byte/scalar tests; no model, decoder backend, resampler or native eval.
final class MiMoV26EncodedAudioDecoderTests: XCTestCase {
    private func u16(_ n: UInt16) -> [UInt8] {
        [UInt8(truncatingIfNeeded: n), UInt8(truncatingIfNeeded: n >> 8)]
    }
    private func u32(_ n: UInt32) -> [UInt8] {
        [
            UInt8(truncatingIfNeeded: n), UInt8(truncatingIfNeeded: n >> 8),
            UInt8(truncatingIfNeeded: n >> 16), UInt8(truncatingIfNeeded: n >> 24),
        ]
    }
    private func chunk(_ name: String, _ body: [UInt8]) -> [UInt8] {
        Array(name.utf8) + u32(UInt32(body.count)) + body + (body.count % 2 == 1 ? [0] : [])
    }
    private func wave(
        _ bytes: [UInt8], code: UInt16 = 1, bits: UInt16 = 16,
        channels: UInt16 = 1, rate: UInt32 = 24000,
        leading: [UInt8] = [], trailing: [UInt8] = []
    ) -> Data {
        let block = channels * (bits / 8)
        let format =
            u16(code) + u16(channels) + u32(rate) + u32(rate * UInt32(block)) + u16(block)
            + u16(bits)
        let body =
            Array("WAVE".utf8) + leading + chunk("fmt ", format) + chunk("data", bytes) + trailing
        return Data(Array("RIFF".utf8) + u32(UInt32(body.count)) + body)
    }
    private let limits = MiMoV26EncodedAudioDecoder.Limits(
        maximumEncodedBytes: 65536,
        maximumFrames: 8192, maximumWorkingBytes: 65536)
    func testPCM16BytesProduceExactNativeScalarsAndFrameCount() throws {
        let bits: [UInt16] = [0x8000, 0xffff, 0, 1, 0x7fff]
        let bytes = wave(bits.flatMap(u16))
        let plan = try MiMoV26EncodedAudioDecoder.inspect(bytes, limits: limits)
        let pcm = try MiMoV26EncodedAudioDecoder.decode(plan)
        let expectedBits: [UInt32] = [0xbf80_0000, 0xb800_0000, 0, 0x3800_0000, 0x3f7f_fe00]
        XCTAssertEqual(plan.encoding, .pcm16)
        XCTAssertEqual(plan.frameCount, 5)
        XCTAssertEqual(plan.encodedByteCount, bytes.count)
        XCTAssertEqual(pcm.descriptor.frameCount, 5)
        XCTAssertEqual(pcm.descriptor.channels, 1)
        XCTAssertEqual(pcm.descriptor.sampleRate, 24000)
        XCTAssertEqual(pcm.samples.map(\.bitPattern), expectedBits)
        XCTAssertEqual(pcm.descriptor.sourceIdentity, plan.sourceIdentity)
        XCTAssertEqual(plan.decodedByteBound, 5 * 4 + 1024)
    }
    func testOpenRouterPCM8At22050PreservesOriginalRateForNativeResampling() throws {
        let bytes = wave(
            [0, 64, 128, 192, 255] + Array(repeating: 128, count: 47043), bits: 8, rate: 22050)
        let large = MiMoV26EncodedAudioDecoder.Limits(
            maximumEncodedBytes: 65536,
            maximumFrames: 65536, maximumWorkingBytes: 256 << 10)
        let plan = try MiMoV26EncodedAudioDecoder.inspect(bytes, limits: large)
        let pcm = try MiMoV26EncodedAudioDecoder.decode(plan)
        XCTAssertEqual(plan.encoding, .pcm8)
        XCTAssertEqual(plan.frameCount, 47048)
        XCTAssertEqual(plan.sampleCount, 47048)
        XCTAssertEqual(plan.decodedByteBound, 47048 * 4 + 1024)
        XCTAssertEqual(pcm.descriptor.sampleRate, 22050)
        XCTAssertEqual(Array(pcm.samples.prefix(5)), [-1, -0.5, 0, 0.5, 127.0 / 128])
        let resample = try MiMoV26AudioResamplePlan.make(
            originalRate: 22050, frames: 47048,
            maximumCoefficients: 1_000_000)
        XCTAssertEqual(resample.outputFrames, (47048 * 24000 + 22049) / 22050)
        XCTAssertFalse(resample.isIdentity)
    }

    func testStereoPCMDeinterleavesBeforeNativeResamplingAndMixing() throws {
        let bytes = wave(
            [UInt16(0x8000), 0x4000, 0x2000, 0xc000].flatMap(u16), channels: 2, rate: 44100)
        let plan = try MiMoV26EncodedAudioDecoder.inspect(bytes, limits: limits)
        let pcm = try MiMoV26EncodedAudioDecoder.decode(plan)
        XCTAssertEqual(plan.frameCount, 2)
        XCTAssertEqual(plan.sampleCount, 4)
        XCTAssertEqual(plan.decodedByteBound, 4 * 4 + 1024)
        XCTAssertEqual(pcm.descriptor.channels, 2)
        XCTAssertEqual(pcm.descriptor.sampleRate, 44100)
        XCTAssertEqual(pcm.samples, [-1, 0.25, 0.5, -0.5])
        XCTAssertThrowsError(
            try MiMoV26EncodedAudioDecoder.inspect(
                bytes,
                limits: .init(
                    maximumEncodedBytes: 65536, maximumFrames: 8192, maximumWorkingBytes: 65536,
                    maximumChannels: 1)))
        XCTAssertThrowsError(
            try MiMoV26EncodedAudioDecoder.inspect(
                bytes,
                limits: .init(
                    maximumEncodedBytes: 65536, maximumFrames: 8192, maximumWorkingBytes: 65536,
                    maximumSampleRate: 24000)))
    }

    func testPCM24AndPCM32SignAndAmplitudeConversion() throws {
        let pcm24 = try MiMoV26EncodedAudioDecoder.decode(
            MiMoV26EncodedAudioDecoder.inspect(
                wave(
                    [0, 0, 128, 255, 255, 255, 0, 0, 0, 0, 0, 64, 255, 255, 127], bits: 24,
                    rate: 48000), limits: limits))
        XCTAssertEqual(pcm24.samples, [-1, -1.0 / 8_388_608, 0, 0.5, 8388607.0 / 8_388_608])
        let bits: [UInt32] = [0x8000_0000, 0xffff_ffff, 0, 0x4000_0000, 0x7fff_ffff]
        let pcm32 = try MiMoV26EncodedAudioDecoder.decode(
            MiMoV26EncodedAudioDecoder.inspect(
                wave(bits.flatMap(u32), bits: 32, rate: 96000), limits: limits))
        XCTAssertEqual(pcm32.samples, [-1, -1.0 / 2_147_483_648, 0, 0.5, 1])
    }

    func testFloat32PreservesBitsSignedZeroAndFiniteOutOfUnitRangeWithoutNormalization() throws {
        let values: [Float] = [
            Float(bitPattern: 0x8000_0000), 0, 1.25, -2, Float.leastNormalMagnitude,
        ]
        let bytes = wave(values.flatMap { u32($0.bitPattern) }, code: 3, bits: 32)
        let pcm = try MiMoV26EncodedAudioDecoder.decode(
            MiMoV26EncodedAudioDecoder.inspect(bytes, limits: limits))
        XCTAssertEqual(pcm.samples.map(\.bitPattern), values.map(\.bitPattern))
        XCTAssertEqual(pcm.descriptor.frameCount, values.count)
    }
    func testOddMetadataPaddingDataSliceAndFactCountDoNotChangeSamples() throws {
        let bytes = wave(
            u16(0x1234) + u16(0x8000),
            leading: chunk("JUNK", [1, 2, 3]), trailing: chunk("fact", u32(2)))
        var backing = Data([9, 8, 7])
        backing.append(bytes)
        let sliced = backing[backing.index(backing.startIndex, offsetBy: 3)...]
        let a = try MiMoV26EncodedAudioDecoder.decode(
            MiMoV26EncodedAudioDecoder.inspect(bytes, limits: limits))
        let b = try MiMoV26EncodedAudioDecoder.decode(
            MiMoV26EncodedAudioDecoder.inspect(sliced, limits: limits))
        XCTAssertEqual(a.samples.map(\.bitPattern), b.samples.map(\.bitPattern))
        XCTAssertEqual(a.descriptor.sourceIdentity, b.descriptor.sourceIdentity)
        XCTAssertThrowsError(
            try MiMoV26EncodedAudioDecoder.inspect(
                wave(u16(1), trailing: chunk("fact", u32(2))), limits: limits))
    }
    func testMalformedTruncatedDuplicateAndUnalignedDataRefuseBeforePCMReturn() throws {
        let good = wave(u16(1))
        var badLength = good
        badLength[4] = 0xff
        var badBlock = good
        badBlock[32] = 0
        var malformed = [
            Data(), Data("RIFF".utf8), Data(good.dropLast()), badLength, badBlock,
            wave([1]), wave([], code: 1), wave(u16(1), trailing: chunk("data", u16(2))),
        ]
        var wrongEndian = good
        wrongEndian[3] = UInt8(ascii: "X")
        malformed.append(wrongEndian)
        for data in malformed {
            XCTAssertThrowsError(try MiMoV26EncodedAudioDecoder.inspect(data, limits: limits))
        }
    }
    func testChannelsRateEncodingAndNonfiniteValuesRemainExplicitRefusals() throws {
        for data in [
            wave(u16(1) + u16(2) + u16(3), channels: 3), wave(u16(1), rate: 7999),
            wave(u16(1), rate: 192001), wave(u16(1), code: 6),
        ] {
            XCTAssertThrowsError(try MiMoV26EncodedAudioDecoder.inspect(data, limits: limits))
        }
        for value in [Float.infinity, Float.nan] {
            let plan = try MiMoV26EncodedAudioDecoder.inspect(
                wave(u32(value.bitPattern), code: 3, bits: 32), limits: limits)
            XCTAssertThrowsError(try MiMoV26EncodedAudioDecoder.decode(plan)) {
                XCTAssertEqual($0 as? MiMoV26EncodedAudioDecoder.Failure, .nonfiniteSamples)
            }
        }
    }
    func testExactFrameWorkingByteAndChunkCeilingsRejectBeforeAllocation() throws {
        let data = wave(u16(1) + u16(2))
        for cap in [
            MiMoV26EncodedAudioDecoder.Limits(
                maximumEncodedBytes: data.count - 1, maximumFrames: 2, maximumWorkingBytes: 65536),
            .init(maximumEncodedBytes: 65536, maximumFrames: 1, maximumWorkingBytes: 65536),
            .init(maximumEncodedBytes: 65536, maximumFrames: 2, maximumWorkingBytes: 1031),
            .init(
                maximumEncodedBytes: 65536, maximumFrames: 2, maximumWorkingBytes: 65536,
                maximumChunks: 1),
        ] {
            XCTAssertThrowsError(try MiMoV26EncodedAudioDecoder.inspect(data, limits: cap)) {
                XCTAssertEqual($0 as? MiMoV26EncodedAudioDecoder.Failure, .limit)
            }
        }
    }
}
