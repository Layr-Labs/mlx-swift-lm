// Copyright © 2026 Eigen Labs.
// Bounded transport decode only: mono24k WAV PCM16/IEEE Float32. No MLX,
// resampling, channel mixing, amplitude normalization, URL or file access.
import CryptoKit
import Foundation
import MLXLLM

public enum MiMoV26EncodedAudioDecoder {
    public enum Failure: Error, Equatable, Sendable {
        case invalidWave, unsupportedEncoding, unsupportedChannels, unsupportedSampleRate
        case nonfiniteSamples, limit, arithmeticOverflow
    }
    public struct Limits: Sendable {
        public let maximumEncodedBytes, maximumFrames, maximumWorkingBytes, maximumChunks: Int
        public init(
            maximumEncodedBytes: Int, maximumFrames: Int, maximumWorkingBytes: Int,
            maximumChunks: Int = 4096
        ) {
            self.maximumEncodedBytes = maximumEncodedBytes
            self.maximumFrames = maximumFrames
            self.maximumWorkingBytes = maximumWorkingBytes
            self.maximumChunks = maximumChunks
        }
    }
    public struct Plan: Sendable {
        public enum Encoding: Sendable, Equatable { case pcm16, float32 }
        public let encoding: Encoding
        public let frameCount: Int
        public let sampleRate = 24000
        public let encodedByteCount, decodedByteBound: Int
        public let sourceIdentity: String
        fileprivate let data: Data
        fileprivate let sampleOffset, sampleBytes: Int
        fileprivate init(
            data: Data, encoding: Encoding, frames: Int, offset: Int,
            bytes: Int, decodedBytes: Int
        ) {
            self.data = data
            self.encoding = encoding
            frameCount = frames
            sampleOffset = offset
            sampleBytes = bytes
            encodedByteCount = data.count
            // Input-byte provenance, NOT model/codec-weight authentication.
            sourceIdentity = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            decodedByteBound = decodedBytes
        }
    }
    private static func add(_ a: Int, _ b: Int) throws -> Int {
        let (n, overflow) = a.addingReportingOverflow(b)
        guard a >= 0, b >= 0, !overflow else { throw Failure.arithmeticOverflow }
        return n
    }
    private static func multiply(_ a: Int, _ b: Int) throws -> Int {
        let (n, overflow) = a.multipliedReportingOverflow(by: b)
        guard a >= 0, b >= 0, !overflow else { throw Failure.arithmeticOverflow }
        return n
    }
    private static func u16(_ bytes: UnsafeRawBufferPointer, _ at: Int) throws -> UInt16 {
        guard at >= 0, at <= bytes.count - 2 else { throw Failure.invalidWave }
        return UInt16(bytes[at]) | (UInt16(bytes[at + 1]) << 8)
    }
    private static func u32(_ bytes: UnsafeRawBufferPointer, _ at: Int) throws -> UInt32 {
        guard at >= 0, at <= bytes.count - 4 else { throw Failure.invalidWave }
        return UInt32(bytes[at]) | (UInt32(bytes[at + 1]) << 8)
            | (UInt32(bytes[at + 2]) << 16) | (UInt32(bytes[at + 3]) << 24)
    }

    /// Header scan only; no Float buffer is allocated. Host reserves the
    /// aggregate decodedByteBound before decode(), not after expansion.
    public static func inspect(_ data: Data, limits: Limits) throws -> Plan {
        try Task.checkCancellation()
        guard limits.maximumEncodedBytes > 0, limits.maximumFrames > 0,
            limits.maximumWorkingBytes > 0, limits.maximumChunks > 0,
            data.count <= limits.maximumEncodedBytes
        else { throw Failure.limit }
        return try data.withUnsafeBytes { bytes in
            guard bytes.count >= 12, try u32(bytes, 0) == 0x4646_4952,
                try u32(bytes, 8) == 0x4556_4157
            else { throw Failure.invalidWave }  // RIFF/WAVE
            let end = try add(Int(u32(bytes, 4)), 8)
            guard end == bytes.count else { throw Failure.invalidWave }
            var cursor = 12
            var chunks = 0
            var format: (code: UInt16, bits: UInt16, block: Int)?
            var samples: (offset: Int, bytes: Int)?
            var factFrames: Int?
            while cursor < end {
                try Task.checkCancellation()
                chunks += 1
                guard chunks <= limits.maximumChunks else { throw Failure.limit }
                guard cursor <= end - 8 else { throw Failure.invalidWave }
                let kind = try u32(bytes, cursor)
                let size = try Int(u32(bytes, cursor + 4))
                let start = try add(cursor, 8)
                let stop = try add(start, size)
                guard stop <= end else { throw Failure.invalidWave }
                switch kind {
                case 0x2074_6d66:  // fmt
                    guard format == nil, size == 16 || size == 18 else { throw Failure.invalidWave }
                    let code = try u16(bytes, start)
                    let channels = try u16(bytes, start + 2)
                    let rate = try u32(bytes, start + 4)
                    let byteRate = try u32(bytes, start + 8)
                    let block = try Int(u16(bytes, start + 12))
                    let bits = try u16(bytes, start + 14)
                    guard channels == 1 else { throw Failure.unsupportedChannels }
                    guard rate == 24000 else { throw Failure.unsupportedSampleRate }
                    guard (code == 1 && bits == 16) || (code == 3 && bits == 32) else {
                        throw Failure.unsupportedEncoding
                    }
                    if size == 18 {
                        guard try u16(bytes, start + 16) == 0 else {
                            throw Failure.unsupportedEncoding
                        }
                    }
                    guard block == Int(bits / 8), byteRate == UInt32(24000 * block) else {
                        throw Failure.invalidWave
                    }
                    format = (code, bits, block)
                case 0x6174_6164:  // data
                    guard samples == nil, size > 0 else { throw Failure.invalidWave }
                    samples = (start, size)
                case 0x7463_6166:  // optional fact count; do not accept contradictory duration
                    guard factFrames == nil, size >= 4 else { throw Failure.invalidWave }
                    factFrames = try Int(u32(bytes, start))
                default: break  // bounded unknown metadata; no recursive container/URL interpretation
                }
                cursor = try add(stop, size & 1)
                guard cursor <= end else { throw Failure.invalidWave }  // mandatory odd-chunk pad byte
            }
            guard cursor == end, let format, let samples,
                samples.bytes % format.block == 0
            else { throw Failure.invalidWave }
            let frames = samples.bytes / format.block
            guard frames > 0, frames <= limits.maximumFrames else { throw Failure.limit }
            guard factFrames == nil || factFrames == frames else { throw Failure.invalidWave }
            // Additional Float storage plus fixed header/descriptor slack.
            // The encoded owner is independently priced by the host promise.
            let decodedBytes = try add(multiply(frames, MemoryLayout<Float>.stride), 1024)
            guard decodedBytes <= limits.maximumWorkingBytes else { throw Failure.limit }
            return Plan(
                data: data, encoding: format.code == 1 ? .pcm16 : .float32,
                frames: frames, offset: samples.offset, bytes: samples.bytes,
                decodedBytes: decodedBytes)
        }
    }

    /// Preserves signed zero/Float32 values and exact PCM16 /32768 conversion.
    /// No clipping, peak normalization, channel mixing or sample-rate conversion.
    public static func decode(_ plan: Plan) throws -> MiMoV26DecodedPCM {
        try Task.checkCancellation()
        let values = try plan.data.withUnsafeBytes { bytes -> [Float] in
            var values: [Float] = []
            values.reserveCapacity(plan.frameCount)
            let step: Int = plan.encoding == .pcm16 ? 2 : 4
            for index in 0 ..< plan.frameCount {
                if index.isMultiple(of: 1024) { try Task.checkCancellation() }
                let at = plan.sampleOffset + index * step
                let value: Float
                switch plan.encoding {
                case .pcm16: value = Float(Int16(bitPattern: try u16(bytes, at))) / 32768
                case .float32: value = Float(bitPattern: try u32(bytes, at))
                }
                guard value.isFinite else { throw Failure.nonfiniteSamples }
                values.append(value)
            }
            return values
        }
        return try MiMoV26DecodedPCM(
            samples: values,
            descriptor: .init(
                sourceIdentity: plan.sourceIdentity, channels: 1,
                frameCount: plan.frameCount, sampleRate: plan.sampleRate))
    }
}
