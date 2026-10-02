import Foundation
import MLX
import XCTest

@testable import MLXLLM
@testable import MLXVLM

final class MiMoV26AudioFrontendTests: XCTestCase {
    func testReflectIndicesAndPeriodicWindowAreNotEdgeOrSymmetricPadding() throws {
        let indices = try MiMoV26AudioFrontend.reflectFrameIndices(
            samples: 481, startFrame: 0, count: 1, fftSize: 960, hop: 240)
        XCTAssertEqual(Array(indices.prefix(4)), [480, 479, 478, 477])
        XCTAssertEqual(indices[479], 1)
        XCTAssertEqual(indices[480], 0)
        XCTAssertEqual(indices[481], 1)
        let end = try MiMoV26AudioFrontend.reflectFrameIndices(
            samples: 481, startFrame: 2, count: 1, fftSize: 960, hop: 240)
        XCTAssertEqual(end[480], 480)
        XCTAssertEqual(end[481], 479)
        XCTAssertEqual(end.last, 1)
        try mimoAudioInputRequireNative()
        let window = MiMoV26AudioFrontend.periodicHann(960).asArray(Float.self)
        XCTAssertEqual(window[0], 0)
        XCTAssertEqual(window[480], 1, accuracy: 0.000001)
        XCTAssertGreaterThan(window[959], 0)
        XCTAssertEqual(window[1], window[959], accuracy: 0.000001)
    }

    func testPCMValidationAndCancelledPreparationBeforeGraph() throws {
        let d = MiMoV26AudioPCMDescriptor(
            sourceIdentity: "pcm", channels: 1, frameCount: 481, sampleRate: 24000)
        XCTAssertThrowsError(try MiMoV26DecodedPCM(samples: [0], descriptor: d))
        var samples = Array(repeating: Float(0), count: 481)
        samples[0] = .nan
        XCTAssertThrowsError(try MiMoV26DecodedPCM(samples: samples, descriptor: d))
        let c = try MiMoV26AudioInputConfiguration.fixture(melBands: 128)
        let plan = try MiMoV26AudioInputPlan.make(
            clips: [d], configuration: c, limits: mimoAudioInputTestLimits())
        let pcm = try MiMoV26DecodedPCM(samples: Array(repeating: 0, count: 481), descriptor: d)
        XCTAssertThrowsError(
            try MiMoV26AudioFrontend(configuration: c).prepare(
                pcm: pcm, plan: plan, clipIndex: 0, isCancelled: { true })
        ) {
            XCTAssertEqual($0 as? MiMoV26AudioInputError, .cancelled)
        }
    }

    func testIdentitySilenceStereoCancellationAndAmplitudeAreNotNormalized() throws {
        try mimoAudioInputRequireNative()
        let c = try MiMoV26AudioInputConfiguration.fixture(melBands: 128)
        let frontend = MiMoV26AudioFrontend(configuration: c)
        func prepare(_ samples: [Float], channels: Int) throws -> [Float] {
            let d = MiMoV26AudioPCMDescriptor(
                sourceIdentity: "synthetic", channels: channels,
                frameCount: samples.count / channels, sampleRate: 24000)
            let p = try MiMoV26AudioInputPlan.make(
                clips: [d], configuration: c, limits: mimoAudioInputTestLimits())
            return try frontend.prepare(
                pcm: .init(samples: samples, descriptor: d), plan: p, clipIndex: 0
            ).values.asArray(Float.self)
        }
        let zeros = try prepare(Array(repeating: 0, count: 2400), channels: 1)
        XCTAssertTrue(zeros.allSatisfy { abs($0 - log(Float(1e-7))) < 0.00001 })
        let tone = (0 ..< 2400).map { Float(sin(2 * Double.pi * 1000 * Double($0) / 24000)) }
        let cancel = try prepare(tone + tone.map { -$0 }, channels: 2)
        XCTAssertEqual(cancel, zeros)
        let ordinary = try prepare(tone, channels: 1)
        let loud = try prepare(tone.map { $0 * 2 }, channels: 1)
        let active = ordinary.indices.filter { ordinary[$0] > -5 }
        XCTAssertFalse(active.isEmpty)
        for i in active { XCTAssertEqual(loud[i] - ordinary[i], log(Float(2)), accuracy: 0.001) }
    }

    func testPinnedTorchFrontendOracleIntermediates() throws {
        try mimoAudioInputRequireNative()
        struct Fixture: Decodable {
            let name: String
            let rate, channels, frames: Int
            let samples, resampled, kernel, mel, window, bank: [Float]
            let spectrumReal, spectrumImaginary, magnitude: [Float]
            let melFrames: Int
        }
        let root = try mimoAudioInputFixtureRoot()
        let fixtures = try JSONDecoder().decode(
            [Fixture].self,
            from: Data(contentsOf: root.appendingPathComponent("frontend-oracle.json")))
        XCTAssertGreaterThanOrEqual(fixtures.count, 4)
        let c = try MiMoV26AudioInputConfiguration.fixture(melBands: 128)
        for f in fixtures {
            let d = MiMoV26AudioPCMDescriptor(
                sourceIdentity: f.name, channels: f.channels, frameCount: f.frames,
                sampleRate: f.rate)
            let p = try MiMoV26AudioInputPlan.make(
                clips: [d], configuration: c, limits: mimoAudioInputTestLimits())
            XCTAssertEqual(p.melFrameCounts, [f.melFrames])
            let coefficients = try MiMoV26AudioFrontend.resampleCoefficients(p.resampling[0])
            XCTAssertEqual(coefficients.count, f.kernel.count)
            for i in coefficients.indices {
                XCTAssertEqual(coefficients[i], f.kernel[i], accuracy: 0.000001, f.name)
            }
            let wave = MLXArray(f.samples).reshaped(f.channels, f.frames)
            let actualWave = MiMoV26AudioFrontend.resample(
                wave, plan: p.resampling[0], coefficients: coefficients
            ).asArray(Float.self)
            XCTAssertEqual(actualWave.count, f.resampled.count)
            for i in actualWave.indices {
                XCTAssertEqual(actualWave[i], f.resampled[i], accuracy: 0.00002, f.name)
            }
            let window = MiMoV26AudioFrontend.periodicHann(960).asArray(Float.self)
            for i in window.indices {
                XCTAssertEqual(window[i], f.window[i], accuracy: 0.000001, f.name)
            }
            let indices = try MiMoV26AudioFrontend.reflectFrameIndices(
                samples: f.resampled.count / f.channels,
                startFrame: 0, count: 1, fftSize: 960, hop: 240)
            let mono = mean(MLXArray(actualWave).reshaped(f.channels, -1), axis: 0)
            let frame =
                take(mono, MLXArray(indices), axis: 0) * MiMoV26AudioFrontend.periodicHann(960)
            let spectrum = MLXFFT.rfft(frame, n: 960)
            let real = spectrum.realPart().asArray(Float.self)
            let imaginary = spectrum.imaginaryPart().asArray(Float.self)
            let magnitude = abs(spectrum).asArray(Float.self)
            XCTAssertEqual(real.count, 481)
            for i in real.indices {
                XCTAssertEqual(real[i], f.spectrumReal[i], accuracy: 0.0002, f.name)
                XCTAssertEqual(imaginary[i], f.spectrumImaginary[i], accuracy: 0.0002, f.name)
                XCTAssertEqual(magnitude[i], f.magnitude[i], accuracy: 0.0002, f.name)
            }
            let bank = MiMoV26AudioFrontend.melBank(c).asArray(Float.self)
            XCTAssertEqual(bank.count, f.bank.count)
            for i in bank.indices { XCTAssertEqual(bank[i], f.bank[i], accuracy: 0.00002, f.name) }
            let actual = try MiMoV26AudioFrontend(configuration: c).prepare(
                pcm: .init(samples: f.samples, descriptor: d), plan: p, clipIndex: 0
            ).values.asArray(Float.self)
            XCTAssertEqual(actual.count, f.mel.count)
            for i in actual.indices { XCTAssertEqual(actual[i], f.mel[i], accuracy: 0.003, f.name) }
        }
    }
}
