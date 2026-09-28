// Copyright © 2026 Eigen Labs.
// Derived algorithm: pinned Torchaudio2.11 (BSD), SGLang67 (Apache-2.0).
// Decoded PCM only. No URL/file decoding, normalization or native execution at init.
import Foundation
import MLX
import MLXLLM

public struct MiMoV26DecodedPCM: Sendable {
    /// Channel-major planar samples. Swift value ownership prevents a caller
    /// mutation of its original array from changing this captured PCM value.
    public let samples: [Float]
    public let descriptor: MiMoV26AudioPCMDescriptor
    public init(samples: [Float], descriptor: MiMoV26AudioPCMDescriptor) throws {
        guard descriptor.channels > 0, descriptor.frameCount > 0,
              try MiMoV26AudioChecked.product([descriptor.channels,descriptor.frameCount],"PCM shape") == samples.count,
              samples.allSatisfy(\.isFinite) else { throw MiMoV26AudioInputError.input("PCM shape/nonfinite samples") }
        self.samples = samples; self.descriptor = descriptor
    }
}

public struct MiMoV26PreparedMel {
    public let values: MLXArray
    public let clipIndex: Int
    public let inputPlanIdentity: Data
    // Only the frontend can label a tensor as this prepared PCM result.
    fileprivate init(values: MLXArray, clipIndex: Int, inputPlanIdentity: Data) {
        self.values = values; self.clipIndex = clipIndex; self.inputPlanIdentity = inputPlanIdentity
    }
}

public final class MiMoV26AudioFrontend {
    public let configuration: MiMoV26AudioInputConfiguration
    public init(configuration: MiMoV26AudioInputConfiguration) { self.configuration = configuration }

    /// Builds a lazy native graph. Caller retains its admitted input/work owner
    /// through evaluation/drain. This method does not change global MLX state.
    public func prepare(pcm: MiMoV26DecodedPCM, plan: MiMoV26AudioInputPlan, clipIndex: Int,
                        isCancelled: () -> Bool = { false }) throws -> MiMoV26PreparedMel {
        guard plan.configuration == configuration, let descriptors = plan.pcmDescriptors,
              descriptors.indices.contains(clipIndex), descriptors[clipIndex] == pcm.descriptor,
              plan.resampling.indices.contains(clipIndex) else {
            throw MiMoV26AudioInputError.input("PCM does not match its preparation plan")
        }
        if isCancelled() { throw MiMoV26AudioInputError.cancelled }
        let identity = try plan.preparationIdentityData(), d = pcm.descriptor, r = plan.resampling[clipIndex]
        let coefficients = try Self.resampleCoefficients(r,isCancelled:isCancelled)
        if isCancelled() { throw MiMoV26AudioInputError.cancelled }
        let wave = MLXArray(pcm.samples).reshaped(d.channels,d.frameCount)
        let resampled = Self.resample(wave,plan:r,coefficients:coefficients)
        // Order is intentional: filtering each channel precedes averaging.
        let mono = mean(resampled,axis:0)
        let window = Self.periodicHann(configuration.fftSize)
        let bank = Self.melBank(configuration)
        var blocks: [MLXArray] = []
        let count = plan.melFrameCounts[clipIndex], block = plan.limits.frontendFrameBlockSize
        for start in stride(from:0,to:count,by:block) {
            if isCancelled() { throw MiMoV26AudioInputError.cancelled }
            let n = min(block,count-start)
            let indices = try Self.reflectFrameIndices(samples:r.outputFrames,startFrame:start,count:n,
                fftSize:configuration.fftSize,hop:configuration.hopLength)
            let frames = take(mono,MLXArray(indices),axis:0).reshaped(n,configuration.fftSize)
            let spectrum = MLXFFT.rfft(frames*window,n:configuration.fftSize,axis:-1)
            let magnitudes = abs(spectrum) // power1, no FFT/window normalization
            blocks.append(log(maximum(matmul(magnitudes,bank),Float(1e-7))))
        }
        let values = blocks.count == 1 ? blocks[0] : concatenated(blocks,axis:0)
        guard values.shape == [count,configuration.melBands], values.dtype == .float32 else {
            throw MiMoV26AudioInputError.input("native frontend shape/dtype")
        }
        return .init(values:values,clipIndex:clipIndex,inputPlanIdentity:identity)
    }

    static func resampleCoefficients(_ p: MiMoV26AudioResamplePlan,
                                     isCancelled: () -> Bool = { false }) throws -> [Float] {
        guard !p.isIdentity else { return [] }
        let base = Double(min(p.originalRatio,p.targetRatio))*0.99
        var output = [Float]();output.reserveCapacity(p.coefficients)
        for phase in 0..<p.targetRatio {
            if isCancelled() { throw MiMoV26AudioInputError.cancelled }
            // torch.arange(integer)/q first rounds to the recorded default
            // F32, then addition to the F64 index promotes it to Double.
            let offset = Double(Float(-phase)/Float(p.targetRatio))
            for index in -p.width..<(p.width+p.originalRatio) {
                let t = min(6.0,max(-6.0,(offset+Double(index)/Double(p.originalRatio))*base))
                let hann = cos((t*Double.pi/6)/2)
                let angle = t*Double.pi
                let sinc = angle == 0 ? 1 : sin(angle)/angle
                let window = hann*hann, scale = base/Double(p.originalRatio)
                output.append(Float(sinc*(window*scale)))
            }
        }
        guard output.count == p.coefficients, output.allSatisfy(\.isFinite) else {
            throw MiMoV26AudioInputError.input("resample coefficient construction")
        }
        return output
    }

    static func resample(_ wave: MLXArray, plan p: MiMoV26AudioResamplePlan, coefficients: [Float]) -> MLXArray {
        if p.isIdentity { return wave }
        let input = padded(wave.expandedDimensions(axis:2),widths:[0,IntOrPair((p.width,p.width+p.originalRatio)),0])
        let kernel = MLXArray(coefficients).reshaped(p.targetRatio,p.taps,1)
        // MLX NLC output is already [channels, coarse-time, phase], matching
        // Torch's output.transpose(1,2).reshape before its exact trim.
        let output = conv1d(input,kernel,stride:p.originalRatio).reshaped(wave.dim(0),-1)
        return output[0...,0..<p.outputFrames]
    }

    static func periodicHann(_ size: Int) -> MLXArray {
        let position = MLXArray(0..<size).asType(.float32)
        // Torch TensorFactories.hann_window delegates to hamming(alpha=.5,
        // beta=.5): multiply, cos, multiply(-.5), add(.5), in that order.
        return cos(position*Float(2*Double.pi/Double(size)))*Float(-0.5)+Float(0.5)
    }

    static func reflectFrameIndices(samples: Int, startFrame: Int, count: Int,
                                    fftSize: Int, hop: Int) throws -> [Int32] {
        guard samples > fftSize/2, samples <= Int(Int32.max), startFrame >= 0, count > 0,
              fftSize > 0, hop > 0 else { throw MiMoV26AudioInputError.input("reflect/STFT geometry") }
        guard startFrame <= samples/hop+1, count <= samples/hop+1-startFrame else {
            throw MiMoV26AudioInputError.input("FFT frame range")
        }
        let elements = try MiMoV26AudioChecked.product([count,fftSize],"frame indices")
        guard elements <= Int(Int32.max) else { throw MiMoV26AudioInputError.limit("flat native frame-index shape") }
        var result = [Int32]();result.reserveCapacity(elements)
        for frame in startFrame..<(try MiMoV26AudioChecked.add(startFrame,count,"FFT frames")) {
            let center = try MiMoV26AudioChecked.product([frame,hop],"frame center")
            for feature in 0..<fftSize {
                let position = center-fftSize/2+feature
                let index = position < 0 ? -position : position >= samples ? 2*samples-2-position : position
                guard index >= 0, index < samples else { throw MiMoV26AudioInputError.input("reflect frame exceeds declared waveform") }
                result.append(Int32(index))
            }
        }
        return result
    }

    static func melBank(_ c: MiMoV26AudioInputConfiguration) -> MLXArray {
        let frequency = MLXArray.linspace(Float(0),Float(c.sampleRate/2),count:c.fftSize/2+1)
        let maxMel = Float(2595*log10(1+Double(c.sampleRate/2)/700))
        let mel = MLXArray.linspace(Float(0),maxMel,count:c.melBands+2)
        let hz = Float(700)*(pow(Float(10),mel/Float(2595))-Float(1))
        let differences = hz[1...] - hz[..<(c.melBands+1)]
        let slopes = hz.expandedDimensions(axis:0)-frequency.expandedDimensions(axis:1)
        let down = -slopes[0...,0..<c.melBands]/differences[0..<c.melBands]
        let up = slopes[0...,2...]/differences[1...]
        return maximum(Float(0),minimum(down,up)) // no Slaney normalization
    }
}
