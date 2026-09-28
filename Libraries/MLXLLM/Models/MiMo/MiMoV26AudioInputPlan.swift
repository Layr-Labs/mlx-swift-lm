// Copyright © 2026 Eigen Labs. Pure checked metadata; no native execution.
import Foundation

public enum MiMoV26AudioChecked {
    public static func add(_ a: Int, _ b: Int, _ label: String) throws -> Int {
        let (n,o) = a.addingReportingOverflow(b)
        guard a >= 0, b >= 0, !o else { throw MiMoV26AudioInputError.limit(label + " overflow") }; return n
    }
    public static func product(_ values: [Int], _ label: String) throws -> Int {
        try values.reduce(1) { a,b in
            let (n,o) = a.multipliedReportingOverflow(by:b)
            guard a >= 0, b >= 0, !o else { throw MiMoV26AudioInputError.limit(label + " overflow") }; return n
        }
    }
    public static func ceilDivide(_ n: Int, _ d: Int) throws -> Int {
        guard n >= 0, d > 0 else { throw MiMoV26AudioInputError.limit("invalid division") }
        return n/d + (n%d == 0 ? 0 : 1)
    }
}

public struct MiMoV26AudioPCMDescriptor: Codable, Equatable, Sendable {
    public let sourceIdentity: String
    public let channels, frameCount, sampleRate: Int
    public init(sourceIdentity: String, channels: Int, frameCount: Int, sampleRate: Int) {
        self.sourceIdentity = sourceIdentity; self.channels = channels
        self.frameCount = frameCount; self.sampleRate = sampleRate
    }
}

/// Explicit caller bounds, not advertised model/context capacity or a process
/// memory permit. Actual native allocations require root's separate admission.
public struct MiMoV26AudioInputLimits: Equatable, Sendable {
    public let maximumClips, maximumChannels, maximumSampleRate: Int
    public let maximumInputSamples, maximumResampledSamples, maximumResampleCoefficients: Int
    public let maximumMelFrames, maximumSegments, maximumPaddedMelFrames: Int
    public let maximumWorkingElements, frontendFrameBlockSize, rvqTileFrames: Int
    public init(maximumClips: Int, maximumChannels: Int, maximumSampleRate: Int,
                maximumInputSamples: Int, maximumResampledSamples: Int, maximumResampleCoefficients: Int,
                maximumMelFrames: Int, maximumSegments: Int, maximumPaddedMelFrames: Int,
                maximumWorkingElements: Int, frontendFrameBlockSize: Int, rvqTileFrames: Int) {
        self.maximumClips = maximumClips; self.maximumChannels = maximumChannels; self.maximumSampleRate = maximumSampleRate
        self.maximumInputSamples = maximumInputSamples; self.maximumResampledSamples = maximumResampledSamples
        self.maximumResampleCoefficients = maximumResampleCoefficients; self.maximumMelFrames = maximumMelFrames
        self.maximumSegments = maximumSegments; self.maximumPaddedMelFrames = maximumPaddedMelFrames
        self.maximumWorkingElements = maximumWorkingElements; self.frontendFrameBlockSize = frontendFrameBlockSize
        self.rvqTileFrames = rvqTileFrames
    }
    func validate() throws {
        guard [maximumClips,maximumChannels,maximumSampleRate,maximumInputSamples,maximumResampledSamples,
               maximumResampleCoefficients,maximumMelFrames,maximumSegments,maximumPaddedMelFrames,
               maximumWorkingElements,frontendFrameBlockSize,rvqTileFrames].allSatisfy({ $0 > 0 }),
              [maximumChannels,maximumSampleRate,frontendFrameBlockSize,rvqTileFrames].allSatisfy({ $0 <= Int(Int32.max) }) else {
            throw MiMoV26AudioInputError.limit("invalid explicit audio bounds")
        }
    }
}

public struct MiMoV26AudioResamplePlan: Codable, Equatable, Sendable {
    public let originalRate, targetRate, originalFrames, outputFrames: Int
    public let originalRatio, targetRatio, width, taps, coefficients: Int
    public var isIdentity: Bool { originalRate == targetRate }

    public static func make(originalRate: Int, frames: Int, targetRate: Int = 24000,
                            maximumCoefficients: Int) throws -> Self {
        guard originalRate > 0, originalRate <= Int(Int32.max), targetRate > 0,
              targetRate <= Int(Int32.max), frames > 0, frames <= Int(Int32.max), maximumCoefficients > 0 else {
            throw MiMoV26AudioInputError.input("invalid resample geometry")
        }
        if originalRate == targetRate {
            return .init(originalRate:originalRate,targetRate:targetRate,originalFrames:frames,outputFrames:frames,
                originalRatio:1,targetRatio:1,width:0,taps:0,coefficients:0)
        }
        var a = originalRate, b = targetRate
        while b != 0 { let r = a%b; a = b; b = r }
        let p = originalRate/a, q = targetRate/a, base = Double(min(p,q))*0.99
        let support = ceil(6*Double(p)/base)
        guard support.isFinite, support <= Double(Int32.max) else { throw MiMoV26AudioInputError.limit("resample support") }
        let width = Int(support), taps = try MiMoV26AudioChecked.add(try MiMoV26AudioChecked.product([2,width],"resample taps"),p,"resample taps")
        let coefficients = try MiMoV26AudioChecked.product([q,taps],"resample coefficients")
        guard coefficients <= maximumCoefficients, coefficients <= Int(Int32.max) else { throw MiMoV26AudioInputError.limit("resample coefficient ceiling/flat native shape") }
        let numerator = try MiMoV26AudioChecked.product([q,frames],"resample length")
        let rawFrames = try MiMoV26AudioChecked.product([frames/p+1,q],"untrimmed resample shape")
        guard rawFrames <= Int(Int32.max) else { throw MiMoV26AudioInputError.limit("untrimmed native resample shape") }
        guard numerator <= 9_007_199_254_740_992 else { throw MiMoV26AudioInputError.limit("resample Float64 integer precision") }
        // Match pinned torchaudio's Python division -> default-F32 tensor ->
        // ceil conversion, NOT exact rational ceil. Identity bypasses this.
        let converted = Float(Double(numerator)/Double(p)).rounded(.up)
        guard converted.isFinite, converted > 0, Double(converted) <= Double(Int32.max) else {
            throw MiMoV26AudioInputError.limit("resampled frame count")
        }
        return .init(originalRate:originalRate,targetRate:targetRate,originalFrames:frames,outputFrames:Int(converted),
            originalRatio:p,targetRatio:q,width:width,taps:taps,coefficients:coefficients)
    }
}

public struct MiMoV26AudioInputPlan: Equatable, Sendable {
    public struct Segment: Codable, Equatable, Sendable {
        public let clipIndex, melStart, melFrames, convFrames, codeFrames: Int
    }
    public struct Group: Codable, Equatable, Sendable {
        public let segmentIndices: [Int]
        public let maximumMelFrames, maximumConvFrames, pooledWidth, validMelFrames, paddedMelFrames: Int
        public func repeatsLastBeforePooling(segment: Segment) -> Bool {
            segment.convFrames%2 == 1 && segment.convFrames < maximumConvFrames
        }
    }
    public let configuration: MiMoV26AudioInputConfiguration
    public let limits: MiMoV26AudioInputLimits
    public let pcmDescriptors: [MiMoV26AudioPCMDescriptor]?
    public let sourceIdentities: [String]
    public let resampling: [MiMoV26AudioResamplePlan]
    public let melFrameCounts, codeFrameCounts, patchCounts: [Int]
    public let segments: [Segment]
    public let groups: [Group]
    /// Logical FP32-equivalent element ceiling, not measured physical usage.
    public let workingElementUpperBound: Int
    public var totalCodeFrames: Int { codeFrameCounts.reduce(0,+) }
    public var totalPatches: Int { patchCounts.reduce(0,+) }

    public static func make(clips: [MiMoV26AudioPCMDescriptor], configuration: MiMoV26AudioInputConfiguration,
                            limits: MiMoV26AudioInputLimits) throws -> Self {
        try limits.validate()
        guard clips.count <= limits.maximumClips else { throw MiMoV26AudioInputError.limit("audio clip count") }
        var resampling: [MiMoV26AudioResamplePlan] = [], mel: [Int] = []
        var inputSamples = 0, outputSamples = 0, frontWorking = 0
        for clip in clips {
            guard !clip.sourceIdentity.isEmpty, clip.sourceIdentity.utf8.count <= 1024,
                  clip.channels > 0, clip.channels <= limits.maximumChannels,
                  clip.sampleRate > 0, clip.sampleRate <= limits.maximumSampleRate else {
                throw MiMoV26AudioInputError.input("PCM descriptor")
            }
            let count = try MiMoV26AudioChecked.product([clip.channels,clip.frameCount],"PCM samples")
            guard count <= Int(Int32.max) else { throw MiMoV26AudioInputError.limit("flat native PCM shape") }
            inputSamples = try MiMoV26AudioChecked.add(inputSamples,count,"total PCM")
            let resample = try MiMoV26AudioResamplePlan.make(originalRate:clip.sampleRate,frames:clip.frameCount,
                targetRate:configuration.sampleRate,maximumCoefficients:limits.maximumResampleCoefficients)
            guard resample.outputFrames > configuration.fftSize/2 else {
                throw MiMoV26AudioInputError.input("centered reflect STFT requires more than480 resampled frames")
            }
            let output = try MiMoV26AudioChecked.product([clip.channels,resample.outputFrames],"resampled samples")
            outputSamples = try MiMoV26AudioChecked.add(outputSamples,output,"total resampled PCM")
            let frames = resample.outputFrames/configuration.hopLength+1
            resampling.append(resample); mel.append(frames)
            let rawFrames = resample.isIdentity ? resample.outputFrames : try MiMoV26AudioChecked.product(
                [resample.originalFrames/resample.originalRatio+1,resample.targetRatio],"untrimmed resample frames")
            let paddedFrames = resample.isIdentity ? clip.frameCount : try MiMoV26AudioChecked.add(clip.frameCount,
                MiMoV26AudioChecked.add(resample.width,resample.width+resample.originalRatio,"resample padding"),"padded PCM")
            guard paddedFrames <= Int(Int32.max) else { throw MiMoV26AudioInputError.limit("padded native PCM shape") }
            let rawAndPadded = try MiMoV26AudioChecked.product([clip.channels,
                MiMoV26AudioChecked.add(rawFrames,paddedFrames,"resample buffers")],"resample buffers")
            frontWorking = try MiMoV26AudioChecked.add(frontWorking,
                MiMoV26AudioChecked.product([4,try MiMoV26AudioChecked.add(count,rawAndPadded,"PCM coexistence")],"PCM coexistence"),"frontend work")
            frontWorking = try MiMoV26AudioChecked.add(frontWorking,
                MiMoV26AudioChecked.product([4,resample.coefficients],"kernel coefficient intermediates"),"frontend work")
            frontWorking = try MiMoV26AudioChecked.add(frontWorking,
                MiMoV26AudioChecked.product([frames,configuration.fftSize,8],"FFT/magnitude/frame graph"),"frontend work")
        }
        guard inputSamples <= limits.maximumInputSamples, outputSamples <= limits.maximumResampledSamples else {
            throw MiMoV26AudioInputError.limit("decoded/resampled audio sample ceiling")
        }
        return try build(mel:mel,identities:clips.map(\.sourceIdentity),configuration:configuration,limits:limits,
            pcm:clips,resampling:resampling,frontWorking:frontWorking)
    }

    /// Explicit precomputed-mel namespace, never labeled decoded PCM.
    public static func makeMelInputs(frameCounts: [Int], sourceIdentities: [String],
        configuration: MiMoV26AudioInputConfiguration, limits: MiMoV26AudioInputLimits) throws -> Self {
        try limits.validate()
        return try build(mel:frameCounts,identities:sourceIdentities,configuration:configuration,limits:limits,
                         pcm:nil,resampling:[],frontWorking:0)
    }

    private static func build(mel: [Int], identities: [String], configuration c: MiMoV26AudioInputConfiguration,
        limits: MiMoV26AudioInputLimits, pcm: [MiMoV26AudioPCMDescriptor]?, resampling: [MiMoV26AudioResamplePlan],
        frontWorking: Int) throws -> Self {
        guard mel.count == identities.count, mel.count <= limits.maximumClips,
              identities.allSatisfy({ !$0.isEmpty && $0.utf8.count <= 1024 }),
              mel.allSatisfy({ $0 > 0 && $0 <= Int(Int32.max) }) else { throw MiMoV26AudioInputError.input("mel item descriptors") }
        let totalMel = try mel.reduce(0) { try MiMoV26AudioChecked.add($0,$1,"total mel frames") }
        guard totalMel <= limits.maximumMelFrames else { throw MiMoV26AudioInputError.limit("mel frame ceiling") }
        var segments: [Segment] = [], codes = Array(repeating:0,count:mel.count)
        for (clip,length) in mel.enumerated() {
            var start = 0
            while start < length {
                guard segments.count < limits.maximumSegments else { throw MiMoV26AudioInputError.limit("segment ceiling") }
                let count = min(c.segmentSize,length-start), conv = try MiMoV26AudioChecked.ceilDivide(count,2)
                let code = try MiMoV26AudioChecked.ceilDivide(conv,2)
                segments.append(.init(clipIndex:clip,melStart:start,melFrames:count,convFrames:conv,codeFrames:code))
                codes[clip] = try MiMoV26AudioChecked.add(codes[clip],code,"per-clip code frames"); start += count
            }
        }
        var groupIndices: [[Int]] = [], current: [Int] = [], currentSum = 0
        for (i,segment) in segments.enumerated() {
            if currentSum+segment.melFrames > c.maximumGroupValidMelFrames && !current.isEmpty {
                groupIndices.append(current); current = []; currentSum = 0
            }
            current.append(i);currentSum += segment.melFrames
        }
        if !current.isEmpty { groupIndices.append(current) }
        var paddedTotal = 0, working = frontWorking
        let groups = try groupIndices.map { indices -> Group in
            let maximum = indices.map { segments[$0].melFrames }.max()!, conv = try MiMoV26AudioChecked.ceilDivide(maximum,2)
            let padded = try MiMoV26AudioChecked.product([indices.count,maximum],"original padded mel group")
            paddedTotal = try MiMoV26AudioChecked.add(paddedTotal,padded,"padded groups")
            let group = Group(segmentIndices:indices,maximumMelFrames:maximum,maximumConvFrames:conv,
                pooledWidth:try MiMoV26AudioChecked.ceilDivide(conv,2),
                validMelFrames:indices.reduce(0) { $0+segments[$1].melFrames },paddedMelFrames:padded)
            working = try MiMoV26AudioChecked.add(working,
                MiMoV26AudioChecked.product([padded,c.hiddenSize,8],"padded convolution/skip/coexistence"),"codec work")
            for i in indices {
                let t = segments[i].convFrames
                let attention = try MiMoV26AudioChecked.product([t,t,c.heads,8,c.layers],"attention graph ceiling")
                let ffn = try MiMoV26AudioChecked.product([t,c.ffnSize,4,c.layers],"FFN graph ceiling")
                working = try MiMoV26AudioChecked.add(working,MiMoV26AudioChecked.add(attention,ffn,"transformer graph"),"codec work")
            }
            return group
        }
        guard paddedTotal <= limits.maximumPaddedMelFrames else { throw MiMoV26AudioInputError.limit("original padded group ceiling; regrouping is forbidden") }
        let patches = try codes.map { try MiMoV26AudioChecked.ceilDivide($0,c.groupSize) }
        let rvq = totalMel == 0 ? 0 : try MiMoV26AudioChecked.product([limits.rvqTileFrames,max(c.hiddenSize,c.codebookSizes.max()!),8,c.quantizers],"RVQ tile graph")
        working = try MiMoV26AudioChecked.add(working,rvq,"working elements")
        working = try MiMoV26AudioChecked.add(working,MiMoV26AudioChecked.product([totalMel,c.melBands,4],"retained mels"),"working elements")
        guard working <= limits.maximumWorkingElements else { throw MiMoV26AudioInputError.limit("audio working-element ceiling") }
        return .init(configuration:c,limits:limits,pcmDescriptors:pcm,sourceIdentities:identities,resampling:resampling,
            melFrameCounts:mel,codeFrameCounts:codes,patchCounts:patches,segments:segments,groups:groups,workingElementUpperBound:working)
    }

    /// Canonical identity MATERIAL, not authentication. Provider must bind its
    /// real decoded content/tenant/weight identities before persistent reuse.
    public func preparationIdentityData() throws -> Data {
        struct Record: Encodable {
            let codec, frontend, inputKind: String
            let configuration: [String:MiMoV26JSONValue]
            let sourceIdentities: [String]
            let pcm: [MiMoV26AudioPCMDescriptor]?
            let resampling: [MiMoV26AudioResamplePlan]
            let segments: [Segment]; let groups: [Group]
            let fftBlock, rvqTile: Int; let synthetic: Bool
        }
        let record = Record(codec:MiMoV26AudioInputConfiguration.codecProfile,
            frontend:MiMoV26AudioInputConfiguration.frontendProfile,inputKind:pcmDescriptors == nil ? "mel" : "pcm",
            configuration:configuration.rawFields,sourceIdentities:sourceIdentities,pcm:pcmDescriptors,resampling:resampling,
            segments:segments,groups:groups,fftBlock:limits.frontendFrameBlockSize,rvqTile:limits.rvqTileFrames,synthetic:configuration.synthetic)
        let encoder = JSONEncoder();encoder.outputFormatting = [.sortedKeys];return try encoder.encode(record)
    }
}
