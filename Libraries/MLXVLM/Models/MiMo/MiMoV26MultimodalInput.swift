// Copyright © 2026 Eigen Labs.
// Decoded-input contract; SGLang67bb6a58 native ordering, not an encoded-media API.
import CryptoKit
import Foundation
import Jinja
import MLXLLM
import MLXLMCommon

public enum MiMoV26MultimodalError: Error, Equatable, Sendable {
    case invalidInput(String), unsupportedProfile(String), limit(String)
    case missingAudioCodec, incompatibleOwner, invalidatedOwner, incompatiblePlan
    case cancelled, invalidFeatures, reservationRejected, drainFailed
}

/// Data-only projection of ProviderCoreFoundation.PromptRenderDate. No caller
/// closure is stored or admitted. The provider passes the SAME resolved day
/// used by its ordinary prompt pipeline; the actual renderer stays Jinja.
public struct MiMoV26MediaRequestClock: Sendable, Equatable {
    public let value: String
    public init(utcGregorianDay value: String) throws {
        let bytes = Array(value.utf8)
        guard bytes.count == 10, bytes[4] == 45, bytes[7] == 45,
              bytes.enumerated().allSatisfy({ index, byte in
                  index == 4 || index == 7 || (48...57).contains(byte)
              }) else { throw MiMoV26MultimodalError.invalidInput("invalid request clock snapshot") }
        let year = Int(value.prefix(4))!
        let month = Int(String(value.dropFirst(5).prefix(2)))!
        let day = Int(value.suffix(2))!
        guard year > 0, (1...12).contains(month) else {
            throw MiMoV26MultimodalError.invalidInput("invalid request clock snapshot")
        }
        let leap = year.isMultiple(of: 4) && (!year.isMultiple(of: 100) || year.isMultiple(of: 400))
        let days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        guard (1...days[month - 1]).contains(day) else {
            throw MiMoV26MultimodalError.invalidInput("invalid request clock snapshot")
        }
        self.value = value
    }
    // Mirrors the existing PromptRenderDate.function arity, kwargs and
    // fallback exactly. Different ambient fallback clocks are NOT byte-parity
    // evidence; only the supported literal day call is deterministic.
    var templateValue: Jinja.Value {
        .function { args, kwargs, environment in
            if args.count == 1, kwargs.isEmpty,
               case .string("%Y-%m-%d") = args[0] {
                return .string(value)
            }
            return try Globals.strftimeNow(args, kwargs, environment)
        }
    }
}

/// Already sampled, complete, explicitly SILENT visual clip. Timestamps are
/// source FP32 seconds. No frame sampling, rotation, decoding or track removal.
public struct MiMoV26SilentVideo: Sendable {
    public let frames: [MiMoV26Pixels.DecodedRGB]
    public let timestamps: [Float]
    public init(frames: [MiMoV26Pixels.DecodedRGB], timestamps: [Float]) {
        self.frames = frames; self.timestamps = timestamps
    }
}

public enum MiMoV26MultimodalContent: Sendable {
    case text(String)
    case image(MiMoV26Pixels.DecodedRGB)
    case silentVideo(MiMoV26SilentVideo)
    case audio(MiMoV26DecodedPCM)
    case audiovisual(MiMoV26DecodedAudiovisual)
}

/// Fields have already passed the host's authoritative native normalization.
/// This layer preserves them; it does not repair history or tool arguments.
public struct MiMoV26MultimodalMessage: Sendable {
    public let role: Chat.Message.Role
    public let content: [MiMoV26MultimodalContent]
    public let templateFields: [String: any Sendable]
    public init(role: Chat.Message.Role, content: [MiMoV26MultimodalContent],
                templateFields: [String: any Sendable] = [:]) {
        self.role = role; self.content = content; self.templateFields = templateFields
    }
}

public struct MiMoV26MultimodalInput: Sendable {
    public let messages: [MiMoV26MultimodalMessage]
    public let tools: [ToolSpec]?
    public let additionalContext: [String: any Sendable]?
    public let maximumOutputTokens: Int
    public init(messages: [MiMoV26MultimodalMessage], tools: [ToolSpec]? = nil,
                additionalContext: [String: any Sendable]? = nil, maximumOutputTokens: Int) {
        self.messages = messages; self.tools = tools; self.additionalContext = additionalContext
        self.maximumOutputTokens = maximumOutputTokens
    }
}

/// Explicit request bounds, not model capacity or a physical-memory permit.
public struct MiMoV26MultimodalLimits: Sendable {
    public let maximumMedia, maximumVideoFrames, maximumPromptTokens: Int
    public let maximumMetadataBytes, maximumMetadataNodes, maximumMetadataDepth: Int
    public let pixels: MiMoV26Pixels.Limits
    public let vision: MiMoV26VisionLimits
    public let audio: MiMoV26AudioInputLimits
    public let audioPatch: MiMoV26AudioPatchLimits
    public init(maximumMedia: Int, maximumVideoFrames: Int, maximumPromptTokens: Int,
                maximumMetadataBytes: Int, maximumMetadataNodes: Int, maximumMetadataDepth: Int,
                pixels: MiMoV26Pixels.Limits, vision: MiMoV26VisionLimits,
                audio: MiMoV26AudioInputLimits, audioPatch: MiMoV26AudioPatchLimits) {
        self.maximumMedia = maximumMedia; self.maximumVideoFrames = maximumVideoFrames
        self.maximumPromptTokens = maximumPromptTokens; self.maximumMetadataBytes = maximumMetadataBytes
        self.maximumMetadataNodes = maximumMetadataNodes; self.maximumMetadataDepth = maximumMetadataDepth
        self.pixels = pixels; self.vision = vision; self.audio = audio; self.audioPatch = audioPatch
    }
}

/// The host must return its actual request reservation and validate this exact
/// sealed plan. A conforming object/digest is NOT authentication by the SDK.
/// Its owner is retained through preparation drain and the queued request.
public protocol MiMoV26MediaWorkReservation: AnyObject {
    func validate(plan: MiMoV26MultimodalPlan) throws
    /// Mandatory ownership transfer, not a notification. Retain this opaque
    /// bundle and the actual lease in the host's failed-work quarantine until
    /// successful recovery or process exit. Fault/retire the serving lane.
    func retainAfterFailedDrain(_ work: MiMoV26FailedMediaWork)
}

public struct MiMoV26MultimodalSpan: Equatable, Sendable {
    public enum Kind: String, Sendable { case image, video, audio }
    public let kind: Kind
    public let mediaIndex, featureOffset, tokenOffset, length: Int
    public var engineSpan: CBv2ImageSpan { .init(tokenOffset: tokenOffset, length: length) }
}

/// No public initializer or mutable fields: plans cannot be relabelled with a
/// different processor/owner or changed media after host admission.
public struct MiMoV26MultimodalPlan: Sendable {
    public let promptTokens: [Int]
    public let spans: [MiMoV26MultimodalSpan]
    public let maximumOutputTokens: Int
    public let preparationSHA256: String
    public let profile: String
    public let loadedOwnerIdentity: UUID
    public let configurationSHA256, templateSHA256: String
    /// Actual per-item geometry for host work admission, keyed by ordered media
    /// index. Counts/buffer envelopes are not measured physical residency.
    public let visionGeometryByMediaIndex: [Int: MiMoV26MediaGeometry.Plan]
    public let audioPlan: MiMoV26AudioInputPlan?
    public let decodedElements, patchElements, featureElements: Int
    /// Logical retained-buffer counts only. Host must add real transient/Metal
    /// bounds and existing reserves; these are not a measured peak or ledger.
    public let logicalFeatureBytes: Int
    let processorIdentity: UUID
    let parts: [MiMoV26PlannedMedia]
}

struct MiMoV26PlannedMedia: Sendable {
    let content: MiMoV26MultimodalContent
    let geometry: MiMoV26MediaGeometry.Plan?
    let audioIndex: Int?
}

struct MiMoV26MultimodalProfile {
    static let name = "sglang67-decoded-rgb-silent-f32-timestamps-hf5711-pcm-v1"
    static let spellings = ["vision_start_token_id":"<|vision_start|>",
        "vision_end_token_id":"<|vision_end|>", "image_token_id":"<|image_pad|>",
        "video_token_id":"<|video_pad|>", "video_start_token_id":"<|mimo_video_start|>",
        "video_end_token_id":"<|mimo_video_end|>", "audio_start_token_id":"<|mimo_audio_start|>",
        "audio_end_token_id":"<|mimo_audio_end|>", "audio_token_id":"<|audio_pad|>"]
    let settings: MiMoV26MediaGeometry.Settings
    let tokens: [String: Int]
    init(configuration c: MiMoV26Configuration, tokenizer: any Tokenizer) throws {
        guard let p = c.processorFields, let vision = c.vision,
              p["rope_type"] == .string("rope"), p["use_video_timestamps"] == .bool(true),
              p["use_per_grid_t_timestamps"] == .bool(false),
              try Self.integer(p, "temporal_compression_ratio") == 1 else {
            throw MiMoV26MultimodalError.unsupportedProfile("native 1D causal profile")
        }
        settings = try .init(patchSize: Self.integer(p,"patch_size"), mergeSize: Self.integer(p,"merge_size"),
            temporalPatchSize: Self.integer(p,"temporal_patch_size"), temporalCompressionRatio: 1,
            imageMinPixels: Self.integer(p,"image_min_pixels"), imageMaxPixels: Self.integer(p,"image_max_pixels"),
            videoMinPixels: Self.integer(p,"video_min_pixels"), videoMaxPixels: Self.integer(p,"video_max_pixels"),
            videoTotalMaxPixels: Self.integer(p,"video_total_max_pixels"))
        guard settings.patchSize == vision.patchSize, settings.mergeSize == vision.spatialMergeSize,
              settings.temporalPatchSize == vision.temporalPatchSize else {
            throw MiMoV26MultimodalError.unsupportedProfile("processor/tower geometry")
        }
        var resolved: [String: Int] = [:]
        for (key, spelling) in Self.spellings {
            let id = try Self.integer(p,key)
            guard id >= 0, id < c.vocabularySize, tokenizer.convertTokenToId(spelling) == id,
                  tokenizer.convertIdToToken(id) == spelling,
                  c.tokenIDs[key] == nil || c.tokenIDs[key] == id else {
                throw MiMoV26MultimodalError.unsupportedProfile("tokenizer/processor token namespace")
            }
            resolved[key] = id
        }
        guard Set(resolved.values).count == resolved.count else {
            throw MiMoV26MultimodalError.unsupportedProfile("aliased media token IDs")
        }
        tokens = resolved
    }
    static func integer(_ values: [String: MiMoV26JSONValue], _ key: String) throws -> Int {
        guard case .number(let value) = values[key] else {
            throw MiMoV26MultimodalError.unsupportedProfile(key)
        }
        let number = NSDecimalNumber(decimal: value).int64Value
        guard Decimal(number) == value, let result = Int(exactly: number) else {
            throw MiMoV26MultimodalError.unsupportedProfile(key)
        }
        return result
    }
    static func timestamp(_ value: Float) throws -> String {
        guard value.isFinite, value >= 0, Double(value) < Double(Int.max) else {
            throw MiMoV26MultimodalError.invalidInput("nonrepresentable timestamp")
        }
        let whole = Int(value)
        let minutes = String(whole / 60), seconds = String(whole % 60)
        let paddedMinutes = String(repeating: "0", count: max(0, 2 - minutes.count)) + minutes
        let paddedSeconds = String(repeating: "0", count: max(0, 2 - seconds.count)) + seconds
        return paddedMinutes + ":" + paddedSeconds
    }
}

enum MiMoV26MediaMetadata {
    static func validate(_ value: Any, limits: MiMoV26MultimodalLimits,
                         nodes: inout Int, bytes: inout Int, depth: Int = 0) throws {
        nodes = try MiMoV26AudioChecked.add(nodes,1,"metadata nodes")
        guard nodes <= limits.maximumMetadataNodes, depth <= limits.maximumMetadataDepth else {
            throw MiMoV26MultimodalError.limit("metadata traversal")
        }
        if let native = value as? Jinja.Value {
            switch native {
            case .null, .boolean, .int: return
            case .double(let number):
                guard number.isFinite else { throw MiMoV26MultimodalError.invalidInput("nonfinite template number") }
                return
            case .string(let string):
                bytes = try MiMoV26AudioChecked.add(bytes,string.utf8.count,"metadata bytes")
                guard bytes <= limits.maximumMetadataBytes,
                      !MiMoV26MultimodalProfile.spellings.values.contains(where:string.contains) else {
                    throw MiMoV26MultimodalError.invalidInput("metadata limit or ambiguous literal media marker")
                }
                return
            case .array(let values):
                for child in values { try validate(child,limits:limits,nodes:&nodes,bytes:&bytes,depth:depth+1) }
                return
            case .object(let values):
                for (key,child) in values {
                    try validate(key,limits:limits,nodes:&nodes,bytes:&bytes,depth:depth+1)
                    try validate(child,limits:limits,nodes:&nodes,bytes:&bytes,depth:depth+1)
                }
                return
            default:
                throw MiMoV26MultimodalError.invalidInput("template fields must be JSON data")
            }
        }
        if let string = value as? String {
            bytes = try MiMoV26AudioChecked.add(bytes,string.utf8.count,"metadata bytes")
            guard bytes <= limits.maximumMetadataBytes,
                  !MiMoV26MultimodalProfile.spellings.values.contains(where: string.contains) else {
                throw MiMoV26MultimodalError.invalidInput("metadata limit or ambiguous literal media marker")
            }
        } else if let array = value as? [Any] {
            for child in array { try validate(child,limits:limits,nodes:&nodes,bytes:&bytes,depth:depth+1) }
        } else if let object = value as? [String: Any] {
            for (key,child) in object {
                try validate(key,limits:limits,nodes:&nodes,bytes:&bytes,depth:depth+1)
                try validate(child,limits:limits,nodes:&nodes,bytes:&bytes,depth:depth+1)
            }
        } else if value is NSNull || value is Bool || value is Int || value is Int64 || value is UInt64 {
            return
        } else if let number = value as? Double, number.isFinite { return }
        else { throw MiMoV26MultimodalError.invalidInput("non-JSON native template field") }
    }
}

extension MiMoV26MultimodalProcessor {
    /// Host-only planning. Does not allocate MLX arrays or execute a tower.
    public func plan(_ input: MiMoV26MultimodalInput) throws -> MiMoV26MultimodalPlan {
        try checkOwner()
        let c = configuration, l = limits, p = profile
        guard !input.messages.isEmpty, input.maximumOutputTokens > 0,
              [l.maximumMedia,l.maximumVideoFrames,l.maximumPromptTokens,l.maximumMetadataBytes,
               l.maximumMetadataNodes,l.maximumMetadataDepth,l.pixels.maximumInputElements,
               l.pixels.maximumOutputElements,l.pixels.maximumWorkingBytes,l.vision.maximumPatches,
               l.vision.maximumAttentionScoreElements,l.audioPatch.maximumClips,l.audioPatch.maximumFrames,
               l.audioPatch.maximumPatches,l.audioPatch.maximumWorkingElements].allSatisfy({ $0 > 0 }) else {
            throw MiMoV26MultimodalError.invalidInput("empty request or invalid limits")
        }
        guard l.maximumMetadataDepth <= 128, l.maximumMetadataNodes <= 1_000_000 else {
            throw MiMoV26MultimodalError.limit("native metadata safety bounds")
        }
        var nodes = 0, bytes = 0, messages: [Message] = [], parts: [MiMoV26PlannedMedia] = []
        var clips: [MiMoV26DecodedPCM] = [], decoded = 0, patches = 0
        var hash = SHA256()
        func hashInteger(_ n: Int) { var v = UInt64(n).bigEndian; withUnsafeBytes(of:&v) { hash.update(data:Data($0)) } }
        func hashFloats(_ a: [Float]) {
            // Bounded host scratch, not one allocation per pixel or an entire
            // second raster. Explicit little-endian Float32 bit identity.
            var words: [UInt32] = []; words.reserveCapacity(4096)
            for value in a {
                words.append(value.bitPattern.littleEndian)
                if words.count == 4096 { words.withUnsafeBytes { hash.update(data:Data($0)) }; words.removeAll(keepingCapacity:true) }
            }
            if !words.isEmpty { words.withUnsafeBytes { hash.update(data:Data($0)) } }
        }
        func validateFrame(_ frame: MiMoV26Pixels.DecodedRGB) throws {
            guard frame.height > 0, frame.width > 0,
                  try MiMoV26AudioChecked.product([3,frame.height,frame.width],"RGB shape") == frame.planarRGB.count,
                  frame.planarRGB.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 255 }) else {
                throw MiMoV26MultimodalError.invalidInput("decoded planar RGB")
            }
            decoded = try MiMoV26AudioChecked.add(decoded,frame.planarRGB.count,"decoded elements")
            guard decoded <= l.pixels.maximumInputElements else { throw MiMoV26MultimodalError.limit("aggregate RGB elements") }
            hashInteger(frame.height); hashInteger(frame.width); hashFloats(frame.planarRGB)
        }
        if let tools = input.tools { try MiMoV26MediaMetadata.validate(tools,limits:l,nodes:&nodes,bytes:&bytes) }
        var renderContext = input.additionalContext
        if var context = input.additionalContext {
            if let clock = context.removeValue(forKey:"_darkbloom_request_clock") {
                guard let snapshot = clock as? MiMoV26MediaRequestClock else {
                    throw MiMoV26MultimodalError.invalidInput("request clock requires a data-only snapshot")
                }
                try MiMoV26MediaMetadata.validate(snapshot.value,limits:l,nodes:&nodes,bytes:&bytes)
                renderContext?["_darkbloom_request_clock"] = snapshot.templateValue
            }
            guard Set(context.keys).isSubset(of:["enable_thinking","add_generation_prompt"]),
                  context.values.allSatisfy({ type(of:$0) == Bool.self }) else {
                throw MiMoV26MultimodalError.invalidInput("native Boolean render controls only")
            }
            try MiMoV26MediaMetadata.validate(context,limits:l,nodes:&nodes,bytes:&bytes)
        }
        for message in input.messages {
            let structuralKeys: Set<String> = ["role","content","image","images","image_url","video","videos",
                "video_url","audio","audio_url","input_audio","audio_chunks","pixel_values"]
            guard Set(message.templateFields.keys).isDisjoint(with:structuralKeys) else {
                throw MiMoV26MultimodalError.invalidInput("structured role/content override or unbound media field")
            }
            try MiMoV26MediaMetadata.validate(message.templateFields,limits:l,nodes:&nodes,bytes:&bytes)
            var native = message.templateFields, content: [[String: any Sendable]] = []
            for item in message.content {
                if case .text(let text) = item {
                    try MiMoV26MediaMetadata.validate(text,limits:l,nodes:&nodes,bytes:&bytes)
                    content.append(["type":"text","text":text]); continue
                }
                guard parts.count < l.maximumMedia else { throw MiMoV26MultimodalError.limit("media count") }
                var geometry: MiMoV26MediaGeometry.Plan?, audioIndex: Int?
                switch item {
                case .image(let frame):
                    try validateFrame(frame)
                    geometry = try MiMoV26MediaGeometry.image(height:frame.height,width:frame.width,settings:p.settings)
                    content.append(["type":"image"])
                case .silentVideo(let video):
                    guard !video.frames.isEmpty, video.frames.count <= l.maximumVideoFrames,
                          video.frames.count == video.timestamps.count,
                          let first = video.frames.first else { throw MiMoV26MultimodalError.invalidInput("complete silent sampled video") }
                    var previous: Float = -1
                    for (frame,time) in zip(video.frames,video.timestamps) {
                        guard time.isFinite, time >= 0, time > previous,
                              frame.height == first.height, frame.width == first.width else {
                            throw MiMoV26MultimodalError.invalidInput("video geometry or strictly increasing FP32 timestamps")
                        }
                        _ = try MiMoV26MultimodalProfile.timestamp(time); previous = time
                        try validateFrame(frame)
                    }
                    hashFloats(video.timestamps)
                    geometry = try MiMoV26MediaGeometry.video(height:first.height,width:first.width,
                        sampledFrames:video.frames.count,sampledFrameLimit:l.maximumVideoFrames,settings:p.settings)
                    content.append(["type":"video"])
                case .audio(let clip):
                    guard audioCodec != nil else { throw MiMoV26MultimodalError.missingAudioCodec }
                    audioIndex = clips.count; clips.append(clip)
                    hashInteger(clip.descriptor.channels); hashInteger(clip.descriptor.frameCount)
                    hashInteger(clip.descriptor.sampleRate); hashFloats(clip.samples)
                    content.append(["type":"audio"])
                case .audiovisual(let av):
                    guard let codec = audioCodec else { throw MiMoV26MultimodalError.missingAudioCodec }
                    guard let fields = c.processorFields,
                          (fields["video_audio_interleave_length"] ?? .number(0)) == .number(0),
                          (fields["audio_input_id_per_second"] ?? .number(25)) == .number(25),
                          codec.input.configuration.groupSize == 4,
                          p.settings.temporalCompressionRatio == 1 else {
                        throw MiMoV26MultimodalError.unsupportedProfile("decoded AV interleave0/6.25 patch rate")
                    }
                    guard !av.frames.isEmpty, av.frames.count <= l.maximumVideoFrames,
                          av.frames.count == av.timestamps.count, let first = av.frames.first else {
                        throw MiMoV26MultimodalError.invalidInput("complete decoded AV sampled clip")
                    }
                    var previous: Float = -1
                    for (frame,time) in zip(av.frames,av.timestamps) {
                        guard time.isFinite, time >= 0, time > previous,
                              frame.height == first.height, frame.width == first.width else {
                            throw MiMoV26MultimodalError.invalidInput("decoded AV RGB/time geometry")
                        }
                        _ = try MiMoV26MultimodalProfile.timestamp(time); previous = time
                        try validateFrame(frame)
                    }
                    hash.update(data:Data(MiMoV26AudiovisualLayout.profileName.utf8))
                    hashFloats(av.timestamps)
                    switch av.segmentEnd {
                    case .float32(let end): hashInteger(32); hashFloats([end])
                    case .float64(let end):
                        hashInteger(64)
                        var bits = end.bitPattern.littleEndian
                        withUnsafeBytes(of:&bits) { hash.update(data:Data($0)) }
                    }
                    geometry = try MiMoV26MediaGeometry.video(height:first.height,width:first.width,
                        sampledFrames:av.frames.count,sampledFrameLimit:l.maximumVideoFrames,settings:p.settings)
                    audioIndex = clips.count; clips.append(av.wholeAudio)
                    hashInteger(av.wholeAudio.descriptor.channels); hashInteger(av.wholeAudio.descriptor.frameCount)
                    hashInteger(av.wholeAudio.descriptor.sampleRate); hashFloats(av.wholeAudio.samples)
                    // One native video placeholder becomes linked AV units;
                    // never append an independent whole-audio message/span.
                    content.append(["type":"video"])
                case .text: preconditionFailure("handled above")
                }
                if let geometry {
                    patches = try MiMoV26AudioChecked.add(patches,geometry.patchElementCount,"patch elements")
                    guard geometry.patchElementCount <= l.pixels.maximumOutputElements,
                          geometry.patchCount <= l.vision.maximumPatches else { throw MiMoV26MultimodalError.limit("vision plan") }
                    let framePatches = try MiMoV26AudioChecked.product([geometry.gridH,geometry.gridW],"frame patches")
                    let scores = try MiMoV26AudioChecked.product([c.vision!.queryHeads,framePatches,framePatches],"attention scores")
                    guard scores <= l.vision.maximumAttentionScoreElements else { throw MiMoV26MultimodalError.limit("vision attention") }
                }
                parts.append(.init(content:item,geometry:geometry,audioIndex:audioIndex))
            }
            native["role"] = message.role.rawValue; native["content"] = content; messages.append(native)
        }
        guard !parts.isEmpty else { throw MiMoV26MultimodalError.invalidInput("use the native text processor for text-only input") }
        let audioPlan = try clips.isEmpty ? nil : MiMoV26AudioInputPlan.make(clips:clips.map(\.descriptor),
            configuration:audioCodec!.input.configuration,limits:l.audio)
        if let audioPlan {
            guard clips.count <= l.audioPatch.maximumClips,
                  audioPlan.totalCodeFrames <= l.audioPatch.maximumFrames,
                  audioPlan.totalPatches <= l.audioPatch.maximumPatches else { throw MiMoV26MultimodalError.limit("audio patch plan") }
            hash.update(data:try audioPlan.preparationIdentityData())
        }
        let symbolic = try tokenizer.applyChatTemplate(messages:messages,chatTemplate:chatTemplate,
            tools:input.tools,additionalContext:renderContext)
        let reserved = Set(p.tokens.values)
        var output: [Int] = [], spans: [MiMoV26MultimodalSpan] = [], slot = 0, index = 0, featureElements = 0
        func append(_ ids: [Int]) throws {
            let count = try MiMoV26AudioChecked.add(output.count,ids.count,"expanded prompt")
            guard count <= l.maximumPromptTokens, count <= c.maxPositionEmbeddings-input.maximumOutputTokens,
                  ids.allSatisfy({ $0 >= 0 && $0 < c.vocabularySize }) else { throw MiMoV26MultimodalError.limit("expanded native context/vocabulary") }
            output.append(contentsOf:ids)
        }
        func span(_ kind: MiMoV26MultimodalSpan.Kind, _ pad: Int, _ length: Int, _ featureOffset: Int) throws {
            guard length > 0, length <= l.maximumPromptTokens-output.count,
                  length <= c.maxPositionEmbeddings-input.maximumOutputTokens-output.count else { throw MiMoV26MultimodalError.limit("media span") }
            spans.append(.init(kind:kind,mediaIndex:slot,featureOffset:featureOffset,tokenOffset:output.count,length:length))
            featureElements = try MiMoV26AudioChecked.add(featureElements,
                MiMoV26AudioChecked.product([length,c.hiddenSize],"features"),"features")
            try append(Array(repeating:pad,count:length))
        }
        guard input.maximumOutputTokens < c.maxPositionEmbeddings else { throw MiMoV26MultimodalError.limit("output context") }
        while index < symbolic.count {
            let id = symbolic[index]
            guard reserved.contains(id) else { try append([id]); index += 1; continue }
            guard slot < parts.count, index+2 < symbolic.count else { throw MiMoV26MultimodalError.invalidInput("unbound media marker") }
            let item = parts[slot], start = p.tokens["vision_start_token_id"]!, end = p.tokens["vision_end_token_id"]!
            switch item.content {
            case .image:
                guard Array(symbolic[index..<(index+3)]) == [start,p.tokens["image_token_id"]!,end] else { throw MiMoV26MultimodalError.invalidInput("ordered image marker") }
                try append([start]); try span(.image,p.tokens["image_token_id"]!,item.geometry!.mediaTokens,0); try append([end])
            case .silentVideo(let video):
                guard Array(symbolic[index..<(index+3)]) == [start,p.tokens["video_token_id"]!,end] else { throw MiMoV26MultimodalError.invalidInput("ordered video marker") }
                let geometry = item.geometry!, count = geometry.gridH*geometry.gridW/(p.settings.mergeSize*p.settings.mergeSize)
                try append([p.tokens["video_start_token_id"]!])
                for group in 0..<geometry.timestampCount {
                    let timestamp = video.timestamps[min(group*p.settings.temporalPatchSize,video.timestamps.count-1)]
                    let timestampIDs = tokenizer.encode(text:try MiMoV26MultimodalProfile.timestamp(timestamp),addSpecialTokens:true)
                    guard !timestampIDs.isEmpty, reserved.isDisjoint(with:timestampIDs) else { throw MiMoV26MultimodalError.invalidInput("timestamp tokenization") }
                    try append(timestampIDs); try append([start]); try span(.video,p.tokens["video_token_id"]!,count,group*count); try append([end])
                }
                try append([p.tokens["video_end_token_id"]!])
            case .audio:
                let a = p.tokens["audio_start_token_id"]!, z = p.tokens["audio_end_token_id"]!, pad = p.tokens["audio_token_id"]!
                guard Array(symbolic[index..<(index+3)]) == [a,pad,z] else { throw MiMoV26MultimodalError.invalidInput("ordered audio marker") }
                try append([a]); try span(.audio,pad,audioPlan!.patchCounts[item.audioIndex!],0); try append([z])
            case .audiovisual(let av):
                guard Array(symbolic[index..<(index+3)]) == [start,p.tokens["video_token_id"]!,end],
                      let geometry = item.geometry, let audioIndex = item.audioIndex,
                      let audioPlan else { throw MiMoV26MultimodalError.invalidInput("ordered decoded AV marker") }
                let layout = try MiMoV26AudiovisualLayout.make(timestamps:av.timestamps,
                    segmentEnd:av.segmentEnd,temporalPatchSize:p.settings.temporalPatchSize,
                    wholeAudioPatches:audioPlan.patchCounts[audioIndex],maximumUnits:l.maximumVideoFrames)
                guard layout.units.count == geometry.timestampCount,
                      layout.alignedFrames == geometry.alignedFrames else { throw MiMoV26MultimodalError.incompatiblePlan }
                let count = geometry.gridH*geometry.gridW/(p.settings.mergeSize*p.settings.mergeSize)
                try append([p.tokens["video_start_token_id"]!])
                for unit in layout.units {
                    let stamp = tokenizer.encode(text:try MiMoV26MultimodalProfile.timestamp(unit.timestamp),addSpecialTokens:true)
                    guard !stamp.isEmpty, reserved.isDisjoint(with:stamp) else { throw MiMoV26MultimodalError.invalidInput("timestamp tokenization") }
                    try append(stamp)
                    try append([start]); try span(.video,p.tokens["video_token_id"]!,count,unit.visualGroup*count); try append([end])
                    try append([p.tokens["audio_start_token_id"]!])
                    try span(.audio,p.tokens["audio_token_id"]!,unit.audioRange.count,unit.audioRange.lowerBound)
                    try append([p.tokens["audio_end_token_id"]!])
                }
                try append([p.tokens["video_end_token_id"]!])
                // Slices retain the FULL encoded/patch backing. Price unused
                // prefix/tail features too; prompt tokens still use only spans.
                featureElements = try MiMoV26AudioChecked.add(featureElements,
                    MiMoV26AudioChecked.product([layout.unusedAudioPatches,c.hiddenSize],"whole AV audio backing"),
                    "whole AV audio backing")
            case .text: preconditionFailure("text is not a media slot")
            }
            index += 3; slot += 1
        }
        guard slot == parts.count else { throw MiMoV26MultimodalError.invalidInput("missing media placeholders") }
        hash.update(data:Data(profileIdentity().utf8))
        hashInteger(input.maximumOutputTokens); hashInteger(parts.count)
        for token in output { hashInteger(token) }
        let totalDecoded = try clips.reduce(decoded) { try MiMoV26AudioChecked.add($0,$1.samples.count,"total decoded elements") }
        return .init(promptTokens:output,spans:spans,maximumOutputTokens:input.maximumOutputTokens,
            preparationSHA256:hash.finalize().map { String(format:"%02x",$0) }.joined(),
            profile:parts.contains(where: { if case .audiovisual = $0.content { return true }; return false })
                ? MiMoV26AudiovisualLayout.profileName : MiMoV26MultimodalProfile.name,
            loadedOwnerIdentity:loadedOwnerIdentity,configurationSHA256:configurationSHA256,templateSHA256:templateSHA256,
            visionGeometryByMediaIndex:Dictionary(uniqueKeysWithValues:parts.enumerated().compactMap { index,item in
                item.geometry.map { (index,$0) }
            }),
            audioPlan:audioPlan,decodedElements:totalDecoded,patchElements:patches,featureElements:featureElements,
            logicalFeatureBytes:try MiMoV26AudioChecked.product([featureElements,configuration.dtype == "float32" ? 4 : 2],"feature bytes"),
            processorIdentity:identity,parts:parts)
    }
}
