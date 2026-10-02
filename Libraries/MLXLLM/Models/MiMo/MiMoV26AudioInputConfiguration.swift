// Copyright © 2026 Eigen Labs.
// HF5711 codec; pinned SGLang67/Torchaudio2.11 PCM frontend. No MLX dependency.
import Foundation

public enum MiMoV26AudioInputError: Error, Equatable, Sendable {
    case configuration(String)
    case input(String)
    case limit(String)
    case weights(String)
    case weightsNotLoaded, cancelled, nonfiniteResult
}

/// Input codec, distinct from MiMoV26AudioConfiguration (the patch encoder).
public struct MiMoV26AudioInputConfiguration: Equatable, Sendable {
    public static let codecProfile = "hf5711-bf16-registered-rope-rvq-v1"
    public static let frontendProfile = "sglang67-torchaudio2.11-pcm-default-f32-v1"
    public let rawFields: [String: MiMoV26JSONValue]
    public let hiddenSize, heads, headDim, layers, ffnSize, skipLayerIndex: Int
    public let localLookBack, quantizers: Int
    public let codebookSizes: [Int]
    public let sampleRate, fftSize, hopLength, windowSize, melBands: Int
    public let segmentSize, groupSize, maximumGroupValidMelFrames: Int
    public let ropeTheta: Float
    public let maxAudioSecondsMetadata: Int
    public let audioStartToken, audioToken, audioEndToken: Int
    let synthetic: Bool

    public init(sidecarJSON: Data, mainConfiguration: MiMoV26Configuration) throws {
        guard !sidecarJSON.isEmpty, sidecarJSON.count <= 1 << 20 else {
            throw MiMoV26AudioInputError.limit("audio sidecar config bytes")
        }
        let fields = try JSONDecoder().decode([String: MiMoV26JSONValue].self, from: sidecarJSON)
        try self.init(fields: fields, synthetic: false)
        guard let audio = mainConfiguration.audio, let p = mainConfiguration.processorFields,
            audio.channels == quantizers, audio.groupSize == groupSize,
            audio.segmentSize == segmentSize
        else {
            throw MiMoV26AudioInputError.configuration("main audio-patch/codec contract mismatch")
        }
        for (name, value) in [
            "audio_avg_pooler": 2, "audio_channels": quantizers, "audio_group_size": groupSize,
            "audio_hop_length": hopLength, "audio_kernel_size": 3, "audio_n_mels": melBands,
            "audio_nfft": fftSize,
            "audio_sampling_rate": sampleRate, "audio_segment_size": segmentSize,
            "audio_stride_size": 2,
            "audio_window_size": windowSize, "audio_start_token_id": audioStartToken,
            "audio_token_id": audioToken, "audio_end_token_id": audioEndToken, "audio_fmin": 0,
        ] {
            guard try Self.integer(p, name) == value else {
                throw MiMoV26AudioInputError.configuration(name)
            }
        }
        guard p["audio_fmax"] == .null,
            mainConfiguration.tokenIDs["audio_start_token_id"] == audioStartToken,
            mainConfiguration.tokenIDs["audio_token_id"] == audioToken,
            mainConfiguration.tokenIDs["audio_end_token_id"] == audioEndToken
        else {
            throw MiMoV26AudioInputError.configuration(
                "main audio token/frontend metadata mismatch")
        }
    }

    public func encodedSidecar() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(rawFields)
    }
    public func usesLocalAttention(layer: Int) -> Bool { layer % 2 == 0 }
    public func allowsAttention(query: Int, key: Int, layer: Int) -> Bool {
        key >= 0 && query >= key
            && (!usesLocalAttention(layer: layer) || query - key <= localLookBack)
    }

    static func integer(_ fields: [String: MiMoV26JSONValue], _ key: String) throws -> Int {
        guard case .number(let value) = fields[key] else {
            throw MiMoV26AudioInputError.configuration(key)
        }
        let number = NSDecimalNumber(decimal: value)
        let integer = number.int64Value
        guard Decimal(integer) == value, let result = Int(exactly: integer) else {
            throw MiMoV26AudioInputError.configuration(key + " is not a representable integer")
        }
        return result
    }
    private init(fields: [String: MiMoV26JSONValue], synthetic: Bool) throws {
        self.rawFields = fields
        self.synthetic = synthetic
        func int(_ key: String) throws -> Int { try Self.integer(fields, key) }
        hiddenSize = try int("d_model")
        heads = try int("encoder_attention_heads")
        layers = try int("encoder_layers")
        ffnSize = try int("encoder_ffn_dim")
        let skip = try int("encoder_skip_layer_id")
        guard hiddenSize > 0, hiddenSize <= Int(Int32.max), heads > 0,
            hiddenSize % heads == 0, (hiddenSize / heads) % 2 == 0,
            layers > 0, layers <= 1024, ffnSize > 0, ffnSize <= Int(Int32.max),
            (1 ... layers).contains(skip)
        else { throw MiMoV26AudioInputError.configuration("codec dimensions") }
        headDim = hiddenSize / heads
        skipLayerIndex = skip - 1
        guard case .array(let window) = fields["encoder_attn_window_size"], window.count == 2,
            case .number(let left) = window[0], case .number(let right) = window[1], right == 0,
            left >= 0, left <= Decimal(Int32.max),
            Decimal(NSDecimalNumber(decimal: left).intValue) == left
        else {
            throw MiMoV26AudioInputError.configuration("encoder_attn_window_size")
        }
        localLookBack = NSDecimalNumber(decimal: left).intValue
        guard fields["encoder_causal"] == .bool(true), fields["hybrid_attention"] == .bool(true),
            try int("swa_per_block") == 2, try int("hybrid_block_size") == 8,
            fields["scale_embedding"] == .bool(false),
            fields["activation_function"] == .string("gelu"),
            fields["position_embedding_type"] == .string("rope"),
            fields["rope_type"] == .string("default"),
            fields["ln_type"] == .string("LayerNorm"), try int("kernel_size") == 3,
            try int("stride_size") == 2, try int("avg_pooler") == 2
        else {
            throw MiMoV26AudioInputError.configuration("unsupported HF5711 codec semantics")
        }
        let theta = try int("rope_theta")
        guard theta == 10000 else { throw MiMoV26AudioInputError.configuration("rope_theta") }
        ropeTheta = Float(theta)
        quantizers = try int("num_quantizers")
        guard case .array(let bins) = fields["codebook_size"], bins.count == quantizers,
            quantizers > 0
        else {
            throw MiMoV26AudioInputError.configuration("codebook_size")
        }
        codebookSizes = try bins.map { value in
            let n = try Self.integer(["value": value], "value")
            guard n > 1, n <= 1024 else {
                throw MiMoV26AudioInputError.configuration("codebook range")
            }
            return n
        }
        sampleRate = try int("sampling_rate")
        fftSize = try int("nfft")
        hopLength = try int("hop_length")
        windowSize = try int("window_size")
        melBands = try int("n_mels")
        maxAudioSecondsMetadata = try int("max_audio_seconds")  // metadata, not an execution cap
        guard sampleRate == 24000, fftSize == 960, hopLength == 240, windowSize == 960,
            melBands > 0, melBands <= 128, maxAudioSecondsMetadata > 0,
            try int("fmin") == 0, fields["fmax"] == .null
        else {
            throw MiMoV26AudioInputError.configuration("frontend geometry")
        }
        segmentSize = 6000
        groupSize = 4
        maximumGroupValidMelFrames = 256000
        audioStartToken = 151673
        audioToken = 151669
        audioEndToken = 151674
        if !synthetic {
            guard hiddenSize == 1024, heads == 16, layers == 24, ffnSize == 4096, skip == 3,
                localLookBack == 128, melBands == 128, quantizers == 20,
                codebookSizes == [1024, 1024, 256] + Array(repeating: 128, count: 17)
            else {
                throw MiMoV26AudioInputError.configuration(
                    "not selected MiMo V2.6 input codec geometry")
            }
        }
    }

    /// Internal fixture construction executes the same algorithm at tiny shapes;
    /// no public loader/serving path accepts this as the selected artifact.
    static func fixture(
        hiddenSize: Int = 8, heads: Int = 2, layers: Int = 4,
        melBands: Int = 2, bins: [Int] = [4, 4], lookBack: Int = 2
    ) throws -> Self {
        guard hiddenSize > 0, hiddenSize <= Int(Int32.max) / 2 else {
            throw MiMoV26AudioInputError.configuration("fixture width")
        }
        var f: [String: MiMoV26JSONValue] = [
            "encoder_causal": .bool(true), "hybrid_attention": .bool(true),
            "scale_embedding": .bool(false), "activation_function": .string("gelu"),
            "position_embedding_type": .string("rope"),
            "rope_type": .string("default"), "ln_type": .string("LayerNorm"), "fmax": .null,
        ]
        for (key, value) in [
            "d_model": hiddenSize, "encoder_attention_heads": heads, "encoder_layers": layers,
            "encoder_ffn_dim": hiddenSize * 2, "encoder_skip_layer_id": min(3, layers),
            "swa_per_block": 2, "hybrid_block_size": 8,
            "kernel_size": 3, "stride_size": 2, "avg_pooler": 2, "rope_theta": 10000,
            "num_quantizers": bins.count,
            "sampling_rate": 24000, "nfft": 960, "hop_length": 240, "window_size": 960,
            "n_mels": melBands,
            "max_audio_seconds": 300, "fmin": 0,
        ] { f[key] = .number(Decimal(value)) }
        f["codebook_size"] = .array(bins.map { .number(Decimal($0)) })
        f["encoder_attn_window_size"] = .array([.number(Decimal(lookBack)), .number(0)])
        return try .init(fields: f, synthetic: true)
    }
}
