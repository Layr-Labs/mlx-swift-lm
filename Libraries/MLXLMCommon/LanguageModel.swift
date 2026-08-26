// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Abstract form of a model that processes language.
public protocol BaseLanguageModel: Module {
    /// Optionally preprocess the weights and modify / remove values as needed.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]

    /// Optionally preprocess the weights with access to safetensor metadata.
    ///
    /// The default implementation forwards to ``sanitize(weights:)``.
    /// Models can override this to inspect metadata (e.g. check `metadata["format"] == "mlx"`)
    /// and skip or customize sanitization accordingly.
    func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray]
}

/// Weight files a model needs that no naming convention or `model.safetensors.index.json`
/// selects.
///
/// A checkpoint can ship weights in a file that neither the conventional `model*.safetensors`
/// names nor its own index cover: `jinaai/jina-reranker-v3-mlx` keeps its reranking head in
/// `projector.safetensors` and maps only the transformer shards in its index, so the head is
/// never read and the model fails to load. The reference implementation has the same gap and
/// closes it the same way -- the checkpoint's `rerank.py` loads that file by name.
///
/// Conform a model to this protocol to name those files. Being explicit rather than widening
/// the selection is what keeps unrelated weights out: a stray tensor whose name a model's
/// `sanitize(weights:)` rewrites is loaded silently rather than reported.
public protocol AdditionalWeightFilesProviding {
    /// File names, relative to the model directory.
    ///
    /// They are loaded after the selected weight files, so a file that is already selected is
    /// not loaded twice, and names that are not present are ignored.
    var additionalWeightFiles: [String] { get }
}

/// Optional metadata a model wants written into converted safetensors.
///
/// Model-specific metadata lets future loaders distinguish transformed MLX-native
/// checkpoints from original upstream checkpoints without relying only on tensor shapes.
public protocol ModelConversionMetadataProvider {
    var modelConversionMetadata: [String: String] { get }
}

extension BaseLanguageModel {
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        sanitize(weights: weights)
    }
}

/// Removes checkpoint tensors owned by an `lm_head` module when the model uses its token
/// embedding as the output projection instead.
///
/// Quantized linear layers carry parameters in addition to `weight` (for example `scales`
/// and `biases`). Filtering by the module path keeps those parameters from being loaded into
/// the absent head. Matching a complete path component also supports weights that have already
/// been namespaced by a wrapper model without affecting similarly named modules.
package func filterLMHeadWeights(
    from weights: [String: MLXArray], tiedWordEmbeddings: Bool
) -> [String: MLXArray] {
    guard tiedWordEmbeddings else { return weights }

    return weights.filter { key, _ in
        !key.split(separator: ".").contains("lm_head")
    }
}

/// Time/Height/Width struct to represent information about input images.
public struct THW: Sendable {

    public let t: Int
    public let h: Int
    public let w: Int

    public init(_ t: Int, _ h: Int, _ w: Int) {
        self.t = t
        self.h = h
        self.w = w
    }

    public var values: (Int, Int, Int) {
        (t, h, w)
    }

    public var product: Int { t * h * w }
}

/// Representation of ``LanguageModel`` input.
///
/// This can contain text (tokens), prepared images (`MLXArray`), or other media as
/// needed. ``LMInput`` is produced by ``UserInputProcessor`` in response
/// to ``UserInput``.
///
/// The ``ModelContext`` holds the ``UserInputProcessor`` associated with a
/// ``LanguageModel``.
public struct LMInput {
    public let text: Text
    public let image: ProcessedImage?
    public let video: ProcessedVideo?
    public let audio: ProcessedAudio?

    /// Representation of tokenized input text.
    public struct Text {

        /// input token array
        public let tokens: MLXArray

        /// optional mask array
        public let mask: MLXArray?

        public init(tokens: MLXArray, mask: MLXArray? = nil) {
            self.tokens = tokens
            self.mask = mask
        }

        public subscript(
            indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default
        ) -> Text {
            Text(tokens: tokens[indices, stream: stream], mask: mask?[indices, stream: stream])
        }

        public subscript(
            text indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default
        ) -> Text {
            Text(tokens: tokens[indices, stream: stream], mask: mask)
        }

        /// Per-batch sequence lengths derived from the optional attention mask.
        public var sequenceLengths: [Int]? {
            if let mask {
                return mask.asType(.int32).sum(axis: -1).asArray(Int.self)
            }
            guard tokens.ndim == 2 else { return nil }
            return Array(repeating: tokens.dim(1), count: tokens.dim(0))
        }

        /// Number of logical sequence positions consumed by one model call.
        /// Batch dimensions do not duplicate the shared cache timeline.
        @inline(__always)
        package var cacheSequenceLength: Int {
            tokens.ndim == 0 ? 0 : tokens.dim(-1)
        }
    }

    /// Representation of prepared input image(s).
    public struct ProcessedImage {

        /// Concatenated pixels from one or more images
        public let pixels: MLXArray
        /// Optional per-patch position ids for encoder-free vision embedders.
        public let positionIds: MLXArray?
        /// Time, height, and width of the images
        public let frames: [THW]?

        public init(
            pixels: MLXArray, positionIds: MLXArray? = nil, frames: [THW]? = nil
        ) {
            self.pixels = pixels
            self.positionIds = positionIds
            self.frames = frames
        }
    }

    /// Representation of prepared input video(s).
    /// For now, this is virtually identical to ProcessedImage.
    public struct ProcessedVideo {

        public let pixels: MLXArray
        public let positionIds: MLXArray?
        public let frames: [THW]?

        public init(
            pixels: MLXArray, positionIds: MLXArray? = nil, frames: [THW]? = nil
        ) {
            self.pixels = pixels
            self.positionIds = positionIds
            self.frames = frames
        }
    }

    /// Representation of prepared audio features.
    public struct ProcessedAudio {
        public let features: MLXArray
        public let mask: MLXArray?

        public init(features: MLXArray, mask: MLXArray? = nil) {
            self.features = features
            self.mask = mask
        }

        public init(samples: MLXArray) {
            self.init(features: samples)
        }
    }

    public init(tokens: MLXArray, mask: MLXArray? = nil) {
        self.init(text: .init(tokens: tokens, mask: mask))
    }

    public init(
        text: LMInput.Text,
        image: LMInput.ProcessedImage? = nil,
        video: LMInput.ProcessedVideo? = nil,
        audio: LMInput.ProcessedAudio? = nil
    ) {
        self.text = text
        self.image = image
        self.video = video
        self.audio = audio
    }
}

/// ``LanguageModel`` step output. This is consumed internally
/// by the ``TokenIterator``.
public struct LMOutput {

    /// logits (one hot vector of probabilities for tokens)
    public let logits: MLXArray

    /// optional ``State`` to carry forward into the next step
    public let state: State?

    /// typed key for use in ``State``
    public struct Key<T>: Identifiable, Sendable {
        public let id: String

        public init(_ id: String) {
            self.id = id
        }
    }

    /// Dictionary of typed ``Key`` to carry state between steps.
    public struct State {
        private var contents: [String: Any]

        public init() {
            self.contents = [:]
        }

        init(serializedArrays: [String: MLXArray]) {
            self.contents = serializedArrays.mapValues { $0 as Any }
        }

        func serializedArrays() throws -> [String: MLXArray] {
            var arrays: [String: MLXArray] = [:]
            for (key, value) in contents {
                guard let array = value as? MLXArray else {
                    throw SerializationError.unsupportedValue(
                        key: key, type: String(describing: type(of: value)))
                }
                arrays[key] = array
            }
            return arrays
        }

        public subscript<T>(_ key: Key<T>) -> T? {
            get {
                contents[key.id] as? T
            }
            set {
                contents[key.id] = newValue
            }
        }

        enum SerializationError: LocalizedError {
            case unsupportedValue(key: String, type: String)

            var errorDescription: String? {
                switch self {
                case .unsupportedValue(let key, let type):
                    "LMOutput.State key '\(key)' contains unsupported value type '\(type)'"
                }
            }
        }
    }

    public init(logits: MLXArray, state: LMOutput.State? = nil) {
        self.logits = logits
        self.state = state
    }
}

/// The result of the call to ``LanguageModel/prepare(_:cache:state:prefill:)``
public enum PrepareResult {
    /// tokens to process by the ``TokenIterator``
    case tokens(LMInput.Text)

    /// logits representing the next token
    case logits(LMOutput)
}

/// Interface for all Language Models (e.g. LLM, VLM).
///
/// The language model is typically called by the ``TokenIterator`` and it:
///
/// - consumes the ``LMInput``
/// - calls ``prepare(_:cache:state:prefill:)`` to initialize the KVCache and consume the prompt
/// - calls ``callAsFunction(_:cache:state:)-9kuvf`` for each token, producing an ``LMOutput``
/// - the ``TokenIterator`` accumulates this information into a ``GenerateResult``
public protocol LanguageModel: BaseLanguageModel, ChatConventionsProviding {

    /// Legacy prepare entry point retained for model and test-double source
    /// compatibility while callers migrate to the stateful/prefill-aware API.
    @available(
        *, deprecated, renamed: "prepare(_:cache:state:prefill:)",
        message: "prefill now defaults to balanced chunking; use the stateful overload"
    )
    func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult

    /// Prepare the cache state and consume the ``LMInput``.
    ///
    /// `state` is the ``LMOutput/state`` a caller carried over from earlier
    /// evaluation against the same `cache` — present when `cache` is already
    /// warm (a multi-turn chat, a tool-call restart, a restored prompt
    /// cache). Models that keep per-call positional state (e.g. the M-RoPE
    /// `ropeDeltas` of the Qwen VLM families) use it to anchor the new
    /// tokens' positions at the cache offset; models without such state can
    /// ignore it. In the typical cold call it is `nil`.
    ///
    /// This can return:
    /// - ``PrepareResult/tokens(_:)`` if the caller should evaluate the (remaining) tokens normally
    /// - ``PrepareResult/logits(_:)`` to produce the next token from the prompt
    ///
    /// Implementations that chunk the prompt should drive the loop with
    /// ``PrefillParameters/forEachChunk(total:reserving:defaultStepSize:maximumStepSize:_:)``,
    /// which owns cancellation, pooling, and per-chunk progress. An
    /// implementation returning `.logits` owns its whole
    /// ``PrefillParameters/progress`` sequence, including the terminal
    /// `(total, total)`; one returning `.tokens` reports only its own chunks —
    /// the iterator that evaluates the remainder completes the sequence.
    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    )
        throws -> PrepareResult

    /// Primary entry point to produce a step (single token) from the model
    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput

    /// Models may implement this simplified interface if they do not produce any ``LMOutput/State``
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray

    /// Create a new array of ``KVCache`` appropriate for this model.
    ///
    /// Implementations must honor ``GenerateParameters/maxKVSize`` for any
    /// attention layer that can be token-windowed. Hybrid attention / state-space
    /// models apply the limit to attention caches only (see
    /// ``makeAttentionKVCache(parameters:)``).
    ///
    /// - Throws: ``KVCacheConfigurationError`` when the request or a model-defined
    ///   cache size is invalid.
    ///
    /// Automatic implementation if self implements ``KVCacheDimensionProvider``.
    func newCache(parameters: GenerateParameters?) throws -> [KVCache]

    /// Authoritative planned status of the cache ``newCache(parameters:)`` produces.
    ///
    /// Use this from ``ModelContainer`` / ``ChatSession`` (or directly on the model)
    /// to inspect topology, requested capacity, and strategy compatibility without
    /// allocating or casting probe caches in application code.
    ///
    /// The default implementation derives the description from ``newCache(parameters:)``.
    /// Models may override with a zero-allocation declarative path, but must stay
    /// consistent with ``newCache(parameters:)``.
    func cacheStatus(parameters: GenerateParameters?) throws -> KVCacheStatus
}

extension LanguageModel {
    /// Default legacy entry point for implementations that provide the newer
    /// stateful/prefill-aware method instead.
    public func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int?
    ) throws -> PrepareResult {
        try prepare(input, cache: cache, state: nil, prefill: .init(stepSize: windowSize))
    }

    /// Compatibility bridge for implementations that still provide only the
    /// legacy `windowSize` entry point.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        try prepare(input, cache: cache, windowSize: prefill.stepSize)
    }

    @available(
        *, deprecated, renamed: "prepare(_:cache:state:prefill:)",
        message:
            "prefill now defaults to balanced chunking; use prefill.chunking = .remainder for the legacy chunk boundaries"
    )
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, windowSize: Int?
    ) throws -> PrepareResult {
        try prepare(input, cache: cache, state: state, prefill: .init(stepSize: windowSize))
    }

    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let logits = callAsFunction(input.tokens, cache: cache)
        return .init(logits: logits)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("callAsFunction(inputs:cache:) not implemented for \(Self.self)")
    }

    /// Default: classify the caches that ``newCache(parameters:)`` constructs.
    ///
    /// Empty caches allocate negligible state (no tensors until the first update),
    /// so this is always consistent with the runtime path. Prefer a declarative
    /// override only when construction itself is expensive.
    public func cacheStatus(parameters: GenerateParameters?) throws -> KVCacheStatus {
        let plan = try parameters?.kvCachePlan() ?? .disabled
        return KVCacheStatus(
            cache: try newCache(parameters: parameters),
            plan: plan,
            phase: .planned)
    }
}

/// Protocol for models that support Multi-Token Prediction (MTP) speculative decoding.
///
/// MTP embeds a lightweight draft head inside the model. After each backbone forward,
/// the head proposes token t+2 from the pre-norm hidden state at t+1. A subsequent
/// 2-token verify forward confirms or rejects the draft in one pass.
///
/// Eligibility: single-sequence batches only (`batchSize == 1`).
/// Throughput (greedy, accept rate p): ~1.74× at p≈1.0, ~1.30× at p≈0.5.
///
/// Port of omlx commit 696d90a:
///   patches/mlx_lm_mtp/qwen35_model.py (_patch_text_model / _patch_outer_model)
///   patches/mlx_lm_mtp/batch_generator.py (_is_mtp_eligible)
public protocol MTPCapable: LanguageModel {
    /// True when the MTP head is attached and operational.
    /// Returns false when the model was loaded without MTP (e.g. no --mtp flag),
    /// in which case `mtpForward` must not be called.
    var hasMTPHead: Bool { get }

    /// Run the MTP head. `hidden` is post-norm hidden from the backbone at position t
    /// (i.e., after `model.norm` — the "post_norm" variant). The MTP head's
    /// `pre_fc_norm_hidden` weights were trained on post-norm inputs.
    /// `nextTokenIds` shape: [1, S] (usually S=1). Returns logits [1, S, vocab].
    /// omlx: TextModel.mtp_forward
    /// MTPLX: hidden_variant="post_norm" (default for all generation commands)
    func mtpForward(hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]) -> MLXArray

    /// Allocate fresh KV caches for the MTP head layers.
    /// Called once per draft proposal — never reused across cycles.
    /// omlx: TextModel.make_mtp_cache
    func makeMTPCache() -> [any KVCache]

    /// Forward pass that also returns pre-norm hidden states.
    /// - Parameters:
    ///   - input: token input for the target forward pass.
    ///   - cache: target model caches advanced by the forward pass.
    ///   - nConfirmed: Confirmed prefix length for the 2-token verify input
    ///     (`0` for a standard forward pass).
    /// - Returns: `(logits [B, S, vocab], preNormHidden [B, S, hiddenSize])`
    ///
    /// The returned hidden is the raw backbone output BEFORE `model.norm`. The MTP head applies
    /// `pre_fc_norm_hidden` itself, so passing post-norm would cause double-normalization.
    /// PR #990: `return out, hidden  # pre-norm hidden for MTP head`
    /// omlx: TextModel.__call__ with return_hidden=True + n_confirmed
    func callWithHidden(input: LMInput.Text, cache: [any KVCache], nConfirmed: Int) -> (
        MLXArray, MLXArray
    )
}

/// Optional protocol that can be implemented by ``LanguageModel`` and will
/// provide an automatic implementation of ``LanguageModel/newCache(parameters:)``
public protocol KVCacheDimensionProvider {
    var kvHeads: [Int] { get }
}

extension LanguageModel where Self: KVCacheDimensionProvider {
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        // Create one cache per layer (kvHeads.count = number of layers)
        // The number of heads per layer (kvHeads[i]) is not used for cache creation
        let numLayers = kvHeads.count
        return try (0 ..< numLayers).map { _ in
            try makeAttentionKVCache(parameters: parameters)
        }
    }

    // Note: do not specialize ``cacheStatus(parameters:)`` here. Hybrid models
    // commonly conform to ``KVCacheDimensionProvider`` while overriding ``newCache``;
    // a kvHeads-based default would mis-report those layouts. The base
    // ``LanguageModel`` implementation classifies the caches ``newCache`` builds.
}
