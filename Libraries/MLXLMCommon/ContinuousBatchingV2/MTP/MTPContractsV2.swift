// MTPContractsV2.swift
//
// ContinuousBatchingV2 — Gemma-4-style MTP (multi-token prediction /
// speculative decoding) integration contracts.
//
// Design (mirrors the KV-less frozen-KV drafter that vLLM and SGLang
// independently converged on for Gemma 4, and our own v1 engine shipped):
//
//   - The DRAFTER writes no KV. Each round it attends a snapshot of the
//     target's KV at exactly TWO layers (the last non-shared full-attention
//     layer and the last non-shared sliding-attention layer), with a
//     CONSTANT query RoPE position per round (the anchor = the absolute
//     position of the row's newest confirmed-but-unfed token).
//   - Per round, for each speculating row: chain k drafter forwards
//     (greedy argmax, seed = newest confirmed token + the target's pre-norm
//     hidden at the position before it), then verify [seed, d_1..d_k] in ONE
//     rectangular [B, 1+k] target forward. The accept-walk
//     (`Gemma4SpeculativeWalk` semantics: accept while target argmax ==
//     draft, always emit target argmax at the first divergence / bonus
//     position) emits a+1 tokens; the k−a rejected tokens are rolled back
//     per row (exact — see `CBv2SequenceKV.supportsSpeculativeWrites`).
//   - GREEDY-ONLY losslessness is the parity invariant: MTP-on output is
//     token-exact vs MTP-off for temperature-0 requests. Non-greedy rows
//     never speculate.
//
// Layering: everything here is model-family-agnostic. The Gemma-4 drafter
// module, its masks, and the accept-walk live in MLXLLM; they reach the
// engine through `CBv2MTPDrafter` / `CBv2MTPForwardable` (same pattern as
// `CBv2EmbeddingForwardable` for multimodal).

import Foundation
import MLX

// MARK: - Model seam (verify forward + capture geometry)

/// Which layer indices the engine snapshots for the drafter's frozen KV.
/// Indices are MODEL layer indices (== positions in the engine's per-layer
/// caches array). Both referenced layers must OWN storage (non-KV-shared).
public struct CBv2MTPCaptureLayers: Sendable, Equatable {
    /// Last non-shared full-attention layer.
    public var full: Int
    /// Last non-shared sliding-attention layer.
    public var sliding: Int
    public init(full: Int, sliding: Int) {
        self.full = full
        self.sliding = sliding
    }
}

/// Model-level surface for `LanguageModel` conformers reached through
/// `CBv2SteppableLanguageModelAdapter` (Gemma4TextModel conforms): the
/// KVCache-shaped twin of the `CBv2MTPSteppableModel` requirements.
public protocol CBv2MTPForwardable: AnyObject {
    /// nil when this model cannot drive MTP (no capture layers).
    var cbv2MTPCaptureLayers: CBv2MTPCaptureLayers? { get }
    /// Forward returning (softcapped) logits [B, L, vocab] AND the pre-norm
    /// last-decoder-layer hidden [B, L, hidden] — the tensor the Gemma-4
    /// drafter was trained against. Must be numerically identical to the
    /// plain forward on the logits side.
    func cbv2ForwardWithHidden(_ tokens: MLXArray, caches: [KVCache])
        -> (logits: MLXArray, lastHidden: MLXArray)
}

extension CBv2MTPForwardable {
    /// Identity may normalize an outer language-model wrapper to the inner
    /// text target that actually owns embeddings, logits, hidden, and KV.
    public var cbv2MTPTargetIdentity: ObjectIdentifier { ObjectIdentifier(self) }
}

/// Recurrent target counterpart to `CBv2MTPForwardable`. Hybrid targets must
/// receive the same request-owned transaction objects used by ordinary CBv2
/// decode; the assistant never owns or aliases these states.
public protocol CBv2RecurrentMTPForwardable:
    AnyObject, CBv2RecurrentLanguageModelForwardable
{
    var cbv2MTPTargetIdentity: ObjectIdentifier { get }
    func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray)
}

extension CBv2RecurrentMTPForwardable {
    public var cbv2MTPTargetIdentity: ObjectIdentifier { ObjectIdentifier(self) }
}

/// Steppable models that can drive MTP rounds. Additive refinement of
/// `CBv2SteppableModel`; the engine speculates only when the bound model
/// conforms AND `mtpCaptureLayers` is non-nil AND a drafter is configured.
public protocol CBv2MTPSteppableModel: CBv2SteppableModel {
    /// nil when the underlying model cannot drive MTP (adapters over
    /// arbitrary models answer at runtime).
    var mtpCaptureLayers: CBv2MTPCaptureLayers? { get }
    /// True only when the target has the recurrent hidden/state transaction
    /// seam required by a request-stateful assistant.
    var supportsRequestStatefulMTP: Bool { get }
    /// Identity of the exact target instance that owns verification logits,
    /// hidden states, and KV. nil means compatibility cannot be proven and
    /// must fail safe to plain decode.
    var mtpTargetIdentity: ObjectIdentifier? { get }
    /// Forward returning logits [B, L, vocab] and pre-norm last hidden
    /// [B, L, hidden]. Same cache/attention semantics as `forward`.
    func forwardWithHidden(tokens: MLXArray, caches: [CBv2AttendingLayerCache])
        -> (logits: MLXArray, lastHidden: MLXArray)
}

extension CBv2MTPSteppableModel {
    public var mtpTargetIdentity: ObjectIdentifier? { nil }
    public var supportsRequestStatefulMTP: Bool { false }
}

/// Engine-facing recurrent hidden-capture refinement. A recurrent MTP target
/// is only activated when this seam and request-owned recurrent state are both
/// present, so no call can silently fall through to the stateless forward.
public protocol CBv2RecurrentMTPSteppableModel:
    CBv2MTPSteppableModel, CBv2RecurrentSteppableModel
{
    func forwardWithHidden(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray)
}

// MARK: - Drafter seam

/// One speculating row's frozen-KV capture for a round: snapshot views of
/// the target's retained KV at the two capture layers, plus the row's
/// anchor geometry. Views are per-row (no padding — the drafter pads and
/// masks internally, so mixed retained lengths across rows are fine).
public struct CBv2MTPRowCapture {
    /// Full-attention capture layer: [1, kvHeads, Tfull, headDim], temporal
    /// order, post-RoPE (captured from storage the target attended).
    public var fullKeys: MLXArray
    public var fullValues: MLXArray
    /// Sliding-attention capture layer: [1, kvHeads, Tslide, headDim],
    /// temporal order. Window-limited by storage eviction.
    public var slidingKeys: MLXArray
    public var slidingValues: MLXArray
    /// Absolute position of the FIRST retained sliding entry (the sliding
    /// KV covers positions [slidingStart, anchor)). The full capture always
    /// starts at 0.
    public var slidingStart: Int
    /// The round's frozen query position: absolute position of the row's
    /// newest confirmed-but-unfed token (== the row's absoluteOffset).
    public var anchor: Int

    public init(
        fullKeys: MLXArray, fullValues: MLXArray,
        slidingKeys: MLXArray, slidingValues: MLXArray,
        slidingStart: Int, anchor: Int
    ) {
        self.fullKeys = fullKeys
        self.fullValues = fullValues
        self.slidingKeys = slidingKeys
        self.slidingValues = slidingValues
        self.slidingStart = slidingStart
        self.anchor = anchor
    }
}

/// Opaque round-scoped state a drafter builds once per round from the
/// per-row captures (padded/stacked batch KV, per-row masks, positions).
public protocol CBv2MTPPreparedCapture: AnyObject {}

/// The engine's view of a drafter. Implemented in MLXLLM by an adapter over
/// `Gemma4AssistantDraftModel` bound to the engine's target model (the
/// adapter owns target-embedding lookup, mask construction, and greedy
/// argmax). All methods are called on the engine thread while building the
/// step graph; they MUST NOT force evaluation (no host syncs).
public protocol CBv2MTPDrafter: AnyObject {
    /// Identity of the exact target instance whose embeddings and geometry
    /// this drafter consumes. nil means compatibility cannot be proven and
    /// must fail safe to plain decode.
    var mtpTargetIdentity: ObjectIdentifier? { get }
    /// A drafter may narrow target verification, depth, or batch policy for
    /// correctness. nil preserves the Gemma/default engine configuration.
    var requiredVerificationMode: CBv2MTPVerificationMode? { get }
    var maximumDraftTokens: Int? { get }
    var maximumSpeculativeBatch: Int? { get }
    /// Variable request-owned residency outside target KV. Admission charges
    /// this conservatively for every reserved token when the drafter is active.
    var requestStateBytesPerToken: Int { get }
    /// Build round-scoped batch state from per-row captures. `rows` order
    /// == the round's speculating-row order.
    func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture
    /// One draft-chain step over all speculating rows.
    ///  - tokens: [B, 1] int32 (lazy) — seed tokens (round start: each
    ///    row's newest confirmed token; later steps: previous draft).
    ///  - hidden: [B, 1, H] — round start: the target's pre-norm hidden at
    ///    the position BEFORE the seed token; later steps: the drafter's
    ///    own previous output hidden.
    ///  - Returns greedy next-token ids [B] int32 (lazy) and the drafter's
    ///    output hidden [B, 1, H] for chaining.
    func draftStep(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture
    ) -> (tokens: MLXArray, hidden: MLXArray)
}

extension CBv2MTPDrafter {
    public var mtpTargetIdentity: ObjectIdentifier? { nil }
    public var requiredVerificationMode: CBv2MTPVerificationMode? { nil }
    public var maximumDraftTokens: Int? { nil }
    public var maximumSpeculativeBatch: Int? { nil }
    public var requestStateBytesPerToken: Int { 0 }
}

/// Opaque, request-owned assistant state. It is deliberately distinct from
/// target recurrent state and target attention KV.
public protocol CBv2MTPRequestState: AnyObject {
    var committedInputCount: Int { get }
    var stagedInputCount: Int { get }
}

/// Alternate drafter seam for autoregressive assistants such as Qwen3.5/3.6.
/// Calls are row-local so histories and assistant-cache offsets may differ;
/// the target verifier may still batch the resulting columns as `[B,1]`.
public protocol CBv2MTPRequestStatefulDrafter: CBv2MTPDrafter {
    func makeRequestState() -> any CBv2MTPRequestState
    func draftStep(
        tokens: MLXArray, hidden: MLXArray,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray)
    /// Device arrays that make assistant-cache mutation part of the round's
    /// evaluation fence.
    func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray]
    /// Complete one staged round. The drafter stages the seed plus every
    /// proposal it consumes while chaining; `confirmedInputTokens` is the
    /// exact prefix of those inputs that became canonical target history.
    func finalizeRound(
        requestState: any CBv2MTPRequestState, confirmedInputTokens: Int)
    /// Reject/discard all assistant writes made by the current round.
    func discardRound(requestState: any CBv2MTPRequestState)
    /// Explicitly sever device-array ownership on finish/cancel/preemption.
    func releaseRequestState(_ requestState: any CBv2MTPRequestState)
}

// MARK: - Config

/// How the target scores one MTP draft chain.
///
/// `serialTarget` is the correctness baseline: every column uses the same
/// `[B, 1]` eager target forward as ordinary decode. It works independently
/// of chip-specific multi-position kernel numerics. `rectangular` is an
/// explicit optimization that scores all `1+k` columns in one `[B, 1+k]`
/// forward and therefore requires separate numerical certification.
public enum CBv2MTPVerificationMode: String, Sendable, Equatable {
    case serialTarget = "serial_target"
    case rectangular
    case automatic
}

/// Engine-level MTP configuration (parallel to `CBv2CompiledDecodeConfig`).
public struct CBv2MTPConfig: Sendable {
    /// The largest draft depth covered by the production rectangular-shape
    /// validation matrix (`verify width = 1 + k`, widths 1...8).
    public static let testedMaxDraftTokens = 7
    /// CBv2's production rectangular batch ceiling.
    public static let testedMaxSpeculativeBatch = 8

    /// Master switch. The engine also requires a drafter instance and a
    /// conforming model; `enabled == true` without both is inert.
    public var enabled: Bool
    /// Max draft tokens per round (k). Rounds verify 1+k and emit 1...k+1.
    /// Clamped to the production-tested `0...7` range.
    public var maxDraftTokens: Int
    /// Optional deterministic override. nil selects the adaptive controller;
    /// a value selects a fixed step-global depth, clamped to
    /// `0...maxDraftTokens`. Fixed zero is an explicit target-only mode that
    /// keeps MTP construction and metrics active for bring-up.
    public var fixedDraftTokens: Int?
    /// Hard operational gate on decode rows in one plan, clamped to 1...8.
    /// Together with the k<=7 bound this caps staged window-KV to 64 token
    /// rows per storage-owning layer and one in-flight step. The adaptive
    /// controller is separately keyed by a planned-decode-row bucket.
    public var maxSpeculativeBatch: Int
    /// Target scoring strategy. Automatic verification is the safe default:
    /// it uses rectangular scoring only within the configured work envelope
    /// and otherwise clamps depth before draft work. Serial target scoring
    /// remains an explicit correctness fallback.
    public var verificationMode: CBv2MTPVerificationMode
    /// Maximum `batch * (1+k)` target rows eligible for automatic
    /// rectangular verification. The planner clamps larger work to a safe
    /// depth, including ordinary target-only decode when no positive depth
    /// fits. Defaults to ZERO: a positive envelope is the integrator's
    /// explicit claim that rectangular target evaluation is argmax-exact for
    /// the deployed chip/OS/MLX/model tuple at every shape inside it. With
    /// no envelope, automatic mode performs no speculative work. Ignored by
    /// explicit serial/rectangular modes.
    public var maxAutomaticRectangularTokens: Int

    /// Process-level kill switch: `DARKBLOOM_CBV2_MTP=0/false/no/off`
    /// disables MTP even when the provider enables it (same convention as
    /// `DARKBLOOM_CBV2_COMPILED`). Unset or any other value: no override.
    public static let envEnabled: Bool = {
        if let raw = ProcessInfo.processInfo.environment["DARKBLOOM_CBV2_MTP"] {
            return !["0", "false", "no", "off"].contains(raw.lowercased())
        }
        return true
    }()

    public init(
        enabled: Bool = false,
        maxDraftTokens: Int = Self.testedMaxDraftTokens,
        maxSpeculativeBatch: Int = 8,
        fixedDraftTokens: Int? = nil,
        verificationMode: CBv2MTPVerificationMode = .automatic,
        maxAutomaticRectangularTokens: Int = 0
    ) {
        self.enabled = enabled
        let resolvedMax = min(max(maxDraftTokens, 0), Self.testedMaxDraftTokens)
        self.maxDraftTokens = resolvedMax
        self.maxSpeculativeBatch = min(
            max(maxSpeculativeBatch, 1), Self.testedMaxSpeculativeBatch)
        self.fixedDraftTokens = fixedDraftTokens.map {
            min(max($0, 0), resolvedMax)
        }
        self.verificationMode = verificationMode
        self.maxAutomaticRectangularTokens = max(0, maxAutomaticRectangularTokens)
    }

    /// The effective on/off state (config AND env kill switch).
    public var effectiveEnabled: Bool { enabled && Self.envEnabled }
}

// MARK: - Metrics

/// One controller wall-cost input exposed in a lock-safe metrics snapshot.
/// Cost is measured at the existing finalize boundary; collecting it adds no
/// MLX evaluation or tensor readback.
public struct CBv2MTPCostInput: Sendable, Equatable {
    public var decodeRowBucket: Int
    public var depth: Int
    public var samples: Int
    public var ewmaWallTimeNanos: UInt64
    public var totalWallTimeNanos: UInt64

    public init(
        decodeRowBucket: Int, depth: Int, samples: Int,
        ewmaWallTimeNanos: UInt64, totalWallTimeNanos: UInt64
    ) {
        self.decodeRowBucket = decodeRowBucket
        self.depth = depth
        self.samples = samples
        self.ewmaWallTimeNanos = ewmaWallTimeNanos
        self.totalWallTimeNanos = totalWallTimeNanos
    }
}

/// Cumulative MTP counters (engine-thread mutated, snapshot under the
/// engine's stats lock). Per-position acceptance is the tuning signal for
/// `maxDraftTokens`.
public struct CBv2MTPMetrics: Sendable {
    /// True for every non-nil `EngineV2.mtpMetricsSnapshot()`. Kept explicit
    /// so provider telemetry can serialize one stable shape.
    public var active: Bool = true
    /// Target scoring strategy used by every round in this engine.
    public var verificationMode: CBv2MTPVerificationMode = .automatic
    /// Configured automatic rectangular work cap, exposed so benchmark
    /// validators can distinguish intentional target-only fallback from a
    /// failure to run the requested fixed depth.
    public var maxAutomaticRectangularTokens: Int = 0
    /// Actual target-verification rounds by strategy. Automatic mode can use
    /// both across batch/depth regimes.
    public var rectangularVerificationRounds: Int = 0
    public var serialVerificationRounds: Int = 0
    /// Most recently selected step-global depth (zero means target-only).
    public var selectedDepth: Int = 0
    /// Planned decode-row bucket used for the most recent selection.
    public var decodeRowBucket: Int = 0
    /// Rounds that drafted (k ≥ 1) and verified.
    public var rounds: Int = 0
    /// Seed steps (eligible rows that decoded eagerly with hidden capture
    /// to establish the drafter carry — no drafts yet).
    public var seedSteps: Int = 0
    /// Total draft tokens proposed across all rounds.
    public var draftedTokens: Int = 0
    /// Total draft tokens accepted across all rounds.
    public var acceptedTokens: Int = 0
    /// Total tokens emitted by MTP rounds (accepted + bonus/correction).
    public var emittedTokens: Int = 0
    /// perPositionAccepted[i] = rounds in which draft position i (0-based)
    /// was accepted. Monotonically non-increasing over i within a run.
    public var perPositionAccepted: [Int] = []
    /// Rows that were round-eligible but clamped to plain decode, keyed by
    /// reason ("batch_gate", "kv_headroom", "carry_invalid", ...).
    public var skippedRows: [String: Int] = [:]
    /// Step selections by depth, including depth zero.
    public var depthSelections: [Int: Int] = [:]
    /// Stable controller/fallback reasons (warmup, exploration, hysteresis,
    /// unprofitable depth zero, batch gate, token/KV headroom, and so on).
    public var controllerFallbacks: [String: Int] = [:]
    /// Conditional acceptance rate at each draft position: P(position i is
    /// accepted | every earlier draft position was accepted).
    public var conditionalAcceptance: [Double] = []
    /// Outlier-clamped wall-cost EWMAs and raw cumulative inputs, sorted by
    /// decode-row bucket then depth in snapshots.
    public var costInputs: [CBv2MTPCostInput] = []
    /// Sum of measured wall time for cost-eligible speculative rounds.
    public var totalRoundWallTimeNanos: UInt64 = 0

    public init() {}

    /// Preferred proposal-count spelling. `draftedTokens` remains stored for
    /// compatibility with the first engine/provider seam.
    public var proposedTokens: Int { draftedTokens }

    /// Mean accepted drafts per round (nil before any round).
    public var meanAcceptedPerRound: Double? {
        rounds > 0 ? Double(acceptedTokens) / Double(rounds) : nil
    }
}
