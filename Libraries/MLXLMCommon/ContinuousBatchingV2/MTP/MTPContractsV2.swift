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

/// Additive target capability for extracting exact top-two policy evidence
/// from logits without forcing device evaluation.
public protocol CBv2MTPPolicyTopTwoProviding: AnyObject {
    func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray)
}

/// Runtime availability for type-erased model adapters whose static wrapper
/// type cannot express whether the wrapped target implements top-two.
public protocol CBv2MTPPolicyTopTwoCapabilityProviding: AnyObject {
    var cbv2MTPPolicyTopTwoAvailable: Bool { get }
}

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

/// Capture-verify refinement of `CBv2RecurrentMTPForwardable` (MTPLX GDN
/// capture-commit pattern): ONE forward over the whole `[B, 1+k]` verify
/// window during which every recurrent layer stages per-position captured
/// conv/SSM stacks via `CBv2RecurrentStateEvaluation.stageCaptured`. Commit
/// selects the captured state at the accepted position on device; rollback
/// keeps the pre-verify committed state. Attention KV rolls back by trim as
/// usual, so no repair forward is ever needed.
public protocol CBv2RecurrentCaptureMTPForwardable: CBv2RecurrentMTPForwardable {
    /// Same contract as `cbv2ForwardWithHidden`, but each recurrent
    /// transaction receives captured `[L, ...]` per-position stacks
    /// (`L == tokens.dim(1)`) instead of a single final state.
    func cbv2ForwardWithHiddenCaptured(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray)
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
    /// True only when `forwardWithHiddenCaptured` stages per-position
    /// captured recurrent stacks (MTP capture-verify). Without it,
    /// rectangular verification over a recurrent target is impossible and
    /// the driver falls back to the serial oracle.
    var supportsCapturedVerifyWindow: Bool { get }
    /// Capture-verify forward over the whole verify window. Only called when
    /// `supportsCapturedVerifyWindow == true`.
    func forwardWithHiddenCaptured(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray)
}

extension CBv2RecurrentMTPSteppableModel {
    /// Fail-safe defaults for first-generation recurrent targets: no
    /// captured-window support (the serial oracle remains the verify path).
    public var supportsCapturedVerifyWindow: Bool { false }

    public func forwardWithHiddenCaptured(
        tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        preconditionFailure(
            "CBv2 capture-verify forward called on a model without captured-window support")
    }
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
    /// True when this drafter's rounds may accept via target-prefix
    /// pre-sampling; see the extension default for the full contract.
    var supportsTargetPrefixAcceptance: Bool { get }
    /// Variable request-owned residency outside target KV. Admission charges
    /// this conservatively for every reserved token when the drafter is active.
    var requestStateBytesPerToken: Int { get }
    /// Physical allocation granularity of the request-owned state, in tokens.
    /// A value greater than one makes admission charge whole allocation blocks.
    var requestStateTokenGranularity: Int { get }
    /// Maximum physical high-water tokens retained beyond committed logical state.
    var requestStateTokenAllocationPadding: Int { get }
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
    /// How many DISTINCT candidates one drafter forward can return at a
    /// position. 1 means argmax only, which is what a chain needs and what
    /// every drafter supports; a tree shape asking for rank `r` requires
    /// `maximumDraftCandidates > r`. See `CBv2MTPTreeShape`.
    var maximumDraftCandidates: Int { get }
    /// One draft-chain step that also returns the runners-up of the SAME
    /// forward. `candidates` is the number of ranks wanted (>= 1).
    ///  - Returns `tokens` `[B, candidates]` int32 (lazy), column 0 being
    ///    the greedy argmax so `tokens[.., 0]` is bit-identical to
    ///    `draftStep`'s result, and the drafter's output hidden `[B, 1, H]`
    ///    for chaining the ARGMAX branch (a tree's alternates are leaves of
    ///    the forward they came from; nothing chains from a runner-up).
    func draftStepCandidates(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture,
        candidates: Int
    ) -> (tokens: MLXArray, hidden: MLXArray)
}

extension CBv2MTPDrafter {
    /// Default: argmax only. A drafter that does not override this cannot
    /// carry a tree, and `CBv2MTPTreeShape.maximumCandidateRank` is the
    /// number the caller must check against.
    public var maximumDraftCandidates: Int { 1 }

    /// Default: satisfy a rank-1 request from `draftStep`, refuse anything
    /// deeper rather than inventing a candidate.
    public func draftStepCandidates(
        tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture,
        candidates: Int
    ) -> (tokens: MLXArray, hidden: MLXArray) {
        precondition(
            candidates == 1,
            "CBv2MTPDrafter: \(type(of: self)) returns only its argmax; "
                + "a tree shape needs draftStepCandidates")
        let step = draftStep(tokens: tokens, hidden: hidden, prepared: prepared)
        return (step.tokens.reshaped([-1, 1]), step.hidden)
    }

    public var mtpTargetIdentity: ObjectIdentifier? { nil }
    public var requiredVerificationMode: CBv2MTPVerificationMode? { nil }
    public var maximumDraftTokens: Int? { nil }
    public var maximumSpeculativeBatch: Int? { nil }
    public var requestStateBytesPerToken: Int { 0 }
    public var requestStateTokenGranularity: Int { 1 }
    public var requestStateTokenAllocationPadding: Int { 0 }
    /// True when this drafter's rounds may accept via target-prefix
    /// pre-sampling (accept draft iff it equals a token pre-sampled from the
    /// target's real per-request sampler distribution; the committed token is
    /// always that target sample — exact for the output distribution at ANY
    /// temperature/top-p/top-k). Lifts the engine's `temperature == 0`
    /// eligibility gate when the installed sampler also supports MTP verify
    /// sampling. Default false: greedy argmax acceptance only.
    public var supportsTargetPrefixAcceptance: Bool { false }
}

/// Opaque, request-owned assistant state. It is deliberately distinct from
/// target recurrent state and target attention KV.
public protocol CBv2MTPRequestState: AnyObject {
    var committedInputCount: Int { get }
    var stagedInputCount: Int { get }
    /// Actual materialized device-array residency currently owned by this state.
    var materializedBytes: Int { get }
}

extension CBv2MTPRequestState {
    public var materializedBytes: Int { 0 }
}

/// Trusted target inputs and their corresponding pre-norm hidden rows.
/// Both arrays remain lazy and device-resident until the engine's existing
/// finalize fence.
public struct CBv2MTPCommittedTargetObservation {
    public let tokens: MLXArray
    public let hidden: MLXArray

    public init(tokens: MLXArray, hidden: MLXArray) {
        self.tokens = tokens
        self.hidden = hidden
    }
}

/// Alternate drafter seam for autoregressive assistants such as Qwen3.5/3.6.
/// Calls are row-local so histories and assistant-cache offsets may differ;
/// the target verifier may still batch the resulting columns as `[B,1]`.
public protocol CBv2MTPRequestStatefulDrafter: CBv2MTPDrafter {
    func makeRequestState() -> any CBv2MTPRequestState
    /// Queue trusted target inputs/hidden rows without evaluating them.
    func observeCommittedTarget(
        _ observation: CBv2MTPCommittedTargetObservation,
        requestState: any CBv2MTPRequestState)

    /// Draft-head shortlist opt-in. Non-nil K asks target verification to
    /// additionally surface each verify position's top-K token ids and
    /// their probability mass; finalize threads the accepted position's ids
    /// into the next round's carry when the mass clears the engine coverage
    /// threshold. nil keeps full-head draft scoring and adds zero verify
    /// work. Default nil.
    var draftShortlistSize: Int? { get }
    /// One request-local draft proposal.
    /// - shortlist: [K] int32 target top-K token ids captured at the carry
    ///   position, non-nil only when `draftShortlistSize` is set AND the
    ///   captured mass cleared the coverage threshold. When present the
    ///   drafter MUST propose a token from these ids (it may score only the
    ///   matching head rows instead of the full vocabulary).
    func draftStep(
        tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?,
        requestState: any CBv2MTPRequestState
    ) -> (tokens: MLXArray, hidden: MLXArray)
    /// Device arrays that make assistant-cache mutation part of the round's
    /// evaluation fence.
    func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray]
    /// Complete one staged round. The drafter stages the seed plus every
    /// proposal it consumes while chaining; `confirmedInputTokens` is the
    /// exact prefix of those inputs that became canonical target history.
    func finalizeRound(
        requestState: any CBv2MTPRequestState,
        confirmedInputTokens: Int,
        committedDraftTokens: MLXArray,
        committedTargetHidden: MLXArray)
    /// Reject/discard all assistant writes made by the current round.
    func discardRound(requestState: any CBv2MTPRequestState)
    /// Explicitly sever device-array ownership on finish/cancel/preemption.
    func releaseRequestState(_ requestState: any CBv2MTPRequestState)
}

extension CBv2MTPRequestStatefulDrafter {
    public var draftShortlistSize: Int? { nil }
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
    /// One captured rectangular target transaction whose shape-sensitive
    /// arithmetic is explicitly M1-equivalent at the model boundary.
    case rectangularExact = "rectangular_exact"
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
/// Round-cost switches. Each is one behaviour, off by default, readable at
/// process start so a stack test can enable exactly the arms it is measuring.
///
/// [engage] MTPLX_MTP_PIPELINED_DRAFT_SUBMIT
public enum CBv2MTPRoundSwitches {
    /// Submit each stateless draft step as it is built instead of leaving the
    /// whole round unsubmitted until `executeMTPRound`'s single `asyncEval`.
    ///
    /// UNCONDITIONAL BY DESIGN. It submits at every depth, every batch size and
    /// every context length, because the thing that decides whether the extra
    /// submission pays — how expensive one drafter forward is — is not
    /// something a prompt length can be used to guess. This drafter attends the
    /// target's retained KV, so its forward does grow with context, and it
    /// would be easy to "help" by gating the submit on prompt length. Do not.
    /// A length-gated submit makes every measurement a measurement of the gate
    /// and mis-serves every length nobody tuned. If it loses somewhere, the
    /// answer is to turn the switch off, not to add a threshold.
    ///
    /// The stateless path (Gemma 4) builds k drafter forwards AND the
    /// rectangular verify graph with nothing queued for the GPU, then submits
    /// once. Every one of those builds is host time the GPU spends idle. The
    /// request-stateful path already publishes its first generation mid-build
    /// for a related reason ("nonblocking and joins the round's sole finalize
    /// fence"); this is the same move, per step, for the stateless drafter.
    ///
    /// Safety: `asyncEval` is non-blocking and the same arrays still ride the
    /// round's finalize fence, so the emitted stream is unchanged. Evaluating
    /// the drafter early can only materialize the pre-write KV capture views
    /// EARLIER than the verify's writes, never later, and on contiguous
    /// storage the capture (`..<absoluteOffset`) and the verify's writes
    /// (`absoluteOffset...`) are disjoint ranges of the same buffer.
    public static let pipelinedDraftSubmit = flag("MTPLX_MTP_PIPELINED_DRAFT_SUBMIT")

    /// [engage] MTPLX_MTP_TREE_DRAFT=<shape>
    ///
    /// Spend part of the rectangular WIDTH budget on siblings of the draft
    /// chain instead of on more chain. nil (unset, `off`) keeps the shipped
    /// chain exactly. See `CBv2MTPTreeShape` for the grammar, the column
    /// order and — before enabling this — the arithmetic that says an
    /// alternate only beats one more chain column when per-position
    /// acceptance is at or below roughly 0.72.
    ///
    /// SHAPE, NOT LENGTH. The value names a tree, never a prompt size. A
    /// shape too wide for the plan's width budget is SHORTENED
    /// (`CBv2MTPTreeShape.clamped`), never silently widened: the rectangular
    /// cap is a correctness certification.
    public static let treeDraft: CBv2MTPTreeShape? = {
        guard let raw = ProcessInfo.processInfo.environment["MTPLX_MTP_TREE_DRAFT"] else {
            return nil
        }
        return CBv2MTPTreeShape.parse(raw)
    }()

    /// [engage] MTPLX_MTP_FUSED_VERIFY_ATTENTION
    ///
    /// Score the whole verify rectangle in ONE attention call per layer
    /// instead of one call per column.
    ///
    /// Rectangular verification currently sets
    /// `mtpSerializesRectangularAttention`, which sends a `[1, 1+k]` verify
    /// through `CBv2AttentionV1.attendQueryBlocks` at `blockSize: 1`: `1+k`
    /// separate SDPA calls, each slicing `keys[visibleStart ..< historyCount
    /// + offset + 1]`. On a full-attention layer `visibleStart` is 0, so
    /// EVERY column re-reads the entire retained KV. The round therefore
    /// pays the target's long-context attention `1+k` times, and the per-
    /// draft cost `c` grows with context instead of being amortised across
    /// the rectangle — the opposite of what a rectangular verify is for.
    ///
    /// With this on, the verify takes the ordinary `L > 1` path, whose
    /// `maskMode` is `.causal` for a full layer and the existing
    /// causal-and-window array mask for a windowed one. That is the SAME
    /// visibility per column, computed once: mathematically identical, and
    /// the reason it is a switch rather than the default is that it is a
    /// different KERNEL, so it is not bit-identical to the serialized path
    /// the M5 rectangular envelope was certified on. Greedy parity is the
    /// gate; run it with the benchmark's parity recording before promoting.
    ///
    /// Contiguous storage only. `PagedLayerCache` branches on the same flag
    /// and its non-serialized `L > 1` path is packed-prefill masking over
    /// gathered pages, which is a separate certification; a paged bank keeps
    /// serializing whatever this says.
    public static let fusedVerifyAttention = flag("MTPLX_MTP_FUSED_VERIFY_ATTENTION")

    private static func flag(_ key: String) -> Bool {
        guard let raw = ProcessInfo.processInfo.environment[key] else { return false }
        return ["1", "true", "yes", "on"].contains(raw.lowercased())
    }
}

/// Per-stage HOST timing for MTP rounds, summed over every round in the run.
///
/// The stages are cut where the GPU's view of the round changes, because the
/// thing that decides whether speculation pays is not the total round wall
/// clock but how much of it the GPU spent idle. MTP rounds never chain (see
/// `CBv2InFlightStep.mtpRound`), so unlike plain decode there is no successor
/// step queued while the host finalizes: every nanosecond between the
/// acceptance readback and the next round's submit is dead GPU time.
///
///   hostGapNanos    previous round's finalize end -> this round's submit.
///                   The dead window. Contains plan, capture, draft build,
///                   verify build, and submit — those are broken out below so
///                   the residue (scheduling, leases, gauges, dispatch) is
///                   what is left over.
///   captureNanos    pre-write KV snapshots handed to the drafter.
///   draftBuildNanos k drafter forwards, GRAPH BUILD only (lazy).
///   verifyBuildNanos rectangular target verification graph build.
///   submitNanos     `asyncEval` of the whole round.
///   packetWaitNanos the blocking `asArray` on the acceptance packet: the
///                   round's GPU time as the host sees it. Not overlapped.
///   acceptWalkNanos target-authoritative prefix resolution for every row.
///   rowFinalizeNanos KV rollback/commit, streaming handoff, carry store,
///                   controller bookkeeping.
///
/// A round's accounted wall clock is
/// `hostGap + packetWait + acceptWalk + rowFinalize`, and the speculation
/// budget is everything in it that is not `packetWait`.
public struct CBv2MTPRoundTiming: Sendable, Equatable {
    /// Rounds these sums cover (verify rounds only; seed-only steps do not
    /// produce an acceptance packet and are counted by `seedSteps`).
    public var rounds: UInt64 = 0
    public var hostGapNanos: UInt64 = 0
    public var captureNanos: UInt64 = 0
    public var draftBuildNanos: UInt64 = 0
    public var verifyBuildNanos: UInt64 = 0
    public var submitNanos: UInt64 = 0
    public var packetWaitNanos: UInt64 = 0
    public var acceptWalkNanos: UInt64 = 0
    public var rowFinalizeNanos: UInt64 = 0
    /// Smallest and largest accounted round wall clock seen, so a reader can
    /// tell a single-shape run from one that mixed prompt lengths. The sums
    /// above are means-in-waiting, and a mean over a run whose rounds span a
    /// 64-token prompt and a full-context one describes neither. Zero means
    /// no round has been recorded.
    public var minRoundNanos: UInt64 = 0
    public var maxRoundNanos: UInt64 = 0

    public init() {}

    /// This round's accounted wall clock. Only meaningful on a single-round
    /// value (the per-round struct the engine fills in), not on a sum.
    var accountedWallNanos: UInt64 {
        hostGapNanos &+ packetWaitNanos &+ acceptWalkNanos &+ rowFinalizeNanos
    }

    /// Mean accounted round wall clock (nil before any round).
    public var meanRoundNanos: Double? {
        guard rounds > 0 else { return nil }
        return Double(accountedWallNanos) / Double(rounds)
    }

    /// Host time per round that is NOT the GPU waiting on the verify: the
    /// number a 200 tok/s target has to drive to zero.
    public var meanFixedCostNanos: Double? {
        guard rounds > 0 else { return nil }
        let fixed = hostGapNanos &+ acceptWalkNanos &+ rowFinalizeNanos
        return Double(fixed) / Double(rounds)
    }

    /// Timers are `DispatchTime.now()` reads on the engine thread — about
    /// eight per round against a round measured in milliseconds. On by
    /// default; `MTPLX_MTP_ROUND_TIMING=0` takes even that off a control run.
    public static let enabled: Bool = {
        let raw = ProcessInfo.processInfo.environment["MTPLX_MTP_ROUND_TIMING"]
        guard let raw else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    @inline(__always)
    public static func now() -> UInt64 {
        enabled ? DispatchTime.now().uptimeNanoseconds : 0
    }

    /// Elapsed since `start`, or zero when timing is off or the clock went
    /// backwards (never trusted into a sum).
    @inline(__always)
    public static func since(_ start: UInt64) -> UInt64 {
        guard enabled, start != 0 else { return 0 }
        let now = DispatchTime.now().uptimeNanoseconds
        return now > start ? now &- start : 0
    }

    mutating func add(_ other: CBv2MTPRoundTiming) {
        if other.rounds > 0 {
            let wall = other.accountedWallNanos
            minRoundNanos = minRoundNanos == 0 ? wall : Swift.min(minRoundNanos, wall)
            maxRoundNanos = Swift.max(maxRoundNanos, wall)
        }
        rounds &+= other.rounds
        hostGapNanos &+= other.hostGapNanos
        captureNanos &+= other.captureNanos
        draftBuildNanos &+= other.draftBuildNanos
        verifyBuildNanos &+= other.verifyBuildNanos
        submitNanos &+= other.submitNanos
        packetWaitNanos &+= other.packetWaitNanos
        acceptWalkNanos &+= other.acceptWalkNanos
        rowFinalizeNanos &+= other.rowFinalizeNanos
    }
}

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
    /// Per-stage host timing, summed over rounds. See `CBv2MTPRoundTiming`.
    public var roundTiming = CBv2MTPRoundTiming()

    public init() {}

    /// Preferred proposal-count spelling. `draftedTokens` remains stored for
    /// compatibility with the first engine/provider seam.
    public var proposedTokens: Int { draftedTokens }

    /// Mean accepted drafts per round (nil before any round).
    public var meanAcceptedPerRound: Double? {
        rounds > 0 ? Double(acceptedTokens) / Double(rounds) : nil
    }
}
