// Copyright © 2026 Eigen Labs.

import Foundation
import MLX

/// Request-owned drafter context: every layer's keys and values for the last
/// `window` committed target positions.
///
/// Storage is two buffers `[layers, kvHeads, window, headDimensions]`,
/// allocated once at the first write. A row lives in the slot of its absolute
/// position, so an append overwrites the oldest rows in place. Only committed
/// target rows are ever written: a block's own keys and values exist only
/// inside its forward, so there is nothing to crop after a round.
///
/// The state is advisory. Losing it, or starting it at a later position,
/// changes which tokens are proposed and never which tokens are emitted.
public final class MuseGlimmerDFlashContextState {
    public let layers: Int
    public let kvHeads: Int
    public let headDimensions: Int
    public let window: Int
    public let blockSize: Int
    public let dtype: DType

    private var keys: MLXArray?
    private var values: MLXArray?
    /// Absolute position stored in slot zero's first lap.
    private var basePosition = 0
    /// Oldest absolute position still held.
    public private(set) var firstPosition = 0
    /// Absolute position the next committed row takes; the next anchor.
    public private(set) var nextPosition = 0

    public init(configuration c: MuseGlimmerDFlashConfiguration, dtype: DType) {
        layers = c.hiddenLayers
        kvHeads = c.kvHeads
        headDimensions = c.headDimensions
        window = c.slidingWindow
        blockSize = c.blockSize
        self.dtype = dtype
    }

    /// Committed positions currently held, at most `window`.
    public var count: Int { nextPosition - firstPosition }
    public var isAllocated: Bool { keys != nil }

    /// Bytes of both buffers once written.
    public static func residentBytes(
        configuration c: MuseGlimmerDFlashConfiguration, dtype: DType
    ) -> Int {
        2 * c.hiddenLayers * c.kvHeads * c.slidingWindow * c.headDimensions * dtype.size
    }

    /// The buffers, for the owner's evaluation fence.
    public var arrays: [MLXArray] { [keys, values].compactMap { $0 } }

    /// Forget every row and continue at `position`. Storage is kept.
    public func restart(at position: Int) {
        precondition(position >= 0, "DFlash context position must not be negative")
        basePosition = position
        firstPosition = position
        nextPosition = position
    }

    /// Drop the buffers.
    public func release() {
        keys = nil
        values = nil
        restart(at: 0)
    }

    /// Append keys and values `[layers, kvHeads, rows, headDimensions]` for
    /// the positions after `skipping` unread ones. `skipping` is nonzero only
    /// when more than `window` rows arrive at once: the older rows can never
    /// be read, so the caller leaves them out and passes exactly `window`.
    func append(keys newKeys: MLXArray, values newValues: MLXArray, skipping: Int = 0) {
        let rows = newKeys.dim(2)
        precondition(
            newKeys.shape == [layers, kvHeads, rows, headDimensions]
                && newValues.shape == newKeys.shape && rows > 0 && rows <= window
                && skipping >= 0 && (skipping == 0 || rows == window),
            "DFlash context append does not match the ring")
        if keys == nil {
            let shape = [layers, kvHeads, window, headDimensions]
            // Distinct wrappers: slice assignment replaces a wrapper's array.
            keys = MLXArray.zeros(shape, dtype: dtype)
            values = MLXArray.zeros(shape, dtype: dtype)
        }
        let start = nextPosition + skipping
        if skipping > 0 {
            basePosition = start
            firstPosition = start
        }
        var written = 0
        while written < rows {
            let slot = (start + written - basePosition) % window
            let run = min(rows - written, window - slot)
            let source = written ..< (written + run)
            keys![0..., 0..., slot ..< (slot + run), 0...] =
                newKeys[0..., 0..., source, 0...].asType(dtype)
            values![0..., 0..., slot ..< (slot + run), 0...] =
                newValues[0..., 0..., source, 0...].asType(dtype)
            written += run
        }
        nextPosition = start + rows
        firstPosition = max(firstPosition, nextPosition - window)
    }

    /// Absolute position of each held row, in the order `visibleContext`
    /// returns them.
    var heldPositions: [Int] {
        guard count == window else { return Array(firstPosition ..< nextPosition) }
        let oldestSlot = (firstPosition - basePosition) % window
        return (0 ..< window).map { slot in
            firstPosition + (slot - oldestSlot + window) % window
        }
    }

    /// Which keys each block query may attend, row-major
    /// `[blockSize, contextPositions.count + blockSize]`; nil when all are
    /// visible. A query at absolute position `q` sees a context key at `p`
    /// when `q - p <= window`, and every block key. This is the reference
    /// mask `abs(q - p) <= sliding_window`; the block is never wider than the
    /// window.
    static func visibility(
        anchor: Int, blockSize: Int, window: Int, contextPositions: [Int]
    ) -> [Bool]? {
        guard let oldest = contextPositions.min(), anchor + blockSize - 1 - oldest > window
        else { return nil }
        var mask: [Bool] = []
        mask.reserveCapacity(blockSize * (contextPositions.count + blockSize))
        for query in 0 ..< blockSize {
            for position in contextPositions {
                mask.append(anchor + query - position <= window)
            }
            mask.append(contentsOf: [Bool](repeating: true, count: blockSize))
        }
        return mask
    }

    /// The held rows for a block anchored at `nextPosition`: keys and values
    /// `[layers, kvHeads, count, headDimensions]` and the query mask, or nil
    /// when nothing is held. Row order is storage order, which attention does
    /// not depend on because keys already carry their positions.
    public func visibleContext() -> (keys: MLXArray, values: MLXArray, mask: MLXArray?)? {
        guard count > 0, let keys, let values else { return nil }
        // Always new arrays, so a later append never changes what a caller holds.
        let rows = 0 ..< count
        let visibleKeys = keys[0..., 0..., rows, 0...]
        let visibleValues = values[0..., 0..., rows, 0...]
        let mask = Self.visibility(
            anchor: nextPosition, blockSize: blockSize, window: window,
            contextPositions: heldPositions
        ).map { MLXArray($0, [blockSize, count + blockSize]) }
        return (visibleKeys, visibleValues, mask)
    }
}

extension MuseGlimmerDFlashDraftModel {
    public func makeContextState() -> MuseGlimmerDFlashContextState {
        MuseGlimmerDFlashContextState(configuration: configuration, dtype: activationDType)
    }

    /// Commit target feature rows `[rows, contextWidth]` for the absolute
    /// positions `position ..< position + rows`.
    ///
    /// Rows the state already holds are skipped. A gap after the held rows
    /// restarts the state at `position`: the context is always one run of
    /// consecutive positions. Only the last `window` rows are projected.
    /// Nothing is evaluated here.
    public func commit(
        features: MLXArray, at position: Int, to state: MuseGlimmerDFlashContextState
    ) {
        precondition(
            features.ndim == 2 && features.dim(1) == configuration.contextWidth && position >= 0,
            "DFlash context features must be [rows, contextWidth]")
        let rows = features.dim(0)
        let duplicates = max(0, state.nextPosition - position)
        guard rows > duplicates else { return }
        if position > state.nextPosition { state.restart(at: position) }
        let fresh = rows - duplicates
        let unread = max(0, fresh - state.window)
        let used = features[(duplicates + unread)..., 0...]
        let context = projectContext(used)
        let projected = contextKeyValues(context, at: state.nextPosition + unread)
        state.append(keys: projected.keys, values: projected.values, skipping: unread)
    }

    /// Draft one block anchored at `state.nextPosition` from block embeddings
    /// `[blockSize, hiddenSize]`. Returns final-normed hidden states
    /// `[blockSize, hiddenSize]`; row `i >= 1` predicts the token at
    /// `nextPosition + i`. The state is not changed.
    public func draftBlock(
        embeddings: MLXArray, state: MuseGlimmerDFlashContextState
    ) -> MLXArray {
        let visible = state.visibleContext()
        return blockForward(
            embeddings: embeddings, at: state.nextPosition,
            context: visible.map { ($0.keys, $0.values) }, mask: visible?.mask)
    }

    /// `draftBlock` for several states in one forward (`blockForwards`): one
    /// block per state, each anchored at that state's `nextPosition`. No
    /// state is changed.
    public func draftBlocks(
        embeddings: [MLXArray], states: [MuseGlimmerDFlashContextState]
    ) -> [MLXArray] {
        precondition(
            embeddings.count == states.count, "DFlash blocks need one state per block")
        let visible = states.map { $0.visibleContext() }
        return blockForwards(
            embeddings: embeddings, at: states.map(\.nextPosition),
            contexts: visible.map { $0.map { ($0.keys, $0.values) } },
            masks: visible.map { $0?.mask })
    }

    /// One commit's rows for one state: features `[rows, contextWidth]` for
    /// the positions from `position`, or from the state's next position
    /// (after the earlier entries for that state) when nil.
    public struct ContextCommit {
        public let features: MLXArray
        public let position: Int?
        public let state: MuseGlimmerDFlashContextState

        public init(features: MLXArray, at position: Int?, to state: MuseGlimmerDFlashContextState)
        {
            self.features = features
            self.position = position
            self.state = state
        }
    }

    /// `commit(features:at:to:)` for each entry in order, with one context
    /// projection and one key / value projection per layer for all of them.
    ///
    /// Each state ends with the rows, positions and restarts the calls one
    /// by one give; only the projections' accumulation order may differ.
    /// Norms and rotations run per entry at its own positions. Nothing is
    /// evaluated here.
    public func commit(_ commits: [ContextCommit]) {
        // Plan against each state's next position as the earlier entries
        // leave it; nothing is written until the projections exist.
        struct Planned {
            let state: MuseGlimmerDFlashContextState
            let rows: Range<Int>
            let keyPosition: Int
            let restart: Int?
            let skipping: Int
        }
        var next: [ObjectIdentifier: Int] = [:]
        var planned: [Planned] = []
        var used: [MLXArray] = []
        var total = 0
        for entry in commits {
            let features = entry.features
            precondition(
                features.ndim == 2 && features.dim(1) == configuration.contextWidth,
                "DFlash context features must be [rows, contextWidth]")
            let id = ObjectIdentifier(entry.state)
            let held = next[id] ?? entry.state.nextPosition
            let position = entry.position ?? held
            precondition(position >= 0, "DFlash context position must not be negative")
            let rows = features.dim(0)
            let duplicates = max(0, held - position)
            guard rows > duplicates else { continue }
            let restart = position > held ? position : nil
            let start = restart ?? held
            let fresh = rows - duplicates
            let unread = max(0, fresh - entry.state.window)
            used.append(features[(duplicates + unread)..., 0...])
            planned.append(
                Planned(
                    state: entry.state, rows: total ..< (total + fresh - unread),
                    keyPosition: start + unread, restart: restart, skipping: unread))
            total += fresh - unread
            next[id] = start + fresh
        }
        guard !planned.isEmpty else { return }
        let context = projectContext(used.count == 1 ? used[0] : concatenated(used, axis: 0))
        let projected = contextKeyValues(
            context, segments: planned.map { ($0.rows, $0.keyPosition) })
        for (entry, keyValues) in zip(planned, projected) {
            if let restart = entry.restart { entry.state.restart(at: restart) }
            entry.state.append(
                keys: keyValues.keys, values: keyValues.values, skipping: entry.skipping)
        }
    }
}
