import Foundation
@preconcurrency import MLX
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}


// Shared struct for expert range tracking across projections
public struct ExpertRange: Sendable {
    public let id: Int
    public let start: Int
    public let end: Int
}

// MARK: - SwitchGLU

public class SwitchGLU: Module, @unchecked Sendable {
    @ModuleInfo(key: "gate_proj") public var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") public var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") public var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray

    // ── Async pipeline state (SSD streaming optimization) ──
    // Persistent buffers: allocated once per layer, reused across tokens.
    // Avoids per-token buffer allocation + eval overhead.
    private var _persistentGate: [MLXArray]?
    private var _persistentUp: [MLXArray]?
    private var _persistentDown: [MLXArray]?
    // Previous token's expert routing per layer for speculative prefetch.
    private var _previousExpertIds: [Int]?

    // ── Cache-slot tunable (env-tunable via `MLX_MOE_CACHE_SLOTS=N`) ──
    // Number of resident expert slots used by SSD-streaming paths that keep
    // experts cached across tokens. Default 16 is a good balance for top-k=8
    // routing on Apple Silicon: enough slack for prev-token spec prefetch +
    // current-token misses without over-pressuring the unified-memory
    // allocator. Larger values trade RAM for hit-rate. Minimum is 6 (must
    // accommodate top-k plus a small eviction margin).
    static let MAX_CACHE_SLOTS: Int = {
        if let v = ProcessInfo.processInfo.environment["MLX_MOE_CACHE_SLOTS"],
           let n = Int(v), n >= 6 {
            return n
        }
        // RAM-aware fallback when no explicit byte budget is set (see
        // resolvedCacheSlots below): ~0.75 slots per GB of physical RAM,
        // clamped to [16, 96].
        // Measured on DSV4-Flash-4bit @ M5 Max 128GB: 64 slots → 81% hit, ~5-7 tok/s;
        // 96 slots → 90% hit, ~12.3 tok/s; 128 slots → 95% hit, 0.13 GB/tok
        // (tok/s then limited by per-layer sync overhead, not I/O).
        let ramGB = Int(ProcessInfo.processInfo.physicalMemory >> 30)
        return min(96, max(16, ramGB * 3 / 4))
    }()

    // Process-wide resolved slot count, computed once from the first layer's
    // actual slab size so every layer agrees. nil until first cold init.
    nonisolated(unsafe) private static var _resolvedSlots: Int?
    private static let _slotsLock = NSLock()

    /// Resolve the per-layer slot count, machine-size aware.
    /// Priority: MLX_MOE_CACHE_SLOTS env → host-provided byte budget
    /// (ExpertStreamingConfig.expertSlotBudgetBytes / moeLayerCountHint, divided
    /// by this layer's measured per-slot slab bytes) → RAM heuristic.
    /// - Parameter slabBytesPerSlot: bytes of stacked weight buffer ONE slot
    ///   occupies in ONE layer (all projections combined).
    static func resolvedCacheSlots(slabBytesPerSlot: Int) -> Int {
        _slotsLock.lock()
        defer { _slotsLock.unlock() }
        if let s = _resolvedSlots { return s }
        var slots = MAX_CACHE_SLOTS
        if ProcessInfo.processInfo.environment["MLX_MOE_CACHE_SLOTS"] == nil,
           let budget = ExpertStreamingConfig.shared.expertSlotBudgetBytes,
           let layers = ExpertStreamingConfig.shared.moeLayerCountHint,
           layers > 0, slabBytesPerSlot > 0 {
            // Floor 8 (top-k + eviction margin). Cap 96: EXP-001 (2026-06-11,
            // M5 Max 128 GB, DSV4-Flash-4bit) measured the memory cliff —
            // 128 slots → 88 GB active GPU → swap thrash → ~0.1 tok/s
            // (~100× collapse); 96 slots → 70.5 GB active → 10.5 tok/s;
            // 64 slots matched 96 in warm decode at −17 GB. Until the budget
            // reserves real workspace+page-cache headroom and is tuned
            // against the sustained-drift gate, 96 is the measured-safe
            // ceiling. MLX_MOE_CACHE_SLOTS still overrides explicitly.
            slots = min(96, max(8, budget / (layers * slabBytesPerSlot)))
            print("[SwitchLayers] Slot budget: \(budget >> 20) MB across \(layers) layers @ \(slabBytesPerSlot >> 20) MB/slot → \(slots) slots/layer")
        }
        _resolvedSlots = slots
        return slots
    }

    // ── Verbose streaming diagnostics (env-gated MLX_MOE_VERBOSE=1) ──
    // Per-miss sync prints cost real time at 40+ MoE layers per token; off by default.
    static let verboseStreaming: Bool = {
        let v = ProcessInfo.processInfo.environment["MLX_MOE_VERBOSE"] ?? ""
        return v == "1" || v.lowercased() == "true"
    }()

    // ── Speculative prev-token prefetch (env-gated MLX_MOE_SPEC_PREFETCH) ──
    // EXP-006 evidence: corpus router-trace simulation (66k layer-calls) shows
    // only 6.6% of slot-cache misses are predictable from the previous token's
    // experts — yet this prefetch BLOCKS every streamed layer entry on a
    // concurrentPerform fetching those predictions, and its admitted-target
    // count grows with free/cold slots (the EXP-004 TTFT-vs-slots anomaly).
    // Default decided by EXP-006's A/B; =0/=1 overrides.
    static let specPrefetchEnabled: Bool = {
        let v = ProcessInfo.processInfo.environment["MLX_MOE_SPEC_PREFETCH"] ?? ""
        if v == "0" || v.lowercased() == "false" { return false }
        return true
    }()

    // ── Full-GPU drain before slot overwrite (env-gated MLX_MOE_SAFE_DRAIN=1) ──
    // The idx.asArray() routing sync already orders slot writes after all prior reads
    // (see runStackedFastPath); the drain is kept only as a debugging escape hatch.
    static let safeDrain: Bool = {
        let v = ProcessInfo.processInfo.environment["MLX_MOE_SAFE_DRAIN"] ?? ""
        return v == "1" || v.lowercased() == "true"
    }()

    // ── Stacked-buffer fused-matmul fast path (env-gated MLX_MOE_STACKED=1) ──
    // When enabled, allocate a single stacked weight buffer of shape
    // `[CACHE_SLOTS, intermediate, hidden]` per projection (instead of
    // CACHE_SLOTS individual `[1, intermediate, hidden]` buffers) and
    // populate slots via `MLXFast.preadIntoOffset`, which writes one
    // expert into a byte-offset region of the stacked tensor in place.
    //
    // The win is dispatch reduction: `gatherQuantizedMM` runs ONCE per
    // projection per layer (using `rhsIndices = slotPerToken`), instead
    // of `top_k` separate dispatches per projection per layer. On Apple
    // Silicon each Metal dispatch carries ~30 µs of CPU→GPU
    // encode/submit overhead, which dominates per-token compute on
    // SSD-streamed MoE models.
    //
    // Eligible layers: all 3 projections quantized + SSD streaming
    // resolveable + `idx.size <= 32` (single-token generation). Anything
    // else falls through to the existing N-buffer path. The flag is
    // ON by default (it is only reachable when SSD streaming is active and all
    // projections are quantized); set MLX_MOE_STACKED=0 to fall back to the
    // legacy N-buffer path.
    private static let useStackedBuffers: Bool = {
        let v = ProcessInfo.processInfo.environment["MLX_MOE_STACKED"] ?? ""
        if v == "0" || v.lowercased() == "false" { return false }
        return true
    }()
    private var _stackedGate: MLXArray?
    private var _stackedUp: MLXArray?
    private var _stackedDown: MLXArray?
    // Per-slot expert occupant; nil means empty.
    private var _slotExpert: [Int?]?
    // Per-slot last-used token counter, used as the LRU tie-break.
    private var _slotLastUsed: [Int]?
    // Per-EXPERT routing-frequency score (with periodic decay) — the primary
    // eviction key. MoE popularity is power-law, so keeping *hot* experts
    // resident beats pure LRU. Tracked per expert (not per slot) so the score
    // PERSISTS across evict→re-admit (a re-admitted hot expert keeps its
    // history) — the W-TinyLFU substrate. Size = numExperts. Decayed every
    // `hotnessDecayTokens`. Also gates speculative admission (don't let a cold
    // prefetch guess evict a hotter resident expert).
    private var _routeHotness: [Int]?
    // Per-layer token counter — incremented per fast-path call.
    private var _tokenCounter: Int = 0
    // Halve all slot frequencies every N tokens so the policy tracks drift.
    private static let hotnessDecayTokens = 16
    // Bytes per expert slab in a stacked buffer; computed once on cold init.
    private var _stackedBytesPerExpert: Int = 0
    private var _stackedDownBytesPerExpert: Int = 0

    // ── Fused gate+up SwiGLU mode (env-gated MLX_MOE_FUSE_GATEUP=1) ──
    // SwiGLU MLP is `silu(gate(x)) * up(x)`; gate and up are independent
    // matmuls of identical shape. When enabled (and useStackedBuffers is
    // also enabled), allocate ONE combined buffer of shape
    // `[CACHE_SLOTS, 2 * intermediate, hidden]` and write each expert's
    // gate weights into the first half of its slot and up weights into
    // the second half (offsets `slot * 2 * bpe` and `slot * 2 * bpe + bpe`).
    // A single `gatherQuantizedMM` then produces `[..., 2 * intermediate]`,
    // which is split into the two halves and fed into `silu(g) * u`.
    //
    // Saves ONE projection-level dispatch per layer per token (the gate
    // and up matmuls collapse into one). Requires useStackedBuffers; no
    // effect on the legacy N-buffer or non-stacked paths. ON by default;
    // set MLX_MOE_FUSE_GATEUP=0 to disable.
    private static let useFusedGateUp: Bool = {
        let v = ProcessInfo.processInfo.environment["MLX_MOE_FUSE_GATEUP"] ?? ""
        if v == "0" || v.lowercased() == "false" { return false }
        return true
    }()
    private var _stackedGateUp: MLXArray?
    // Pre-concatenated gate+up scales/biases along the intermediate axis,
    // computed once at cold init so the runtime `MLX.take` is a single op.
    private var _combinedGateUpScales: MLXArray?
    private var _combinedGateUpBiases: MLXArray?
    // Bytes for ONE projection (gate or up) per slot in the combined buffer;
    // total bytes per slot = 2 * _stackedGateUpBytesPerProj.
    private var _stackedGateUpBytesPerProj: Int = 0

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray = MLXNN.silu,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)
        super.init()
    }

    /// Stacked-buffer fused-matmul fast path. Returns nil if the layer is not
    /// eligible (any projection is non-quantized, missing SSD info, idx.size > 32,
    /// or no slot is available); the caller then falls through to the existing
    /// N-buffer path.
    ///
    /// Cold path allocates one `[CACHE_SLOTS, intermediate, hidden]` weight
    /// buffer per projection. Subsequent calls reuse the buffer; an LRU array
    /// rotates slot occupants; misses are written via `MLXFast.preadIntoOffset`
    /// directly into the slot's byte region. Compute issues a single
    /// `gatherQuantizedMM` per projection per layer (vs. `top_k` per projection
    /// in the legacy path).
    private func runStackedFastPath(x: MLXArray, indices: MLXArray) -> MLXArray? {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])
        let doSort = true  // SSD path always sorts so expert ranges are contiguous
        var idx = indices
        var inverseOrder = MLXArray()
        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }
        guard idx.size <= 32,
              gateProj is QuantizedSwitchLinear,
              upProj is QuantizedSwitchLinear,
              downProj is QuantizedSwitchLinear,
              let gateSSD = gateProj.resolveSSDInfo(),
              let upSSD = upProj.resolveSSDInfo(),
              let downSSD = downProj.resolveSSDInfo() else {
            return nil  // ineligible — fall through to legacy path
        }

        // Per-slot slab bytes for THIS layer (gate+up+down packed weights), from
        // lazy shape metadata (no eval). Drives the machine-size-aware slot count.
        let slabBytesPerSlot =
            (gateProj.weight.nbytes + upProj.weight.nbytes + downProj.weight.nbytes)
            / max(1, gateProj.weight.dim(0))
        let CACHE_SLOTS = SwitchGLU.resolvedCacheSlots(slabBytesPerSlot: slabBytesPerSlot)
        let isFused = SwitchGLU.useFusedGateUp

        if _stackedGate == nil && _stackedGateUp == nil {
            if isFused {
                // Combined gate+up buffer: shape [CACHE_SLOTS, 2*intermediate, hidden].
                _stackedGateUp = MLXArray.zeros(
                    [CACHE_SLOTS, 2 * gateProj.weight.dim(1), gateProj.weight.dim(2)]
                ).asType(gateProj.weight.dtype)
                _stackedDown = MLXArray.zeros(
                    [CACHE_SLOTS, downProj.weight.dim(1), downProj.weight.dim(2)]
                ).asType(downProj.weight.dtype)
                // Pre-concatenate gate+up scales/biases (one-time at cold init).
                if let qGate = gateProj as? QuantizedSwitchLinear, let qUp = upProj as? QuantizedSwitchLinear {
                    _combinedGateUpScales = MLX.concatenated([qGate.scales, qUp.scales], axis: 1)
                    if let gb = qGate.biases, let ub = qUp.biases {
                        _combinedGateUpBiases = MLX.concatenated([gb, ub], axis: 1)
                    }
                }
                _slotExpert = Array(repeating: nil, count: CACHE_SLOTS)
                _slotLastUsed = Array(repeating: 0, count: CACHE_SLOTS)
                _routeHotness = Array(repeating: 0, count: numExperts)
                _tokenCounter = 0
                var coldEvalList: [MLXArray] = [idx, _stackedGateUp!, _stackedDown!, _combinedGateUpScales!]
                if let cb = _combinedGateUpBiases { coldEvalList.append(cb) }
                MLX.eval(coldEvalList)
                _stackedGateUpBytesPerProj = _stackedGateUp!.nbytes / CACHE_SLOTS / 2
                _stackedBytesPerExpert = _stackedGateUpBytesPerProj
                _stackedDownBytesPerExpert = _stackedDown!.nbytes / CACHE_SLOTS
            } else {
                _stackedGate = MLXArray.zeros(
                    [CACHE_SLOTS, gateProj.weight.dim(1), gateProj.weight.dim(2)]
                ).asType(gateProj.weight.dtype)
                _stackedUp = MLXArray.zeros(
                    [CACHE_SLOTS, upProj.weight.dim(1), upProj.weight.dim(2)]
                ).asType(upProj.weight.dtype)
                _stackedDown = MLXArray.zeros(
                    [CACHE_SLOTS, downProj.weight.dim(1), downProj.weight.dim(2)]
                ).asType(downProj.weight.dtype)
                _slotExpert = Array(repeating: nil, count: CACHE_SLOTS)
                _slotLastUsed = Array(repeating: 0, count: CACHE_SLOTS)
                _routeHotness = Array(repeating: 0, count: numExperts)
                _tokenCounter = 0
                MLX.eval([idx, _stackedGate!, _stackedUp!, _stackedDown!])
                _stackedBytesPerExpert = _stackedGate!.nbytes / CACHE_SLOTS
                _stackedDownBytesPerExpert = _stackedDown!.nbytes / CACHE_SLOTS
            }
        } else {
            // Warm path: kick off GPU work asynchronously while we
            // speculatively prefetch the prev-token's experts. The pread
            // overlaps with the GPU-side resolution of `idx`.
            asyncEval(idx)
        }
        _tokenCounter += 1
        // Periodic hotness decay: halve every slot's frequency so the eviction
        // policy adapts as the routed-expert distribution drifts over the text.
        if _routeHotness != nil, _tokenCounter % SwitchGLU.hotnessDecayTokens == 0 {
            for e in 0..<_routeHotness!.count { _routeHotness![e] >>= 1 }
        }

        // ── Speculative prefetch: pre-load prev-token's experts that are
        //    NOT already cached, evicting LRU slots and pre-claiming them so
        //    the current-token resolution sees them as hits. Token-to-token
        //    expert overlap is high in steady-state generation, so most of
        //    this work pays off on the same call.
        var expertToSlotPre = [Int: Int]()
        for (slot, eid) in _slotExpert!.enumerated() {
            if let eid = eid { expertToSlotPre[eid] = slot }
        }
        // Coldest-occupant-first victim (per-EXPERT hotness, LRU tiebreak);
        // returns -1 if all candidates are claimed. Empty slots preferred.
        func pickColdestSlot(excluding claimed: Set<Int>) -> Int {
            for s in 0..<CACHE_SLOTS {
                if claimed.contains(s) { continue }
                if _slotExpert![s] == nil { return s }
            }
            var bestSlot = -1, bestHits = Int.max, bestTs = Int.max
            for s in 0..<CACHE_SLOTS {
                if claimed.contains(s) { continue }
                let h = _routeHotness![_slotExpert![s]!]
                if h < bestHits || (h == bestHits && _slotLastUsed![s] < bestTs) {
                    bestHits = h; bestTs = _slotLastUsed![s]; bestSlot = s
                }
            }
            return bestSlot
        }
        var specTargets: [(slot: Int, expertId: Int)] = []
        if SwitchGLU.specPrefetchEnabled, let prevIds = _previousExpertIds {
            var specClaimed = Set<Int>()
            for eid in prevIds {
                if expertToSlotPre[eid] != nil { continue }  // already cached
                let slot = pickColdestSlot(excluding: specClaimed)
                if slot < 0 { break }  // shouldn't happen — CACHE_SLOTS > top_k
                // W-TinyLFU admission: a SPECULATIVE guess may only evict a
                // resident expert if it is at least as hot — never let a cold
                // prediction pollute the cache. (Empty slots always admit.)
                if let victim = _slotExpert![slot],
                   _routeHotness![eid] < _routeHotness![victim] { continue }
                if let old = _slotExpert![slot] { expertToSlotPre.removeValue(forKey: old) }
                _slotExpert![slot] = eid  // claim slot speculatively
                expertToSlotPre[eid] = slot
                specClaimed.insert(slot)
                specTargets.append((slot, eid))
            }
        }
        if !specTargets.isEmpty {
            let bpe = _stackedBytesPerExpert
            let downBpe = _stackedDownBytesPerExpert
            let errState = ThreadSafeError()
            DispatchQueue.concurrentPerform(iterations: specTargets.count * 3) { [specTargets] i in
                errState.catchError {
                let mIdx = i / 3
                let proj = i % 3
                let info = specTargets[mIdx]
                switch proj {
                case 0:
                    let ssd = self.gateProj.resolveSSDInfo(expertIndex: info.expertId) ?? (gateSSD.path, gateSSD.tensorName, UInt32(info.expertId))
                    if isFused {
                        // Gate -> first half of slot in combined buffer.
                        let off = info.slot * 2 * bpe
                        timedPread(bytes: bpe, prefetch: true) {
                            MLXFast.preadIntoOffset(self._stackedGateUp!, safetensorsPath: ssd.path,
                                                    tensorName: ssd.tensorName, expertIndex: ssd.readIndex, dstOffset: off)
                        }
                    } else {
                        timedPread(bytes: bpe, prefetch: true) {
                            MLXFast.preadIntoOffset(self._stackedGate!, safetensorsPath: ssd.path,
                                                    tensorName: ssd.tensorName, expertIndex: ssd.readIndex, dstOffset: info.slot * bpe)
                        }
                    }
                case 1:
                    let ssd = self.upProj.resolveSSDInfo(expertIndex: info.expertId) ?? (upSSD.path, upSSD.tensorName, UInt32(info.expertId))
                    if isFused {
                        // Up -> second half of slot in combined buffer.
                        let off = info.slot * 2 * bpe + bpe
                        timedPread(bytes: bpe, prefetch: true) {
                            MLXFast.preadIntoOffset(self._stackedGateUp!, safetensorsPath: ssd.path,
                                                    tensorName: ssd.tensorName, expertIndex: ssd.readIndex, dstOffset: off)
                        }
                    } else {
                        timedPread(bytes: bpe, prefetch: true) {
                            MLXFast.preadIntoOffset(self._stackedUp!, safetensorsPath: ssd.path,
                                                    tensorName: ssd.tensorName, expertIndex: ssd.readIndex, dstOffset: info.slot * bpe)
                        }
                    }
                default:
                    let ssd = self.downProj.resolveSSDInfo(expertIndex: info.expertId) ?? (downSSD.path, downSSD.tensorName, UInt32(info.expertId))
                    timedPread(bytes: downBpe, prefetch: true) {
                        MLXFast.preadIntoOffset(self._stackedDown!, safetensorsPath: ssd.path,
                                                tensorName: ssd.tensorName, expertIndex: ssd.readIndex, dstOffset: info.slot * downBpe)
                    }
                }
                }
            }
            errState.check()
        }

        if idx.size == 0 {
            var outShape = x.shape
            outShape[outShape.count - 1] = downProj.outputDims
            let result = MLXArray.zeros(outShape).asType(.float16)
            if doSort {
                return MLX.squeezed(scatterUnsort(x: result, invOrder: inverseOrder, shape: indices.shape), axis: -2)
            }
            return MLX.squeezed(result, axis: -2)
        }

        // Parse routing — `idx.asArray()` is the actual sync point on GPU.
        // By now, GPU work (current attention + router) is mostly done, AND
        // most of this token's experts are already in cache via spec prefetch.
        let cpuIndices = idx.asArray(UInt32.self)
        var ranges = [ExpertRange]()
        var startIdx = 0
        while startIdx < cpuIndices.count {
            let eid = Int(cpuIndices[startIdx])
            var endIdx = startIdx + 1
            while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == eid { endIdx += 1 }
            ranges.append(ExpertRange(id: eid, start: startIdx, end: endIdx))
            startIdx = endIdx
        }

        // Bump per-expert hotness for EVERY routed expert this token (the
        // persistent W-TinyLFU frequency signal, survives evict/re-admit).
        for r in ranges { _routeHotness![r.id] += 1 }

        // ── Slot resolution: route each range to a slot ──
        var expertToSlot = [Int: Int]()
        for (slot, eid) in _slotExpert!.enumerated() {
            if let eid = eid { expertToSlot[eid] = slot }
        }
        // Evict the slot whose OCCUPANT expert is coldest (lowest per-expert
        // hotness), LRU as the tie-break. Empty slots first. Current-token
        // routed experts are mandatory, so they always admit (unlike the
        // speculative path, which is admission-gated above).
        func pickEvictionSlot(excluding claimed: Set<Int>) -> Int {
            for s in 0..<CACHE_SLOTS {
                if claimed.contains(s) { continue }
                if _slotExpert![s] == nil { return s }
            }
            var bestSlot = -1, bestHits = Int.max, bestTs = Int.max
            for s in 0..<CACHE_SLOTS {
                if claimed.contains(s) { continue }
                let h = _routeHotness![_slotExpert![s]!]
                if h < bestHits || (h == bestHits && _slotLastUsed![s] < bestTs) {
                    bestHits = h; bestTs = _slotLastUsed![s]; bestSlot = s
                }
            }
            return bestSlot
        }
        var slotForRange: [Int] = []
        var missesNeedingPread: [(slot: Int, expertId: Int)] = []
        var claimedSlots = Set<Int>()
        for r in ranges {
            if let slot = expertToSlot[r.id], !claimedSlots.contains(slot) {
                slotForRange.append(slot)
                claimedSlots.insert(slot)
                _slotLastUsed![slot] = _tokenCounter
            } else {
                let slot = pickEvictionSlot(excluding: claimedSlots)
                if slot < 0 { return nil }  // no slot available; fall back
                if let old = _slotExpert![slot] { expertToSlot.removeValue(forKey: old) }
                _slotExpert![slot] = r.id
                expertToSlot[r.id] = slot
                _slotLastUsed![slot] = _tokenCounter
                claimedSlots.insert(slot)
                slotForRange.append(slot)
                missesNeedingPread.append((slot, r.id))
            }
        }

        // ── Pread misses into stacked-buffer slots ──
        if !missesNeedingPread.isEmpty {
            let bpe = _stackedBytesPerExpert
            let downBpe = _stackedDownBytesPerExpert

            // SYNCHRONIZATION: by this point `idx.asArray()` (above) has blocked on the
            // routing chain, which transitively forces the COMPLETE evaluation of the
            // previous token's graph (its sampled token is an input to this one) — including
            // every prior gatherQuantizedMM read of these stacked buffers. Overwriting slots
            // here is therefore already ordered after all in-flight reads, and the previous
            // full `Stream.gpu.synchronize()` (a whole-GPU drain per layer per token) is
            // unnecessary. MLX_MOE_SAFE_DRAIN=1 restores it as a debugging escape hatch.
            if SwitchGLU.safeDrain {
                Stream.gpu.synchronize()
                if SwitchGLU.verboseStreaming {
                    print("[SwitchLayers] SSD Sync: GPU drained. Misses=\(missesNeedingPread.count)")
                    fflush(stdout)
                }
            }

            // Async fetch: preads overlap the remaining graph build for this and
            // the next layer; SSDFetchPool.waitAll() barriers (SwitchGLU entry +
            // generation loop) order them before any graph execution.
            for i in 0 ..< missesNeedingPread.count * 3 {
                let mIdx = i / 3
                let proj = i % 3
                let info = missesNeedingPread[mIdx]
                SSDFetchPool.shared.submit {
                switch proj {
                case 0:
                    let ssd = self.gateProj.resolveSSDInfo(expertIndex: info.expertId) ?? (gateSSD.path, gateSSD.tensorName, UInt32(info.expertId))
                    if isFused {
                        let off = info.slot * 2 * bpe
                        timedPread(bytes: bpe) {
                            MLXFast.preadIntoOffset(self._stackedGateUp!, safetensorsPath: ssd.path,
                                                    tensorName: ssd.tensorName, expertIndex: ssd.readIndex, dstOffset: off)
                        }
                    } else {
                        timedPread(bytes: bpe) {
                            MLXFast.preadIntoOffset(self._stackedGate!, safetensorsPath: ssd.path,
                                                    tensorName: ssd.tensorName, expertIndex: ssd.readIndex, dstOffset: info.slot * bpe)
                        }
                    }
                case 1:
                    let ssd = self.upProj.resolveSSDInfo(expertIndex: info.expertId) ?? (upSSD.path, upSSD.tensorName, UInt32(info.expertId))
                    if isFused {
                        let off = info.slot * 2 * bpe + bpe
                        timedPread(bytes: bpe) {
                            MLXFast.preadIntoOffset(self._stackedGateUp!, safetensorsPath: ssd.path,
                                                    tensorName: ssd.tensorName, expertIndex: ssd.readIndex, dstOffset: off)
                        }
                    } else {
                        timedPread(bytes: bpe) {
                            MLXFast.preadIntoOffset(self._stackedUp!, safetensorsPath: ssd.path,
                                                    tensorName: ssd.tensorName, expertIndex: ssd.readIndex, dstOffset: info.slot * bpe)
                        }
                    }
                default:
                    let ssd = self.downProj.resolveSSDInfo(expertIndex: info.expertId) ?? (downSSD.path, downSSD.tensorName, UInt32(info.expertId))
                    timedPread(bytes: downBpe) {
                        MLXFast.preadIntoOffset(self._stackedDown!, safetensorsPath: ssd.path,
                                                tensorName: ssd.tensorName, expertIndex: ssd.readIndex, dstOffset: info.slot * downBpe)
                    }
                }
                }
            }
        }
        // Hit/miss accounting for the streaming objective (bytes/token ∝ miss rate).
        SSDStreamMetrics.shared.recordResolution(
            hits: ranges.count - missesNeedingPread.count,
            misses: missesNeedingPread.count)
        // EXP-000c: optional router-decision trace (MLX_ROUTER_TRACE=<path>).
        // Prices prefetch policies (stickiness), frequency-keyed quant (histogram),
        // and speculative expert-union amortization (U on multi-token batches).
        RouterTrace.shared.record(layerKey: downSSD.tensorName,
                                  expertIds: ranges.map { $0.id })
        _previousExpertIds = ranges.map { $0.id }

        // ── Build slotPerToken + slotExperts arrays for fused compute ──
        var slotPerTokenArr = [Int32](repeating: 0, count: cpuIndices.count)
        for (rIdx, r) in ranges.enumerated() {
            let s = Int32(slotForRange[rIdx])
            for t in r.start..<r.end { slotPerTokenArr[t] = s }
        }
        let slotPerToken = MLXArray(slotPerTokenArr).asType(.uint32)
        let slotExperts = _slotExpert!.map { Int32($0 ?? 0) }

        // ── Fused compute: ONE gatherQuantizedMM per projection ──
        let intermediate: MLXArray
        if isFused {
            // SINGLE matmul over combined gate+up buffer; split the output into halves.
            let (xGate, xUp) = self.runFusedGateUpMatmul(
                x: x,
                gateProj: gateProj,
                slotPerToken: slotPerToken,
                slotExperts: slotExperts)
            intermediate = activation(xGate) * xUp
        } else {
            let xGate = gateProj.computeExpertsFused(x, stackedBuffer: _stackedGate!,
                                                  slotPerToken: slotPerToken, slotExperts: slotExperts)
            let xUp = upProj.computeExpertsFused(x, stackedBuffer: _stackedUp!,
                                              slotPerToken: slotPerToken, slotExperts: slotExperts)
            intermediate = activation(xGate) * xUp
        }
        x = downProj.computeExpertsFused(intermediate, stackedBuffer: _stackedDown!,
                                      slotPerToken: slotPerToken, slotExperts: slotExperts)

        if doSort {
            return MLX.squeezed(scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape), axis: -2)
        }
        return MLX.squeezed(x, axis: -2)
    }

    /// Single fused `gatherQuantizedMM` over the combined `_stackedGateUp`
    /// buffer (shape `[CACHE_SLOTS, 2 * intermediate, hidden]`), then splits
    /// the output `[..., 2 * intermediate]` into `(xGate, xUp)` halves.
    ///
    /// Pre-conditions (guaranteed by `runStackedFastPath` cold init when
    /// `useFusedGateUp` is true):
    ///   - `_stackedGateUp` populated with gate -> first half, up -> second half per slot
    ///   - `_combinedGateUpScales` = `concat(gateProj.scales, upProj.scales, axis: 1)`
    ///   - `_combinedGateUpBiases` = `concat(gateProj.biases, upProj.biases, axis: 1)` (or nil)
    private func runFusedGateUpMatmul(
        x: MLXArray,
        gateProj: SwitchLinear,
        slotPerToken: MLXArray,
        slotExperts: [Int32]
    ) -> (MLXArray, MLXArray) {
        let qGate = gateProj as! QuantizedSwitchLinear
        let slotExpertsMLX = MLXArray(slotExperts).asType(.uint32)
        // Gather the combined scales/biases for the experts currently in our slots.
        // _combinedGateUpScales is [numExperts, 2 * intermediate, hidden / groupSize].
        let stackedScales = MLX.take(_combinedGateUpScales!, slotExpertsMLX, axis: 0)
        var stackedBiases: MLXArray? = nil
        if let cb = _combinedGateUpBiases {
            stackedBiases = MLX.take(cb, slotExpertsMLX, axis: 0)
        }

        // ONE dispatch instead of two.
        let combined = MLX.gatherQuantizedMM(
            x, _stackedGateUp!,
            scales: stackedScales,
            biases: stackedBiases,
            rhsIndices: slotPerToken,
            transpose: true,
            groupSize: qGate.groupSize, bits: qGate.bits, mode: qGate.mode, sortedIndices: true
        )

        // Split [..., 2 * intermediate] into (xGate, xUp).
        let interDim = qGate.outputDims
        let leadingShape = Array(x.shape.dropLast())
        var combinedReshaped = combined
        let canonicalCombined = leadingShape + [2 * interDim]
        if combinedReshaped.shape != canonicalCombined {
            combinedReshaped = combinedReshaped.reshaped(canonicalCombined)
        }
        let xGate = combinedReshaped[.ellipsis, 0 ..< interDim]
        let xUp = combinedReshaped[.ellipsis, interDim ..< 2 * interDim]
        return (xGate, xUp)
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        // Order any in-flight async expert fetches before this layer can trigger
        // execution of graphs that read the slot buffers (asyncEval/idx.asArray).
        SSDFetchPool.shared.waitAll()
        // ── FP8 Memory-Resident Path ──
        // FP8 models are fully loaded in memory (35GB fits in 64GB UMA).
        // Bypass the SSD streaming / BATCH path completely, which is built for
        // QuantizedSwitchLinear and eager BF16 dequantization.
        let isFP8 = gateProj.weightScaleInv != nil && !ExpertStreamingConfig.shared.isEnabled
        if isFP8 {
            var xSorted = MLX.expandedDimensions(x, axes: [-2, -3])
            var idx = indices
            var inverseOrder = MLXArray()
            
            let doSort = indices.size >= 64
            if doSort {
                (xSorted, idx, inverseOrder) = gatherSort(x: xSorted, indices: indices)
            }
            
            let xGate = gateProj(xSorted, idx, sortedIndices: doSort)
            let xUp = upProj(xSorted, idx, sortedIndices: doSort)
            let intermediate = self.activation(xGate) * xUp
            let result = downProj(intermediate, idx, sortedIndices: doSort)
            
            if doSort {
                let scattered = scatterUnsort(x: result, invOrder: inverseOrder, shape: indices.shape)
                return scattered.dim(-2) == 1 ? MLX.squeezed(scattered, axis: -2) : scattered
            }
            return result.dim(-2) == 1 ? MLX.squeezed(result, axis: -2) : result
        }

        // Stacked-buffer fused-matmul fast path (env-gated MLX_MOE_STACKED=1).
        // Early-out into the stacked path when applicable; otherwise fall
        // through to the existing SSD-streaming / legacy code below.
        if SwitchGLU.useStackedBuffers,
           ExpertStreamingConfig.shared.isEnabled,
           let result = self.runStackedFastPath(x: x, indices: indices) {
            return result
        }

        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        // We must force sorting/flattening when SSD streaming is active to properly batch
        // expert kernel dispatches dynamically over contiguous arrays.
        let isSSDStreaming = ExpertStreamingConfig.shared.isEnabled
        // NOTE: indices eval deferred to inside the cross-projection path below,
        // where it's merged with buffer allocation into fewer eval calls.
        let doSort = (indices.size >= 64) || isSSDStreaming

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        // ── Cross-projection batched SSD streaming path ──────────────────
        // When all 3 projections are quantized and SSD-streaming is active,
        // orchestrate buffer allocation, pread, and compute across all 3
        // projections to minimize MLX.eval() calls:
        //   - Single-token (fast path): 1 eval merges idx + buffer alloc
        //   - Prompt (large batch): 2 evals (idx, then buffers)
        //   - NO final eval — next layer's eval(idx) forces this layer
        // This reduces from 4 evals/layer (original) to 1 eval/layer.
        if isSSDStreaming,
           let gateSSD = gateProj.resolveSSDInfo(),
           let upSSD = upProj.resolveSSDInfo(),
           let downSSD = downProj.resolveSSDInfo() {

            // ── EVAL REDUCTION STRATEGY ──────────────────────────────────────
            // For single-token generation (idx.size ≤ 32), we merge the sorted-
            // indices eval and buffer-allocation eval into ONE call, cutting from
            // 3 evals/layer to 1.  The final MLX.eval(x) is removed entirely:
            // the NEXT layer's SwitchGLU eval(idx) transitively forces this
            // layer's full output (including KV cache) through the lazy
            // dependency chain.  For the last layer, the generation loop's eval
            // of logits handles it.
            // ─────────────────────────────────────────────────────────────────

            if idx.size <= 32 {
                // ── FAST PATH: single-token generation with async I/O-GPU pipeline ──
                //
                // STRATEGY: Overlap NVMe I/O with GPU compute using asyncEval.
                //
                // Cold path (first token): Allocate persistent buffers, merged eval,
                //   full pread — same as ssd-opt-v1 baseline.
                //
                // Warm path (subsequent tokens): asyncEval(idx) starts GPU work
                //   (prev layer expert compute + current attention/router) while
                //   CPU speculatively preads predicted experts (from previous token's
                //   routing) into persistent buffers. After GPU sync, only ~30% of
                //   experts need on-demand pread (misses). Saves ~60ms/token by
                //   hiding I/O behind GPU compute.
                //
                // Memory cost: ~5GB for persistent buffers across 48 layers
                //   (vs ~13GB for the failed in-memory cache approach).

                let maxBuffers = idx.size  // typically 8 (top_k)

                if _persistentGate == nil {
                    // ── COLD PATH: first token, allocate persistent buffers ──
                    _persistentGate = gateProj.allocateExpertBuffers(maxBuffers)
                    _persistentUp = upProj.allocateExpertBuffers(maxBuffers)
                    _persistentDown = downProj.allocateExpertBuffers(maxBuffers)


                    // Merged eval: idx + buffer allocations (same as ssd-opt-v1)
                    var toEval: [MLXArray] = [idx]
                    toEval.append(contentsOf: _persistentGate!)
                    toEval.append(contentsOf: _persistentUp!)
                    toEval.append(contentsOf: _persistentDown!)
                    MLX.eval(toEval)

                    // Handle empty indices
                    if idx.size == 0 {
                        var outShape = x.shape
                        outShape[outShape.count - 1] = downProj.outputDims
                        let result = MLXArray.zeros(outShape).asType(.float16)
                        if doSort {
                            return MLX.squeezed(scatterUnsort(x: result, invOrder: inverseOrder, shape: indices.shape), axis: -2)
                        }
                        return MLX.squeezed(result, axis: -2)
                    }

                    // Parse routing
                    let cpuIndices = idx.asArray(UInt32.self)
                    var ranges = [ExpertRange]()
                    var startIdx = 0
                    while startIdx < cpuIndices.count {
                        let eid = Int(cpuIndices[startIdx])
                        var endIdx = startIdx + 1
                        while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == eid { endIdx += 1 }
                        ranges.append(ExpertRange(id: eid, start: startIdx, end: endIdx))
                        startIdx = endIdx
                    }

                    // Full concurrent pread (baseline path)
                    let totalReads = ranges.count * 3
                    let errState = ThreadSafeError()
                    DispatchQueue.concurrentPerform(iterations: totalReads) { [ranges] i in
                        errState.catchError {
                        let expertIdx = i / 3
                        let projIdx = i % 3
                        let r = ranges[expertIdx]
                        switch projIdx {
                        case 0:
                            let ssd = self.gateProj.resolveSSDInfo(expertIndex: r.id) ?? (gateSSD.path, gateSSD.tensorName, UInt32(r.id))
                            MLXFast.preadInto(self._persistentGate![expertIdx], safetensorsPath: ssd.path,
                                              tensorName: ssd.tensorName, expertIndex: ssd.readIndex)
                        case 1:
                            let ssd = self.upProj.resolveSSDInfo(expertIndex: r.id) ?? (upSSD.path, upSSD.tensorName, UInt32(r.id))
                            MLXFast.preadInto(self._persistentUp![expertIdx], safetensorsPath: ssd.path,
                                              tensorName: ssd.tensorName, expertIndex: ssd.readIndex)
                        default:
                            let ssd = self.downProj.resolveSSDInfo(expertIndex: r.id) ?? (downSSD.path, downSSD.tensorName, UInt32(r.id))
                            MLXFast.preadInto(self._persistentDown![expertIdx], safetensorsPath: ssd.path,
                                              tensorName: ssd.tensorName, expertIndex: ssd.readIndex)
                        }
                        }
                    }
                    errState.check()

                    // Store routing for next token's predictions
                    _previousExpertIds = ranges.map { $0.id }

                    // Lazy compute
                    let usedGate = Array(_persistentGate![0..<ranges.count])
                    let usedUp = Array(_persistentUp![0..<ranges.count])
                    let usedDown = Array(_persistentDown![0..<ranges.count])
                    let xGate = gateProj.computeExperts(x, buffers: usedGate, ranges: ranges)
                    let xUp = upProj.computeExperts(x, buffers: usedUp, ranges: ranges)
                    let intermediate = activation(xGate) * xUp
                    x = downProj.computeExperts(intermediate, buffers: usedDown, ranges: ranges)

                } else {
                    // ── WARM PATH: asyncEval + speculative pread pipeline ──

                    // Start GPU work asynchronously: forces prev layer's expert
                    // compute + current layer's attention + router.
                    // GPU time: ~2.7ms. CPU is free immediately.
                    asyncEval(idx)

                    // Speculative pread during GPU async window.
                    // Load previous token's experts into persistent buffers.
                    // ~70% will match this token's routing (expert stickiness).
                    // The 1.7ms of pread overlaps with 2.7ms of GPU work.
                    if let prevIds = _previousExpertIds {
                        let specCount = min(prevIds.count, maxBuffers)
                        let specReads = specCount * 3
                        let errState = ThreadSafeError()
                        DispatchQueue.concurrentPerform(iterations: specReads) { i in
                            errState.catchError {
                            let slot = i / 3
                            let proj = i % 3
                            let expertId = prevIds[slot]
                            switch proj {
                            case 0:
                                let ssd = self.gateProj.resolveSSDInfo(expertIndex: expertId) ?? (gateSSD.path, gateSSD.tensorName, UInt32(expertId))
                                MLXFast.preadInto(self._persistentGate![slot], safetensorsPath: ssd.path,
                                                  tensorName: ssd.tensorName, expertIndex: ssd.readIndex)
                            case 1:
                                let ssd = self.upProj.resolveSSDInfo(expertIndex: expertId) ?? (upSSD.path, upSSD.tensorName, UInt32(expertId))
                                MLXFast.preadInto(self._persistentUp![slot], safetensorsPath: ssd.path,
                                                  tensorName: ssd.tensorName, expertIndex: ssd.readIndex)
                            default:
                                let ssd = self.downProj.resolveSSDInfo(expertIndex: expertId) ?? (downSSD.path, downSSD.tensorName, UInt32(expertId))
                                MLXFast.preadInto(self._persistentDown![slot], safetensorsPath: ssd.path,
                                                  tensorName: ssd.tensorName, expertIndex: ssd.readIndex)
                            }
                            }
                        }
                        errState.check()
                    }

                    // Sync on idx (blocks until GPU finishes attention + router)
                    if idx.size == 0 {
                        var outShape = x.shape
                        outShape[outShape.count - 1] = downProj.outputDims
                        let result = MLXArray.zeros(outShape).asType(.float16)
                        if doSort {
                            return MLX.squeezed(scatterUnsort(x: result, invOrder: inverseOrder, shape: indices.shape), axis: -2)
                        }
                        return MLX.squeezed(result, axis: -2)
                    }

                    // Parse actual routing
                    let cpuIndices = idx.asArray(UInt32.self)
                    var ranges = [ExpertRange]()
                    var startIdx = 0
                    while startIdx < cpuIndices.count {
                        let eid = Int(cpuIndices[startIdx])
                        var endIdx = startIdx + 1
                        while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == eid { endIdx += 1 }
                        ranges.append(ExpertRange(id: eid, start: startIdx, end: endIdx))
                        startIdx = endIdx
                    }
                    let actualIds = ranges.map { $0.id }

                    // Map actual experts to persistent buffer slots.
                    // Hits: buffer slot already has correct data from speculative pread.
                    // Misses: assign to a free slot, pread on demand.
                    var usedGate = [MLXArray]()
                    var usedUp = [MLXArray]()
                    var usedDown = [MLXArray]()

                    if let prevIds = _previousExpertIds {
                        var prevSlotMap = [Int: Int]()  // expertId -> buffer slot
                        for (slot, eid) in prevIds.enumerated() {
                            prevSlotMap[eid] = slot
                        }

                        var usedSlots = Set<Int>()
                        var missInfo = [(rangeIdx: Int, expertId: Int, bufferSlot: Int)]()
                        var slotExhausted = false

                        for (ri, r) in ranges.enumerated() {
                            if let slot = prevSlotMap[r.id], !usedSlots.contains(slot) {
                                // HIT: persistent buffer[slot] has correct expert data
                                usedGate.append(_persistentGate![slot])
                                usedUp.append(_persistentUp![slot])
                                usedDown.append(_persistentDown![slot])
                                usedSlots.insert(slot)
                            } else {
                                // MISS: find a free slot
                                guard let freeSlot = (0..<maxBuffers).first(where: { !usedSlots.contains($0) }) else {
                                    // All buffer slots exhausted — fall through to
                                    // full-pread path below (Issue #87)
                                    slotExhausted = true
                                    break
                                }
                                usedGate.append(_persistentGate![freeSlot])
                                usedUp.append(_persistentUp![freeSlot])
                                usedDown.append(_persistentDown![freeSlot])
                                usedSlots.insert(freeSlot)
                                missInfo.append((ri, r.id, freeSlot))
                            }
                        }

                        // Pread only misses (~30% of experts, ~6 reads at QD=6)
                        if !slotExhausted && !missInfo.isEmpty {
                            let totalMissReads = missInfo.count * 3
                            let errState = ThreadSafeError()
                            DispatchQueue.concurrentPerform(iterations: totalMissReads) { [missInfo] i in
                                errState.catchError {
                                let mIdx = i / 3
                                let proj = i % 3
                                let info = missInfo[mIdx]
                                switch proj {
                                case 0:
                                    let ssd = self.gateProj.resolveSSDInfo(expertIndex: info.expertId) ?? (gateSSD.path, gateSSD.tensorName, UInt32(info.expertId))
                                    MLXFast.preadInto(self._persistentGate![info.bufferSlot],
                                                      safetensorsPath: ssd.path,
                                                      tensorName: ssd.tensorName,
                                                      expertIndex: ssd.readIndex)
                                case 1:
                                    let ssd = self.upProj.resolveSSDInfo(expertIndex: info.expertId) ?? (upSSD.path, upSSD.tensorName, UInt32(info.expertId))
                                    MLXFast.preadInto(self._persistentUp![info.bufferSlot],
                                                      safetensorsPath: ssd.path,
                                                      tensorName: ssd.tensorName,
                                                      expertIndex: ssd.readIndex)
                                default:
                                    let ssd = self.downProj.resolveSSDInfo(expertIndex: info.expertId) ?? (downSSD.path, downSSD.tensorName, UInt32(info.expertId))
                                    MLXFast.preadInto(self._persistentDown![info.bufferSlot],
                                                      safetensorsPath: ssd.path,
                                                      tensorName: ssd.tensorName,
                                                      expertIndex: ssd.readIndex)
                                }
                                }
                            }
                            errState.check()
                        }
                    }

                    // Slot exhaustion or no predictions — full pread fallback
                    if usedGate.count != ranges.count {
                        usedGate.removeAll()
                        usedUp.removeAll()
                        usedDown.removeAll()
                        for i in 0..<ranges.count {
                            usedGate.append(_persistentGate![i])
                            usedUp.append(_persistentUp![i])
                            usedDown.append(_persistentDown![i])
                        }
                        let totalReads = ranges.count * 3
                        let errState = ThreadSafeError()
                        DispatchQueue.concurrentPerform(iterations: totalReads) { [ranges] i in
                            errState.catchError {
                            let expertIdx = i / 3
                            let projIdx = i % 3
                            let r = ranges[expertIdx]
                            switch projIdx {
                            case 0:
                                let ssd = self.gateProj.resolveSSDInfo(expertIndex: r.id) ?? (gateSSD.path, gateSSD.tensorName, UInt32(r.id))
                                MLXFast.preadInto(self._persistentGate![expertIdx], safetensorsPath: ssd.path,
                                                  tensorName: ssd.tensorName, expertIndex: ssd.readIndex)
                            case 1:
                                let ssd = self.upProj.resolveSSDInfo(expertIndex: r.id) ?? (upSSD.path, upSSD.tensorName, UInt32(r.id))
                                MLXFast.preadInto(self._persistentUp![expertIdx], safetensorsPath: ssd.path,
                                                  tensorName: ssd.tensorName, expertIndex: ssd.readIndex)
                            default:
                                let ssd = self.downProj.resolveSSDInfo(expertIndex: r.id) ?? (downSSD.path, downSSD.tensorName, UInt32(r.id))
                                MLXFast.preadInto(self._persistentDown![expertIdx], safetensorsPath: ssd.path,
                                                  tensorName: ssd.tensorName, expertIndex: ssd.readIndex)
                            }
                            }
                        }
                        errState.check()
                    }

                    // Update routing for next token's predictions
                    _previousExpertIds = actualIds

                    // Lazy compute (no eval — next layer forces it)
                    let xGate = gateProj.computeExperts(x, buffers: usedGate, ranges: ranges)
                    let xUp = upProj.computeExperts(x, buffers: usedUp, ranges: ranges)
                    let intermediate = activation(xGate) * xUp
                    x = downProj.computeExperts(intermediate, buffers: usedDown, ranges: ranges)
                }

            } else {
                // ── PROMPT PATH: larger batches ──
                // Eval indices first (needed for range count), then allocate exact buffers.
                MLX.eval(idx)

                // Handle empty indices
                if idx.size == 0 {
                    var outShape = x.shape
                    outShape[outShape.count - 1] = downProj.outputDims
                    let result = MLXArray.zeros(outShape).asType(.float16)
                    if doSort {
                        return MLX.squeezed(scatterUnsort(x: result, invOrder: inverseOrder, shape: indices.shape), axis: -2)
                    }
                    return MLX.squeezed(result, axis: -2)
                }

                // Parse expert ranges
                let cpuIndices = idx.asArray(UInt32.self)
                var ranges = [ExpertRange]()
                var startIdx = 0
                while startIdx < cpuIndices.count {
                    let eid = Int(cpuIndices[startIdx])
                    var endIdx = startIdx + 1
                    while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == eid { endIdx += 1 }
                    ranges.append(ExpertRange(id: eid, start: startIdx, end: endIdx))
                    startIdx = endIdx
                }

                // Allocate exact buffer count and eval
                let gateBuffers = gateProj.allocateExpertBuffers(ranges.count)
                let upBuffers = upProj.allocateExpertBuffers(ranges.count)
                let downBuffers = downProj.allocateExpertBuffers(ranges.count)
                MLX.eval(gateBuffers + upBuffers + downBuffers)

                // Concurrent pread (same as fast path)
                let totalReads = ranges.count * 3
                let errState = ThreadSafeError()
                DispatchQueue.concurrentPerform(iterations: totalReads) { [ranges] i in
                    errState.catchError {
                    let expertIdx = i / 3
                    let projIdx = i % 3
                    let r = ranges[expertIdx]
                    switch projIdx {
                    case 0:
                        let ssd = self.gateProj.resolveSSDInfo(expertIndex: r.id) ?? (gateSSD.path, gateSSD.tensorName, UInt32(r.id))
                        timedPread(bytes: gateBuffers[expertIdx].nbytes) {
                            MLXFast.preadInto(gateBuffers[expertIdx], safetensorsPath: ssd.path,
                                              tensorName: ssd.tensorName, expertIndex: ssd.readIndex)
                        }
                    case 1:
                        let ssd = self.upProj.resolveSSDInfo(expertIndex: r.id) ?? (upSSD.path, upSSD.tensorName, UInt32(r.id))
                        timedPread(bytes: upBuffers[expertIdx].nbytes) {
                            MLXFast.preadInto(upBuffers[expertIdx], safetensorsPath: ssd.path,
                                              tensorName: ssd.tensorName, expertIndex: ssd.readIndex)
                        }
                    default:
                        let ssd = self.downProj.resolveSSDInfo(expertIndex: r.id) ?? (downSSD.path, downSSD.tensorName, UInt32(r.id))
                        timedPread(bytes: downBuffers[expertIdx].nbytes) {
                            MLXFast.preadInto(downBuffers[expertIdx], safetensorsPath: ssd.path,
                                              tensorName: ssd.tensorName, expertIndex: ssd.readIndex)
                        }
                    }
                    }
                }
                errState.check()

                // Lazy compute (no eval — next layer forces it)
                let xGate = gateProj.computeExperts(x, buffers: gateBuffers, ranges: ranges)
                let xUp = upProj.computeExperts(x, buffers: upBuffers, ranges: ranges)
                let intermediate = activation(xGate) * xUp
                x = downProj.computeExperts(intermediate, buffers: downBuffers, ranges: ranges)
            }

            if doSort {
                x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
            }
            return MLX.squeezed(x, axis: -2)
        }

        // ── Fallback: original sequential path (non-SSD or non-quantized) ──
        let xUp = upProj(x, idx, sortedIndices: doSort)
        let xGate = gateProj(x, idx, sortedIndices: doSort)
        x = downProj(
            activation(xGate) * xUp,
            idx,
            sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") public var weight: MLXArray
    @ModuleInfo(key: "bias") public var bias: MLXArray?
    public var weightScaleInv: MLXArray?

    // SSD streaming map for unstacked experts: expertId -> (path, tensorName)
    public var unstackedSSDMap: [Int: (path: String, tensorName: String)]?
    public var tensorName: String?

    public let inputDims: Int
    public let outputDims: Int
    public let numExperts: Int

    public func resolveSSDInfo() -> (path: String, tensorName: String)? {
        #if os(macOS)
        guard ExpertStreamingConfig.shared.useDirectNVMe else { return nil }
        if let map = unstackedSSDMap, let first = map[0] {
            return (first.path, first.tensorName)
        }
        guard let tName = self.tensorName,
              let filename = ExpertStreamerManager.shared?.getFile(for: tName),
              let dir = ExpertStreamingConfig.shared.modelDirectory else { return nil }
        let path = dir.appendingPathComponent(filename).path
        return (path, tName)
        #else
        return nil
        #endif
    }

    public func resolveSSDInfo(expertIndex: Int) -> (path: String, tensorName: String, readIndex: UInt32)? {
        #if os(macOS)
        guard ExpertStreamingConfig.shared.useDirectNVMe else { return nil }
        if let unstacked = self.unstackedSSDMap?[expertIndex] {
            return (unstacked.path, unstacked.tensorName, 0)
        }
        guard let base = resolveSSDInfo() else { return nil }
        return (base.path, base.tensorName, UInt32(expertIndex))
        #else
        return nil
        #endif
    }

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = MLXArray.zeros([numExperts, outputDims, inputDims], type: Float16.self)

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        // weightScaleInv is a plain var (not @ModuleInfo), populated dynamically.
        // Expert weights are pre-dequanted in sanitize; no loader population needed.

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }
    
    private lazy var fp8GatherGemvKernel = {
        let metalSource = """
            uint base_row = threadgroup_position_in_grid.x * ROWS_PER_TG;
            uint token_idx = threadgroup_position_in_grid.y;
            uint ti_idx = thread_position_in_threadgroup.x;
            uint tg_size = threads_per_threadgroup.x;
            
            int expert_idx = indices[token_idx];
            if (expert_idx < 0 || expert_idx >= NUM_EXPERTS) {
                if (ti_idx == 0) {
                    for (uint r = 0; r < ROWS_PER_TG; r++) {
                        uint row = base_row + r;
                        if (row < OUT_DIM) out[token_idx * OUT_DIM + row] = (bfloat)0.0f;
                    }
                }
                return;
            }
            
            int scale_cols = (IN_DIM + BS - 1) / BS;
            int scale_expert_offset = expert_idx * ((OUT_DIM + BS - 1)/BS) * scale_cols;
            int w_expert_offset = expert_idx * OUT_DIM * IN_DIM;
            
            device const uint8_t *w_expert = (device const uint8_t *)w + w_expert_offset;
            device const bfloat *scales_expert = (device const bfloat *)scales + scale_expert_offset;
            device const bfloat *x_token = (device const bfloat *)x + token_idx * IN_DIM;
            
            for (uint r = 0; r < ROWS_PER_TG; r++) {
                uint row = base_row + r;
                if (row >= OUT_DIM) continue;
                
                float sum = 0.0f;
                for (int col = ti_idx; col < IN_DIM; col += tg_size) {
                    int scale_idx = (row / BS) * scale_cols + (col / BS);
                    float scale_val = (float)scales_expert[scale_idx];
                    
                    uint8_t w_byte = w_expert[row * IN_DIM + col];
                    
                    float w_val = 0.0f;
                    if (w_byte != 0 && w_byte != 0x80) {
                        uint s = (w_byte >> 7) & 1;
                        uint e = (w_byte >> 3) & 0xF;
                        uint m = w_byte & 0x7;
                        float sign = s ? -1.0f : 1.0f;
                        if (e == 0) {
                            w_val = sign * exp2(-6.0f) * (m / 8.0f);
                        } else if (!(e == 15 && m == 7)) {
                            w_val = sign * exp2(float(e) - 7.0f) * (1.0f + m / 8.0f);
                        }
                    }
                    
                    w_val *= scale_val;
                    float x_val = (float)x_token[col];
                    
                    sum += w_val * x_val;
                }
                
                threadgroup float shared_sum[1024];
                shared_sum[ti_idx] = sum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
                    if (ti_idx < stride) {
                        shared_sum[ti_idx] += shared_sum[ti_idx + stride];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                
                if (ti_idx == 0) {
                    out[token_idx * OUT_DIM + row] = (bfloat)shared_sum[0];
                }
            }
        """
        return { (rowsPerTg: Int) in
            let actualSource = """
            #define IN_DIM \(self.inputDims)
            #define OUT_DIM \(self.outputDims)
            #define NUM_EXPERTS \(self.numExperts)
            #define BS 128
            #define ROWS_PER_TG \(rowsPerTg)
            
            \(metalSource)
            """
            return MLXFast.metalKernel(
                name: "fp8_gather_gemv",
                inputNames: ["x", "w", "scales", "indices"],
                outputNames: ["out"],
                source: actualSource
            )
        }
    }()
    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let w = self.weight
        var result: MLXArray
        
        if let inv = self.weightScaleInv, inv.size > 0 {
            var numTokens = x.size / inputDims
            if numTokens == 0 {
                var outShape = x.shape
                outShape[outShape.count - 1] = outputDims
                return MLXArray.zeros(outShape).asType(x.dtype)
            }
            
            var x = x
            let expectedXShape = Array(indices.shape) + [inputDims]
            
            // If x doesn't match the expected broadcasting shape for indices, broadcast it
            // gatherMM natively broadcasts, but our fp8GatherGemvKernel does not.
            if x.shape != expectedXShape && x.size < indices.size * inputDims {
                var xToBroadcast = x
                if x.ndim == 5 {
                    // x is [B, S, 1, 1, D], we need [B, S, 1, D] to broadcast to [B, S, topK, D]
                    xToBroadcast = x.reshaped([x.dim(0), x.dim(1), 1, inputDims])
                }
                x = MLX.broadcast(xToBroadcast, to: expectedXShape)
            }
            
            numTokens = x.size / inputDims
            
            let xFlat = x.reshaped([numTokens, inputDims]).contiguous()
            let indicesFlat = indices.reshaped([numTokens]).contiguous()
            
            let outShape = [numTokens, outputDims]
            let safeInv = inv.asType(.bfloat16).contiguous()
            let wContig = w.contiguous()
            
            let isBatch = numTokens >= 64
            let rowsPerTg = isBatch ? 16 : 1
            let outDimGrid = (outputDims + rowsPerTg - 1) / rowsPerTg
            
            result = fp8GatherGemvKernel(rowsPerTg)(
                [xFlat, wContig, safeInv, indicesFlat],
                grid: (outDimGrid * 256, numTokens, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [outShape],
                outputDTypes: [x.dtype]
            )[0]
            result = result.reshaped(Array(x.shape.dropLast()) + [outputDims])
        } else {
            let weightT = w.swappedAxes(-1, -2)
            result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)
        }

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    // MARK: - Cross-projection batching helpers (SSD streaming)

    /// Allocate zero-filled weight buffers for `count` experts (lazy, not yet eval'd).
    public func allocateExpertBuffers(_ count: Int) -> [MLXArray] {
        var buffers = [MLXArray]()
        for _ in 0..<count {
            buffers.append(MLXArray.zeros([1, self.outputDims, self.inputDims]).asType(self.weight.dtype))
        }
        return buffers
    }

    public func computeExperts(_ x: MLXArray, buffers: [MLXArray], ranges: [ExpertRange]) -> MLXArray {
        var expertResults = [MLXArray]()
        for (i, r) in ranges.enumerated() {
            let rangeX = x[r.start ..< r.end]
            let expertIndices = MLXArray.zeros([rangeX.dim(0)], type: UInt32.self)
            
            var w = buffers[i]
            // DUMMY DEPENDENCY: Prevent MLX from caching fromFp8.
            // Since `buffers[i]` is mutated via C++ memcpy (preadInto), MLX doesn't know it changed.
            // We use a random value that evaluates to 0 (uint8) to force a new graph node.
            let dummy = MLXRandom.uniform(low: 0.0, high: 0.1).asType(.uint8)
            w = w + dummy

            if let inv = self.weightScaleInv, inv.size > 0 {
                // Swift MLX safetensors loader maps F8_E4M3 → uint8 (raw bit patterns).
                // We must call MLXFast.fromFp8 explicitly to get the same signed float values.
                let wFp = MLXFast.fromFp8(w, dtype: .bfloat16)
                // w is [1, outDim, inDim] (one expert loaded from SSD)
                let bs = 128
                let (m, n) = (wFp.dim(1), wFp.dim(2))
                let outBlocks = (m + bs - 1) / bs
                let inBlocks  = (n + bs - 1) / bs
                let padBottom = (bs - m % bs) % bs
                let padSide   = (bs - n % bs) % bs
                if i == 0 {

                }
                var padded = MLX.padded(wFp, widths: [IntOrPair((0, 0)), IntOrPair((0, padBottom)), IntOrPair((0, padSide))])
                if i == 0 {

                }
                padded = padded.reshaped([1, outBlocks, bs, inBlocks, bs])

                // inv may be:
                //  - 3D stacked: [numExperts, outBlocks, inBlocks]  (memory-resident path)
                //  - 2D per-expert: [outBlocks, inBlocks]           (if loaded directly)
                let invSlice: MLXArray
                if inv.ndim == 3 {
                    // Stacked: pick this expert's row and unsqueeze batch dim
                    invSlice = inv[r.id ..< r.id + 1]  // [1, outBlocks, inBlocks]
                    if i == 0 {

                    }
                } else {
                    // Already 2D — unsqueeze for batch broadcast
                    invSlice = MLX.expandedDimensions(inv, axis: 0)  // [1, outBlocks, inBlocks]
                    if i == 0 {

                    }
                }

                let scaled = padded * invSlice[0..., 0..., .newAxis, 0..., .newAxis]
                let dequantized = scaled.reshaped([1, m + padBottom, n + padSide])[0..., 0 ..< m, 0 ..< n]
                w = dequantized.asType(x.dtype)
            } else {
                if i == 0 { print("[SwitchLayers] computeExperts: NO weightScaleInv found! w shape=\(w.shape), dtype=\(w.dtype)") }
            }

            var expertOutput = MLX.gatherMM(
                rangeX, w.swappedAxes(-1, -2),
                rhsIndices: expertIndices,
                sortedIndices: true
            )
            if let bias = self.bias {
                let biasSlice = bias[r.id ..< r.id + 1]
                expertOutput = expertOutput + MLX.expandedDimensions(biasSlice[expertIndices], axis: -2)
            }
            let leadingShape = Array(rangeX.shape.dropLast())
            let canonicalShape = leadingShape + [self.outputDims]
            if expertOutput.shape != canonicalShape {
                expertOutput = expertOutput.reshaped(canonicalShape)
            }
            expertResults.append(expertOutput)
        }
        return MLX.concatenated(expertResults, axis: 0)
    }

    public func computeExpertsFused(
        _ x: MLXArray, stackedBuffer: MLXArray, slotPerToken: MLXArray, slotExperts: [Int32]
    ) -> MLXArray {
        // Fallback for unquantized/FP8 - emulate the fused gather by evaluating active slots sequentially
        let slots = slotPerToken.asArray(Int32.self)
        if slots.isEmpty {
            return MLXArray.zeros(x.shape).asType(x.dtype)
        }
        
        var currentSlot = slots.first ?? 0
        var currentStart = 0
        var ranges = [(slot: Int32, start: Int, end: Int)]()
        for (i, slot) in slots.enumerated() {
            if slot != currentSlot {
                ranges.append((slot: currentSlot, start: currentStart, end: i))
                currentSlot = slot
                currentStart = i
            }
        }
        ranges.append((slot: currentSlot, start: currentStart, end: slots.count))

        var expertResults = [MLXArray]()
        for r in ranges {
            let expertId = Int(slotExperts[Int(r.slot)])
            let rangeX = x[r.start ..< r.end]
            let expertIndices = MLXArray.zeros([rangeX.dim(0)], type: UInt32.self)
            
            var w = stackedBuffer[Int(r.slot)][.newAxis, 0..., 0...] // [1, outDim, inDim]
            
            // CACHE BREAKER: Invalidate MLX graph cache for this buffer slot.
            // Since we mutate the underlying memory via pread (C++), we must change the ID.
            let dummy = MLXRandom.uniform(low: 0.0, high: 0.001)
            w = MLX.depends(input: w, dependencies: [dummy])
            
            if let inv = self.weightScaleInv, inv.size > 0 {
                let wFp = MLXFast.fromFp8(w, dtype: .bfloat16)
                
                MLX.eval(wFp)
                
                let bs = 128
                let (m, n) = (wFp.dim(1), wFp.dim(2))
                let padBottom = (bs - m % bs) % bs
                let padSide   = (bs - n % bs) % bs
                var padded = MLX.padded(wFp, widths: [IntOrPair((0, 0)), IntOrPair((0, padBottom)), IntOrPair((0, padSide))])
                padded = padded.reshaped([wFp.dim(0), (m + padBottom) / bs, bs, (n + padSide) / bs, bs])
                let invSlice = inv[expertId ..< expertId + 1]
                let scaled = padded * invSlice[0..., 0..., .newAxis, 0..., .newAxis]
                let dequantized = scaled.reshaped([wFp.dim(0), m + padBottom, n + padSide])[0..., 0 ..< m, 0 ..< n]
                w = dequantized.asType(x.dtype)
            } else {
                print("[SwitchLayers] computeExpertsFused: FATAL ERROR: NO weightScaleInv found! w shape=\(w.shape), dtype=\(w.dtype)")
                fflush(stdout)
            }
            
            var expertOutput = MLX.gatherMM(
                rangeX, w.swappedAxes(-1, -2),
                rhsIndices: expertIndices,
                sortedIndices: true
            )
            
            if let bias = self.bias {
                let biasSlice = bias[expertId ..< expertId + 1]
                expertOutput = expertOutput + MLX.expandedDimensions(biasSlice[expertIndices], axis: -2)
            }
            
            let leadingShape = Array(rangeX.shape.dropLast())
            let canonicalShape = leadingShape + [self.outputDims]
            if expertOutput.shape != canonicalShape {
                expertOutput = expertOutput.reshaped(canonicalShape)
            }
            expertResults.append(expertOutput)
        }
        
        return MLX.concatenated(expertResults, axis: 0)
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        if ExpertStreamingConfig.shared.isEnabled {
            SSDFetchPool.shared.waitAll()
            MLX.eval(indices)
            if indices.size == 0 {
                var outShape = x.shape
                outShape[outShape.count - 1] = self.outputDims
                return MLXArray.zeros(outShape).asType(.float16)
            }

            let cpuIndices = indices.asArray(UInt32.self)
            var expertResults = [MLXArray]()
            var startIdx = 0

            // macOS directNVMe: resolve the safetensors shard + tensor offset once.
            // iOS mmapPageCache: ssdInfo = nil → falls through to mmap prefault below.
            let ssdInfo: (path: String, tensorName: String)? = {
                #if os(macOS)
                guard ExpertStreamingConfig.shared.useDirectNVMe,
                      let tName = self.tensorName,
                      let filename = ExpertStreamerManager.shared?.getFile(for: tName),
                      let dir = ExpertStreamingConfig.shared.modelDirectory else { return nil }
                let path = dir.appendingPathComponent(filename).path
                return (path, tName)
                #else
                return nil  // iOS always uses mmap fallback
                #endif
            }()

            // ---- Parse expert ranges ----
            var ranges = [ExpertRange]()
            while startIdx < cpuIndices.count {
                let eid = Int(cpuIndices[startIdx])
                var endIdx = startIdx + 1
                while endIdx < cpuIndices.count && Int(cpuIndices[endIdx]) == eid { endIdx += 1 }
                ranges.append(ExpertRange(id: eid, start: startIdx, end: endIdx))
                startIdx = endIdx
            }

            if let info = ssdInfo {
                // ---- Batch-allocate weight buffers (1 eval for all) ----
                var buffers = [MLXArray]()
                for _ in ranges {
                    buffers.append(MLXArray.zeros([1, self.weight.dim(1), self.weight.dim(2)]).asType(self.weight.dtype))
                }
                MLX.eval(buffers)

                // ---- Concurrent pread into each fresh buffer ----
                // (was sequential — single-threaded pread leaves most of the
                // NVMe queue-depth bandwidth on the table during prefill)
                let preadErr = ThreadSafeError()
                DispatchQueue.concurrentPerform(iterations: ranges.count) { [ranges, buffers] i in
                    preadErr.catchError {
                        let r = ranges[i]
                        let ssd = self.resolveSSDInfo(expertIndex: r.id) ?? (info.path, info.tensorName, UInt32(r.id))
                        timedPread(bytes: buffers[i].nbytes) {
                            MLXFast.preadInto(
                                buffers[i],
                                safetensorsPath: ssd.path,
                                tensorName: ssd.tensorName,
                                expertIndex: ssd.readIndex
                            )
                        }
                    }
                }
                preadErr.check()

                // ---- GPU compute for all experts ----
                for (i, r) in ranges.enumerated() {
                    let rangeX = x[r.start ..< r.end]
                    let expertIndices = MLXArray.zeros([rangeX.dim(0)], type: UInt32.self)
                    let expertScales = self.scales[r.id ..< r.id + 1]
                    var expertBiases: MLXArray? = nil
                    if let b = self.biases { expertBiases = b[r.id ..< r.id + 1] }

                    var expertOutput = MLX.gatherQuantizedMM(
                        rangeX, buffers[i],
                        scales: expertScales, biases: expertBiases,
                        rhsIndices: expertIndices, transpose: true,
                        groupSize: self.groupSize, bits: self.bits, mode: mode, sortedIndices: true
                    )
                    if let bias = self.bias {
                        let biasSlice = bias[r.id ..< r.id + 1]
                        expertOutput = expertOutput + MLX.expandedDimensions(biasSlice[expertIndices], axis: -2)
                    }
                    let leadingShape = Array(rangeX.shape.dropLast())
                    let canonicalShape = leadingShape + [self.outputDims]
                    if expertOutput.shape != canonicalShape {
                        expertOutput = expertOutput.reshaped(canonicalShape)
                    }
                    expertResults.append(expertOutput)
                }
            } else {
                // iOS mmap fallback — original sequential path with per-expert eval
                for r in ranges {
                    let rangeX = x[r.start ..< r.end]
                    let expertIndices = MLXArray.zeros([rangeX.dim(0)], type: UInt32.self)
                    let w = self.weight[r.id ..< r.id + 1]
                    MLX.eval(w)
                    MLXFast.prefault(w)
                    let expertScales = self.scales[r.id ..< r.id + 1]
                    var expertBiases: MLXArray? = nil
                    if let b = self.biases { expertBiases = b[r.id ..< r.id + 1] }
                    var expertOutput = MLX.gatherQuantizedMM(
                        rangeX, w,
                        scales: expertScales, biases: expertBiases,
                        rhsIndices: expertIndices, transpose: true,
                        groupSize: self.groupSize, bits: self.bits, mode: mode, sortedIndices: true
                    )
                    if let bias = self.bias {
                        let biasSlice = bias[r.id ..< r.id + 1]
                        expertOutput = expertOutput + MLX.expandedDimensions(biasSlice[expertIndices], axis: -2)
                    }
                    let leadingShape = Array(rangeX.shape.dropLast())
                    let canonicalShape = leadingShape + [self.outputDims]
                    if expertOutput.shape != canonicalShape {
                        expertOutput = expertOutput.reshaped(canonicalShape)
                    }
                    MLX.eval(expertOutput)
                    expertResults.append(expertOutput)
                }
            }

            // Batch eval all expert outputs at once (directNVMe path)
            if let _ = ssdInfo, !expertResults.isEmpty {
                MLX.eval(expertResults)
            }

            if expertResults.isEmpty {
                var outShape = x.shape
                outShape[outShape.count - 1] = self.outputDims
                return MLXArray.zeros(outShape).asType(.float16)
            }
            // PAPPS Heuristic: Prefetch exactly these experts so they are in cache for the N+1 token.
            if let _ = ssdInfo {
                let uniqueIndices = Set(cpuIndices)
                for _ in uniqueIndices {
                    // MLXFast.pappsPrefetch(
                    //     safetensorsPath: info.path,
                    //     tensorName: info.tensorName,
                    //     expertIndex: idx
                    // )
                }
            }

            return MLX.concatenated(expertResults, axis: 0)
        }

        var result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }


    /// Allocate zero-filled weight buffers for `count` experts (lazy, not yet eval'd).
    override public func allocateExpertBuffers(_ count: Int) -> [MLXArray] {
        var buffers = [MLXArray]()
        for _ in 0..<count {
            buffers.append(MLXArray.zeros([1, self.weight.dim(1), self.weight.dim(2)]).asType(self.weight.dtype))
        }
        return buffers
    }

    /// Load expert weights from SSD into pre-allocated (eval'd) buffers.
    public func loadExpertWeights(_ buffers: [MLXArray], ranges: [ExpertRange], ssdInfo: (path: String, tensorName: String)) {
        for (i, r) in ranges.enumerated() {
            let ssd = self.resolveSSDInfo(expertIndex: r.id) ?? (ssdInfo.path, ssdInfo.tensorName, UInt32(r.id))
            MLXFast.preadInto(
                buffers[i],
                safetensorsPath: ssd.path,
                tensorName: ssd.tensorName,
                expertIndex: ssd.readIndex
            )
        }
    }

    /// Compute expert outputs using pre-loaded weight buffers. Returns LAZY result (no eval).
    override public func computeExperts(_ x: MLXArray, buffers: [MLXArray], ranges: [ExpertRange]) -> MLXArray {
        var expertResults = [MLXArray]()
        for (i, r) in ranges.enumerated() {
            let rangeX = x[r.start ..< r.end]
            let expertIndices = MLXArray.zeros([rangeX.dim(0)], type: UInt32.self)
            let expertScales = self.scales[r.id ..< r.id + 1]
            var expertBiases: MLXArray? = nil
            if let b = self.biases { expertBiases = b[r.id ..< r.id + 1] }

            var expertOutput = MLX.gatherQuantizedMM(
                rangeX, buffers[i],
                scales: expertScales, biases: expertBiases,
                rhsIndices: expertIndices, transpose: true,
                groupSize: self.groupSize, bits: self.bits, mode: mode, sortedIndices: true
            )
            if let bias = self.bias {
                let biasSlice = bias[r.id ..< r.id + 1]
                expertOutput = expertOutput + MLX.expandedDimensions(biasSlice[expertIndices], axis: -2)
            }
            let leadingShape = Array(rangeX.shape.dropLast())
            let canonicalShape = leadingShape + [self.outputDims]
            if expertOutput.shape != canonicalShape {
                expertOutput = expertOutput.reshaped(canonicalShape)
            }
            expertResults.append(expertOutput)
        }

        if expertResults.isEmpty {
            var outShape = x.shape
            outShape[outShape.count - 1] = self.outputDims
            return MLXArray.zeros(outShape).asType(.float16)
        }
        return MLX.concatenated(expertResults, axis: 0)
    }

    /// Stacked-buffer fused-matmul variant of `computeExperts`. Replaces the
    /// per-expert `gatherQuantizedMM` loop (one dispatch per expert) with a
    /// single dispatch over the full stacked weight buffer.
    ///
    /// - Parameters:
    ///   - x: input activations, shape `[totalTokens, ..., inputDims]`.
    ///   - stackedBuffer: weight buffer, shape `[CACHE_SLOTS, outputDims, inputDims]`.
    ///       Slots are populated externally via `MLXFast.preadIntoOffset`.
    ///   - slotPerToken: uint32 array mapping each token (along axis 0 of `x`)
    ///       to a slot index in `stackedBuffer`. Built from the routing.
    ///   - slotExperts: per-slot expert IDs (`0..<numExperts`). Used to gather
    ///       per-slot scales/biases from `self.scales` and `self.biases`.
    override public func computeExpertsFused(
        _ x: MLXArray,
        stackedBuffer: MLXArray,
        slotPerToken: MLXArray,
        slotExperts: [Int32]
    ) -> MLXArray {
        let slotExpertsMLX = MLXArray(slotExperts).asType(.uint32)
        // Gather scales/biases for the experts currently in our slots.
        // Result shape: [N_slots, outputDims, inputDims / groupSize].
        let stackedScales = MLX.take(self.scales, slotExpertsMLX, axis: 0)
        var stackedBiases: MLXArray? = nil
        if let b = self.biases { stackedBiases = MLX.take(b, slotExpertsMLX, axis: 0) }

        var output = MLX.gatherQuantizedMM(
            x, stackedBuffer,
            scales: stackedScales,
            biases: stackedBiases,
            rhsIndices: slotPerToken,
            transpose: true,
            groupSize: self.groupSize, bits: self.bits, mode: mode, sortedIndices: true
        )

        // Optional per-token bias add (gathered from per-slot bias).
        if let bias = self.bias {
            let stackedBias = MLX.take(bias, slotExpertsMLX, axis: 0)             // [N_slots, outputDims]
            let perTokenBias = MLX.take(stackedBias, slotPerToken, axis: 0)       // [tokens, outputDims]
            output = output + MLX.expandedDimensions(perTokenBias, axis: -2)
        }

        let leadingShape = Array(x.shape.dropLast())
        let canonicalShape = leadingShape + [self.outputDims]
        if output.shape != canonicalShape {
            output = output.reshaped(canonicalShape)
        }
        return output
    }
}

public class ExpertStreamerManager {
    nonisolated(unsafe) public static var shared: ExpertStreamerManager?

    public let weightMap: [String: String]

    public init(modelDirectory: URL) {
        var map = [String: String]()
        let indexUrl = modelDirectory.appendingPathComponent("model.safetensors.index.json")
        if let data = try? Data(contentsOf: indexUrl),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let weightMapJson = json["weight_map"] as? [String: String] {
            map = weightMapJson
        }
        self.weightMap = map
    }

    public func getFile(for tensorName: String) -> String? {
        return weightMap[tensorName]
    }
}

/// Live telemetry for the SSD expert-streaming hot path.
///
/// Fed directly from the Swift `preadInto`/`preadIntoOffset` call sites (the
/// C-level `mlx_ssd_metrics` counters are only incremented by the legacy
/// `LoadSSDExpert` op, which the stacked/pread paths do not use). Everything
/// the streaming optimization loop needs as an objective lives here:
/// bytes-streamed-per-token, slot-cache hit rate, and NVMe throughput.
///
/// Emits a one-line summary every 10 s (only when streaming traffic occurred).
public final class SSDStreamMetrics: @unchecked Sendable {
    public static let shared = SSDStreamMetrics()

    // Lifetime counters (never reset).
    private var lifetimeBytes: UInt64 = 0
    private var lifetimeReads: UInt64 = 0
    private var lifetimeIONs: UInt64 = 0
    private var lifetimeHits: UInt64 = 0
    private var lifetimeMisses: UInt64 = 0
    private var lifetimePrefetchReads: UInt64 = 0
    private var lifetimeTokens: UInt64 = 0
    // Page-cache split (EXP-000a): a pread served from the page cache runs at
    // memcpy speed (≫8 GB/s); a real NVMe random read of an expert slab can't
    // (≤~4 GB/s). Classify per read by effective throughput. Heuristic, but
    // validated in the F_NOCACHE floor state where cached must read ~0.
    private var lifetimeCachedBytes: UInt64 = 0
    private var lifetimeCachedReads: UInt64 = 0
    private var lifetimeDeviceIONs: UInt64 = 0

    // Current 10 s window (reset on each log).
    private var windowBytes: UInt64 = 0
    private var windowReads: UInt64 = 0
    private var windowIONs: UInt64 = 0
    private var windowHits: UInt64 = 0
    private var windowMisses: UInt64 = 0
    private var windowTokens: UInt64 = 0
    private var windowCachedBytes: UInt64 = 0
    private var windowCachedReads: UInt64 = 0
    private var windowDeviceIONs: UInt64 = 0
    private var windowStartNs: UInt64 = DispatchTime.now().uptimeNanoseconds

    private let lock = NSLock()
    private static let logIntervalNs: UInt64 = 10_000_000_000  // 10 s
    /// Reads faster than this are classified as page-cache-served.
    /// bytes/ns == GB/s; Apple NVMe random read tops out well below this.
    private static let pageCacheGBpsThreshold = 8.0

    /// Record one pread of expert bytes (call from every pread site).
    /// - Parameters:
    ///   - bytes: bytes read from NVMe
    ///   - timeNs: wall-clock duration of the pread
    ///   - prefetch: true when issued speculatively (prev-token experts)
    public func record(bytes: Int, timeNs: UInt64, prefetch: Bool = false) {
        let cached = Double(bytes) / Double(max(timeNs, 1)) >= Self.pageCacheGBpsThreshold
        lock.lock()
        lifetimeBytes += UInt64(bytes)
        lifetimeReads += 1
        lifetimeIONs += timeNs
        if prefetch { lifetimePrefetchReads += 1 }
        if cached {
            lifetimeCachedBytes += UInt64(bytes)
            lifetimeCachedReads += 1
            windowCachedBytes += UInt64(bytes)
            windowCachedReads += 1
        } else {
            lifetimeDeviceIONs += timeNs
            windowDeviceIONs += timeNs
        }
        windowBytes += UInt64(bytes)
        windowReads += 1
        windowIONs += timeNs
        let line = logLineIfDueLocked()
        lock.unlock()
        if let line { print(line); fflush(stdout) }
    }

    /// Record slot-cache resolution results for one layer call.
    public func recordResolution(hits: Int, misses: Int) {
        lock.lock()
        lifetimeHits += UInt64(hits)
        lifetimeMisses += UInt64(misses)
        windowHits += UInt64(hits)
        windowMisses += UInt64(misses)
        lock.unlock()
    }

    /// Mark one generated token (drives bytes/token in the log + snapshot).
    public func tokenTick() {
        lock.lock()
        lifetimeTokens += 1
        windowTokens += 1
        let line = logLineIfDueLocked()
        lock.unlock()
        if let line { print(line); fflush(stdout) }
    }

    /// Builds the 10 s summary line and resets the window. Must hold `lock`.
    private func logLineIfDueLocked() -> String? {
        let now = DispatchTime.now().uptimeNanoseconds
        guard now - windowStartNs >= Self.logIntervalNs, windowReads > 0 else { return nil }
        let dt = Double(now - windowStartNs) / 1e9
        let devBytes = windowBytes - windowCachedBytes
        let devReads = windowReads - windowCachedReads
        let devMbps = Double(devBytes) / (1024 * 1024) / dt
        let pcMbps = Double(windowCachedBytes) / (1024 * 1024) / dt
        let resolved = windowHits + windowMisses
        let hitPct = resolved > 0 ? 100.0 * Double(windowHits) / Double(resolved) : 0
        let devAvgMs = devReads > 0 ? Double(windowDeviceIONs) / 1e6 / Double(devReads) : 0
        let gib = Double(1 << 30)
        let gbPerTok = windowTokens > 0
            ? Double(windowBytes) / Double(windowTokens) / gib : 0
        let devGbPerTok = windowTokens > 0
            ? Double(devBytes) / Double(windowTokens) / gib : 0
        let line = String(
            format: "[⚡️ SSD Stream] dev %.0f MB/s | pc %.0f MB/s | %llu reads | dev avg %.2f ms | hit %.0f%% | %.2f GB/tok (dev %.2f) | total %.1f GB",
            devMbps, pcMbps, windowReads, devAvgMs, hitPct, gbPerTok, devGbPerTok,
            Double(lifetimeBytes) / gib)
        windowBytes = 0; windowReads = 0; windowIONs = 0
        windowHits = 0; windowMisses = 0; windowTokens = 0
        windowCachedBytes = 0; windowCachedReads = 0; windowDeviceIONs = 0
        windowStartNs = now
        return line
    }

    /// Point-in-time totals for /metrics endpoints and benchmark reports.
    public struct Snapshot: Sendable {
        public let totalBytesRead: UInt64
        public let totalReads: UInt64
        public let totalIONs: UInt64
        public let cacheHits: UInt64
        public let cacheMisses: UInt64
        public let prefetchReads: UInt64
        public let tokens: UInt64
        /// Page-cache-served preads (effective throughput ≥ 8 GB/s).
        public let cachedBytes: UInt64
        public let cachedReads: UInt64
        /// Wall-time of device-classified reads only.
        public let deviceIONs: UInt64

        public var hitRate: Double {
            let r = cacheHits + cacheMisses
            return r > 0 ? Double(cacheHits) / Double(r) : 0
        }
        public var bytesPerToken: Double {
            tokens > 0 ? Double(totalBytesRead) / Double(tokens) : 0
        }
        public var deviceBytes: UInt64 { totalBytesRead - cachedBytes }
        public var deviceReads: UInt64 { totalReads - cachedReads }
        public var deviceBytesPerToken: Double {
            tokens > 0 ? Double(deviceBytes) / Double(tokens) : 0
        }
    }

    public func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            totalBytesRead: lifetimeBytes, totalReads: lifetimeReads,
            totalIONs: lifetimeIONs, cacheHits: lifetimeHits,
            cacheMisses: lifetimeMisses, prefetchReads: lifetimePrefetchReads,
            tokens: lifetimeTokens,
            cachedBytes: lifetimeCachedBytes, cachedReads: lifetimeCachedReads,
            deviceIONs: lifetimeDeviceIONs)
    }
}

/// Times one pread call and reports it to `SSDStreamMetrics`.
@inline(__always)
func timedPread(bytes: Int, prefetch: Bool = false, _ body: () -> Void) {
    let t0 = DispatchTime.now().uptimeNanoseconds
    body()
    SSDStreamMetrics.shared.record(
        bytes: bytes, timeNs: DispatchTime.now().uptimeNanoseconds - t0, prefetch: prefetch)
}


/// EXP-000c: opt-in router-decision trace for offline analysis (stickiness,
/// per-expert frequency histograms, speculative-verify expert unions).
///
/// Enable with `MLX_ROUTER_TRACE=/path/to/trace.csv`. Disabled (and free) when
/// the env var is unset. One CSV line per streamed MoE layer call:
///
///     t=<token>,l=<layer-key>,e=<id>+<id>+...
///
/// where `t` is the generated-token counter (ticks with SSDStreamMetrics) and
/// `e` lists the UNIQUE experts the layer call resolved (decode B=1 → the
/// token's top-k; batched prefill/verify → the batch's expert union).
public final class RouterTrace: @unchecked Sendable {
    public static let shared = RouterTrace()

    private let handle: FileHandle?
    private var token: UInt64 = 0
    private var pending = ""
    private var pendingLines = 0
    private let lock = NSLock()
    private static let flushEvery = 256

    private init() {
        if let path = ProcessInfo.processInfo.environment["MLX_ROUTER_TRACE"],
            !path.isEmpty
        {
            FileManager.default.createFile(atPath: path, contents: nil)
            handle = FileHandle(forWritingAtPath: path)
            FileHandle.standardError.write(
                Data("[SSD] router trace → \(path)\n".utf8))
        } else {
            handle = nil
        }
    }

    /// Mark one generated token (called alongside SSDStreamMetrics.tokenTick).
    public func tokenTick() {
        guard handle != nil else { return }
        lock.lock()
        token += 1
        lock.unlock()
    }

    /// Record one streamed MoE layer call's resolved expert set.
    public func record(layerKey: String, expertIds: [Int]) {
        guard let handle else { return }
        let ids = expertIds.map(String.init).joined(separator: "+")
        lock.lock()
        pending += "t=\(token),l=\(layerKey),e=\(ids)\n"
        pendingLines += 1
        let out: Data? =
            pendingLines >= Self.flushEvery ? Data(pending.utf8) : nil
        if out != nil {
            pending = ""
            pendingLines = 0
        }
        lock.unlock()
        if let out { handle.write(out) }
    }

    /// Flush buffered lines (server shutdown / end of benchmark).
    public func flush() {
        guard let handle else { return }
        lock.lock()
        let out = Data(pending.utf8)
        pending = ""
        pendingLines = 0
        lock.unlock()
        if !out.isEmpty { handle.write(out) }
    }
}
/// Asynchronous SSD fetch pool for expert slot misses.
///
/// Miss preads used to run synchronously in the decode thread (a hard
/// ~2-4 ms stall per layer-with-misses). Submitting them here lets the
/// fetch overlap the graph-building work between MoE layers. Correctness
/// contract: `waitAll()` MUST run before anything can trigger execution of
/// a graph that reads the slot buffers — i.e. at every SwitchGLU /
/// QuantizedSwitchLinear entry (covers the per-layer `idx.asArray()` /
/// `asyncEval`) and in the generation loop before evaluating logits.
/// MLX_MOE_ASYNC_FETCH=0 reverts to synchronous fetching.
public final class SSDFetchPool: @unchecked Sendable {
    public static let shared = SSDFetchPool()

    private let queue = DispatchQueue(
        label: "swiftlm.ssd.fetch", qos: .userInitiated, attributes: .concurrent)
    private let group = DispatchGroup()
    private let errState = ThreadSafeError()

    // DEFAULT ON since EXP-005 (2026-06-11, M5 Max 128 GB, DSV4-Flash-4bit,
    // 48 slots): kernel-verified within-run pairs measured nocache decode
    // 4.53→9.24 t/s (+104%) and warm 6.86→10.57 (+54%) at identical
    // bytes/hit/golden — the engine was queue-depth-starved (device QD≈2 of
    // a ≥9 GB/s @QD16 NVMe), and overlapping miss preads claims that
    // headroom. Sustained probe: 12×256-tok flat at 10.6 t/s, compressor
    // stable. The earlier "UMA contention (pread 1→4-8ms)" rejection did not
    // reproduce under kernel-verified measurement — it was page-cache memcpy
    // contention misattributed to the device. MLX_MOE_ASYNC_FETCH=0 reverts.
    public let enabled: Bool = {
        let v = ProcessInfo.processInfo.environment["MLX_MOE_ASYNC_FETCH"] ?? ""
        if v == "0" || v.lowercased() == "false" { return false }
        return true
    }()

    /// Run `work` asynchronously (or inline when disabled). Errors are
    /// latched and surfaced at the next `waitAll()`.
    public func submit(_ work: @escaping () -> Void) {
        if enabled {
            queue.async(group: group) { [errState] in
                errState.catchError { work() }
            }
        } else {
            errState.catchError { work() }
        }
    }

    /// Block until all in-flight fetches complete, then surface any error.
    public func waitAll() {
        if enabled { group.wait() }
        if errState.error != nil {
            errState.check()  // posts to the SSD streaming error latch
            errState.lock.withLock { errState.error = nil }
        }
    }
}

