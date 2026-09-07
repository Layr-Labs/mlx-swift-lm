import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Diagnostic: the MTP capture-verify window against the serial path on the
/// real weights. Opt-in from diag-parity with MLXFAST_DIAG_CAPTURED_WINDOW=1.
///
/// Two fresh single-row sessions prefill the golden's seed in one forward.
/// The serial session then feeds the golden's first two decode tokens one
/// at a time; the captured session feeds the same two tokens as one verify
/// window. The report is the logit L-infinity between the paths at each
/// window position and each path's top-2 margin, which separates kernel
/// rounding (small delta, argmax flips only at a near tie) from a state or
/// position defect (large delta).
enum BenchWorkerCapturedProbe {
    static let environmentSwitch = "MLXFAST_DIAG_CAPTURED_WINDOW"

    static func isRequested(_ environment: [String: String] = ProcessInfo.processInfo.environment) -> Bool {
        let raw = environment[environmentSwitch]?.trimmingCharacters(in: .whitespaces).lowercased()
        return ["1", "true", "yes", "on"].contains(raw ?? "")
    }

    private final class Session {
        let model: Qwen4ExpModel
        let caches: [Qwen4ExpCBv2LayerCache]
        let recurrent: CBv2RecurrentRequestState
        private let backend: CBv2ContiguousKVBackend
        private let rows: [CBv2SequenceKV?]

        init(model: Qwen4ExpModel, kvBytesCapacity: Int, maxLength: Int) throws {
            self.model = model
            self.backend = CBv2ContiguousKVBackend(
                config: CBv2ContiguousBackendConfig(bytesCapacity: kvBytesCapacity))
            self.rows = try backend.makeSequenceState(
                layerKinds: model.cbv2LayerKinds, promptLength: 0, maxLength: maxLength)
            let caches = model.newCacheV2().map { $0 as! Qwen4ExpCBv2LayerCache }
            for (index, cache) in caches.enumerated() {
                guard let row = rows[index] else {
                    preconditionFailure("captured probe: no row for full-attention layer \(index)")
                }
                cache.setRows([row])
            }
            self.caches = caches
            self.recurrent = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
        }

        var kvCaches: [KVCache] { caches.map { $0 as KVCache } }

        /// One committed plain forward; returns logits `[1, L, V]`.
        func forward(_ tokens: [Int]) throws -> MLXArray {
            let ids = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
            let evaluation = try recurrent.bind()
            let out = model.cbv2ForwardWithHidden(
                ids, caches: kvCaches, recurrentState: [evaluation], positionIds: nil)
            try evaluation.evaluate()
            try evaluation.commit()
            eval(out.logits)
            return out.logits
        }

        /// One captured window; commits `keep` positions and rolls the KV
        /// rows back by the rest, the way finalize does. Returns logits
        /// `[1, L, V]`.
        func capturedWindow(_ tokens: [Int], serializeAttention: Bool, keep: Int) throws -> MLXArray {
            let ids = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
            let evaluation = try recurrent.bind()
            for cache in caches { cache.mtpSerializesRectangularAttention = serializeAttention }
            let out = model.cbv2ForwardWithHiddenCaptured(
                ids, caches: kvCaches, recurrentState: [evaluation], positionIds: nil)
            for cache in caches { cache.mtpSerializesRectangularAttention = false }
            try evaluation.evaluate()
            try evaluation.commit(keepPositions: keep)
            let rejected = tokens.count - keep
            if rejected > 0 {
                for row in rows.compactMap({ $0 }) { row.rollback(rejected) }
            }
            for row in rows.compactMap({ $0 }) { row.commitSpeculativeWrite() }
            eval(out.logits)
            return out.logits
        }

        var kvOffset: Int { rows.compactMap { $0 }.first?.absoluteOffset ?? -1 }
        var tapeLength: Int { caches.first?.indexerTapeLength ?? -1 }
    }

    private static func report(_ logits: MLXArray, label: String, emit: (String) -> Void) -> MLXArray {
        let row = logits.asType(.float32)
        let top = argMax(row, axis: -1).item(Int.self)
        let sorted = MLX.sorted(row, axis: -1)
        let n = sorted.dim(-1)
        let margin = (sorted[.ellipsis, n - 1] - sorted[.ellipsis, n - 2]).item(Float.self)
        emit(String(format: "captured-probe: %@ argmax %d margin %.4f", label, top, margin))
        return row
    }

    static func run(
        runner: any Runner, seed: [Int], tokens: [Int], kvBytesCapacity: Int,
        emit: (String) -> Void
    ) throws {
        guard let model = runner.servingModel as? Qwen4ExpModel else {
            emit("captured-probe: serving model is not Qwen4Exp; skipped")
            return
        }
        guard tokens.count >= 2 else {
            emit("captured-probe: need two decode tokens; skipped")
            return
        }
        let window = Array(tokens.prefix(2))
        let maxLength = seed.count + 16
        let capacity = max(kvBytesCapacity, 1 << 30)

        let serial = try Session(model: model, kvBytesCapacity: capacity, maxLength: maxLength)
        _ = try serial.forward(seed)
        let serial0 = report(try serial.forward([window[0]])[0..., -1, 0...], label: "serial pos0", emit: emit)
        let serial1 = report(try serial.forward([window[1]])[0..., -1, 0...], label: "serial pos1", emit: emit)

        emit("captured-probe: serial kv offset \(serial.kvOffset) tape \(serial.tapeLength) after seed+2")

        for serialize in [true, false] {
            let captured = try Session(model: model, kvBytesCapacity: capacity, maxLength: maxLength)
            _ = try captured.forward(seed)
            let logits = try captured.capturedWindow(window, serializeAttention: serialize, keep: 2)
            let tag = serialize ? "captured(serialized attention)" : "captured(batched attention)"
            let c0 = report(logits[0..., 0, 0...], label: "\(tag) pos0", emit: emit)
            let c1 = report(logits[0..., 1, 0...], label: "\(tag) pos1", emit: emit)
            let d0 = MLX.abs(c0 - serial0).max().item(Float.self)
            let d1 = MLX.abs(c1 - serial1).max().item(Float.self)
            emit(String(format: "captured-probe: %@ L_inf vs serial pos0 %.5f pos1 %.5f", tag, d0, d1))
            emit("captured-probe: \(tag) kv offset \(captured.kvOffset) tape \(captured.tapeLength) after seed+window(keep 2)")
        }

        // The finalize shape that failed on the box: window [t0, t1], the
        // draft at position 1 rejected, so commit keep 1, roll the KV back by
        // one, then feed t1 plainly. Must equal the serial path's logits for
        // t1 exactly.
        let rejected = try Session(model: model, kvBytesCapacity: capacity, maxLength: maxLength)
        _ = try rejected.forward(seed)
        _ = try rejected.capturedWindow(window, serializeAttention: true, keep: 1)
        emit("captured-probe: after window(keep 1)+rollback: kv offset \(rejected.kvOffset) tape \(rejected.tapeLength) (serial after t0 would be \(seed.count + 1))")
        let after = report(try rejected.forward([window[1]])[0..., -1, 0...], label: "plain t1 after keep-1 commit", emit: emit)
        let dAfter = MLX.abs(after - serial1).max().item(Float.self)
        emit(String(format: "captured-probe: plain t1 after keep-1 commit L_inf vs serial pos1 %.5f", dAfter))
        emit("captured-probe: after that forward: kv offset \(rejected.kvOffset) tape \(rejected.tapeLength) (serial \(serial.kvOffset))")

        // ---- isolating variants (2026-09-07) ----
        // A: a ONE-position captured window [t0], keep 1, then plain t1. No rollback involved:
        //    if this is wrong the captured stage/commit of a single position is wrong.
        do {
            let a = try Session(model: model, kvBytesCapacity: capacity, maxLength: maxLength)
            _ = try a.forward(seed)
            _ = try a.capturedWindow([window[0]], serializeAttention: true, keep: 1)
            emit("captured-probe: A window[t0] keep 1: kv offset \(a.kvOffset) tape \(a.tapeLength)")
            let afterA = report(try a.forward([window[1]])[0..., -1, 0...], label: "A plain t1 after 1-window keep-1", emit: emit)
            emit(String(format: "captured-probe: A L_inf vs serial pos1 %.5f", MLX.abs(afterA - serial1).max().item(Float.self)))
        }
        // B: window [t0, t1] keep 2 (full acceptance), then plain t2 vs serial pos2.
        if tokens.count >= 3 {
            let serial2 = report(try serial.forward([tokens[2]])[0..., -1, 0...], label: "serial pos2", emit: emit)
            let b = try Session(model: model, kvBytesCapacity: capacity, maxLength: maxLength)
            _ = try b.forward(seed)
            _ = try b.capturedWindow(window, serializeAttention: true, keep: 2)
            emit("captured-probe: B window[t0,t1] keep 2: kv offset \(b.kvOffset) tape \(b.tapeLength)")
            let afterB = report(try b.forward([tokens[2]])[0..., -1, 0...], label: "B plain t2 after keep-2", emit: emit)
            emit(String(format: "captured-probe: B L_inf vs serial pos2 %.5f", MLX.abs(afterB - serial2).max().item(Float.self)))
        }
        // C: window [t0, t1] keep 1 with BATCHED attention, then plain t1.
        do {
            let c = try Session(model: model, kvBytesCapacity: capacity, maxLength: maxLength)
            _ = try c.forward(seed)
            _ = try c.capturedWindow(window, serializeAttention: false, keep: 1)
            let afterC = report(try c.forward([window[1]])[0..., -1, 0...], label: "C plain t1 after keep-1 (batched attn)", emit: emit)
            emit(String(format: "captured-probe: C L_inf vs serial pos1 %.5f", MLX.abs(afterC - serial1).max().item(Float.self)))
        }
        // E: the committed recurrent state after window keep-1 vs the serial state after t0, per layer.
        do {
            let e = try Session(model: model, kvBytesCapacity: capacity, maxLength: maxLength)
            _ = try e.forward(seed)
            _ = try e.capturedWindow(window, serializeAttention: true, keep: 1)
            let s0 = try Session(model: model, kvBytesCapacity: capacity, maxLength: maxLength)
            _ = try s0.forward(seed)
            _ = try s0.forward([window[0]])
            let ev = try e.recurrent.bind()
            let sv = try s0.recurrent.bind()
            var worstConv: (Int, Float) = (-1, 0); var worstSsm: (Int, Float) = (-1, 0); var firstBad = -1; var compared = 0
            for spec in model.cbv2RecurrentStateSpec.layers {
                let li = spec.modelLayerIndex
                guard let x = ev.inputState(modelLayerIndex: li), let y = sv.inputState(modelLayerIndex: li) else { continue }
                compared += 1
                var dc: Float = 0; var ds: Float = 0
                if let xc = x.conv, let yc = y.conv, xc.shape == yc.shape { dc = MLX.abs(xc.asType(.float32) - yc.asType(.float32)).max().item(Float.self) } else if x.conv != nil || y.conv != nil { dc = Float.infinity }
                if let xs = x.ssm, let ys = y.ssm, xs.shape == ys.shape { ds = MLX.abs(xs.asType(.float32) - ys.asType(.float32)).max().item(Float.self) } else if x.ssm != nil || y.ssm != nil { ds = Float.infinity }
                if dc > worstConv.1 { worstConv = (li, dc) }
                if ds > worstSsm.1 { worstSsm = (li, ds) }
                if firstBad < 0, dc > 1e-3 || ds > 1e-3 { firstBad = li; emit(String(format: "captured-probe: E first differing layer %d conv L_inf %.5f ssm L_inf %.5f conv shape %@ ssm shape %@", li, dc, ds, String(describing: x.conv?.shape ?? []), String(describing: x.ssm?.shape ?? []))) }
            }
            emit(String(format: "captured-probe: E compared %d layers; worst conv layer %d L_inf %.5f; worst ssm layer %d L_inf %.5f; first bad %d", compared, worstConv.0, worstConv.1, worstSsm.0, worstSsm.1, firstBad))
            try ev.rollback(); try sv.rollback()
        }
        MLXMemoryReporter().drain()
    }
}
