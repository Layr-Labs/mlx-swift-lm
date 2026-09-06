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
        MLXMemoryReporter().drain()
    }
}
