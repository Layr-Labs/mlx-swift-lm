// EngineLoopV2+MTPTargetVerification.swift
//
// Target-authoritative scoring strategies for one known MTP draft chain.

import MLX

extension EngineLoopV2 {

    /// Serial mode is the chip-independent authority path: every column
    /// executes the same `[B, 1]` eager forward used by ordinary decode,
    /// while one surrounding speculative KV transaction defers commit until
    /// the accept walk. Rectangular mode is an explicit optimized strategy.
    func mtpBuildTargetVerification(
        columns: [MLXArray], rows: [CBv2MTPRowWork], driver mtp: CBv2MTPRoundDriver
    ) -> (argmax: MLXArray, hidden: MLXArray, cacheInnerState: [MLXArray]) {
        precondition(!columns.isEmpty, "CBv2 MTP: target verification requires a seed column")
        let caches = eagerCaches(rowStates: rows.map { kvStates[$0.rec.id]! })
        let argmax: MLXArray
        let hidden: MLXArray

        switch mtp.config.verificationMode {
        case .serialTarget:
            var argmaxColumns: [MLXArray] = []
            var hiddenColumns: [MLXArray] = []
            argmaxColumns.reserveCapacity(columns.count)
            hiddenColumns.reserveCapacity(columns.count)
            for column in columns {
                precondition(column.dim(1) == 1, "CBv2 MTP: serial target column must have L=1")
                let output = mtp.model.forwardWithHidden(tokens: column, caches: caches)
                let columnArgmax = argMax(output.logits, axis: -1).asType(.int32)
                // Building several eager decode calls in one lazy graph can
                // let mutable KV buffers observe a later version. Complete
                // each canonical target step before constructing the next.
                eval([columnArgmax, output.lastHidden] + eagerCacheInnerState(caches))
                argmaxColumns.append(columnArgmax)
                hiddenColumns.append(output.lastHidden)
            }
            argmax = concatenated(argmaxColumns, axis: 1)
            hidden = concatenated(hiddenColumns, axis: 1)

        case .rectangular:
            let tokens = concatenated(columns, axis: 1)
            let output = mtp.model.forwardWithHidden(tokens: tokens, caches: caches)
            argmax = argMax(output.logits, axis: -1).asType(.int32)
            hidden = output.lastHidden
        }

        return (argmax, hidden, eagerCacheInnerState(caches))
    }
}
