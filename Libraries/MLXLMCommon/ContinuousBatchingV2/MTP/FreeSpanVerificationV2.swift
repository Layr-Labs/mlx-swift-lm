// Copyright © 2026 Eigen Labs.

import MLX

/// Built-in samplers keep the same committed grammar state for ordinary and MTP steps.
protocol CBv2FreeSpanSampler: CBv2StepSampler {
    func confirmedConstraintState(for id: CBv2RequestID, outputTokenCount: Int) -> Int?
    func maskFreeSpanVerification(_ logits: MLXArray, requestIDs: [CBv2RequestID]) -> MLXArray
}

struct CBv2MTPFreeSpan {
    let constraint: any CBv2FreeSpanTokenConstraint
    let state: Int

    func continues(with token: Int) -> Bool {
        constraint.isFreeSpanContinuation(state: state, tokenID: token)
    }
}

extension EngineLoopV2 {
    func mtpFreeSpan(_ rec: CBv2ScheduledRequest) -> CBv2MTPFreeSpan? {
        guard mtp?.config.freeSpanSpeculation == true, nativeShutdownState == nil,
            let constraint = rec.request.tokenConstraint as? any CBv2FreeSpanTokenConstraint,
            let spanSampler = sampler as? any CBv2FreeSpanSampler
        else { return nil }
        // Before the first ordinary sample, only seed planning reads the initial state.
        let state =
            rec.generatedTokenCount == 0
            ? constraint.initialState
            : spanSampler.confirmedConstraintState(
                for: rec.id, outputTokenCount: rec.generatedTokenCount)
        guard let state, constraint.isFreeSpan(state: state) else { return nil }
        return CBv2MTPFreeSpan(constraint: constraint, state: state)
    }
}
