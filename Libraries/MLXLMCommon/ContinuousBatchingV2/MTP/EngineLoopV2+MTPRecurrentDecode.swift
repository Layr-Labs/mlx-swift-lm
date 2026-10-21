// EngineLoopV2+MTPRecurrentDecode.swift
//
// Decode groups of an MTP step over a recurrent target: the hidden-returning
// rows and, under the opt-in request-scoped media fallback, the target-only
// media rows. Every row binds its own request-owned recurrent transaction.

import MLX

extension EngineLoopV2 {

    static func mtpDecodePhase(_ rows: [CBv2MTPRowWork]) -> CBv2ForwardPhase {
        rows.allSatisfy { $0.start < $0.rec.request.promptTokens.count }
            ? .prefill
            : (rows.allSatisfy { $0.start >= $0.rec.request.promptTokens.count }
                ? .decode : .mixedFrontier)
    }

    /// Hidden-returning `[B, 1]` forward of `rows`. The bound evaluations ride
    /// the step and are committed at its finalize.
    func mtpRecurrentForwardWithHidden(
        rows: [CBv2MTPRowWork], tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        model recurrentModel: any CBv2RecurrentMTPSteppableModel,
        cacheInnerState: inout [MLXArray],
        recurrentEvaluations: inout [CBv2RequestID: CBv2RecurrentStateEvaluation]
    ) throws -> (logits: MLXArray, lastHidden: MLXArray) {
        let evaluations = rows.map { row -> CBv2RecurrentStateEvaluation in
            guard let state = recurrentStates[row.rec.id] else {
                preconditionFailure("CBv2 recurrent MTP seed state missing")
            }
            do { return try state.bind() } catch {
                preconditionFailure("CBv2 recurrent MTP seed bind failed: \(error)")
            }
        }
        let positionIds = CBv2PositionState.decodePositionIds(
            states: rows.map(\.rec.request.positionState),
            cacheOffsets: rows.map { Self.positionOffset(kvStates[$0.rec.id]!) })
        let output = try withQwen4PositionScope(
            ids: rows.map(\.rec.id), positionIds: positionIds
        ) {
            try checkedModelForward(phase: Self.mtpDecodePhase(rows)) {
                recurrentModel.forwardWithHidden(
                    tokens: tokens, caches: caches, recurrentState: evaluations,
                    positionIds: positionIds)
            }
        }
        for (row, evaluation) in zip(rows, evaluations) {
            do {
                cacheInnerState.append(contentsOf: try evaluation.evaluate())
            } catch {
                preconditionFailure("CBv2 recurrent MTP seed evaluation failed: \(error)")
            }
            recurrentEvaluations[row.rec.id] = evaluation
        }
        return output
    }

    /// Target-only media decode rows: the ordinary target forward that
    /// `executeMixed` runs, so these rows compute what a non-speculating
    /// engine computes for them. Returns `[B, 1, vocab]` logits.
    func mtpRecurrentTargetOnlyDecode(
        rows: [CBv2MTPRowWork], tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        cacheInnerState: inout [MLXArray],
        recurrentEvaluations: inout [CBv2RequestID: CBv2RecurrentStateEvaluation]
    ) throws -> MLXArray {
        let positionIds = CBv2PositionState.decodePositionIds(
            states: rows.map(\.rec.request.positionState),
            cacheOffsets: rows.map { Self.positionOffset(kvStates[$0.rec.id]!) })
        let forward = try targetForward(
            tokens: tokens, caches: caches, ids: rows.map(\.rec.id),
            positionIds: positionIds, phase: Self.mtpDecodePhase(rows))
        cacheInnerState.append(contentsOf: forward.innerState)
        recurrentEvaluations.merge(forward.recurrent) { _, _ in
            preconditionFailure("duplicate recurrent evaluation")
        }
        return forward.logits
    }
}
