// EngineLoopV2+MTPRecurrentDecode.swift
//
// Decode groups over a recurrent MTP target: the hidden-returning rows of an
// MTP step or of an ordinary step that feeds persistent history and, under
// the opt-in request-scoped media fallback, the target-only media rows. Every
// row binds its own request-owned recurrent transaction.

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
        let ids = rows.map(\.rec.id)
        let forward = try mtpRecurrentForwardWithHidden(
            ids: ids, tokens: tokens, caches: caches,
            positionIds: CBv2PositionState.decodePositionIds(
                states: rows.map(\.rec.request.positionState),
                cacheOffsets: rows.map { Self.positionOffset(kvStates[$0.rec.id]!) }),
            phase: Self.mtpDecodePhase(rows), model: recurrentModel)
        cacheInnerState.append(contentsOf: forward.innerState)
        for (id, evaluation) in zip(ids, forward.evaluations) {
            recurrentEvaluations[id] = evaluation
        }
        return (forward.logits, forward.lastHidden)
    }

    /// The same forward over explicit row ids and positions, also used by
    /// ordinary decode rows that feed persistent history (`decodeLogits`):
    /// it binds the same transactions and runs the same trunk graph as the
    /// plain recurrent `targetForward`, so the logits match it.
    func mtpRecurrentForwardWithHidden(
        ids: [CBv2RequestID], tokens: MLXArray, caches: [CBv2AttendingLayerCache],
        positionIds: MLXArray?, phase: CBv2ForwardPhase,
        model recurrentModel: any CBv2RecurrentMTPSteppableModel
    ) throws -> (
        logits: MLXArray, lastHidden: MLXArray,
        evaluations: [CBv2RecurrentStateEvaluation], innerState: [MLXArray]
    ) {
        let evaluations = ids.map { id -> CBv2RecurrentStateEvaluation in
            guard let state = recurrentStates[id] else {
                preconditionFailure("CBv2 recurrent MTP state missing for \(id)")
            }
            do { return try state.bind() } catch {
                preconditionFailure("CBv2 recurrent MTP bind failed for \(id): \(error)")
            }
        }
        let output = try withQwen4PositionScope(ids: ids, positionIds: positionIds) {
            try checkedModelForward(phase: phase) {
                recurrentModel.forwardWithHidden(
                    tokens: tokens, caches: caches, recurrentState: evaluations,
                    positionIds: positionIds)
            }
        }
        var innerState: [MLXArray] = []
        for evaluation in evaluations {
            do { innerState.append(contentsOf: try evaluation.evaluate()) } catch {
                preconditionFailure("CBv2 recurrent MTP evaluation failed: \(error)")
            }
        }
        return (output.logits, output.lastHidden, evaluations, innerState)
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
