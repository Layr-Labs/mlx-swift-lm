// EngineLoopV2+MTPOrdinaryHistory.swift
//
// Ordinary, chainable decode for persistent-history MTP rows that cannot
// speculate this step (decode rows above the drafter's speculative batch).
// The target forward also returns its trusted hidden rows
// (`decodeLogits(returnsHidden:)`); finalize feeds them to assistant history
// and refreshes the carry, so a later round still needs no seed step. No
// policy top-two and no extra readback.

import Foundation
import MLX

/// Trusted target rows of one ordinary decode step, observed at finalize.
struct CBv2MTPOrdinaryDecodeHistory {
    /// [B, 1] int32 decode inputs, in decode-batch order.
    let tokens: MLXArray
    /// [B, 1, H] pre-norm target hidden, in decode-batch order.
    let hidden: MLXArray
    /// History-owning rows, their decode-batch index, and their preemption
    /// count at launch (a later preemption already released their history).
    let rows: [(id: CBv2RequestID, index: Int, preemptions: Int)]
}

extension EngineLoopV2 {

    /// Native completion tracking keeps its established MTP-step schedule.
    var mtpFeedsHistoryFromOrdinaryDecode: Bool {
        nativeShutdownState == nil && mtp?.feedsHistoryFromOrdinaryDecode == true
    }

    var mtpSeedsCarryFromPrefill: Bool {
        nativeShutdownState == nil && mtp?.seedsCarryFromPrefill == true
    }

    /// Decode rows whose trusted hidden must reach persistent history when
    /// they run as ordinary decode. Same gates as MTP-step observation.
    func mtpOrdinaryHistoryRows(
        _ ids: [CBv2RequestID]
    ) -> [(id: CBv2RequestID, index: Int, preemptions: Int)] {
        guard mtpFeedsHistoryFromOrdinaryDecode else { return [] }
        return ids.enumerated().compactMap { index, id in
            guard let rec = scheduler.record(for: id), mtpBasicEligible(rec),
                let state = kvStates[id], Self.mtpStorageEligible(state)
            else { return nil }
            return (id, index, rec.preemptionCount)
        }
    }

    /// Runs after the ordinary finalize loop confirmed this step's tokens and
    /// finished stopped rows. Observations land in step order because steps
    /// finalize in launch order. A row with no successor in flight receives
    /// the carry a seed step would have stored (without a policy margin);
    /// otherwise any older carry is stale and dropped, so a consumed history
    /// transition is never observed twice.
    func finalizeMTPOrdinaryHistory(_ step: CBv2InFlightStep) {
        guard let mtp, let history = step.mtpOrdinaryHistory else { return }
        for (id, index, preemptions) in history.rows {
            // A chained plan may preempt a row of this in-flight step; its
            // history was released and restarts with the recompute.
            guard !step.discard.contains(id), let rec = scheduler.record(for: id),
                rec.preemptionCount == preemptions
            else { continue }
            let hidden = history.hidden[index ..< index + 1, 0..., 0...]
            do {
                guard let state = try mtpTakeHistoryState(for: rec) else { continue }
                if let carry = mtp.pendingHistoryCarry(for: id) {
                    mtp.observeCommittedTarget(
                        id: id,
                        observation: CBv2MTPCommittedTargetObservation(
                            tokens: MLXArray([Int32(carry.token)]).reshaped([1, 1]),
                            hidden: carry.hidden),
                        detachedState: state)
                }
                mtp.observeCommittedTarget(
                    id: id,
                    observation: CBv2MTPCommittedTargetObservation(
                        tokens: history.tokens[index ..< index + 1, 0...], hidden: hidden),
                    detachedState: state)
                mtp.restoreAssistantState(state, for: id)
            } catch {
                finishRequest(id, reason: .error("assistant history: \(error)"))
                continue
            }
            if rec.pendingSamples == 0 {
                mtp.storeCarry(
                    id: id, token: rec.tokens.last!, hidden: hidden,
                    tokensCount: rec.tokens.count, kvOffset: rec.numComputedTokens)
            } else {
                mtp.dropCarry(id)
            }
        }
    }
}
