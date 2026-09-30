// Copyright © 2026 Eigen Labs.

import Foundation

extension EngineLoopV2 {
    /// Engine-queue only. Memory feasibility precedes either calibrated or
    /// legacy service conversion; the caller owns the unchanged clock verdict.
    func firstTokenProjectedWork(
        _ projection: CBv2FirstTokenWorkProjection,
        request: CBv2Request,
        reusedPrefix: Bool,
        targetComputedTokens: Int,
        admission policy: CBv2FirstTokenDeadlineAdmission,
        hasInFlightWork: Bool = true
    ) -> CBv2FirstTokenProjectedWork {
        switch projection {
        case .bounded(let work, let capacityOperations):
            guard work.prefillTokens >= 0,
                work.decodeTokens >= 0,
                work.scheduledSteps >= 0,
                work.mixedSteps >= 0,
                work.mixedSteps <= work.scheduledSteps,
                work.mixedSteps == 0
                    || (work.prefillTokens > 0 && work.decodeTokens > 0)
            else {
                return .unbounded(reason: .invalidWorkTotals)
            }
            if let capacity {
                let pool = (backend as? PagedKVBackend)?.pool
                let physicalProjection: (([CBv2RequestID: Int]) -> Int?)?
                if let pool, pool.segmentGrant != nil {
                    physicalProjection = { [layerKinds] tokens in
                        pool.projectedPhysicalBytes(reservedTokens: tokens, layerKinds: layerKinds)
                    }
                } else {
                    physicalProjection = nil
                }
                guard let admission = capacity as? AdmissionV2 else {
                    return .unbounded(reason: .capacityModelUnsupported)
                }
                guard
                    admission.canGuarantee(
                        projectedOperations: capacityOperations,
                        projectedPhysicalBytes: physicalProjection)
                else {
                    return .unbounded(reason: .capacityNotGuaranteed)
                }
            }

            func phaseSeconds(tokens: Int, rate: Double?) -> Double? {
                guard tokens > 0 else { return 0 }
                guard let rate, rate.isFinite, rate > 0 else { return nil }
                let seconds = Double(tokens) / rate
                guard seconds.isFinite, seconds > 0 else { return nil }
                return seconds
            }

            // Serial phase envelope. For a mixed step this charges decode and
            // prefill independently instead of treating radically different
            // work as one token currency. Missing either required lower-bound
            // rate makes the posture unbounded and enforcement fails closed.
            let calibratedSeconds = policy.nativeTargetPrefill == nil ? policy.calibration?.serviceSeconds(
                work: work, promptTokens: request.promptTokens.count,
                reusedPrefix: reusedPrefix,
                activeRequests: scheduler.running.count + scheduler.waiting.count,
                maxOutputTokens: request.maxTokens,
                existingSchedulerContextTokensMax: existingDeadlineContextMaximum(
                    excluding: request.id),
                targetComputedTokens: targetComputedTokens,
                clock: config.clock) : nil
            let fallbackPrefill: Double?
            if let native = policy.nativeTargetPrefill {
                guard let media = request.multimodal, media.nativeMediaToken != nil,
                    media.attention == .causal, media.positionState == nil,
                    media.deepstackEmbeddings == nil, !reusedPrefix,
                    targetComputedTokens == 0,
                    work.prefillTokens >= request.promptTokens.count else {
                    return .unbounded(reason: .prefillRateUnavailable)
                }
                if let bootstrap = native.bootstrap,
                    bootstrap.isValid(request: request, clock: config.clock),
                    !hasInFlightWork, scheduler.running.count + scheduler.waiting.count == 1,
                    work.prefillTokens == request.promptTokens.count,
                    work.decodeTokens == 0, work.mixedSteps == 0 {
                    return .unmeasuredNativeMedia(work: work)
                }
                guard let targetRate = native.observation?.rate(request: request,
                    reusedPrefix: reusedPrefix, targetComputedTokens: targetComputedTokens,
                    clock: config.clock) else {
                    return .unbounded(reason: .prefillRateUnavailable)
                }
                // Only this target gets its observed native-media rate. Work
                // ahead still consumes the original phase rate; a faster target
                // can never make unknown or slow queued text appear free.
                let target = phaseSeconds(tokens: request.promptTokens.count, rate: targetRate)
                let preceding = phaseSeconds(
                    tokens: work.prefillTokens - request.promptTokens.count,
                    rate: policy.conservativePrefillTokensPerSecond)
                fallbackPrefill = target.flatMap { t in preceding.map { t + $0 } }
            } else {
                fallbackPrefill = phaseSeconds(tokens: work.prefillTokens,
                    rate: policy.conservativePrefillTokensPerSecond)
            }
            let fallbackDecode = phaseSeconds(
                tokens: work.decodeTokens,
                rate: policy.conservativeDecodeTokensPerSecond)
            let fallbackSeconds = fallbackPrefill.flatMap { prefill in
                fallbackDecode.map { prefill + $0 }
            }
            guard let seconds = calibratedSeconds ?? fallbackSeconds else {
                // Classify only after both existing conversion paths failed.
                // Calibration can supply a duration without legacy phase rates.
                if fallbackPrefill == nil {
                    return .unbounded(
                        reason:
                            Self.usableDeadlineRate(policy.conservativePrefillTokensPerSecond)
                            ? .serviceDurationInvalid : .prefillRateUnavailable)
                }
                if fallbackDecode == nil {
                    return .unbounded(
                        reason:
                            Self.usableDeadlineRate(policy.conservativeDecodeTokensPerSecond)
                            ? .serviceDurationInvalid : .decodeRateUnavailable)
                }
                return .unbounded(reason: .serviceDurationInvalid)
            }
            // Duration.seconds(_:) traps when its scaled Int128 conversion
            // overflows. Int64.max seconds is a deliberately narrower safe
            // bound; durations beyond it cannot be useful for admission.
            guard seconds.isFinite,
                seconds >= 0,
                seconds <= Double(Int64.max)
            else {
                return .unbounded(reason: .serviceDurationInvalid)
            }
            let serviceDuration = Duration.seconds(seconds)
            guard work.scheduledTokens == 0 || serviceDuration > .zero else {
                return .unbounded(reason: .serviceDurationUnderflow)
            }
            return .bounded(
                work: work,
                serviceDuration: serviceDuration)
        case .unbounded(let reason):
            return .unbounded(reason: reason)
        }
    }

    private static func usableDeadlineRate(_ rate: Double?) -> Bool {
        guard let rate else { return false }
        return rate.isFinite && rate > 0
    }

    private func existingDeadlineContextMaximum(excluding id: CBv2RequestID) -> Int {
        func include(_ result: Int, _ row: CBv2ScheduledRequest) -> Int {
            guard row.id != id else { return result }
            let (context, overflow) = row.request.promptTokens.count.addingReportingOverflow(
                max(0, row.request.maxTokens))
            return max(result, overflow ? Int.max : context)
        }
        return scheduler.waiting.reduce(scheduler.running.reduce(0, include), include)
    }
}
