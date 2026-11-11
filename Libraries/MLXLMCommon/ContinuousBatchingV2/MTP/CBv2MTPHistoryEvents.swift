// CBv2MTPHistoryEvents.swift
//
// Drops the completion events that a finished step's arrays keep while
// request-owned assistant history still views them.

import Cmlx
import MLX

enum CBv2MTPHistoryEvents {
    /// Drops the completion event of every array in `arrays` whose evaluation
    /// has finished. Nothing waits and nothing is scheduled.
    ///
    /// `asyncEval` attaches its completion event (on Metal an
    /// `MTLSharedEvent`, a finite system-wide resource) to every array it
    /// evaluates. MLX drops the event only when the array is waited on, or is
    /// read as an input after the event fired. Persistent assistant history
    /// keeps lazy views of each committed step's token and hidden arrays until
    /// a draft round reads them. A batch-gated row never drafts, so each of
    /// its steps kept one event alive until the row drafted alone or ended.
    ///
    /// Call this after the step's readback, when its events have fired. Values
    /// and evaluation are unchanged: only the handle of a fired event is
    /// dropped, as MLX itself does when it next reads the array. Lazy arrays
    /// and unfinished work are left as they are.
    static func release(_ arrays: [MLXArray]) {
        for array in arrays {
            var available = false
            _ = _mlx_array_is_available(&available, array.ctx)
        }
    }
}

extension EngineLoopV2 {
    /// `CBv2MTPHistoryEvents.release` for a finalized step's history arrays.
    func releaseMTPHistoryEvents(_ arrays: [MLXArray]) {
        guard !mtpHistoryEventReleaseDisabledForTesting else { return }
        CBv2MTPHistoryEvents.release(arrays)
    }
}
