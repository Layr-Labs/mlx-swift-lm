import Foundation

public enum CBv2ForwardPhase: String, Codable, Sendable {
    case prefill, decode
    case mtpVerification = "mtp_verification"
    case mixedFrontier = "mixed_prefill_decode"
}

public enum CBv2ForwardKind: String, Codable, Sendable {
    case target
    case compiledComponent = "compiled_component"
}

public enum CBv2CompiledComponent: String, Codable, Sendable {
    case siluProduct, weightedExpertSum, gelu, swiGLU, geGLU
    case gemmaGelu, gemmaSoftcap, gptossExperts
}

/// Scalars only. Component leading rows may flatten tokens/experts and are
/// never interpreted as live request batch rows.
public struct CBv2ForwardAxes: Hashable, Codable, Sendable {
    public let phase: CBv2ForwardPhase
    public let kind: CBv2ForwardKind
    public let liveBatchRows: Int
    public let sequenceWidth: Int
    public let physicalBatchRows: Int
    public let physicalComponentRows: Int?
    public let component: CBv2CompiledComponent?
}

public struct CBv2ForwardShapeCount: Codable, Sendable {
    public let axes: CBv2ForwardAxes
    /// Actual dispatches entered, not GPU submissions or kernel launches.
    public var submittedCalls: UInt64
    /// Calls whose owning step reached its existing successful readback.
    /// Acceptance/rollback of speculative tokens does not change this count.
    public var completedCalls: UInt64
}

/// Opt-in benchmark receipt for one completed engine step. This is the
/// existing launch-to-readback wall interval, never a GPU-kernel duration.
public struct CBv2CompletedStepTiming: Codable, Sendable {
    public let phase: CBv2ForwardPhase
    public let wallNanos: UInt64
}

public struct CBv2ForwardShapeSnapshot: Codable, Sendable {
    public let schema: Int
    public let scope: UInt64
    public let enabled: Bool
    public let entries: [CBv2ForwardShapeCount]
    public let pendingSteps: Int
    public let abandonedSteps: UInt64
    public let unobservedDispatches: UInt64
    public let droppedCalls: UInt64
    /// Present only with the opt-in recorder. Bounded; a nonzero drop count
    /// makes a full-run latency percentile ineligible for qualification.
    public var completedStepTimings: [CBv2CompletedStepTiming]? = nil
    public var droppedStepTimings: UInt64? = nil
    public var confirmedTokenTimings: [CBv2ConfirmedTokenTiming]? = nil
    public var droppedTokenTimings: UInt64? = nil

    public static let disabled = CBv2ForwardShapeSnapshot(
        schema: 1, scope: 0, enabled: false,
        entries: [], pendingSteps: 0, abandonedSteps: 0, unobservedDispatches: 0, droppedCalls: 0)
}

public enum CBv2ForwardShapeError: Error {
    case engineBusy, scopeExhausted
}

/// Engine-queue confined; no tensors, token IDs, text or exported request IDs.
final class CBv2ForwardShapeRecorder {
    static let maximumBuckets = 256
    static let maximumStepTimings = 8192
    private(set) var scope: UInt64 = 0
    private var entries: [CBv2ForwardAxes: CBv2ForwardShapeCount] = [:]
    private var pending = 0
    private var abandoned: UInt64 = 0
    private var unobserved: UInt64 = 0
    private var dropped: UInt64 = 0
    private var stepTimings: [CBv2CompletedStepTiming] = []
    private var droppedTimings: UInt64 = 0
    private var tokenTimings = CBv2ConfirmedTokenTimings()

    func reset() throws {
        guard pending == 0 else { throw CBv2ForwardShapeError.engineBusy }
        guard scope < UInt64.max else { throw CBv2ForwardShapeError.scopeExhausted }
        scope += 1
        entries.removeAll(keepingCapacity: true)
        abandoned = 0
        unobserved = 0
        dropped = 0
        stepTimings.removeAll(keepingCapacity: true)
        droppedTimings = 0
        tokenTimings = CBv2ConfirmedTokenTimings()
    }

    func beginStep() -> CBv2ForwardShapeStep {
        pending += 1
        return CBv2ForwardShapeStep(owner: self)
    }

    fileprivate func submit(_ axes: CBv2ForwardAxes) -> Bool {
        guard (1 ... 256).contains(axes.liveBatchRows),
            (1 ... 1_048_576).contains(axes.sequenceWidth),
            (axes.liveBatchRows ... 1_048_576).contains(axes.physicalBatchRows),
            axes.physicalComponentRows.map({ (1 ... 16_777_216).contains($0) }) ?? true,
            entries[axes] != nil || entries.count < Self.maximumBuckets
        else {
            dropped = Self.add(dropped, 1)
            return false
        }
        var value = entries[axes] ?? .init(axes: axes, submittedCalls: 0, completedCalls: 0)
        value.submittedCalls = Self.add(value.submittedCalls, 1)
        entries[axes] = value
        return true
    }

    fileprivate func missingDispatch() { unobserved = Self.add(unobserved, 1) }

    fileprivate func confirmTokens(
        row: ObjectIdentifier, firstToken: Bool, count: Int, nanos: UInt64
    ) {
        tokenTimings.record(row: row, firstToken: firstToken, count: count, nanos: nanos)
    }

    fileprivate func retire(
        _ counts: [CBv2ForwardAxes: UInt64], completed: Bool, wallNanos: UInt64? = nil,
        targetPhasesComplete: Bool = true
    ) {
        precondition(pending > 0)
        pending -= 1
        if completed {
            if let wallNanos, wallNanos > 0 {
                let phases = Set(counts.keys.filter { $0.kind == .target }.map(\.phase))
                let prefill = phases.contains(.prefill) || phases.contains(.mixedFrontier)
                let decode =
                    phases.contains(.decode) || phases.contains(.mtpVerification)
                    || phases.contains(.mixedFrontier)
                if targetPhasesComplete && !phases.isEmpty {
                    let phase: CBv2ForwardPhase =
                        prefill && decode ? .mixedFrontier : prefill ? .prefill : .decode
                    if stepTimings.count < Self.maximumStepTimings {
                        stepTimings.append(.init(phase: phase, wallNanos: wallNanos))
                    } else {
                        droppedTimings = Self.add(droppedTimings, 1)
                    }
                } else {
                    // The wall sample exists, but dropped/unobserved target
                    // axes cannot classify it reliably, even if another call
                    // in this same step retained a known phase.
                    droppedTimings = Self.add(droppedTimings, 1)
                }
            }
            for (axes, count) in counts {
                guard var value = entries[axes] else { continue }
                value.completedCalls = Self.add(value.completedCalls, count)
                entries[axes] = value
            }
        } else if !counts.isEmpty {
            abandoned = Self.add(abandoned, 1)
        }
    }

    func snapshot() -> CBv2ForwardShapeSnapshot {
        let ordered = entries.values.sorted {
            let a = $0.axes
            let b = $1.axes
            return [
                a.phase.rawValue, a.kind.rawValue, String(a.liveBatchRows), String(a.sequenceWidth),
                String(a.physicalBatchRows), String(a.physicalComponentRows ?? 0),
                a.component?.rawValue ?? "",
            ]
            .lexicographicallyPrecedes([
                b.phase.rawValue, b.kind.rawValue, String(b.liveBatchRows),
                String(b.sequenceWidth), String(b.physicalBatchRows),
                String(b.physicalComponentRows ?? 0), b.component?.rawValue ?? "",
            ])
        }
        var result = CBv2ForwardShapeSnapshot(
            schema: 1, scope: scope, enabled: true, entries: ordered,
            pendingSteps: pending, abandonedSteps: abandoned,
            unobservedDispatches: unobserved, droppedCalls: dropped)
        result.completedStepTimings = stepTimings
        result.droppedStepTimings = droppedTimings
        result.confirmedTokenTimings = tokenTimings.receipts
        result.droppedTokenTimings = tokenTimings.dropped
        return result
    }

    static func add(_ a: UInt64, _ b: UInt64) -> UInt64 {
        let (sum, overflow) = a.addingReportingOverflow(b)
        return overflow ? .max : sum
    }
}

/// One engine step's scalar dispatch receipts. Abandonment does not assert
/// that no GPU work ran: model implementations may evaluate internally.
final class CBv2ForwardShapeStep {
    private let owner: CBv2ForwardShapeRecorder
    private var counts: [CBv2ForwardAxes: UInt64] = [:]
    private var attached = false
    private var retired = false
    private var targetPhasesComplete = true

    init(owner: CBv2ForwardShapeRecorder) { self.owner = owner }
    deinit { abandon() }
    func attach() { attached = true }
    func finishBuilding() { if !attached { abandon() } }
    func complete(wallNanos: UInt64? = nil) {
        guard !retired else { return }
        retired = true
        owner.retire(
            counts, completed: true, wallNanos: wallNanos,
            targetPhasesComplete: targetPhasesComplete)
    }
    private func abandon() {
        guard !retired else { return }
        retired = true
        owner.retire(counts, completed: false)
    }
    func submit(_ axes: CBv2ForwardAxes) {
        guard !retired else { return }
        guard owner.submit(axes) else {
            if axes.kind == .target { targetPhasesComplete = false }
            return
        }
        counts[axes] = CBv2ForwardShapeRecorder.add(counts[axes, default: 0], 1)
    }
    func missingDispatch() {
        guard !retired else { return }
        targetPhasesComplete = false
        owner.missingDispatch()
    }
    func confirmTokens(row: ObjectIdentifier, firstToken: Bool, count: Int, nanos: UInt64) {
        guard !retired else { return }
        owner.confirmTokens(row: row, firstToken: firstToken, count: count, nanos: nanos)
    }
}
