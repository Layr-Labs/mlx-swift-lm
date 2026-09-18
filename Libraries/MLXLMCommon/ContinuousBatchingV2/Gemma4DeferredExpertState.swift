// Copyright © 2026 Eigen Labs.

/// Forward-confined one-shot choice: materialize once, or consume in a fused
/// operation. No synchronization, global memoizer or model/request retention.
struct Gemma4DeferredExpertState<Pending, Value> {
    private enum Storage { case pending(Pending), resolved(Value), consumed }
    private var storage: Storage

    init(pending: Pending) { storage = .pending(pending) }
    init(resolved: Value) { storage = .resolved(resolved) }

    var pending: Pending? {
        if case .pending(let value) = storage { return value }
        return nil
    }

    mutating func resolve(using resolver: (Pending) -> Value) -> Value? {
        switch storage {
        case .pending(let pending):
            let value = resolver(pending)
            storage = .resolved(value)
            return value
        case .resolved(let value): return value
        case .consumed: return nil
        }
    }

    mutating func consumePending() -> Pending? {
        guard case .pending(let value) = storage else { return nil }
        storage = .consumed
        return value
    }
}
