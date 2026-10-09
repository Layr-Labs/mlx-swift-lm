// Copyright © 2026 Eigen Labs.

/// One-forward, one-consumer normalization handoff. A mismatched source clears
/// the handoff rather than retaining stale work for a later layer or request.
/// The trunk owns this value on its stack; no engine/global state is involved.
public struct Gemma4PrefillNormalizationCarry<Value: AnyObject> {
    private var pending: (source: Value, normalized: Value)?

    public init() {}

    public mutating func publish(source: Value, normalized: Value?) {
        pending = normalized.map { (source: source, normalized: $0) }
    }

    public mutating func take(for source: Value) -> Value? {
        let value = pending
        pending = nil
        guard let value, value.source === source else { return nil }
        return value.normalized
    }
}
