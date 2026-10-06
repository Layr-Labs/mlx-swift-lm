// DeferredHostFillV2.swift
//
// "Graph before the lookup" (mlx-serve 26.9.1 Flash-Next, Fusion one-sync
// window): a model whose step needs a HOST-side table read keyed by the
// step's own input token (Qwen4 n-gram PLE: 16 mmap'd rows per token) can
// leave a pre-allocated, host-writable input buffer in the lazy graph and
// register the fill here instead of forcing the token onto the host in the
// middle of graph construction. The engine runs every registered fill after
// the whole step graph is built and immediately before it submits the graph
// (`asyncEval`). The chained decode therefore builds step N+1 while step N
// is still on the GPU; only the fill itself — one token readback plus a few
// row copies — sits between step N finishing and step N+1 being submitted.
//
// Scope discipline: a scope is opened ONLY by steps whose input tokens are
// still lazy (the chained decode); every other forward (prefill, mixed step,
// MTP rounds, teacher forcing, TokenIterator) sees no scope and models take
// their ordinary eager path. Dropping a scope that still holds fills is a
// programming error and traps — a graph submitted against an unfilled
// placeholder would otherwise decode from stale rows silently.
//
// Thread confinement: the scope is published through the current thread's
// dictionary; `open` / `register` / `close` must happen on the thread that
// builds the graph (the engine queue), which is the only thread that ever
// runs a step forward.

import Foundation

public final class CBv2DeferredHostFill {
    private static let threadKey = "darkbloom.cbv2.deferredHostFill"

    /// The scope open on this thread, if any.
    public static var current: CBv2DeferredHostFill? {
        Thread.current.threadDictionary[threadKey] as? CBv2DeferredHostFill
    }

    private enum State { case open, closed, ran }
    private var state: State = .open
    private var fills: [() -> Void] = []
    private var registered = 0

    private init() {}

    deinit {
        precondition(
            fills.isEmpty, "CBv2DeferredHostFill dropped with \(fills.count) unrun fill(s)")
    }

    /// Open a scope on the current thread. Exactly one may be open per thread.
    public static func open() -> CBv2DeferredHostFill {
        precondition(current == nil, "CBv2DeferredHostFill scope already open on this thread")
        let scope = CBv2DeferredHostFill()
        Thread.current.threadDictionary[threadKey] = scope
        return scope
    }

    /// Number of fills registered over the scope's lifetime.
    public var registeredCount: Int { registered }

    /// Model side: defer `fill` until the step graph is complete. Fills run
    /// in registration order on the engine thread.
    public func register(_ fill: @escaping () -> Void) {
        precondition(state == .open, "CBv2DeferredHostFill.register after close")
        precondition(Self.current === self, "CBv2DeferredHostFill.register outside its building thread")
        fills.append(fill)
        registered += 1
    }

    /// Model-side eager boundaries (first-use kernel checks, explicit layer
    /// drains, diagnostics) may evaluate before graph construction ends.
    /// Resolve inputs registered so far BEFORE that evaluation. Keep the
    /// scope open so later layers can still register their own inputs.
    ///
    /// This is not a submission or a global synchronization. The callbacks
    /// perform only their normal input readback/row copy. While they run the
    /// scope is detached: callbacks must not register new deferred work.
    public static func resolveBeforeEvaluation() {
        guard let scope = current, !scope.fills.isEmpty else { return }
        precondition(scope.state == .open)
        let pending = scope.fills
        scope.fills.removeAll(keepingCapacity: true)
        Thread.current.threadDictionary.removeObject(forKey: threadKey)
        defer {
            precondition(current == nil, "Deferred input fill opened a nested model scope")
            Thread.current.threadDictionary[threadKey] = scope
        }
        for fill in pending { fill() }
    }

    /// Detach from the thread once graph construction is finished. Further
    /// `register` calls trap; `run` is still pending.
    public func close() {
        guard state == .open else { return }
        state = .closed
        if Thread.current.threadDictionary[Self.threadKey] as? CBv2DeferredHostFill === self {
            Thread.current.threadDictionary.removeObject(forKey: Self.threadKey)
        }
    }

    /// Engine side: run every registered fill. Call after `close()` and
    /// immediately before submitting the step graph. Idempotent.
    public func run() {
        precondition(state != .open, "CBv2DeferredHostFill.run before close")
        guard state == .closed else { return }
        state = .ran
        let pending = fills
        fills.removeAll(keepingCapacity: false)
        for fill in pending { fill() }
    }
}
