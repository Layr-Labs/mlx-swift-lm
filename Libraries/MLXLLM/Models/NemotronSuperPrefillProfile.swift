import Foundation
import MLX

/// Opt-in wall-clock breakdown of Super forwards (prefill and decode), enabled
/// by `NEMOTRON_SUPER_PREFILL_PROFILE=1` in the process environment.
///
/// Each profiled stage first evaluates its inputs (timed separately as that
/// layer type's "glue": the splits, reshapes and copies between stages), then
/// times its own graph to completion. Profiling therefore synchronizes the
/// host at every stage and makes the forward slower than the unprofiled one;
/// the shares, not the sum, are what it measures. It never changes results.
/// Only Super (ModelOpt) layers are profiled; embedding, block norms and
/// residual adds, the final norm and the LM head are not, so the caller's own
/// wall time minus ``report()``'s total is everything outside the stages.
///
/// Times and routing are kept per layer and per call: a layer's n-th call is
/// the n-th prefill chunk (then decode steps) since ``reset()``.
///
/// Usage: `reset()`, run and evaluate the prefill, then print `report()` and
/// optionally write `json()`. A test or tool that must not see other
/// concurrent forwards binds its own ``NemotronSuperProfileRecorder`` with
/// `$recorder.withValue(_:operation:)`; that scope is profiled whatever the
/// environment says.
public enum NemotronSuperPrefillProfile {
    public static let isEnabled =
        ProcessInfo.processInfo.environment["NEMOTRON_SUPER_PREFILL_PROFILE"] == "1"

    /// Process-wide totals used when ``isEnabled``.
    public static let shared = NemotronSuperProfileRecorder()

    /// A scoped recorder; it takes precedence over ``shared``.
    @TaskLocal public static var recorder: NemotronSuperProfileRecorder?

    private static var current: NemotronSuperProfileRecorder? {
        recorder ?? (isEnabled ? shared : nil)
    }

    /// Runs `body` as one stage of `layer` when `active` and profiling, else
    /// returns `body()` as is.
    static func stage(
        _ name: String, layer: AnyObject, active: Bool, inputs: [MLXArray],
        _ body: () -> MLXArray
    ) -> MLXArray {
        stages(name, layer: layer, active: active, inputs: inputs) { [body()] }[0]
    }

    /// ``stage(_:layer:active:inputs:_:)`` for a stage with several outputs.
    static func stages(
        _ name: String, layer: AnyObject, active: Bool, inputs: [MLXArray],
        _ body: () -> [MLXArray]
    ) -> [MLXArray] {
        guard active, let current else { return body() }
        let glueStart = DispatchTime.now().uptimeNanoseconds
        eval(inputs)
        let base = current.recordsMemory ? Memory.activeMemory : 0
        if current.recordsMemory { Memory.peakMemory = 0 }
        let start = DispatchTime.now().uptimeNanoseconds
        let output = body()
        eval(output)
        let end = DispatchTime.now().uptimeNanoseconds
        current.add(
            name, layer: ObjectIdentifier(layer), glue: Double(start - glueStart) / 1e9,
            seconds: Double(end - start) / 1e9)
        if current.recordsMemory {
            current.addMemory(
                name, layer: ObjectIdentifier(layer),
                .init(
                    transientBytes: max(0, Memory.peakMemory - base),
                    retainedBytes: Memory.activeMemory - base))
        }
        return output
    }

    /// Records the rows routed to each expert by one MoE layer call.
    static func recordRouting(layer: AnyObject, indices: MLXArray, experts: Int) {
        guard let current else { return }
        let ids = indices.flattened().asType(.int32).asArray(Int32.self)
        current.addRouting(layer: ObjectIdentifier(layer), ids: ids, experts: experts)
    }

    /// Clears ``shared``.
    public static func reset() { shared.reset() }

    /// ``NemotronSuperProfileRecorder/report()`` of ``shared``.
    public static func report() -> String { shared.report() }

    /// ``NemotronSuperProfileRecorder/routingCounts()`` of ``shared``.
    public static func routingCounts() -> [[Int]] { shared.routingCounts() }

    /// ``NemotronSuperProfileRecorder/routingCountsPerCall()`` of ``shared``.
    public static func routingCountsPerCall() -> [[[Int]]] { shared.routingCountsPerCall() }

    /// ``NemotronSuperProfileRecorder/secondsPerCall()`` of ``shared``.
    public static func secondsPerCall() -> [String: [[Double]]] { shared.secondsPerCall() }

    /// ``NemotronSuperProfileRecorder/json()`` of ``shared``.
    public static func json() throws -> Data { try shared.json() }
}

/// Stage times and routing counts of profiled Super forwards, per layer and call.
public final class NemotronSuperProfileRecorder: @unchecked Sendable {
    /// The grouped NVFP4 GEMM's row tile; padding to it is the tile waste.
    static let rowTile = 32

    /// MLX memory of one profiled stage call, measured after its inputs were
    /// evaluated: `transientBytes` is the active-memory peak above that base
    /// while the stage ran, `retainedBytes` what stayed active afterwards
    /// (negative when the stage released inputs it owned).
    public struct StageMemory: Sendable, Equatable {
        public let transientBytes: Int
        public let retainedBytes: Int
    }

    /// With memory recording each stage resets MLX's process-wide peak, so a
    /// caller's own peak measurement spanning a profiled forward is void.
    public let recordsMemory: Bool

    /// One layer's values in call order; `order` is the layer's rank among the
    /// layers that recorded the same stage (model order).
    private struct Series<Value> {
        var order: Int
        var calls: [Value] = []
    }

    private struct Totals {
        var order: [String] = []
        var seconds: [String: [ObjectIdentifier: Series<Double>]] = [:]
        var routing: [ObjectIdentifier: Series<[Int]>] = [:]
        var memory: [String: [ObjectIdentifier: Series<StageMemory>]] = [:]
    }

    private let lock = NSLock()
    private var totals = Totals()

    public init(recordsMemory: Bool = false) {
        self.recordsMemory = recordsMemory
    }

    func addMemory(_ name: String, layer: ObjectIdentifier, _ value: StageMemory) {
        lock.withLock {
            var layers = totals.memory[name] ?? [:]
            var series = layers[layer] ?? Series(order: layers.count)
            series.calls.append(value)
            layers[layer] = series
            totals.memory[name] = layers
        }
    }

    /// Stage memory per profiled module (model order), per call; empty unless
    /// ``recordsMemory``.
    public func memoryPerCall() -> [String: [[StageMemory]]] {
        lock.withLock { totals.memory.mapValues { ordered($0) } }
    }

    /// Appends the stage's call; its glue goes to the layer type's glue
    /// series at the same call index (summed over that layer's stages).
    func add(_ name: String, layer: ObjectIdentifier, glue: Double, seconds: Double) {
        let layerType = name.split(separator: " ").first.map(String.init) ?? name
        lock.withLock {
            let call = add(name, layer, seconds, at: nil)
            add("\(layerType) glue (between stages)", layer, glue, at: call)
        }
    }

    /// Adds `seconds` at call index `call` (padding with zeros), or appends a
    /// new call when nil; returns the index.
    @discardableResult
    private func add(_ name: String, _ layer: ObjectIdentifier, _ seconds: Double, at call: Int?)
        -> Int
    {
        if totals.seconds[name] == nil { totals.order.append(name) }
        var layers = totals.seconds[name] ?? [:]
        var series = layers[layer] ?? Series(order: layers.count)
        let index = call ?? series.calls.count
        while series.calls.count <= index { series.calls.append(0) }
        series.calls[index] += seconds
        layers[layer] = series
        totals.seconds[name] = layers
        return index
    }

    func addRouting(layer: ObjectIdentifier, ids: [Int32], experts: Int) {
        var call = [Int](repeating: 0, count: experts)
        for id in ids where id >= 0 && Int(id) < experts { call[Int(id)] += 1 }
        lock.withLock {
            var series = totals.routing[layer] ?? Series(order: totals.routing.count)
            series.calls.append(call)
            totals.routing[layer] = series
        }
    }

    public func reset() {
        lock.withLock { totals = Totals() }
    }

    private func ordered<Value>(_ layers: [ObjectIdentifier: Series<Value>]) -> [[Value]] {
        layers.values.sorted { $0.order < $1.order }.map(\.calls)
    }

    /// Rows routed to each expert: per MoE layer (model order), per call.
    public func routingCountsPerCall() -> [[[Int]]] {
        lock.withLock { ordered(totals.routing) }
    }

    /// Rows routed to each expert summed over calls, per MoE layer.
    public func routingCounts() -> [[Int]] {
        routingCountsPerCall().map { calls in
            calls.dropFirst().reduce(calls.first ?? []) { zip($0, $1).map(+) }
        }
    }

    /// Seconds per stage: per profiled module (model order; a MoE layer's
    /// expert stages are its switch MLP's), per call.
    public func secondsPerCall() -> [String: [[Double]]] {
        lock.withLock { totals.seconds.mapValues { ordered($0) } }
    }

    /// Per stage, the seconds of call index n summed over that stage's layers.
    private static func perCallIndex(_ layers: [[Double]]) -> [Double] {
        var sums: [Double] = []
        for calls in layers {
            for (index, seconds) in calls.enumerated() {
                if index == sums.count { sums.append(0) }
                sums[index] += seconds
            }
        }
        return sums
    }

    /// `{"stage_order", "seconds_per_call": {stage: [layer][call]},
    /// "seconds_per_call_index": {stage: [call]} (summed over layers),
    /// "routing_per_call": [layer][call][expert]}`, with ``recordsMemory`` also
    /// `"memory_per_call": {stage: [layer][call] [transient, retained]}`.
    public func json() throws -> Data {
        let (order, seconds, routing) = lock.withLock {
            (totals.order, totals.seconds.mapValues { ordered($0) }, ordered(totals.routing))
        }
        var object: [String: Any] = [
            "stage_order": order,
            "seconds_per_call": seconds,
            "seconds_per_call_index": seconds.mapValues(Self.perCallIndex),
            "routing_per_call": routing,
        ]
        if recordsMemory {
            object["memory_per_call"] = memoryPerCall().mapValues { layers in
                layers.map { calls in calls.map { [$0.transientBytes, $0.retainedBytes] } }
            }
        }
        return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    /// Stage totals in first-seen order with per-layer-type subtotals; each
    /// stage's milliseconds per call index (summed over its layers; first 16
    /// calls); then per MoE layer the rows per expert summed over calls (min,
    /// p10, median, p90, max, a bucketed histogram), the per-call maximum and
    /// the share of 32-row tile rows that carry real rows.
    public func report() -> String {
        let (order, seconds, routing) = lock.withLock {
            (totals.order, totals.seconds.mapValues { ordered($0) }, ordered(totals.routing))
        }
        let stageTotals = seconds.mapValues { $0.joined().reduce(0, +) }
        let total = stageTotals.values.reduce(0, +)
        func percent(_ seconds: Double) -> Double { total > 0 ? 100 * seconds / total : 0 }
        func milliseconds(_ values: [Double]) -> String {
            values.prefix(16).map { String(format: "%.1f", 1000 * $0) }.joined(separator: " ")
        }
        var lines = [String(format: "Super profile: %.3f s in profiled stages", total)]
        var types: [String] = []
        var typeCalls: [String: [Double]] = [:]
        for name in order {
            let type = name.split(separator: " ").first.map(String.init) ?? name
            if typeCalls[type] == nil { types.append(type) }
            let perIndex = Self.perCallIndex(seconds[name]!)
            var sums = typeCalls[type] ?? []
            for (index, value) in perIndex.enumerated() {
                if index == sums.count { sums.append(0) }
                sums[index] += value
            }
            typeCalls[type] = sums
        }
        for type in types {
            let typeTotal = typeCalls[type]!.reduce(0, +)
            lines.append(
                String(format: "%@ total: %.3f s (%.1f%%)", type, typeTotal, percent(typeTotal))
                    + "; ms per call index: " + milliseconds(typeCalls[type]!))
            for name in order where name.hasPrefix(type + " ") {
                let layers = seconds[name]!
                lines.append(
                    String(
                        format: "  %@: %.3f s over %d calls (%.1f%%)", name, stageTotals[name]!,
                        layers.map(\.count).reduce(0, +), percent(stageTotals[name]!))
                        + "; ms per call index: " + milliseconds(Self.perCallIndex(layers)))
            }
        }
        let buckets = [0, 1, 16, 32, 64, 128, 256]
        let labels = ["0", "1-15", "16-31", "32-63", "64-127", "128-255", "256+"]
        let tile = Self.rowTile
        for (index, calls) in routing.enumerated() {
            let counts = calls.dropFirst().reduce(calls.first ?? []) { zip($0, $1).map(+) }
            let sorted = counts.sorted()
            func at(_ fraction: Double) -> Int {
                sorted.isEmpty
                    ? 0 : sorted[min(sorted.count - 1, Int(fraction * Double(sorted.count)))]
            }
            var histogram = [Int](repeating: 0, count: buckets.count)
            for rows in counts { histogram[buckets.lastIndex { rows >= $0 }!] += 1 }
            let rows = counts.reduce(0, +)
            let padded = calls.joined().reduce(0) { $0 + ($1 + tile - 1) / tile * tile }
            let callMax = calls.map { $0.max() ?? 0 }.map(String.init).joined(separator: " ")
            let bucketText = zip(labels, histogram).map { "\($0)=\($1)" }.joined(separator: " ")
            lines.append(
                "MoE layer \(index) (\(calls.count) calls, \(rows) routed rows): rows per expert "
                    + "min \(sorted.first ?? 0) p10 \(at(0.1)) median \(at(0.5)) p90 \(at(0.9)) "
                    + "max \(sorted.last ?? 0); experts \(bucketText); max per call \(callMax); "
                    + String(
                        format: "32-row tile fill %.1f%%",
                        padded > 0 ? 100 * Double(rows) / Double(padded) : 0))
        }
        let memory = memoryPerCall()
        for name in order {
            guard let layers = memory[name], !layers.isEmpty else { continue }
            let calls = layers.joined()
            let transient = calls.map(\.transientBytes).max() ?? 0
            let retained = calls.map(\.retainedBytes).max() ?? 0
            lines.append(
                String(
                    format: "memory %@: max transient %.1f MB, max retained %.1f MB over %d calls",
                    name, Double(transient) / 1e6, Double(retained) / 1e6, calls.count))
        }
        return lines.joined(separator: "\n")
    }
}
