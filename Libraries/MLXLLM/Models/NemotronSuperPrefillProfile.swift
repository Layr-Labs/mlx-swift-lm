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
/// Usage: `reset()`, run and evaluate the prefill, then print `report()`.
/// A test or tool that must not see other concurrent forwards binds its own
/// ``NemotronSuperProfileRecorder`` with `$recorder.withValue(_:operation:)`;
/// that scope is profiled whatever the environment says.
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

    /// Runs `body` as one stage when `active` and profiling, else returns `body()` as is.
    static func stage(
        _ name: String, active: Bool, inputs: [MLXArray], _ body: () -> MLXArray
    ) -> MLXArray {
        stages(name, active: active, inputs: inputs) { [body()] }[0]
    }

    /// ``stage(_:active:inputs:_:)`` for a stage with several outputs.
    static func stages(
        _ name: String, active: Bool, inputs: [MLXArray], _ body: () -> [MLXArray]
    ) -> [MLXArray] {
        guard active, let current else { return body() }
        let glueStart = DispatchTime.now().uptimeNanoseconds
        eval(inputs)
        let start = DispatchTime.now().uptimeNanoseconds
        let output = body()
        eval(output)
        let end = DispatchTime.now().uptimeNanoseconds
        current.add(name, glue: Double(start - glueStart) / 1e9, seconds: Double(end - start) / 1e9)
        return output
    }

    /// Accumulates the rows routed to each expert by one MoE layer call.
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
}

/// Accumulated stage times and routing counts of profiled Super forwards.
public final class NemotronSuperProfileRecorder: @unchecked Sendable {
    /// The grouped NVFP4 GEMM's row tile; padding to it is the tile waste.
    static let rowTile = 32

    private struct Routing {
        var order: Int
        var counts: [Int]
        var calls = 0
        var rows = 0
        var paddedRows = 0
    }

    private struct Totals {
        var stages: [String: (seconds: Double, calls: Int)] = [:]
        var order: [String] = []
        var routing: [ObjectIdentifier: Routing] = [:]
    }

    private let lock = NSLock()
    private var totals = Totals()

    public init() {}

    func add(_ name: String, glue: Double, seconds: Double) {
        let layerType = name.split(separator: " ").first.map(String.init) ?? name
        lock.withLock {
            add("\(layerType) glue (between stages)", glue)
            add(name, seconds)
        }
    }

    private func add(_ name: String, _ seconds: Double) {
        if totals.stages[name] == nil { totals.order.append(name) }
        let previous = totals.stages[name] ?? (0, 0)
        totals.stages[name] = (previous.seconds + seconds, previous.calls + 1)
    }

    func addRouting(layer: ObjectIdentifier, ids: [Int32], experts: Int) {
        var call = [Int](repeating: 0, count: experts)
        for id in ids where id >= 0 && Int(id) < experts { call[Int(id)] += 1 }
        let tile = Self.rowTile
        lock.withLock {
            var entry =
                totals.routing[layer]
                ?? Routing(order: totals.routing.count, counts: [Int](repeating: 0, count: experts))
            for (expert, rows) in call.enumerated() {
                entry.counts[expert] += rows
                entry.paddedRows += (rows + tile - 1) / tile * tile
            }
            entry.calls += 1
            entry.rows += ids.count
            totals.routing[layer] = entry
        }
    }

    public func reset() {
        lock.withLock { totals = Totals() }
    }

    /// Rows routed to each expert, summed over calls, one array per MoE layer
    /// in model order (the order layers first ran after ``reset()``).
    public func routingCounts() -> [[Int]] {
        lock.withLock {
            totals.routing.values.sorted { $0.order < $1.order }.map(\.counts)
        }
    }

    /// Stage totals in first-seen order with per-layer-type subtotals, then per
    /// MoE layer the rows per expert summed over calls (min, p10, median, p90,
    /// max, empty experts, a bucketed histogram) and the share of 32-row tile
    /// rows that carry real rows (per call).
    public func report() -> String {
        lock.withLock {
            let total = totals.stages.values.reduce(0) { $0 + $1.seconds }
            func percent(_ seconds: Double) -> Double { total > 0 ? 100 * seconds / total : 0 }
            var lines = [String(format: "Super profile: %.3f s in profiled stages", total)]
            var types: [String] = []
            var typeSeconds: [String: Double] = [:]
            for name in totals.order {
                let type = name.split(separator: " ").first.map(String.init) ?? name
                if typeSeconds[type] == nil { types.append(type) }
                typeSeconds[type, default: 0] += totals.stages[name]!.seconds
            }
            for type in types {
                lines.append(
                    String(
                        format: "%@ total: %.3f s (%.1f%%)", type, typeSeconds[type]!,
                        percent(typeSeconds[type]!)))
                for name in totals.order where name.hasPrefix(type + " ") {
                    let value = totals.stages[name]!
                    lines.append(
                        String(
                            format: "  %@: %.3f s over %d calls (%.1f%%)", name, value.seconds,
                            value.calls, percent(value.seconds)))
                }
            }
            let buckets = [0, 1, 16, 32, 64, 128, 256]
            for (index, entry) in totals.routing.values.sorted(by: { $0.order < $1.order })
                .enumerated()
            {
                let sorted = entry.counts.sorted()
                func at(_ fraction: Double) -> Int {
                    sorted.isEmpty
                        ? 0 : sorted[min(sorted.count - 1, Int(fraction * Double(sorted.count)))]
                }
                var histogram = [Int](repeating: 0, count: buckets.count)
                for rows in entry.counts {
                    histogram[buckets.lastIndex { rows >= $0 }!] += 1
                }
                let labels = ["0", "1-15", "16-31", "32-63", "64-127", "128-255", "256+"]
                let bucketText = zip(labels, histogram).map { "\($0)=\($1)" }.joined(separator: " ")
                lines.append(
                    "MoE layer \(index) (\(entry.calls) calls, \(entry.rows) routed rows): rows per expert "
                        + "min \(sorted.first ?? 0) p10 \(at(0.1)) median \(at(0.5)) p90 \(at(0.9)) "
                        + "max \(sorted.last ?? 0); experts \(bucketText); "
                        + String(
                            format: "32-row tile fill %.1f%%",
                            entry.paddedRows > 0
                                ? 100 * Double(entry.rows) / Double(entry.paddedRows) : 0))
            }
            return lines.joined(separator: "\n")
        }
    }
}
