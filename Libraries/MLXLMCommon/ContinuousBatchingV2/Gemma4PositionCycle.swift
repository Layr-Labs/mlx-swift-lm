// Copyright © 2026 Eigen Labs.
import Foundation

public enum Gemma4UnifiedPositionPolicy {
    public static let enabled = ProcessInfo.processInfo.environment["DARKBLOOM_GEMMA4_UNIFIED_POSITIONS"] == "1"
}

/// Engine-thread-confined write ordering; no device values or row ownership.
final class Gemma4PositionCycle {
    enum Completion: Equatable { case more, advanced, declined }
    let layers: Int
    private var next = 0
    private var width: Int?
    private var pending = false
    var isIdle: Bool { next == 0 && !pending }

    init(layers: Int) { self.layers = layers }

    func reset() { next = 0; width = nil; pending = false }

    func begin(layer: Int, count: Int, inputMatches: Bool) -> Bool {
        guard layers > 0, inputMatches, !pending, layer == next,
            count > 0, count <= Int(Int32.max), width == nil || width == count else { return false }
        width = count
        pending = true
        return true
    }

    func finish(layer: Int, count: Int) -> Completion {
        guard pending, layer == next, width == count else { return .declined }
        pending = false
        next += 1
        if next == layers { reset(); return .advanced }
        return .more
    }
}
