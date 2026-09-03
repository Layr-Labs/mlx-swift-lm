// RESIDENCY-001 (C6) --- put the decode working set into MTL residency sets.
//
// WHAT WAS ALREADY THERE. The pinned MLX already implements the technique
// `memory/dwarfstar4-borrowable-techniques.md` describes: `ResidencySets`
// (`mlx/backend/metal/resident.cpp`) creates size-capped `MTL::ResidencySet`s,
// holds a standing `requestResidency()` on each, attaches every set to every
// command queue before each commit, and inserts each allocation as
// `MetalAllocator::malloc` hands it out. No Metal code and no MLX core change
// is needed for C6.
//
// WHY IT IS DARK ANYWAY. The whole mechanism is budgeted by ONE number:
//
//     size_t MetalAllocator::set_wired_limit(size_t limit) {
//       std::swap(limit, wired_limit_);
//       residency_sets_.resize(wired_limit_);   // <- the budget
//     }
//
// and `ResidencySets::capacity_` starts at ZERO ("0 by default, i.e. nothing
// is wired unless asked for", resident.h). `ResidencySets::insert` then bails
// on `total_wired_ + bytes > capacity_` for every allocation, so with no wired
// limit set NOTHING is ever added to a set and the sets stay empty. Nothing in
// the serving path asks: `provider-swift` sets `Memory.memoryLimit` and
// `Memory.cacheLimit` (`MLXMemoryGuard`) and never takes a
// `WiredMemoryTicket`, so the whole model is unwired and every command buffer
// pays its own residency decisions.
//
// WHAT THIS DOES. Sets the budget once, on the first engine step, so the
// ~14.5 GB of weights already allocated are retro-added by `resize()` and
// every KV page and per-step buffer allocated afterwards is inserted at
// `malloc`. Bit-exact by construction: this is an allocation POLICY and
// touches no arithmetic, no kernel and no dispatch.
//
// WHAT THE BUDGET IS, AND WHAT IT IS NOT. It is a CEILING on how many bytes
// MLX may keep wired, not an allocation and not a reservation --- the process
// only ever wires what it actually allocates. It is also NOT the box's
// `iogpu.wired_limit_mb` knob (`memory/never-exceed-the-memory-knob.md`):
// `metal::set_wired_limit` refuses any value above the device's
// `recommendedMaxWorkingSetSize` and throws, so this can never raise the
// machine's ceiling, and the default below is clamped to exactly that value.
//
// `DARKBLOOM_METAL_RESIDENCY_SET`:
//   * `0`/`false`/`no`/`off` --- never touch the wired limit (stock).
//   * a positive integer --- that many MEBIBYTES of budget, clamped to the
//     device maximum. For bisecting how much of the working set matters.
//   * unset or anything else --- the device maximum, i.e. everything MLX
//     allocates becomes resident.
//
// Engage mark: `metal-residency-set`.

import Cmlx
import Foundation
import MLX

public enum CBv2MetalResidencySetV1 {

    /// Parsed once; the switch is a process-level decision.
    ///
    /// `Sendable` is spelled out rather than inferred: a PUBLIC type never
    /// gets the implicit conformance, and without it the immutable
    /// `configured` global below is a strict-concurrency error
    /// (`#MutableGlobalVariable`). The conformance is sound on its own terms
    /// --- a frozen-in-practice value type whose only payload is an `Int`,
    /// with no reference storage to share.
    public enum Setting: Equatable, Sendable {
        case off
        /// Explicit budget in bytes.
        case bytes(Int)
        /// The device's `recommendedMaxWorkingSetSize`.
        case deviceMaximum
    }

    /// Device-free parse of the switch, so the policy is testable.
    public static func setting(from raw: String?) -> Setting {
        guard let raw = raw?.trimmingCharacters(in: .whitespaces), !raw.isEmpty else {
            return .deviceMaximum
        }
        if ["0", "false", "no", "off"].contains(raw.lowercased()) { return .off }
        if let mib = Int(raw), mib > 0 {
            // Overflow-safe: a value this large is already past any device.
            guard mib < Int.max / (1024 * 1024) else { return .deviceMaximum }
            return .bytes(mib * 1024 * 1024)
        }
        return .deviceMaximum
    }

    /// The budget actually applied. `nil` means "leave the wired limit alone".
    /// `deviceMaximumBytes` is `GPU.maxRecommendedWorkingSetBytes()`; a device
    /// that does not report one keeps the stock behaviour, because
    /// `metal::set_wired_limit` throws above that value and there is nothing
    /// safe to clamp to.
    public static func budget(
        for setting: Setting, deviceMaximumBytes: Int?
    ) -> Int? {
        guard let deviceMaximumBytes, deviceMaximumBytes > 0 else { return nil }
        switch setting {
        case .off:
            return nil
        case .deviceMaximum:
            return deviceMaximumBytes
        case .bytes(let requested):
            return min(requested, deviceMaximumBytes)
        }
    }

    private static let configured: Setting = setting(
        from: ProcessInfo.processInfo.environment["DARKBLOOM_METAL_RESIDENCY_SET"])

    private static let lock = NSLock()
    nonisolated(unsafe) private static var armed = false

    /// Idempotent. Safe to call on every step: after the first successful (or
    /// refused) attempt this is one atomic-ish flag read under an uncontended
    /// lock.
    public static func armIfNeeded() {
        lock.lock()
        if armed {
            lock.unlock()
            return
        }
        armed = true
        lock.unlock()
        // `budget` answers nil for the off state, so this is the whole switch.
        apply()
    }

    private static func apply() {
        guard
            let budget = budget(
                for: configured, deviceMaximumBytes: GPU.maxRecommendedWorkingSetBytes())
        else { return }
        var previous: size_t = 0
        let status = mlx_set_wired_limit(&previous, size_t(budget))
        guard status == 0 else { return }
        CBv2EngageMark.once(
            "metal-residency-set budget=\(budget >> 20)MiB previous=\(Int(previous) >> 20)MiB")
    }
}
