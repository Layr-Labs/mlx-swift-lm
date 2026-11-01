// Copyright 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX

/// Small kernel arguments (epsilons, scales, token counts, positions, a
/// counter slot) as shared read-only arrays, built once per distinct value
/// instead of on every call. Each `MLXArray(values)` allocates a device
/// buffer, copies the values and wraps them; a decode step built about 210.
///
/// A cached array is a data leaf with the same shape, dtype and bytes as the
/// one it replaces, and kernels only read it (like a weight), so every result
/// is unchanged. The cache holds at most `capacity` arrays and empties when
/// full, so values that change every step (a DSA layer's query position) are
/// built once per step instead of once per layer without accumulating.
final class GLM5NextKernelConstants: @unchecked Sendable {
    static let shared = GLM5NextKernelConstants()
    static let capacity = 256

    private struct Key: Hashable {
        let dtype: DType
        /// 0-D (a kernel `constant T&`) rather than `[count]`.
        let scalar: Bool
        let bits: [UInt32]
    }

    private let lock = NSLock()
    private var arrays: [Key: MLXArray] = [:]

    /// FP32 `[values.count]`, as `MLXArray(values)`.
    func floats(_ values: [Float]) -> MLXArray {
        array(Key(dtype: .float32, scalar: false, bits: values.map(\.bitPattern))) {
            MLXArray(values)
        }
    }

    /// FP32 0-D, as `MLXArray(value)`.
    func float(_ value: Float) -> MLXArray {
        array(Key(dtype: .float32, scalar: true, bits: [value.bitPattern])) { MLXArray(value) }
    }

    /// Int32 `[values.count]`, as `MLXArray(values)`.
    func int32s(_ values: [Int32]) -> MLXArray {
        array(Key(dtype: .int32, scalar: false, bits: values.map { UInt32(bitPattern: $0) })) {
            MLXArray(values)
        }
    }

    /// Int32 0-D, as `MLXArray(value)`.
    func int32(_ value: Int32) -> MLXArray {
        array(Key(dtype: .int32, scalar: true, bits: [UInt32(bitPattern: value)])) {
            MLXArray(value)
        }
    }

    /// Boolean all-true `shape` (an all-valid decode or verification step's
    /// validity), as `MLXArray.ones(shape, dtype: .bool)` without its fill
    /// launch; nil above `maximumAllTrueElements` elements. A DSA cache's
    /// first append may keep it as its validity storage; like every stored
    /// array it is never written (`GLM5NextDSACache.adopt`).
    func allTrue(_ shape: [Int]) -> MLXArray? {
        let count = shape.reduce(1, *)
        guard count > 0, count <= Self.maximumAllTrueElements, shape.allSatisfy({ $0 > 0 })
        else { return nil }
        return array(Key(dtype: .bool, scalar: shape.isEmpty, bits: shape.map { UInt32($0) })) {
            MLXArray([Bool](repeating: true, count: count), shape)
        }
    }

    static let maximumAllTrueElements = 64

    /// Arrays currently held (a test seam).
    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return arrays.count
    }

    private func array(_ key: Key, _ make: () -> MLXArray) -> MLXArray {
        lock.lock()
        defer { lock.unlock() }
        if let array = arrays[key] { return array }
        if arrays.count >= Self.capacity { arrays.removeAll(keepingCapacity: true) }
        let array = make()
        arrays[key] = array
        return array
    }
}
