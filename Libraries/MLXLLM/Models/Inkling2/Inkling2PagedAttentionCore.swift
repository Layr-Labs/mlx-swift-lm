// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import MLXLMCommon

/// The CBv2 attention core on `Inkling2PagedAttention`: the step's tokens and
/// the slot's pages go to the paged kernel, which also writes decode and
/// verification tokens into their pages. Every query runs the kernel's one
/// arithmetic, so decode, verification and prefill rows agree bit for bit and
/// no row depends on its batch.
///
/// One core serves every engine on a loaded model, and engines build their
/// steps on their own threads: the core's mutable state (plans, bucketing) is
/// behind `lock`. The kernel, launch and segment-offset caches it calls into
/// have their own locks.
public final class Inkling2KernelAttentionCore: Inkling2PagedAttentionCore, @unchecked Sendable {
    private let lock = NSLock()
    /// Plans of the current step, keyed by engine (the layer caches' pool) and
    /// row geometry: one engine's 48 slot calls of a step share two (sliding
    /// and full). A plan is immutable once stored, but its fused workspace is
    /// device scratch its calls reuse one after another; that order holds
    /// within one engine (each call depends on the one before) and not across
    /// engines, so engines never share a plan. Guarded by `lock`.
    private var plans: [PlanKey: Inkling2PagedAttention.Plan] = [:]
    /// Live plans kept: two per engine per step, so a few engines fit.
    static let planCapacity = 16

    private struct PlanKey: Hashable {
        let engine: ObjectIdentifier
        let rows: [CBv2ModelAttentionPages.Row]
        let queries: Int
        let window: Int?
        let scale: Float
    }

    /// Guarded by `lock`.
    private var bucketingSetting: CBv2ModelAttentionBucketing? =
        Inkling2KernelAttentionCore.defaultBucketing
    /// Segment bucketing of decode and verification calls; nil reads a
    /// gathered copy instead (tests and benchmarks compare the two).
    var bucketing: CBv2ModelAttentionBucketing? {
        get { lock.withLock { bucketingSetting } }
        set { lock.withLock { bucketingSetting = newValue } }
    }

    public init() {}

    public func attend(
        slot: Int, cache: any CBv2AttendingLayerCache, rowOffsets: [Int],
        _ inputs: Inkling2AttentionInputs
    ) -> MLXArray {
        guard let paged = cache as? PagedLayerCache else {
            preconditionFailure("Inkling2 slot \(slot): the paged kernel needs a PagedLayerCache")
        }
        let (b, l) = (inputs.queries.dim(0), inputs.queries.dim(2))
        let width = inputs.queries.dim(1) * inputs.queries.dim(3)
        // The call site builds head-major views of token-major tensors;
        // transposing back gives the original buffers, with no copy.
        let queries = inputs.queries.transposed(0, 2, 1, 3)
        let keys = inputs.keys.transposed(0, 2, 1, 3)
        let values = inputs.values.transposed(0, 2, 1, 3)
        let sinks = inputs.sinks.transposed(0, 2, 1)
        // A prompt chunk writes through the cache, so its kernel is known here.
        let prefillKernel =
            Self.kernel(
                queries: l, prefill: inputs.isPrefill, writes: false, dtype: keys.dtype,
                headDim: keys.dim(3), relativeWidth: inputs.relativeProjection.dim(0),
                window: inputs.window) == .prefill
        let bucketing = bucketing
        let engine = paged.poolIdentity
        let output = paged.attendWithModelKernel(
            keys: inputs.keys, values: inputs.values,
            // Prompt chunks keep the cache's write path so any chunk length can
            // take the prefill kernel (which reads only). Calls on the prefill
            // kernel (one threadgroup reads a row's pages) take no buckets: past
            // the binding limit they read a gathered copy.
            fusedWriteTokens: inputs.isPrefill ? 0 : Inkling2PagedAttention.splitMaximumQueries,
            bucketing: prefillKernel ? nil : bucketing
        ) { pages in
            precondition(pages.window == inputs.window, "Inkling2 slot \(slot): window mismatch")
            let plan = self.plan(
                engine: engine, pages.rows, queries: l, window: pages.window,
                scale: inputs.scale)
            let cache: Inkling2PagedAttention.PagedKV
            switch pages.storage {
            case .slab(let k, let v):
                cache = .init(
                    keys: k, values: v, tables: pages.tables, writeFence: pages.writeFence)
            case .segments(let segments):
                cache = .init(
                    segments: segments, tables: pages.tables, writeFence: pages.writeFence)
            case .segmentBuckets(let buckets):
                cache = .init(
                    buckets: buckets.map {
                        .init(
                            segments: $0.segments, ranges: $0.ranges, items: $0.items,
                            itemCount: $0.itemCount)
                    }, tables: pages.tables, writeFence: pages.writeFence)
            }
            let result = Inkling2PagedAttention.attend(
                queries: queries, newKeys: keys, newValues: values,
                relativeStates: inputs.relativeStates, proj: inputs.relativeProjection,
                sinks: sinks, cache: cache, plan: plan, write: pages.writesTokens,
                layout: .tokenMajor,
                kernel: Self.kernel(
                    queries: l, prefill: inputs.isPrefill, writes: pages.writesTokens,
                    dtype: keys.dtype, headDim: keys.dim(3),
                    relativeWidth: inputs.relativeProjection.dim(0), window: pages.window))
            return (result.output, result.writeFence)
        }
        return output ?? MLXArray.zeros([b, l, width], dtype: inputs.queries.dtype)
    }

    /// Prefill chunks run on the neural accelerators when the GPU has them:
    /// every prompt chunk (`isPrefill`, any length), and, for callers that do
    /// not set the flag, any call longer than a verification window
    /// (`splitMaximumQueries`). Decode steps and verify windows keep the
    /// decode arithmetic. Other K/V dtypes and shapes (FP32 test models) keep
    /// the decode kernels.
    static func kernel(
        queries: Int, prefill: Bool = false, writes: Bool, dtype: DType, headDim: Int,
        relativeWidth: Int, window: Int?
    ) -> Inkling2PagedAttention.Kernel {
        let accelerated =
            Inkling2PagedAttention.neuralAcceleratorsAvailable && !writes
            && (dtype == .bfloat16 || dtype == .float16) && headDim % 32 == 0
            && relativeWidth <= 16 && (window == nil || slidingPrefillOnAccelerators)
        return accelerated && (prefill || queries > Inkling2PagedAttention.splitMaximumQueries)
            ? .prefill : .automatic
    }

    /// Calls over more segments than one dispatch binds run one split
    /// dispatch per bucket of at most `maximumBucketSegments` segments (same
    /// partitions, same bits) instead of reading a gathered copy.
    static let defaultBucketing = CBv2ModelAttentionBucketing(
        partitionTokens: Inkling2PagedAttention.partitionTokens,
        maximumQueries: Inkling2PagedAttention.splitMaximumQueries,
        maximumSegments: Inkling2PagedAttention.maximumBucketSegments)

    /// Sliding layers' prefill on the accelerators too (their 1,024-key window
    /// needs the relative bias on every tile).
    static let slidingPrefillOnAccelerators = true

    private func plan(
        engine: ObjectIdentifier, _ rows: [CBv2ModelAttentionPages.Row], queries: Int,
        window: Int?, scale: Float
    ) -> Inkling2PagedAttention.Plan {
        let key = PlanKey(
            engine: engine, rows: rows, queries: queries, window: window, scale: scale)
        if let cached = lock.withLock({ plans[key] }) { return cached }
        // Built outside the lock (it makes the row metadata's device copy); the
        // first plan stored for a key wins, so every call of one key gets that
        // plan and its workspace.
        let made = Inkling2PagedAttention.Plan(
            rows: rows.map {
                .init(queryStart: $0.queryStart, base: $0.base, tableLength: $0.tableLength)
            }, queries: queries, window: window, scale: scale)
        return lock.withLock {
            if let cached = plans[key] { return cached }
            if plans.count >= Self.planCapacity { plans.removeAll(keepingCapacity: true) }
            plans[key] = made
            return made
        }
    }
}
