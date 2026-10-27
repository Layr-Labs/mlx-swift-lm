// Copyright 2026 Eigen Labs.
// Modified in 2026: Swift/MLXFast port of oMLX's GLM-5.3 KDA prefill
// recurrence (per-core and blocked kernels) onto Darkbloom's long KDA step,
// with Darkbloom's FP32 rounding boundaries, [B,H,K,V] state and validity.
// Source: omlx/patches/glm53_kda_recurrence.py, github.com/jundot/omlx
// (Apache-2.0), commit 3e043977bfaa885989d56c9762b3ea54b7a9a991.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
import MLX

#if canImport(IOKit)
import IOKit
#endif

/// Long-step KDA recurrence with each value row's FP32 state in registers,
/// over contiguous ranges of the H·D value rows.
///
/// For every token, head h, key k and value row v (state [B,H,K,V]):
///   S[k,v] *= decay[k];  p[v] = Σ_k S[k,v]·key[k];
///   S[k,v] += key[k]·(value[v] − p[v])·beta;  out[v] = Σ_k S[k,v]·query[k].
///
/// Launches (`Launch`), one kernel source:
/// - Per-core: one threadgroup per GPU core, each over one contiguous range
///   of the flattened (head, value row) axis (about 52 rows per core for 32
///   heads on an 80-core M5 Ultra, about 103 for 64). A range may cross one
///   head boundary; its block then stages q, k, decay and beta for both heads.
/// - Blocked: 64 rows per threadgroup, two threadgroups per head, one head
///   staged per block. Used when the GPU core count is unknown, a step has
///   too few rows per core, or a core would need more than 128 rows.
///
/// One value row per 8 lanes, 16 keys per lane (keys `i·8 + lane`, i < 16, as
/// `GLM5NextLinearDecodeKernel`'s long recurrence). Blocks of tokens are
/// fetched into registers while the previous block runs, then normalized,
/// gated and staged as FP32 in threadgroup memory in a lane-major order, so
/// each lane reads its 16 keys as four `float4` (128 contiguous bytes per
/// 8-lane row). Full blocks are unrolled; the query sums of four steps are
/// reduced together (a reduce-scatter with the same xor 4, 2, 1 tree, so the
/// same bits) after the next step's decay and key sums.
///
/// `Order.serving` sums each lane's 16 products in key order, exactly as the
/// existing long recurrence does, so with the fused inputs every output and
/// state bit is designed to match `GLM5NextLinearDecodeKernel`'s
/// `longRecurrenceKernel`: same staged FP32 values (same expressions, norms
/// from the same lane-strided partials and `simd_sum`), same products and
/// sums, `fp contract(off)`. `Order.paired` is oMLX's order (four `float4`
/// partials, (x + y) + (z + w)); it changes FP32 rounding, so it is not
/// bitwise to either path.
enum GLM5NextKDAPrefillKernel {
    /// Head width the kernels read: 8 lanes x 16 keys per value row.
    static let dimension = 128
    /// Shortest step `launch(tokens:heads:dimension:)` admits.
    static let minimumTokens = 32
    /// Tokens per staged block. Per-core blocks hold two heads' FP32 q, k and
    /// decay; 8 tokens keep that within 32 KiB of threadgroup memory.
    static let perCoreBlock = 8
    static let blockedBlock = 16
    /// Value rows of one blocked threadgroup.
    static let blockedRows = 64
    /// Rows per core below which per-core ranges leave SIMD units idle; such
    /// steps (few heads) keep the blocked launch.
    static let minimumRowsPerCore = 32
    static let threadgroupMemoryLimit = 32 * 1024

    enum Order: Sendable, Hashable {
        /// Designed bit-identical to the value-split long recurrence
        /// (`GLM5NextLinearDecodeKernel`'s `longRecurrenceKernel`).
        case serving
        /// oMLX's four-accumulator order; not bitwise.
        case paired
    }

    struct Launch: Sendable, Hashable {
        /// Threadgroups per batch row, each over a contiguous range of rows.
        let threadgroups: Int
        /// Row capacity per threadgroup: 64 or 128, i.e. 512 or 1024 threads,
        /// the sizes for which MLX recompiles a pipeline whose register use
        /// would otherwise lower its thread limit below the launch.
        let capacity: Int
        /// Heads staged per block: 2 when a range may cross a head boundary.
        let slots: Int
        /// Tokens per staged block, a multiple of four.
        let block: Int
        /// Row bounds of the pipelined kernel (`Pipelined`): threadgroup i runs
        /// rows `[ranges[i], ranges[i + 1])`. nil: the single-buffer kernel's
        /// even split `[i·W / NTG, (i + 1)·W / NTG)`.
        var ranges: [Int32]? = nil

        var threads: Int { 8 * capacity }

        var threadgroupBytes: Int {
            let d = GLM5NextKDAPrefillKernel.dimension
            // The pipelined kernel double-buffers the values of single-head ranges.
            let values = ranges == nil ? 1 : 2
            return 3 * slots * block * d * 4 + values * block * capacity * 4 + slots * block * 4
        }

        /// One range per core; nil when a range would exceed 128 rows.
        static func perCore(heads: Int, cores: Int) -> Launch? {
            guard heads > 0, cores > 0 else { return nil }
            let rows = heads * GLM5NextKDAPrefillKernel.dimension
            let groups = min(cores, rows)
            let widest = (rows + groups - 1) / groups
            guard widest <= 128 else { return nil }
            return Launch(
                threadgroups: groups, capacity: widest <= 64 ? 64 : 128, slots: 2,
                block: GLM5NextKDAPrefillKernel.perCoreBlock)
        }

        static func blocked(heads: Int) -> Launch? {
            guard heads > 0 else { return nil }
            return Launch(
                threadgroups: heads * GLM5NextKDAPrefillKernel.dimension
                    / GLM5NextKDAPrefillKernel.blockedRows,
                capacity: GLM5NextKDAPrefillKernel.blockedRows, slots: 1,
                block: GLM5NextKDAPrefillKernel.blockedBlock)
        }

        /// At most `cores` ranges of at most 64 rows (512 threads) for the
        /// pipelined kernel, balanced by `Pipelined.cost`: the most expensive
        /// range is as cheap as the greedy longest-first cut allows (optimal,
        /// since a range's cost only falls as its start moves right). For 32
        /// heads on 80 cores each head pair is cut 56/56/32/56/56: the 56-row
        /// ranges stay within one head, the 32-row range crosses the boundary.
        /// nil when no such cut exists (e.g. 64 heads on 80 cores). Cached per
        /// (heads, cores).
        static func pipelined(heads: Int, cores: Int) -> Launch? {
            guard heads > 0, cores > 0 else { return nil }
            let key = [heads, cores]
            if let cached = Pipelined.cuts.value(key) { return cached }
            let launch = Pipelined.cut(
                rows: heads * GLM5NextKDAPrefillKernel.dimension, ranges: cores
            )
            .map {
                Launch(
                    threadgroups: $0.count - 1, capacity: Pipelined.capacity, slots: 2,
                    block: GLM5NextKDAPrefillKernel.perCoreBlock, ranges: $0)
            }
            Pipelined.cuts.store(key, launch)
            return launch
        }
    }

    /// The launch for a long step on this device, or nil to keep the existing
    /// recurrence: Neural-Accelerator GPUs (M5 and later, where oMLX measured
    /// the per-core kernel ahead), 128-wide heads, at least `minimumTokens`.
    /// Per-core steps take the pipelined ranges when they exist (32 heads on
    /// 80 cores: one tensor-parallel rank), else the even per-core split.
    static func launch(tokens: Int, heads: Int, dimension: Int) -> Launch? {
        guard glm5NextNAXAvailable, dimension == Self.dimension, tokens >= minimumTokens,
            heads > 0
        else { return nil }
        if let cores = gpuCoreCount, heads * dimension >= minimumRowsPerCore * cores {
            if pipelinedEnabled, let launch = Launch.pipelined(heads: heads, cores: cores) {
                return launch
            }
            if let launch = Launch.perCore(heads: heads, cores: cores) { return launch }
        }
        return Launch.blocked(heads: heads)
    }

    /// `GLM53_KDA_PIPELINED=0` keeps the even per-core split for long steps
    /// (an A/B of the same bits), read once per process.
    static let pipelinedEnabled =
        ProcessInfo.processInfo.environment["GLM53_KDA_PIPELINED"] != "0"

    /// The accelerator's IORegistry `gpu-core-count` (80 on an M5 Ultra), read
    /// once per process, as `ClusterDeviceProfileSampler.gpuCoreCount` does.
    /// Neither Metal nor `GPU.deviceInfo()` exposes the core count.
    static let gpuCoreCount: Int? = {
        #if os(macOS) && canImport(IOKit)
        var iterator: io_iterator_t = 0
        guard
            IOServiceGetMatchingServices(
                kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &iterator)
                == KERN_SUCCESS
        else { return nil }
        defer { IOObjectRelease(iterator) }
        var result: Int?
        while true {
            let service = IOIteratorNext(iterator)
            guard service != 0 else { break }
            defer { IOObjectRelease(service) }
            if let property = IORegistryEntryCreateCFProperty(
                service, "gpu-core-count" as CFString, kCFAllocatorDefault, 0),
                let number = property.takeRetainedValue() as? NSNumber,
                (1 ... 4096).contains(number.intValue)
            {
                result = max(result ?? 0, number.intValue)
            }
        }
        return result
        #else
        return nil
        #endif
    }()

    // MLXFast has stubs where Metal is absent, so these compile everywhere;
    // `launch` is nil there (no Neural Accelerators), so they never run.

    /// The long step's recurrence, same inputs and outputs as
    /// `GLM5NextLinearDecodeKernel.longRecurrence`: activated q/k/v
    /// `[B,T,3·H·D]` (after convolution, SiLU and the activation-dtype
    /// rounding, before the L2 norms), projected forget gate `[B,T,H·D]`,
    /// beta logits read at `token·betaLayout[0] + betaLayout[1] + h`, FP32
    /// state `[B,H,K,V]` and validity `[B,T]`. Returns core outputs
    /// `[B,T,H,D]` in the activation dtype (before the gated output norm)
    /// and the new FP32 state. An invalid token's decay is one and its
    /// beta zero, so it leaves the state unchanged.
    static func recurrence(
        activated: MLXArray, forget: MLXArray, betaLogits: MLXArray, betaLayout: [Int32],
        aLog: MLXArray, dtBias: MLXArray, state: MLXArray, validity: MLXArray,
        gateLowerBound: Float, queryScale: Float, launch: Launch, order: Order = .serving
    ) -> (core: MLXArray, state: MLXArray) {
        let (batch, heads, d) = (state.dim(0), state.dim(1), state.dim(2))
        let tokens = activated.dim(1)
        precondition(
            d == dimension && state.dim(3) == d && state.dtype == .float32
                && activated.shape == [batch, tokens, 3 * heads * d]
                && forget.size == batch * tokens * heads * d
                && validity.shape == [batch, tokens] && betaLayout.count == 2,
            "long KDA recurrence geometry")
        check(launch, heads: heads)
        if let ranges = launch.ranges {
            precondition(order == .serving, "the pipelined kernel runs the serving order")
            let outputs = pipelinedKernel(
                [
                    activated, forget, betaLogits, aLog, dtBias, state, validity,
                    MLXArray([gateLowerBound, queryScale]), MLXArray(Int32(tokens)),
                    MLXArray(betaLayout), GLM5NextKernelConstants.shared.int32s(ranges),
                ],
                template: [
                    ("T", activated.dtype), ("H", heads), ("D", d), ("MR", launch.capacity),
                ],
                grid: (launch.threads * launch.threadgroups, 1, batch),
                threadGroup: (launch.threads, 1, 1),
                outputShapes: [[batch, tokens, heads, d], state.shape],
                outputDTypes: [activated.dtype, .float32])
            return (outputs[0], outputs[1])
        }
        let outputs = kernels[Variant(fused: true, order: order, block: launch.block)]!(
            [
                activated, forget, betaLogits, aLog, dtBias, state, validity,
                MLXArray([gateLowerBound, queryScale]), MLXArray(Int32(tokens)),
                MLXArray(betaLayout),
            ],
            template: [
                ("T", activated.dtype), ("H", heads), ("D", d), ("MR", launch.capacity),
                ("NTG", launch.threadgroups), ("SLOTS", launch.slots),
            ],
            grid: (launch.threads * launch.threadgroups, 1, batch),
            threadGroup: (launch.threads, 1, 1),
            outputShapes: [[batch, tokens, heads, d], state.shape],
            outputDTypes: [activated.dtype, .float32])
        return (outputs[0], outputs[1])
    }

    /// The same recurrence over `GLM5NextLinearAttention.apply`'s
    /// precomputed FP32 inputs (the `GLM5NextLinearKernel` contract): q
    /// normalized and scaled, k normalized, v, per-key log decay (zero for
    /// an invalid token) `[B,T,H,D]`, beta `[B,T,H]` (zero for an invalid
    /// token), state `[B,H,K,V]`. Returns FP32 values `[B,T,H,D]` and state.
    static func recurrence(
        query: MLXArray, key: MLXArray, value: MLXArray, logDecay: MLXArray,
        beta: MLXArray, state: MLXArray, launch: Launch, order: Order = .serving
    ) -> (values: MLXArray, state: MLXArray) {
        let (batch, tokens, heads, d) = (query.dim(0), query.dim(1), query.dim(2), query.dim(3))
        precondition(
            d == dimension
                && [key, value, logDecay].allSatisfy { $0.shape == query.shape }
                && beta.shape == [batch, tokens, heads]
                && state.shape == [batch, heads, d, d]
                && [query, key, value, logDecay, beta, state].allSatisfy {
                    $0.dtype == .float32
                },
            "FP32 KDA recurrence geometry")
        check(launch, heads: heads)
        precondition(launch.ranges == nil, "the pipelined kernel reads the fused inputs only")
        let outputs = kernels[Variant(fused: false, order: order, block: launch.block)]!(
            [query, key, value, logDecay, beta, state, MLXArray(Int32(tokens))],
            template: [
                ("H", heads), ("D", d), ("MR", launch.capacity),
                ("NTG", launch.threadgroups), ("SLOTS", launch.slots),
            ],
            grid: (launch.threads * launch.threadgroups, 1, batch),
            threadGroup: (launch.threads, 1, 1),
            outputShapes: [query.shape, state.shape],
            outputDTypes: [.float32, .float32])
        return (outputs[0], outputs[1])
    }

    /// Every range fits its row capacity and staged heads, and the
    /// threadgroup arrays fit; the kernel itself does not check.
    private static func check(_ launch: Launch, heads: Int) {
        let rows = heads * dimension
        if let ranges = launch.ranges {
            precondition(
                launch.capacity == Pipelined.capacity && launch.slots == 2
                    && launch.block == perCoreBlock && ranges.count == launch.threadgroups + 1
                    && ranges.first == 0 && ranges.last == Int32(rows)
                    && zip(ranges, ranges.dropFirst()).allSatisfy { start, end in
                        start < end && Int(end - start) <= launch.capacity
                            && Int(end - 1) / dimension - Int(start) / dimension <= 1
                    }
                    && launch.threadgroupBytes <= threadgroupMemoryLimit,
                "KDA pipelined launch \(launch) for \(heads) heads")
            return
        }
        let widest = (rows + launch.threadgroups - 1) / launch.threadgroups
        precondition(
            [64, 128].contains(launch.capacity) && widest <= launch.capacity
                && (launch.slots == 2
                    || (rows % launch.threadgroups == 0 && dimension % widest == 0))
                && [1, 2].contains(launch.slots)
                && [perCoreBlock, blockedBlock].contains(launch.block)
                && launch.threadgroupBytes <= threadgroupMemoryLimit,
            "KDA prefill launch \(launch) for \(heads) heads")
    }

    private struct Variant: Hashable {
        let fused: Bool
        let order: Order
        let block: Int
    }

    private static let kernels: [Variant: MLXFast.MLXFastKernel] = Dictionary(
        uniqueKeysWithValues: [true, false].flatMap { fused in
            [Order.serving, .paired].flatMap { order in
                [perCoreBlock, blockedBlock].map { block in
                    let variant = Variant(fused: fused, order: order, block: block)
                    return (variant, makeKernel(variant))
                }
            }
        })

    private static func makeKernel(_ variant: Variant) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: "glm5_next_kda_prefill_" + (variant.fused ? "fused" : "fp32")
                + (variant.order == .paired ? "_paired" : "") + "_tb\(variant.block)",
            inputNames: variant.fused
                ? [
                    "activated", "forget", "beta_logits", "a_log", "dt_bias", "state_in",
                    "validity", "params", "token_count", "beta_layout",
                ]
                : ["query", "key", "value", "log_decay", "beta", "state_in", "token_count"],
            outputNames: variant.fused ? ["core", "state_out"] : ["y", "state_out"],
            source: Source.text(fused: variant.fused, order: variant.order, block: variant.block),
            ensureRowContiguous: true)
    }

    private static let pipelinedKernel = MLXFast.metalKernel(
        name: "glm5_next_kda_prefill_pipelined",
        inputNames: [
            "activated", "forget", "beta_logits", "a_log", "dt_bias", "state_in", "validity",
            "params", "token_count", "beta_layout", "row_bounds",
        ],
        outputNames: ["core", "state_out"],
        source: Pipelined.text(),
        ensureRowContiguous: true)

    // MARK: - Pipelined kernel (fused inputs, serving order)

    /// The per-core serving recurrence with three changes, none of which
    /// touches a value row's FP32 operations or their order:
    ///
    /// 1. Ranges come from a table (`Launch.pipelined`) that keeps every range
    ///    within one head at most 56 rows (7 SIMD groups per issue partition)
    ///    and gives the range across a head boundary, which stages two heads,
    ///    the fewest rows. The even split puts 52 rows and two staged heads on
    ///    the slowest cores.
    /// 2. A range within one head (at most 60 rows, so its last SIMD group
    ///    holds no rows) double-buffers its 8-token blocks: block n + 1 is
    ///    staged at the top of block n's straight-line step code, where the
    ///    compiler interleaves it with the steps, and each block ends in one
    ///    barrier. Ranges across a boundary keep the single-buffer schedule.
    /// 3. The q/k norms' `x / denominator` (about 45 issue slots in safe math)
    ///    is `y = 1 / denominator` once per row and Markstein's correction per
    ///    element: `q0 = x·y; r = fma(−q0, denominator, x); fma(r, y, q0)`.
    ///    For 16-bit activations this equals Metal's correctly rounded `/` for
    ///    every pair of significands (checked exhaustively on the GPU: all
    ///    2^30 bf16 × f32 and 2^33 f16 × f32 pairs at several exponent
    ///    offsets), hence everywhere all intermediates stay normal:
    ///    |x| ≥ 2^-80 and denominator < 2^40. A row with a numerator outside
    ///    that range is divided again with `/` and overwritten (never seen in
    ///    practice); FP32 activations always use `/`.
    ///
    /// Every other expression (norms, gates, beta rounding, steps, the
    /// four-step reduce-scatter of the query sums) is the single-buffer
    /// kernel's, so the outputs and states are bit-identical to it.
    enum Pipelined {
        /// Rows per range at most; 512 threads.
        static let capacity = 64

        /// Kernel time in 0.01 ms (T = 4,096, M5 Ultra) of a range by its SIMD
        /// groups per issue partition, ceil(rows / 8): within one head
        /// (double-buffered while it leaves a SIMD group without rows, i.e. up
        /// to 60 rows; 61–64 rows run the single-buffer schedule) and across a
        /// head boundary (two heads staged in one buffer). They only steer
        /// `cut`: no bit depends on where the ranges fall.
        static let singleHeadCost = [0, 100, 110, 120, 135, 153, 168, 193, 206]
        static let fullSingleHeadCost = 235
        static let crossingCost = [0, 165, 174, 183, 194, 213, 227, 247, 263]

        static func cost(start: Int, rows: Int) -> Int {
            let d = GLM5NextKDAPrefillKernel.dimension
            let groups = (rows + 7) / 8
            switch (start + rows - 1) / d - start / d {
            case 0: return rows <= capacity - 4 ? singleHeadCost[groups] : fullSingleHeadCost
            case 1: return crossingCost[groups]
            default: return .max
            }
        }

        /// Bounds of at most `ranges` ranges of whole SIMD groups (4 rows) over
        /// `rows` rows minimizing the largest `cost`: for each candidate bound,
        /// ascending, cut greedily (each range as long as the bound allows).
        /// A range's cost never rises as it grows shorter from the left, and
        /// never falls as it grows longer to the right, so the greedy cut uses
        /// the fewest ranges for its bound and the first bound that fits is
        /// optimal.
        static func cut(rows: Int, ranges: Int) -> [Int32]? {
            guard rows > 0, rows % 4 == 0, ranges > 0 else { return nil }
            let bounds = Set(singleHeadCost.dropFirst() + crossingCost.dropFirst())
                .union([fullSingleHeadCost]).sorted()
            for bound in bounds {
                var cut: [Int32] = [0]
                var start = 0
                while start < rows, cut.count <= ranges {
                    var length = 0
                    for candidate in stride(from: 4, through: min(capacity, rows - start), by: 4)
                    where cost(start: start, rows: candidate) <= bound {
                        length = candidate
                    }
                    guard length > 0 else { break }
                    start += length
                    cut.append(Int32(start))
                }
                if start == rows, cut.count - 1 <= ranges { return cut }
            }
            return nil
        }

        /// `Launch.pipelined` results per (heads, cores).
        final class Cuts: @unchecked Sendable {
            private let lock = NSLock()
            private var launches: [[Int]: Launch?] = [:]

            func value(_ key: [Int]) -> Launch?? {
                lock.lock()
                defer { lock.unlock() }
                return launches[key]
            }

            func store(_ key: [Int], _ launch: Launch?) {
                lock.lock()
                defer { lock.unlock() }
                launches[key] = launch
            }
        }

        static let cuts = Cuts()

        /// Kernel body. Template arguments: `T` (activation dtype), `H`, `D`
        /// (128), `MR` (64).
        static func text() -> String {
            """
            #pragma clang fp reassociate(off)
            #pragma clang fp contract(off)
            constexpr int TB = 8;
            constexpr int NT = MR * 8;
            constexpr int NSG = NT / 32;
            constexpr int W = H * D;
            constexpr int Q4 = D / 4;
            constexpr bool FAST_DIVISION = sizeof(T) == 2;
            static_assert(D == 128, "eight lanes of sixteen keys per value row");
            static_assert(MR == 64, "512 threads: two value buffers fit 32 KiB");
            // Index 0/1 of the staged arrays is the head of a range across a
            // boundary, or the buffer of a range within one head. Rows are
            // lane-major: key i * 8 + l (i < 16, l < 8) sits at
            // 32 * (i / 4) + 4 * l + i % 4.
            threadgroup float4 q_s[2][TB][Q4];
            threadgroup float4 k_s[2][TB][Q4];
            threadgroup float4 g_s[2][TB][Q4];
            threadgroup float v_s[2][TB][MR];
            threadgroup float b_s[2][TB];

            const int tid = int(thread_position_in_threadgroup.x);
            const int tg = int(threadgroup_position_in_grid.x);
            const int b = int(threadgroup_position_in_grid.z);
            const int tokens = token_count;
            const size_t row0 = size_t(b) * size_t(tokens);
            const int r0 = row_bounds[tg];
            const int r1 = row_bounds[tg + 1];
            const int nrows = r1 - r0;
            const int hA = r0 / D;
            const int nh = (r1 - 1) / D - hA + 1;
            const int sg = tid / 32;
            const int lane = tid % 32;
            // Eight lanes per value row; lane ln owns keys i * 8 + ln.
            const int rg = tid / 8;
            const int ln = tid % 8;
            const bool active = rg < nrows;
            const int slot = active ? (r0 + rg) / D - hA : 0;
            const bool qb2 = (ln & 4) != 0;
            const bool qb1 = (ln & 2) != 0;
            const int ysel = (qb2 ? 2 : 0) + (qb1 ? 1 : 0);
            const int grow = active ? r0 + rg : r0;
            const size_t state_base = (size_t(b) * H + size_t(grow / D)) * D * D + size_t(grow % D);
            float4 st[4];
            for (int j = 0; j < 4; ++j) {
                for (int c = 0; c < 4; ++c) {
                    st[j][c] = state_in[state_base + size_t((4 * j + c) * 8 + ln) * D];
                }
            }
            const float bound = params[0];
            const float query_scale = params[1];
            const size_t beta_stride = size_t(beta_layout[0]);
            const size_t beta_column = size_t(beta_layout[1]);
            const int nblocks = (tokens + TB - 1) / TB;
            // SIMD groups [0, csg) hold value rows.
            const int csg = (nrows + 3) / 4;
            float yq0 = 0.0f;
            float yq1 = 0.0f;
            float yq2 = 0.0f;
            float yq3 = 0.0f;

            if (nh == 1 && csg < NSG) {
            \(doubleBuffered())
            } else {
            \(singleBuffered())
            }

            if (active) {
                for (int j = 0; j < 4; ++j) {
                    for (int c = 0; c < 4; ++c) {
                        state_out[state_base + size_t((4 * j + c) * 8 + ln) * D] = st[j][c];
                    }
                }
            }
            """
        }

        // MARK: Ranges within one head: two buffers

        /// Block n + 1 is staged while block n runs. In steady blocks (this and
        /// the next block full) the row SIMD groups stage at the top of the
        /// unrolled steps and fetch block n + 2 after staging; other blocks run
        /// the steps, then stage.
        private static func doubleBuffered() -> String {
            var steps = ""
            for t in 0 ..< 8 {
                steps += step("\(t)", index: "buf", reducing: t == 4 ? 0 : nil, keeping: t % 4)
            }
            steps += reduceFour(4)
            let tail =
                "for (int t = 0; t < tt; ++t) "
                + step("t", index: "buf", reducing: nil, keeping: nil)
            return """
                    // SIMD group sg stages the q (even) or k (odd) row of block token
                    // sg / 2; each thread the decay of channel tid % D at block tokens
                    // tid / D and tid / D + 4, and the value of token tid / MR, row
                    // tid % MR; the last SIMD group (no rows) the betas.
                    const int a_kind = sg % 2;
                    const int a_tok = sg / 2;
                    const int g_c = tid % D;
                    const int g_r = tid / D;
                    const int g_pos = 32 * (g_c / 32) + 4 * (g_c % 8) + (g_c / 8) % 4;
                    const float a_rate = metal::precise::exp(static_cast<float>(a_log[hA]));
                    const float a_bias = static_cast<float>(dt_bias[hA * D + g_c]);
                    const int v_r = tid / MR;
                    const int v_c = tid % MR;
                    const bool beta_lane = sg == NSG - 1 && lane < TB;
                    float aq[4];
                    float ag0;
                    float ag1;
                    bool agv0;
                    bool agv1;
                    float av;
                    float ab;
                    bool abv;
                    bool a_fix = false;
                    \(fetchOne("0"))
                    \(stageOne("0", count: "min(TB, tokens)", full: false))
                    \(betaOne("0", count: "min(TB, tokens)"))
                    \(fixOne("0", first: "0", count: "min(TB, tokens)"))
                    \(fetchOne("TB"))
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (int blk = 0; blk < nblocks; ++blk) {
                        const int t0 = blk * TB;
                        const int tt = min(TB, tokens - t0);
                        const int buf = blk & 1;
                        if (tt == TB && t0 + 2 * TB <= tokens) {
                            if (sg < csg) {
                                \(stageOne("buf ^ 1", count: "TB", full: true))
                                \(fetchOne("t0 + 2 * TB"))
                \(steps)
                            } else {
                                \(stageOne("buf ^ 1", count: "TB", full: true))
                                \(betaOne("buf ^ 1", count: "TB"))
                                \(fetchOne("t0 + 2 * TB"))
                            }
                            \(fixOne("buf ^ 1", first: "t0 + TB", count: "TB"))
                        } else {
                            if (tt == TB) {
                \(steps)
                            } else if (active) {
                \(tail)
                            }
                            if (blk + 1 < nblocks) {
                                \(stageOne("buf ^ 1", count: "min(TB, tokens - t0 - TB)", full: false))
                                \(betaOne("buf ^ 1", count: "min(TB, tokens - t0 - TB)"))
                                \(fixOne("buf ^ 1", first: "t0 + TB", count: "min(TB, tokens - t0 - TB)"))
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                """
        }

        /// Loads of the block from token `first` into registers. Token indices
        /// are clamped, so the loads need no branch; values past the last token
        /// are never staged.
        private static func fetchOne(_ first: String) -> String {
            """
            {
                                const int f0 = \(first);
                                const size_t qrow = row0 + size_t(min(f0 + a_tok, tokens - 1));
                                const size_t qbase = qrow * size_t(3 * W) + size_t(a_kind * W + hA * D + lane);
                                for (int i = 0; i < 4; ++i) {
                                    aq[i] = static_cast<float>(activated[qbase + size_t(32 * i)]);
                                }
                                const size_t gt0 = row0 + size_t(min(f0 + g_r, tokens - 1));
                                const size_t gt1 = row0 + size_t(min(f0 + g_r + 4, tokens - 1));
                                ag0 = static_cast<float>(forget[gt0 * size_t(W) + size_t(hA * D + g_c)]);
                                ag1 = static_cast<float>(forget[gt1 * size_t(W) + size_t(hA * D + g_c)]);
                                agv0 = validity[gt0];
                                agv1 = validity[gt1];
                                const size_t vt = row0 + size_t(min(f0 + v_r, tokens - 1));
                                av = static_cast<float>(
                                    activated[vt * size_t(3 * W) + size_t(2 * W + r0 + min(v_c, nrows - 1))]);
                                const size_t bt = row0 + size_t(min(f0 + lane % TB, tokens - 1));
                                ab = static_cast<float>(beta_logits[bt * beta_stride + beta_column + size_t(hA)]);
                                abv = validity[bt];
                            }
            """
        }

        /// Registers -> buffer `buffer` for `count` tokens (`full`: count is TB,
        /// no token guards): the q/k norm (each lane's four squares in channel
        /// order, then `simd_sum`), decay exp(bound·sigmoid(rate·(f + dt_bias)))
        /// and the value.
        private static func stageOne(_ buffer: String, count: String, full: Bool) -> String {
            """
            {
                                const int sb = \(buffer);
                                const int sn = \(count);
                                (void)sn;
                                \(full ? "" : "if (a_tok < sn) "){
                                    float squares = 0.0f;
                                    for (int i = 0; i < 4; ++i) {
                                        const float square = aq[i] * aq[i];
                                        squares += square;
                                    }
                                    squares = simd_sum(squares);
                                    const float denominator = metal::precise::sqrt(squares + 1e-6f);
                                    threadgroup float* row = a_kind == 0
                                        ? (threadgroup float*)&q_s[sb][a_tok][0]
                                        : (threadgroup float*)&k_s[sb][a_tok][0];
                                    \(division("aq", fix: "a_fix", query: "a_kind == 0"))
                                }
                                \(gate("ag0", valid: "agv0", token: "g_r", guarded: !full))
                                \(gate("ag1", valid: "agv1", token: "g_r + 4", guarded: !full))
                                \(full ? "" : "if (v_r < sn) ")if (v_c < nrows) { v_s[sb][v_r][v_c] = av; }
                            }
            """
        }

        private static func gate(_ f: String, valid: String, token: String, guarded: Bool) -> String
        {
            """
            \(guarded ? "if (\(token) < sn) " : ""){
                                    const float gate_input = \(f) + a_bias;
                                    const float x = a_rate * gate_input;
                                    const float e = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
                                    const float log_decay = \(valid) ? bound * (x < 0.0f ? e : 1.0f - e) : 0.0f;
                                    ((threadgroup float*)&g_s[sb][\(token)][0])[g_pos] = metal::precise::exp(log_decay);
                                }
            """
        }

        /// Beta rounded to the activation dtype; zero when invalid.
        private static func betaOne(_ buffer: String, count: String) -> String {
            """
            if (beta_lane && lane < \(count)) {
                                const float e = 1.0f / (1.0f + metal::precise::exp(metal::abs(ab)));
                                const float rounded_beta = static_cast<float>(static_cast<T>(ab < 0.0f ? e : 1.0f - e));
                                b_s[\(buffer)][lane] = abv ? rounded_beta : 0.0f;
                            }
            """
        }

        /// The exact fallback of a staged q/k row (reloaded: the registers
        /// already hold the next block).
        private static func fixOne(_ buffer: String, first: String, count: String) -> String {
            """
            if (simd_any(a_fix)) {
                                if (a_tok < \(count)) {
                                    const size_t qrow = row0 + size_t(\(first) + a_tok);
                                    const size_t qbase = qrow * size_t(3 * W) + size_t(a_kind * W + hA * D + lane);
                                    float xq[4];
                                    for (int i = 0; i < 4; ++i) {
                                        xq[i] = static_cast<float>(activated[qbase + size_t(32 * i)]);
                                    }
                                    \(exactRow("xq", row: "a_kind == 0 ? (threadgroup float*)&q_s[\(buffer)][a_tok][0] : (threadgroup float*)&k_s[\(buffer)][a_tok][0]", query: "a_kind == 0"))
                                }
                                a_fix = false;
                            }
            """
        }

        /// `numerators[i] / denominator` into `row` (times query_scale for a
        /// query row) by the reciprocal and Markstein's correction; sets `fix`
        /// when the row needs the exact fallback.
        private static func division(_ n: String, fix: String, query: String) -> String {
            """
            const float inverse = 1.0f / denominator;
                                    bool exact = denominator < 0x1.0p40f;
                                    float unit[4];
                                    for (int i = 0; i < 4; ++i) {
                                        const float q0 = \(n)[i] * inverse;
                                        const float residual = metal::fma(-q0, denominator, \(n)[i]);
                                        unit[i] = FAST_DIVISION ? metal::fma(residual, inverse, q0) : \(n)[i] / denominator;
                                        exact = exact && metal::abs(\(n)[i]) >= 0x1.0p-80f;
                                    }
                                    for (int i = 0; i < 4; ++i) {
                                        row[32 * i + 4 * (lane % 8) + lane / 8] = (\(query)) ? unit[i] * query_scale : unit[i];
                                    }
                                    \(fix) = \(fix) || (FAST_DIVISION && !exact);
            """
        }

        /// The single-buffer kernel's norm of `x` (a whole SIMD group) into
        /// `row`, with `/`.
        private static func exactRow(_ x: String, row: String, query: String) -> String {
            """
            threadgroup float* row = \(row);
                                    float squares = 0.0f;
                                    for (int i = 0; i < 4; ++i) {
                                        const float square = \(x)[i] * \(x)[i];
                                        squares += square;
                                    }
                                    squares = simd_sum(squares);
                                    const float denominator = metal::precise::sqrt(squares + 1e-6f);
                                    for (int i = 0; i < 4; ++i) {
                                        const float unit = \(x)[i] / denominator;
                                        row[32 * i + 4 * (lane % 8) + lane / 8] = (\(query)) ? unit * query_scale : unit;
                                    }
            """
        }

        // MARK: Ranges across a head boundary: one buffer, two heads

        /// The single-buffer kernel's schedule (fetch, steps, barrier, stage,
        /// barrier), indexing the staged arrays by head slot.
        private static func singleBuffered() -> String {
            var steps = ""
            for t in 0 ..< 8 {
                steps += step("\(t)", index: "slot", reducing: t == 4 ? 0 : nil, keeping: t % 4)
            }
            steps += reduceFour(4)
            let tail =
                "for (int t = 0; t < tt; ++t) "
                + step("t", index: "slot", reducing: nil, keeping: nil)
            return """
                    constexpr int NQUAD = 2 * TB * Q4;
                    constexpr int NG = (NQUAD + NT - 1) / NT;
                    constexpr int NV = (TB * MR + NT - 1) / NT;
                    constexpr int NTASK = 4 * TB;
                    constexpr int QKR = (NTASK + NSG - 1) / NSG;
                    // A q/k row is fetched by one SIMD group (lane holds channels
                    // lane + 32 i); a gate quad holds channels 32 j + 8 m + l (m < 4).
                    float gate_bias[NG][4];
                    float gate_rate[NG];
                    for (int j = 0; j < NG; ++j) {
                        const int p = tid + j * NT;
                        const int h = hA + min((p / Q4) / TB, nh - 1);
                        const int quad = p % Q4;
                        gate_rate[j] = metal::precise::exp(static_cast<float>(a_log[h]));
                        for (int m = 0; m < 4; ++m) {
                            gate_bias[j][m] = static_cast<float>(dt_bias[h * D + 32 * (quad / 8) + 8 * m + quad % 8]);
                        }
                    }
                    float fq[QKR][4];
                    float fg[NG][4];
                    bool fgv[NG];
                    float fv[NV];
                    float fb = 0.0f;
                    bool fbv = false;
                    for (int j = 0; j < QKR; ++j) {
                        for (int i = 0; i < 4; ++i) { fq[j][i] = 0.0f; }
                    }
                    for (int j = 0; j < NG; ++j) {
                        fgv[j] = false;
                        for (int m = 0; m < 4; ++m) { fg[j][m] = 0.0f; }
                    }
                    for (int j = 0; j < NV; ++j) { fv[j] = 0.0f; }
                    \(fetchTwo("0"))
                    \(stageTwo("min(TB, tokens)"))
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    for (int t0 = 0; t0 < tokens; t0 += TB) {
                        const int tt = min(TB, tokens - t0);
                        const bool more = t0 + TB < tokens;
                        if (more) \(fetchTwo("t0 + TB"))
                        if (active) {
                            if (tt == TB) {
                \(steps)
                            } else {
                \(tail)
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        if (more) {
                            \(stageTwo("min(TB, tokens - t0 - TB)"))
                            threadgroup_barrier(mem_flags::mem_threadgroup);
                        }
                    }
                """
        }

        private static func fetchTwo(_ first: String) -> String {
            """
            {
                        const int f0 = \(first);
                        const int fn = min(TB, tokens - f0);
                        for (int j = 0; j < QKR; ++j) {
                            const int task = sg + j * NSG;
                            const int s = (task / 2) / TB;
                            const int r = (task / 2) % TB;
                            if (task < NTASK && s < nh && r < fn) {
                                const size_t base = (row0 + size_t(f0 + r)) * size_t(3 * W)
                                    + size_t((task % 2) * W + (hA + s) * D + lane);
                                for (int i = 0; i < 4; ++i) {
                                    fq[j][i] = static_cast<float>(activated[base + size_t(32 * i)]);
                                }
                            }
                        }
                        for (int j = 0; j < NG; ++j) {
                            const int p = tid + j * NT;
                            const int s = (p / Q4) / TB;
                            const int r = (p / Q4) % TB;
                            const int quad = p % Q4;
                            if (p < NQUAD && s < nh && r < fn) {
                                const size_t token = row0 + size_t(f0 + r);
                                const size_t base = token * size_t(W)
                                    + size_t((hA + s) * D + 32 * (quad / 8) + quad % 8);
                                for (int m = 0; m < 4; ++m) {
                                    fg[j][m] = static_cast<float>(forget[base + size_t(8 * m)]);
                                }
                                fgv[j] = validity[token];
                            }
                        }
                        for (int j = 0; j < NV; ++j) {
                            const int p = tid + j * NT;
                            const int r = p / MR;
                            const int c = p % MR;
                            if (r < fn && c < nrows) {
                                fv[j] = static_cast<float>(
                                    activated[(row0 + size_t(f0 + r)) * size_t(3 * W) + size_t(2 * W + r0 + c)]);
                            }
                        }
                        if (tid < 2 * TB) {
                            const int s = tid / TB;
                            const int r = tid % TB;
                            if (s < nh && r < fn) {
                                const size_t token = row0 + size_t(f0 + r);
                                fb = static_cast<float>(beta_logits[token * beta_stride + beta_column + size_t(hA + s)]);
                                fbv = validity[token];
                            }
                        }
                    }
            """
        }

        private static func stageTwo(_ count: String) -> String {
            """
            {
                        const int sn = \(count);
                        bool b_fix[QKR];
                        for (int j = 0; j < QKR; ++j) {
                            b_fix[j] = false;
                            const int task = sg + j * NSG;
                            const int s = (task / 2) / TB;
                            const int r = (task / 2) % TB;
                            // Uniform per SIMD group: the whole group stages one row.
                            if (task < NTASK && s < nh && r < sn) {
                                float squares = 0.0f;
                                for (int i = 0; i < 4; ++i) {
                                    const float square = fq[j][i] * fq[j][i];
                                    squares += square;
                                }
                                squares = simd_sum(squares);
                                const float denominator = metal::precise::sqrt(squares + 1e-6f);
                                threadgroup float* row = (task % 2) == 0
                                    ? (threadgroup float*)&q_s[s][r][0]
                                    : (threadgroup float*)&k_s[s][r][0];
                                \(division("fq[j]", fix: "b_fix[j]", query: "(task % 2) == 0"))
                            }
                        }
                        for (int j = 0; j < NG; ++j) {
                            const int p = tid + j * NT;
                            const int s = (p / Q4) / TB;
                            const int r = (p / Q4) % TB;
                            if (p < NQUAD && s < nh && r < sn) {
                                float4 decay;
                                for (int m = 0; m < 4; ++m) {
                                    const float gate_input = fg[j][m] + gate_bias[j][m];
                                    const float x = gate_rate[j] * gate_input;
                                    const float e = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
                                    const float log_decay = fgv[j] ? bound * (x < 0.0f ? e : 1.0f - e) : 0.0f;
                                    decay[m] = metal::precise::exp(log_decay);
                                }
                                g_s[s][r][p % Q4] = decay;
                            }
                        }
                        for (int j = 0; j < NV; ++j) {
                            const int p = tid + j * NT;
                            const int r = p / MR;
                            const int c = p % MR;
                            if (r < sn && c < nrows) { v_s[0][r][c] = fv[j]; }
                        }
                        if (tid < 2 * TB) {
                            const int s = tid / TB;
                            const int r = tid % TB;
                            if (s < nh && r < sn) {
                                const float e = 1.0f / (1.0f + metal::precise::exp(metal::abs(fb)));
                                const float rounded_beta = static_cast<float>(static_cast<T>(fb < 0.0f ? e : 1.0f - e));
                                b_s[s][r] = fbv ? rounded_beta : 0.0f;
                            }
                        }
                        for (int j = 0; j < QKR; ++j) {
                            if (simd_any(b_fix[j])) {
                                const int task = sg + j * NSG;
                                \(exactRow("fq[j]", row: "(task % 2) == 0 ? (threadgroup float*)&q_s[(task / 2) / TB][(task / 2) % TB][0] : (threadgroup float*)&k_s[(task / 2) / TB][(task / 2) % TB][0]", query: "(task % 2) == 0"))
                            }
                        }
                    }
            """
        }

        // MARK: Steps

        private static func store(_ token: String, value: String) -> String {
            "if (active) { core[(row0 + size_t(t0 + \(token))) * size_t(W) + size_t(r0 + rg)] = static_cast<T>(\(value)); }"
        }

        /// One step at block token `t` (a literal, or "t" in the tail loop),
        /// reading index `index` of the staged arrays (the buffer, or the head
        /// slot). `reducing`: reduce the pending query sums of steps
        /// reducing..+3 after this step's decay and key sums. `keeping`: hold
        /// this step's query sum in yq<keeping>; nil reduces it at once. Rows
        /// past the range run the step on staged values and store nothing.
        private static func step(_ t: String, index: String, reducing: Int?, keeping: Int?)
            -> String
        {
            let pending = reducing.map { reduceFour($0) } ?? ""
            let finish = keeping.map { "yq\($0) = o;" } ?? reduceOne(t)
            let value = index == "slot" ? "v_s[0][\(t)][rg]" : "v_s[\(index)][\(t)][rg]"
            return """

                                { // step \(t)
                                    const threadgroup float4* g4 = &g_s[\(index)][\(t)][0];
                                    const threadgroup float4* k4 = &k_s[\(index)][\(t)][0];
                                    const threadgroup float4* q4 = &q_s[\(index)][\(t)][0];
                                    const float bt = b_s[\(index)][\(t)];
                                    const float vt = \(value);
                                    float4 kf[4];
                                    for (int j = 0; j < 4; ++j) { kf[j] = k4[8 * j + ln]; }
                                    float p = 0.0f;
                                    for (int j = 0; j < 4; ++j) {
                                        st[j] = st[j] * g4[8 * j + ln];
                                        const float4 product = st[j] * kf[j];
                                        p += product.x;
                                        p += product.y;
                                        p += product.z;
                                        p += product.w;
                                    }
                                    \(pending)
                                    p += simd_shuffle_xor(p, ushort(4));
                                    p += simd_shuffle_xor(p, ushort(2));
                                    p += simd_shuffle_xor(p, ushort(1));
                                    const float delta = (vt - p) * bt;
                                    float o = 0.0f;
                                    for (int j = 0; j < 4; ++j) {
                                        const float4 update = kf[j] * delta;
                                        st[j] = st[j] + update;
                                        const float4 product = st[j] * q4[8 * j + ln];
                                        o += product.x;
                                        o += product.y;
                                        o += product.z;
                                        o += product.w;
                                    }
                                    \(finish)
                                }
                """
        }

        /// Query sum of one step -> lane 0 of its row (xor 4, 2, 1).
        private static func reduceOne(_ t: String) -> String {
            """
            {
                                    float yv = o;
                                    yv += simd_shuffle_xor(yv, ushort(4));
                                    yv += simd_shuffle_xor(yv, ushort(2));
                                    yv += simd_shuffle_xor(yv, ushort(1));
                                    if (ln == 0) { \(store(t, value: "yv")) }
                                }
            """
        }

        /// Query sums of steps first..first+3 (yq0..yq3) as one reduce-scatter
        /// with the per-step tree: lanes (b2, b1, 0) end with step
        /// first + 2·b2 + b1, equal to its per-step lane-0 sum (fp addition
        /// commutes), and write it.
        private static func reduceFour(_ first: Int) -> String {
            """

                            {
                                const float ya = (qb2 ? yq2 : yq0) + simd_shuffle_xor(qb2 ? yq0 : yq2, ushort(4));
                                const float yb = (qb2 ? yq3 : yq1) + simd_shuffle_xor(qb2 ? yq1 : yq3, ushort(4));
                                float yv = (qb1 ? yb : ya) + simd_shuffle_xor(qb1 ? ya : yb, ushort(2));
                                yv += simd_shuffle_xor(yv, ushort(1));
                                if ((ln & 1) == 0) { \(store("\(first) + ysel", value: "yv")) }
                            }
            """
        }
    }

    // MARK: - Metal source (standard library only)

    /// Kernel bodies. Template arguments: `T` (fused only, the activation
    /// dtype), `H`, `D` (128), `MR` (row capacity), `NTG` (threadgroups per
    /// batch row), `SLOTS` (staged heads).
    enum Source {
        static func text(fused: Bool, order: Order, block: Int) -> String {
            precondition(block > 0 && block % 4 == 0, "token blocks are whole groups of four")
            var full = ""
            for t in 0 ..< block {
                full += step(
                    "\(t)", fused: fused, order: order,
                    reducing: t % 4 == 0 && t > 0 ? t - 4 : nil, keeping: t % 4)
            }
            full += reduceFour(block - 4, fused: fused)
            let tail =
                "for (int t = 0; t < tt; ++t) "
                + step("t", fused: fused, order: order, reducing: nil, keeping: nil)
            return """
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                constexpr int TB = \(block);
                constexpr int NT = MR * 8;
                constexpr int W = H * D;
                constexpr int Q4 = D / 4;
                constexpr int NQUAD = SLOTS * TB * Q4;
                constexpr int NG = (NQUAD + NT - 1) / NT;
                constexpr int NV = (TB * MR + NT - 1) / NT;
                static_assert(D == 128, "eight lanes of sixteen keys per value row");
                static_assert(MR % 4 == 0 && MR <= 128, "whole SIMD groups, at most two heads");
                static_assert(SLOTS == 1 || SLOTS == 2, "one or two staged heads");
                // Staged block, FP32. q/k/decay rows are lane-major: key
                // i * 8 + l (i < 16, l < 8) sits at 32 * (i / 4) + 4 * l + i % 4,
                // so lane l of a value row reads float4 8 * j + l, j < 4.
                threadgroup float4 q_s[SLOTS][TB][Q4];
                threadgroup float4 k_s[SLOTS][TB][Q4];
                threadgroup float4 g_s[SLOTS][TB][Q4];
                threadgroup float v_s[TB][MR];
                threadgroup float b_s[SLOTS][TB];

                const int tid = int(thread_position_in_threadgroup.x);
                const int tg = int(threadgroup_position_in_grid.x);
                const int b = int(threadgroup_position_in_grid.z);
                const int tokens = token_count;
                const size_t row0 = size_t(b) * size_t(tokens);
                // Range [r0, r1) of the flattened (head, value row) axis; it
                // covers nh (1 or 2) heads from hA.
                const int r0 = (tg * W) / NTG;
                const int r1 = ((tg + 1) * W) / NTG;
                const int nrows = r1 - r0;
                const int hA = r0 / D;
                const int nh = (r1 - 1) / D - hA + 1;
                // Eight lanes per value row; lane ln owns keys i * 8 + ln.
                const int rg = tid / 8;
                const int ln = tid % 8;
                const bool active = rg < nrows;
                const int slot = active ? (r0 + rg) / D - hA : 0;
                const bool qb2 = (ln & 4) != 0;
                const bool qb1 = (ln & 2) != 0;
                const int ysel = (qb2 ? 2 : 0) + (qb1 ? 1 : 0);

                const int grow = active ? r0 + rg : r0;
                const size_t state_base = (size_t(b) * H + size_t(grow / D)) * D * D + size_t(grow % D);
                float4 st[4];
                for (int j = 0; j < 4; ++j) {
                    for (int c = 0; c < 4; ++c) {
                        st[j][c] = state_in[state_base + size_t((4 * j + c) * 8 + ln) * D];
                    }
                }
                \(fused ? fusedSetup : precomputedSetup)

                \(fused ? fusedFetch("0") : precomputedFetch("0"))
                \(fused ? fusedStage("min(TB, tokens)") : precomputedStage("min(TB, tokens)"))
                threadgroup_barrier(mem_flags::mem_threadgroup);

                float yq0 = 0.0f;
                float yq1 = 0.0f;
                float yq2 = 0.0f;
                float yq3 = 0.0f;
                for (int t0 = 0; t0 < tokens; t0 += TB) {
                    const int tt = min(TB, tokens - t0);
                    const bool more = t0 + TB < tokens;
                    if (more) \(fused ? fusedFetch("t0 + TB") : precomputedFetch("t0 + TB"))
                    if (active) {
                        if (tt == TB) {
                \(full)
                        } else {
                \(tail)
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (more) {
                        \(fused ? fusedStage("min(TB, tokens - t0 - TB)") : precomputedStage("min(TB, tokens - t0 - TB)"))
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }
                }

                if (active) {
                    for (int j = 0; j < 4; ++j) {
                        for (int c = 0; c < 4; ++c) {
                            state_out[state_base + size_t((4 * j + c) * 8 + ln) * D] = st[j][c];
                        }
                    }
                }
                """
        }

        // Fused inputs: block-invariant gate constants per staged quad, and
        // the registers one block is prefetched into. A q/k row is fetched by
        // one SIMD group (lane holds channels lane + 32 i, the long
        // recurrence's norm order); a gate quad holds channels
        // 32 j + 8 m + l (m < 4), which stage as one float4.
        private static let fusedSetup = """
            constexpr int NSG = NT / 32;
            constexpr int NTASK = 2 * SLOTS * TB;
            const int sg = tid / 32;
            const int lane = tid % 32;
            constexpr int QKR = (NTASK + NSG - 1) / NSG;
            const float bound = params[0];
            const float query_scale = params[1];
            const size_t beta_stride = size_t(beta_layout[0]);
            const size_t beta_column = size_t(beta_layout[1]);
            float gate_bias[NG][4];
            float gate_rate[NG];
            for (int j = 0; j < NG; ++j) {
                const int p = tid + j * NT;
                const int h = hA + min((p / Q4) / TB, nh - 1);
                const int quad = p % Q4;
                gate_rate[j] = metal::precise::exp(static_cast<float>(a_log[h]));
                for (int m = 0; m < 4; ++m) {
                    gate_bias[j][m] = static_cast<float>(dt_bias[h * D + 32 * (quad / 8) + 8 * m + quad % 8]);
                }
            }
            float fq[QKR][4];
            float fg[NG][4];
            bool fgv[NG];
            float fv[NV];
            float fb = 0.0f;
            bool fbv = false;
            for (int j = 0; j < QKR; ++j) {
                for (int i = 0; i < 4; ++i) { fq[j][i] = 0.0f; }
            }
            for (int j = 0; j < NG; ++j) {
                fgv[j] = false;
                for (int m = 0; m < 4; ++m) { fg[j][m] = 0.0f; }
            }
            for (int j = 0; j < NV; ++j) { fv[j] = 0.0f; }
            """

        private static let precomputedSetup = """
            float4 fq4[NG];
            float4 fk4[NG];
            float4 fl4[NG];
            float fv[NV];
            float fb = 0.0f;
            for (int j = 0; j < NG; ++j) {
                fq4[j] = float4(0.0f);
                fk4[j] = float4(0.0f);
                fl4[j] = float4(0.0f);
            }
            for (int j = 0; j < NV; ++j) { fv[j] = 0.0f; }
            """

        /// Global loads of the block starting at token `first` into registers.
        private static func fusedFetch(_ first: String) -> String {
            """
            {
                    const int f0 = \(first);
                    const int fn = min(TB, tokens - f0);
                    for (int j = 0; j < QKR; ++j) {
                        const int task = sg + j * NSG;
                        const int s = (task / 2) / TB;
                        const int r = (task / 2) % TB;
                        if (task < NTASK && s < nh && r < fn) {
                            const size_t base = (row0 + size_t(f0 + r)) * size_t(3 * W)
                                + size_t((task % 2) * W + (hA + s) * D + lane);
                            for (int i = 0; i < 4; ++i) {
                                fq[j][i] = static_cast<float>(activated[base + size_t(32 * i)]);
                            }
                        }
                    }
                    for (int j = 0; j < NG; ++j) {
                        const int p = tid + j * NT;
                        const int s = (p / Q4) / TB;
                        const int r = (p / Q4) % TB;
                        const int quad = p % Q4;
                        if (p < NQUAD && s < nh && r < fn) {
                            const size_t token = row0 + size_t(f0 + r);
                            const size_t base = token * size_t(W)
                                + size_t((hA + s) * D + 32 * (quad / 8) + quad % 8);
                            for (int m = 0; m < 4; ++m) {
                                fg[j][m] = static_cast<float>(forget[base + size_t(8 * m)]);
                            }
                            fgv[j] = validity[token];
                        }
                    }
                    for (int j = 0; j < NV; ++j) {
                        const int p = tid + j * NT;
                        const int r = p / MR;
                        const int c = p % MR;
                        if (r < fn && c < nrows) {
                            fv[j] = static_cast<float>(
                                activated[(row0 + size_t(f0 + r)) * size_t(3 * W) + size_t(2 * W + r0 + c)]);
                        }
                    }
                    if (tid < SLOTS * TB) {
                        const int s = tid / TB;
                        const int r = tid % TB;
                        if (s < nh && r < fn) {
                            const size_t token = row0 + size_t(f0 + r);
                            fb = static_cast<float>(beta_logits[token * beta_stride + beta_column + size_t(hA + s)]);
                            fbv = validity[token];
                        }
                    }
                }
            """
        }

        /// Registers -> threadgroup memory for `count` tokens: q/k L2 norms
        /// (each lane's four squares in channel order, then `simd_sum`, as the
        /// long recurrence), decay exp(bound·sigmoid(rate·(f + dt_bias))) and
        /// beta rounded to the activation dtype; one and zero when invalid.
        private static func fusedStage(_ count: String) -> String {
            """
            {
                    const int sn = \(count);
                    for (int j = 0; j < QKR; ++j) {
                        const int task = sg + j * NSG;
                        const int s = (task / 2) / TB;
                        const int r = (task / 2) % TB;
                        // Uniform per SIMD group: the whole group stages one row.
                        if (task < NTASK && s < nh && r < sn) {
                            float squares = 0.0f;
                            for (int i = 0; i < 4; ++i) {
                                const float square = fq[j][i] * fq[j][i];
                                squares += square;
                            }
                            squares = simd_sum(squares);
                            const float denominator = metal::precise::sqrt(squares + 1e-6f);
                            threadgroup float* row = (task % 2) == 0
                                ? (threadgroup float*)&q_s[s][r][0]
                                : (threadgroup float*)&k_s[s][r][0];
                            for (int i = 0; i < 4; ++i) {
                                const float unit = fq[j][i] / denominator;
                                row[32 * i + 4 * (lane % 8) + lane / 8] = (task % 2) == 0 ? unit * query_scale : unit;
                            }
                        }
                    }
                    for (int j = 0; j < NG; ++j) {
                        const int p = tid + j * NT;
                        const int s = (p / Q4) / TB;
                        const int r = (p / Q4) % TB;
                        if (p < NQUAD && s < nh && r < sn) {
                            float4 decay;
                            for (int m = 0; m < 4; ++m) {
                                const float gate_input = fg[j][m] + gate_bias[j][m];
                                const float x = gate_rate[j] * gate_input;
                                const float e = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
                                const float log_decay = fgv[j] ? bound * (x < 0.0f ? e : 1.0f - e) : 0.0f;
                                decay[m] = metal::precise::exp(log_decay);
                            }
                            g_s[s][r][p % Q4] = decay;
                        }
                    }
                    for (int j = 0; j < NV; ++j) {
                        const int p = tid + j * NT;
                        const int r = p / MR;
                        const int c = p % MR;
                        if (r < sn && c < nrows) { v_s[r][c] = fv[j]; }
                    }
                    if (tid < SLOTS * TB) {
                        const int s = tid / TB;
                        const int r = tid % TB;
                        if (s < nh && r < sn) {
                            const float e = 1.0f / (1.0f + metal::precise::exp(metal::abs(fb)));
                            const float rounded_beta = static_cast<float>(static_cast<T>(fb < 0.0f ? e : 1.0f - e));
                            b_s[s][r] = fbv ? rounded_beta : 0.0f;
                        }
                    }
                }
            """
        }

        private static func precomputedFetch(_ first: String) -> String {
            """
            {
                    const int f0 = \(first);
                    const int fn = min(TB, tokens - f0);
                    for (int j = 0; j < NG; ++j) {
                        const int p = tid + j * NT;
                        const int s = (p / Q4) / TB;
                        const int r = (p / Q4) % TB;
                        const int quad = p % Q4;
                        if (p < NQUAD && s < nh && r < fn) {
                            const size_t base = (row0 + size_t(f0 + r)) * size_t(W)
                                + size_t((hA + s) * D + 32 * (quad / 8) + quad % 8);
                            for (int m = 0; m < 4; ++m) {
                                fq4[j][m] = query[base + size_t(8 * m)];
                                fk4[j][m] = key[base + size_t(8 * m)];
                                fl4[j][m] = log_decay[base + size_t(8 * m)];
                            }
                        }
                    }
                    for (int j = 0; j < NV; ++j) {
                        const int p = tid + j * NT;
                        const int r = p / MR;
                        const int c = p % MR;
                        if (r < fn && c < nrows) {
                            fv[j] = value[(row0 + size_t(f0 + r)) * size_t(W) + size_t(r0 + c)];
                        }
                    }
                    if (tid < SLOTS * TB) {
                        const int s = tid / TB;
                        const int r = tid % TB;
                        if (s < nh && r < fn) {
                            fb = beta[(row0 + size_t(f0 + r)) * size_t(H) + size_t(hA + s)];
                        }
                    }
                }
            """
        }

        /// Decay is exp(log decay) with MLX's precise exponential, as in
        /// `GLM5NextLinearKernel`.
        private static func precomputedStage(_ count: String) -> String {
            """
            {
                    const int sn = \(count);
                    for (int j = 0; j < NG; ++j) {
                        const int p = tid + j * NT;
                        const int s = (p / Q4) / TB;
                        const int r = (p / Q4) % TB;
                        if (p < NQUAD && s < nh && r < sn) {
                            q_s[s][r][p % Q4] = fq4[j];
                            k_s[s][r][p % Q4] = fk4[j];
                            float4 decay;
                            for (int m = 0; m < 4; ++m) { decay[m] = metal::precise::exp(fl4[j][m]); }
                            g_s[s][r][p % Q4] = decay;
                        }
                    }
                    for (int j = 0; j < NV; ++j) {
                        const int p = tid + j * NT;
                        const int r = p / MR;
                        const int c = p % MR;
                        if (r < sn && c < nrows) { v_s[r][c] = fv[j]; }
                    }
                    if (tid < SLOTS * TB) {
                        const int s = tid / TB;
                        const int r = tid % TB;
                        if (s < nh && r < sn) { b_s[s][r] = fb; }
                    }
                }
            """
        }

        private static func store(_ fused: Bool, token: String, value: String) -> String {
            let index = "(row0 + size_t(t0 + \(token))) * size_t(W) + size_t(r0 + rg)"
            return fused
                ? "core[\(index)] = static_cast<T>(\(value));"
                : "y[\(index)] = \(value);"
        }

        /// One step at block token `t` (a literal, or "t" in the tail loop).
        /// `reducing`: reduce the pending query sums of steps reducing..+3
        /// after this step's decay and key sums. `keeping`: hold this step's
        /// query sum in yq<keeping>; nil reduces it at once.
        private static func step(
            _ t: String, fused: Bool, order: Order, reducing: Int?, keeping: Int?
        ) -> String {
            let keySum: String
            let querySum: String
            switch order {
            case .serving:
                keySum = """
                    float p = 0.0f;
                                for (int j = 0; j < 4; ++j) {
                                    st[j] = st[j] * g4[8 * j + ln];
                                    const float4 product = st[j] * kf[j];
                                    p += product.x;
                                    p += product.y;
                                    p += product.z;
                                    p += product.w;
                                }
                    """
                querySum = """
                    float o = 0.0f;
                                for (int j = 0; j < 4; ++j) {
                                    const float4 update = kf[j] * delta;
                                    st[j] = st[j] + update;
                                    const float4 product = st[j] * q4[8 * j + ln];
                                    o += product.x;
                                    o += product.y;
                                    o += product.z;
                                    o += product.w;
                                }
                    """
            case .paired:
                keySum = """
                    float4 p4 = float4(0.0f);
                                for (int j = 0; j < 4; ++j) {
                                    st[j] = st[j] * g4[8 * j + ln];
                                    const float4 product = st[j] * kf[j];
                                    p4 = p4 + product;
                                }
                                float p = (p4.x + p4.y) + (p4.z + p4.w);
                    """
                querySum = """
                    float4 o4 = float4(0.0f);
                                for (int j = 0; j < 4; ++j) {
                                    const float4 update = kf[j] * delta;
                                    st[j] = st[j] + update;
                                    const float4 product = st[j] * q4[8 * j + ln];
                                    o4 = o4 + product;
                                }
                                const float o = (o4.x + o4.y) + (o4.z + o4.w);
                    """
            }
            let pending = reducing.map { reduceFour($0, fused: fused) } ?? ""
            let finish = keeping.map { "yq\($0) = o;" } ?? reduceOne(t, fused: fused)
            return """

                        { // step \(t)
                            const threadgroup float4* g4 = &g_s[slot][\(t)][0];
                            const threadgroup float4* k4 = &k_s[slot][\(t)][0];
                            const threadgroup float4* q4 = &q_s[slot][\(t)][0];
                            const float bt = b_s[slot][\(t)];
                            const float vt = v_s[\(t)][rg];
                            float4 kf[4];
                            for (int j = 0; j < 4; ++j) { kf[j] = k4[8 * j + ln]; }
                            \(keySum)
                            \(pending)
                            p += simd_shuffle_xor(p, ushort(4));
                            p += simd_shuffle_xor(p, ushort(2));
                            p += simd_shuffle_xor(p, ushort(1));
                            const float delta = (vt - p) * bt;
                            \(querySum)
                            \(finish)
                        }
                """
        }

        /// Query sum of one step -> lane 0 of its row (xor 4, 2, 1).
        private static func reduceOne(_ t: String, fused: Bool) -> String {
            """
            {
                                float yv = o;
                                yv += simd_shuffle_xor(yv, ushort(4));
                                yv += simd_shuffle_xor(yv, ushort(2));
                                yv += simd_shuffle_xor(yv, ushort(1));
                                if (ln == 0) { \(store(fused, token: t, value: "yv")) }
                            }
            """
        }

        /// Query sums of steps first..first+3 (yq0..yq3) as one reduce-scatter
        /// with the per-step tree: lanes (b2, b1, 0) end with step
        /// first + 2·b2 + b1, equal to its per-step lane-0 sum (fp addition
        /// commutes), and write it.
        private static func reduceFour(_ first: Int, fused: Bool) -> String {
            """

                        {
                            const float ya = (qb2 ? yq2 : yq0) + simd_shuffle_xor(qb2 ? yq0 : yq2, ushort(4));
                            const float yb = (qb2 ? yq3 : yq1) + simd_shuffle_xor(qb2 ? yq1 : yq3, ushort(4));
                            float yv = (qb1 ? yb : ya) + simd_shuffle_xor(qb1 ? ya : yb, ushort(2));
                            yv += simd_shuffle_xor(yv, ushort(1));
                            if ((ln & 1) == 0) { \(store(fused, token: "\(first) + ysel", value: "yv")) }
                        }
            """
        }
    }
}
