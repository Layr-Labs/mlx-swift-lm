// RaggedTwoPassDecodeAttentionV1.swift
//
// TRIMMED FOR PR 1 (base): the batch-8 ragged / D512 two-pass decode ROAD is
// gone — it was unreachable at the owner's single-prompt B=1 shape
// (`CBv2AttentionV1.updateAndAttend` returns through the per-row path before any
// of it) and every call site was collapsed to its stock fallback.
//
// What remains is ONLY the machinery RING-READ-FOLD-B1 reuses at B=1: the
// two-pass configuration (partition count, combine packing), the pass-B combine
// kernels (`passBKernel` / `passBFoldKernel` via `passBActive`), and the
// one-row ring attend (`attendRingB1` / `ringB1Eligible` / `ringPassAB1Kernel`).
// These are byte-identical to the measured serial-levers tree; `attendRingB1`
// is bit-for-bit the B=8 `attendRing` for `batch_index == 0`. See
// `CBv2RingReadFoldB1` for the B=1 call site and the near-tie exactness note.
//
// The pass-B combine (COMBINE-PACK-001 / COMBINE-HOIST-001) is adapted from
// samfenwick's public Yukon submission 0ca873cb-d1b3-43e0-b75d-88d49c206812.
//
// `DARKBLOOM_CBV2_RAGGED_TWO_PASS_ATTENTION` survives as the kernel-family
// co-switch (default ON); the fold's primary kill switch is
// `DARKBLOOM_CBV2_RING_READ_FOLD_B1` (see `CBv2RingReadFoldB1`).

import Foundation
import MLX
import MLXFast

public enum CBv2RaggedTwoPassDecodeAttentionV1 {
    private static let enabled: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_RAGGED_TWO_PASS_ATTENTION"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()


    private static let queryHeads = 16
    private static let kvHeads = 8
    private static let gqa = 2
    private static let headDim = 256
    private static let sequenceLength = 1024

    /// PARTITION-001: the stock partition count, sized for ONE row.
    ///
    /// `sdpa_vector_2pass` picks `blocks` from the architecture letter, the key
    /// length and `n_simds = GQA * qL` alone
    /// (`scaled_dot_product_attention.cpp:443-476`). Batch never enters the
    /// choice, because the stock call it was tuned for dispatches one row: at
    /// N=1024/GQA=2 it wants 128 threadgroup columns on a `d` part to keep the
    /// machine busy with `kvHeads * 1 * blocks` threadgroups.
    ///
    /// This dispatch is batch-wide. `kvHeads * batch * blocks` threadgroups is
    /// eight times what the heuristic was solving for, so the stock answer
    /// over-partitions by 8x and pays for it in the pass-A/pass-B scratch
    /// round trip, which is `batch * queryHeads * blocks * D` BF16 written and
    /// read once each per sliding layer.
    private static let stockBlocks: Int = {
        switch MLX.GPU.deviceInfo().architecture.last {
        case "s": return 64
        case "d": return 128
        default: return 32
        }
    }()

    /// PARTITION-002: the partition this dispatch actually uses.
    ///
    /// PARTITION-001 stopped at 32 for a reason that was about the merge
    /// kernel, not about the machine: pass B indexed its columns with one SIMD
    /// lane each and looped `BLOCKS / simd_width` times, so a partition below a
    /// simdgroup silently merged nothing. That loop is now lane-guarded and
    /// admits any partition that divides the ring, which lets the occupancy
    /// argument finish.
    ///
    /// The stock heuristic's answer is an occupancy target expressed in
    /// threadgroups: on a `d` part at N=1024 and `n_simds = 2` it asks for
    /// `kvHeads * 1 * 128 = 1024` of them, because the call it was tuned for
    /// dispatches one row. This dispatch carries eight rows, so the partition
    /// that reaches the same target is `128 / 8 = 16`, and the whole span from
    /// 8 to 16 clears it: at 8 the launch is still 512 threadgroups of two
    /// simdgroups, and each of those simdgroups keeps a 512-byte K load and a
    /// 512-byte V load outstanding, so roughly 1 MB is in flight against the
    /// ~225 KB a 450 GB/s part needs to cover its own DRAM latency.
    ///
    /// Everything below the target is scratch that does not have to be written.
    /// `MLX_SDPA_BLOCKS` keeps its stock meaning and still wins, so a process
    /// can never run mismatched partitions; `DARKBLOOM_CBV2_2PASS_BLOCKS`
    /// restores the stock answer (`=0`) or selects any other divisor of the
    /// ring for bisection.
    private static let blocks: Int = {
        if let raw = ProcessInfo.processInfo.environment["MLX_SDPA_BLOCKS"],
            let value = Int(raw), value > 0
        {
            return value
        }
        if let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_CBV2_2PASS_BLOCKS"], let value = Int(raw)
        {
            return value > 0 && sequenceLength.isMultiple(of: value)
                ? value : stockBlocks
        }
        return min(8, stockBlocks)
    }()

    /// Attribution: COMBINE-PACK-001 and COMBINE-HOIST-001 below are adapted
    /// from samfenwick's public Yukon submission
    /// `0ca873cb-d1b3-43e0-b75d-88d49c206812` (`3dcce32`). That sealed run
    /// passed parity and measured the fastest absolute decode window among the
    /// recent public candidates (2.085229 seconds).
    /// Off-cadence retest: the first stacked run also passed parity and kept
    /// 13.871 ms of absolute decode gain, but missed promotion after its paired
    /// serial prefill control shifted by 25.274 ms versus the record run.
    ///
    /// COMBINE-PACK-001: how many partition columns one simdgroup of the merge
    /// dispatch carries.
    ///
    /// The merge indexes a partition column with a lane, so with the partition
    /// PARTITION-002 settled on it runs eight live lanes and twenty-four dead
    /// ones in every simdgroup of every threadgroup. Packing `32 / COLS`
    /// output groups into the one simdgroup fills those lanes instead. It is
    /// the same reduction over the same columns in the same order, so the
    /// merge is unchanged arithmetically; only the lane a column lands on and
    /// the number of threads launched move.
    ///
    /// At a partition of a simdgroup or wider this is `32`, `sets` is one and
    /// the packing is the incumbent one thread per column, so the stock
    /// partitions and the `DARKBLOOM_CBV2_2PASS_BLOCKS` bisection route keep
    /// the shape they had.
    private static let combineColumns: Int = {
        let capped = min(blocks, 32)
        return capped > 0 && (capped & (capped - 1)) == 0 ? capped : 32
    }()

    /// Output groups per simdgroup. `D / values_per_lane` is always 32, so the
    /// merge needs `32 / combineSets` simdgroups to cover the head.
    private static let combineSets = 32 / combineColumns
    private static let combineThreads = (32 / combineSets) * 32

    private static let passBKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name:
            "cbv2_ragged8_sdpa_2pass_b_direct_bf16_d256_b\(blocks)"
            + "_c\(combineColumns)_vec4_v6",
        inputNames: ["partials", "sums", "maxs"],
        outputNames: ["out"],
        source: """
            typedef vec<T, 4> T4;
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;
            constexpr int vectors_per_lane = values_per_lane / 4;
            static_assert(values_per_lane % 4 == 0, "lane run is four-wide");
            // COMBINE-PACK-001: a lane owns one partition column of one output
            // group, and a simdgroup carries `sets` output groups side by
            // side. COLS is min(BLOCKS, simd_width) rounded to a power of two,
            // so every lane holds a live column and the surplus-lane guard
            // below is the constant true whenever the partition fits a
            // simdgroup. At COLS == simd_width this is the incumbent one
            // output group per simdgroup, one column per lane.
            constexpr int sets = simd_width / COLS;
            constexpr int rounds = (BLOCKS + COLS - 1) / COLS;

            const int batch_head = int(threadgroup_position_in_grid.x);
            const int lane = int(thread_index_in_simdgroup);
            const int block_lane = lane % COLS;
            const int output_group =
                int(simdgroup_index_in_threadgroup) * sets + lane / COLS;

            partials += batch_head * BLOCKS * D
                + output_group * values_per_lane;
            sums += batch_head * BLOCKS;
            maxs += batch_head * BLOCKS;
            out += batch_head * D + output_group * values_per_lane;

            thread float accumulator[values_per_lane];
            for (int element = 0; element < values_per_lane; ++element) {
                accumulator[element] = 0.0f;
            }
            // COMBINE-HOIST-001: a lane's column summaries are invariant
            // across the three passes below, and its rescale factor is
            // invariant across the last two. The incumbent re-read `maxs`
            // three times and `sums` once, and evaluated the same
            // `fast::exp` twice per column. Each is kept in a register
            // instead. `rounds` is a compile-time constant, so these are
            // named registers rather than an indexed stack array.
            thread float lane_max[rounds];
            thread float lane_sum[rounds];
            thread float lane_factor[rounds];
            float sum_exp_score = 0.0f;
            float max_score = -3.402823466e+38F;
            for (int round = 0; round < rounds; ++round) {
                const int column = block_lane + COLS * round;
                const bool live = column < BLOCKS;
                lane_max[round] = live ? maxs[column] : -3.402823466e+38F;
                lane_sum[round] = live ? sums[column] : 0.0f;
                max_score = max(max_score, lane_max[round]);
            }
            // The columns of one output group sit on the contiguous lane run
            // [set * COLS, set * COLS + COLS), so an ascending xor butterfly
            // bounded at COLS never leaves the set. It is the same tree the
            // full-width reduction ran over the live columns, with the rounds
            // that only folded in the identity dropped.
            for (int stride = 1; stride < COLS; stride <<= 1) {
                max_score =
                    max(max_score, simd_shuffle_xor(max_score, ushort(stride)));
            }

            for (int round = 0; round < rounds; ++round) {
                lane_factor[round] = fast::exp(lane_max[round] - max_score);
                sum_exp_score += lane_factor[round] * lane_sum[round];
            }
            for (int stride = 1; stride < COLS; stride <<= 1) {
                sum_exp_score += simd_shuffle_xor(sum_exp_score, ushort(stride));
            }

            // A lane's run of the column is contiguous, so it is read as
            // four-wide vectors of the same element type. Each component is
            // widened where it is multiplied, so every product and every
            // accumulator update is the one the element walk performed.
            for (int round = 0; round < rounds; ++round) {
                const int column = block_lane + COLS * round;
                if (column < BLOCKS) {
                    const float factor = lane_factor[round];
                    const device T4* partial_vectors =
                        reinterpret_cast<const device T4*>(
                            partials + column * D);
                    #pragma clang loop unroll(full)
                    for (int chunk = 0; chunk < vectors_per_lane; ++chunk) {
                        const T4 partial_vector = partial_vectors[chunk];
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < 4; ++j) {
                            accumulator[chunk * 4 + j] +=
                                factor * float(partial_vector[j]);
                        }
                    }
                }
            }

            for (int element = 0; element < values_per_lane; ++element) {
                float reduced = accumulator[element];
                for (int stride = 1; stride < COLS; stride <<= 1) {
                    reduced += simd_shuffle_xor(reduced, ushort(stride));
                }
                if (block_lane == 0) {
                    out[element] = T(
                        sum_exp_score == 0.0f
                            ? reduced
                            : reduced / sum_exp_score);
                }
            }
        """,
        ensureRowContiguous: true
    )

    private static let passBFoldKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name:
            "cbv2_ragged8_sdpa_2pass_b_direct_bf16_d256_b\(blocks)"
            + "_c\(combineColumns)_vec4_xfold_v1",
        inputNames: ["partials", "sums", "maxs"],
        outputNames: ["out"],
        source: """
            typedef vec<T, 4> T4;
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;
            constexpr int vectors_per_lane = values_per_lane / 4;
            static_assert(values_per_lane % 4 == 0, "lane run is four-wide");
            // COMBINE-PACK-001: a lane owns one partition column of one output
            // group, and a simdgroup carries `sets` output groups side by
            // side. COLS is min(BLOCKS, simd_width) rounded to a power of two,
            // so every lane holds a live column and the surplus-lane guard
            // below is the constant true whenever the partition fits a
            // simdgroup. At COLS == simd_width this is the incumbent one
            // output group per simdgroup, one column per lane.
            constexpr int sets = simd_width / COLS;
            constexpr int rounds = (BLOCKS + COLS - 1) / COLS;

            const int batch_head = int(threadgroup_position_in_grid.x);
            const int lane = int(thread_index_in_simdgroup);
            const int block_lane = lane % COLS;
            const int output_group =
                int(simdgroup_index_in_threadgroup) * sets + lane / COLS;

            partials += batch_head * BLOCKS * D
                + output_group * values_per_lane;
            sums += batch_head * BLOCKS;
            maxs += batch_head * BLOCKS;
            out += batch_head * D + output_group * values_per_lane;

            thread float accumulator[values_per_lane];
            for (int element = 0; element < values_per_lane; ++element) {
                accumulator[element] = 0.0f;
            }
            // COMBINE-HOIST-001: a lane's column summaries are invariant
            // across the three passes below, and its rescale factor is
            // invariant across the last two. The incumbent re-read `maxs`
            // three times and `sums` once, and evaluated the same
            // `fast::exp` twice per column. Each is kept in a register
            // instead. `rounds` is a compile-time constant, so these are
            // named registers rather than an indexed stack array.
            thread float lane_max[rounds];
            thread float lane_sum[rounds];
            thread float lane_factor[rounds];
            float sum_exp_score = 0.0f;
            float max_score = -3.402823466e+38F;
            for (int round = 0; round < rounds; ++round) {
                const int column = block_lane + COLS * round;
                const bool live = column < BLOCKS;
                lane_max[round] = live ? maxs[column] : -3.402823466e+38F;
                lane_sum[round] = live ? sums[column] : 0.0f;
                max_score = max(max_score, lane_max[round]);
            }
            // The columns of one output group sit on the contiguous lane run
            // [set * COLS, set * COLS + COLS), so an ascending xor butterfly
            // bounded at COLS never leaves the set. It is the same tree the
            // full-width reduction ran over the live columns, with the rounds
            // that only folded in the identity dropped.
            for (int stride = 1; stride < COLS; stride <<= 1) {
                max_score =
                    max(max_score, simd_shuffle_xor(max_score, ushort(stride)));
            }

            for (int round = 0; round < rounds; ++round) {
                lane_factor[round] = fast::exp(lane_max[round] - max_score);
                sum_exp_score += lane_factor[round] * lane_sum[round];
            }
            for (int stride = 1; stride < COLS; stride <<= 1) {
                sum_exp_score += simd_shuffle_xor(sum_exp_score, ushort(stride));
            }

            // A lane's run of the column is contiguous, so it is read as
            // four-wide vectors of the same element type. Each component is
            // widened where it is multiplied, so every product and every
            // accumulator update is the one the element walk performed.
            for (int round = 0; round < rounds; ++round) {
                const int column = block_lane + COLS * round;
                if (column < BLOCKS) {
                    const float factor = lane_factor[round];
                    const device T4* partial_vectors =
                        reinterpret_cast<const device T4*>(
                            partials + column * D);
                    #pragma clang loop unroll(full)
                    for (int chunk = 0; chunk < vectors_per_lane; ++chunk) {
                        const T4 partial_vector = partial_vectors[chunk];
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < 4; ++j) {
                            accumulator[chunk * 4 + j] +=
                                factor * float(partial_vector[j]);
                        }
                    }
                }
            }

            // COMBINE-XFOLD-001: fold the lane run with ONE cross-lane
            // butterfly instead of `values_per_lane` independent chains, and
            // let every lane store its own element.
            //
            // The incumbent ran one log2(COLS) shuffle chain per element, so
            // every lane carried every element to the last step, and then the
            // single lane with `block_lane == 0` issued the whole run of
            // stores while the other COLS-1 lanes sat in the branch shadow.
            //
            // Step k pairs the lanes that differ in bit k of `block_lane`,
            // exactly as chain step k did. Each lane keeps the half of the
            // element set whose low index bit equals its own bit k and sends
            // the half its partner keeps, so `simd_shuffle_xor(upper ? a : b,
            // stride)` returns the partner's evaluation of that select and
            // the partner has the opposite `upper`. After the last step a
            // lane holds, in `fold[j]`, the complete sum for element
            // `j * COLS + block_lane`.
            //
            // Every element's additions therefore still fold lane bit 0
            // first, then bit 1, and so on in the same ascending order, over
            // the same values, so each stored element is bit for bit the one
            // the chains produced. `sum_exp_score` is bitwise equal on every
            // lane of the set already: its own butterfly gives each lane the
            // same operands in commuted pairs, and IEEE addition is
            // commutative.
            //
            // Each step is a literal-bound block rather than a loop over a
            // runtime delta, so the array is addressed with compile-time
            // indices and cannot be spilled.
            constexpr bool lane_fold = values_per_lane >= COLS;
            if (lane_fold) {
                float fold[values_per_lane];
                #pragma clang loop unroll(full)
                for (int element = 0; element < values_per_lane; ++element) {
                    fold[element] = accumulator[element];
                }
                if (COLS > 1) {
                    const bool upper = (block_lane & 1) != 0;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < values_per_lane / 2; ++j) {
                        const float a = fold[2 * j];
                        const float b = fold[2 * j + 1];
                        fold[j] = (upper ? b : a)
                            + simd_shuffle_xor(upper ? a : b, ushort(1));
                    }
                }
                if (COLS > 2) {
                    const bool upper = (block_lane & 2) != 0;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < values_per_lane / 4; ++j) {
                        const float a = fold[2 * j];
                        const float b = fold[2 * j + 1];
                        fold[j] = (upper ? b : a)
                            + simd_shuffle_xor(upper ? a : b, ushort(2));
                    }
                }
                if (COLS > 4) {
                    const bool upper = (block_lane & 4) != 0;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < values_per_lane / 8; ++j) {
                        const float a = fold[2 * j];
                        const float b = fold[2 * j + 1];
                        fold[j] = (upper ? b : a)
                            + simd_shuffle_xor(upper ? a : b, ushort(4));
                    }
                }
                if (COLS > 8) {
                    const bool upper = (block_lane & 8) != 0;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < values_per_lane / 16; ++j) {
                        const float a = fold[2 * j];
                        const float b = fold[2 * j + 1];
                        fold[j] = (upper ? b : a)
                            + simd_shuffle_xor(upper ? a : b, ushort(8));
                    }
                }
                if (COLS > 16) {
                    const bool upper = (block_lane & 16) != 0;
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < values_per_lane / 32; ++j) {
                        const float a = fold[2 * j];
                        const float b = fold[2 * j + 1];
                        fold[j] = (upper ? b : a)
                            + simd_shuffle_xor(upper ? a : b, ushort(16));
                    }
                }
                // The run the lane kept is contiguous across the set, so the
                // COLS stores of one output group are one coalesced run
                // instead of a serial run issued by one lane.
                #pragma clang loop unroll(full)
                for (int j = 0; j < values_per_lane / COLS; ++j) {
                    out[j * COLS + block_lane] = T(
                        sum_exp_score == 0.0f
                            ? fold[j]
                            : fold[j] / sum_exp_score);
                }
            } else {
                // A partition wider than the lane run cannot be folded into
                // the lane index; that shape keeps the incumbent chains.
                for (int element = 0; element < values_per_lane; ++element) {
                    float reduced = accumulator[element];
                    for (int stride = 1; stride < COLS; stride <<= 1) {
                        reduced += simd_shuffle_xor(reduced, ushort(stride));
                    }
                    if (block_lane == 0) {
                        out[element] = T(
                            sum_exp_score == 0.0f
                                ? reduced
                                : reduced / sum_exp_score);
                    }
                }
            }
        """,
        ensureRowContiguous: true
    )

    /// COMBINE-XFOLD-001 selector. ON by default;
    /// `DARKBLOOM_GEMMA4_PASSA_COMBFOLD=0` restores the incumbent merge
    /// kernel byte for byte, including its cached pipeline name.
    private static let combineFold: Bool = {
        guard let raw = ProcessInfo.processInfo.environment[
            "DARKBLOOM_GEMMA4_PASSA_COMBFOLD"]
        else { return true }
        return !["0", "false", "no", "off"].contains(raw.lowercased())
    }()

    private static var passBActive: MLXFast.MLXFastKernel {
        if combineFold {
            CBv2EngageMark.once("passb-xfold")
            return passBFoldKernel
        }
        return passBKernel
    }


    /// RING-READ-FOLD-B1: `attendRing` for a SINGLE row (B=1 decode). A
    /// one-buffer transcription of `ringPassAKernel` — same modular temporal
    /// walk (`slot = (start + block + i*BLOCKS) & ring_mask`), same online-
    /// softmax reduction order, same `passBActive` combine — so it is bit-for-bit
    /// the B=8 `attendRing` for `batch_index == 0`. That two-pass is a NEAR-TIE
    /// (not bitwise) match to `MLXFast.scaledDotProductAttention` — MLX's own
    /// decode SDPA the pre-fold B=1 path called over the `temporalOrder` concat —
    /// because the two reduce the key axis in different block partitions, so one
    /// greedy argmax in ~6,000 tokens flips (HumanEval-gated, the D512-two-pass
    /// class). The READ is bitwise identical (unit-tested); only the reduction
    /// order differs. Reads the ring K/V IN PLACE via the modular walk; no window
    /// materialisation. Returns nil (no work) on any gate miss so the caller
    /// keeps the concat+SDPA path.
    static func attendRingB1(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        start: Int, scale: Float, slidingWindowLength: Int
    ) -> MLXArray? {
        guard enabled,
            blocks > 0,
            sequenceLength.isMultiple(of: blocks),
            scale == 1.0,
            slidingWindowLength == sequenceLength,
            queries.dtype == .bfloat16,
            queries.shape == [1, queryHeads, 1, headDim],
            keys.dtype == .bfloat16,
            values.dtype == .bfloat16,
            keys.shape == [1, kvHeads, sequenceLength, headDim],
            values.shape == keys.shape,
            0 <= start, start < sequenceLength
        else { return nil }

        let startArray = MLXArray([UInt32(start)], [1])
        let partialShape = [1, queryHeads, 1, blocks, headDim]
        let summaryShape = [1, queryHeads, 1, blocks]
        let passA = ringPassAB1Kernel(
            [queries, keys, values, startArray],
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("N", sequenceLength),
                ("GQA", gqa),
                ("BLOCKS", blocks),
            ],
            grid: (kvHeads * 32, gqa, blocks),
            threadGroup: (32, gqa, 1),
            outputShapes: [partialShape, summaryShape, summaryShape],
            outputDTypes: [.bfloat16, .float32, .float32]
        )
        return passBActive(
            passA,
            template: [
                ("T", queries.dtype),
                ("D", headDim),
                ("BLOCKS", blocks),
                ("COLS", combineColumns),
            ],
            grid: (queryHeads * combineThreads, 1, 1),
            threadGroup: (combineThreads, 1, 1),
            outputShapes: [[1, queryHeads, 1, headDim]],
            outputDTypes: [.bfloat16]
        )[0]
    }

    /// Whether `attendRingB1` would run for these inputs — every guard except
    /// the runtime `start` range (always valid for a full ring). The caller
    /// checks this BEFORE the ring write so the post-write attend cannot fail.
    static func ringB1Eligible(
        queries: MLXArray, keys: MLXArray, values: MLXArray, slidingWindowLength: Int
    ) -> Bool {
        enabled
            && blocks > 0
            && sequenceLength.isMultiple(of: blocks)
            && slidingWindowLength == sequenceLength
            && queries.dtype == .bfloat16
            && queries.shape == [1, queryHeads, 1, headDim]
            && keys.dtype == .bfloat16
            && values.dtype == .bfloat16
            && keys.shape == [1, kvHeads, sequenceLength, headDim]
            && values.shape == keys.shape
    }

    /// One-row (`batch_index == 0`) transcription of `ringPassAKernel`: a
    /// single K/V buffer, no batch switch. Every other line is identical to the
    /// B=8 kernel, so row 0's arithmetic — the modular walk and the online
    /// softmax — is bit-for-bit the B=8 kernel's.
    private static let ringPassAB1Kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "cbv2_ring_2pass_a_b1_bf16_d256_g\(gqa)_b\(blocks)_v1",
        inputNames: ["queries", "k0", "v0", "starts"],
        outputNames: ["partials", "sums", "maxs"],
        source: """
            constexpr int simd_width = 32;
            constexpr int values_per_lane = D / simd_width;
            static_assert((N & (N - 1)) == 0, "ring length must be a power of two");
            constexpr int ring_mask = N - 1;

            const int kv_head = int(threadgroup_position_in_grid.x);
            const int block = int(threadgroup_position_in_grid.z);
            const int query_head_in_group = int(thread_position_in_threadgroup.y);
            const int query_head = GQA * kv_head + query_head_in_group;
            const int batch_head = query_head;   // batch_index == 0
            const int lane = int(thread_index_in_simdgroup);

            const device T* keys = k0;
            const device T* values = v0;
            const uint start = starts[0];

            const device T* query =
                queries + batch_head * D + lane * values_per_lane;
            keys += kv_head * N * D + lane * values_per_lane;
            values += kv_head * N * D + lane * values_per_lane;
            int slot = int((start + uint(block)) & uint(ring_mask));
            device T* partial = partials
                + batch_head * BLOCKS * D + block * D + lane * values_per_lane;
            device float* sum_out = sums + batch_head * BLOCKS + block;
            device float* max_out = maxs + batch_head * BLOCKS + block;

            thread float q[values_per_lane];
            thread float accumulator[values_per_lane];
            for (int element = 0; element < values_per_lane; ++element) {
                q[element] = 1.0f * float(query[element]);
                accumulator[element] = 0.0f;
            }

            float max_score = -3.402823466e+38F;
            float sum_exp_score = 0.0f;
            for (int token = block; token < N; token += BLOCKS) {
                const device T* k = keys + slot * D;
                const device T* v = values + slot * D;
                float score = 0.0f;
                for (int element = 0; element < values_per_lane; ++element) {
                    score += q[element] * float(k[element]);
                }
                score = simd_sum(score);

                const float new_max = max(max_score, score);
                const float old_factor = fast::exp(max_score - new_max);
                const float score_factor = fast::exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * old_factor + score_factor;
                for (int element = 0; element < values_per_lane; ++element) {
                    accumulator[element] = accumulator[element] * old_factor
                        + score_factor * float(v[element]);
                }

                slot = (slot + BLOCKS) & ring_mask;
            }

            if (lane == 0) {
                sum_out[0] = sum_exp_score;
                max_out[0] = max_score;
            }
            for (int element = 0; element < values_per_lane; ++element) {
                partial[element] = T(accumulator[element]);
            }
        """,
        ensureRowContiguous: true
    )
}
