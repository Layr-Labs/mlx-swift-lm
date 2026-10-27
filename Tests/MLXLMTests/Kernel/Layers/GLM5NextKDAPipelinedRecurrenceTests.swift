// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// `GLM5NextKDAPrefillKernel`'s pipelined launch (`Launch.pipelined`,
    /// `Pipelined`) against the single-buffer per-core kernel that
    /// `GLM5NextLinearDecodeKernel.longRecurrence` ran before it (even split,
    /// `Launch.perCore`) and against the value split: every bit of the core
    /// outputs and FP32 states, over the rank geometry (32 heads, 80 cores),
    /// other core counts and head counts, chunked prefills that carry the
    /// state, short and tail blocks, invalid tokens, a second batch row, beta
    /// read from fused rows, and bf16/f16/f32 activations. Plus an opt-in
    /// timing report (`GLM53_KDA_PIPELINED_BENCH=1`).
    @Suite(.serialized)
    struct GLM5NextKDAPipelinedRecurrenceTests {
        typealias Kernel = GLM5NextKDAPrefillKernel
        typealias Fused = GLM5NextKDAPrefillKernelTests.Fused

        /// Bit patterns (signed zeros and NaN payloads included).
        static func bitwise(_ a: MLXArray, _ b: MLXArray) -> Bool {
            let bits: DType = a.dtype == .float32 ? .uint32 : .uint16
            return a.dtype == b.dtype && a.shape == b.shape
                && arrayEqual(a.view(dtype: bits), b.view(dtype: bits)).item(Bool.self)
        }

        static func sizes(_ launch: Kernel.Launch) -> [Int] {
            let ranges = launch.ranges ?? []
            return zip(ranges.dropFirst(), ranges).map { Int($0 - $1) }
        }

        // MARK: Ranges (no GPU)

        @Test func pipelinedRanges() {
            // One tensor-parallel rank: 32 heads on an 80-core M5 Ultra.
            let rank = Kernel.Launch.pipelined(heads: 32, cores: 80)
            #expect(rank?.threadgroups == 80 && rank?.capacity == 64 && rank?.slots == 2)
            #expect(
                rank.map(Self.sizes)
                    == Array(repeating: [56, 56, 32, 56, 56], count: 16).flatMap { $0 })
            // 64 heads need ranges of more than 64 rows on 80 cores.
            #expect(Kernel.Launch.pipelined(heads: 64, cores: 80) == nil)
            for (heads, cores) in [
                (32, 80), (32, 64), (32, 40), (31, 80), (33, 80), (16, 40), (40, 80), (3, 7),
                (2, 4), (1, 2),
            ] {
                guard let launch = Kernel.Launch.pipelined(heads: heads, cores: cores),
                    let ranges = launch.ranges
                else { continue }
                #expect(launch.threadgroups <= cores && ranges.count == launch.threadgroups + 1)
                #expect(ranges.first == 0 && ranges.last == Int32(heads * 128))
                for (start, end) in zip(ranges, ranges.dropFirst()) {
                    #expect(end > start && end - start <= 64 && (end - start) % 4 == 0)
                    #expect((Int(end) - 1) / 128 - Int(start) / 128 <= 1)
                }
                #expect(launch.threadgroupBytes <= Kernel.threadgroupMemoryLimit)
            }
            print(
                "GLM5NextKDAPipelined cores=\(String(describing: Kernel.gpuCoreCount))"
                    + " nax=\(glm5NextNAXAvailable)"
                    + " launch(T=4096,H=32)=\(String(describing: Kernel.launch(tokens: 4096, heads: 32, dimension: 128).map(Self.sizes)))"
            )
        }

        // MARK: Bitwise against the single-buffer kernel

        /// Every case: the pipelined launch for 80, 64, 40 and 37 cores (where
        /// one exists) against the even per-core split (the kernel today's
        /// long step runs) and the value split.
        @Test func pipelinedMatchesSingleBufferBitwise() throws {
            let short = [32, 33, 39, 40, 47, 48, 55, 56, 64, 100].map {
                (1, $0, 32, DType.bfloat16, true, false)
            }
            let cases =
                [
                    (1, 4096, 32, DType.bfloat16, false, false),
                    (1, 4093, 32, .bfloat16, true, false),
                    (2, 4096, 32, .bfloat16, true, true),
                    (1, 4096, 32, .float16, true, false),
                    (1, 1000, 33, .bfloat16, true, true),
                    (1, 300, 31, .bfloat16, false, false),
                    (1, 512, 40, .bfloat16, true, false),
                    (1, 37, 3, .float16, true, false),
                    (1, 45, 2, .float32, true, true),
                    (1, 333, 16, .float32, true, false),
                ] + short
            var compared = 0
            try Device.withDefaultDevice(.gpu) {
                for (batch, tokens, heads, dtype, holes, rows) in cases {
                    let input = Fused(
                        batch: batch, tokens: tokens, heads: heads, dtype: dtype,
                        seed: UInt64(tokens * 13 + heads), holes: holes, projectionRows: rows)
                    let reference = input.kernel(Kernel.Launch.perCore(heads: heads, cores: 80)!)
                    let split = input.valueSplit()
                    eval(reference.core, reference.state, split.core, split.state)
                    let label =
                        "B\(batch) T\(tokens) H\(heads) \(dtype) holes=\(holes) rows=\(rows)"
                    #expect(
                        Self.bitwise(reference.core, split.core)
                            && Self.bitwise(reference.state, split.state), "\(label) reference")
                    var launches = 0
                    for cores in [80, 64, 40, 37] {
                        guard let launch = Kernel.Launch.pipelined(heads: heads, cores: cores)
                        else { continue }
                        launches += 1
                        compared += 1
                        let actual = input.kernel(launch)
                        eval(actual.core, actual.state)
                        #expect(
                            Self.bitwise(actual.core, reference.core),
                            "\(label) cores=\(cores) core \(Self.sizes(launch))")
                        #expect(
                            Self.bitwise(actual.state, reference.state),
                            "\(label) cores=\(cores) state")
                    }
                    #expect(launches > 0, "\(label): no pipelined launch")
                    if tokens == 4096 && heads == 32 && batch == 1 && !holes {
                        // Control: the comparison sees a reordered sum (oMLX's order).
                        let paired = input.kernel(
                            Kernel.Launch.perCore(heads: heads, cores: 80)!, order: .paired)
                        #expect(!Self.bitwise(paired.core, reference.core), "control")
                    }
                }
            }
            print(
                "GLM5NextKDAPipelined bitwise: \(compared) pipelined launches over \(cases.count) cases"
            )
        }

        /// Activations that take the exact-division fallback: q/k rows holding
        /// +0, -0, tiny (2^-90) and subnormal values, and rows so large that
        /// the norm's denominator passes 2^40, in single-head (double-buffered)
        /// and boundary (single-buffer) ranges.
        @Test func exactDivisionFallbackBitwise() throws {
            try Device.withDefaultDevice(.gpu) {
                for dtype in [DType.bfloat16, .float16] {
                    let heads = 32
                    let tokens = 200
                    let width = heads * 128
                    let input = Fused(
                        batch: 1, tokens: tokens, heads: heads, dtype: dtype, seed: 31,
                        holes: true)
                    var values = input.activated.asType(.float32).reshaped(-1).asArray(Float.self)
                    let tiny: Float = dtype == .bfloat16 ? 0x1p-90 : 0x1p-20
                    let subnormal: Float = dtype == .bfloat16 ? 0x1p-130 : 0x1p-20
                    let huge: Float = dtype == .bfloat16 ? 0x1p45 : 60000
                    for t in stride(from: 0, to: tokens, by: 3) {
                        for (h, channel, value) in [
                            (t % heads, (t * 7) % 128, Float(0)), ((t + 5) % heads, 3, -Float(0)),
                            ((t + 9) % heads, 77, tiny), ((t + 13) % heads, 127, subnormal),
                        ] {
                            // q (first third) and k (second third) channels
                            values[t * 3 * width + h * 128 + channel] = value
                            values[t * 3 * width + width + h * 128 + (channel + 1) % 128] = value
                        }
                        if t % 12 == 0 {
                            // a whole k row that large: denominator above 2^40 (bf16 only)
                            let h = (t / 12) % heads
                            for c in 0 ..< 128 where dtype == .bfloat16 {
                                values[t * 3 * width + width + h * 128 + c] = huge
                            }
                        }
                    }
                    let special = Fused(
                        replacing: input,
                        activated: MLXArray(values, input.activated.shape).asType(dtype))
                    let reference = special.kernel(
                        Kernel.Launch.perCore(heads: heads, cores: 80)!)
                    let split = special.valueSplit()
                    let pipelined = special.kernel(
                        Kernel.Launch.pipelined(heads: heads, cores: 80)!)
                    eval(reference.core, reference.state, split.core, split.state)
                    eval(pipelined.core, pipelined.state)
                    #expect(
                        Self.bitwise(reference.core, split.core)
                            && Self.bitwise(reference.state, split.state), "\(dtype) reference")
                    #expect(Self.bitwise(pipelined.core, reference.core), "\(dtype) core")
                    #expect(Self.bitwise(pipelined.state, reference.state), "\(dtype) state")
                }
            }
        }

        /// A 4,096-token prefill in chunks (the state carried from chunk to
        /// chunk) through `GLM5NextLinearDecodeKernel.longRecurrence`, which
        /// takes the pipelined launch on this GPU when there is one, against
        /// the single-buffer kernel over the same chunks; 32 heads (each
        /// tensor-parallel rank) with bf16 activations and invalid tokens.
        @Test func chunkedLongRecurrenceMatchesSingleBufferBitwise() throws {
            try Device.withDefaultDevice(.gpu) {
                let heads = 32
                let launch = Kernel.launch(tokens: 4096, heads: heads, dimension: 128)
                print(
                    "GLM5NextKDAPipelined long step launch: \(String(describing: launch.map(Self.sizes)))"
                )
                if glm5NextNAXAvailable, Kernel.gpuCoreCount == 80, Kernel.pipelinedEnabled {
                    #expect(
                        launch?.ranges != nil, "the rank's long step takes the pipelined ranges")
                }
                for (seed, chunks) in [
                    (21, [4096]), (22, [2048, 2048]), (23, [1000, 33, 2048, 1015]),
                    (24, [32, 40, 56, 3968]),
                ] {
                    let tokens = chunks.reduce(0, +)
                    let input = Fused(
                        batch: 1, tokens: tokens, heads: heads, seed: UInt64(seed), holes: true)
                    var oldState = input.state
                    var newState = input.state
                    var start = 0
                    for chunk in chunks {
                        let width = 3 * heads * 128
                        let activated = input.activated[0..., start ..< start + chunk, 0...]
                        let forget = input.forget[0..., start ..< start + chunk, 0..., 0...]
                        let beta = input.betaLogits[0..., start ..< start + chunk, 0...]
                        let validity = input.validity[0..., start ..< start + chunk]
                        let old = Kernel.recurrence(
                            activated: activated, forget: forget, betaLogits: beta,
                            betaLayout: input.betaLayout, aLog: input.aLog, dtBias: input.dtBias,
                            state: oldState, validity: validity,
                            gateLowerBound: GLM5NextKDAPrefillKernelTests.gateLowerBound,
                            queryScale: GLM5NextKDAPrefillKernelTests.queryScale,
                            launch: Kernel.Launch.perCore(heads: heads, cores: 80)!)
                        let shape = [1, chunk, heads, 128]
                        let zeros = MLXArray.zeros(shape, dtype: activated.dtype)
                        let new = GLM5NextLinearDecodeKernel.longRecurrence(
                            .init(
                                query: zeros, key: zeros, value: zeros,
                                gates: .projected(forget: forget, output: forget),
                                betaLogits: beta,
                                queryConvolution: MLXArray.zeros([width / 3, 1, 4]),
                                keyConvolution: MLXArray.zeros([width / 3, 1, 4]),
                                valueConvolution: MLXArray.zeros([width / 3, 1, 4]),
                                aLog: input.aLog, dtBias: input.dtBias,
                                outputNorm: MLXArray.ones([128]), state: nil, validity: validity),
                            activated: activated, forget: forget, recurrent: newState,
                            gateLowerBound: GLM5NextKDAPrefillKernelTests.gateLowerBound,
                            queryScale: GLM5NextKDAPrefillKernelTests.queryScale)
                        eval(old.core, old.state, new.core, new.state)
                        let label = "chunks \(chunks) at \(start)+\(chunk)"
                        #expect(Self.bitwise(new.core, old.core), "\(label) core")
                        #expect(Self.bitwise(new.state, old.state), "\(label) state")
                        (oldState, newState) = (old.state, new.state)
                        start += chunk
                    }
                }
            }
        }

        // MARK: Timing (opt-in)

        /// Median wall time per call (inputs resident, nothing else on the
        /// GPU) of the single-buffer per-core kernel and the pipelined one at
        /// the rank geometry; "per chunk" is ×34 KDA layers.
        @Test(
            .enabled(
                if: ProcessInfo.processInfo.environment["GLM53_KDA_PIPELINED_BENCH"] == "1"))
        func timing() throws {
            try Device.withDefaultDevice(.gpu) {
                func median(_ body: () -> (core: MLXArray, state: MLXArray)) -> Double {
                    for _ in 0 ..< 5 {
                        let warm = body()
                        eval(warm.core, warm.state)
                    }
                    var samples: [Double] = []
                    for _ in 0 ..< 25 {
                        let start = DispatchTime.now().uptimeNanoseconds
                        let result = body()
                        eval(result.core, result.state)
                        samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
                    }
                    return samples.sorted()[samples.count / 2]
                }
                let cores = Kernel.gpuCoreCount ?? 80
                for (tokens, heads) in [(4096, 32), (2048, 32), (8192, 32)] {
                    guard let pipelined = Kernel.Launch.pipelined(heads: heads, cores: cores),
                        let perCore = Kernel.Launch.perCore(heads: heads, cores: cores)
                    else { continue }
                    let input = Fused(batch: 1, tokens: tokens, heads: heads, seed: 3)
                    let floor = median {
                        let z = MLXArray.zeros([1])
                        return (z, z)
                    }
                    var old: [Double] = []
                    var new: [Double] = []
                    for _ in 0 ..< 3 {
                        old.append(median { input.kernel(perCore) })
                        new.append(median { input.kernel(pipelined) })
                    }
                    let (o, n) = (old.min()!, new.min()!)
                    print(
                        String(
                            format:
                                "GLM53_KDA_PIPELINED T%d H%d cores %d: single-buffer %.3f ms, pipelined %.3f ms (sync floor %.3f): -%.3f ms per layer, -%.1f ms per %d-token chunk over 34 layers (x%.3f)",
                            tokens, heads, cores, o, n, floor, o - n, 34 * (o - n), tokens, o / n))
                }
            }
        }
    }
}

extension KernelTests.GLM5NextKDAPrefillKernelTests.Fused {
    /// `base` with other activated q/k/v.
    init(replacing base: Self, activated: MLXArray) {
        (batch, tokens, heads) = (base.batch, base.tokens, base.heads)
        self.activated = activated
        (forget, betaLogits, aLog, dtBias) = (base.forget, base.betaLogits, base.aLog, base.dtBias)
        (state, validity, betaLayout) = (base.state, base.validity, base.betaLayout)
        eval(activated)
    }
}
