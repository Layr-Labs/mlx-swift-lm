// Copyright 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import Foundation
import MLX
import Testing

@testable import MLXLLM

extension KernelTests {
    /// `GLM5NextKDAPrefillKernel` against the existing long recurrence (every
    /// bit, every launch shape) and a literal FP32 recurrence (tolerance), and
    /// an opt-in timing report (`GLM53_KDA_PREFILL_BENCH=1`).
    ///
    /// Requires INTEGRATION.md Diff A for `valueSplitRecurrence`; before it,
    /// call `GLM5NextLinearDecodeKernel.longRecurrence` (unchanged) instead.
    @Suite(.serialized)
    struct GLM5NextKDAPrefillKernelTests {
        typealias Kernel = GLM5NextKDAPrefillKernel
        static let dimension = 128
        static let gateLowerBound: Float = -5
        static let queryScale = 1 / Foundation.sqrt(Float(128))

        /// The long step's recurrence inputs, as `longStep` passes them.
        struct Fused {
            let batch, tokens, heads: Int
            let activated, forget, betaLogits, aLog, dtBias, state, validity: MLXArray
            let betaLayout: [Int32]

            init(
                batch: Int, tokens: Int, heads: Int, dtype: DType = .bfloat16, seed: UInt64,
                holes: Bool = false, projectionRows: Bool = false
            ) {
                (self.batch, self.tokens, self.heads) = (batch, tokens, heads)
                let width = heads * dimension
                let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 6)
                activated = MLXRandom.normal([batch, tokens, 3 * width], key: keys[0]).asType(dtype)
                forget = MLXRandom.normal([batch, tokens, heads, dimension], key: keys[1])
                    .asType(dtype)
                // rate·(f + dt_bias) mostly near -4: decays ~0.9, long memories.
                aLog = MLXRandom.normal([heads], scale: 0.3, key: keys[2])
                dtBias = MLXRandom.normal([width], loc: -4, key: keys[3]).asType(dtype)
                state = MLXRandom.normal([batch, heads, dimension, dimension], scale: 0.1, key: keys[4])
                if projectionRows {
                    // Beta read in place from wider fused rows: [B·T, N], column c.
                    let stride = 3 * width + 2 * heads + 3
                    let column = 3 * width + heads + 1
                    betaLogits = (MLXRandom.normal([batch * tokens, stride], key: keys[5]) * 2)
                        .asType(dtype)
                    betaLayout = [Int32(stride), Int32(column)]
                } else {
                    betaLogits = (MLXRandom.normal([batch, tokens, heads], key: keys[5]) * 2)
                        .asType(dtype)
                    betaLayout = [Int32(heads), 0]
                }
                let valid = (0 ..< batch * tokens).map { index in
                    let (row, t) = index.quotientAndRemainder(dividingBy: tokens)
                    return !holes || !((row == 1 && t < 3) || t % 97 == 5)
                }
                validity = MLXArray(valid, [batch, tokens])
                eval(activated, forget, betaLogits, aLog, dtBias, state, validity)
            }

            func kernel(_ launch: Kernel.Launch, order: Kernel.Order = .serving)
                -> (core: MLXArray, state: MLXArray)
            {
                Kernel.recurrence(
                    activated: activated, forget: forget, betaLogits: betaLogits,
                    betaLayout: betaLayout, aLog: aLog, dtBias: dtBias, state: state,
                    validity: validity, gateLowerBound: gateLowerBound, queryScale: queryScale,
                    launch: launch, order: order)
            }

            func valueSplit() -> (core: MLXArray, state: MLXArray) {
                GLM5NextLinearDecodeKernel.valueSplitRecurrence(
                    activated: activated, forget: forget, betaLogits: betaLogits,
                    betaLayout: betaLayout, aLog: aLog, dtBias: dtBias, state: state,
                    validity: validity, gateLowerBound: gateLowerBound, queryScale: queryScale)
            }

            /// `GLM5NextLinearAttention.apply`'s FP32 recurrence inputs, by its
            /// own expressions (norms, gate, beta rounding, validity).
            func precomputed() -> (q: MLXArray, k: MLXArray, v: MLXArray, logDecay: MLXArray, beta: MLXArray) {
                let width = heads * dimension
                let parts = split(activated, indices: [width, 2 * width], axis: -1)
                    .map { $0.reshaped(batch, tokens, heads, dimension).asType(.float32) }
                let q =
                    parts[0] / sqrt(parts[0].square().sum(axis: -1, keepDims: true) + 1e-6)
                    * (1 / Foundation.sqrt(Float(dimension)))
                let k = parts[1] / sqrt(parts[1].square().sum(axis: -1, keepDims: true) + 1e-6)
                let gateInput =
                    forget.asType(.float32)
                    + dtBias.asType(.float32).reshaped(1, 1, heads, dimension)
                let rate = exp(aLog.asType(.float32)).reshaped(1, 1, heads, 1)
                let logits =
                    betaLayout[1] == 0 && Int(betaLayout[0]) == heads
                    ? betaLogits
                    : betaLogits[0..., Int(betaLayout[1]) ..< Int(betaLayout[1]) + heads]
                        .reshaped(batch, tokens, heads)
                let beta = GLM5NextLinearAttention.betaGate(logits).asType(.float32)
                let mask = validity.reshaped(batch, tokens, 1, 1)
                return (
                    q, k, parts[2],
                    MLX.where(mask, gateLowerBound * sigmoid(rate * gateInput), 0),
                    MLX.where(validity.reshaped(batch, tokens, 1), beta, 0)
                )
            }
        }

        /// The `.reference` recurrence of `GLM5NextLinearAttention.apply`,
        /// evaluated every 128 tokens to bound the graph.
        static func literal(
            q: MLXArray, k: MLXArray, v: MLXArray, logDecay: MLXArray, beta: MLXArray,
            state: MLXArray
        ) -> (values: MLXArray, state: MLXArray) {
            var current = state
            var outputs: [MLXArray] = []
            for position in 0 ..< q.dim(1) {
                let qt = q[0..., position, 0..., 0...]
                let kt = k[0..., position, 0..., 0...]
                let vt = v[0..., position, 0..., 0...]
                current = current * exp(logDecay[0..., position, 0..., 0...])[.ellipsis, .newAxis]
                let memory = (current * kt[.ellipsis, .newAxis]).sum(axis: -2)
                let delta = (vt - memory) * beta[0..., position, 0...][.ellipsis, .newAxis]
                current = current + kt[.ellipsis, .newAxis] * delta[0..., 0..., .newAxis, 0...]
                outputs.append((current * qt[.ellipsis, .newAxis]).sum(axis: -2))
                if position % 128 == 127 { eval([current] + outputs.suffix(128)) }
            }
            return (stacked(outputs, axis: 1), current)
        }

        static func maximumError(_ actual: MLXArray, _ expected: MLXArray) -> Float {
            abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self)
        }

        /// max |actual − expected| / max |expected|.
        static func relativeError(_ actual: MLXArray, _ expected: MLXArray) -> Float {
            maximumError(actual, expected)
                / max(abs(expected.asType(.float32)).max().item(Float.self), 1e-6)
        }

        static func finite(_ array: MLXArray) -> Bool {
            isFinite(array.asType(.float32)).all().item(Bool.self)
        }

        // MARK: Launch geometry (no GPU)

        @Test func launchGeometry() {
            let tp = Kernel.Launch.perCore(heads: 32, cores: 80)
            #expect(tp == Kernel.Launch(threadgroups: 80, capacity: 64, slots: 2, block: 8))
            #expect(
                Kernel.Launch.perCore(heads: 64, cores: 80)
                    == Kernel.Launch(threadgroups: 80, capacity: 128, slots: 2, block: 8))
            // 64 heads on 40 cores would need 205 rows per core: blocked instead.
            #expect(Kernel.Launch.perCore(heads: 64, cores: 40) == nil)
            #expect(
                Kernel.Launch.blocked(heads: 32)
                    == Kernel.Launch(threadgroups: 64, capacity: 64, slots: 1, block: 16))
            for launch in [
                tp, Kernel.Launch.perCore(heads: 64, cores: 80), Kernel.Launch.blocked(heads: 64),
            ] {
                let bytes = launch?.threadgroupBytes ?? .max
                #expect(bytes <= Kernel.threadgroupMemoryLimit, "\(String(describing: launch)) \(bytes)")
            }
            print(
                "GLM5NextKDAPrefill cores=\(String(describing: Kernel.gpuCoreCount))"
                    + " nax=\(glm5NextNAXAvailable)"
                    + " launch(T=4096,H=32)=\(String(describing: Kernel.launch(tokens: 4096, heads: 32, dimension: 128)))"
                    + " launch(T=4096,H=64)=\(String(describing: Kernel.launch(tokens: 4096, heads: 64, dimension: 128)))"
                    + " launch(T=31,H=32)=\(String(describing: Kernel.launch(tokens: 31, heads: 32, dimension: 128)))")
        }

        // MARK: Bitwise against the existing long recurrence

        /// Every launch shape (ranges crossing head boundaries, one-head
        /// blocks, tails shorter than a block, invalid tokens, a second batch
        /// row, beta read from fused rows) gives the value split's exact bits.
        /// Runs on any Metal GPU: only `launch(...)` is NAX-gated.
        @Test func servingOrderMatchesValueSplitBitwise() throws {
            try Device.withDefaultDevice(.gpu) {
                for (batch, tokens, heads, dtype, holes, rows) in [
                    (1, 4096, 32, DType.bfloat16, false, false),
                    (1, 4096, 64, .bfloat16, false, false),
                    (2, 4093, 32, .bfloat16, true, true),
                    (2, 37, 3, .float16, true, false),
                    (1, 45, 2, .float32, true, true),
                ] {
                    let input = Fused(
                        batch: batch, tokens: tokens, heads: heads, dtype: dtype,
                        seed: UInt64(tokens * 7 + heads), holes: holes, projectionRows: rows)
                    let expected = input.valueSplit()
                    eval(expected.core, expected.state)
                    #expect(Self.finite(expected.core) && Self.finite(expected.state))
                    var launches = [("blocked", Kernel.Launch.blocked(heads: heads)!)]
                    for cores in [80, 64, 40, 37, 7] {
                        if let launch = Kernel.Launch.perCore(heads: heads, cores: cores) {
                            launches.append(("perCore\(cores)", launch))
                        }
                    }
                    for (name, launch) in launches {
                        let actual = input.kernel(launch)
                        eval(actual.core, actual.state)
                        let label = "B\(batch) T\(tokens) H\(heads) \(dtype) \(name)"
                        #expect(actual.core.dtype == dtype && actual.core.shape == expected.core.shape)
                        let coreError = Self.maximumError(actual.core, expected.core)
                        let stateError = Self.maximumError(actual.state, expected.state)
                        #expect(coreError == 0 && stateError == 0, "\(label) core \(coreError) state \(stateError)")
                    }
                }
            }
        }

        // MARK: Tolerance against the literal recurrence

        /// B=1, T=4096, H=32 (one tensor-parallel rank's KDA heads): the fused
        /// kernel (both orders, per-core and blocked) and the FP32 front end
        /// against `.reference`'s per-token FP32 recurrence on the same inputs.
        /// Core outputs are rounded once to bfloat16 (relative half-ulp 2^-9);
        /// everything else differs only in FP32 summation order.
        @Test func matchesLiteralRecurrence() throws {
            try Device.withDefaultDevice(.gpu) {
                let input = Fused(batch: 1, tokens: 4096, heads: 32, seed: 11, holes: true)
                let p = input.precomputed()
                let literal = Self.literal(
                    q: p.q, k: p.k, v: p.v, logDecay: p.logDecay, beta: p.beta, state: input.state)
                eval(literal.values, literal.state)
                let perCore = Kernel.Launch.perCore(heads: 32, cores: Kernel.gpuCoreCount ?? 80)!
                for (name, launch, order) in [
                    ("perCore serving", perCore, Kernel.Order.serving),
                    ("perCore paired", perCore, .paired),
                    ("blocked serving", Kernel.Launch.blocked(heads: 32)!, .serving),
                ] {
                    let actual = input.kernel(launch, order: order)
                    eval(actual.core, actual.state)
                    let coreError = Self.relativeError(actual.core, literal.values)
                    let stateError = Self.relativeError(actual.state, literal.state)
                    print("GLM5NextKDAPrefill \(name): core rel \(coreError) state rel \(stateError)")
                    #expect(coreError <= 8e-3, "\(name) core \(coreError)")
                    #expect(stateError <= 1e-4, "\(name) state \(stateError)")
                }
                let fp32 = Kernel.recurrence(
                    query: p.q, key: p.k, value: p.v, logDecay: p.logDecay, beta: p.beta,
                    state: input.state, launch: perCore)
                eval(fp32.values, fp32.state)
                let valueError = Self.relativeError(fp32.values, literal.values)
                let stateError = Self.relativeError(fp32.state, literal.state)
                print("GLM5NextKDAPrefill fp32 front end: values rel \(valueError) state rel \(stateError)")
                #expect(valueError <= 1e-4 && stateError <= 1e-4)
            }
        }

        // MARK: Timing (opt-in)

        /// Median wall time per call, inputs resident, nothing else on the GPU.
        @Test(.enabled(if: ProcessInfo.processInfo.environment["GLM53_KDA_PREFILL_BENCH"] == "1"))
        func timing() throws {
            try Device.withDefaultDevice(.gpu) {
                func median(_ body: () -> (core: MLXArray, state: MLXArray)) -> Double {
                    for _ in 0 ..< 3 {
                        let warm = body()
                        eval(warm.core, warm.state)
                    }
                    var samples: [Double] = []
                    for _ in 0 ..< 20 {
                        let start = DispatchTime.now().uptimeNanoseconds
                        let result = body()
                        eval(result.core, result.state)
                        samples.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
                    }
                    return samples.sorted()[samples.count / 2]
                }
                let cores = Kernel.gpuCoreCount ?? 80
                for (tokens, heads) in [(4096, 32), (2048, 32), (4096, 64)] {
                    let input = Fused(batch: 1, tokens: tokens, heads: heads, seed: 3)
                    let old = median { input.valueSplit() }
                    var line = "GLM5NextKDAPrefill T\(tokens) H\(heads): valueSplit \(String(format: "%.3f", old)) ms"
                    var variants: [(String, Kernel.Launch, Kernel.Order)] = [
                        ("blocked", Kernel.Launch.blocked(heads: heads)!, .serving)
                    ]
                    if let perCore = Kernel.Launch.perCore(heads: heads, cores: cores) {
                        variants += [("perCore", perCore, .serving), ("perCore paired", perCore, .paired)]
                    }
                    for (name, launch, order) in variants {
                        let time = median { input.kernel(launch, order: order) }
                        line += String(
                            format: " | %@ %.3f ms (x%.2f, %.1f ms per 34 layers saved)", name, time,
                            old / time, 34 * (old - time))
                    }
                    print(line)
                }
            }
        }
    }
}
