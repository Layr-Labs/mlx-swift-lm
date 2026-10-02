import Foundation
import MLX
import Testing

@testable import MLXLLM

/// Deterministic values for the gated delta tests. A plain LCG keeps the
/// data the same on each run and does not touch the MLX random state.
struct GDNKernelValues {
    private var state: UInt64

    init(seed: UInt64) {
        state = (seed &* 0x9E37_79B9_7F4A_7C15) | 1
    }

    /// The next value, uniform in [-1, 1).
    mutating func next() -> Float {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Float((state >> 40) & 0xFF_FFFF) / Float(1 << 23) - 1
    }

    /// `count` values, uniform in [low, high).
    mutating func values(_ count: Int, low: Float, high: Float) -> [Float] {
        var result = [Float]()
        result.reserveCapacity(count)
        for _ in 0 ..< count {
            result.append(low + (next() + 1) * 0.5 * (high - low))
        }
        return result
    }

    /// `rows * width` values. Each row of `width` values has L2 norm 1, as
    /// the Qwen models give the GDN keys and queries.
    mutating func unitRows(_ rows: Int, width: Int) -> [Float] {
        var result = values(rows * width, low: -1, high: 1)
        for row in 0 ..< rows {
            var sum: Float = 0
            for i in 0 ..< width { sum += result[row * width + i] * result[row * width + i] }
            let scale = 1 / Swift.max(sum.squareRoot(), 1e-6)
            for i in 0 ..< width { result[row * width + i] *= scale }
        }
        return result
    }
}

/// One small gated delta problem: the inputs as Swift arrays, the same
/// inputs as MLX arrays, and a plain Swift reference of the recurrence.
struct GDNKernelProblem {
    let batch: Int
    let steps: Int
    let keyHeads: Int
    let valueHeads: Int
    let keyDim: Int
    let valueDim: Int

    let q: [Float]
    let k: [Float]
    let v: [Float]
    let a: [Float]
    let b: [Float]
    let aLog: [Float]
    let dtBias: [Float]
    let state: [Float]

    init(
        batch: Int, steps: Int, keyHeads: Int, valueHeads: Int, keyDim: Int, valueDim: Int,
        seed: UInt64
    ) {
        self.batch = batch
        self.steps = steps
        self.keyHeads = keyHeads
        self.valueHeads = valueHeads
        self.keyDim = keyDim
        self.valueDim = valueDim
        var values = GDNKernelValues(seed: seed)
        q = values.unitRows(batch * steps * keyHeads, width: keyDim)
        k = values.unitRows(batch * steps * keyHeads, width: keyDim)
        v = values.values(batch * steps * valueHeads * valueDim, low: -1, high: 1)
        a = values.values(batch * steps * valueHeads, low: -1, high: 1)
        b = values.values(batch * steps * valueHeads, low: -2, high: 2)
        aLog = values.values(valueHeads, low: -1, high: 0.5)
        dtBias = values.values(valueHeads, low: -1, high: 1)
        state = values.values(batch * valueHeads * valueDim * keyDim, low: -0.5, high: 0.5)
    }

    var qArray: MLXArray { MLXArray(q, [batch, steps, keyHeads, keyDim]) }
    var kArray: MLXArray { MLXArray(k, [batch, steps, keyHeads, keyDim]) }
    var vArray: MLXArray { MLXArray(v, [batch, steps, valueHeads, valueDim]) }
    var aArray: MLXArray { MLXArray(a, [batch, steps, valueHeads]) }
    var bArray: MLXArray { MLXArray(b, [batch, steps, valueHeads]) }
    var aLogArray: MLXArray { MLXArray(aLog, [valueHeads]) }
    var dtBiasArray: MLXArray { MLXArray(dtBias, [valueHeads]) }
    var stateArray: MLXArray { MLXArray(state, [batch, valueHeads, valueDim, keyDim]) }

    /// `g = exp(-exp(A_log) * softplus(a + dt_bias))`, shape [B, T, Hv].
    var gates: [Float] {
        (0 ..< a.count).map { index in
            let head = index % valueHeads
            let x = Double(a[index]) + Double(dtBias[head])
            let softplus = log1p(exp(x))
            return Float(exp(-exp(Double(aLog[head])) * softplus))
        }
    }

    /// `beta = sigmoid(b)`, shape [B, T, Hv].
    var betas: [Float] {
        b.map { Float(1 / (1 + exp(-Double($0)))) }
    }

    /// The result of the plain reference recurrence.
    struct Reference {
        /// [B, T, Hv, Dv]
        var y: [Double]
        /// [B, Hv, Dv, Dk]
        var state: [Double]
        /// [B, T, Hv, Dv, Dk]: the state after each position.
        var stack: [Double]
    }

    /// The gated delta recurrence, one value row at a time, in Double.
    ///
    /// For each position: decay the state, read `kv = S k`, take
    /// `delta = (v - kv) * beta`, write `S += delta k^T`, and output
    /// `y = S q`. Value head `h` reads key head `h / (Hv / Hk)`.
    ///
    /// - Parameters:
    ///   - g: [B, T, Hv], or [B, T, Hv, Dk] when `perKeyDecay` is true.
    ///   - beta: [B, T, Hv].
    ///   - initial: [B, Hv, Dv, Dk], or nil for a zero state.
    ///   - keep: false where the mask drops the update of a value row.
    ///   - zeroMaskedOutput: true for the Metal kernel, which writes 0 at a
    ///     masked position. The ops path writes the computed value there.
    func reference(
        g: [Float], beta: [Float], perKeyDecay: Bool = false, initial: [Float]?,
        keep: (Int, Int, Int, Int) -> Bool = { _, _, _, _ in true },
        zeroMaskedOutput: Bool = false
    ) -> Reference {
        let headSize = valueDim * keyDim
        let batchSize = valueHeads * headSize
        var s =
            initial.map { $0.map { Double($0) } }
            ?? [Double](repeating: 0, count: batch * batchSize)
        var y = [Double](repeating: 0, count: batch * steps * valueHeads * valueDim)
        var stack = [Double](repeating: 0, count: batch * steps * batchSize)
        let ratio = valueHeads / keyHeads
        for bi in 0 ..< batch {
            for t in 0 ..< steps {
                for h in 0 ..< valueHeads {
                    let keyBase = ((bi * steps + t) * keyHeads + h / ratio) * keyDim
                    let gateIndex = (bi * steps + t) * valueHeads + h
                    let betaValue = Double(beta[gateIndex])
                    for dv in 0 ..< valueDim {
                        let stateBase = ((bi * valueHeads + h) * valueDim + dv) * keyDim
                        var row = Array(s[stateBase ..< stateBase + keyDim])
                        var kv = 0.0
                        for dk in 0 ..< keyDim {
                            let decay = perKeyDecay ? g[gateIndex * keyDim + dk] : g[gateIndex]
                            row[dk] *= Double(decay)
                            kv += row[dk] * Double(k[keyBase + dk])
                        }
                        let valueIndex = gateIndex * valueDim + dv
                        let delta = (Double(v[valueIndex]) - kv) * betaValue
                        var out = 0.0
                        for dk in 0 ..< keyDim {
                            row[dk] += Double(k[keyBase + dk]) * delta
                            out += row[dk] * Double(q[keyBase + dk])
                        }
                        let kept = keep(bi, t, h, dv)
                        y[valueIndex] = kept || !zeroMaskedOutput ? out : 0
                        if kept {
                            s.replaceSubrange(stateBase ..< stateBase + keyDim, with: row)
                        }
                    }
                }
                let source = bi * batchSize
                let target = (bi * steps + t) * batchSize
                stack.replaceSubrange(
                    target ..< target + batchSize, with: s[source ..< source + batchSize])
            }
        }
        return Reference(y: y, state: s, stack: stack)
    }
}

extension KernelTests {

    /// Tests of the gated delta (GDN) recurrence in `GatedDelta.swift`.
    ///
    /// Each test compares a production path (the ops fallback, the Metal
    /// kernel, the masked kernel, the stacked kernel, the Qwen4 blocked
    /// prefill kernel) with a plain Swift reference of the recurrence on
    /// the same float32 inputs. The keys and queries have unit L2 norm and
    /// the decay is below 1, so the state stays of order 1.
    @Suite(.serialized)
    struct GatedDeltaKernelTests {

        // Tolerance of the comparisons with the Swift reference: float32
        // GPU sums against a Double reference over at most 128 terms per
        // dot product and 64 positions, on values of order 1. The error is
        // near 1e-6.
        static let tolerance: Float = 1e-4

        /// The largest absolute difference between `actual` and `expected`.
        /// Infinity when the element counts differ.
        static func difference(_ actual: MLXArray, _ expected: [Double]) -> Float {
            let values = actual.asType(.float32).asArray(Float.self)
            guard values.count == expected.count else { return .infinity }
            var worst = 0.0
            for (value, reference) in zip(values, expected) {
                worst = Swift.max(worst, Swift.abs(Double(value) - reference))
            }
            return Float(worst)
        }

        static func array(_ values: [Float], _ shape: [Int]) -> MLXArray {
            MLXArray(values, shape)
        }

        // MARK: - Gates

        @Test func computeGatedDeltaGMatchesTheFormulaInFloat32() {
            let problem = GDNKernelProblem(
                batch: 1, steps: 3, keyHeads: 1, valueHeads: 4, keyDim: 32, valueDim: 4,
                seed: 1)
            let g = computeGatedDeltaG(
                problem.aLogArray, problem.aArray, problem.dtBiasArray)
            eval(g)
            #expect(g.shape == [1, 3, 4])
            #expect(g.dtype == .float32)
            // Tolerance: float32 exp and softplus against a Double formula,
            // on values in (0, 1). The error is near 1e-7.
            let expected = problem.gates.map { Double($0) }
            let difference = Self.difference(g, expected)
            #expect(difference <= 1e-5, "g max difference \(difference)")

            // bfloat16 inputs still give a float32 g. A bfloat16 g forces a
            // kernel recompile against the float32 state.
            let gHalf = computeGatedDeltaG(
                problem.aLogArray.asType(.bfloat16), problem.aArray.asType(.bfloat16),
                problem.dtBiasArray.asType(.bfloat16))
            #expect(gHalf.dtype == .float32)
        }

        // MARK: - Ops fallback

        @Test func opsRecurrenceMatchesReferenceWithGroupedHeads() {
            // Hv = 4 and Hk = 2: the ops path repeats each key head twice.
            let problem = GDNKernelProblem(
                batch: 2, steps: 3, keyHeads: 2, valueHeads: 4, keyDim: 8, valueDim: 4,
                seed: 2)
            let g = Self.array(problem.gates, [2, 3, 4])
            let beta = Self.array(problem.betas, [2, 3, 4])

            for initial in [nil, problem.state] as [[Float]?] {
                let (y, state) = gatedDeltaOps(
                    q: problem.qArray, k: problem.kArray, v: problem.vArray, g: g, beta: beta,
                    state: initial.map { Self.array($0, [2, 4, 4, 8]) })
                eval(y, state)
                let expected = problem.reference(
                    g: problem.gates, beta: problem.betas, initial: initial)
                #expect(y.shape == [2, 3, 4, 4])
                #expect(state.shape == [2, 4, 4, 8])
                #expect(y.dtype == .float32)
                let yDifference = Self.difference(y, expected.y)
                let stateDifference = Self.difference(state, expected.state)
                let label = initial == nil ? "zero state" : "given state"
                #expect(yDifference <= Self.tolerance, "\(label) y difference \(yDifference)")
                #expect(
                    stateDifference <= Self.tolerance,
                    "\(label) state difference \(stateDifference)")
            }
        }

        @Test func opsPerKeyDecayMatchesReference() {
            // A g of shape [B, T, Hv, Dk] decays each key column on its own.
            let problem = GDNKernelProblem(
                batch: 1, steps: 4, keyHeads: 1, valueHeads: 2, keyDim: 8, valueDim: 4,
                seed: 3)
            var values = GDNKernelValues(seed: 33)
            let gPerKey = values.values(1 * 4 * 2 * 8, low: 0.5, high: 1)
            let (y, state) = gatedDeltaOps(
                q: problem.qArray, k: problem.kArray, v: problem.vArray,
                g: Self.array(gPerKey, [1, 4, 2, 8]),
                beta: Self.array(problem.betas, [1, 4, 2]),
                state: problem.stateArray)
            eval(y, state)
            let expected = problem.reference(
                g: gPerKey, beta: problem.betas, perKeyDecay: true, initial: problem.state)
            let yDifference = Self.difference(y, expected.y)
            let stateDifference = Self.difference(state, expected.state)
            #expect(yDifference <= Self.tolerance, "y difference \(yDifference)")
            #expect(stateDifference <= Self.tolerance, "state difference \(stateDifference)")
        }

        @Test func opsMasksOfEveryRankHoldStateOnMaskedSteps() {
            let batch = 2
            let steps = 4
            let heads = 2
            let valueDim = 4
            let problem = GDNKernelProblem(
                batch: batch, steps: steps, keyHeads: 1, valueHeads: heads, keyDim: 8,
                valueDim: valueDim, seed: 4)
            let g = Self.array(problem.gates, [batch, steps, heads])
            let beta = Self.array(problem.betas, [batch, steps, heads])

            // Three mask ranks. Per step [B, T] (a 1-D slice per step), per
            // head [B, T, Hv] (2-D) and per value row [B, T, Hv, Dv] (3-D).
            let perStep: (Int, Int, Int, Int) -> Bool = { b, t, _, _ in (b + t) % 3 != 1 }
            let perHead: (Int, Int, Int, Int) -> Bool = { b, t, h, _ in (b + t + h) % 2 == 0 }
            let perRow: (Int, Int, Int, Int) -> Bool = { b, t, h, d in (b + 2 * t + h + d) % 3 != 0
            }

            var stepValues = [Bool]()
            var headValues = [Bool]()
            var rowValues = [Bool]()
            for b in 0 ..< batch {
                for t in 0 ..< steps {
                    stepValues.append(perStep(b, t, 0, 0))
                    for h in 0 ..< heads {
                        headValues.append(perHead(b, t, h, 0))
                        for d in 0 ..< valueDim {
                            rowValues.append(perRow(b, t, h, d))
                        }
                    }
                }
            }
            let cases: [(String, MLXArray, (Int, Int, Int, Int) -> Bool)] = [
                ("per step", MLXArray(stepValues, [batch, steps]), perStep),
                ("per head", MLXArray(headValues, [batch, steps, heads]), perHead),
                ("per row", MLXArray(rowValues, [batch, steps, heads, valueDim]), perRow),
            ]
            for (label, mask, keep) in cases {
                let (y, state) = gatedDeltaOps(
                    q: problem.qArray, k: problem.kArray, v: problem.vArray, g: g, beta: beta,
                    state: problem.stateArray, mask: mask)
                eval(y, state)
                // The ops path computes y at a masked position but does not
                // keep the new state.
                let expected = problem.reference(
                    g: problem.gates, beta: problem.betas, initial: problem.state, keep: keep)
                let yDifference = Self.difference(y, expected.y)
                let stateDifference = Self.difference(state, expected.state)
                #expect(yDifference <= Self.tolerance, "\(label) y difference \(yDifference)")
                #expect(
                    stateDifference <= Self.tolerance,
                    "\(label) state difference \(stateDifference)")
            }
        }

        // MARK: - Metal kernel

        @Test func kernelMatchesReferenceAndOpsPath() {
            // Dk = 32 gives one key value per SIMD lane and Dk = 64 gives two.
            for keyDim in [32, 64] {
                let problem = GDNKernelProblem(
                    batch: 2, steps: 5, keyHeads: 1, valueHeads: 2, keyDim: keyDim,
                    valueDim: 8, seed: UInt64(keyDim))
                let g = Self.array(problem.gates, [2, 5, 2])
                let beta = Self.array(problem.betas, [2, 5, 2])
                let (y, state) = gatedDeltaKernel(
                    q: problem.qArray, k: problem.kArray, v: problem.vArray, g: g, beta: beta,
                    state: problem.stateArray)
                let (opsY, opsState) = gatedDeltaOps(
                    q: problem.qArray, k: problem.kArray, v: problem.vArray, g: g, beta: beta,
                    state: problem.stateArray)
                eval(y, state, opsY, opsState)
                #expect(y.shape == [2, 5, 2, 8])
                #expect(state.shape == [2, 2, 8, keyDim])
                #expect(state.dtype == .float32)

                let expected = problem.reference(
                    g: problem.gates, beta: problem.betas, initial: problem.state)
                let yDifference = Self.difference(y, expected.y)
                let stateDifference = Self.difference(state, expected.state)
                #expect(yDifference <= Self.tolerance, "Dk \(keyDim) y difference \(yDifference)")
                #expect(
                    stateDifference <= Self.tolerance,
                    "Dk \(keyDim) state difference \(stateDifference)")

                // Tolerance: two float32 paths with a different order of sums.
                let opsDifference = SyntheticModel.maxAbsDifference(y, opsY)
                let opsStateDifference = SyntheticModel.maxAbsDifference(state, opsState)
                #expect(opsDifference <= Self.tolerance, "Dk \(keyDim) kernel vs ops y")
                #expect(opsStateDifference <= Self.tolerance, "Dk \(keyDim) kernel vs ops state")
            }
        }

        @Test func maskedUpdateZeroesOutputAndHoldsState() {
            let batch = 2
            let steps = 4
            let problem = GDNKernelProblem(
                batch: batch, steps: steps, keyHeads: 1, valueHeads: 2, keyDim: 32,
                valueDim: 4, seed: 6)
            let keepStep: (Int, Int) -> Bool = { b, t in !(b == 0 && t == 1) && !(b == 1 && t >= 2)
            }
            var maskValues = [Bool]()
            for b in 0 ..< batch {
                for t in 0 ..< steps { maskValues.append(keepStep(b, t)) }
            }
            let mask = MLXArray(maskValues, [batch, steps])

            let (y, state) = gatedDeltaUpdate(
                q: problem.qArray, k: problem.kArray, v: problem.vArray,
                a: problem.aArray, b: problem.bArray,
                aLog: problem.aLogArray, dtBias: problem.dtBiasArray,
                state: problem.stateArray, mask: mask)
            eval(y, state)

            let expected = problem.reference(
                g: problem.gates, beta: problem.betas, initial: problem.state,
                keep: { b, t, _, _ in keepStep(b, t) }, zeroMaskedOutput: true)
            let yDifference = Self.difference(y, expected.y)
            let stateDifference = Self.difference(state, expected.state)
            #expect(yDifference <= Self.tolerance, "masked kernel y difference \(yDifference)")
            #expect(
                stateDifference <= Self.tolerance,
                "masked kernel state difference \(stateDifference)")

            // The kernel writes exactly 0 at a masked position.
            let values = y.asArray(Float.self)
            let rowSize = 2 * 4
            for b in 0 ..< batch {
                for t in 0 ..< steps where !keepStep(b, t) {
                    let start = (b * steps + t) * rowSize
                    let masked = values[start ..< start + rowSize]
                    #expect(masked.allSatisfy { $0 == 0 }, "masked y at b \(b) t \(t)")
                }
            }

            // The ops path keeps the same state on masked steps.
            let (_, opsState) = gatedDeltaOps(
                q: problem.qArray, k: problem.kArray, v: problem.vArray,
                g: computeGatedDeltaG(problem.aLogArray, problem.aArray, problem.dtBiasArray),
                beta: Self.array(problem.betas, [batch, steps, 2]),
                state: problem.stateArray, mask: mask)
            let opsDifference = SyntheticModel.maxAbsDifference(state, opsState)
            #expect(opsDifference <= Self.tolerance, "masked kernel vs ops state \(opsDifference)")
        }

        // MARK: - Public update

        @Test func updateComputesGatesAndPromotesStateToFloat32() {
            let problem = GDNKernelProblem(
                batch: 1, steps: 3, keyHeads: 2, valueHeads: 4, keyDim: 32, valueDim: 8,
                seed: 7)

            // No state: the recurrence starts from a float32 zero state.
            let (y, state) = gatedDeltaUpdate(
                q: problem.qArray, k: problem.kArray, v: problem.vArray,
                a: problem.aArray, b: problem.bArray,
                aLog: problem.aLogArray, dtBias: problem.dtBiasArray)
            eval(y, state)
            #expect(state.dtype == .float32)
            #expect(state.shape == [1, 4, 8, 32])
            let zeroStart = problem.reference(
                g: problem.gates, beta: problem.betas, initial: nil)
            let yDifference = Self.difference(y, zeroStart.y)
            let stateDifference = Self.difference(state, zeroStart.state)
            #expect(yDifference <= Self.tolerance, "zero state y difference \(yDifference)")
            #expect(stateDifference <= Self.tolerance, "zero state difference \(stateDifference)")

            // A float16 state is promoted to float32 before the recurrence.
            let halfState = problem.stateArray.asType(.float16)
            let promoted = halfState.asType(.float32).asArray(Float.self)
            let (yHalf, stateHalf) = gatedDeltaUpdate(
                q: problem.qArray, k: problem.kArray, v: problem.vArray,
                a: problem.aArray, b: problem.bArray,
                aLog: problem.aLogArray, dtBias: problem.dtBiasArray, state: halfState)
            eval(yHalf, stateHalf)
            #expect(stateHalf.dtype == .float32)
            let fromHalf = problem.reference(
                g: problem.gates, beta: problem.betas, initial: promoted)
            let yHalfDifference = Self.difference(yHalf, fromHalf.y)
            let stateHalfDifference = Self.difference(stateHalf, fromHalf.state)
            #expect(yHalfDifference <= Self.tolerance, "float16 state y \(yHalfDifference)")
            #expect(stateHalfDifference <= Self.tolerance, "float16 state \(stateHalfDifference)")
        }

        @Test func chunkedUpdateCarriesStateExactly() {
            let problem = GDNKernelProblem(
                batch: 1, steps: 6, keyHeads: 1, valueHeads: 2, keyDim: 32, valueDim: 8,
                seed: 8)
            let (oneShotY, oneShotState) = gatedDeltaUpdate(
                q: problem.qArray, k: problem.kArray, v: problem.vArray,
                a: problem.aArray, b: problem.bArray,
                aLog: problem.aLogArray, dtBias: problem.dtBiasArray,
                state: problem.stateArray)

            var state: MLXArray? = problem.stateArray
            var outputs = [MLXArray]()
            for range in [0 ..< 2, 2 ..< 3, 3 ..< 6] {
                let (y, next) = gatedDeltaUpdate(
                    q: problem.qArray[0..., range], k: problem.kArray[0..., range],
                    v: problem.vArray[0..., range],
                    a: problem.aArray[0..., range], b: problem.bArray[0..., range],
                    aLog: problem.aLogArray, dtBias: problem.dtBiasArray, state: state)
                outputs.append(y)
                state = next
            }
            let chunkedY = concatenated(outputs, axis: 1)
            let chunkedState = state ?? MLXArray.zeros([1])
            eval(oneShotY, oneShotState, chunkedY, chunkedState)

            // Exact: the same kernel body runs each position on the same
            // inputs, and the float32 state crosses a chunk edge unchanged.
            let yDifference = SyntheticModel.maxAbsDifference(oneShotY, chunkedY)
            let stateDifference = SyntheticModel.maxAbsDifference(oneShotState, chunkedState)
            #expect(yDifference == 0, "chunked y difference \(yDifference)")
            #expect(stateDifference == 0, "chunked state difference \(stateDifference)")

            let expected = problem.reference(
                g: problem.gates, beta: problem.betas, initial: problem.state)
            let referenceDifference = Self.difference(chunkedY, expected.y)
            #expect(referenceDifference <= Self.tolerance, "chunked y vs reference")
        }

        // MARK: - Qwen4 blocked prefill flag

        @Test func blockedFlagOnShortWindowUsesTheStockKernel() {
            // T = 4 is below the blocked kernel minimum of 64, so the flag
            // only selects the fused Qwen4 gates and the stock kernel runs.
            #expect(
                !Qwen4ExpGDNBlockedSeq.shouldDispatch(
                    T: 4, keyHeadDim: 32, valueHeadDim: 8, hasMask: false, environment: [:]))
            let problem = GDNKernelProblem(
                batch: 1, steps: 4, keyHeads: 1, valueHeads: 2, keyDim: 32, valueDim: 8,
                seed: 9)
            let (y, state) = gatedDeltaUpdate(
                q: problem.qArray, k: problem.kArray, v: problem.vArray,
                a: problem.aArray, b: problem.bArray,
                aLog: problem.aLogArray, dtBias: problem.dtBiasArray,
                state: problem.stateArray, useBlockedSeq: true)
            eval(y, state)
            let expected = problem.reference(
                g: problem.gates, beta: problem.betas, initial: problem.state)
            let yDifference = Self.difference(y, expected.y)
            let stateDifference = Self.difference(state, expected.state)
            #expect(yDifference <= Self.tolerance, "short blocked flag y \(yDifference)")
            #expect(
                stateDifference <= Self.tolerance, "short blocked flag state \(stateDifference)")
        }

        @Test func blockedPrefillMatchesReference() {
            // The smallest geometry the blocked kernel accepts: T = 64,
            // Dk = 128, Dv = 32, no mask. Two value heads share one key head.
            #expect(
                Qwen4ExpGDNBlockedSeq.shouldDispatch(
                    T: 64, keyHeadDim: 128, valueHeadDim: 32, hasMask: false, environment: [:]))
            #expect(
                !Qwen4ExpGDNBlockedSeq.shouldDispatch(
                    T: 64, keyHeadDim: 128, valueHeadDim: 32, hasMask: true, environment: [:]))
            let problem = GDNKernelProblem(
                batch: 1, steps: 64, keyHeads: 1, valueHeads: 2, keyDim: 128, valueDim: 32,
                seed: 10)
            let (y, state) = gatedDeltaUpdate(
                q: problem.qArray, k: problem.kArray, v: problem.vArray,
                a: problem.aArray, b: problem.bArray,
                aLog: problem.aLogArray, dtBias: problem.dtBiasArray,
                state: problem.stateArray, useBlockedSeq: true)
            eval(y, state)
            #expect(y.shape == [1, 64, 2, 32])
            #expect(state.shape == [1, 2, 32, 128])
            #expect(state.dtype == .float32)
            let expected = problem.reference(
                g: problem.gates, beta: problem.betas, initial: problem.state)
            let yDifference = Self.difference(y, expected.y)
            let stateDifference = Self.difference(state, expected.state)
            #expect(yDifference <= Self.tolerance, "blocked prefill y \(yDifference)")
            #expect(stateDifference <= Self.tolerance, "blocked prefill state \(stateDifference)")
        }

        // MARK: - Stacked update

        @Test func stackedUpdateReturnsTheStateAfterEveryPosition() throws {
            let problem = GDNKernelProblem(
                batch: 2, steps: 3, keyHeads: 1, valueHeads: 2, keyDim: 32, valueDim: 8,
                seed: 11)
            let halfState = problem.stateArray.asType(.float16)
            let promoted = halfState.asType(.float32).asArray(Float.self)
            let cases: [(String, MLXArray?, [Float]?)] = [
                ("zero state", nil, nil),
                ("float16 state", halfState, promoted),
            ]
            for (label, initial, initialValues) in cases {
                let result = try #require(
                    gatedDeltaUpdateStacked(
                        q: problem.qArray, k: problem.kArray, v: problem.vArray,
                        a: problem.aArray, b: problem.bArray,
                        aLog: problem.aLogArray, dtBias: problem.dtBiasArray,
                        state: initial))
                eval(result.y, result.stateStack)
                #expect(result.y.shape == [2, 3, 2, 8])
                #expect(result.stateStack.shape == [2, 3, 2, 8, 32])
                #expect(result.stateStack.dtype == .float32)
                let expected = problem.reference(
                    g: problem.gates, beta: problem.betas, initial: initialValues)
                let yDifference = Self.difference(result.y, expected.y)
                let stackDifference = Self.difference(result.stateStack, expected.stack)
                #expect(yDifference <= Self.tolerance, "\(label) stacked y \(yDifference)")
                #expect(
                    stackDifference <= Self.tolerance,
                    "\(label) stacked states \(stackDifference)")
            }
        }
    }
}
