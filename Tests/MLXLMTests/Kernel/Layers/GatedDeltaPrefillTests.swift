import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {

    /// Tests of the GDN prompt-window kernels (`GatedDeltaPrefill`) at the
    /// Qwen3.5-family head geometries.
    ///
    /// `gatedDeltaUpdate` runs a prompt kernel only where it reproduces the
    /// stock `gated_delta_step` bit for bit, so the candidates are compared
    /// with the stock kernel bitwise. A Double reference bounds all of them
    /// on the short windows.
    @Suite(.serialized)
    struct GatedDeltaPrefillTests {

        struct Geometry: CustomStringConvertible, Sendable {
            let name: String
            let batch: Int
            let keyHeads: Int
            let valueHeads: Int

            var description: String {
                "\(name) B=\(batch) Hk=\(keyHeads) Hv=\(valueHeads)"
            }

            func instantiation(_ dtype: DType) -> GatedDeltaPrefill.Geometry {
                .init(
                    dtype: dtype, keyHeads: keyHeads, valueHeads: valueHeads,
                    valueHeadDim: GatedDeltaPrefillTests.valueDim)
            }
        }

        static let keyDim = 128
        static let valueDim = 128

        /// Qwen3.5-9B and Qwen3.5/3.6-35B-A3B have 16 key and 32 value heads;
        /// Qwen3.8-27B has 16 key and 48 value heads; all use Dk = Dv = 128.
        /// The 35B case runs two rows, as a packed prefill rectangle does.
        static let geometries: [Geometry] = [
            Geometry(name: "qwen3.5-9b", batch: 1, keyHeads: 16, valueHeads: 32),
            Geometry(name: "qwen3.8-27b", batch: 1, keyHeads: 16, valueHeads: 48),
            Geometry(name: "qwen3.5-35b-a3b", batch: 2, keyHeads: 16, valueHeads: 32),
        ]

        /// T = 1 and 7 sit below `minT` (the update keeps the stock kernel
        /// there); the candidates still run them here.
        static let steps = [1, 7, 64, 512, 2048]

        struct Inputs {
            let q: MLXArray
            let k: MLXArray
            let v: MLXArray
            let g: MLXArray
            let beta: MLXArray
            let a: MLXArray
            let b: MLXArray
            let aLog: MLXArray
            let dtBias: MLXArray
            let state: MLXArray
        }

        /// Qwen-shaped inputs: RMS-normalized q (scale Dk^-1) and k (scale
        /// Dk^-1/2), gates from the production formula with A in [1, 16) and
        /// dt in [1e-3, 1e-1), and a non-zero float32 state.
        static func inputs(
            _ geometry: Geometry, steps: Int, dtype: DType, seed: UInt64
        ) -> Inputs {
            let key = MLXRandom.key(seed)
            let keys = MLXRandom.split(key: key, into: 8)
            let B = geometry.batch
            let invScale = pow(Float(keyDim), -0.5)
            let qRaw = MLXRandom.normal([B, steps, geometry.keyHeads, keyDim], key: keys[0])
            let kRaw = MLXRandom.normal([B, steps, geometry.keyHeads, keyDim], key: keys[1])
            let q =
                (invScale * invScale)
                * MLXFast.rmsNorm(qRaw, weight: MLXArray.mlxNone, eps: 1e-6)
            let k = invScale * MLXFast.rmsNorm(kRaw, weight: MLXArray.mlxNone, eps: 1e-6)
            let v = MLXRandom.normal([B, steps, geometry.valueHeads, valueDim], key: keys[2])
            let a = MLXRandom.normal([B, steps, geometry.valueHeads], key: keys[3])
            let b = MLXRandom.normal([B, steps, geometry.valueHeads], key: keys[4])
            let aLog = log(
                MLXRandom.uniform(low: 1, high: 16, [geometry.valueHeads], key: keys[5]))
            let dt = exp(
                MLXRandom.uniform(
                    low: log(Float(1e-3)), high: log(Float(1e-1)), [geometry.valueHeads],
                    key: keys[6]))
            let dtBias = dt + log(-expm1(-dt))
            let state =
                MLXRandom.normal(
                    [B, geometry.valueHeads, valueDim, keyDim], key: keys[7]) * Float(0.02)
            let g = computeGatedDeltaG(aLog, a, dtBias)
            let beta = sigmoid(b).asType(.float32)
            let result = Inputs(
                q: q.asType(dtype), k: k.asType(dtype), v: v.asType(dtype), g: g, beta: beta,
                a: a, b: b, aLog: aLog, dtBias: dtBias, state: state)
            eval(
                result.q, result.k, result.v, result.g, result.beta, result.a, result.b,
                result.aLog, result.dtBias, result.state)
            return result
        }

        static func bitsEqual(_ x: MLXArray, _ y: MLXArray) -> Bool {
            GatedDeltaPrefill.bitsEqual(x, y)
        }

        /// Max absolute difference, and that over the reference's largest
        /// magnitude.
        static func difference(_ x: MLXArray, _ reference: MLXArray) -> (abs: Float, rel: Float) {
            let worst = SyntheticModel.maxAbsDifference(x, reference)
            return (
                worst, worst / Swift.max(SyntheticModel.maxAbs(reference), .leastNormalMagnitude)
            )
        }

        static func run(
            _ candidate: GatedDeltaPrefillKernel, _ x: Inputs
        ) throws -> (y: MLXArray, state: MLXArray) {
            let (y, state) = try #require(
                gatedDeltaKernelPrefill(
                    candidate, q: x.q, k: x.k, v: x.v, g: x.g, beta: x.beta, state: x.state))
            eval(y, state)
            return (y, state)
        }

        // MARK: - Parity with the stock kernel

        /// The rows kernel keeps the stock per-row source and `simd_sum`, so
        /// it reproduces the stock kernel on every GPU. The packed kernel does
        /// where `simd_sum` is the ascending butterfly (Apple M5); there the
        /// probe selects it, and its parity must then hold on every shape.
        @Test(arguments: geometries, steps)
        func candidatesMatchStockKernelBitwise(_ geometry: Geometry, steps: Int) throws {
            for dtype in [DType.bfloat16, .float32] {
                let x = Self.inputs(geometry, steps: steps, dtype: dtype, seed: UInt64(steps))
                let (stockY, stockState) = gatedDeltaKernel(
                    q: x.q, k: x.k, v: x.v, g: x.g, beta: x.beta, state: x.state)
                eval(stockY, stockState)
                for candidate in GatedDeltaPrefillKernel.allCases {
                    let (y, state) = try Self.run(candidate, x)
                    #expect(y.shape == [geometry.batch, steps, geometry.valueHeads, Self.valueDim])
                    #expect(y.dtype == dtype)
                    #expect(state.dtype == .float32)
                    let exact = Self.bitsEqual(y, stockY) && Self.bitsEqual(state, stockState)
                    let yDifference = Self.difference(y, stockY)
                    let stateDifference = Self.difference(state, stockState)
                    print(
                        "GDN \(candidate) vs stock \(geometry) T=\(steps) \(dtype): bitwise \(exact); y max abs \(yDifference.abs) rel \(yDifference.rel); state max abs \(stateDifference.abs) rel \(stateDifference.rel)"
                    )
                    let probeSelects =
                        GatedDeltaPrefill.probe(
                            geometry.instantiation(dtype), candidates: [candidate]) == candidate
                    if candidate == .rows || probeSelects {
                        #expect(exact, "\(candidate) \(geometry) T=\(steps) \(dtype)")
                    }
                    // Mathematically the same recurrence either way.
                    #expect(stateDifference.rel <= 1e-5, "\(candidate) state \(stateDifference)")
                }
            }
        }

        /// The probe's answer is deterministic and cached per instantiation.
        @Test(arguments: geometries)
        func probedKernelIsStable(_ geometry: Geometry) {
            for dtype in [DType.bfloat16, .float32] {
                let instantiation = geometry.instantiation(dtype)
                let first = GatedDeltaPrefill.probedKernel(instantiation)
                #expect(GatedDeltaPrefill.probedKernel(instantiation) == first)
                #expect(GatedDeltaPrefill.probe(instantiation) == first)
                print(
                    "GDN probed prefill kernel for \(geometry) \(dtype): \(first?.rawValue ?? "stock")"
                )
            }
        }

        /// The probe and the parity checks compare raw bits. Signed zeros,
        /// which `==` treats as equal, and values one ulp apart do not match;
        /// the same NaN does.
        @Test func bitsEqualComparesRawBits() {
            let values = MLXArray([Float(1), 0, .nan])
            for dtype in [DType.bfloat16, .float32] {
                let x = values.asType(dtype)
                let same = MLXArray([Float(1), 0, .nan]).asType(dtype)
                let negativeZero = MLXArray([Float(1), -0.0, .nan]).asType(dtype)
                #expect(GatedDeltaPrefill.bitsEqual(x, same))
                #expect(!GatedDeltaPrefill.bitsEqual(x, negativeZero))
                #expect(!GatedDeltaPrefill.bitsEqual(x, x.reshaped(3, 1)))
            }
            let oneUlpAbove = MLXArray([Float(1).nextUp, 0, .nan])
            #expect(!GatedDeltaPrefill.bitsEqual(values, oneUlpAbove))
            #expect(!GatedDeltaPrefill.bitsEqual(values, values.asType(.bfloat16)))
        }

        // MARK: - Double reference

        /// Float32 inputs against `GDNKernelProblem`'s Double recurrence. The
        /// bound is the existing GDN kernel tolerance (float32 sums of 128
        /// terms over at most 64 positions on values of order 1).
        @Test(arguments: geometries)
        func candidatesMatchDoubleReference(_ geometry: Geometry) throws {
            for steps in [7, 64] {
                let problem = GDNKernelProblem(
                    batch: geometry.batch, steps: steps, keyHeads: geometry.keyHeads,
                    valueHeads: geometry.valueHeads, keyDim: Self.keyDim,
                    valueDim: Self.valueDim, seed: UInt64(100 + steps))
                let shape = [geometry.batch, steps, geometry.valueHeads]
                let expected = problem.reference(
                    g: problem.gates, beta: problem.betas, initial: problem.state)
                for candidate in GatedDeltaPrefillKernel.allCases {
                    let (y, state) = try #require(
                        gatedDeltaKernelPrefill(
                            candidate, q: problem.qArray, k: problem.kArray, v: problem.vArray,
                            g: MLXArray(problem.gates, shape),
                            beta: MLXArray(problem.betas, shape), state: problem.stateArray))
                    eval(y, state)
                    let yError = Self.errors(y, expected.y)
                    let stateError = Self.errors(state, expected.state)
                    print(
                        "GDN \(candidate) vs Double \(geometry) T=\(steps): y max abs \(yError.abs) rel \(yError.rel); state max abs \(stateError.abs) rel \(stateError.rel)"
                    )
                    #expect(yError.abs <= 1e-4, "\(candidate) \(geometry) T=\(steps) y \(yError)")
                    #expect(
                        stateError.abs <= 1e-4,
                        "\(candidate) \(geometry) T=\(steps) state \(stateError)")
                }
            }
        }

        /// Max absolute error, and max absolute error over the reference's
        /// largest magnitude.
        static func errors(_ actual: MLXArray, _ expected: [Double]) -> (abs: Double, rel: Double) {
            let values = actual.asType(.float32).asArray(Float.self)
            precondition(values.count == expected.count)
            var worst = 0.0
            var scale = 0.0
            for (value, reference) in zip(values, expected) {
                worst = Swift.max(worst, Swift.abs(Double(value) - reference))
                scale = Swift.max(scale, Swift.abs(reference))
            }
            return (worst, worst / Swift.max(scale, .leastNormalMagnitude))
        }

        // MARK: - Public update

        /// `gatedDeltaUpdate` on a prompt window returns exactly the stock
        /// result.
        @Test func updateOnPrefillWindowMatchesStockKernel() {
            let geometry = Self.geometries[0]
            let x = Self.inputs(geometry, steps: 512, dtype: .bfloat16, seed: 512)
            let (y, state) = gatedDeltaUpdate(
                q: x.q, k: x.k, v: x.v, a: x.a, b: x.b, aLog: x.aLog, dtBias: x.dtBias,
                state: x.state)
            let (stockY, stockState) = gatedDeltaKernel(
                q: x.q, k: x.k, v: x.v, g: x.g, beta: x.beta, state: x.state)
            eval(y, state, stockY, stockState)
            #expect(Self.bitsEqual(y, stockY))
            #expect(Self.bitsEqual(state, stockState))
        }

        /// Any chunk partition, including chunks short enough to stay on the
        /// stock kernel, reproduces the one-shot recurrence bit for bit. This
        /// is the property recurrent checkpoints rely on: a state captured at
        /// a chunk boundary and resumed equals the state a cold prefill holds
        /// there, whichever kernel ran each chunk.
        @Test func chunkPartitionsCarryStateBitwise() {
            let geometry = Self.geometries[1]
            let total = 600
            let x = Self.inputs(geometry, steps: total, dtype: .bfloat16, seed: 600)
            let (oneShotY, oneShotState) = gatedDeltaUpdate(
                q: x.q, k: x.k, v: x.v, a: x.a, b: x.b, aLog: x.aLog, dtBias: x.dtBias,
                state: x.state)
            eval(oneShotY, oneShotState)

            let partitions: [[Range<Int>]] = [
                [0 ..< 256, 256 ..< 600],
                [0 ..< 5, 5 ..< 300, 300 ..< 597, 597 ..< 600],
                [0 ..< 1, 1 ..< 17, 17 ..< 600],
            ]
            for partition in partitions {
                var state = x.state
                var outputs = [MLXArray]()
                for range in partition {
                    let (y, next) = gatedDeltaUpdate(
                        q: x.q[0..., range], k: x.k[0..., range], v: x.v[0..., range],
                        a: x.a[0..., range], b: x.b[0..., range], aLog: x.aLog,
                        dtBias: x.dtBias, state: state)
                    outputs.append(y)
                    state = next
                }
                let joined = concatenated(outputs, axis: 1)
                eval(joined, state)
                #expect(Self.bitsEqual(joined, oneShotY), "partition \(partition): y")
                #expect(Self.bitsEqual(state, oneShotState), "partition \(partition): state")
            }
        }

        // MARK: - Dispatch rules

        @Test func dispatchRules() {
            let minT = GatedDeltaPrefill.minT
            let geometry = Self.geometries[0].instantiation(.bfloat16)
            func kernel(
                T: Int, keyHeadDim: Int = 128, valueHeadDim: Int = 128, hasMask: Bool = false,
                mode: GatedDeltaPrefill.Mode = .forced(.rows)
            ) -> GatedDeltaPrefillKernel? {
                GatedDeltaPrefill.kernel(
                    T: T, keyHeadDim: keyHeadDim, hasMask: hasMask,
                    geometry: .init(
                        dtype: geometry.dtype, keyHeads: geometry.keyHeads,
                        valueHeads: geometry.valueHeads, valueHeadDim: valueHeadDim),
                    mode: mode)
            }
            #expect(kernel(T: minT) == .rows)
            #expect(kernel(T: 4096, mode: .forced(.packed)) == .packed)
            #expect(kernel(T: minT, valueHeadDim: 64) == .rows)
            // Decode and the longest MTP verify window.
            #expect(kernel(T: 1) == nil)
            #expect(kernel(T: 1 + CBv2MTPConfig.testedMaxDraftTokens) == nil)
            #expect(kernel(T: minT - 1) == nil)
            #expect(kernel(T: 4096, keyHeadDim: 64) == nil)
            #expect(kernel(T: 4096, valueHeadDim: 126) == nil)
            #expect(kernel(T: 4096, hasMask: true) == nil)
            #expect(kernel(T: 4096, mode: .stock) == nil)
            #expect(kernel(T: 4096, mode: .probe) == GatedDeltaPrefill.probedKernel(geometry))
        }

        // MARK: - Timing (opt-in)

        static let benchmarkEnabled =
            ProcessInfo.processInfo.environment["DARKBLOOM_GDN_PREFILL_BENCH"] == "1"

        /// Per-layer GPU time of each kernel on one prompt window, for
        /// calibrating the selection on other GPUs. Rounds alternate the
        /// kernel order; each sample chains four launches through the state.
        /// `DARKBLOOM_GDN_PREFILL_BENCH=1 swift test --filter GatedDeltaPrefillTests`
        @Test(.enabled(if: benchmarkEnabled))
        func prefillKernelTimings() throws {
            typealias Arm = (name: String, run: (Inputs, MLXArray) -> (MLXArray, MLXArray))
            let arms: [Arm] =
                [
                    (
                        "stock",
                        { x, s in
                            gatedDeltaKernel(q: x.q, k: x.k, v: x.v, g: x.g, beta: x.beta, state: s)
                        }
                    )
                ]
                + GatedDeltaPrefillKernel.allCases.map { candidate in
                    (
                        candidate.rawValue,
                        { x, s in
                            gatedDeltaKernelPrefill(
                                candidate, q: x.q, k: x.k, v: x.v, g: x.g, beta: x.beta, state: s)!
                        }
                    )
                }
            print(
                "| GPU | geometry | T | " + arms.map { "\($0.name) ms" }.joined(separator: " | ")
                    + " |")
            for geometry in Self.geometries.prefix(2) {
                for steps in [2048, 4096] {
                    let x = Self.inputs(geometry, steps: steps, dtype: .bfloat16, seed: 7)
                    var samples = [[Double]](repeating: [], count: arms.count)
                    for round in 0 ..< 8 {
                        let order =
                            round % 2 == 0 ? Array(arms.indices) : Array(arms.indices.reversed())
                        for index in order {
                            let start = DispatchTime.now().uptimeNanoseconds
                            var state = x.state
                            var outputs = [MLXArray]()
                            for _ in 0 ..< 4 {
                                let (y, next) = arms[index].run(x, state)
                                outputs.append(y)
                                state = next
                            }
                            eval(outputs + [state])
                            let elapsed = DispatchTime.now().uptimeNanoseconds - start
                            // Round 0 compiles and warms each kernel.
                            if round > 0 { samples[index].append(Double(elapsed) / 4e6) }
                        }
                    }
                    let medians = samples.map { $0.sorted()[$0.count / 2] }
                    print(
                        "| \(GPU.deviceInfo().architecture) | \(geometry.name) | \(steps) | "
                            + medians.map { String(format: "%.3f", $0) }.joined(separator: " | ")
                            + " |")
                }
            }
        }

        @Test func killSwitchParsing() {
            typealias Mode = GatedDeltaPrefill.Mode
            #expect(GatedDeltaPrefill.envFlag == "DARKBLOOM_TTFT_GDN_BLOCKED")
            #expect(Mode(environmentValue: nil) == .probe)
            for probe in ["", "1", "on", "true", "auto"] {
                #expect(Mode(environmentValue: probe) == .probe, "\(probe)")
            }
            for off in ["0", "false", "FALSE", "no", " off "] {
                #expect(Mode(environmentValue: off) == .stock, "\(off)")
            }
            #expect(Mode(environmentValue: "packed") == .forced(.packed))
            #expect(Mode(environmentValue: " ROWS ") == .forced(.rows))
        }
    }
}
