import Foundation
import MLX
import Testing

@_spi(DiffusionGemmaDiagnostics) @testable import MLXLMCommon

extension KernelTests {

    /// Kernel tests of the DiffusionGemma canvas sampler in
    /// `DiffusionGemmaSampler.swift` and of the compiled sampler graph in
    /// `DiffusionGemmaCompiledSampler.swift`.
    ///
    /// The compiled graph runs only on one production shape, so these tests
    /// call `DiffusionGemmaCompiledSampler.call` directly on tiny inputs. The
    /// graph has fixed constants: entropy bound 0.1, confidence threshold
    /// 0.005, stability threshold 1 and float32 conditioning. These are the
    /// defaults of `DiffusionGemmaGenerationConfiguration`.
    ///
    /// The suite is serialized because the compiled sampler diagnostics are
    /// process-wide state.
    @Suite(.serialized)
    struct DiffusionGemmaSamplerKernelTests {

        // MARK: - Helpers

        /// True when both arrays have the same shape, dtype and bytes.
        static func identical(_ a: MLXArray, _ b: MLXArray) -> Bool {
            a.shape == b.shape && a.dtype == b.dtype
                && a.asData(access: .copy).data == b.asData(access: .copy).data
        }

        /// The 13 graph inputs in the order that `stepNative` uses.
        static func graphInputs(
            processed: MLXArray, bits: MLXArray, noise: MLXArray, canvas: MLXArray,
            argmax: MLXArray, finished: MLXArray, steps: MLXArray, conditioning: MLXArray,
            history: MLXArray, samplingLogits: MLXArray
        ) -> [MLXArray] {
            [
                processed, bits, noise, canvas, argmax, finished, steps, conditioning, history,
                samplingLogits, MLXArray(Float(UInt32.max)), MLXArray(Float(1).nextDown),
                MLXArray(-Float.greatestFiniteMagnitude),
            ]
        }

        /// Logits for two rows of 4 positions and 8 tokens. Row 0 is confident:
        /// position p has logit 12 at token p and -12 at the other tokens. Row 1
        /// has seeded random logits, so its entropy is high.
        static func twoRowLogits(step: Int) -> MLXArray {
            let tokens = MLXArray(0 ..< 8).reshaped(1, 1, 8)
            let positions = MLXArray(0 ..< 4).reshaped(1, 4, 1)
            let confident = which(tokens .== positions, MLXArray(Float(12)), MLXArray(Float(-12)))
            let uncertain = MLXRandom.normal([1, 4, 8], key: MLXRandom.key(UInt64(40 + step))) * 2
            return concatenated([confident, uncertain], axis: 0)
        }

        // MARK: - Compiled graph against the plain sampler

        /// The compiled graph and the plain `step` get the same logits, the
        /// same random keys and the same state. All outputs must be identical
        /// at each of 3 steps, for a greedy draw and for a temperature draw.
        /// The 3 steps use the flag sets (hasPrevious, hasHistory, lastStep)
        /// = (F, F, F), (T, T, F) and (T, T, T).
        @Test func compiledGraphMatchesPlainStepOverThreeSteps() throws {
            for samplingTemperature: Float in [0, 0.7] {
                let configuration = try DiffusionGemmaGenerationConfiguration(maxDenoisingSteps: 3)
                let initial = MLXArray([Int32(1), 2, 3, 4, 5, 6, 7, 0]).reshaped(2, 4)
                let state = try DiffusionGemmaDenoisingState(
                    initialCanvas: initial, vocabularySize: 8, embeddingDType: .float32,
                    configuration: configuration)

                var canvas = initial
                var argmax = initial
                var finished = MLXArray.zeros([2], dtype: .bool)
                var steps = MLXArray.zeros([2], dtype: .int32)
                var conditioning: MLXArray?
                var history: MLXArray?

                for step in 0 ..< 3 {
                    let raw = Self.twoRowLogits(step: step)
                    let remaining = configuration.maxDenoisingSteps - step
                    let temperature = try configuration.temperature(remainingStep: remaining)
                    let processed = raw.asType(.float32) / temperature
                    let sampleKey = MLXRandom.key(UInt64(100 + step))
                    let noiseKey = MLXRandom.key(UInt64(200 + step))

                    let accepted = try state.step(
                        rawLogits: raw,
                        sample: { values in
                            if samplingTemperature == 0 {
                                return values.argMax(axis: -1).asType(.int32)
                            }
                            return MLXRandom.categorical(
                                values / samplingTemperature, key: sampleKey
                            ).asType(.int32)
                        },
                        noise: { MLXRandom.randInt(Int32(0) ..< Int32(8), [2, 4], key: noiseKey) })

                    let bits =
                        samplingTemperature == 0
                        ? MLXArray(UInt32(0))
                        : DiffusionGemmaCompiledSampler.randomBits(
                            shape: processed.shape, key: sampleKey)
                    let noise = MLXRandom.randInt(Int32(0) ..< Int32(8), [2, 4], key: noiseKey)
                    let samplingLogits =
                        samplingTemperature == 0 ? processed : processed / samplingTemperature
                    let result = DiffusionGemmaCompiledSampler.call(
                        Self.graphInputs(
                            processed: processed, bits: bits, noise: noise, canvas: canvas,
                            argmax: argmax, finished: finished, steps: steps,
                            conditioning: conditioning ?? processed, history: history ?? argmax,
                            samplingLogits: samplingLogits),
                        greedy: samplingTemperature == 0, hasPrevious: conditioning != nil,
                        hasHistory: history != nil, lastStep: remaining == 1)
                    #expect(result.count == 10)

                    // Exact: the same operations run on the same data. All
                    // compared outputs are integers, booleans or copies of
                    // the processed logits.
                    let label = "temperature \(samplingTemperature), step \(step)"
                    #expect(Self.identical(result[0], state.currentCanvas), "canvas, \(label)")
                    #expect(Self.identical(result[1], state.argmaxCanvas), "argmax, \(label)")
                    #expect(Self.identical(result[2], state.finishedRows), "finished, \(label)")
                    let plainConditioning = try #require(state.selfConditioningLogits)
                    #expect(Self.identical(result[3], plainConditioning), "conditioning, \(label)")
                    #expect(Self.identical(result[4], state.stepsUsed), "steps, \(label)")
                    #expect(Self.identical(result[5], accepted), "acceptance, \(label)")
                    #expect(Self.identical(result[7], noise), "noise passthrough, \(label)")
                    #expect(result[8].item(Bool.self) && result[9].item(Bool.self))

                    canvas = result[0]
                    argmax = result[1]
                    finished = result[2]
                    conditioning = result[3]
                    steps = result[4]
                    history = result[1]
                }

                // Row 0 is confident at every step. It is stable from step 2,
                // so it finishes after 2 steps. Row 1 finishes on the last step.
                #expect(state.stepsUsed.asArray(Int32.self) == [2, 3])
                #expect(state.finishedRows.asArray(Bool.self) == [true, true])
                #expect(state.currentCanvas[0].asArray(Int32.self) == [0, 1, 2, 3])
                #expect(try state.finalizedCanvas()[0].asArray(Int32.self) == [0, 1, 2, 3])
                #expect(state.remainingSteps == 0)
            }
        }

        // MARK: - Compiled graph, hand-computed values

        /// One greedy graph step on 2 rows, 3 positions and 4 tokens, with
        /// values computed by hand from the graph source.
        ///
        /// Row 0 is not finished. Its position entropies are about 1.2e-7
        /// (logits 20, 0, 0, 0), 1.268 (logits 1, 0, 0, 0) and 1.386 (logits
        /// 0, 0, 0, 0.01). In entropy order the sums that exclude the current
        /// entry are 0, 1.2e-7 and 1.268, so the first two positions are
        /// accepted and the last position takes the noise token. Row 1 is
        /// finished, so its canvas, argmax, conditioning and step count do
        /// not change.
        @Test func greedyGraphStepHasHandComputedOutputs() {
            let processed = MLXArray(
                [
                    Float(20), 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.01,
                    20, 0, 0, 0, 20, 0, 0, 0, 20, 0, 0, 0,
                ],
                [2, 3, 4])
            let noise = MLXArray([Int32(3), 2, 1, 1, 1, 1], [2, 3])
            let canvas = MLXArray([Int32(2), 2, 2, 3, 3, 3], [2, 3])
            let argmax = MLXArray([Int32(1), 1, 1, 2, 2, 2], [2, 3])
            let finished = MLXArray([false, true])
            let steps = MLXArray([Int32(4), 2])
            let previous = MLXArray.full([2, 3, 4], values: MLXArray(Float(-5)))
            let history = MLXArray([Int32(0), 0, 3, 2, 2, 2], [2, 3])

            for lastStep in [false, true] {
                let result = DiffusionGemmaCompiledSampler.call(
                    Self.graphInputs(
                        processed: processed, bits: MLXArray(UInt32(0)), noise: noise,
                        canvas: canvas, argmax: argmax, finished: finished, steps: steps,
                        conditioning: previous, history: history, samplingLogits: processed),
                    greedy: true, hasPrevious: true, hasHistory: true, lastStep: lastStep)
                #expect(result.count == 10)
                #expect(result[0].asArray(Int32.self) == [0, 0, 1, 3, 3, 3])
                #expect(result[1].asArray(Int32.self) == [0, 0, 3, 2, 2, 2])
                // Row 0 is stable (history equals the new argmax) but its
                // mean entropy (about 0.88) is not below 0.005.
                #expect(result[2].asArray(Bool.self) == (lastStep ? [true, true] : [false, true]))
                // Exact: `which` copies the values.
                #expect(Self.identical(result[3][0], processed[0]))
                #expect(Self.identical(result[3][1], previous[1]))
                #expect(result[3].dtype == .float32)
                #expect(result[4].asArray(Int32.self) == [5, 2])
                #expect(result[5].asArray(Bool.self) == [true, true, false, true, true, true])
                #expect(result[6].asArray(Int32.self) == [0, 0, 3, 0, 0, 0])
                #expect(Self.identical(result[7], noise))
                #expect(result[8].item(Bool.self))
                #expect(result[9].item(Bool.self))
            }
        }

        /// A confident row finishes only when the graph has a history and the
        /// history equals the new argmax. Without a previous conditioning
        /// tensor, the graph returns the processed logits.
        @Test func graphStabilityNeedsHistoryAndMatchingArgmax() {
            // Entropy of logits (20, 0, 0, 0) is about 1.2e-7 < 0.005.
            let processed = MLXArray([Float(20), 0, 0, 0, 0, 20, 0, 0], [1, 2, 4])
            let canvas = MLXArray([Int32(3), 3], [1, 2])
            let noise = MLXArray([Int32(2), 2], [1, 2])
            let finished = MLXArray([false])
            let steps = MLXArray([Int32(0)])
            let cases: [(hasHistory: Bool, history: [Int32], finishes: Bool)] = [
                (false, [0, 1], false),
                (true, [0, 1], true),
                (true, [0, 2], false),
            ]
            for item in cases {
                let result = DiffusionGemmaCompiledSampler.call(
                    Self.graphInputs(
                        processed: processed, bits: MLXArray(UInt32(0)), noise: noise,
                        canvas: canvas, argmax: canvas, finished: finished, steps: steps,
                        conditioning: processed, history: MLXArray(item.history, [1, 2]),
                        samplingLogits: processed),
                    greedy: true, hasPrevious: false, hasHistory: item.hasHistory,
                    lastStep: false)
                #expect(result[1].asArray(Int32.self) == [0, 1])
                #expect(result[0].asArray(Int32.self) == [0, 1])
                #expect(result[2].asArray(Bool.self) == [item.finishes])
                #expect(result[4].asArray(Int32.self) == [1])
                // Exact: the conditioning is a float32 copy of the input.
                #expect(Self.identical(result[3], processed))
            }
        }

        /// The temperature branch draws `argmax(gumbel(u) + samplingLogits)`
        /// with `u = min(bits / UInt32.max, nextDown(1))`. Gumbel noise grows
        /// with `u`, so with equal sampling logits the largest bits win. The
        /// draw uses the sampling logits, not the processed logits.
        @Test func temperatureGraphDrawUsesBitsAndSamplingLogits() {
            let processed = MLXArray([Float(9), 0, 0, 0, 9, 0, 0, 0], [1, 2, 4])
            let sampling = MLXArray.zeros([1, 2, 4], dtype: .float32)
            let half = UInt32(1) << 31
            let bits = MLXArray(
                [half, half, UInt32.max, half, UInt32.max, half, half, half], [1, 2, 4])
            let result = DiffusionGemmaCompiledSampler.call(
                Self.graphInputs(
                    processed: processed, bits: bits, noise: MLXArray([Int32(1), 1], [1, 2]),
                    canvas: MLXArray([Int32(3), 3], [1, 2]),
                    argmax: MLXArray([Int32(3), 3], [1, 2]), finished: MLXArray([false]),
                    steps: MLXArray([Int32(0)]), conditioning: processed,
                    history: MLXArray([Int32(3), 3], [1, 2]), samplingLogits: sampling),
                greedy: false, hasPrevious: false, hasHistory: false, lastStep: false)
            #expect(result[6].asArray(Int32.self) == [2, 0])
            // The argmax output still uses the processed logits.
            #expect(result[1].asArray(Int32.self) == [0, 0])
            #expect(result[8].item(Bool.self))
        }

        /// The validity outputs are false when the noise canvas has a token
        /// outside `0 ..< vocabularySize`.
        @Test func graphFlagsNoiseOutsideTheVocabulary() {
            let processed = MLXArray.zeros([1, 2, 4], dtype: .float32)
            let cases: [(noise: [Int32], valid: Bool)] = [
                ([0, 3], true), ([4, 0], false), ([0, -1], false),
            ]
            for (noise, valid) in cases {
                let result = DiffusionGemmaCompiledSampler.call(
                    Self.graphInputs(
                        processed: processed, bits: MLXArray(UInt32(0)),
                        noise: MLXArray(noise, [1, 2]), canvas: MLXArray([Int32(0), 0], [1, 2]),
                        argmax: MLXArray([Int32(0), 0], [1, 2]), finished: MLXArray([false]),
                        steps: MLXArray([Int32(0)]), conditioning: processed,
                        history: MLXArray([Int32(0), 0], [1, 2]), samplingLogits: processed),
                    greedy: true, hasPrevious: false, hasHistory: false, lastStep: false)
                #expect(result[9].item(Bool.self) == valid, "noise \(noise)")
                #expect(result[8].item(Bool.self))
            }
        }

        /// `randomBits` uses the same bits as MLX uniform sampling: the
        /// uniform values made from them equal `MLXRandom.uniform` with the
        /// same key.
        @Test func randomBitsReproduceNativeUniformDraws() {
            let key = MLXRandom.key(17)
            let bits = DiffusionGemmaCompiledSampler.randomBits(shape: [3, 5], key: key)
            #expect(bits.shape == [3, 5])
            #expect(bits.dtype == .uint32)
            #expect(
                Self.identical(
                    bits, DiffusionGemmaCompiledSampler.randomBits(shape: [3, 5], key: key)))
            #expect(
                !Self.identical(
                    bits,
                    DiffusionGemmaCompiledSampler.randomBits(shape: [3, 5], key: MLXRandom.key(18))
                ))
            // Exact: this is the same divide and clamp that MLX uniform uses.
            let uniform = minimum(
                bits / MLXArray(Float(UInt32.max)), MLXArray(Float(1).nextDown)
            ).asType(.float32)
            let native = MLXRandom.uniform(Float(0) ..< Float(1), [3, 5], key: key)
            #expect(Self.identical(uniform, native))
        }

        /// The compiled sampler is never eligible for tiny shapes.
        @Test func compiledSamplerIsNotEligibleForTinyShapes() throws {
            let configuration = try DiffusionGemmaGenerationConfiguration()
            #expect(
                !DiffusionGemmaCompiledSampler.eligible(
                    logits: MLXArray.zeros([1, 4, 8], dtype: .float32),
                    canvas: MLXArray.zeros([1, 4], dtype: .int32), embeddingDType: .float32,
                    configuration: configuration))
        }

        /// The dispatch counter counts graph calls only while it is armed.
        @Test func diagnosticsCountDispatchesOnlyWhileArmed() {
            let processed = MLXArray.zeros([1, 1, 2], dtype: .float32)
            let token = MLXArray([Int32(0)], [1, 1])
            let inputs = Self.graphInputs(
                processed: processed, bits: MLXArray(UInt32(0)), noise: token, canvas: token,
                argmax: token, finished: MLXArray([false]), steps: MLXArray([Int32(0)]),
                conditioning: processed, history: token, samplingLogits: processed)
            let dispatch = {
                _ = DiffusionGemmaCompiledSampler.call(
                    inputs, greedy: true, hasPrevious: false, hasHistory: false, lastStep: true)
            }

            DiffusionGemmaCompiledSamplerDiagnostics.clearAndArm()
            #expect(
                DiffusionGemmaCompiledSamplerDiagnostics.snapshot()
                    == .init(armed: true, calls: 0))
            dispatch()
            dispatch()
            #expect(
                DiffusionGemmaCompiledSamplerDiagnostics.snapshot()
                    == .init(armed: true, calls: 2))
            #expect(DiffusionGemmaCompiledSamplerDiagnostics.snapshotAndDisarm() == 2)
            dispatch()
            #expect(
                DiffusionGemmaCompiledSamplerDiagnostics.snapshot()
                    == .init(armed: false, calls: 2))
        }

        // MARK: - Plain sampler

        /// `tokenEntropy` and `acceptanceMask` reject a wrong rank, an empty
        /// axis, an integer dtype and an invalid bound.
        @Test func entropyAndAcceptanceRejectInvalidInputs() throws {
            let logitsError = DiffusionGemmaSamplingError.invalidShape("logits")
            #expect(throws: logitsError) {
                try DiffusionGemmaSampling.tokenEntropy(MLXArray.zeros([2, 3]))
            }
            #expect(throws: logitsError) {
                try DiffusionGemmaSampling.tokenEntropy(MLXArray.zeros([1, 0, 3]))
            }
            #expect(throws: logitsError) {
                try DiffusionGemmaSampling.tokenEntropy(MLXArray.zeros([1, 2, 3], dtype: .int32))
            }
            let entropyError = DiffusionGemmaSamplingError.invalidShape("entropy/bound")
            let entropy = MLXArray([Float(0.2), 0.3], [1, 2])
            #expect(throws: entropyError) {
                try DiffusionGemmaSampling.acceptanceMask(
                    entropy: MLXArray.zeros([1, 1, 2]), bound: 0.1)
            }
            #expect(throws: entropyError) {
                try DiffusionGemmaSampling.acceptanceMask(
                    entropy: MLXArray.zeros([1, 2], dtype: .int32), bound: 0.1)
            }
            for bound in [Float(0), -1, .infinity, .nan] {
                #expect(throws: entropyError) {
                    try DiffusionGemmaSampling.acceptanceMask(entropy: entropy, bound: bound)
                }
            }

            // Uniform logits over 4 tokens have entropy ln 4. Tolerance 1e-6:
            // one float32 log-sum-exp.
            let uniform = try DiffusionGemmaSampling.tokenEntropy(MLXArray.zeros([2, 3, 4]))
            #expect(uniform.shape == [2, 3])
            #expect(abs(uniform - Float(log(4.0))).max().item(Float.self) < 1e-6)
        }

        /// The initializer rejects a canvas with the wrong rank, dtype or
        /// tokens, a bad vocabulary size and an integer embedding dtype. A
        /// uint32 canvas is accepted.
        @Test func initializerValidatesTheCanvas() throws {
            let configuration = try DiffusionGemmaGenerationConfiguration(maxDenoisingSteps: 3)
            let shapeError = DiffusionGemmaSamplingError.invalidShape("initial canvas")
            let valid = MLXArray([Int32(0), 1], [1, 2])
            let invalidShapes: [(MLXArray, Int, DType)] = [
                (MLXArray([Int32(0), 1]), 4, .float32),
                (MLXArray([Float(0), 1], [1, 2]), 4, .float32),
                (valid, 0, .float32),
                (valid, Int(Int32.max) + 1, .float32),
                (valid, 4, .int32),
            ]
            for (canvas, vocabulary, dtype) in invalidShapes {
                #expect(throws: shapeError) {
                    try DiffusionGemmaDenoisingState(
                        initialCanvas: canvas, vocabularySize: vocabulary, embeddingDType: dtype,
                        configuration: configuration)
                }
            }
            for tokens: [Int32] in [[0, 4], [-1, 0]] {
                #expect(throws: DiffusionGemmaSamplingError.invalidToken) {
                    try DiffusionGemmaDenoisingState(
                        initialCanvas: MLXArray(tokens, [1, 2]), vocabularySize: 4,
                        embeddingDType: .float32, configuration: configuration)
                }
            }

            let state = try DiffusionGemmaDenoisingState(
                initialCanvas: MLXArray([UInt32(3), 0, 1, 2], [2, 2]), vocabularySize: 4,
                embeddingDType: .float16, configuration: configuration)
            #expect(state.remainingSteps == 3)
            #expect(state.vocabularySize == 4 && state.embeddingDType == .float16)
            #expect(state.finishedRows.asArray(Bool.self) == [false, false])
            #expect(state.stepsUsed.dtype == .int32)
            #expect(state.stepsUsed.asArray(Int32.self) == [0, 0])
            #expect(state.argmaxCanvas.asArray(UInt32.self) == [3, 0, 1, 2])
            #expect(state.selfConditioningLogits == nil)
        }

        /// `step` rejects bad logits and bad callback outputs, and it changes
        /// no state when it throws.
        @Test func stepRejectsInvalidLogitsAndCallbackOutputs() throws {
            let initial = MLXArray([Int32(0), 1], [1, 2])
            let state = try DiffusionGemmaDenoisingState(
                initialCanvas: initial, vocabularySize: 4, embeddingDType: .float32,
                configuration: .init(maxDenoisingSteps: 2))
            let logits = MLXArray.zeros([1, 2, 4])
            let logitsError = DiffusionGemmaSamplingError.invalidShape("decoder logits")
            #expect(throws: logitsError) {
                try state.step(
                    rawLogits: MLXArray.zeros([1, 2, 5]), sample: { _ in initial },
                    noise: { initial })
            }
            #expect(throws: logitsError) {
                try state.step(
                    rawLogits: MLXArray.zeros([1, 2, 4], dtype: .int32), sample: { _ in initial },
                    noise: { initial })
            }
            #expect(throws: DiffusionGemmaSamplingError.invalidShape("sampled canvas")) {
                try state.step(
                    rawLogits: logits, sample: { _ in MLXArray([UInt32(0), 1], [1, 2]) },
                    noise: { initial })
            }
            #expect(throws: DiffusionGemmaSamplingError.invalidShape("sampled canvas")) {
                try state.step(
                    rawLogits: logits, sample: { _ in MLXArray([Int32(0), 1, 2], [1, 3]) },
                    noise: { initial })
            }
            #expect(throws: DiffusionGemmaSamplingError.invalidShape("noise canvas")) {
                try state.step(
                    rawLogits: logits, sample: { _ in initial },
                    noise: { MLXArray([Int32(0), 1]) })
            }
            #expect(throws: DiffusionGemmaSamplingError.invalidToken) {
                try state.step(
                    rawLogits: logits, sample: { _ in initial },
                    noise: { MLXArray([Int32(0), 4], [1, 2]) })
            }
            #expect(throws: DiffusionGemmaSamplingError.invalidDistribution) {
                try state.step(
                    rawLogits: MLXArray.full([1, 2, 4], values: MLXArray(Float.nan)),
                    sample: { _ in initial }, noise: { initial })
            }
            #expect(state.remainingSteps == 2)
            #expect(state.currentCanvas.asArray(Int32.self) == [0, 1])
            #expect(state.stepsUsed.asArray(Int32.self) == [0])
            #expect(state.selfConditioningLogits == nil)
            #expect(state.retainedPayloadBytes == 2 * 8 + 1 + 4)
        }

        /// With stability threshold 2, a confident row finishes only when the
        /// last 2 argmax canvases both equal the new argmax. The history keeps
        /// at most 2 canvases, which the payload byte count shows.
        ///
        /// Steps use logits A (argmax 0, 1), then B (argmax 2, 3) 3 times.
        /// Step 3 sees history [A, B] and is not stable. Step 4 sees [B, B].
        @Test func stabilityThresholdComparesTheWholeHistoryWindow() throws {
            let configuration = try DiffusionGemmaGenerationConfiguration(
                maxDenoisingSteps: 6, stabilityThreshold: 2)
            let initial = MLXArray([Int32(3), 3], [1, 2])
            let state = try DiffusionGemmaDenoisingState(
                initialCanvas: initial, vocabularySize: 4, embeddingDType: .float32,
                configuration: configuration)
            // Canvases are 2 int32 values (8 bytes); finished is 1 bool; steps
            // is 1 int32; conditioning is 1 x 2 x 4 float32 (32 bytes).
            #expect(state.retainedPayloadBytes == 8 + 8 + 1 + 4)
            let a = MLXArray([Float(20), -20, -20, -20, -20, 20, -20, -20], [1, 2, 4])
            let b = MLXArray([Float(-20), -20, 20, -20, -20, -20, -20, 20], [1, 2, 4])
            let expected: [(logits: MLXArray, finished: Bool, bytes: Int)] = [
                (a, false, 21 + 8 + 32),
                (b, false, 21 + 16 + 32),
                (b, false, 21 + 16 + 32),
                (b, true, 21 + 16 + 32),
            ]
            for (index, item) in expected.enumerated() {
                try state.step(
                    rawLogits: item.logits,
                    sample: { $0.argMax(axis: -1).asType(.int32) }, noise: { initial })
                #expect(state.finishedRows.asArray(Bool.self) == [item.finished], "step \(index)")
                #expect(state.retainedPayloadBytes == item.bytes, "step \(index)")
            }
            #expect(state.stepsUsed.asArray(Int32.self) == [4])
            #expect(state.remainingSteps == 2)
            #expect(try state.finalizedCanvas().asArray(Int32.self) == [2, 3])
        }

        /// `stepNative` rejects a sampling temperature that is not finite or
        /// is negative, before it takes a key or changes state.
        @Test func stepNativeRejectsInvalidSamplingTemperature() throws {
            let state = try DiffusionGemmaDenoisingState(
                initialCanvas: MLXArray([Int32(0), 1], [1, 2]), vocabularySize: 4,
                embeddingDType: .float32, configuration: .init(maxDenoisingSteps: 2))
            let keys = DGSamplerKernelKeyCounter()
            for temperature in [Float.nan, -1, .infinity] {
                #expect(throws: DiffusionGemmaSamplingError.invalidShape("sampling temperature")) {
                    try state.stepNative(
                        rawLogits: MLXArray.zeros([1, 2, 4]), samplingTemperature: temperature,
                        nextKey: keys.next)
                }
            }
            #expect(keys.count == 0)
            #expect(state.remainingSteps == 2)
            #expect(state.selfConditioningLogits == nil)
        }
    }
}

/// Counts the keys that `stepNative` takes.
private final class DGSamplerKernelKeyCounter {
    var count = 0
    func next() -> MLXArray {
        count += 1
        return MLXRandom.key(UInt64(count))
    }
}
