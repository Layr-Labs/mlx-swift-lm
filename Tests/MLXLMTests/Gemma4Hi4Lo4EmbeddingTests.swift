import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite("Gemma 4 Hi4/Lo4 Bit-Plane Embedding", .serialized)
struct Gemma4Hi4Lo4EmbeddingTests {

    // MARK: - Helpers

    private func makeFixture(
        vocabSize: Int = 4096,
        hiddenSize: Int = 256,
        dtype: DType = .bfloat16,
        chunkRows: Int = 512,
        seed: UInt64 = 2026
    ) throws -> (stock: QuantizedEmbedding, hiLo: Gemma4Hi4Lo4Embedding, dequantizedW: MLXArray) {
        MLXRandom.seed(seed)
        let rawWeight = (MLXRandom.normal([vocabSize, hiddenSize]) * Float(0.05)).asType(dtype)
        let stock = QuantizedEmbedding(weight: rawWeight, groupSize: 64, bits: 8, mode: .affine)
        eval(stock)
        let hiLo = try #require(Gemma4Hi4Lo4Embedding.from(stock, chunkRows: chunkRows))
        let wDeq = dequantized(
            stock.weight, scales: stock.scales, biases: stock.biases,
            groupSize: 64, bits: 8, mode: .affine, dtype: .float32)
        eval(wDeq)
        return (stock, hiLo, wDeq)
    }

    /// Build `[B, H]` hidden states with two planted near-competitor tokens inside
    /// Pass 1's Cauchy-Schwarz error band and a clear margin above bulk vocab rows.
    private func makePeakedHidden(
        batchSize: Int,
        vocabSize: Int,
        dequantizedW: MLXArray,
        dtype: DType,
        competitorsPerRow: Int = 2,
        seed: UInt64 = 9001
    ) -> MLXArray {
        MLXRandom.seed(seed)
        let hiddenSize = dequantizedW.dim(1)
        var rows: [MLXArray] = []
        rows.reserveCapacity(batchSize)
        for b in 0 ..< batchSize {
            var acc = MLXRandom.normal([hiddenSize]) * Float(0.15)
            for c in 0 ..< competitorsPerRow {
                let tok = (b * 389 + c * 197 + 17) % vocabSize
                let weight = Float(16.0) - Float(c) * Float(0.12)
                acc = acc + weight * dequantizedW[tok]
            }
            rows.append(acc.reshaped([1, hiddenSize]).asType(dtype))
        }
        let hidden = concatenated(rows, axis: 0)
        eval(hidden)
        return hidden
    }

    /// Reference exact dense logits matching single-row `qmv` + softcap semantics.
    private func denseReferenceLogits(
        stock: QuantizedEmbedding,
        hidden: MLXArray,
        softcap: Float
    ) -> MLXArray {
        let x2D = hidden.ndim == 3 ? hidden.squeezed(axis: 1) : hidden
        let batchSize = x2D.dim(0)
        var perRow: [MLXArray] = []
        perRow.reserveCapacity(batchSize)
        for b in 0 ..< batchSize {
            perRow.append(stock.asLinear(x2D[b ..< b + 1]))
        }
        let raw = perRow.count == 1 ? perRow[0] : concatenated(perRow, axis: 0)
        return Gemma4Hi4Lo4Embedding.applySoftcap(raw, softcap: softcap)
    }

    private func bitExactEqual(_ lhs: MLXArray, _ rhs: MLXArray) -> Bool {
        guard lhs.shape == rhs.shape, lhs.dtype == rhs.dtype else { return false }
        eval(lhs, rhs)
        if lhs.size == 0 { return true }
        switch lhs.dtype {
        case .bfloat16, .float16:
            let a = lhs.reshaped([-1]).view(dtype: .uint16)
            let b = rhs.reshaped([-1]).view(dtype: .uint16)
            return arrayEqual(a, b).item(Bool.self)
        case .float32:
            let a = lhs.reshaped([-1]).view(dtype: .uint32)
            let b = rhs.reshaped([-1]).view(dtype: .uint32)
            return arrayEqual(a, b).item(Bool.self)
        default:
            return arrayEqual(lhs, rhs).item(Bool.self)
        }
    }

    private func makeReferenceSampling(
        vocabSize: Int,
        rows: [CBv2SamplerRow],
        logits: MLXArray,
        fallbackSeed: UInt64 = 12345
    ) -> (tokens: [Int32], expNoise: MLXArray) {
        let pipeline = LogitsPipelineV2(vocabSize: vocabSize)
        let sampler = SamplerV2(fallbackSeed: fallbackSeed)
        pipeline.setRows(rows)
        sampler.setRows(rows)

        let noiseSpec: [(seed: UInt64?, id: UInt64, step: UInt64, greedy: Bool)] = rows.map { row in
            (
                seed: row.params.seed,
                id: row.id.raw,
                step: UInt64(row.outputTokens.count),
                greedy: row.params.temperature < LogitsPipelineV2.greedyEpsilon
            )
        }
        let expNoise = sampler.verifyExponentialNoise(rows: noiseSpec, vocab: vocabSize)
        let processed = pipeline.process(logits)
        let sampled = sampler.sample(from: processed.sampling)
        eval(expNoise, sampled)
        return (sampled.asArray(Int32.self), expNoise)
    }

    // MARK: - Env Flag & Eligibility

    @Test("Environment flag defaults off and parses exact/refine/off modes")
    func envFlagResolution() {
        #expect(Gemma4Hi4Lo4Embedding.envKey == "DARKBLOOM_GEMMA4_HI4_LM_HEAD")
        #expect(!Gemma4Hi4Lo4Embedding.isEnabled(environment: [:]))
        #expect(Gemma4Hi4Lo4Embedding.resolvedStrategy(environment: [:]) == .off)
        #expect(
            Gemma4Hi4Lo4Embedding.resolvedStrategy(environment: [Gemma4Hi4Lo4Embedding.envKey: ""])
                == .off)
        for exactVal in ["1", "true", "on", "exact", " EXACT "] {
            #expect(
                Gemma4Hi4Lo4Embedding.resolvedStrategy(
                    environment: [Gemma4Hi4Lo4Embedding.envKey: exactVal]) == .exact)
            #expect(
                Gemma4Hi4Lo4Embedding.isEnabled(
                    environment: [Gemma4Hi4Lo4Embedding.envKey: exactVal]))
        }
        for refineVal in ["refine", "fast", "top64", " REFINE "] {
            #expect(
                Gemma4Hi4Lo4Embedding.resolvedStrategy(
                    environment: [Gemma4Hi4Lo4Embedding.envKey: refineVal]) == .refine)
            #expect(
                Gemma4Hi4Lo4Embedding.isEnabled(
                    environment: [Gemma4Hi4Lo4Embedding.envKey: refineVal]))
        }
        for offValue in ["0", "false", "FALSE", "no", "off", " 0 ", "unknown"] {
            #expect(
                !Gemma4Hi4Lo4Embedding.isEnabled(
                    environment: [Gemma4Hi4Lo4Embedding.envKey: offValue]))
        }
    }

    @Test(
        "Refine strategy (.refine) matches dense 8-bit exact greedy, gumbel, and filtered across B=1,4"
    )
    func refineStrategyMatchesExactDense() throws {
        let vocabSize = 4096
        let (stock, _, wDeq) = try makeFixture(
            vocabSize: vocabSize, hiddenSize: 256, dtype: .bfloat16, chunkRows: 1024, seed: 941)
        let refineEmb = try #require(
            Gemma4Hi4Lo4Embedding.from(stock, samplingStrategy: .refine, chunkRows: 1024))

        for batchSize in [1, 4] {
            let hidden = makePeakedHidden(
                batchSize: batchSize, vocabSize: vocabSize, dequantizedW: wDeq,
                dtype: .bfloat16, competitorsPerRow: 4, seed: UInt64(950 + batchSize))
            let refLogits = denseReferenceLogits(stock: stock, hidden: hidden, softcap: 30.0)

            let greedyRef = argMax(refLogits, axis: -1).asType(.int32)
            let greedyDiag = try #require(
                refineEmb.sparseExactHeadSampleWithDiagnostics(
                    hidden: hidden, mode: .greedy, softcap: 30.0))
            eval(greedyRef, greedyDiag.tokens, greedyDiag.nonFinite)
            #expect(greedyDiag.tokens.asArray(Int32.self) == greedyRef.asArray(Int32.self))
            #expect(
                greedyDiag.nonFinite.asArray(Int32.self)
                    == Array(repeating: Int32(0), count: batchSize))

            let params = (0 ..< batchSize).map { i in
                CBv2SamplingParams(
                    temperature: 0.7, topP: 0.9, topK: 16, minP: 0.05, seed: UInt64(960 + i))
            }
            let rows = (0 ..< batchSize).map { i in
                CBv2SamplerRow(
                    id: CBv2RequestID(UInt64(960 + i)), params: params[i], promptTokens: [1])
            }
            let (filtRef, expNoise) = makeReferenceSampling(
                vocabSize: vocabSize, rows: rows, logits: refLogits)
            let filtDiag = try #require(
                refineEmb.sparseExactHeadSampleWithDiagnostics(
                    hidden: hidden,
                    mode: .filtered(
                        temperatures: MLXArray(params.map(\.temperature)),
                        topK: params.map(\.topK),
                        topP: params.map(\.topP),
                        minP: params.map(\.minP),
                        expNoise: expNoise),
                    softcap: 30.0))
            eval(filtDiag.tokens, filtDiag.nonFinite)
            #expect(filtDiag.tokens.asArray(Int32.self) == filtRef)
            #expect(
                filtDiag.nonFinite.asArray(Int32.self)
                    == Array(repeating: Int32(0), count: batchSize))
        }
    }

    @Test("Unsupported QuantizedEmbedding configurations return nil cleanly from from(_:)")
    func unsupportedQuantizedEmbeddingsReturnNil() throws {
        MLXRandom.seed(7)
        let w = MLXRandom.normal([128, 256]).asType(.bfloat16)

        let q4 = QuantizedEmbedding(weight: w, groupSize: 64, bits: 4, mode: .affine)
        eval(q4)
        #expect(Gemma4Hi4Lo4Embedding.from(q4) == nil)

        let q8g32 = QuantizedEmbedding(weight: w, groupSize: 32, bits: 8, mode: .affine)
        eval(q8g32)
        #expect(Gemma4Hi4Lo4Embedding.from(q8g32) == nil)

        let q8g128 = QuantizedEmbedding(weight: w, groupSize: 128, bits: 8, mode: .affine)
        eval(q8g128)
        #expect(Gemma4Hi4Lo4Embedding.from(q8g128) == nil)

        let q8NoBiases = QuantizedEmbedding(
            weight: MLXArray.zeros([128, 64], dtype: .uint32),
            scales: MLXArray.ones([128, 4], type: Float.self).asType(.bfloat16),
            biases: nil,
            groupSize: 64, bits: 8, mode: .affine)
        eval(q8NoBiases)
        #expect(Gemma4Hi4Lo4Embedding.from(q8NoBiases) == nil)

        // Valid 8-bit group-64 affine converts cleanly with zero extra weight bytes.
        let valid = QuantizedEmbedding(weight: w, groupSize: 64, bits: 8, mode: .affine)
        eval(valid)
        let converted = try #require(Gemma4Hi4Lo4Embedding.from(valid, chunkRows: 64))
        #expect(converted.shape == (128, 256))
        #expect(converted.bits == 8)
        #expect(converted.groupSize == 64)
        #expect(converted.mode == .affine)
        #expect(converted.hiWeight.shape == [128, 32])
        #expect(converted.loWeight.shape == [128, 32])
        #expect(converted.rowResidualNorm.shape == [128])
        #expect(converted.rowResidualNorm.dtype == .float32)
        #expect(converted.hiWeight.nbytes + converted.loWeight.nbytes == valid.weight.nbytes)
    }

    // MARK: - Bit-Packing Verification Against MLX Quantized / Dequantized

    @Test("MLX 8-bit and 4-bit little-endian bit-packing matches MLX.quantized and MLX.dequantized")
    func bitPackingOrderMatchesMLXDequantized() {
        // Explicit deterministic byte pattern verifying little-endian byte and nibble order:
        // q = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0] repeated 8x across 64 elements.
        let w0: UInt32 = 0x7856_3412
        let w1: UInt32 = 0xF0DE_BC9A
        var words: [UInt32] = []
        words.reserveCapacity(16)
        for _ in 0 ..< 8 {
            words.append(w0)
            words.append(w1)
        }
        let w8 = MLXArray(words, [1, 16])
        let (hi, lo) = Gemma4Hi4Lo4Embedding.splitW8ToHiLo(w8)
        let roundTrip = Gemma4Hi4Lo4Embedding.interleaveHiLoToW8(hi: hi, lo: lo)
        eval(hi, lo, roundTrip)

        // Expected little-endian nibble packs:
        // hi nibbles = [1, 3, 5, 7, 9, B, D, F] -> 0xFDB9_7531
        // lo nibbles = [2, 4, 6, 8, A, C, E, 0] -> 0x0ECA_8642
        let hiVals = hi.asArray(UInt32.self)
        let loVals = lo.asArray(UInt32.self)
        #expect(hiVals.count == 8)
        #expect(loVals.count == 8)
        for col in 0 ..< 8 {
            #expect(hiVals[col] == 0xFDB9_7531)
            #expect(loVals[col] == 0x0ECA_8642)
        }
        #expect(arrayEqual(w8, roundTrip).item(Bool.self))

        // Verify MLX's own dequantized(hi, bits: 4) and dequantized(lo, bits: 4)
        // reconstruct exact dequantized(w8, bits: 8) element-for-element.
        let scales = MLXArray([Float(1.0)], [1, 1])
        let biases = MLXArray([Float(0.0)], [1, 1])
        let deq8 = dequantized(
            w8, scales: scales, biases: biases, groupSize: 64, bits: 8, mode: .affine)
        let deqHi = dequantized(
            hi, scales: scales, biases: biases, groupSize: 64, bits: 4, mode: .affine)
        let deqLo = dequantized(
            lo, scales: scales, biases: biases, groupSize: 64, bits: 4, mode: .affine)
        let reconstructed = deqHi * Float(16) + deqLo
        eval(deq8, reconstructed)
        #expect(arrayEqual(deq8, reconstructed).item(Bool.self))

        // Verify random MLX.quantized tensor split -> dequantize -> interleave round-trip.
        MLXRandom.seed(314)
        let randomW = MLXRandom.normal([64, 192]).asType(.float32)
        let (qW, qScales, qBiases) = quantized(randomW, groupSize: 64, bits: 8, mode: .affine)
        let (rHi, rLo) = Gemma4Hi4Lo4Embedding.splitW8ToHiLo(qW)
        let rW8 = Gemma4Hi4Lo4Embedding.interleaveHiLoToW8(hi: rHi, lo: rLo)
        let rDeq8 = dequantized(
            qW, scales: qScales, biases: qBiases, groupSize: 64, bits: 8, mode: .affine)
        let rDeqHi = dequantized(
            rHi, scales: qScales * Float(16), biases: qBiases! + Float(7.5) * qScales,
            groupSize: 64, bits: 4, mode: .affine)
        let rDeqLo = dequantized(
            rLo, scales: qScales, biases: -Float(7.5) * qScales,
            groupSize: 64, bits: 4, mode: .affine)
        let maxErr = MLX.abs(rDeq8 - (rDeqHi + rDeqLo)).max().item(Float.self)
        eval(rW8)
        #expect(arrayEqual(qW, rW8).item(Bool.self))
        #expect(maxErr < 1e-5, "Hi4+Lo4 algebraic identity drift: \(maxErr)")
    }

    // MARK: - 0-ULP Bit-Exact callAsFunction & asLinear

    @Test("callAsFunction is 0-ULP bit-exact vs stock QuantizedEmbedding across dtypes and shapes")
    func callAsFunctionBitExactVsStockQuantizedEmbedding() throws {
        for dtype in [DType.bfloat16, .float16, .float32] {
            let (stock, hiLo, _) = try makeFixture(
                vocabSize: 512, hiddenSize: 256, dtype: dtype, chunkRows: 128, seed: 101)

            let indices1D = MLXArray([Int32(0), 7, 127, 255, 511, -1, -512])
            let indices2D = MLXArray([Int32(3), 42, 199, 500, 12, 333], [2, 3])
            let indicesInt64 = MLXArray([Int64(5), 64, 300, 510])
            let indicesEmpty = MLXArray.zeros([0], dtype: .int32)

            #expect(bitExactEqual(hiLo(indices1D), stock(indices1D)))
            #expect(bitExactEqual(hiLo(indices2D), stock(indices2D)))
            #expect(bitExactEqual(hiLo(indicesInt64), stock(indicesInt64)))
            let emptyOut = hiLo(indicesEmpty)
            #expect(emptyOut.shape == [0, 256])
            #expect(emptyOut.dtype == dtype)
        }
    }

    @Test("asLinear is 0-ULP bit-exact vs stock QuantizedEmbedding (unchunked and chunked)")
    func asLinearBitExactVsStockQuantizedEmbedding() throws {
        for dtype in [DType.bfloat16, .float16, .float32] {
            let (stock, hiLo, _) = try makeFixture(
                vocabSize: 512, hiddenSize: 256, dtype: dtype, chunkRows: 128, seed: 202)
            MLXRandom.seed(303)

            for shape in [[1, 256], [4, 256], [8, 256], [2, 3, 256]] {
                let x = MLXRandom.normal(shape).asType(dtype)
                let expected = stock.asLinear(x)
                let actual = hiLo.asLinear(x)
                let actualChunked = hiLo.asLinear(x, chunkRows: 128)
                #expect(bitExactEqual(actual, expected))
                #expect(bitExactEqual(actualChunked, expected))
            }
        }
    }

    // MARK: - Sparse Exact Greedy Sampling

    @Test(
        "sparseExactHeadSample(.greedy) is 0-ULP bit-exact across B=1,4,8 with and without softcap and prunes vocab"
    )
    func sparseExactGreedyMatchesDenseAcrossBatchSizesAndSoftcap() throws {
        let vocabSize = 4096
        let (stock, hiLo, wDeq) = try makeFixture(
            vocabSize: vocabSize, hiddenSize: 256, dtype: .bfloat16, chunkRows: 1024, seed: 404)

        for softcap in [Float(30.0), Float(0.0)] {
            for batchSize in [1, 4, 8] {
                let hidden = makePeakedHidden(
                    batchSize: batchSize, vocabSize: vocabSize, dequantizedW: wDeq,
                    dtype: .bfloat16, competitorsPerRow: 2, seed: UInt64(1000 + batchSize))

                // Exercise both 2D [B, H] and 3D [B, 1, H] decode hidden shapes.
                let inputHidden = batchSize == 4 ? hidden.expandedDimensions(axis: 1) : hidden
                let refLogits = denseReferenceLogits(
                    stock: stock, hidden: hidden, softcap: softcap)
                let refTokens = argMax(refLogits, axis: -1).asType(.int32)

                let diag = try #require(
                    hiLo.sparseExactHeadSampleWithDiagnostics(
                        hidden: inputHidden, mode: .greedy, softcap: softcap))
                eval(
                    refTokens, diag.tokens, diag.nonFinite, diag.candidateCounts, diag.overflow,
                    diag.needsFallback)

                let actualTokens = diag.tokens.asArray(Int32.self)
                let expectedTokens = refTokens.asArray(Int32.self)
                let nonFinite = diag.nonFinite.asArray(Int32.self)
                let counts = diag.candidateCounts.asArray(Int32.self)
                let overflow = diag.overflow.asArray(UInt8.self)
                let fallback = diag.needsFallback.asArray(UInt8.self)

                #expect(actualTokens == expectedTokens)
                #expect(nonFinite == Array(repeating: Int32(0), count: batchSize))
                #expect(overflow == Array(repeating: UInt8(0), count: batchSize))
                #expect(fallback == Array(repeating: UInt8(0), count: batchSize))
                for c in counts {
                    #expect(
                        c > 0 && c < Int32(vocabSize / 10),
                        "Expected pruning, got \(c)/\(vocabSize)")
                }
            }
        }
    }

    // MARK: - Sparse Exact Gumbel Sampling

    @Test(
        "sparseExactHeadSample(.gumbel) matches exact SamplerV2 + LogitsPipelineV2 on the same expNoise"
    )
    func sparseExactGumbelMatchesSamplerV2BitExact() throws {
        let vocabSize = 4096
        let (stock, hiLo, wDeq) = try makeFixture(
            vocabSize: vocabSize, hiddenSize: 256, dtype: .bfloat16, chunkRows: 1024, seed: 505)

        let tempsList: [Float] = [0.0, 0.35, 0.7, 1.0, 1.4, 0.5, 0.85, 0.0]
        let batchSize = tempsList.count
        let hidden = makePeakedHidden(
            batchSize: batchSize, vocabSize: vocabSize, dequantizedW: wDeq,
            dtype: .bfloat16, competitorsPerRow: 2, seed: 506)

        let rows: [CBv2SamplerRow] = tempsList.enumerated().map { idx, temp in
            CBv2SamplerRow(
                id: CBv2RequestID(UInt64(idx + 1)),
                params: CBv2SamplingParams(
                    temperature: temp, topP: 1.0, topK: 0, minP: 0,
                    seed: UInt64(700 + idx)),
                promptTokens: [1, 2, 3])
        }

        for softcap in [Float(30.0), Float(0.0)] {
            let refLogits = denseReferenceLogits(stock: stock, hidden: hidden, softcap: softcap)
            let (refTokens, expNoise) = makeReferenceSampling(
                vocabSize: vocabSize, rows: rows, logits: refLogits)

            let temps = MLXArray(tempsList)
            let diag = try #require(
                hiLo.sparseExactHeadSampleWithDiagnostics(
                    hidden: hidden,
                    mode: .gumbel(temperatures: temps, expNoise: expNoise),
                    softcap: softcap))
            eval(
                diag.tokens, diag.nonFinite, diag.candidateCounts, diag.overflow, diag.needsFallback
            )

            #expect(diag.tokens.asArray(Int32.self) == refTokens)
            #expect(
                diag.nonFinite.asArray(Int32.self) == Array(repeating: Int32(0), count: batchSize))
            #expect(
                diag.overflow.asArray(UInt8.self) == Array(repeating: UInt8(0), count: batchSize))
            #expect(
                diag.needsFallback.asArray(UInt8.self)
                    == Array(repeating: UInt8(0), count: batchSize))
            for c in diag.candidateCounts.asArray(Int32.self) {
                #expect(
                    c > 0 && c < Int32(vocabSize / 5),
                    "Expected Gumbel pruning, got \(c)/\(vocabSize)")
            }
        }
    }

    // MARK: - Sparse Exact Filtered Sampling (topK, minP, topP, mixed)

    @Test(
        "sparseExactHeadSample(.filtered) matches exact LogitsPipelineV2 + SamplerV2 across topK, minP, topP, and mixed batches"
    )
    func sparseExactFilteredMatchesPipelineAndSamplerV2BitExact() throws {
        let vocabSize = 4096
        let (stock, hiLo, wDeq) = try makeFixture(
            vocabSize: vocabSize, hiddenSize: 256, dtype: .bfloat16, chunkRows: 1024, seed: 606)

        struct FilterCase {
            let name: String
            let temps: [Float]
            let topK: [Int]
            let topP: [Float]
            let minP: [Float]
            let expectSparsePruning: [Bool]
            let competitorsPerRow: Int
        }

        let cases: [FilterCase] = [
            FilterCase(
                name: "topK-only mixed per-row k",
                temps: [0.8, 0.7, 1.0, 0.6],
                topK: [1, 3, 32, 64],
                topP: [1.0, 1.0, 1.0, 1.0],
                minP: [0.0, 0.0, 0.0, 0.0],
                // With 3 peaked competitors above a flat Gaussian bulk, topK <= 3 prunes
                // tightly while topK = 32, 64 sit inside the noise cloud and fall back exactly.
                expectSparsePruning: [true, true, false, false],
                competitorsPerRow: 3
            ),
            FilterCase(
                name: "minP-only",
                temps: [0.8, 0.9, 0.6, 1.1],
                topK: [0, 0, 0, 0],
                topP: [1.0, 1.0, 1.0, 1.0],
                minP: [0.02, 0.05, 0.1, 0.25],
                expectSparsePruning: [true, true, true, true],
                competitorsPerRow: 3
            ),
            FilterCase(
                name: "combined topK + topP + minP (incl. Gemma 4 default topK=64 + topP=0.95)",
                temps: [0.8, 0.7, 0.9, 1.0],
                topK: [40, 25, 0, 64],
                topP: [0.9, 0.85, 0.95, 0.95],
                minP: [0.05, 0.0, 0.05, 0.0],
                expectSparsePruning: [true, true, true, true],
                competitorsPerRow: 3
            ),
            FilterCase(
                name: "topP-only including topP=0 greedy collapse",
                temps: [0.7, 0.8, 0.9, 0.8],
                topK: [0, 0, 0, 0],
                topP: [0.5, 0.85, 0.95, 0.0],
                minP: [0.0, 0.0, 0.0, 0.0],
                expectSparsePruning: [true, true, true, true],
                competitorsPerRow: 3
            ),
            FilterCase(
                name: "mixed greedy + topK/topP + minP + unfiltered stochastic",
                temps: [0.0, 0.75, 0.85, 0.9],
                topK: [50, 20, 0, 0],
                topP: [0.9, 0.95, 1.0, 1.0],
                minP: [0.1, 0.0, 0.05, 0.0],
                expectSparsePruning: [true, true, true, false],
                competitorsPerRow: 3
            ),
        ]

        for (caseIdx, testCase) in cases.enumerated() {
            let batchSize = testCase.temps.count
            let hidden = makePeakedHidden(
                batchSize: batchSize, vocabSize: vocabSize, dequantizedW: wDeq,
                dtype: .bfloat16, competitorsPerRow: testCase.competitorsPerRow,
                seed: UInt64(800 + caseIdx * 17))
            let refLogits = denseReferenceLogits(stock: stock, hidden: hidden, softcap: 30.0)

            let rows: [CBv2SamplerRow] = (0 ..< batchSize).map { i in
                CBv2SamplerRow(
                    id: CBv2RequestID(UInt64(caseIdx * 100 + i + 1)),
                    params: CBv2SamplingParams(
                        temperature: testCase.temps[i],
                        topP: testCase.topP[i],
                        topK: testCase.topK[i],
                        minP: testCase.minP[i],
                        seed: UInt64(1200 + caseIdx * 31 + i)),
                    promptTokens: [4, 5])
            }
            let (refTokens, expNoise) = makeReferenceSampling(
                vocabSize: vocabSize, rows: rows, logits: refLogits)

            let diag = try #require(
                hiLo.sparseExactHeadSampleWithDiagnostics(
                    hidden: hidden,
                    mode: .filtered(
                        temperatures: MLXArray(testCase.temps),
                        topK: testCase.topK,
                        topP: testCase.topP,
                        minP: testCase.minP,
                        expNoise: expNoise),
                    softcap: 30.0))
            eval(
                diag.tokens, diag.nonFinite, diag.candidateCounts, diag.overflow, diag.needsFallback
            )

            #expect(
                diag.tokens.asArray(Int32.self) == refTokens,
                "Filtered case '\(testCase.name)' mismatch")
            #expect(
                diag.nonFinite.asArray(Int32.self) == Array(repeating: Int32(0), count: batchSize))

            let counts = diag.candidateCounts.asArray(Int32.self)
            for i in 0 ..< batchSize where testCase.expectSparsePruning[i] {
                #expect(
                    counts[i] > 0 && counts[i] < Int32(vocabSize / 2),
                    "Case '\(testCase.name)' row \(i) expected pruning, got \(counts[i])/\(vocabSize)"
                )
            }
        }
    }

    // MARK: - Overflow Fallback Exactness

    @Test(
        "Overflow fallback (K_cap exceeded) produces 0-ULP bit-exact results across all modes and ties"
    )
    func overflowFallbackProducesExactResults() throws {
        let vocabSize = 2048
        let (stock, hiLo, wDeq) = try makeFixture(
            vocabSize: vocabSize, hiddenSize: 256, dtype: .bfloat16, chunkRows: 512, seed: 707)

        // Plant 8 near-equal competitors per row and cap K_cap = 2 so overflow is guaranteed.
        let batchSize = 4
        let hidden = makePeakedHidden(
            batchSize: batchSize, vocabSize: vocabSize, dequantizedW: wDeq,
            dtype: .bfloat16, competitorsPerRow: 8, seed: 708)
        let refLogits = denseReferenceLogits(stock: stock, hidden: hidden, softcap: 30.0)

        // 1. Greedy overflow fallback
        let refGreedy = argMax(refLogits, axis: -1).asType(.int32)
        let greedyDiag = try #require(
            hiLo.sparseExactHeadSampleWithDiagnostics(
                hidden: hidden, mode: .greedy, softcap: 30.0, candidateCapacity: 2))
        eval(
            refGreedy, greedyDiag.tokens, greedyDiag.nonFinite, greedyDiag.overflow,
            greedyDiag.needsFallback)
        #expect(greedyDiag.tokens.asArray(Int32.self) == refGreedy.asArray(Int32.self))
        #expect(greedyDiag.nonFinite.asArray(Int32.self) == [0, 0, 0, 0])
        #expect(greedyDiag.overflow.asArray(UInt8.self).contains(1))
        #expect(greedyDiag.needsFallback.asArray(UInt8.self).contains(1))

        // 2. Gumbel overflow fallback
        let gumbelTemps: [Float] = [0.7, 1.0, 0.5, 0.0]
        let gumbelRows: [CBv2SamplerRow] = gumbelTemps.enumerated().map { i, t in
            CBv2SamplerRow(
                id: CBv2RequestID(UInt64(200 + i)),
                params: CBv2SamplingParams(temperature: t, seed: UInt64(2100 + i)),
                promptTokens: [1])
        }
        let (refGumbel, gumbelNoise) = makeReferenceSampling(
            vocabSize: vocabSize, rows: gumbelRows, logits: refLogits)
        let gumbelDiag = try #require(
            hiLo.sparseExactHeadSampleWithDiagnostics(
                hidden: hidden,
                mode: .gumbel(temperatures: MLXArray(gumbelTemps), expNoise: gumbelNoise),
                softcap: 30.0, candidateCapacity: 2))
        eval(gumbelDiag.tokens, gumbelDiag.nonFinite, gumbelDiag.overflow, gumbelDiag.needsFallback)
        #expect(gumbelDiag.tokens.asArray(Int32.self) == refGumbel)
        #expect(gumbelDiag.nonFinite.asArray(Int32.self) == [0, 0, 0, 0])
        #expect(gumbelDiag.overflow.asArray(UInt8.self).contains(1))
        #expect(gumbelDiag.needsFallback.asArray(UInt8.self).contains(1))

        // 3. Filtered overflow fallback (topK, minP, topP all exercised inside fallback kernel)
        let filtTemps: [Float] = [0.8, 0.7, 0.9, 0.6]
        let filtTopK = [32, 16, 0, 24]
        let filtTopP: [Float] = [0.9, 1.0, 0.85, 0.75]
        let filtMinP: [Float] = [0.02, 0.05, 0.01, 0.0]
        let filtRows: [CBv2SamplerRow] = (0 ..< batchSize).map { i in
            CBv2SamplerRow(
                id: CBv2RequestID(UInt64(300 + i)),
                params: CBv2SamplingParams(
                    temperature: filtTemps[i], topP: filtTopP[i], topK: filtTopK[i],
                    minP: filtMinP[i], seed: UInt64(3100 + i)),
                promptTokens: [1])
        }
        let (refFiltered, filtNoise) = makeReferenceSampling(
            vocabSize: vocabSize, rows: filtRows, logits: refLogits)
        let filtDiag = try #require(
            hiLo.sparseExactHeadSampleWithDiagnostics(
                hidden: hidden,
                mode: .filtered(
                    temperatures: MLXArray(filtTemps),
                    topK: filtTopK, topP: filtTopP, minP: filtMinP,
                    expNoise: filtNoise),
                softcap: 30.0, candidateCapacity: 2))
        eval(filtDiag.tokens, filtDiag.nonFinite, filtDiag.needsFallback)
        #expect(filtDiag.tokens.asArray(Int32.self) == refFiltered)
        #expect(filtDiag.nonFinite.asArray(Int32.self) == [0, 0, 0, 0])
        #expect(filtDiag.needsFallback.asArray(UInt8.self) == [1, 1, 1, 1])

        // 4. All-equal zero-hidden logits tie-breaking under overflow (smallest token ID wins greedy,
        //    and topK/topP tie budgets match LogitsPipelineV2 ascending token-ID order).
        let zeroHidden = MLXArray.zeros([2, 256], dtype: .bfloat16)
        let zeroRefLogits = denseReferenceLogits(stock: stock, hidden: zeroHidden, softcap: 30.0)
        let zeroGreedyRef = argMax(zeroRefLogits, axis: -1).asType(.int32)
        let zeroGreedyDiag = try #require(
            hiLo.sparseExactHeadSampleWithDiagnostics(
                hidden: zeroHidden, mode: .greedy, softcap: 30.0, candidateCapacity: 4))
        eval(zeroGreedyRef, zeroGreedyDiag.tokens, zeroGreedyDiag.overflow)
        #expect(zeroGreedyDiag.tokens.asArray(Int32.self) == zeroGreedyRef.asArray(Int32.self))
        #expect(zeroGreedyDiag.overflow.asArray(UInt8.self) == [1, 1])

        let tieRows = [
            CBv2SamplerRow(
                id: CBv2RequestID(401),
                params: CBv2SamplingParams(temperature: 1.0, topP: 0.5, topK: 64, seed: 4001),
                promptTokens: [1]),
            CBv2SamplerRow(
                id: CBv2RequestID(402),
                params: CBv2SamplingParams(
                    temperature: 1.0, topP: 0.25, topK: 0, minP: 0.5, seed: 4002),
                promptTokens: [1]),
        ]
        let (zeroFiltRef, zeroFiltNoise) = makeReferenceSampling(
            vocabSize: vocabSize, rows: tieRows, logits: zeroRefLogits)
        let zeroFiltDiag = try #require(
            hiLo.sparseExactHeadSampleWithDiagnostics(
                hidden: zeroHidden,
                mode: .filtered(
                    temperatures: MLXArray([Float(1.0), Float(1.0)]),
                    topK: [64, 0], topP: [0.5, 0.25], minP: [0.0, 0.5],
                    expNoise: zeroFiltNoise),
                softcap: 30.0, candidateCapacity: 4))
        eval(zeroFiltDiag.tokens, zeroFiltDiag.nonFinite, zeroFiltDiag.needsFallback)
        #expect(zeroFiltDiag.tokens.asArray(Int32.self) == zeroFiltRef)
        #expect(zeroFiltDiag.nonFinite.asArray(Int32.self) == [0, 0])
        #expect(zeroFiltDiag.needsFallback.asArray(UInt8.self) == [1, 1])
    }

    // MARK: - Non-Finite Isolation

    @Test("Non-finite hidden row sets nonFinite flag without corrupting finite sibling rows")
    func nonFiniteHiddenProducesNonFiniteFlag() throws {
        let vocabSize = 2048
        let (stock, hiLo, wDeq) = try makeFixture(
            vocabSize: vocabSize, hiddenSize: 256, dtype: .bfloat16, chunkRows: 512, seed: 808)

        let cleanHidden = makePeakedHidden(
            batchSize: 4, vocabSize: vocabSize, dequantizedW: wDeq,
            dtype: .bfloat16, competitorsPerRow: 2, seed: 809)
        let nanRow = MLXArray.full([1, 256], values: MLXArray(Float.nan)).asType(.bfloat16)
        let corruptedHidden = concatenated(
            [cleanHidden[0 ..< 1], nanRow, cleanHidden[2 ..< 4]], axis: 0)
        eval(corruptedHidden)

        let refLogits = denseReferenceLogits(stock: stock, hidden: cleanHidden, softcap: 30.0)
        let refGreedy = argMax(refLogits, axis: -1).asType(.int32)
        eval(refGreedy)
        let expectedGreedy = refGreedy.asArray(Int32.self)

        // Greedy with normal capacity and forced overflow capacity on sibling rows.
        for cap in [Gemma4Hi4Lo4Embedding.defaultCandidateCapacity, 1] {
            let diag = try #require(
                hiLo.sparseExactHeadSampleWithDiagnostics(
                    hidden: corruptedHidden, mode: .greedy, softcap: 30.0, candidateCapacity: cap))
            eval(diag.tokens, diag.nonFinite)
            let tokens = diag.tokens.asArray(Int32.self)
            let nf = diag.nonFinite.asArray(Int32.self)
            #expect(nf == [0, 1, 0, 0])
            #expect(tokens[0] == expectedGreedy[0])
            #expect(tokens[2] == expectedGreedy[2])
            #expect(tokens[3] == expectedGreedy[3])
        }

        // Filtered mode with non-finite row 1.
        let rows: [CBv2SamplerRow] = (0 ..< 4).map { i in
            CBv2SamplerRow(
                id: CBv2RequestID(UInt64(900 + i)),
                params: CBv2SamplingParams(
                    temperature: 0.8, topP: 0.9, topK: 32, minP: 0.05, seed: UInt64(9100 + i)),
                promptTokens: [1])
        }
        let (refFiltered, expNoise) = makeReferenceSampling(
            vocabSize: vocabSize, rows: rows, logits: refLogits)
        let filtDiag = try #require(
            hiLo.sparseExactHeadSampleWithDiagnostics(
                hidden: corruptedHidden,
                mode: .filtered(
                    temperatures: MLXArray([Float(0.8), 0.8, 0.8, 0.8]),
                    topK: [32, 32, 32, 32],
                    topP: [0.9, 0.9, 0.9, 0.9],
                    minP: [0.05, 0.05, 0.05, 0.05],
                    expNoise: expNoise),
                softcap: 30.0))
        eval(filtDiag.tokens, filtDiag.nonFinite)
        let filtTokens = filtDiag.tokens.asArray(Int32.self)
        let filtNF = filtDiag.nonFinite.asArray(Int32.self)
        #expect(filtNF == [0, 1, 0, 0])
        #expect(filtTokens[0] == refFiltered[0])
        #expect(filtTokens[2] == refFiltered[2])
        #expect(filtTokens[3] == refFiltered[3])
    }

    @Test(
        "DefaultSamplerV2 routes topP<=0 to .filtered and Gemma4BatchDecodeProjection handles Hi4/Lo4"
    )
    func samplerPlanTopPZeroAndBatchDecodeEmbeddingParity() throws {
        let vocabSize = 512
        let (stock, hiLo, _) = try makeFixture(
            vocabSize: vocabSize, hiddenSize: 256, dtype: .bfloat16, chunkRows: 128, seed: 919)

        let sampler = CBv2DefaultSampler()
        let rowID = CBv2RequestID(77)
        let params = CBv2SamplingParams(temperature: 0.8, topP: 0.0, topK: 0, minP: 0.0, seed: 42)
        let plan = try #require(
            sampler.sparseDecodeSamplingPlan(
                vocabSize: vocabSize, params: [params], requestIDs: [rowID], stepIndex: 0,
                pendingSampledTokens: nil,
                rowContext: { [CBv2SamplerRow(id: rowID, params: params, promptTokens: [1])] }))
        if case .filtered(_, _, let topP, _, _) = plan {
            #expect(topP == [0.0])
        } else {
            Issue.record("Expected .filtered mode for topP <= 0, got \(plan)")
        }

        // Batch-decode projection parity for B = 4 inside Gemma4BatchDecodeScope.
        MLXRandom.seed(920)
        let x = MLXRandom.normal([4, 1, 256]).asType(.bfloat16)
        let dummyTokens = MLXArray.zeros([4, 1], dtype: .int32)
        let expected = Gemma4BatchDecodeScope.withDecodeFrame(tokens: dummyTokens, enabled: true) {
            Gemma4BatchDecodeProjection.embedding(stock, x)
        }
        let actual = Gemma4BatchDecodeScope.withDecodeFrame(tokens: dummyTokens, enabled: true) {
            Gemma4BatchDecodeProjection.embedding(hiLo, x)
        }
        #expect(bitExactEqual(actual, expected))
    }

    @Test(
        "Refine mode fastChunkTopKKernel matches exact 8-bit argmax on V=8192 B=8 and detects NaN")
    func refineFastChunkTopKLargeVocabMatchesAndDetectsNaN() throws {
        MLXRandom.seed(777)
        let vocab = 8192
        let hidden = 128
        let batch = 8
        let rawWeight = (MLXRandom.normal([vocab, hidden]) * Float(0.05)).asType(.bfloat16)
        let stock = QuantizedEmbedding(weight: rawWeight, groupSize: 64, bits: 8, mode: .affine)
        eval(stock)
        let hiLo = try #require(Gemma4Hi4Lo4Embedding.from(stock, samplingStrategy: .refine))

        let x = MLXRandom.normal([batch, hidden]).asType(.bfloat16)
        let denseLogits = stock.asLinear(x).asType(.float32)
        let exactArgmax = argMax(denseLogits, axis: -1).asType(.int32)

        let greedyDiag = try #require(
            hiLo.sparseExactHeadSampleWithDiagnostics(
                hidden: x, mode: .greedy, softcap: 30, candidateCapacity: 64))
        eval(exactArgmax, greedyDiag.tokens, greedyDiag.nonFinite)
        #expect(greedyDiag.tokens.asArray(Int32.self) == exactArgmax.asArray(Int32.self))
        #expect(greedyDiag.nonFinite.asArray(Int32.self).allSatisfy { $0 == 0 })

        // Filtered mode exercises EMIT=4 (128 candidates) on V=8192 B=8.
        let expNoise = MLXArray.ones([batch, vocab], dtype: .float32)
        let filteredDiag = try #require(
            hiLo.sparseExactHeadSampleWithDiagnostics(
                hidden: x,
                mode: .filtered(
                    temperatures: MLXArray(Array(repeating: Float(0.7), count: batch)),
                    topK: Array(repeating: 64, count: batch),
                    topP: Array(repeating: Float(0.95), count: batch),
                    minP: Array(repeating: Float(0), count: batch),
                    expNoise: expNoise),
                softcap: 30,
                candidateCapacity: 64))
        eval(filteredDiag.tokens, filteredDiag.nonFinite)
        #expect(filteredDiag.nonFinite.asArray(Int32.self).allSatisfy { $0 == 0 })

        // Inject NaN in row 3 -> must flag row 3 nonFinite without corrupting other rows.
        var xRows = x.asType(.float32).asArray(Float.self)
        xRows[3 * hidden] = Float.nan
        let xNaN = MLXArray(xRows, [batch, hidden]).asType(.bfloat16)
        let nanDiag = try #require(
            hiLo.sparseExactHeadSampleWithDiagnostics(
                hidden: xNaN, mode: .greedy, softcap: 30, candidateCapacity: 64))
        eval(nanDiag.tokens, nanDiag.nonFinite)
        let flags = nanDiag.nonFinite.asArray(Int32.self)
        #expect(flags[3] != 0)
        for i in 0 ..< batch where i != 3 {
            #expect(flags[i] == 0)
            #expect(nanDiag.tokens[i].item(Int32.self) == exactArgmax[i].item(Int32.self))
        }

        // topK > 128 on V=8192 must not truncate effectiveK below targetK.
        let wideTopKDiag = try #require(
            hiLo.sparseExactHeadSampleWithDiagnostics(
                hidden: x,
                mode: .filtered(
                    temperatures: MLXArray(Array(repeating: Float(0.7), count: batch)),
                    topK: Array(repeating: 200, count: batch),
                    topP: Array(repeating: Float(0.95), count: batch),
                    minP: Array(repeating: Float(0), count: batch),
                    expNoise: expNoise),
                softcap: 30,
                candidateCapacity: 64))
        eval(wideTopKDiag.candidateCounts, wideTopKDiag.nonFinite)
        #expect(wideTopKDiag.candidateCounts.asArray(Int32.self).allSatisfy { $0 >= 200 })
        #expect(wideTopKDiag.nonFinite.asArray(Int32.self).allSatisfy { $0 == 0 })
    }
}
