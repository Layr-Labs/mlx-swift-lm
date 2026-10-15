// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {
    /// The default-off absorbed DSA path against the unchanged expanded reference
    /// on tiny synthetic weights: parity, padding, MTP, packed-layout semantics
    /// and zero added residency. Not real-checkpoint, quality or speed evidence.
    ///
    /// Parity tolerances are max |absorbed - reference| over each output tensor,
    /// relative to that tensor's reference peak:
    /// - FP32 on CPU differs only by association: 2e-5.
    /// - FP32 on GPU may use TF32 GEMMs in both paths (MLX_ENABLE_TF32 defaults
    ///   on); 2^-11 operand rounding allows 2^-8.
    /// - Q4 with BF16 on GPU (serving kernels accumulate in FP32): the reference
    ///   rounds expanded K/V to BF16, absorbed rounds q W_k and Σ p c, and
    ///   multi-row kernels in either path round dequantized weights to BF16.
    ///   Emulating both orders with every weight-rounding mixture peaked at
    ///   2.1% over 9,000 trials (p99 at most 1.3%): 2^-5.
    /// - Q4 with BF16 on CPU: MLX's CPU quantized kernels round every BF16
    ///   operation, in both orders. Emulation peaked at 5.7% over 1,500 trials
    ///   (p99 5.0%): 2^-3. This only checks structure; GPU bounds carry precision.
    @Suite(.serialized)
    struct GLM5NextAbsorbedAttentionTests {
        private struct Fixture {
            let configuration: GLM5NextConfiguration
            let inventory: GLM5NextWeightInventory
            var text: GLM5NextTextConfiguration { configuration.textConfig }
            var activation: DType { inventory.storage == .vontraQ4 ? .bfloat16 : .float32 }

            /// Multi-head DSA with distinct Dq, Dv and C and an 8-token budget,
            /// so contexts cross both padded and strictly sparse selections.
            init(q4: Bool = false) throws {
                let base = try GLM5NextInventoryFixture.configuration(layers: 4, q4: q4)
                var root = try #require(
                    JSONSerialization.jsonObject(with: JSONEncoder().encode(base))
                        as? [String: Any])
                var text = try #require(root["text_config"] as? [String: Any])
                let geometry: [String: Any] =
                    q4
                    ? [
                        "num_attention_heads": 2, "num_key_value_heads": 2, "qk_head_dim": 48,
                        "qk_nope_head_dim": 48, "v_head_dim": 32, "index_n_heads": 2,
                        "index_topk": 8,
                    ]
                    : [
                        "num_attention_heads": 3, "num_key_value_heads": 3, "qk_head_dim": 4,
                        "qk_nope_head_dim": 4, "v_head_dim": 3, "kv_lora_rank": 5,
                        "q_lora_rank": 6, "index_n_heads": 2, "index_head_dim": 3,
                        "index_topk": 8,
                    ]
                text.merge(geometry) { _, replacement in replacement }
                root["text_config"] = text
                configuration = try JSONDecoder().decode(
                    GLM5NextConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
                let storage: GLM5NextWeightInventory.Storage = q4 ? .vontraQ4 : .float32
                inventory = try GLM5NextWeightInventory.validate(
                    configuration: configuration,
                    tensors: GLM5NextWeightInventory.expectedTensors(
                        configuration: configuration, storage: storage,
                        shard: "absorbed.safetensors"),
                    storage: storage)
            }

            /// 2^-5 (1 + odd/128): exact in BF16, with an 8-bit significand.
            static func scale(_ index: Int) -> Float {
                (1 + Float(2 * (index % 64) + 1) / 128) / 32
            }

            func tensor(_ name: String) throws -> MLXArray {
                let header = try #require(inventory.entries[name]).tensor
                let count = header.shape.reduce(1, *)
                let salt = name.utf8.reduce(0) { ($0 * 31 + Int($1)) % 1009 }
                if header.dtype == "U32" {
                    return MLXArray(
                        (0 ..< count).map {
                            UInt32(
                                truncatingIfNeeded: (UInt64($0 + salt * 7919)
                                    &* 0x9E37_79B9_7F4A_7C15) >> 29)
                        }, header.shape)
                }
                let sparse =
                    name.contains("layers.3.self_attn.") || name.contains("layers.4.self_attn.")
                let values: [Float] = (0 ..< count).map { index in
                    let wave = Float(sin(Double(index) * 0.731 + Double(salt)))
                    if name.hasSuffix(".scales") { return Self.scale(index) }
                    if name.hasSuffix(".biases") { return -8 * Self.scale(index) }
                    if name.contains("norm.weight") || name.hasSuffix("_scale") {
                        return 1 + 0.25 * wave
                    }
                    if name.hasSuffix("A_log") { return -0.7 }
                    if name.hasSuffix("dt_bias") { return -0.2 }
                    if name.hasSuffix("_base") || name.hasSuffix("e_score_correction_bias") {
                        return 0
                    }
                    if name.hasSuffix(".bias") { return 0.05 * wave }
                    // Peaked, non-uniform sparse attention; damped elsewhere.
                    return (sparse ? 0.5 : 0.125) * wave
                }
                let array = MLXArray(values, header.shape)
                return header.dtype == "BF16" ? array.asType(.bfloat16) : array
            }

            /// `blockBytes` overrides the absorbed FP32 latent block bound; one
            /// byte forces one query per block.
            func attention(
                _ sparseAttention: GLM5NextExecutionPolicy.SparseAttention,
                blockBytes: Int? = nil
            ) throws -> GLM5NextDSAAttention {
                let prefix = "model.language_model.layers.3.self_attn"
                let policy = GLM5NextExecutionPolicy(sparseAttention: sparseAttention)
                guard let blockBytes else {
                    return try GLM5NextDSAAttention(
                        configuration: text, sourcePrefix: prefix, inventory: inventory,
                        tensor: tensor, releaseSourceTensors: { _ in }, executionPolicy: policy)
                }
                return try GLM5NextDSAAttention(
                    text, prefix: prefix,
                    loader: GLM5NextTensorLoader(inventory, tensor, releaseSourceTensors: { _ in }),
                    executionPolicy: policy, absorbedBlockBytes: blockBytes)
            }

            func attention(policy: GLM5NextExecutionPolicy) throws -> GLM5NextDSAAttention {
                try GLM5NextDSAAttention(
                    configuration: text, sourcePrefix: "model.language_model.layers.3.self_attn",
                    inventory: inventory, tensor: tensor, releaseSourceTensors: { _ in },
                    executionPolicy: policy)
            }

            func target(_ sparseAttention: GLM5NextExecutionPolicy.SparseAttention) throws
                -> GLM5NextTextModel
            {
                try target(policy: .init(sparseAttention: sparseAttention))
            }

            func target(policy: GLM5NextExecutionPolicy) throws -> GLM5NextTextModel {
                guard inventory.storage.isPacked else {
                    return try GLM5NextTextModel(
                        configuration: text, inventory: inventory, tensor: tensor,
                        executionPolicy: policy)
                }
                // Q4 construction materializes each expert stack as it is built.
                return try GLM5NextTextModel(
                    configuration: text, inventory: inventory, tensor: tensor,
                    materializeExpertStack: { array in
                        try withError { error in
                            eval(array)
                            try error.check()
                        }
                    }, releaseSourceTensors: { _ in }, executionPolicy: policy)
            }

            func hidden(batch: Int, positions: Range<Int>) -> MLXArray {
                let width = text.hiddenSize
                var values: [Float] = []
                for row in 0 ..< batch {
                    for position in positions {
                        for channel in 0 ..< width {
                            let seed = ((row * 1000 + position) * width + channel)
                            values.append(Float(sin(Double(seed) * 0.377 + 0.2)))
                        }
                    }
                }
                return MLXArray(values, [batch, positions.count, width]).asType(activation)
            }

            func tolerance(gpu: Bool) -> Float {
                switch (activation == .bfloat16, gpu) {
                case (true, true): 1.0 / 32
                case (true, false): 1.0 / 8
                case (false, true): 1.0 / 256
                case (false, false): 2e-5
                }
            }
        }

        private func expectClose(
            _ actual: MLXArray, _ expected: MLXArray, tolerance: Float, _ context: String,
            sourceLocation: SourceLocation = #_sourceLocation
        ) {
            #expect(actual.shape == expected.shape, "\(context)", sourceLocation: sourceLocation)
            #expect(actual.dtype == expected.dtype, "\(context)", sourceLocation: sourceLocation)
            let values = actual.asType(.float32).asArray(Float.self)
            let reference = expected.asType(.float32).asArray(Float.self)
            guard values.allSatisfy(\.isFinite), reference.allSatisfy(\.isFinite) else {
                Issue.record("\(context): nonfinite output", sourceLocation: sourceLocation)
                return
            }
            let bound = tolerance * (reference.map(abs).max() ?? 0)
            let difference = zip(values, reference).map { abs($0 - $1) }.max() ?? 0
            #expect(
                difference <= bound, "\(context): max difference \(difference), bound \(bound)",
                sourceLocation: sourceLocation)
        }

        @Test(arguments: [false, true], [false, true])
        func contiguousForwardMatchesExpandedReference(gpu: Bool, q4: Bool) throws {
            try Device.withDefaultDevice(gpu ? .gpu : .cpu) {
                let f = try Fixture(q4: q4)
                let reference = try f.attention(.reference)
                let absorbed = try f.attention(.absorbed)
                #expect(
                    reference.executionPolicy.sparseAttention == .reference
                        && reference.absorbed == nil)
                #expect(
                    absorbed.executionPolicy.sparseAttention == .absorbed
                        && absorbed.absorbed != nil)
                let dense = try f.attention(.absorbedDenseHistory)
                #expect(dense.absorbed != nil)
                let referenceCache = GLM5NextDSACache()
                let absorbedCache = GLM5NextDSACache()
                let denseCache = GLM5NextDSACache()
                // Row 1 starts with two padding tokens. Chunk ends cross pool
                // boundaries; the chunk [5, 10) straddles the 8-token budget, so
                // its first queries take the dense prefix and the rest the
                // selector. Then decode one at a time.
                var start = 0
                for end in [5, 10, 11, 12, 15, 16, 20, 21] {
                    let input = f.hidden(batch: 2, positions: start ..< end)
                    let validity = MLXArray(
                        (0 ..< 2).flatMap { row in (start ..< end).map { row == 0 || $0 >= 2 } },
                        [2, end - start])
                    let expected = try reference.forward(
                        input, cache: referenceCache, validity: validity)
                    let actual = try absorbed.forward(
                        input, cache: absorbedCache, validity: validity)
                    let denseActual = try dense.forward(input, cache: denseCache, validity: validity)
                    expectClose(
                        actual, expected, tolerance: f.tolerance(gpu: gpu), "chunk ending \(end)")
                    expectClose(
                        denseActual, expected, tolerance: f.tolerance(gpu: gpu),
                        "dense history, chunk ending \(end)")
                    if start < 2 {
                        let padded = 0 ..< (2 - start)
                        #expect(actual[1, padded].asArray(Float.self).allSatisfy { $0 == 0 })
                        #expect(expected[1, padded].asArray(Float.self).allSatisfy { $0 == 0 })
                        #expect(denseActual[1, padded].asArray(Float.self).allSatisfy { $0 == 0 })
                    }
                    start = end
                }
            }
        }

        /// `SparseDecode.nativeKernels` on the contiguous text path: ragged B2
        /// (row 1 left-padded), chunks of at most eight tokens past the 8-token
        /// budget take the native selection and split-K attention, a nine-token
        /// chunk keeps the graphs. Under either absorbed member and selector.
        @Test(arguments: [false, true])
        func nativeSparseDecodeMatchesExpandedReference(q4: Bool) throws {
            try Device.withDefaultDevice(.gpu) {
                let f = try Fixture(q4: q4)
                let reference = try f.attention(.reference)
                var policies: [GLM5NextExecutionPolicy] = []
                let members: [GLM5NextExecutionPolicy.SparseAttention] = [
                    .absorbed, .absorbedDenseHistory,
                ]
                let selectors: [GLM5NextSparseAttention.SelectionStrategy] = [
                    .reference, .parallelThreshold,
                ]
                for sparse in members {
                    for selection in selectors {
                        policies.append(
                            .init(
                                sparseAttention: sparse,
                                sparseSelection: .init(selection: selection, queryBlocks: true),
                                sparseDecode: .nativeKernels))
                    }
                }
                let candidates = try policies.map { try f.attention(policy: $0) }
                let referenceCache = GLM5NextDSACache()
                let caches = candidates.map { _ in GLM5NextDSACache() }
                var start = 0
                for end in [5, 10, 11, 12, 15, 16, 20, 21, 29, 30, 39, 40] {
                    let input = f.hidden(batch: 2, positions: start ..< end)
                    let validity = MLXArray(
                        (0 ..< 2).flatMap { row in (start ..< end).map { row == 0 || $0 >= 2 } },
                        [2, end - start])
                    let expected = try reference.forward(
                        input, cache: referenceCache, validity: validity)
                    for (index, (candidate, cache)) in zip(candidates, caches).enumerated() {
                        expectClose(
                            try candidate.forward(input, cache: cache, validity: validity),
                            expected, tolerance: f.tolerance(gpu: true),
                            "policy \(index), chunk ending \(end)")
                    }
                    start = end
                }
            }
        }

        /// `SparsePrefill.nativeKernel` on the contiguous text path: ragged B2,
        /// chunks with more than eight sparse queries past the 8-token budget
        /// attend through `GLM5NextSparsePrefillKernel` (the portable path at
        /// this geometry), with and without native decode for the short steps.
        @Test(arguments: [false, true])
        func nativeSparsePrefillMatchesExpandedReference(q4: Bool) throws {
            try Device.withDefaultDevice(.gpu) {
                let f = try Fixture(q4: q4)
                let reference = try f.attention(.reference)
                var policies: [GLM5NextExecutionPolicy] = []
                for sparse in [GLM5NextExecutionPolicy.SparseAttention.absorbed, .absorbedDenseHistory] {
                    for decode in [GLM5NextExecutionPolicy.SparseDecode.reference, .nativeKernels] {
                        policies.append(
                            .init(
                                sparseAttention: sparse,
                                sparseSelection: .init(
                                    selection: .parallelThreshold, queryBlocks: true),
                                sparseDecode: decode, sparsePrefill: .nativeKernel))
                    }
                }
                let candidates = try policies.map { try f.attention(policy: $0) }
                let referenceCache = GLM5NextDSACache()
                let caches = candidates.map { _ in GLM5NextDSACache() }
                var start = 0
                for end in [5, 20, 33, 34, 50] {
                    let input = f.hidden(batch: 2, positions: start ..< end)
                    let validity = MLXArray(
                        (0 ..< 2).flatMap { row in (start ..< end).map { row == 0 || $0 >= 2 } },
                        [2, end - start])
                    let expected = try reference.forward(
                        input, cache: referenceCache, validity: validity)
                    for (index, (candidate, cache)) in zip(candidates, caches).enumerated() {
                        expectClose(
                            try candidate.forward(input, cache: cache, validity: validity),
                            expected, tolerance: f.tolerance(gpu: true),
                            "policy \(index), chunk ending \(end)")
                    }
                    start = end
                }
            }
        }

        /// The native contiguous CBv2 rows (FP32 storage) under
        /// `SparsePrefill.nativeKernel`: ragged prefill, a 12-token step with
        /// ten sparse queries, then decode.
        @Test(arguments: [false, true])
        func nativeSparsePrefillContiguousCBv2MatchesReference(q4: Bool) throws {
            try Device.withDefaultDevice(.gpu) {
                let f = try Fixture(q4: q4)
                var policy = GLM5NextExecutionPolicy.serving
                policy.sparseDecode = .nativeKernels
                policy.sparsePrefill = .nativeKernel
                let reference = try f.target(.reference)
                let candidate = try f.target(policy: policy)
                let referenceRows = try ContiguousRows(model: reference, batch: 2)
                let candidateRows = try ContiguousRows(model: candidate, batch: 2)
                var steps = [
                    (
                        MLXArray([Int32(4), 5, 6, 7, 8, 9, 0, 0, 4, 5, 6, 7], [2, 6]),
                        MLXArray(
                            (0 ..< 12).map { $0 < 6 || $0 >= 8 }, [2, 6])
                    ),
                    (
                        MLXArray((0 ..< 24).map { Int32(1 + ($0 * 7) % 13) }, [2, 12]),
                        MLXArray.ones([2, 12], dtype: .bool)
                    ),
                ]
                for step in 0 ..< 3 {
                    steps.append(
                        (
                            MLXArray([Int32(10 + step % 5), Int32(3 + step % 7)], [2, 1]),
                            MLXArray.ones([2, 1], dtype: .bool)
                        ))
                }
                for (index, (tokens, validity)) in steps.enumerated() {
                    let expected = try referenceRows.forward(tokens, validity: validity)
                    let actual = try candidateRows.forward(tokens, validity: validity)
                    expectClose(
                        actual.logits, expected.logits, tolerance: q4 ? 1.0 / 16 : 1.0 / 256,
                        "candidate step \(index)")
                }
            }
        }

        @Test func nativeSparsePrefillRequiresAbsorbedAttention() throws {
            let f = try Fixture()
            #expect(throws: GLM5NextLoadError.self) {
                try f.attention(policy: .init(sparsePrefill: .nativeKernel))
            }
        }

        @Test func nativeSparseDecodeRequiresAbsorbedAttentionAndRefusesCPU() throws {
            let f = try Fixture()
            #expect(throws: GLM5NextLoadError.self) {
                try f.attention(policy: .init(sparseDecode: .nativeKernels))
            }
            try Device.withDefaultDevice(.cpu) {
                let native = try f.attention(
                    policy: .init(sparseAttention: .absorbed, sparseDecode: .nativeKernels))
                let cache = GLM5NextDSACache()
                // The first selector step already runs the native kernels.
                #expect(throws: GLM5NextSparseDecodeKernel.Failure.unsupportedDeviceOrStream) {
                    try native.forward(
                        f.hidden(batch: 1, positions: 0 ..< 4), cache: cache,
                        validity: MLXArray.ones([1, 4], dtype: .bool))
                }
            }
        }

        /// Query blocks only bound the transient. Forced one-query blocks (the
        /// form every paged source uses) take the multi-block slicing and
        /// concatenation path and must match one block of the whole chunk.
        @Test(arguments: [false, true], [false, true])
        func absorbedQueryBlocksMatchOneBlock(gpu: Bool, q4: Bool) throws {
            try Device.withDefaultDevice(gpu ? .gpu : .cpu) {
                let f = try Fixture(q4: q4)
                let single = try f.attention(.absorbed)
                let blocked = try f.attention(.absorbed, blockBytes: 1)
                #expect(single.absorbedBlockBytes == GLM5NextDSAAttention.absorbedLatentBlockBytes)
                let singleCache = GLM5NextDSACache()
                let blockedCache = GLM5NextDSACache()
                var start = 0
                for end in [9, 20, 21] {
                    let input = f.hidden(batch: 2, positions: start ..< end)
                    let validity = MLXArray(
                        (0 ..< 2).flatMap { row in (start ..< end).map { row == 0 || $0 >= 2 } },
                        [2, end - start])
                    expectClose(
                        try blocked.forward(input, cache: blockedCache, validity: validity),
                        try single.forward(input, cache: singleCache, validity: validity),
                        tolerance: f.tolerance(gpu: gpu) / 8, "blocks, chunk ending \(end)")
                    start = end
                }
            }
        }

        @Test func pagedStageForwardMatchesExpandedReference() throws {
            try Device.withDefaultDevice(.gpu) {
                let f = try Fixture()
                let admission = AdmissionV2(
                    layerKinds: [], bytesCapacity: 64 << 20, config: .init(watermarkFraction: 0))
                let store = try GLM5NextPagedSparseStore(
                    geometry: GLM5NextSparseStateGeometry(
                        maximumTokens: 64, maximumStagedTokens: 64,
                        indexDimension: f.text.indexHeadDim, latentDimension: f.text.kvLoraRank,
                        indexDType: .float32, latentDType: .float32),
                    admission: admission, capacityBytes: 8 << 20, segmentBytes: 64 << 10,
                    maximumBufferBytes: 1 << 20)
                let reference = try f.attention(.reference)
                let absorbed = try f.attention(.absorbed)
                // One batched block per chunk, against the paged per-query blocks.
                let contiguous = try f.attention(.absorbed)
                let contiguousCache = GLM5NextDSACache()
                func bind(_ attention: GLM5NextDSAAttention) throws -> (
                    GLM5NextSparseStateBinding, GLM5NextPagedSparseState
                ) {
                    let binding = GLM5NextSparseStateBinding(
                        modelIdentity: ObjectIdentifier(attention), cacheSalt: "absorbed-parity",
                        modelLayerIndex: 3)
                    return (binding, try store.makeState(binding: binding))
                }
                let dense = try f.attention(.absorbedDenseHistory)
                // Native selection over the paged index view; paged latent
                // reads keep the absorbed gathers.
                let native = try f.attention(
                    policy: .init(sparseAttention: .absorbed, sparseDecode: .nativeKernels))
                let (referenceBinding, referenceState) = try bind(reference)
                let (absorbedBinding, absorbedState) = try bind(absorbed)
                let (denseBinding, denseState) = try bind(dense)
                let (nativeBinding, nativeState) = try bind(native)
                func stage(
                    _ attention: GLM5NextDSAAttention, _ state: GLM5NextPagedSparseState,
                    _ binding: GLM5NextSparseStateBinding, _ input: MLXArray
                ) throws -> MLXArray {
                    let transaction = try state.begin(at: state.committedTokenCount)
                    let staged = try attention.stageForward(
                        input, state: state, transaction: transaction, binding: binding,
                        validity: MLXArray.ones([1, input.dim(1)], dtype: .bool))
                    try staged.withEvaluationLease { roots in
                        try withError { error in
                            eval(roots)
                            try error.check()
                        }
                    }
                    try state.commit(transaction, accepting: input.dim(1))
                    return staged.values
                }
                var start = 0
                for end in [3, 4, 5, 9, 10, 17, 18] {
                    let input = f.hidden(batch: 1, positions: start ..< end)
                    let paged = try stage(absorbed, absorbedState, absorbedBinding, input)
                    expectClose(
                        paged, try stage(reference, referenceState, referenceBinding, input),
                        tolerance: f.tolerance(gpu: true), "paged chunk ending \(end)")
                    // Queries below the 8-token budget take the dense prefix,
                    // later ones the selector, inside one staged step.
                    expectClose(
                        try stage(dense, denseState, denseBinding, input), paged,
                        tolerance: f.tolerance(gpu: true) / 8, "paged dense, chunk ending \(end)")
                    expectClose(
                        paged,
                        try contiguous.forward(
                            input, cache: contiguousCache,
                            validity: MLXArray.ones([1, input.dim(1)], dtype: .bool)),
                        tolerance: f.tolerance(gpu: true) / 8,
                        "paged blocks, chunk ending \(end)")
                    expectClose(
                        try stage(native, nativeState, nativeBinding, input), paged,
                        tolerance: f.tolerance(gpu: true), "paged native, chunk ending \(end)")
                    start = end
                }
                try referenceState.close()
                try absorbedState.close()
                try denseState.close()
                try nativeState.close()
            }
        }

        private final class ContiguousRows {
            let model: GLM5NextTextModel
            let backend: CBv2ContiguousKVBackend
            let bank: CBv2LayerCacheBank
            let rows: [[CBv2SequenceKV?]]
            let recurrent: [CBv2RecurrentRequestState]
            init(model: GLM5NextTextModel, batch: Int) throws {
                let backend = CBv2ContiguousKVBackend(
                    config: .init(bytesCapacity: 16 << 20, kvDType: .float32))
                self.model = model
                self.backend = backend
                rows = try (0 ..< batch).map { _ in
                    try backend.makeSequenceState(
                        layerKinds: model.cbv2LayerKinds, promptLength: 0, maxLength: 64)
                }
                bank = CBv2LayerCacheBank(layerKinds: model.cbv2LayerKinds)
                recurrent = try (0 ..< batch).map { _ in
                    try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
                }
            }
            func forward(_ tokens: MLXArray, validity: MLXArray) throws -> GLM5NextTextOutput {
                let evaluations = try recurrent.map { try $0.bind() }
                let caches = bank.layerCaches(rowStates: rows).map { $0 as! any KVCache }
                let result = try model.prepareCBv2Forward(
                    tokens: tokens, caches: caches, recurrentState: evaluations,
                    validity: validity)
                let roots =
                    try evaluations.flatMap { try $0.evaluate() }
                    + caches.flatMap { $0.innerState() }
                try withError { fault in
                    eval(roots + [result.logits, result.hidden])
                    StreamOrDevice.default.stream.synchronize()
                    try fault.check()
                }
                for evaluation in evaluations { try evaluation.commit() }
                return result
            }
            deinit {
                bank.releaseBoundRows()
                for row in rows { backend.release(row) }
                for state in recurrent { try? state.release() }
            }
        }

        @Test func nativeContiguousCBv2AndGraphForwardMatchReferenceTarget() throws {
            try Device.withDefaultDevice(.cpu) {
                let f = try Fixture()
                let reference = try f.target(.reference)
                let absorbed = try f.target(.absorbed)
                let dense = try f.target(.absorbedDenseHistory)
                let referenceRows = try ContiguousRows(model: reference, batch: 2)
                let absorbedRows = try ContiguousRows(model: absorbed, batch: 2)
                let denseRows = try ContiguousRows(model: dense, batch: 2)
                // Ragged B2 prefill (row 1 left-padded), then batched decode past the budget.
                var steps = [
                    (
                        MLXArray([Int32(4), 5, 6, 7, 8, 9, 0, 0, 4, 5, 6, 7], [2, 6]),
                        MLXArray(
                            [
                                true, true, true, true, true, true, false, false, true, true, true,
                                true,
                            ],
                            [2, 6])
                    )
                ]
                for step in 0 ..< 8 {
                    steps.append(
                        (
                            MLXArray([Int32(10 + step % 5), Int32(3 + step % 7)], [2, 1]),
                            MLXArray.ones([2, 1], dtype: .bool)
                        ))
                }
                for (index, (tokens, validity)) in steps.enumerated() {
                    let expected = try referenceRows.forward(tokens, validity: validity)
                    let actual = try absorbedRows.forward(tokens, validity: validity)
                    expectClose(actual.logits, expected.logits, tolerance: 1e-4, "step \(index)")
                    expectClose(actual.hidden, expected.hidden, tolerance: 1e-4, "step \(index)")
                    let denseActual = try denseRows.forward(tokens, validity: validity)
                    expectClose(
                        denseActual.logits, expected.logits, tolerance: 1e-4, "dense step \(index)")
                }
                let tokens = MLXArray([Int32(2), 5, 3, 9, 7, 1, 4, 8, 6, 2, 11], [1, 11])
                expectClose(
                    try absorbed.forward(tokens: tokens, cache: nil).logits,
                    try reference.forward(tokens: tokens, cache: nil).logits, tolerance: 1e-4,
                    "graph forward")
                expectClose(
                    try dense.forward(tokens: tokens, cache: nil).logits,
                    try reference.forward(tokens: tokens, cache: nil).logits, tolerance: 1e-4,
                    "dense graph forward")
            }
        }

        /// The serving path (native contiguous CBv2 rows) under every default-off
        /// candidate: ragged prefill, multi-token verify-sized steps and decode.
        @Test(arguments: [false, true])
        func nativeContiguousCBv2UnderEveryCandidateMatchesReference(q4: Bool) throws {
            try Device.withDefaultDevice(.gpu) {
                let f = try Fixture(q4: q4)
                var fastest = GLM5NextExecutionPolicy.serving
                fastest.sparseAttention = .absorbedDenseHistory
                fastest.linearAttention = .nativeRecurrenceChunkedPrefill
                fastest.normalization = .fusedKernel
                fastest.linearDecodeStep = .fusedKernel
                fastest.routing = .nativeKernel
                fastest.hyperConnection = .nativeNormFused
                fastest.projections = .fusedInputs
                fastest.stepEvaluation = .earlyAsync
                fastest.sparseDecode = .nativeKernels
                let reference = try f.target(.reference)
                let candidate = try f.target(policy: fastest)
                let referenceRows = try ContiguousRows(model: reference, batch: 2)
                let candidateRows = try ContiguousRows(model: candidate, batch: 2)
                var steps = [
                    (
                        MLXArray([Int32(4), 5, 6, 7, 8, 9, 0, 0, 4, 5, 6, 7], [2, 6]),
                        MLXArray(
                            [
                                true, true, true, true, true, true, false, false, true, true, true,
                                true,
                            ],
                            [2, 6])
                    ),
                    (MLXArray([Int32(3), 9, 2, 7, 5, 1], [2, 3]), MLXArray.ones([2, 3], dtype: .bool)),
                ]
                for step in 0 ..< 6 {
                    steps.append(
                        (
                            MLXArray([Int32(10 + step % 5), Int32(3 + step % 7)], [2, 1]),
                            MLXArray.ones([2, 1], dtype: .bool)
                        ))
                }
                for (index, (tokens, validity)) in steps.enumerated() {
                    let expected = try referenceRows.forward(tokens, validity: validity)
                    let actual = try candidateRows.forward(tokens, validity: validity)
                    expectClose(
                        actual.logits, expected.logits, tolerance: q4 ? 1.0 / 16 : 1.0 / 256,
                        "candidate step \(index)")
                }
            }
        }

        private func greedy(_ model: GLM5NextTextModel, prompt: [Int], count: Int) throws -> [Int] {
            let cache = model.newTextCache()
            var input = prompt
            var tokens: [Int] = []
            for _ in 0 ..< count {
                let output = try model.forward(
                    tokens: MLXArray(input.map(Int32.init), [1, input.count]), cache: cache)
                let next = output.logits[0, -1].argMax().item(Int.self)
                tokens.append(next)
                input = [next]
            }
            return tokens
        }

        /// Returns the engine tokens and the paged binding's selected-row count.
        private func pagedGreedy(_ model: GLM5NextTextModel, prompt: [Int], count: Int)
            async throws -> (tokens: [Int], selectedRows: Int)
        {
            let backend = try model.makeCBv2PagedBackend(
                kvBytesCapacity: 128 << 20, maximumBatchSize: 1, maximumQueryTokens: 16,
                maximumContextTokens: 128, maximumScratchBytes: 256 << 20,
                segmentBytes: 64 << 10, maximumBufferBytes: 32 << 20)
            let process = GLM5NextTestProcessBudget()
            let engine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(model),
                layerKinds: model.cbv2LayerKinds, backend: backend,
                cacheProvider: CBv2LayerCacheBank(caches: backend.makeLayerCaches()),
                sampler: CBv2GreedySampler(),
                schedulerConfig: .init(
                    maxConcurrentRequests: 1, maxBatchedTokensPerStep: 16,
                    prefillChunkSize: 4, maxWaiting: 8, enablePrefixCache: false),
                admissionConfig: model.cbv2NativeAdmissionConfig, processMemoryOwner: process)
            let stream = try engine.submit(
                .init(
                    id: .init(1301), promptTokens: prompt, sampling: .init(temperature: 0),
                    maxTokens: count))
            var tokens: [Int] = []
            for await event in stream {
                if case .delta(_, let values, _) = event { tokens += values }
            }
            await engine.shutdown()
            #expect(backend.bytesInUse == 0 && backend.bytesReserved == 0 && process.bytes == 0)
            #expect(engine.loopForTesting.recurrentStates.isEmpty)
            return (tokens, backend.glm5NextModelBinding?.observedSelectedRows ?? 0)
        }

        @Test func pagedEngineServesAbsorbedTargetAndReadsOnlyLiveColumns() async throws {
            let f = try Fixture()
            let prompt = [4, 5, 6, 7, 8, 9, 10]
            let reference = try f.target(.reference)
            let absorbed = try f.target(.absorbed)
            let referenceRun = try await pagedGreedy(reference, prompt: prompt, count: 4)
            let absorbedRun = try await pagedGreedy(absorbed, prompt: prompt, count: 4)
            #expect(referenceRun.tokens == (try greedy(reference, prompt: prompt, count: 4)))
            #expect(absorbedRun.tokens == (try greedy(absorbed, prompt: prompt, count: 4)))
            // The reference reads full budget rows. With an identical schedule the
            // absorbed path skips the first chunk's statically padded columns.
            let width = f.text.indexTopK + 3
            #expect(referenceRun.selectedRows > 0 && referenceRun.selectedRows % width == 0)
            #expect(
                absorbedRun.selectedRows > 0 && absorbedRun.selectedRows < referenceRun.selectedRows
            )
            // Dense-prefix queries read the first budget latents once per step;
            // the greedy tokens follow the contiguous dense-history target.
            let dense = try f.target(.absorbedDenseHistory)
            let denseRun = try await pagedGreedy(dense, prompt: prompt, count: 4)
            #expect(denseRun.tokens == (try greedy(dense, prompt: prompt, count: 4)))
            #expect(denseRun.selectedRows > 0)
        }

        @Test func mtpLayerFollowsTargetPolicyAndMatchesReferenceDrafts() throws {
            try Device.withDefaultDevice(.cpu) {
                let f = try Fixture()
                let referenceTarget = try f.target(.reference)
                let absorbedTarget = try f.target(.absorbed)
                let referenceMTP = try GLM5NextMTP(target: referenceTarget, tensor: f.tensor)
                let absorbedMTP = try GLM5NextMTP(target: absorbedTarget, tensor: f.tensor)
                #expect(referenceMTP.executionPolicy == referenceTarget.executionPolicy)
                #expect(absorbedMTP.executionPolicy.sparseAttention == .absorbed)
                let prompt = MLXArray((0 ..< 10).map { Int32(3 + $0 % 9) }, [1, 10])
                // One trusted target boundary feeds both predictors, isolating MTP.
                let hidden = try referenceTarget.forward(tokens: prompt, cache: nil).hidden
                let referenceState = try referenceMTP.newRequestState()
                let absorbedState = try absorbedMTP.newRequestState()
                let carry = MLXArray([Int32(7)], [1, 1])
                let expected = try referenceMTP.primePrompt(
                    tokens: prompt, targetHidden: hidden, carryToken: carry, state: referenceState)
                let actual = try absorbedMTP.primePrompt(
                    tokens: prompt, targetHidden: hidden, carryToken: carry, state: absorbedState)
                expectClose(actual.hidden, expected.hidden, tolerance: 1e-4, "MTP prime hidden")
                expectClose(actual.logits, expected.logits, tolerance: 1e-4, "MTP prime logits")
                for step in 0 ..< 3 {
                    let token = MLXArray([Int32(5 + step)], [1, 1])
                    let row = hidden[0..., step ..< step + 1]
                    try referenceMTP.beginRound(state: referenceState)
                    try absorbedMTP.beginRound(state: absorbedState)
                    let expected = try referenceMTP.draftStep(
                        shiftedToken: token, targetHidden: row, state: referenceState)
                    let actual = try absorbedMTP.draftStep(
                        shiftedToken: token, targetHidden: row, state: absorbedState)
                    expectClose(actual.logits, expected.logits, tolerance: 1e-4, "draft \(step)")
                    try referenceMTP.commitRound(state: referenceState, acceptingInputs: 1)
                    try absorbedMTP.commitRound(state: absorbedState, acceptingInputs: 1)
                }
                #expect(absorbedState.committedInputCount == 13)
            }
        }

        @Test func mtpRefusesAttentionPolicyDifferentFromItsTarget() throws {
            try Device.withDefaultDevice(.cpu) {
                let f = try Fixture()
                let target = try f.target(.reference)
                let loader = GLM5NextTensorLoader(f.inventory, f.tensor)
                let p = "model.language_model.layers.\(f.text.hiddenLayers)"
                #expect(
                    throws: GLM5NextMTPError.invalid("MTP execution policy must match the target")
                ) {
                    try GLM5NextMTP(
                        target: target, embeddingNorm: loader.array(p + ".enorm.weight"),
                        hiddenNorm: loader.array(p + ".hnorm.weight"),
                        projection: loader.linear(p + ".eh_proj"),
                        attention: GLM5NextDSAAttention(
                            f.text, prefix: p + ".self_attn", loader: loader,
                            executionPolicy: .init(sparseAttention: .absorbed)),
                        inputNorm: loader.array(p + ".input_layernorm.weight"),
                        postNorm: loader.array(p + ".post_attention_layernorm.weight"),
                        mlp: loader.moe(f.text, prefix: p + ".mlp", materialize: nil),
                        outputNorm: loader.array(p + ".shared_head.norm.weight"))
                }
            }
        }

        @Test func defaultIsReferenceAndAbsorbedKeepsTheCheckpointParameterTree() throws {
            try Device.withDefaultDevice(.cpu) {
                let f = try Fixture()
                let model = try GLM5NextTextModel(
                    configuration: f.text, inventory: f.inventory, tensor: f.tensor)
                let sparse = model.model.layers.compactMap(\.sparseAttention)
                #expect(model.executionPolicy == .reference)
                #expect(!sparse.isEmpty)
                #expect(
                    sparse.allSatisfy { $0.executionPolicy == .reference && $0.absorbed == nil })
                let absorbed = try f.target(.absorbed)
                #expect(
                    absorbed.model.layers.compactMap(\.sparseAttention).allSatisfy {
                        $0.executionPolicy == absorbed.executionPolicy && $0.absorbed != nil
                    })
                func tree(_ model: GLM5NextTextModel) -> [String: [Int]] {
                    Dictionary(
                        uniqueKeysWithValues: model.parameters().flattened().map {
                            ($0.0, $0.1.shape)
                        })
                }
                #expect(tree(absorbed) == tree(model))
            }
        }

        /// Every absorbed operand is a view sharing kv_b's buffers, and together the
        /// views cover exactly kv_b's bytes: weight reads per token are unchanged.
        @Test(arguments: [false, true])
        func absorbedOperandsAreViewsSharingKVB(q4: Bool) throws {
            try Device.withDefaultDevice(.cpu) {
                let f = try Fixture(q4: q4)
                let attention = try f.attention(.absorbed)
                let projections = try #require(attention.absorbed)
                let kvB = attention.kvB
                var sources: [MLXArray] = []
                var pairs: [(view: MLXArray, source: MLXArray)] = []
                switch projections.weights {
                case .quantized(let key, let value, let groupSize, let bits):
                    let packed = try #require(kvB as? QuantizedLinear)
                    let biases = try #require(packed.biases)
                    #expect(q4 && groupSize == 64 && bits == 4)
                    sources = [packed.weight, packed.scales, biases]
                    pairs = [
                        (key.words, packed.weight), (key.scales, packed.scales),
                        (key.biases, biases), (value.words, packed.weight),
                        (value.scales, packed.scales), (value.biases, biases),
                    ]
                case .dense(let key, let value):
                    #expect(!q4)
                    sources = [kvB.weight]
                    pairs = [(key, kvB.weight), (value, kvB.weight)]
                }
                #expect(
                    pairs.map(\.view.nbytes).reduce(0, +) == sources.map(\.nbytes).reduce(0, +))
                for (view, source) in pairs {
                    let viewBuffer = try #require(try view.evaluatedBufferInfo())
                    let sourceBuffer = try #require(try source.evaluatedBufferInfo())
                    #expect(viewBuffer.allocatedBytes == sourceBuffer.allocatedBytes)
                    #expect(!viewBuffer.isUnique)
                }
            }
        }

        /// Pins the packed-layout reasoning on the actual kernels. transpose:false
        /// contracts each head's Dq key rows and expands the packed C axis;
        /// transpose:true contracts C against each head's Dv value rows. One
        /// query uses qvm/qmv; 33 queries use qmm_n/qmm (or qmm_nax), which
        /// dequantize weights into BF16 tiles. A wrong axis, group or head
        /// stride produces errors of the order of the peak.
        @Test(arguments: [false, true], [1, 33])
        func q4ProjectionKernelsMatchScalarOracle(gpu: Bool, queries: Int) throws {
            try Device.withDefaultDevice(gpu ? .gpu : .cpu) {
                let f = try Fixture(q4: true)
                let c = f.text
                let projections = try #require(try f.attention(.absorbed).absorbed)
                let p = "model.language_model.layers.3.self_attn.kv_b_proj"
                let words = try f.tensor(p + ".weight").asArray(UInt32.self)
                let scales = try f.tensor(p + ".scales").asType(.float32).asArray(Float.self)
                let biases = try f.tensor(p + ".biases").asType(.float32).asArray(Float.self)
                let heads = c.attentionHeads
                let (keyWidth, valueWidth, latent) = (c.qkNopeHeadDim, c.vHeadDim, c.kvLoraRank)
                func weight(_ row: Int, _ column: Int) -> Double {
                    let index = row * latent + column
                    let nibble = (words[index / 8] >> UInt32(4 * (index % 8))) & 0xF
                    return Double(scales[index / 64]) * Double(nibble) + Double(biases[index / 64])
                }
                let q = MLXArray(
                    (0 ..< queries * heads * keyWidth).map { Float(sin(Double($0) * 0.61 + 0.3)) },
                    [1, queries, heads, keyWidth]
                ).asType(.bfloat16)
                let x = MLXArray(
                    (0 ..< heads * queries * latent).map { Float(cos(Double($0) * 0.43)) },
                    [1, heads, queries, latent]
                ).asType(.bfloat16)
                let qValues = q.asType(.float32).asArray(Float.self)
                let xValues = x.asType(.float32).asArray(Float.self)
                /// Exact dot product and the sum of its absolute products.
                func dot(_ terms: [(Double, Double)]) -> OracleValue {
                    terms.reduce(into: OracleValue(exact: 0, magnitude: 0)) {
                        $0.exact += $1.0 * $1.1
                        $0.magnitude += abs($1.0 * $1.1)
                    }
                }
                var keyOracle: [OracleValue] = []
                var valueOracle: [OracleValue] = []
                for head in 0 ..< heads {
                    let base = head * (keyWidth + valueWidth)
                    for query in 0 ..< queries {
                        for column in 0 ..< latent {
                            keyOracle.append(
                                dot(
                                    (0 ..< keyWidth).map {
                                        (
                                            Double(qValues[(query * heads + head) * keyWidth + $0]),
                                            weight(base + $0, column)
                                        )
                                    }))
                        }
                        for row in 0 ..< valueWidth {
                            valueOracle.append(
                                dot(
                                    (0 ..< latent).map {
                                        (
                                            Double(xValues[(head * queries + query) * latent + $0]),
                                            weight(base + keyWidth + row, $0)
                                        )
                                    }))
                        }
                    }
                }
                let absorbedQueries = projections.absorbQueries(q)
                let values = projections.projectValues(x)
                #expect(absorbedQueries.shape == [1, heads, queries, latent])
                #expect(values.shape == [1, heads, queries, valueWidth])
                #expect(absorbedQueries.dtype == .bfloat16 && values.dtype == .bfloat16)
                expectOracle(absorbedQueries, keyOracle, gpu: gpu, "q W_k")
                expectOracle(values, valueOracle, gpu: gpu, "x W_v^T")
            }
        }

        private struct OracleValue {
            var exact: Double
            /// Σ|x_i w_i|, the scale of any per-weight rounding.
            var magnitude: Double
        }

        /// Metal kernels accumulate in FP32 and round the output once to BF16
        /// (unit roundoff 2^-8, so at most 2^-8 |e|). Multi-row kernels also
        /// round each FP32-dequantized weight to BF16, bounding the sum's error
        /// by 2^-8 Σ|x w|. With BF16-exact inputs, 2^-8 (Σ|x w| + |e|) is the
        /// first-order worst-case bound itself, not a margin over it; the
        /// 1e-4 peak term covers FP32 accumulation and second-order terms. Any
        /// observed headroom is a property of this fixture. CPU kernels round
        /// every BF16 operation; there, 2^-4 of the peak only detects layout errors.
        private func expectOracle(
            _ actual: MLXArray, _ oracle: [OracleValue], gpu: Bool, _ context: String,
            sourceLocation: SourceLocation = #_sourceLocation
        ) {
            let values = actual.asType(.float32).asArray(Float.self)
            #expect(values.count == oracle.count, "\(context)", sourceLocation: sourceLocation)
            let peak = oracle.map { abs($0.exact) }.max() ?? 0
            for (index, (value, expected)) in zip(values, oracle).enumerated() {
                let bound =
                    gpu
                    ? (expected.magnitude + abs(expected.exact)) / 256 + peak * 1e-4 : peak / 16
                guard abs(Double(value) - expected.exact) <= bound else {
                    Issue.record(
                        "\(context)[\(index)]: \(value) versus \(expected.exact), bound \(bound)",
                        sourceLocation: sourceLocation)
                    return
                }
            }
        }

        /// Every opt-in is quoted identically to the reference: no policy adds
        /// resident or load bytes. The loader's policy reaches every target
        /// layer and the MTP layer.
        @Test func artifactLoaderPoliciesAddNoLoadBytesAndReachMTP() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try GLM5NextLocalArtifactFixture(q4: true)
                let identity = try fixture.identity()
                func loader(_ policy: GLM5NextExecutionPolicy) throws -> GLM5NextArtifactLoader {
                    try GLM5NextArtifactLoader(
                        directory: fixture.directory, expectedIdentity: identity,
                        storage: fixture.storage, executionPolicy: policy)
                }
                let reference = try GLM5NextArtifactLoader(
                    directory: fixture.directory, expectedIdentity: identity,
                    storage: fixture.storage)
                let everyOptIn = GLM5NextExecutionPolicy(
                    sparseAttention: .absorbed,
                    sparseSelection: .init(selection: .parallelThreshold, deviceRanges: true),
                    pagedMetadataReads: .deviceTable, hyperConnection: .nativeRequired,
                    routedExperts: .nativeCombineSortedPrefill,
                    linearAttention: .nativeRecurrenceChunkedPrefill)
                #expect(reference.executionPolicy == .reference)
                for policy in [GLM5NextExecutionPolicy(sparseAttention: .absorbed), everyOptIn] {
                    let other = try loader(policy)
                    #expect(other.executionPolicy == policy)
                    let (lhs, rhs) = (reference.resources, other.resources)
                    #expect(lhs.conservativeLoadPeakBytes == rhs.conservativeLoadPeakBytes)
                    #expect(lhs.additionalLoadTransientBytes == rhs.additionalLoadTransientBytes)
                    #expect(lhs.textBytes == rhs.textBytes && lhs.mtpBytes == rhs.mtpBytes)
                }
                let absorbed = try loader(.init(sparseAttention: .absorbed))
                let target = try absorbed.loadTextModel()
                #expect(target.executionPolicy == absorbed.executionPolicy)
                #expect(
                    target.model.layers.allSatisfy {
                        $0.executionPolicy == absorbed.executionPolicy
                            && $0.sparseAttention.map { $0.absorbed != nil } ?? true
                    })
                let mtp = try absorbed.loadMTP(target: target)
                #expect(mtp.executionPolicy == absorbed.executionPolicy)
                let logits = try target.forward(
                    tokens: MLXArray([Int32(2), 3, 4, 5, 6], [1, 5]), cache: nil
                ).logits
                #expect(logits.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite })
            }
        }

        @Test func selectionReportsOnlyStaticallyInvalidBudgetColumns() throws {
            try Device.withDefaultDevice(.cpu) {
                for (history, budget, padding) in [
                    (3, 8, 4 ..< 8), (5, 16, 8 ..< 16), (9, 8, 8 ..< 8), (13, 8, 8 ..< 8),
                ] {
                    let selection = try GLM5NextSparseAttention.selectTokens(
                        indexQueries: MLXArray(
                            (0 ..< history * 2).map { Float(sin(Double($0))) }, [1, history, 1, 2]),
                        indexKeys: MLXArray(
                            (0 ..< history * 2).map { Float(cos(Double($0) * 0.3)) },
                            [1, history, 2]),
                        poolGates: MLXArray.zeros([1, history, 2]),
                        poolPositionBias: MLXArray.zeros([4, 2]),
                        headWeights: MLXArray.ones([1, history, 1]),
                        keyValidity: MLXArray.ones([1, history], dtype: .bool),
                        queryValidity: MLXArray.ones([1, history], dtype: .bool),
                        queryStart: 0, tokenBudget: budget)
                    #expect(selection.staticPaddingColumns == padding)
                    let width = budget + 3
                    let indices = selection.indices.asArray(Int32.self)
                    for query in 0 ..< history {
                        #expect(padding.allSatisfy { indices[query * width + $0] == -1 })
                    }
                }
            }
        }
    }
}
