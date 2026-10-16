import Foundation
import MLX
import MLXNN

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Tiny latent-MoE Super trunk with every checkpoint-quantized slot installed
/// as ModelOpt: FP8 Mamba/attention projections and output head, NVFP4 shared
/// and routed experts. Latent projections and the router stay BF16 `Linear`
/// and matmul, as in the checkpoint. Synthetic weights only.
enum NemotronSuperBatchFixture {
    static let vocabulary = 96
    static let imageToken: Int32 = 18

    static func configuration() -> NemotronHConfiguration {
        NemotronHConfiguration(
            vocabSize: vocabulary, hiddenSize: 256, numHiddenLayers: 5,
            numAttentionHeads: 4, numKeyValueHeads: 2, mambaNumHeads: 8, mambaHeadDim: 32,
            ssmStateSize: 16, convKernel: 4, nGroups: 2, intermediateSize: 128,
            moeIntermediateSize: 96, moeSharedExpertIntermediateSize: 192,
            nRoutedExperts: 16, numExpertsPerTok: 6, hybridOverridePattern: "ME*ME", headDim: 64,
            nSharedExperts: 1, routedScalingFactor: 5, moeLatentSize: 64)
    }

    /// Graph execution in both settings; only the product shapes differ.
    enum Products {
        /// `.rowwise` accumulation and `.exactReference` FP8 products.
        case exactSettings
        /// `.tiled` accumulation and `.packed` FP8 products.
        case packedTiled
    }

    static func model(seed: UInt64, dtype: DType, products: Products) throws -> NemotronSuperModel {
        let language = NemotronHModel(configuration())
        var fixtures = ModelOptGraphFixtures(seed: seed)
        var dense: [String: MLXArray] = [:]
        for (name, value) in language.parameters().flattened() {
            if name.hasSuffix("A_log") || name.hasSuffix("e_score_correction_bias") {
                dense[name] = value
                continue
            }
            let magnitude: Float = name.contains("norm") ? 0.5 : 0.08
            var random = MLXArray(fixtures.values(value.size, magnitude: magnitude))
                .reshaped(value.shape)
            if name.contains("norm") { random = random + 1 }
            dense[name] = random.asType(dtype)
        }
        try language.update(parameters: .unflattened(dense), verify: .all)
        var projections: [String: ModelOptCalibratedLinear] = [:]
        var experts: [String: ModelOptNVFP4SwitchLinear] = [:]
        let fp8Slots = ["in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"]
        for (path, module) in language.leafModules().flattened() {
            if let linear = module as? Linear, type(of: linear) == Linear.self {
                let (outputs, inputs) = linear.shape
                if path == "lm_head" || fp8Slots.contains(where: { path.hasSuffix("." + $0) }) {
                    projections[path] = try fixtures.fp8Projection(
                        outputs: outputs, inputs: inputs, magnitude: 64, dtype: dtype)
                } else if path.contains(".shared_experts.") {
                    projections[path] = try fixtures.nvfp4Projection(
                        outputs: outputs, inputs: inputs, magnitude: 64, dtype: dtype)
                }
            }
            if module is SwitchLinear,
                let shape = Dictionary(uniqueKeysWithValues: module.parameters().flattened())[
                    "weight"]?.shape
            {
                experts[path] = try fixtures.expertBank(
                    experts: shape[0], outputs: shape[1], inputs: shape[2], magnitude: 64,
                    dtype: dtype)
            }
        }
        try language.installModelOptProjections(projections)
        try language.installModelOptExperts(experts)
        language.setModelOptExecution(.graph)
        switch products {
        case .exactSettings:
            language.setModelOptAccumulation(.rowwise)
            language.setModelOptFP8Product(.exactReference)
        case .packedTiled:
            language.setModelOptAccumulation(.tiled)
            language.setModelOptFP8Product(.packed)
        }
        eval(language.parameters())
        let width = configuration().hiddenSize
        let images = NemotronSuperImageComposition(
            tower: NemotronRadio(
                NemotronRadioDimensions(
                    width: 2, layers: 1, heads: 1,
                    channels: 1, patch: 2, positionSide: 2, prefix: 1, mlpWidth: 3)),
            projector: NemotronSuperVisionProjector(
                visionWidth: 2, hiddenWidth: 3, outputWidth: width),
            width: width, imageTokenID: imageToken)
        return try NemotronSuperModel(languageModel: language, imageComposition: images)
    }

    /// Deterministic token streams that avoid the image placeholder.
    static func tokens(_ count: Int, seed: Int) -> [Int32] {
        (0 ..< count).map { index in
            let value = Int32((index * 37 + seed * 11 + 5) % vocabulary)
            return value == imageToken ? value + 1 : value
        }
    }
}

// MARK: - Row-exact products (test-only qualification tool)

/// Product slots whose matrix kernels see every decode row of a step at once.
struct NemotronSuperRowProducts: OptionSet {
    let rawValue: Int
    /// FP8 Mamba/attention projections and output head.
    static let fp8 = Self(rawValue: 1)
    /// NVFP4 shared expert projections.
    static let nvfp4Dense = Self(rawValue: 2)
    /// BF16 latent projections.
    static let latent = Self(rawValue: 4)
    /// BF16 router matmul.
    static let router = Self(rawValue: 8)
    static let all: Self = [.fp8, .nvfp4Dense, .latent, .router]
}

/// Runs `body` on every batch row alone (matrix M = 1, the serial decode
/// shape) and restacks the rows. A single-row input passes straight through,
/// so prompt chunks and serial decode keep their exact graphs.
func nemotronSuperEachRow(_ x: MLXArray, _ body: (MLXArray) -> MLXArray) -> MLXArray {
    guard x.ndim >= 2, x.dim(0) > 1 else { return body(x) }
    return concatenated((0 ..< x.dim(0)).map { body(x[$0 ..< ($0 + 1)]) }, axis: 0)
}

final class NemotronSuperRowExactProjection: Module, UnaryLayer {
    @ModuleInfo(key: "inner") var inner: ModelOptCalibratedLinear

    init(_ inner: ModelOptCalibratedLinear) {
        _inner.wrappedValue = inner
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        nemotronSuperEachRow(x) { inner($0) }
    }
}

final class NemotronSuperRowExactLinear: Linear {
    init(rowSplitting original: Linear) {
        super.init(weight: original.weight, bias: original.bias)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        nemotronSuperEachRow(x) { super.callAsFunction($0) }
    }
}

final class NemotronSuperRowExactRouter: NemotronHMoEGate {
    init(rowSplitting original: NemotronHMoEGate, configuration: NemotronHConfiguration) throws {
        super.init(configuration)
        try update(parameters: original.parameters(), verify: .all)
    }

    override func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        guard x.dim(0) > 1 else { return super.callAsFunction(x) }
        let rows = (0 ..< x.dim(0)).map { super.callAsFunction(x[$0 ..< ($0 + 1)]) }
        return (
            concatenated(rows.map(\.0), axis: 0), concatenated(rows.map(\.1), axis: 0)
        )
    }
}

extension NemotronHModel {
    /// Replaces the selected product slots with row-at-a-time wrappers that
    /// share the original weights. Test-only; returns the number replaced.
    @discardableResult
    func installRowExactProducts(_ kinds: NemotronSuperRowProducts) throws -> Int {
        let parents = Dictionary(uniqueKeysWithValues: namedModules())
        var replacements: [(Module, String, Module)] = []
        for (path, module) in namedModules() {
            let parts = path.split(separator: ".").map(String.init)
            guard let key = parts.last,
                let parent = parents[parts.dropLast().joined(separator: ".")]
            else { continue }
            if let projection = module as? ModelOptCalibratedLinear,
                !(parent is NemotronSuperRowExactProjection)
            {
                let kind: NemotronSuperRowProducts =
                    projection.format == .fp8 ? .fp8 : .nvfp4Dense
                if kinds.contains(kind) {
                    replacements.append((parent, key, NemotronSuperRowExactProjection(projection)))
                }
            } else if let linear = module as? Linear, type(of: linear) == Linear.self,
                key.hasSuffix("_latent_proj"), kinds.contains(.latent)
            {
                replacements.append(
                    (parent, key, NemotronSuperRowExactLinear(rowSplitting: linear)))
            } else if let router = module as? NemotronHMoEGate,
                type(of: router) == NemotronHMoEGate.self, kinds.contains(.router)
            {
                replacements.append(
                    (
                        parent, key,
                        try NemotronSuperRowExactRouter(
                            rowSplitting: router, configuration: configuration)
                    ))
            }
        }
        for (parent, key, value) in replacements {
            try parent.update(modules: .unflattened([(key, value)]), verify: .none)
        }
        return replacements.count
    }
}

// MARK: - Request lanes

/// One request: its own KV rows and recurrent state, a prompt and the
/// teacher-forced decode inputs. Logits are recorded per decode step.
final class NemotronSuperLane {
    let prompt: [Int32]
    let inputs: [Int32]
    let rows: [CBv2SequenceKV?]
    let state: CBv2RecurrentRequestState
    var decoded = 0
    var logits: [MLXArray] = []

    init(
        prompt: [Int32], inputs: [Int32], rows: [CBv2SequenceKV?], state: CBv2RecurrentRequestState
    ) {
        self.prompt = prompt
        self.inputs = inputs
        self.rows = rows
        self.state = state
    }

    /// Committed recurrent state and, with `kv`, the attention KV, copied
    /// bit-for-bit (`* 1` keeps signed zeros) and evaluated.
    func snapshot(kv: Bool = true) throws -> [MLXArray] {
        guard let committed = state.confirmedStateSnapshot() else { return [] }
        var arrays: [MLXArray] = []
        for spec in state.spec.layers {
            guard let layer = committed[spec.modelLayerIndex], let conv = layer.conv,
                let ssm = layer.ssm
            else { throw NemotronSuperBatchRunner.Failure.missingState }
            arrays += [conv * 1, ssm * 1]
        }
        for row in kv ? rows.compactMap({ $0 }) : [] {
            let cached = row.snapshot()
            arrays += [
                cached.keys[0..., 0..., ..<cached.offset] * 1,
                cached.values[0..., 0..., ..<cached.offset] * 1,
            ]
        }
        eval(arrays)
        return arrays
    }
}

/// Serial prompt prefill and rectangular `[B, 1]` decode through the
/// production adapter, with per-row commit, rollback and release.
final class NemotronSuperBatchRunner {
    enum Failure: Error { case missingState }
    enum Outcome { case commit, rollback }

    let model: NemotronSuperModel
    let adapter: NemotronSuperRecurrentAdapter
    let backend: CBv2ContiguousKVBackend
    let bank: CBv2LayerCacheBank

    init(
        model: NemotronSuperModel, maximumRows: Int, kvDType: DType,
        products: NemotronSuperRecurrentAdapter.DecodeProducts = .rowExact
    ) throws {
        self.model = model
        adapter = try NemotronSuperRecurrentAdapter(
            model, maximumDecodeRows: maximumRows, decodeProducts: products)
        backend = CBv2ContiguousKVBackend(config: .init(bytesCapacity: 64 << 20, kvDType: kvDType))
        bank = CBv2LayerCacheBank(
            caches: model.languageModel.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
    }

    func lane(prompt: [Int32], inputs: [Int32]) throws -> NemotronSuperLane {
        NemotronSuperLane(
            prompt: prompt, inputs: inputs,
            rows: try backend.makeSequenceState(
                layerKinds: model.cbv2LayerKinds, promptLength: prompt.count,
                maxLength: prompt.count + inputs.count + 1),
            state: try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec))
    }

    /// Binds the bank's caches to `lanes`; the bank rebinds on every call.
    func caches(_ lanes: [NemotronSuperLane]) -> [KVCache] {
        bank.layerCaches(rowStates: lanes.map(\.rows)).map { $0 as! KVCache }
    }

    @discardableResult
    func prefill(_ lane: NemotronSuperLane) throws -> MLXArray {
        let cache = caches([lane])
        let binding = try lane.state.bind()
        let logits = adapter.cbv2RecurrentPrefill(
            MLXArray(lane.prompt).reshaped(1, lane.prompt.count), inputEmbedding: nil,
            cache: cache, recurrentState: [binding], positionIds: nil,
            requirement: .lastPositionLogits)
        eval([logits] + (try binding.evaluate()) + cache.flatMap { $0.innerState() })
        try binding.commit()
        return logits
    }

    /// One decode step of `lanes` in row order; logits rows are recorded for
    /// committed lanes. Returns the `[B, vocab]` logits.
    @discardableResult
    func decode(_ lanes: [NemotronSuperLane], outcomes: [Outcome]? = nil) throws -> MLXArray {
        let cache = caches(lanes)
        let bindings = try lanes.map { try $0.state.bind() }
        let tokens = MLXArray(lanes.map { $0.inputs[$0.decoded] }).reshaped(lanes.count, 1)
        let logits = adapter.cbv2Forward(tokens, caches: cache, recurrentState: bindings)[
            0..., -1, 0...]
        var roots = [logits] + cache.flatMap { $0.innerState() }
        for binding in bindings { roots += try binding.evaluate() }
        eval(roots)
        for (row, lane) in lanes.enumerated() {
            switch outcomes?[row] ?? .commit {
            case .commit:
                try bindings[row].commit()
                lane.logits.append(logits[row])
                lane.decoded += 1
            case .rollback:
                try bindings[row].rollback()
            }
        }
        return logits
    }

    func release(_ lane: NemotronSuperLane) throws {
        try lane.state.release()
        backend.release(lane.rows)
    }
}
