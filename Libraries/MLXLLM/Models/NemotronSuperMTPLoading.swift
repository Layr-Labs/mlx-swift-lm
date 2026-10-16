import MLX
import MLXLMCommon
import MLXNN

extension NemotronSuperMTPModule {
    /// CPU-only conservative array quote: four copies of the
    /// `language_model.mtp.` source payload (read buffers, loaded arrays,
    /// expert-bank stacking and slack). The dense constructor and cast
    /// placeholders are lazy and the loader replaces every one with a loaded
    /// array before its only evaluation (coverage is checked), so they never
    /// allocate; `NemotronSuperMTPLoadBoundTests` checks that none is ever
    /// evaluated. Backend scratch and model/request residency need separate
    /// host allowance. This is not a measured peak or a provider admission grant.
    public static func requiredConstructionBytes(
        configuration c: NemotronHConfiguration, payloadBytes: Int
    ) throws -> Int {
        guard payloadBytes > 0, c.hiddenSize > 0, c.numAttentionHeads > 0,
            c.numKeyValueHeads > 0, (c.moeLatentSize ?? 0) > 0,
            c.moeIntermediateSize > 0, c.moeSharedExpertIntermediateSize > 0,
            c.nRoutedExperts > 0
        else { throw Failure.configuration }
        let (payload, overflow) = payloadBytes.multipliedReportingOverflow(by: 4)
        guard !overflow else { throw Failure.budget }
        return payload
    }

    /// Caller supplies exact model/config/source identity and real LOAD admission
    /// covering the explicit construction allowance plus backend scratch. Must run inside its retained construction
    /// epoch; all allocated owners survive until that caller fences. This loader
    /// neither activates speculation nor qualifies cache rollback or acceptance.
    public static func load(
        configuration: NemotronHConfiguration,
        source: ModelOptCheckpointSource, maximumBytes: Int, constructionAllowanceBytes: Int,
        construction: NativeConstructionScope, cachePolicy: CachePolicy
    ) throws -> NemotronSuperMTPModule {
        try load(
            configuration: configuration, source: source, maximumBytes: maximumBytes,
            constructionAllowanceBytes: constructionAllowanceBytes,
            construction: construction, cachePolicy: cachePolicy, dtype: .bfloat16)
    }

    static func load(
        configuration: NemotronHConfiguration,
        source: ModelOptCheckpointSource, maximumBytes: Int, constructionAllowanceBytes: Int,
        construction: NativeConstructionScope, cachePolicy: CachePolicy, dtype: DType
    ) throws -> NemotronSuperMTPModule {
        guard construction.snapshot.disposition == .active else { throw Failure.construction }
        if case .paged = cachePolicy { throw Failure.cachePolicy }
        guard [.bfloat16, .float32].contains(dtype), maximumBytes > 0 else { throw Failure.budget }
        try construction.capture(StreamOrDevice.cpu.stream)
        try construction.capture(StreamOrDevice.default.stream)
        try construction.willSubmit()
        let prefix = "language_model.mtp."
        let tensors = source.tensors.filter { $0.key.hasPrefix(prefix) }
        var payload = 0
        for tensor in tensors.values {
            let (next, overflow) = payload.addingReportingOverflow(tensor.layout.byteCount)
            guard !overflow, next <= maximumBytes else { throw Failure.budget }
            payload = next
        }
        guard payload > 0 else { throw Failure.inventory("missing MTP tensors") }
        guard
            constructionAllowanceBytes
                >= (try requiredConstructionBytes(
                    configuration: configuration, payloadBytes: payload))
        else {
            throw Failure.budget
        }
        let kName = prefix + "layers.0.mixer.k_proj.k_scale"
        let vName = prefix + "layers.0.mixer.v_proj.v_scale"
        guard let key = tensors[kName], let value = tensors[vName],
            key.layout.shape == [1], value.layout.shape == [1]
        else { throw Failure.inventory("KV calibration") }
        let calibration = try KVCalibration(keyScale: key.scalar(), valueScale: value.scalar())
        let candidate = try NemotronSuperMTPModule(
            configuration: configuration, calibration: calibration, dtype: dtype)
        try construction.retainOwner(candidate)
        let cast = candidate.parameters().flattened().map { name, value in
            (name, name.hasSuffix("e_score_correction_bias") ? value : value.asType(dtype))
        }
        for (_, value) in cast { try construction.retain(value) }
        try candidate.update(parameters: .unflattened(cast), verify: .all)
        // Every placeholder is lazy and replaced below before the only eval.
        try construction.checkpoint("mtpHead.placeholdersBound")
        let leaves = Dictionary(uniqueKeysWithValues: candidate.leafModules().flattened())
        // Validate the exact source namespace before materializing any payload.
        var expectedSources: Set<String> = [kName, vName]
        for (name, parameter) in candidate.parameters().flattened()
        where !name.contains(".switch_mlp.") && !name.contains(".shared_experts.") {
            let sourceName = prefix + name
            guard let tensor = tensors[sourceName], tensor.layout.shape == parameter.shape,
                tensor.layout.dtype == (parameter.dtype == .float32 ? "F32" : "BF16")
            else {
                throw Failure.shape(sourceName)
            }
            expectedSources.insert(sourceName)
        }
        for projection in ["up_proj", "down_proj"] {
            let stems =
                [prefix + "layers.1.mixer.shared_experts." + projection]
                + (0 ..< configuration.nRoutedExperts).map {
                    prefix + "layers.1.mixer.experts.\($0)." + projection
                }
            for stem in stems {
                for suffix in ["weight", "weight_scale", "weight_scale_2", "input_scale"] {
                    expectedSources.insert(stem + "." + suffix)
                }
            }
        }
        guard Set(tensors.keys) == expectedSources else {
            throw Failure.inventory("exact MTP namespace")
        }
        var consumed: Set<String> = [kName, vName]
        var updates: [(String, Module)] = []
        var mapped: [String: MLXArray] = [:]
        let root = "layers.1.mixer"
        for projection in ["up_proj", "down_proj"] {
            let path = root + ".shared_experts." + projection
            guard let original = leaves[path] as? Linear else { throw Failure.inventory(path) }
            let group = source.group(prefix + path)
            let layout = try ModelOptProjectionLayout(
                tensors: group.mapValues(\.layout),
                inputDimensions: original.shape.1, outputDimensions: original.shape.0)
            guard layout.storage == .nvfp4 else { throw Failure.inventory(path) }
            let replacement = try ModelOptProjectionLoader.load(
                sources: group,
                inputDimensions: original.shape.1, outputDimensions: original.shape.0,
                weightDType: dtype, maximumBytes: maximumBytes, construction: construction)
            try construction.retainOwner(replacement)
            updates.append((path, replacement))
            consumed.formUnion(group.values.map(\.name))
        }
        for (raw, native) in [("up_proj", "fc1"), ("down_proj", "fc2")] {
            let path = root + ".switch_mlp." + native
            guard let original = leaves[path] as? SwitchLinear,
                let shape = Dictionary(uniqueKeysWithValues: original.parameters().flattened())[
                    "weight"]?.shape,
                shape.count == 3
            else { throw Failure.inventory(path) }
            let groups = (0 ..< configuration.nRoutedExperts).map {
                source.group(prefix + root + ".experts.\($0)." + raw)
            }
            let replacement = try ModelOptExpertLoader.load(
                groups: groups, expectedExperts: shape[0],
                inputDimensions: shape[2], outputDimensions: shape[1], weightDType: dtype,
                maximumBytes: maximumBytes, construction: construction)
            updates.append((path, replacement))
            for group in groups { consumed.formUnion(group.values.map(\.name)) }
        }
        let parents = Dictionary(uniqueKeysWithValues: candidate.namedModules())
        for (path, replacement) in updates {
            let parts = path.split(separator: ".").map(String.init)
            guard let key = parts.last,
                let parent = parents[parts.dropLast().joined(separator: ".")]
            else { throw Failure.inventory(path) }
            try parent.update(modules: .unflattened([(key, replacement)]), verify: .all)
            for (name, value) in replacement.parameters().flattened() {
                mapped[path + "." + name] = value
            }
        }
        let expected = Dictionary(uniqueKeysWithValues: candidate.parameters().flattened())
        for (name, tensor) in tensors where !consumed.contains(name) {
            try Task.checkCancellation()
            let path = String(name.dropFirst(prefix.count))
            guard let parameter = expected[path], mapped[path] == nil else {
                throw Failure.inventory(name)
            }
            let expectedDType = parameter.dtype == .float32 ? "F32" : "BF16"
            guard tensor.layout.dtype == expectedDType, tensor.layout.shape == parameter.shape
            else { throw Failure.shape(name) }
            mapped[path] = try tensor.array(maximumBytes: maximumBytes, construction: construction)
            consumed.insert(name)
        }
        guard consumed == Set(tensors.keys), Set(mapped.keys) == Set(expected.keys) else {
            throw Failure.inventory("complete MTP coverage")
        }
        try candidate.update(parameters: .unflattened(mapped), verify: .all)
        try MLX.withError { errors in
            eval(candidate.parameters())
            try errors.check()
        }
        try Task.checkCancellation()
        return candidate
    }
}

extension NemotronSuperMTPModule {
    /// LOAD supplement for the embedded head: resident `language_model.mtp.`
    /// payload plus the loader's conservative construction allowance. Header
    /// and configuration arithmetic only; not a measured peak.
    public struct LoadQuote: Sendable, Equatable {
        public let payloadBytes: Int
        public let constructionBytes: Int
    }

    public static func loadQuote(
        configuration: NemotronHConfiguration, source: ModelOptCheckpointSource
    ) throws -> LoadQuote {
        var payload = 0
        for (name, tensor) in source.tensors where name.hasPrefix("language_model.mtp.") {
            let (next, overflow) = payload.addingReportingOverflow(tensor.layout.byteCount)
            guard !overflow else { throw Failure.budget }
            payload = next
        }
        guard payload > 0 else { throw Failure.inventory("missing MTP tensors") }
        return LoadQuote(
            payloadBytes: payload,
            constructionBytes: try requiredConstructionBytes(
                configuration: configuration, payloadBytes: payload))
    }

    /// Loads the unit-scale FP8-KV head for this exact target in its own
    /// exclusive construction epoch; captured streams are fenced on return.
    /// The caller holds LOAD admission for `quote` and keeps a faulted scope.
    public static func load(
        for target: NemotronHModel, source: ModelOptCheckpointSource, quote: LoadQuote,
        construction: NativeConstructionScope
    ) throws -> NemotronSuperMTPModule {
        try load(
            for: target, source: source, quote: quote, construction: construction,
            dtype: .bfloat16)
    }

    static func load(
        for target: NemotronHModel, source: ModelOptCheckpointSource, quote: LoadQuote,
        construction: NativeConstructionScope, dtype: DType
    ) throws -> NemotronSuperMTPModule {
        guard type(of: target) == NemotronHModel.self else { throw Failure.configuration }
        guard construction.snapshot.disposition != .active else { throw Failure.construction }
        guard try loadQuote(configuration: target.configuration, source: source) == quote else {
            throw Failure.budget
        }
        return try construction.withPhase(.filesystemLoad) {
            try load(
                configuration: target.configuration, source: source,
                maximumBytes: quote.payloadBytes,
                constructionAllowanceBytes: quote.constructionBytes,
                construction: construction, cachePolicy: .fp8, dtype: dtype)
        }
    }
}
