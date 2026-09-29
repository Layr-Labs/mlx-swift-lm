import MLX

/// Build-time observation of the tensors the loaded model actually stores.
/// This includes projection/RoPE promotion and owns only a three-token row.
/// No hooks or probe tensors remain on the serving path.
public enum CBv2NativeKVTypeProbe {
    public enum Phase: String, Sendable { case prefill, decode }

    public struct Observation: Sendable, Equatable {
        public let storageIndex: Int
        public let modelLayerIndex: Int
        public let phase: Phase
        public let keysDType: DType
        public let valuesDType: DType
        public let keysShape: [Int]
        public let valuesShape: [Int]
    }

    public struct Result: Sendable, Equatable {
        /// Includes borrowing layers, whose type is copied from their owner.
        public let layerDTypes: [DType]
        /// Only storage-owning layers produce observations, once per phase.
        public let observations: [Observation]
    }

    /// Supply fresh caches built by the model's normal newCacheV2 entry point.
    /// The caller must budget this short load-time forward like other warmup
    /// work and serialize it with model loading. No active engine may own the
    /// supplied caches. The returned value contains no native array aliases.
    public static func run(
        model: any CBv2SteppableModel, layerKinds: [CBv2LayerKind],
        caches: [any CBv2AttendingLayerCache], token: Int32 = 0,
        retaining work: NativeConstructionScope? = nil
    ) throws -> Result {
        if let work {
            return try work.withPhase(.nativeKVProbe) {
                // The tracked baseline is attention-only. Do not claim that
                // arbitrary recurrent transactions' private cleanup is tracked.
                guard !(model is any CBv2RecurrentSteppableModel) else {
                    throw invalid("tracked construction requires an attention-only model")
                }
                try work.retainValue(model)
                try work.retainValue(caches)
                try work.capture(StreamOrDevice.cpu.stream)
                try work.capture(StreamOrDevice.default.stream)
                return try runInScope(
                    model: model, layerKinds: layerKinds,
                    caches: caches, token: token, work: work)
            }
        }
        return try runInScope(
            model: model, layerKinds: layerKinds,
            caches: caches, token: token, work: nil)
    }

    private static func runInScope(
        model: any CBv2SteppableModel, layerKinds: [CBv2LayerKind],
        caches: [any CBv2AttendingLayerCache], token: Int32,
        work: NativeConstructionScope?
    ) throws -> Result {
        guard token >= 0, !layerKinds.isEmpty, caches.count == layerKinds.count,
            layerKinds.allSatisfy({
                $0.kvGeometry != nil && $0.queryHeads > 0
                    && $0.queryHeads.isMultiple(of: $0.kvHeads)
            }),
            zip(caches, layerKinds).allSatisfy({ $0.0.rows.isEmpty && $0.0.kind == $0.1 })
        else { throw invalid("probe requires fresh matching model caches") }
        // Reject invalid ownership before binding rows or calling the model.
        // Attention's nonthrowing borrowing seam has programmer preconditions;
        // this public throwing probe must not defer metadata validation until
        // after that seam has already run (and possibly mutated an owner row).
        for (index, kind) in layerKinds.enumerated() {
            guard let source = kind.sharesKVWithLayer else { continue }
            guard layerKinds.indices.contains(source), layerKinds[source].sharesKVWithLayer == nil,
                layerKinds[source].kvHeads == kind.kvHeads,
                layerKinds[source].headDim == kind.headDim,
                layerKinds[source].valueHeadDim == kind.valueHeadDim,
                layerKinds[source].attention == kind.attention
            else { throw invalid("layer \(index) has an invalid KV owner") }
        }
        let recorders = layerKinds.enumerated().map { index, kind -> RecordingRow? in
            guard kind.sharesKVWithLayer == nil else { return nil }
            return RecordingRow(index: index, kind: kind, captureProjections: work != nil)
        }
        for case let row? in recorders { try work?.retainOwner(row) }
        for index in layerKinds.indices {
            caches[index].setRows(recorders[index].map { [$0] } ?? [])
        }
        var completedForCleanup = work == nil
        defer { if completedForCleanup { for cache in caches { cache.setRows([]) } } }

        let recurrentModel = model as? any CBv2RecurrentSteppableModel
        let recurrent = try recurrentModel?.recurrentStateSpec.map(
            CBv2RecurrentRequestState.init(spec:))
        defer { try? recurrent?.release() }
        do {
            var observations: [Observation] = []
            for phase: Phase in [.prefill, .decode] {
                let count = phase == .prefill ? 2 : 1
                for case let row? in recorders { row.begin(phase: phase, count: count) }
                try withError { error in
                    let tokens = MLXArray(Array(repeating: token, count: count)).reshaped([
                        1, count,
                    ])
                    try work?.retain(tokens)
                    try work?.capture(StreamOrDevice.default.stream)
                    // Model forward may have synchronizing native validation.
                    try work?.willSubmit()
                    try error.check()
                    let transaction = try recurrent?.bind()
                    defer { try? transaction?.rollback() }
                    let output: MLXArray
                    if let transaction, let recurrentModel {
                        output = recurrentModel.forward(
                            tokens: tokens, caches: caches, recurrentState: [transaction])
                    } else {
                        output = model.forward(tokens: tokens, caches: caches)
                    }
                    try work?.retain(output)
                    try work?.checkpoint("probe.forward." + phase.rawValue)
                    try error.check()
                    // Validate before allocating/evaluating the forward's graph.
                    // Shapes and dtypes are graph metadata, not tensor readbacks.
                    for case let row? in recorders { observations.append(try row.observation()) }
                    let recurrentRoots = try transaction?.evaluate() ?? []
                    let roots = recorders.compactMap { $0 }.flatMap { $0.cbv2InnerState() }
                    try work?.retain(arrays: recurrentRoots + roots)
                    eval([output] + recurrentRoots + roots)
                    try work?.checkpoint("probe.evaluated." + phase.rawValue)
                    try error.check()
                    try transaction?.commit()
                }
            }
            var types = [DType?](repeating: nil, count: layerKinds.count)
            for observation in observations {
                let index = observation.storageIndex
                if let previous = types[index], previous != observation.keysDType {
                    throw invalid("layer \(index) changes KV dtype between prefill and decode")
                }
                types[index] = observation.keysDType
            }
            for (index, kind) in layerKinds.enumerated() {
                guard let source = kind.sharesKVWithLayer else { continue }
                types[index] = types[source]
            }
            guard types.allSatisfy({ $0 != nil }) else {
                throw invalid("probe did not observe every KV owner")
            }
            try work?.fence(cause: "native KV probe completion")
            completedForCleanup = true
            return Result(layerDTypes: types.map { $0! }, observations: observations)
        } catch {
            let original = error
            if let work {
                // No unbind on failed completion. The preinstalled scope holds
                // the actual caches, recorders and graph roots after throw.
                if !work.snapshot.isRetainedFault {
                    try work.fence(cause: String(describing: original))
                    completedForCleanup = true
                }
            }
            throw original
        }
    }

    private static func invalid(_ reason: String) -> CBv2KVError {
        .backendIneligible(reason: "native KV probe: " + reason)
    }

    /// The normal attention dispatcher calls this row after all model-side
    /// transforms. A full three-token buffer is also exact for the short
    /// window probe: visibility still comes from the layer's attention kind.
    private final class RecordingRow: CBv2SequenceKV, CBv2InnerStateProviding {
        let index: Int
        let kind: CBv2LayerKind
        let storage: CBv2FullSequenceKV
        private var phase = Phase.prefill
        private var count = 0
        private var observed: Observation?
        private var violation: String?
        private var nativeDType: DType?
        private let captureProjections: Bool
        private var projectionRoots: [MLXArray] = []

        init(index: Int, kind: CBv2LayerKind, captureProjections: Bool) {
            self.captureProjections = captureProjections
            self.index = index
            self.kind = kind
            storage = CBv2FullSequenceKV(
                promptLength: 2, maxLength: 3,
                kvHeads: kind.kvHeads, headDim: kind.headDim, valueHeadDim: kind.valueHeadDim)
        }

        func begin(phase: Phase, count: Int) {
            self.phase = phase
            self.count = count
            observed = nil
            violation = nil
        }

        func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
            // Keep incoming graph roots even if validation/storage subsequently
            // faults before the forward returns its output to the caller.
            if captureProjections { projectionRoots += [keys, values] }
            guard observed == nil, violation == nil else {
                violation = "layer \(index) updated more than once during \(phase.rawValue)"
                return (keys, values)
            }
            let shape = [1, kind.kvHeads, count, kind.headDim]
            let valueShape = [1, kind.kvHeads, count, kind.valueHeadDim]
            guard keys.shape == shape, values.shape == valueShape,
                keys.dtype == values.dtype,
                [.float16, .bfloat16, .float32].contains(keys.dtype)
            else {
                violation =
                    "layer \(index) has incompatible native K/V shape or dtype: \(keys.dtype)/\(values.dtype), \(keys.shape)/\(values.shape)"
                return (keys, values)
            }
            if let nativeDType, nativeDType != keys.dtype {
                violation = "layer \(index) changes KV dtype between prefill and decode"
                return (keys, values)
            }
            nativeDType = keys.dtype
            observed = Observation(
                storageIndex: index, modelLayerIndex: kind.modelLayerIndex ?? index,
                phase: phase, keysDType: keys.dtype, valuesDType: values.dtype,
                keysShape: keys.shape, valuesShape: values.shape)
            return storage.update(keys: keys, values: values)
        }

        func observation() throws -> Observation {
            if let violation { throw invalid(violation) }
            guard let observed else {
                throw invalid("layer \(index) did not update during \(phase.rawValue)")
            }
            return observed
        }

        var absoluteOffset: Int { storage.absoluteOffset }
        var retainedCount: Int { storage.retainedCount }
        var byteCount: Int { storage.byteCount }
        func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) { storage.snapshot() }
        func rollback(_ n: Int) { storage.rollback(n) }
        func cbv2InnerState() -> [MLXArray] { storage.cbv2InnerState() }
    }
}
