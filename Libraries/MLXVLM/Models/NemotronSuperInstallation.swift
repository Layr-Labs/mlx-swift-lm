import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Complete offside installation into an unpublished model. No factory/catalog
/// registration, provider grant, MTP activation or qualification is implied.
public enum NemotronSuperInstallation {
    public enum Failure: Error {
        case unsupportedSemantics(String), inventory(String), shape(String), budget, cancelled
    }
    public struct Receipt {
        public let indexSHA256: String
        public let loadedTensorCount: Int
        public let retainedMTPCount: Int
        public let retainedImageMetadata: [String]
        public let payloadBytes: Int
    }
    /// Ownership stays here until the provider retires GPU work and closes it.
    /// Dropping without close intentionally does not refund the external ledger.
    public final class Installed {
        public private(set) var model: NemotronSuperModel?
        public let receipt: Receipt
        private let reservation: NemotronSuperResidency.Reservation
        private var invalidated = false
        fileprivate init(
            _ model: NemotronSuperModel, receipt: Receipt,
            reservation: NemotronSuperResidency.Reservation
        ) {
            self.model = model
            self.receipt = receipt
            self.reservation = reservation
        }
        /// Native provider setup must use this existing completion owner, not
        /// perform probes outside a retained construction epoch. Host keeps the
        /// scope until restart on fence failure. No admission is granted here.
        public func withNativeSetup<Result>(
            construction: NativeConstructionScope,
            body: (NemotronSuperModel) throws -> Result
        ) throws -> Result {
            guard construction.snapshot.disposition != .active else {
                throw Failure.inventory("exclusive construction epoch required")
            }
            guard let model, !invalidated else { throw Failure.inventory("retired installation") }
            return try construction.withPhase(.nativeSetup) {
                try construction.retainOwner(self)
                try construction.retainOwner(model)
                try construction.invalidateOnFailedCompletion(self) {
                    self.invalidated = true
                    self.model = nil
                }
                try construction.capture(StreamOrDevice.cpu.stream)
                try construction.capture(StreamOrDevice.default.stream)
                try construction.willSubmit()
                let value = try body(model)
                try construction.retainValue(value)
                return value
            }
        }

        /// Must be serialized after all requests/GPU work finish. Do not retain
        /// model aliases beyond this call: the ledger covers this owner only.
        public func closeAfterRetirement() {
            guard !invalidated else { return }
            model = nil
            reservation.release()
        }
    }

    /// Pinned architecture entry. Caller verifies model/config/tokenizer identity
    /// and holds real provider LOAD admission before creating source readers.
    public static func stage(
        configuration: NemotronSuperImageConfiguration,
        source: ModelOptCheckpointSource, residency: NemotronSuperResidency,
        backendAllowanceBytes: Int, construction: NativeConstructionScope
    ) throws -> Installed {
        guard source.indexSHA256 == NemotronSuperCheckpointPin.index,
            source.shardSHA256 == NemotronSuperCheckpointPin.shards
        else {
            throw ModelOptCheckpointSource.Failure.identity
        }
        return try stage(
            source: source, residency: residency, dtype: .bfloat16,
            backendAllowanceBytes: backendAllowanceBytes, construction: construction
        ) {
            try NemotronSuperModel(unloadedConfiguration: configuration)
        }
    }

    // The video tubelet projection is loaded and charged like every other
    // tower weight; only these unused buffers stay retained.
    private static let imageMetadata: Set<String> = [
        "vision_model.radio_model.input_conditioner.norm_mean",
        "vision_model.radio_model.input_conditioner.norm_std", "vision_model.summary_idxs",
    ]

    private struct LoadBudget {
        let payload: Int
        let staging: Int
    }

    private static func budget(
        source: ModelOptCheckpointSource,
        backendAllowanceBytes: Int
    ) throws -> LoadBudget {
        guard backendAllowanceBytes > 0 else { throw Failure.budget }
        func add(_ a: Int, _ b: Int) throws -> Int {
            let (value, overflow) = a.addingReportingOverflow(b)
            guard !overflow else { throw Failure.budget }
            return value
        }
        func multiply(_ a: Int, _ b: Int) throws -> Int {
            let (value, overflow) = a.multipliedReportingOverflow(by: b)
            guard !overflow else { throw Failure.budget }
            return value
        }
        var payload = 0
        var image = 0
        var maximumTensor = 0
        var units: [String: Int] = [:]
        for entry in source.inventory.entries
        where entry.component != .mtp && !imageMetadata.contains(entry.source) {
            guard let tensor = source.tensors[entry.source] else {
                throw Failure.inventory(entry.source)
            }
            let bytes = tensor.layout.byteCount
            payload = try add(payload, bytes)
            maximumTensor = max(maximumTensor, bytes)
            if entry.component != .language {
                image = try add(image, bytes)
                continue
            }
            // One expert bank per up/down projection; individual expert scale
            // sidecars are part of that bank even when split across shards.
            let key: String
            if let range = entry.source.range(of: ".experts.") {
                let suffix = entry.source[range.upperBound...].split(separator: ".")
                guard suffix.count == 3, Int(suffix[0]) != nil,
                    ["up_proj", "down_proj"].contains(String(suffix[1]))
                else {
                    throw Failure.inventory(entry.source)
                }
                key = String(entry.source[..<range.lowerBound]) + ".experts." + suffix[1]
            } else {
                guard let dot = entry.source.lastIndex(of: ".") else {
                    throw Failure.inventory(entry.source)
                }
                key = String(entry.source[..<dot])
            }
            units[key] = try add(units[key, default: 0], bytes)
        }
        guard payload > 0 else { throw Failure.budget }
        // Per-epoch source chunks + concatenation + packed stack; image mapping
        // is completed as one unit, allowing additional transformed storage.
        let transient = max(try multiply(units.values.max() ?? 0, 3), try multiply(image, 4))
        let staging = try add(
            try add(transient, min(16 << 20, maximumTensor)), backendAllowanceBytes)
        return LoadBudget(payload: payload, staging: staging)
    }

    /// CPU-only quote from validated header load units. Resident payload stays
    /// charged across all epochs; completed epochs retire temporary array roots.
    /// Backend/allocator overhead remains an explicit caller-owned allowance.
    public static func requiredLoadBytes(
        source: ModelOptCheckpointSource,
        backendAllowanceBytes: Int
    ) throws -> Int {
        try loadQuote(source: source, backendAllowanceBytes: backendAllowanceBytes).totalBytes
    }

    /// `requiredLoadBytes` in its two parts. `stagingBytes` (per-epoch source
    /// copies of the largest unit or the image mapping, one tensor read and
    /// the backend allowance) is released before `stage` returns; only
    /// `payloadBytes` stays resident, so a later phase of the same load may
    /// reuse the staging bytes once the caller has returned freed buffers.
    public struct LoadQuote: Sendable, Equatable {
        public let payloadBytes: Int
        public let stagingBytes: Int
        public let totalBytes: Int
    }

    public static func loadQuote(
        source: ModelOptCheckpointSource,
        backendAllowanceBytes: Int
    ) throws -> LoadQuote {
        let value = try budget(source: source, backendAllowanceBytes: backendAllowanceBytes)
        let (total, overflow) = value.payload.addingReportingOverflow(value.staging)
        guard !overflow else { throw Failure.budget }
        return LoadQuote(
            payloadBytes: value.payload, stagingBytes: value.staging, totalBytes: total)
    }

    private final class Resources {
        var weights: NemotronSuperResidency.Reservation?
        var staging: NemotronSuperResidency.Reservation?
        var owners: [AnyObject] = []
        func releaseFailure() {
            owners.removeAll()
            weights?.release()
            staging?.release()
        }
    }

    /// Every calibrated language projection and expert bank quantizes its
    /// activations under `activationOverflow`; production uses
    /// ``ModelOptActivationOverflow/nemotronSuper``.
    static func stage(
        source: ModelOptCheckpointSource, residency: NemotronSuperResidency,
        dtype: DType, backendAllowanceBytes: Int, construction: NativeConstructionScope,
        activationOverflow: ModelOptActivationOverflow = .nemotronSuper,
        testingBeforeEpoch: ((Int, NativeConstructionScope) throws -> Void)? = nil,
        makeModel: () throws -> NemotronSuperModel
    ) throws -> Installed {
        // An inner NativeConstructionScope phase does not fence on return.
        // Require our own epoch before granting/refunding installation memory.
        guard construction.snapshot.disposition != .active else {
            throw Failure.inventory("exclusive construction epoch required")
        }
        let resources = Resources()
        do {
            let installed = try stageImpl(
                source: source, residency: residency, dtype: dtype,
                backendAllowanceBytes: backendAllowanceBytes, construction: construction,
                activationOverflow: activationOverflow, resources: resources,
                testingBeforeEpoch: testingBeforeEpoch, makeModel: makeModel)
            resources.owners.removeAll()
            resources.staging?.release()
            return installed
        } catch {
            // A failed completion permanently retains native owners and charges.
            // The host must keep the supplied construction scope until restart.
            if !construction.snapshot.isRetainedFault { resources.releaseFailure() }
            throw error
        }
    }

    // Fixture injection is internal, so a tiny checkpoint cannot advertise the
    // pinned production architecture merely by choosing a permissive factory.
    private static func stageImpl(
        source: ModelOptCheckpointSource, residency: NemotronSuperResidency,
        dtype: DType, backendAllowanceBytes: Int, construction: NativeConstructionScope,
        activationOverflow: ModelOptActivationOverflow, resources: Resources,
        testingBeforeEpoch: ((Int, NativeConstructionScope) throws -> Void)?,
        makeModel: () throws -> NemotronSuperModel
    ) throws -> Installed {
        guard [.float16, .bfloat16, .float32].contains(dtype), backendAllowanceBytes > 0 else {
            throw Failure.budget
        }
        let retained = source.inventory.entries.filter {
            $0.component == .mtp || imageMetadata.contains($0.source)
        }
        let active = source.inventory.entries.filter {
            $0.component != .mtp && !imageMetadata.contains($0.source)
        }
        let quote = try budget(source: source, backendAllowanceBytes: backendAllowanceBytes)
        let payload = quote.payload
        for entry in active
        where entry.source.hasSuffix(".k_scale") || entry.source.hasSuffix(".v_scale") {
            throw Failure.unsupportedSemantics(entry.source)
        }
        let weights = try residency.reserve(.weights, bytes: payload)
        resources.weights = weights
        resources.staging = try residency.reserve(.staging, bytes: quote.staging)
        // No nesting: every return completes captured streams and drops only
        // temporary scope roots. Resources owns all successful resident outputs.
        var epochNumber = 0
        func epoch<T>(_ body: () throws -> T) throws -> T {
            try Task.checkCancellation()
            epochNumber += 1
            try testingBeforeEpoch?(epochNumber, construction)
            return try construction.withPhase(.filesystemLoad) {
                try construction.retainOwner(resources)
                try construction.capture(StreamOrDevice.cpu.stream)
                try construction.capture(StreamOrDevice.default.stream)
                try construction.willSubmit()
                let value = try body()
                try construction.retainValue(value)
                return value
            }
        }
        let candidate = try epoch {
            let value = try makeModel()
            resources.owners.append(value)
            return value
        }
        let language = candidate.languageModel
        // Unloaded random parameters remain lazy and are replaced before eval.
        let cast = language.parameters().flattened().map { name, value in
            (name, language.castPredicate?(name) == false ? value : value.asType(dtype))
        }
        try language.update(parameters: .unflattened(cast), verify: .all)
        let leaves = Dictionary(uniqueKeysWithValues: language.leafModules().flattened())
        var consumed = Set(retained.map(\.source))
        var projections: [String: ModelOptCalibratedLinear] = [:]
        var experts: [String: ModelOptNVFP4SwitchLinear] = [:]
        let quantized = active.filter {
            $0.component == .language && $0.source.hasSuffix(".input_scale")
        }
        var expertRoots = Set<String>()
        for entry in quantized {
            try Task.checkCancellation()
            let prefix = String(entry.source.dropLast(".input_scale".count))
            let local = String(prefix.dropFirst("language_model.".count))
            if let range = local.range(of: ".experts.") {
                expertRoots.insert(String(local[..<range.lowerBound]))
                continue
            }
            guard let original = leaves[local] as? Linear else { throw Failure.inventory(prefix) }
            let group = source.group(prefix)
            projections[local] = try epoch {
                let layer = try ModelOptProjectionLoader.load(
                    sources: group,
                    inputDimensions: original.shape.1, outputDimensions: original.shape.0,
                    weightDType: dtype, maximumBytes: payload, construction: construction,
                    activationOverflow: activationOverflow)
                resources.owners.append(layer)
                try MLX.withError { errors in
                    eval(layer.parameters())
                    try errors.check()
                }
                return layer
            }
            consumed.formUnion(group.values.map(\.name))
        }
        for root in expertRoots.sorted() {
            for (raw, native) in [("up_proj", "fc1"), ("down_proj", "fc2")] {
                let destination = root + ".switch_mlp." + native
                guard let original = leaves[destination] as? SwitchLinear,
                    let shape = Dictionary(uniqueKeysWithValues: original.parameters().flattened())[
                        "weight"]?.shape,
                    shape.count == 3
                else { throw Failure.inventory(destination) }
                let groups = (0 ..< shape[0]).map {
                    source.group("language_model." + root + ".experts.\($0)." + raw)
                }
                experts[destination] = try epoch {
                    let layer = try ModelOptExpertLoader.load(
                        groups: groups, expectedExperts: shape[0],
                        inputDimensions: shape[2], outputDimensions: shape[1], weightDType: dtype,
                        maximumBytes: payload, construction: construction,
                        activationOverflow: activationOverflow)
                    resources.owners.append(layer)
                    return layer
                }
                for group in groups { consumed.formUnion(group.values.map(\.name)) }
            }
        }
        try language.installModelOptProjections(projections)
        try language.installModelOptExperts(experts)
        var mapped: [String: MLXArray] = [:]
        // Installed packed modules already own every one of their parameters.
        for (name, module) in projections.map({ ($0.key, $0.value as Module) })
            + experts.map({ ($0.key, $0.value as Module) })
        {
            for (key, value) in module.parameters().flattened() { mapped[name + "." + key] = value }
        }
        let expected = Dictionary(uniqueKeysWithValues: language.parameters().flattened())
        var imageWeights: [String: MLXArray] = [:]
        for entry in active where !consumed.contains(entry.source) {
            try Task.checkCancellation()
            let tensor = source.tensors[entry.source]!
            guard ["BF16", "F16", "F32"].contains(tensor.layout.dtype) else {
                throw Failure.unsupportedSemantics(entry.source)
            }
            var value = try epoch {
                let value = try tensor.array(maximumBytes: payload, construction: construction)
                // A later epoch failure must retain previously loaded arrays,
                // including those not yet installed into the candidate.
                resources.owners.append(value)
                return value
            }
            if entry.component == .language {
                let destination = entry.destination
                guard let parameter = expected[destination], mapped[destination] == nil else {
                    throw Failure.inventory(destination)
                }
                if destination.hasSuffix("conv1d.weight"), value.shape != parameter.shape,
                    value.ndim == 3
                {
                    value = value.swappedAxes(1, 2)
                }
                // The checkpoint stores A_log in BF16, but the model keeps it in
                // float32 (castPredicate) as the reference computes A_log.float().
                // Only that exact widening is accepted; other mismatches still fail.
                if value.dtype != parameter.dtype, parameter.dtype == .float32,
                    [.bfloat16, .float16].contains(value.dtype),
                    language.castPredicate?(destination) == false
                {
                    value = value.asType(.float32)
                }
                guard value.shape == parameter.shape, value.dtype == parameter.dtype else {
                    throw Failure.shape(destination)
                }
                mapped[destination] = value
            } else {
                imageWeights[entry.source] = value
            }
            consumed.insert(entry.source)
        }
        guard consumed == Set(source.tensors.keys), Set(mapped.keys) == Set(expected.keys) else {
            throw Failure.inventory("complete parameter coverage")
        }
        try epoch {
            try language.update(parameters: .unflattened(mapped), verify: .all)
            try candidate.imageComposition.loadImageWeights(from: imageWeights)
            try MLX.withError { errors in
                eval(candidate.parameters())
                try errors.check()
            }
            try Task.checkCancellation()
        }
        let receipt = Receipt(
            indexSHA256: source.indexSHA256, loadedTensorCount: active.count,
            retainedMTPCount: retained.filter { $0.component == .mtp }.count,
            retainedImageMetadata: retained.filter { $0.component != .mtp }.map(\.source).sorted(),
            payloadBytes: payload)
        let result = Installed(candidate, receipt: receipt, reservation: weights)
        return result
    }
}
