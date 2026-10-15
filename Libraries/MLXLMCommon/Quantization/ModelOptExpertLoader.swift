import Foundation
import MLX

/// Explicit ordered expert loading. The caller supplies all expected expert
/// prefixes in routing-ID order; no sorting, omission or inferred expert count.
/// This bounded fixture/staging API rejects groups above 64 MiB packed payload.
public enum ModelOptExpertLoader {
    public enum Failure: Error { case invalidInventory, payloadLimit, invalidScale }
    public struct Source {
        public let file: ModelOptTensorFile
        public let prefix: String
        public init(file: ModelOptTensorFile, prefix: String) {
            self.file = file
            self.prefix = prefix
        }
    }
    public static func load(
        file: ModelOptTensorFile, prefixes: [String], expectedExperts: Int,
        inputDimensions: Int, outputDimensions: Int, weightDType: DType,
        maximumBytes: Int = 64 << 20
    ) throws -> ModelOptNVFP4SwitchLinear {
        try load(
            sources: prefixes.map { Source(file: file, prefix: $0) },
            expectedExperts: expectedExperts,
            inputDimensions: inputDimensions, outputDimensions: outputDimensions,
            weightDType: weightDType, maximumBytes: maximumBytes)
    }

    /// Sources may reside in different independently verified shards. Explicit
    /// array order is the expert routing order; duplicate source pairs are invalid.
    public static func load(
        sources: [Source], expectedExperts: Int, inputDimensions: Int,
        outputDimensions: Int, weightDType: DType, maximumBytes: Int = 64 << 20
    ) throws -> ModelOptNVFP4SwitchLinear {
        var seen: [ObjectIdentifier: Set<String>] = [:]
        for source in sources {
            guard seen[ObjectIdentifier(source.file), default: []].insert(source.prefix).inserted
            else {
                throw Failure.invalidInventory
            }
        }
        guard expectedExperts > 0, expectedExperts <= 4096, sources.count == expectedExperts,
            !sources.contains(where: { $0.prefix.isEmpty }),
            maximumBytes > 0, maximumBytes <= 64 << 20
        else { throw Failure.invalidInventory }
        guard [DType.float16, .bfloat16, .float32].contains(weightDType) else {
            throw Failure.invalidInventory
        }
        var total = 0
        // Validate every expert before allocating any expert arrays.
        for source in sources {
            let file = source.file
            let prefix = source.prefix
            let stem = prefix + "."
            var tensors: [String: ModelOptProjectionLayout.Tensor] = [:]
            for (key, entry) in file.entries where key.hasPrefix(stem) {
                tensors[String(key.dropFirst(stem.count))] = entry.layout
                let (next, overflow) = total.addingReportingOverflow(entry.layout.byteCount)
                guard !overflow, next <= maximumBytes else { throw Failure.payloadLimit }
                total = next
            }
            let layout = try ModelOptProjectionLayout(
                tensors: tensors, inputDimensions: inputDimensions,
                outputDimensions: outputDimensions)
            guard layout.storage == .nvfp4 else { throw Failure.invalidInventory }
        }
        func scalar(_ file: ModelOptTensorFile, _ name: String) throws -> Float {
            let data = try file.read(name)
            let bits = data.enumerated().reduce(UInt32(0)) {
                $0 | UInt32($1.element) << (8 * $1.offset)
            }
            let value = Float(bitPattern: bits)
            guard data.count == 4, value.isFinite, value > 0,
                value <= Float.greatestFiniteMagnitude / (448 * 6),
                (1 / value).isFinite, ((448 * 6) / (value * (448 * 6))).isFinite
            else { throw Failure.invalidScale }
            return value
        }
        // Host scalars are validated before any packed device allocation.
        let scales = try sources.map { try scalar($0.file, $0.prefix + ".weight_scale_2") }
        let inputs = try sources.map { try scalar($0.file, $0.prefix + ".input_scale") }
        var experts: [ModelOptNVFP4Weights] = []
        for (index, source) in sources.enumerated() {
            let file = source.file
            let prefix = source.prefix
            let weight = MLXArray(Array(try file.read(prefix + ".weight"))).reshaped(
                outputDimensions, inputDimensions / 2)
            let blocks = MLXArray(Array(try file.read(prefix + ".weight_scale"))).reshaped(
                outputDimensions, inputDimensions / 16)
            experts.append(
                try ModelOptNVFP4Weights(
                    packedBytes: weight, e4m3ScaleBytes: blocks, tensorScale: scales[index]))
        }
        let result = try ModelOptNVFP4SwitchLinear(
            experts: experts, inputScales: inputs, weightDType: weightDType)
        // Materialize stacked packed storage before releasing individual arrays.
        eval(result.parameters())
        return result
    }
}

extension ModelOptExpertLoader {
    /// Index-bound groups in explicit expert-ID order; permits split sidecars.
    /// The caller owns a residency grant covering the complete packed stack and
    /// transient source/output copies. No fixed64MiB full-model bypass is inferred.
    public static func load(
        groups: [[String: ModelOptTensorSource]], expectedExperts: Int,
        inputDimensions: Int, outputDimensions: Int, weightDType: DType,
        maximumBytes: Int, construction: NativeConstructionScope? = nil
    ) throws -> ModelOptNVFP4SwitchLinear {
        guard expectedExperts > 0, expectedExperts <= 4096, groups.count == expectedExperts,
            [.float16, .bfloat16, .float32].contains(weightDType)
        else { throw Failure.invalidInventory }
        var total = 0
        var tensorScales: [Float] = []
        var inputScales: [Float] = []
        for group in groups {
            for source in group.values {
                let (next, overflow) = total.addingReportingOverflow(source.layout.byteCount)
                guard !overflow, maximumBytes > 0, next <= maximumBytes else {
                    throw Failure.payloadLimit
                }
                total = next
            }
            let layout = try ModelOptProjectionLayout(
                tensors: group.mapValues(\.layout),
                inputDimensions: inputDimensions, outputDimensions: outputDimensions)
            guard layout.storage == .nvfp4 else { throw Failure.invalidInventory }
            let scale = try group["weight_scale_2"]!.scalar()
            let input = try group["input_scale"]!.scalar()
            for value in [scale, input] {
                guard value <= Float.greatestFiniteMagnitude / (448 * 6),
                    (1 / value).isFinite, ((448 * 6) / (value * (448 * 6))).isFinite
                else { throw Failure.invalidScale }
            }
            tensorScales.append(scale)
            inputScales.append(input)
        }
        var experts: [ModelOptNVFP4Weights] = []
        for (index, group) in groups.enumerated() {
            experts.append(
                try ModelOptNVFP4Weights(
                    packedBytes: group["weight"]!.array(
                        maximumBytes: maximumBytes, construction: construction),
                    e4m3ScaleBytes: group["weight_scale"]!.array(
                        maximumBytes: maximumBytes, construction: construction),
                    tensorScale: tensorScales[index]))
        }
        let layer = try ModelOptNVFP4SwitchLinear(
            experts: experts, inputScales: inputScales, weightDType: weightDType)
        try construction?.retainOwner(layer)
        try MLX.withError { errors in
            eval(layer.parameters())
            try errors.check()
        }
        return layer
    }
}
