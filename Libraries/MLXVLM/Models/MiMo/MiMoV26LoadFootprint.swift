// Copyright © 2026 Eigen Labs.
import Foundation

#if canImport(Darwin)
    import Darwin
#else
    import Glibc
#endif

public enum MiMoV26LoadFootprintError: Error, Equatable, Sendable {
    case unsupportedProfile
    case invalidInventory
    case overflow
}

/// Source-bound estimate for the strict root-bundle loader followed by serial
/// source-handle materialization. This is NOT measured peak residency, a runtime
/// activation/KV reserve, or permission to bypass model-load admission.
///
/// All root components remain resident, including vision/audio-patch/MTP. The
/// separately packaged audio tokenizer/DFlash/original-head assets are outside
/// this loader's root inventory; a consumer loading any of them must reserve
/// their additional residency/transients independently before allocation.
public enum MiMoV26LoadFootprint {
    public static let contract = "mimo-v26-strict-root-serial-source-v1"

    public struct Estimate: Equatable, Encodable, Sendable {
        public let contract: String
        public let residentBytes: UInt64
        public let transientBytes: UInt64
        public let totalBytes: UInt64
        public let largestShardBytes: UInt64
        public let largestTensorBytes: UInt64
        public let largestExpertBlockBytes: UInt64
        public let auxiliaryCopyBytes: UInt64
        public let pageRoundingBytes: UInt64
        public let metadataAllowanceBytes: UInt64
        public let tensorCount: Int
        public let configSHA256, indexSHA256, descriptorSHA256: String
    }

    /// Use only the plan validated by MiMoV26FilesystemWeights. Recheck object
    /// states now and again at the actual loader/materialization boundary.
    public static func estimate(plan: MiMoV26FilesystemLoadPlan) throws -> Estimate {
        #if os(macOS) && arch(arm64)
            // The general reader accepts caller-selected wider limits; this
            // smaller native allowance has its own explicit bounded profile.
            guard plan.metadataBytesRead <= 16 * 1_048_576 else {
                throw MiMoV26LoadFootprintError.unsupportedProfile
            }
            try MiMoV26FilesystemWeights.validateCurrentObjects(plan: plan)
            let files = plan.shards.map { ($0.name, $0.objectState.bytes) }
            let result = try estimateValidatedLayout(
                plan.bundlePlan, files: files, allocationPageBytes: Int(getpagesize()))
            guard UInt64(plan.totalFileBytes) == result.residentBytes else {
                throw MiMoV26LoadFootprintError.invalidInventory
            }
            return result
        #else
            throw MiMoV26LoadFootprintError.unsupportedProfile
        #endif
    }

    /// Internal pure arithmetic seam for metadata-only positive/negative tests.
    /// Public callers cannot replace filesystem validation with this function.
    static func estimateValidatedLayout(
        _ plan: MiMoV26ConvertedLoadPlan,
        files: [(name: String, bytes: Int)], allocationPageBytes: Int
    ) throws -> Estimate {
        let config = plan.configuration
        guard allocationPageBytes == 16_384, config.modelType == "mimo_v2",
            config.dtype == "bfloat16", config.numNextnPredictLayers == 3,
            config.quantization.nativeDefault?.bits == 4,
            plan.provenance.layout == .nativeConversion
                ? (config.quantization.nativeDefault?.mode == "mxfp4"
                    && config.quantization.nativeDefault?.groupSize == 32)
                : (config.quantization.nativeDefault?.mode == "affine"
                    && config.quantization.nativeDefault?.groupSize == 64)
        else { throw MiMoV26LoadFootprintError.unsupportedProfile }
        guard files.count == plan.rootFiles.count, !files.isEmpty,
            Set(files.map(\.name)) == plan.rootFiles,
            plan.descriptors.count <= 2048, !plan.descriptors.isEmpty,
            Set(plan.components.keys) == Set(MiMoV26ConvertedComponent.allCases)
        else { throw MiMoV26LoadFootprintError.invalidInventory }

        var tensorBytes: [String: UInt64] = [:]
        var perFile: [String: UInt64] = [:]
        var totalTensorBytes: UInt64 = 0
        for (name, descriptor) in plan.descriptors {
            guard !descriptor.shape.isEmpty, plan.rootFiles.contains(descriptor.file) else {
                throw MiMoV26LoadFootprintError.invalidInventory
            }
            var bytes = UInt64(descriptor.dtype.bytes)
            for dimension in descriptor.shape {
                guard dimension > 0 else { throw MiMoV26LoadFootprintError.invalidInventory }
                bytes = try multiply(bytes, UInt64(dimension))
            }
            tensorBytes[name] = bytes
            perFile[descriptor.file] = try add(perFile[descriptor.file, default: 0], bytes)
            totalTensorBytes = try add(totalTensorBytes, bytes)
        }
        guard totalTensorBytes == UInt64(plan.tensorBytes) else {
            throw MiMoV26LoadFootprintError.invalidInventory
        }
        var resident: UInt64 = 0
        var largestShard: UInt64 = 0
        for (name, rawBytes) in files {
            guard rawBytes > 8, let payload = perFile[name], UInt64(rawBytes) > payload else {
                throw MiMoV26LoadFootprintError.invalidInventory
            }
            resident = try add(resident, UInt64(rawBytes))
            largestShard = max(largestShard, UInt64(rawBytes))
        }
        // This profile caps total preflight metadata in the public entrypoint;
        // also bound the shard-header difference independently. Price actual
        // file bytes, not only payloads; refuse unrelated oversized inventory.
        guard resident >= totalTensorBytes,
            resident - totalTensorBytes <= 16 * 1_048_576
        else {
            throw MiMoV26LoadFootprintError.invalidInventory
        }

        var seen = Set<String>()
        var auxiliaryCopies: UInt64 = 0
        for component in MiMoV26ConvertedComponent.allCases {
            guard let names = plan.components[component], !names.isEmpty else {
                throw MiMoV26LoadFootprintError.invalidInventory
            }
            for name in names {
                guard seen.insert(name).inserted, let bytes = tensorBytes[name] else {
                    throw MiMoV26LoadFootprintError.invalidInventory
                }
                if component != .target { auxiliaryCopies = try add(auxiliaryCopies, bytes) }
            }
        }
        guard seen == Set(tensorBytes.keys) else {
            throw MiMoV26LoadFootprintError.invalidInventory
        }

        var expertBlocks: [String: UInt64] = [:]
        let sourceNames = Dictionary(
            uniqueKeysWithValues: plan.parameterNames.map { ($0.value, $0.key) })
        for module in plan.targetExpertModulePaths {
            let components = module.split(separator: ".")
            guard components.count > 1 else { throw MiMoV26LoadFootprintError.invalidInventory }
            let parent = components.dropLast().joined(separator: ".")
            guard let weight = sourceNames[module + ".weight"],
                let scale = sourceNames[module + ".scales"]
            else {
                throw MiMoV26LoadFootprintError.invalidInventory
            }
            guard let weightBytes = tensorBytes[weight], let scaleBytes = tensorBytes[scale],
                plan.descriptors[weight]?.dtype == .uint32,
                plan.descriptors[scale]?.dtype == .uint8
            else {
                throw MiMoV26LoadFootprintError.invalidInventory
            }
            expertBlocks[parent] = try add(
                expertBlocks[parent, default: 0],
                try add(weightBytes, scaleBytes))
        }
        let largestTensor = tensorBytes.values.max() ?? 0
        let largestExpert = expertBlocks.values.max() ?? 0
        let rounding = try multiply(try multiply(UInt64(tensorBytes.count), 2), 16_384)
        let metadata: UInt64 = 1 << 30
        // Deliberately SUM, not max, the source-derived copy families:
        // - one complete source shard, despite direct-to-final tensor reads;
        // - one largest source tensor, despite no whole-tensor staging copy;
        // - one whole packed expert block (all three projections);
        // - complete vision/audio-patch/assistant copies (not only reshapes);
        // - two sets of page-rounding allowances and1GiB for small loader state.
        // A consumer using a different loader, dtype conversion or concurrent
        // materialization cannot inherit this contract. Qualification must still
        // measure phase peaks and retain unchanged OS/activation/KV safeguards.
        var transient: UInt64 = 0
        for allowance in [
            largestShard, largestTensor, largestExpert,
            auxiliaryCopies, rounding, metadata,
        ] {
            transient = try add(transient, allowance)
        }
        return Estimate(
            contract: contract, residentBytes: resident,
            transientBytes: transient, totalBytes: try add(resident, transient),
            largestShardBytes: largestShard, largestTensorBytes: largestTensor,
            largestExpertBlockBytes: largestExpert, auxiliaryCopyBytes: auxiliaryCopies,
            pageRoundingBytes: rounding, metadataAllowanceBytes: metadata,
            tensorCount: tensorBytes.count, configSHA256: plan.configSHA256,
            indexSHA256: plan.indexSHA256, descriptorSHA256: plan.descriptorSHA256)
    }

    private static func add(_ a: UInt64, _ b: UInt64) throws -> UInt64 {
        let result = a.addingReportingOverflow(b)
        guard !result.overflow else { throw MiMoV26LoadFootprintError.overflow }
        return result.partialValue
    }
    private static func multiply(_ a: UInt64, _ b: UInt64) throws -> UInt64 {
        let result = a.multipliedReportingOverflow(by: b)
        guard !result.overflow else { throw MiMoV26LoadFootprintError.overflow }
        return result.partialValue
    }
}
