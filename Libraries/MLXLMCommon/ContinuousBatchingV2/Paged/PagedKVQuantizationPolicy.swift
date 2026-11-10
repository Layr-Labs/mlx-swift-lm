import Foundation

/// Attention precision for live execution or checkpoint storage. Weight precision
/// never selects either policy. One type for every owner that builds a paged
/// pool: the provider's engines and the cluster's stage hosts.
public enum PagedKVQuantizationSelection: String, Sendable, Equatable, CaseIterable {
    case balanced
    case k8v4
    case k8v8
    case native

    public var configuration: PagedKVQuantizationConfig? {
        guard self != .native else { return nil }
        return PagedKVQuantizationConfig(
            keyBits: self == .balanced ? 4 : 8, valueBits: self == .k8v8 ? 8 : 4,
            groupSize: 64, rotationBlockSize: 128, recentTokenCount: 128)
    }
}

/// How a selection is named and resolved: the process switch, then a per-model
/// setting, then the global setting, then the purpose's default (live native,
/// checkpoint balanced, or native for `nativeCheckpointModelTypes`). The provider
/// passes its saved settings; a cluster rank receives the value its owner resolved
/// in its launch environment.
public enum PagedKVQuantizationPolicy {
    public static let environmentKey = "DARKBLOOM_CBV2_KV_QUANTIZATION"
    public static let checkpointEnvironmentKey = "DARKBLOOM_CBV2_CHECKPOINT_QUANTIZATION"

    public enum Purpose: Sendable, Equatable {
        case live
        case checkpoint

        public var defaultSelection: PagedKVQuantizationSelection {
            self == .live ? .native : .balanced
        }

        public var environmentKey: String {
            self == .live
                ? PagedKVQuantizationPolicy.environmentKey
                : PagedKVQuantizationPolicy.checkpointEnvironmentKey
        }
    }

    public enum Failure: Error, CustomStringConvertible, Equatable {
        case invalidSelection(String)
        case invalidCheckpointSelection(String)
        case requiresPagedBackend(String)
        case flintRequiresNative

        public var description: String {
            switch self {
            case .invalidSelection(let raw):
                "Invalid live KV quantization \"\(raw)\"; expected balanced, k8v4, k8v8 or native"
            case .invalidCheckpointSelection(let raw):
                "Invalid checkpoint KV quantization \"\(raw)\"; expected balanced, k8v4, k8v8 or native"
            case .requiresPagedBackend(let reason):
                "Live quantized KV requires paged execution; select native live precision for \(reason)"
            case .flintRequiresNative:
                "Flint currently requires native BF16 KV and native checkpoint precision"
            }
        }
    }

    public static func parseSelection(
        _ raw: String, purpose: Purpose = .live
    ) throws -> PagedKVQuantizationSelection {
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "": purpose.defaultSelection
        case "balanced", "k4v4", "int4", "on": .balanced
        case "k8v4": .k8v4
        case "k8v8", "int8": .k8v8
        case "native", "off", "0": .native
        default:
            throw purpose == .live
                ? Failure.invalidSelection(raw)
                : Failure.invalidCheckpointSelection(raw)
        }
    }

    /// Model types whose complete checkpoints store native rows unless a setting says
    /// otherwise. Nemotron 3.5 Lightning served without its MTP head keeps the exact
    /// rows it stored beside that head, so a cache hit's output does not move.
    public static let nativeCheckpointModelTypes: Set<String> = ["nemotron_h"]

    /// MiMo's SDK-owned execution path is excluded before parsing any override.
    public static func resolve(
        modelType: String?, global: String? = nil, byModel: [String: String] = [:],
        modelID: String, environment: [String: String] = ProcessInfo.processInfo.environment,
        purpose: Purpose = .live
    ) throws -> PagedKVQuantizationSelection {
        guard modelType != "mimo_v2" else { return .native }
        if modelType == "flint" {
            // Flint owns native precision. The provider-wide default (checkpoints
            // default to balanced) never applies to it; an explicit per-model or
            // environment selection of anything else is refused.
            let raw = environment[purpose.environmentKey] ?? byModel[modelID]
            if let raw, !raw.isEmpty, try parseSelection(raw, purpose: purpose) != .native {
                throw Failure.flintRequiresNative
            }
            return .native
        }
        let raw = environment[purpose.environmentKey] ?? byModel[modelID]
        if purpose == .checkpoint, let modelType, nativeCheckpointModelTypes.contains(modelType),
            isUnset(raw), isUnset(global) || isSavedCheckpointDefault(global)
        {
            return .native
        }
        return try parseSelection(raw ?? global ?? "", purpose: purpose)
    }

    private static func normalized(_ raw: String) -> String {
        raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    private static func isUnset(_ raw: String?) -> Bool {
        raw.map { normalized($0).isEmpty } ?? true
    }

    /// A saved provider.toml spells out the previous default, so the global word
    /// `balanced` is not a choice. `k4v4` names the same profile as one, and a
    /// per-model setting or the process switch can still say `balanced`.
    private static func isSavedCheckpointDefault(_ raw: String?) -> Bool {
        raw.map { normalized($0) == "balanced" } ?? false
    }
}
