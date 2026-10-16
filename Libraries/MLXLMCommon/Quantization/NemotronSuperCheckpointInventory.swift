import Foundation

/// A metadata-only staging plan for the NVIDIA composite checkpoint namespace.
/// It neither allocates model weights nor declares a checkpoint executable.
/// MTP is retained separately, never silently discarded by the text sanitizer.
public struct NemotronSuperCheckpointInventory: Sendable {
    public enum Component: String, Sendable, CaseIterable {
        case language, vision, projector, mtp
    }
    public struct Entry: Sendable, Equatable {
        public let source: String
        public let destination: String
        public let shard: String
        public let component: Component
    }
    public enum Failure: Error, Equatable {
        case invalidIndex
        case unsafeShard(String)
        case unknownNamespace(String)
        case duplicateDestination(String)
        case incompleteCalibration(String)
        case inconsistentCalibration(String)
    }
    public let entries: [Entry]
    public let shards: [String]

    /// Bounds the JSON input and rejects paths before any caller opens a shard.
    /// Tensor headers, shapes, dtype, payload hashes and completeness against the
    /// constructed model must still be validated by the eventual weight loader.
    public init(indexData: Data) throws {
        guard !indexData.isEmpty, indexData.count <= 64 * 1024 * 1024,
            let root = try JSONSerialization.jsonObject(with: indexData) as? [String: Any],
            let weights = root["weight_map"] as? [String: String],
            !weights.isEmpty, weights.count <= 250_000
        else { throw Failure.invalidIndex }
        var result: [Entry] = []
        var destinations = Set<String>()
        for (source, shard) in weights {
            guard shard.hasSuffix(".safetensors"), !shard.hasPrefix("."),
                !shard.contains("/"), !shard.contains("\\"), !shard.contains(":"),
                !shard.unicodeScalars.contains(where: { $0.value < 32 }), shard.count <= 255
            else { throw Failure.unsafeShard(shard) }
            let component: Component
            let destination: String
            if source.hasPrefix("language_model.mtp.") {
                component = .mtp
                destination = String(source.dropFirst("language_model.".count))
            } else if source.hasPrefix("language_model.backbone.")
                || source.hasPrefix("language_model.lm_head.")
            {
                component = .language
                destination = String(source.dropFirst("language_model.".count))
            } else if source.hasPrefix("vision_model.") {
                component = .vision
                destination = source
            } else if source.hasPrefix("vision_projector.") {
                component = .projector
                destination = source
            } else if source.hasPrefix("mlp1.") {
                component = .projector
                let mappings = [
                    ("mlp1.0.", "norm."), ("mlp1.1.", "linear1."), ("mlp1.3.", "linear2."),
                ]
                guard let mapping = mappings.first(where: { source.hasPrefix($0.0) }) else {
                    throw Failure.unknownNamespace(source)
                }
                destination =
                    "vision_projector.mlp1." + mapping.1 + source.dropFirst(mapping.0.count)
            } else {
                throw Failure.unknownNamespace(source)
            }
            let identity = component.rawValue + ":" + destination
            guard destinations.insert(identity).inserted else {
                throw Failure.duplicateDestination(destination)
            }
            result.append(
                Entry(source: source, destination: destination, shard: shard, component: component))
        }
        entries = result.sorted { $0.source < $1.source }
        shards = Set(weights.values).sorted()
        try validateCalibrationGroups()
    }

    public func entries(for component: Component) -> [Entry] {
        entries.filter { $0.component == component }
    }

    /// ModelOpt NVFP4 and calibrated FP8 sidecars must survive together. This
    /// checks index presence only; it deliberately does not relabel them as MLX
    /// affine .scales/.biases or treat attention k_scale/v_scale as weights.
    private func validateCalibrationGroups() throws {
        let keys = Set(entries.map(\.source))
        let suffixes = [".input_scale", ".weight_scale", ".weight_scale_2"]
        var prefixes = Set<String>()
        for key in keys {
            for suffix in suffixes where key.hasSuffix(suffix) {
                prefixes.insert(String(key.dropLast(suffix.count)))
            }
        }
        for prefix in prefixes {
            guard keys.contains(prefix + ".weight"), keys.contains(prefix + ".weight_scale"),
                keys.contains(prefix + ".input_scale")
            else {
                throw Failure.incompleteCalibration(prefix)
            }
            guard !keys.contains(prefix + ".scales"), !keys.contains(prefix + ".biases") else {
                throw Failure.inconsistentCalibration(prefix)
            }
        }
    }
}
