// Copyright © 2026 Eigen Labs.
import Foundation
import MLXLMCommon

/// Text factory for Prism's folded Qwen3.8-27B pack. Vision remains a VLM-factory concern.
public final class PrismHadamardQwen35TextModel: Qwen35Model, PrismHadamardLoading,
    CheckpointWeightLoadFiltering
{
    public let prismCheckpoint: PrismHadamardCheckpointConfiguration
    private let textConfiguration: Qwen35TextConfiguration

    public init(configurationData: Data) throws {
        prismCheckpoint = try JSONDecoder().decode(PrismHadamardCheckpointConfiguration.self, from: configurationData)
        let config = try JSONDecoder.json5().decode(Qwen35Configuration.self, from: configurationData)
        textConfiguration = config.textConfig
        super.init(config)
    }
    public override var cbv2Capabilities: CBv2ModelCapabilities {
        var value = super.cbv2Capabilities
        value.supportsMTP = false
        return value
    }
    public override var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        textConfiguration.cbv2RecurrentStateSpec(activationDType: .float16)
    }
    public var checkpointWeightLoadFilter: CheckpointWeightLoadFilter {
        { !$0.hasPrefix("vision_tower.") }
    }
    public var skipWholeShardPrefetch: Bool { true }
}
