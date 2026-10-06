extension EngineV2 {
    static func checkpointQwen4Geometries(model: any CBv2SteppableModel,
                                        layerKinds: [CBv2LayerKind]) -> [CBv2Qwen4CheckpointGeometry]? {
        let kinds = layerKinds.enumerated().filter { $0.element.qwen4IndexerCompressRatio != nil }
        if kinds.isEmpty { return [] }
        guard let owner = model as? any CBv2Qwen4CheckpointGeometryProviding,
            let geometries = owner.cbv2Qwen4CheckpointGeometries,
            geometries.count == kinds.count,
            zip(geometries, kinds).allSatisfy({ geometry, entry in
                geometry.layer == (entry.element.modelLayerIndex ?? entry.offset)
                    && geometry.compressRatio == entry.element.qwen4IndexerCompressRatio
            })
        else { return nil }
        return geometries
    }
}
