/// The kernel families a paged pool of these layers dispatches at this
/// precision: native paged attention for every layer, and the packed writer,
/// reader and attention for every quantized owner. An owner probes them before
/// it publishes a serving pool, because an MLX custom-kernel compilation
/// failure can end the process instead of throwing. The provider first runs the
/// same plan in a child process (its `PagedKernelPreflight`); every owner then
/// runs it in its own process to populate the kernel cache.
public struct PagedKernelSmokePlan: Sendable, Equatable {
    public let nativeShapes: [PagedAttentionKernelSmokeShape]
    public let packedShapes: [PagedQuantizedKernelSmokeShape]
    public let quantization: PagedKVQuantizationConfig?

    public init(
        layerKinds: [CBv2LayerKind], quantization: PagedKVQuantizationConfig?,
        nativeLayerIndices: Set<Int> = []
    ) throws {
        nativeShapes = PagedAttentionKernel.smokeShapes(layerKinds: layerKinds)
        packedShapes =
            try quantization.map {
                try PagedQuantizedKernelSmoke.smokeShapes(
                    layerKinds: layerKinds, quantization: $0,
                    nativeLayerIndices: nativeLayerIndices)
            } ?? []
        self.quantization = quantization
    }

    /// Compiles and runs every selected family in this process. The bounded
    /// probe exercises geometry and dtype variants, never request data.
    public func runInProcess() throws {
        try PagedAttentionKernel.runtimeSmoke(shapes: nativeShapes)
        if let quantization {
            try PagedQuantizedKernelSmoke.runtimeSmoke(
                shapes: packedShapes, quantization: quantization)
        }
    }
}
