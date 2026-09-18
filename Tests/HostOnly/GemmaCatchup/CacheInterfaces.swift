// CPU-only interface doubles for compiling the REAL LayerCacheBankV2.swift.
// These deliberately contain no MLX import, tensors, device or model code.
// They validate host binding behavior, not native KV storage/attention.
import Foundation

public protocol CBv2SequenceKV: AnyObject {}
public struct CBv2LayerKind {
    public let sharesKVWithLayer: Int?
    public init(sharesKVWithLayer: Int? = nil) { self.sharesKVWithLayer = sharesKVWithLayer }
}
public protocol CBv2AttendingLayerCache: AnyObject {
    var layerIndex: Int { get }
    var kind: CBv2LayerKind { get }
    func setRows(_ rows: [CBv2SequenceKV])
}
public protocol CBv2LayerCacheProvider: AnyObject {
    func layerCaches(rowStates: [[CBv2SequenceKV?]]) -> [CBv2AttendingLayerCache]
}
public protocol CBv2SpanMaskBinding: AnyObject {}
public protocol CBv2PackedSpanMaskBinding: AnyObject {}
public protocol CBv2MTPRectangularSerializing: AnyObject {}

public final class CBv2LayerCache: CBv2AttendingLayerCache, CBv2KVSourceChunkRetaining, CBv2CoordinatedPositionBinding {
    public let layerIndex: Int
    public let kind: CBv2LayerKind
    var rows: [CBv2SequenceKV] = []
    var binds = 0
    var retainsForBorrowers = true
    var positionBindingCoordinator: (any CBv2PositionBindingCoordinator)?
    var coordinatedBinds = 0
    public init(layerIndex: Int, kind: CBv2LayerKind, attentionSoftcap: Float? = nil) {
        self.layerIndex = layerIndex
        self.kind = kind
    }
    public func setRows(_ rows: [CBv2SequenceKV]) { binds += 1; self.rows = rows }
    func setRowsForPositionBinding(_ rows: [CBv2SequenceKV]) { coordinatedBinds += 1; self.rows = rows }
    public func setRetainsChunkForBorrowers(_ retains: Bool) { retainsForBorrowers = retains }
}
