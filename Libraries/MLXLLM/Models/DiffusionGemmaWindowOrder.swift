/// The pinned reference exposes physical ring order to a singleton encoder
/// query, but temporal order to a multi-token query and the denoiser. Ordering
/// changes reduction rounding even though the logical attention set is equal.
/// Preserve this small native state separately from canonical temporal storage.
struct DiffusionGemmaWindowOrder: Equatable {
    var physicalLength = 0
    var cursor = 0

    func appending(_ count: Int, priorPosition: Int, window: Int) -> Self {
        if count > 1 {
            let length = min(priorPosition, window - 1) + count
            return Self(physicalLength: length, cursor: length)
        }
        var next = self
        if physicalLength == 0 || (priorPosition >= physicalLength && physicalLength < window) {
            next.physicalLength += min(256, window - priorPosition)
            next.cursor = priorPosition
        }
        if next.physicalLength > window {
            next.physicalLength = window
            next.cursor = window
        }
        if next.cursor == window { next.cursor = 0 }
        next.cursor += 1
        return next
    }
}
