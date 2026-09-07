// Physical target-KV byte layout shared by every AdmissionV2 operation.

struct CBv2KVAdmissionStorageLayout {
    let perLayerTokenBytes: [Int?]
    let maximumPerTokenBytes: Int
    let fullPerTokenBytes: Int
    let maximumRequestOverheadBytes: Int

    init(layerKinds: [CBv2LayerKind], config: AdmissionV2.Config,
         residency: any CBv2KVResidencyPolicy) {
        let elementBytes: [Int]
        if let table = config.layerElementBytes {
            precondition(table.count == layerKinds.count,
                         "AdmissionV2: layerElementBytes count does not match layer count")
            elementBytes = table
        } else {
            elementBytes = Array(repeating: config.elementBytes, count: layerKinds.count)
        }
        perLayerTokenBytes = layerKinds.enumerated().map { index, kind in
            if kind.sharesKVWithLayer != nil { return 0 }
            if let exact = config.layerBytesPerToken {
                guard exact.count == layerKinds.count,
                    kind.kvHeads >= 0, kind.headDim >= 0, exact[index] >= 0,
                    exact[index] > 0 || kind.kvHeads == 0 || kind.headDim == 0
                else { return nil }
                return exact[index]
            }
            guard kind.kvHeads >= 0, kind.headDim >= 0, elementBytes[index] >= 0,
                let elements = Self.multiply(kind.kvHeads, kind.headDim),
                let kvElements = Self.multiply(elements, 2)
            else { return nil }
            return Self.multiply(kvElements, elementBytes[index])
        }

        var maximum = 0
        var full = 0
        var requestOverhead = 0
        var invalid = false
        let pagePadding = max(1, residency.rowGranularity) - 1
        for (index, kind) in layerKinds.enumerated() where kind.sharesKVWithLayer == nil {
            guard let bytes = perLayerTokenBytes[index],
                let next = Self.add(maximum, bytes)
            else { invalid = true; break }
            maximum = next
            let overheadRows: Int
            if case .full = kind.attention {
                guard let nextFull = Self.add(full, bytes) else { invalid = true; break }
                full = nextFull
                overheadRows = pagePadding
            } else {
                // Derive the saturated window promise from the backend, including
                // its speculative/chunk exposure; do not restate ring sizing.
                // Leave a full granule for implementations evaluating n+page-1
                // left-to-right (n+page must itself remain representable).
                guard let rows = residency.residentRows(
                    layer: kind, tokens: Int.max - max(1, residency.rowGranularity)), rows >= 0
                else { invalid = true; break }
                overheadRows = rows
            }
            guard let overhead = Self.multiply(overheadRows, bytes),
                let total = Self.add(requestOverhead, overhead)
            else { invalid = true; break }
            requestOverhead = total
        }
        maximumPerTokenBytes = invalid ? Int.max : maximum
        fullPerTokenBytes = invalid ? Int.max : full
        maximumRequestOverheadBytes = invalid ? Int.max : requestOverhead
    }

    private static func multiply(_ lhs: Int, _ rhs: Int) -> Int? {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        return overflow ? nil : value
    }

    private static func add(_ lhs: Int, _ rhs: Int) -> Int? {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? nil : value
    }
}
