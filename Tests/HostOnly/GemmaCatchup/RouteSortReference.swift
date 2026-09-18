import Foundation

/// CPU specification model of the integer stages, not execution of the MSL.
enum RouteSortReference {
    static func run(_ keys: [UInt32], bitset: Bool, parallelScan: Bool) -> (rows: [UInt32], keys: [UInt32], inverse: [UInt32]) {
        precondition(keys.allSatisfy { $0 < 128 })
        let blocks = (keys.count + 255) / 256
        var histogram = Array(repeating: 0, count: blocks * 256)
        for i in keys.indices { histogram[(i / 256) * 256 + Int(keys[i])] += 1 }
        var offsets = Array(repeating: -1, count: histogram.count)
        var base = 0
        for expert in 0..<128 {
            let parts = parallelScan ? 8 : 1
            precondition(blocks.isMultiple(of: parts))
            let per = blocks / parts
            let totals = (0..<parts).map { part in
                (part * per ..< (part + 1) * per).reduce(0) { $0 + histogram[$1 * 256 + expert] }
            }
            let total = totals.reduce(0, +)
            if total > 0 {
                var earlier = 0
                for part in 0..<parts {
                    var running = base + earlier
                    for block in part * per ..< (part + 1) * per {
                        offsets[block * 256 + expert] = running
                        running += histogram[block * 256 + expert]
                    }
                    earlier += totals[part]
                }
            }
            base += total
        }
        var rowOrder = Array(repeating: UInt32.max, count: keys.count)
        var sorted = rowOrder, inverse = rowOrder
        for block in 0..<blocks {
            let start = block * 256, end = min(start + 256, keys.count)
            var bits = Array(repeating: UInt32(0), count: 128 * 8)
            if bitset {
                for index in start..<end {
                    let k = index - start
                    bits[(k / 32) * 128 + Int(keys[index])] |= UInt32(1) << UInt32(k % 32)
                }
            }
            for index in start..<end {
                let key = Int(keys[index]), k = index - start
                let rank: Int
                if bitset {
                    let word = k / 32, bit = k % 32
                    rank = (0..<word).reduce(0) { $0 + bits[$1 * 128 + key].nonzeroBitCount }
                        + (bits[word * 128 + key] & ((UInt32(1) << UInt32(bit)) - 1)).nonzeroBitCount
                } else {
                    rank = (start..<index).filter { keys[$0] == keys[index] }.count
                }
                let position = offsets[block * 256 + key] + rank
                precondition(position >= 0 && position < keys.count && sorted[position] == UInt32.max)
                rowOrder[position] = UInt32(index / 8)
                sorted[position] = keys[index]
                inverse[index] = UInt32(position)
            }
        }
        return (rowOrder, sorted, inverse)
    }
}
