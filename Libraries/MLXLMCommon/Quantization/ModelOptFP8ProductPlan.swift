import MLX

/// How MLX (mlx 83dbdb04 `matmul.cpp`) sums each element of a BF16 or FP16
/// `[rows, K] x [K, outputs]` product with a row-major left operand and a
/// transposed right operand on the GPU. Products of one class add the same
/// terms in the same order, so an output element has the same bits whichever
/// other rows and columns share its product.
enum ModelOptMLXProductClass: Hashable {
    /// `gemv`: a single output column.
    case vector
    /// Steel split-K: `partitions` equal K ranges, each summed in ascending K,
    /// then added in partition order.
    case splitK(partitions: Int)
    /// Steel or NAX GEMM: one pass over K in ascending order.
    case regular
    /// NAX thin GEMM with `kGroups` interleaved K groups. A product with fewer
    /// than 32 rows or columns takes MPP's masked path; larger ones move their
    /// edge blocks inside.
    case thin(kGroups: Int, maskedRows: Bool, maskedColumns: Bool)
    /// NAX split-K over `partitionSize`-deep K ranges.
    case naxSplitK(partitionSize: Int)
}

/// MLX's product routing on one GPU: Neural Accelerators (NAX) or not, and
/// the size class (last letter of the Metal architecture, "d" on an Ultra).
struct ModelOptMLXProductRouting: Hashable {
    let nax: Bool
    let size: Character

    /// The default stream's routing; nil off the GPU.
    static var current: Self? {
        guard MiMoV26NAXGatherQMM.gpuStream(.default) else { return nil }
        return Self(
            nax: ModelOptNAXDenseLinear.naxAvailable,
            size: GPU.deviceInfo().architecture.last ?? " ")
    }

    /// The class `Matmul::eval_gpu` and `steel_matmul_axpby` pick in mlx
    /// 83dbdb04. Nil where it is not modelled: fewer than 16 rows (MLX's wide
    /// vector kernels) or empty shapes.
    func productClass(rows: Int, inputs: Int, outputs: Int) -> ModelOptMLXProductClass? {
        guard rows >= 16, rows <= 1 << 16, inputs > 0, outputs > 0, outputs <= 1 << 24 else {
            return nil
        }
        if outputs == 1 { return .vector }
        guard nax else {
            let tiles = ((rows + 15) / 16) * ((outputs + 15) / 16)
            let depth = inputs / 16
            if tiles <= (size == "s" || size == "d" ? 2048 : 1024), depth >= 8,
                inputs >= max(rows, outputs)
            {
                let large = ((rows + 31) / 32) * ((outputs + 31) / 32)
                return .splitK(partitions: min(max(2, Self.nextPowerOfTwo(depth / large)), 32))
            }
            return .regular
        }
        if outputs <= 64 || rows <= 64 {
            return .thin(
                kGroups: rows < 2048 ? 4 : 2, maskedRows: rows < 32, maskedColumns: outputs < 32)
        }
        let largest = max(rows, outputs)
        guard inputs >= 3 * largest || (largest <= 1024 && inputs > 2 * largest) else {
            return .regular
        }
        var partition = size == "d" && rows <= 2048 && outputs <= 2048 ? 2048 : 4096
        if inputs <= 1024 {
            partition = inputs / 2
        } else if inputs <= 2048 {
            partition = 1024
        } else if inputs <= 4096 {
            partition = 2048
        }
        return .naxSplitK(partitionSize: partition)
    }

    /// MLX's `next_power_of_2` (0 stays 0).
    static func nextPowerOfTwo(_ n: Int) -> Int {
        guard n > 0, n & (n - 1) != 0 else { return n }
        return 1 << (Int.bitWidth - (n - 1).leadingZeroBitCount)
    }
}

/// One product of the multi-row FP8 graph path: the decoded weight rows
/// `columns`, times each row block of the activation (`vectors`: one
/// single-column product per weight row, batched).
struct ModelOptFP8ProductPiece: Hashable {
    var columns: Range<Int>
    var rowBlocks: [Range<Int>]
    var vectors = false

    var dispatches: Int { vectors ? 1 : rowBlocks.count }

    /// Most products one call may take; more falls back to `slices`.
    static let maximumDispatches = 128

    /// The eager reference's output tiles (`eager`, `rows` rows of `inputs`)
    /// regrouped into fewer products of the same ``ModelOptMLXProductClass``,
    /// so every output keeps its bits. Calls with at most twice the one-row
    /// call's tile count (`oneRowTiles`), or off the modelled GPU paths, keep
    /// the eager tiles. A plan that would still need more than
    /// ``maximumDispatches`` products takes `slices` instead (not bit-matched):
    /// near the reference's 64 MiB bound its tiles shrink toward one row.
    /// `routing` is read only when the eager tiles are not kept.
    static func plan(
        rows: Int, inputs: Int, eager: [Range<Int>], oneRowTiles: Int,
        slices: [Range<Int>], routing: @autoclosure () -> ModelOptMLXProductRouting?
    ) -> [Self] {
        let tiles = eager.map { Self(columns: $0, rowBlocks: [0 ..< rows]) }
        guard eager.count > 2 * oneRowTiles else { return tiles }
        func bounded(_ pieces: [Self]) -> [Self] {
            pieces.dispatches <= maximumDispatches
                ? pieces : slices.map { Self(columns: $0, rowBlocks: [0 ..< rows]) }
        }
        guard let routing = routing() else { return bounded(tiles) }
        let maxWidth = max(1, (1 << 25) / inputs)
        var pieces: [Self] = []
        var start = 0
        while start < eager.count {
            let width = eager[start].count
            var end = start + 1
            while end < eager.count, eager[end].count == width { end += 1 }
            let run = eager[start].lowerBound ..< eager[end - 1].upperBound
            guard
                let regrouped = regrouped(
                    run, eagerWidth: width, rows: rows, inputs: inputs, maxWidth: maxWidth,
                    routing: routing)
            else { return bounded(tiles) }
            pieces += regrouped
            start = end
        }
        return bounded(pieces)
    }

    /// `run` (whole eager tiles of `eagerWidth` columns) as the fewest
    /// products of the eager class: column chunks of at most `maxWidth`, each
    /// times balanced row blocks of up to `rows`, 64 or 32 rows. Nil where the
    /// class is not modelled.
    private static func regrouped(
        _ run: Range<Int>, eagerWidth: Int, rows: Int, inputs: Int, maxWidth: Int,
        routing: ModelOptMLXProductRouting
    ) -> [Self]? {
        guard let target = routing.productClass(rows: rows, inputs: inputs, outputs: eagerWidth)
        else { return nil }
        if target == .vector {
            return chunks(run, count: (run.count + maxWidth - 1) / maxWidth).map {
                Self(columns: $0, rowBlocks: [0 ..< rows], vectors: true)
            }
        }
        var best: [Self]?
        for limit in Set([rows, min(rows, 64), min(rows, 32)]).sorted(by: >) {
            let blocks = chunks(0 ..< rows, count: (rows + limit - 1) / limit)
            let heights = Set(blocks.map(\.count))
            func keeps(_ width: Int) -> Bool {
                heights.allSatisfy {
                    routing.productClass(rows: $0, inputs: inputs, outputs: width) == target
                }
            }
            guard keeps(eagerWidth) else { continue }
            // Classes cover intervals of widths, so the widest width that keeps
            // the class bounds every narrower chunk down to the eager width.
            var low = eagerWidth
            var high = max(eagerWidth, min(run.count, maxWidth))
            while low < high {
                let middle = (low + high + 1) / 2
                if keeps(middle) { low = middle } else { high = middle - 1 }
            }
            let columns = chunks(run, count: (run.count + low - 1) / low)
            guard Set(columns.map(\.count)).allSatisfy(keeps) else { continue }
            let candidate = columns.map { Self(columns: $0, rowBlocks: blocks) }
            if best.map({ candidate.dispatches < $0.dispatches }) ?? true {
                best = candidate
            }
        }
        return best
    }

    /// `range` in `count` contiguous parts whose sizes differ by at most one.
    static func chunks(_ range: Range<Int>, count: Int) -> [Range<Int>] {
        let parts = max(1, min(count, range.count))
        let (base, extra) = range.count.quotientAndRemainder(dividingBy: parts)
        var start = range.lowerBound
        return (0 ..< parts).map { index in
            let end = start + base + (index < extra ? 1 : 0)
            defer { start = end }
            return start ..< end
        }
    }
}

extension Array where Element == ModelOptFP8ProductPiece {
    /// The products of all pieces.
    var dispatches: Int { reduce(0) { $0 + $1.dispatches } }
}
