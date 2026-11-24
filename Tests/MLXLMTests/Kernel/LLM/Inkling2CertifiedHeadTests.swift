import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

/// Inkling 2 certified head on a synthetic unembedding: tokens and non-finite flags equal those of
/// the full product (`softcap(rowProduct(h, W))`, MLX `argMax`, the engine's non-finite rule) for
/// greedy and temperature-only rows, including exact ties and non-finite, infinite, huge and zero
/// hidden rows; the public primitives bound and reproduce the product.
extension KernelTests {
    @Suite(.serialized)
    struct Inkling2CertifiedHeadTests {
        static let n = 4096
        static let k = 512
        static let cap: Float = 30

        /// Rows of varied norm; rows 10 and 3000 duplicate row 5 (exact ties).
        nonisolated(unsafe) static let weight: MLXArray = {
            let base = MLXRandom.normal([n, k], key: MLXRandom.key(7)) * Float(0.05)
            let scale = MLXRandom.uniform(
                low: Float(0.5), high: Float(2), [n, 1], key: MLXRandom.key(8))
            var index = Array(Int32(0) ..< Int32(n))
            index[10] = 5
            index[3000] = 5
            let w = take((base * scale).asType(.bfloat16), MLXArray(index), axis: 0)
            eval(w)
            return w
        }()

        static let head: Inkling2CertifiedHead = {
            try! Inkling2CertifiedHead(weight: weight) { Inkling2Math.softcap($0, cap) }
        }()

        static func row(_ values: MLXArray) -> MLXArray {
            values.reshaped([1, k]).asType(.bfloat16)
        }

        /// Ordinary rows plus the special ones: a tie on row 5, NaN, +inf and -inf elements, a huge
        /// finite element (norm past 1e30), and an all-zero row.
        nonisolated(unsafe) static let hidden: MLXArray = {
            let normal = MLXRandom.normal([12, k], key: MLXRandom.key(9)).asType(.bfloat16)
            func with(_ value: Float, at column: Int, of r: Int) -> MLXArray {
                let mask = MLXArray(Array(0 ..< k).map { $0 == column }).reshaped([1, k])
                return which(mask, MLXArray(value), normal[r ..< r + 1].asType(.float32))
                    .asType(.bfloat16)
            }
            let tie = row(weight[5].asType(.float32) * Float(40))
            let rows = [
                normal[0 ..< 4], tie, with(.nan, at: 3, of: 4), with(.infinity, at: 0, of: 5),
                with(-.infinity, at: 7, of: 6), with(1e35, at: 1, of: 7),
                MLXArray.zeros([1, k], dtype: .bfloat16), normal[8 ..< 12],
            ]
            let h = concatenated(rows, axis: 0)
            eval(h)
            return h
        }()

        static func full(_ h: MLXArray) -> MLXArray {
            Inkling2Kernels.$override.withValue(true) {
                Inkling2Kernels.rowProduct(h, transposedWeight: weight)!
            }
        }

        static func noise(rows: Int, seed: UInt64) -> MLXArray {
            let u = MLXRandom.uniform(
                low: Float(0), high: Float(1), [rows, n], key: MLXRandom.key(seed))
            return maximum(-log(1 - u), MLXArray(Float(1e-20)))
        }

        static func ints(_ x: MLXArray) -> [Int32] { x.asType(.int32).asArray(Int32.self) }

        @Test(arguments: [1, 3, 8, 17])
        func exactProductsEqualRowProductBits(rows: Int) {
            let h = MLXRandom.normal([rows, Self.k], key: MLXRandom.key(UInt64(rows))).asType(
                .bfloat16)
            let all = MLXArray(Array(UInt32(0) ..< UInt32(Self.n)))
            let mine = Self.head.exactProducts(h, indices: all, count: MLXArray([UInt32(Self.n)]))
            let theirs = Self.full(h)
            #expect(
                arrayEqual(mine.view(dtype: .uint16), theirs.view(dtype: .uint16)).item(Bool.self))
        }

        @Test func greedyTokensAndFlagsEqualTheFullProduct() {
            let h = Self.hidden
            for size in [1, 2, 5, 8, 17] {
                for start in stride(from: 0, to: h.dim(0), by: size) {
                    let x = h[start ..< min(h.dim(0), start + size)]
                    let f = Inkling2Math.softcap(Self.full(x).asType(.float32), Self.cap)
                    let s = Self.head.greedy(x)
                    #expect(
                        Self.ints(s.tokens) == Self.ints(argMax(f, axis: -1)),
                        "rows \(start)+\(size)")
                    #expect(
                        Self.ints(s.nonFinite) == Self.ints(CBv2NonFiniteLogitsMirror.flags(f)),
                        "rows \(start)+\(size)")
                }
            }
            // The tie row picks the first of rows 5, 10 and 3000.
            #expect(Self.ints(Self.head.greedy(Self.hidden[4 ..< 5]).tokens) == [5])
        }

        @Test(arguments: [Float(0.6), 1.0])
        func temperatureRowsEqualTheSamplerOps(temperature: Float) {
            let h = Self.hidden
            for size in [1, 4, 8] {
                for start in stride(from: 0, to: h.dim(0), by: size) {
                    let x = h[start ..< min(h.dim(0), start + size)]
                    let r = x.dim(0)
                    let greedy = (0 ..< r).map { $0 % 3 == 2 }
                    let t = MLXArray(greedy.map { $0 ? Float(1) : temperature }, [r, 1])
                    let e = Self.noise(rows: r, seed: UInt64(100 + start))
                    let noise = which(MLXArray(greedy).reshaped([r, 1]), MLXArray(Float(1)), e)
                    let f = Inkling2Math.softcap(Self.full(x).asType(.float32), Self.cap)
                    let raced = argMax(softmax(f / t, axis: -1) / noise, axis: -1)
                    let reference = which(MLXArray(greedy), argMax(f, axis: -1), raced)
                    let s = Self.head.select(x, greedy: greedy, temperature: t, noise: noise)
                    #expect(Self.ints(s.tokens) == Self.ints(reference), "rows \(start)+\(size)")
                    #expect(
                        Self.ints(s.nonFinite) == Self.ints(CBv2NonFiniteLogitsMirror.flags(f)),
                        "rows \(start)+\(size)")
                }
            }
        }

        /// This head's census entries: the census is process-wide, and other suites running
        /// concurrently build heads of their own. A variant specialised on K or N belongs to this
        /// head only with this head's K and N; the race verdict kernel (no K or N) is the same
        /// library for every head.
        static func censusOfThisHead() -> Set<String> {
            Inkling2CertifiedHead.censusVariants.filter { key in
                (!key.contains("_K=") || key.contains("_K=\(k)_"))
                    && (!key.contains("_N=") || key.contains("_N=\(n)_"))
            }
        }

        /// Row counts are runtime values in every head kernel: after one warm-up, greedy,
        /// temperature-only and primitive calls at every row count 1 ... 64 ask for no new kernel
        /// library, and their tokens and flags still equal the full product's.
        @Test func noNewKernelVariantAfterWarmUpAtAnyRowCount() {
            Inkling2CertifiedHead.censusEnabled = true
            defer { Inkling2CertifiedHead.censusEnabled = false }
            Self.head.warm()
            let warmed = Self.censusOfThisHead()
            // The warm-up was recorded for every kernel of this head.
            let kernels = Set(warmed.map { String($0.prefix { $0 != "_" }) })
            #expect(
                kernels == [
                    "proxyKernel", "proxyTileKernel", "compactKernel", "exactKernel",
                    "selectKernel",
                    "scoreKernel", "maskKernel", "racePartialKernel", "raceFinalKernel",
                ], "\(kernels.sorted())")
            for r in 1 ... 64 {
                let x = MLXRandom.normal([r, Self.k], key: MLXRandom.key(UInt64(500 + r)))
                    .asType(.bfloat16)
                let f = Inkling2Math.softcap(Self.full(x).asType(.float32), Self.cap)
                let g = Self.head.greedy(x)
                #expect(Self.ints(g.tokens) == Self.ints(argMax(f, axis: -1)), "greedy rows \(r)")
                #expect(
                    Self.ints(g.nonFinite) == Self.ints(CBv2NonFiniteLogitsMirror.flags(f)),
                    "greedy rows \(r)")
                let greedy = (0 ..< r).map { $0 % 3 == 2 }
                let t = MLXArray(greedy.map { $0 ? Float(1) : Float(0.8) }, [r, 1])
                let noise = which(
                    MLXArray(greedy).reshaped([r, 1]), MLXArray(Float(1)),
                    Self.noise(rows: r, seed: UInt64(900 + r)))
                let raced = argMax(softmax(f / t, axis: -1) / noise, axis: -1)
                let reference = which(MLXArray(greedy), argMax(f, axis: -1), raced)
                let s = Self.head.select(x, greedy: greedy, temperature: t, noise: noise)
                #expect(Self.ints(s.tokens) == Self.ints(reference), "race rows \(r)")
                let bounds = Self.head.proxyBounds(x, envelope: .identity())
                let list = Self.head.candidates(
                    upper: bounds.upper, atLeast: bounds.lower.max(axis: -1))
                let listed = Self.head.proxyBounds(x, proxy: .int8, rows: list)
                let exact = Self.head.exactProducts(x, indices: list.indices, count: list.count)
                eval(listed.lower, exact)
            }
            let after = Self.censusOfThisHead()
            #expect(after == warmed, "new variants: \(after.subtracting(warmed).sorted())")
            #expect(warmed.count <= 48, "\(warmed.count) variants")
        }

        /// Locked results from several threads (MLXArray is not Sendable; the arrays are only read).
        final class Shared: @unchecked Sendable {
            let lock = NSLock()
            var builds = 0
            var heads: [ObjectIdentifier] = []
            var mismatches: [String] = []
            let inputs: [(x: MLXArray, greedy: [Bool], t: MLXArray, noise: MLXArray)]
            let weight: MLXArray?
            init(
                _ inputs: [(x: MLXArray, greedy: [Bool], t: MLXArray, noise: MLXArray)],
                weight: MLXArray? = nil
            ) {
                self.inputs = inputs
                self.weight = weight
            }
            func note(_ s: String) { lock.withLock { mismatches.append(s) } }
        }

        /// Two engines on one model share its head: concurrent first uses of the model's cell
        /// build one head; concurrent warm-ups of a head whose kernels were never compiled, and
        /// greedy, temperature-only and primitive calls from several threads (with the census
        /// switched on and off meanwhile) give exactly the single-threaded tokens and flags.
        @Test func concurrentUseFromSeveralThreadsMatchesSerialResults() throws {
            let cell = Inkling2CertifiedHeadCell()
            let fresh = MLXRandom.normal([2056, 256], key: MLXRandom.key(41)).asType(.bfloat16)
            eval(fresh)
            let built = Shared([], weight: fresh)
            DispatchQueue.concurrentPerform(iterations: 6) { _ in
                let head = cell.get {
                    built.lock.withLock { built.builds += 1 }
                    return try Inkling2CertifiedHead(weight: built.weight!) { $0 }
                }
                if let head { built.lock.withLock { built.heads.append(ObjectIdentifier(head)) } }
            }
            #expect(built.builds == 1)
            #expect(built.heads.count == 6 && Set(built.heads).count == 1)
            let freshHead = try #require(cell.get { throw Inkling2Bytes.Failure.unsupported("") })
            DispatchQueue.concurrentPerform(iterations: 2) { _ in freshHead.warm() }

            let inputs = [1, 3, 8, 17, 33].map { r in
                let x = MLXRandom.normal([r, Self.k], key: MLXRandom.key(UInt64(700 + r)))
                    .asType(.bfloat16)
                let greedy = (0 ..< r).map { $0 % 2 == 0 }
                let t = MLXArray(greedy.map { $0 ? Float(1) : Float(0.7) }, [r, 1])
                let noise = which(
                    MLXArray(greedy).reshaped([r, 1]), MLXArray(Float(1)),
                    Self.noise(rows: r, seed: UInt64(800 + r)))
                eval(x, t, noise)
                return (x: x, greedy: greedy, t: t, noise: noise)
            }
            let shared = Shared(inputs)
            @Sendable func run(_ i: Int) -> [[Int32]] {
                let input = shared.inputs[i]
                let g = Self.head.greedy(input.x)
                let s = Self.head.select(
                    input.x, greedy: input.greedy, temperature: input.t, noise: input.noise)
                let b = Self.head.proxyBounds(input.x, envelope: .identity())
                let c = Self.head.candidates(upper: b.upper, atLeast: b.lower.max(axis: -1))
                let e = Self.head.exactProducts(input.x, indices: c.indices, count: c.count)
                // The candidate list is built by atomic appends (any order): compare it sorted.
                let n = Int(c.count.item(UInt32.self))
                let listed = sorted(c.indices[0 ..< n].asType(.int32))
                return [
                    Self.ints(g.tokens), Self.ints(g.nonFinite), Self.ints(s.tokens),
                    Self.ints(s.nonFinite), Self.ints(listed),
                    Self.ints(take(e, listed, axis: 1).view(dtype: .uint16).reshaped(-1)),
                ]
            }
            let count = shared.inputs.count
            let serial = (0 ..< count).map(run)
            let wasOn = Inkling2CertifiedHead.censusEnabled
            DispatchQueue.concurrentPerform(iterations: 7) { worker in
                if worker == 6 {
                    for i in 0 ..< 200 { Inkling2CertifiedHead.censusEnabled = i % 2 == 0 }
                    return
                }
                for pass in 0 ..< 3 {
                    for j in 0 ..< count {
                        let i = (j + worker + pass) % count
                        if run(i) != serial[i] { shared.note("worker \(worker) input \(i)") }
                    }
                }
            }
            Inkling2CertifiedHead.censusEnabled = wasOn
            #expect(shared.mismatches.isEmpty, "\(shared.mismatches)")
        }

        @Test func envelopesAreMonotoneAndCoverInfinities() {
            let identity = Inkling2CertifiedHead.Envelope.identity()
            let softcap = Self.head.softcapEnvelope
            let values = MLXArray(Array(UInt16(0) ... UInt16.max), [65536]).view(dtype: .bfloat16)
                .asType(.float32).asArray(Float.self)
            let order = (0 ..< 65536).filter { !values[$0].isNaN }.sorted {
                values[$0] < values[$1]
            }
            for envelope in [identity, softcap] {
                let lower = envelope.lower.asArray(Float.self)
                let upper = envelope.upper.asArray(Float.self)
                let table = envelope.table.asArray(Float.self)
                let unordered = zip(order, order.dropFirst()).filter { a, b in
                    !(lower[a] <= lower[b] && upper[a] <= upper[b])
                }
                let outside = order.filter { !(lower[$0] <= table[$0] && table[$0] <= upper[$0]) }
                #expect(unordered.isEmpty)
                #expect(outside.isEmpty)
            }
            #expect(identity.lower.asArray(Float.self)[0xFF80] == -.infinity)
            #expect(identity.upper.asArray(Float.self)[0x7F80] == .infinity)
            #expect(softcap.lower.asArray(Float.self)[0xFF80].isFinite)
            #expect(softcap.upper.asArray(Float.self)[0x7F80].isFinite)
        }

        @Test func proxyBoundsContainTheProductAndCandidatesKeepTheMaximum() {
            let h = MLXRandom.normal([6, Self.k], key: MLXRandom.key(21)).asType(.bfloat16)
            let product = Self.full(h).asType(.float32)
            let logits = Inkling2Math.softcap(product, Self.cap)
            for proxy in [Inkling2CertifiedHead.Proxy.int6, .int8] {
                let raw = Self.head.proxyBounds(
                    h, proxy: proxy, envelope: .identity())
                #expect(all(raw.lower .<= product).item(Bool.self))
                #expect(all(product .<= raw.upper).item(Bool.self))
                let capped = Self.head.proxyBounds(h, proxy: proxy)
                #expect(all(capped.lower .<= logits).item(Bool.self))
                #expect(all(logits .<= capped.upper).item(Bool.self))
                let threshold = capped.lower.max(axis: -1)
                let list = Self.head.candidates(upper: capped.upper, atLeast: threshold)
                let count = Int(list.count.item(UInt32.self))
                let listed = Set(list.indices[0 ..< count].asArray(UInt32.self).map(Int.init))
                let upper = capped.upper.asArray(Float.self)
                let tau = threshold.asArray(Float.self)
                let expected = Set(
                    (0 ..< Self.n).filter { j in
                        (0 ..< 6).contains { upper[$0 * Self.n + j] >= tau[$0] }
                    })
                #expect(listed == expected)
                for token in Self.ints(argMax(logits, axis: -1)) {
                    #expect(listed.contains(Int(token)))
                }
                // Listed bounds: only the listed rows are written.
                let relisted = Self.head.proxyBounds(
                    h, proxy: proxy, rows: (list.indices, list.count))
                let lowerListed = relisted.lower.asArray(Float.self)
                for j in [0, 1, Self.n - 1] where !listed.contains(j) {
                    #expect(lowerListed[j] == -.infinity)
                }
            }
        }
    }
}
