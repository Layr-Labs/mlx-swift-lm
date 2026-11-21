// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0

import CryptoKit
import Foundation
import MLX
import MLXLLM
import MLXRandom
import XCTest

/// SHA-256 of `Inkling2PagedAttention` outputs over a fixed set of calls
/// (decode, verify and prefill shapes; sliding and full; both layouts; split
/// and sequential kernels; slabs and segments; fused writes). Two builds whose
/// digests agree produce the same bits for every call in the set. Opt-in:
/// `INKLING2_ATTN_DIGEST=1`.
final class Inkling2PagedAttentionDigestTests: XCTestCase {
    typealias Attention = Inkling2PagedAttention

    func testOutputDigest() throws {
        guard ProcessInfo.processInfo.environment["INKLING2_ATTN_DIGEST"] == "1" else {
            throw XCTSkip("set INKLING2_ATTN_DIGEST=1")
        }
        let (qh, kvh, d, s) = (24, 8, 128, 16)
        let scale = Float(1.0 / Double(d).squareRoot())
        MLXRandom.seed(11)
        let pages = 512
        let kSlab = (MLXRandom.normal([pages, kvh, s, d]) * 0.6).asType(.bfloat16)
        let vSlab = MLXRandom.normal([pages, kvh, s, d]).asType(.bfloat16)
        let proj = (MLXRandom.normal([16, 1024]) * 0.5).asType(.bfloat16)
        eval(kSlab, vSlab, proj)
        let contexts = [37, 700, 2100]
        var total = SHA256()
        var lines: [String] = []

        func digest(_ name: String, _ arrays: [MLXArray]) {
            eval(arrays)
            var one = SHA256()
            for a in arrays {
                let data = a.asData(access: .copy).data
                one.update(data: data)
                total.update(data: data)
            }
            let hex = one.finalize().map { String(format: "%02x", $0) }.joined()
            lines.append("\(name) \(hex.prefix(16))")
        }

        func tables(window: Int?, l: Int) -> (MLXArray, [Attention.Row]) {
            var flat: [Int32] = []
            var rows: [Attention.Row] = []
            let width = 160
            for (r, ctx) in contexts.enumerated() {
                let need = (ctx + l + s - 1) / s
                let length = window == nil ? need : 65
                var table = [Int32](repeating: 0, count: width)
                for j in 0 ..< length { table[j] = Int32((r * 170 + j) % (pages - 1) + 1) }
                flat += table
                rows.append(.init(queryStart: ctx, tableLength: length))
            }
            return (MLXArray(flat, [contexts.count, width]), rows)
        }

        func inputs(_ l: Int, _ layout: Attention.Layout, seed: UInt64) -> [MLXArray] {
            MLXRandom.seed(seed)
            let b = contexts.count
            func r(_ heads: Int, _ inner: Int, _ scale: Float) -> MLXArray {
                let x = (MLXRandom.normal([b, l, heads, inner]) * scale).asType(.bfloat16)
                return layout == .tokenMajor ? x : x.transposed(0, 2, 1, 3).contiguous()
            }
            let sinks = MLXRandom.normal([b, l, qh])
            return [
                r(qh, d, 0.6), r(kvh, d, 0.6), r(kvh, d, 1), r(qh, 16, 0.5),
                layout == .tokenMajor ? sinks : sinks.transposed(0, 2, 1).contiguous(),
            ]
        }

        let fence = MLXArray.zeros([1], dtype: .int32)
        var seed: UInt64 = 100
        for window in [nil, 1024] as [Int?] {
            for l in [1, 4, 16, 40] {
                for layout in [Attention.Layout.tokenMajor, .headMajor] {
                    var kernels: [Attention.Kernel] = [.automatic]
                    if layout == .tokenMajor && l == 4 { kernels.append(.sequential) }
                    if layout == .tokenMajor && l == 40 { kernels.append(.split) }
                    for kernel in kernels {
                        seed += 1
                        let x = inputs(l, layout, seed: seed)
                        let (table, rows) = tables(window: window, l: l)
                        let out = Attention.attend(
                            queries: x[0], newKeys: x[1], newValues: x[2], relativeStates: x[3],
                            proj: proj, sinks: x[4],
                            cache: .init(
                                keys: kSlab, values: vSlab, tables: table, writeFence: fence),
                            rows: rows, window: window, scale: scale, layout: layout, kernel: kernel
                        )
                        digest("w\(window ?? 0) L\(l) \(layout) \(kernel)", [out.output])
                    }
                }
            }
        }
        // One segment holding the slab's pages: entries are `0 << 24 | page`.
        let segment = stacked([kSlab, vSlab])
        eval(segment)
        for l in [1, 40] {
            seed += 1
            let x = inputs(l, .tokenMajor, seed: seed)
            let (table, rows) = tables(window: nil, l: l)
            let out = Attention.attend(
                queries: x[0], newKeys: x[1], newValues: x[2], relativeStates: x[3], proj: proj,
                sinks: x[4], cache: .init(segments: [segment], tables: table, writeFence: fence),
                rows: rows, window: nil, scale: scale, layout: .tokenMajor)
            digest("segments L\(l)", [out.output])
        }
        // Fused writes into copies of the slabs: output and written slabs.
        for (window, l) in [(nil, 1), (1024, 4)] as [(Int?, Int)] {
            seed += 1
            let x = inputs(l, .tokenMajor, seed: seed)
            let (table, rows) = tables(window: window, l: l)
            let k = kSlab + MLXArray.zeros(kSlab.shape, dtype: kSlab.dtype)
            let v = vSlab + MLXArray.zeros(vSlab.shape, dtype: vSlab.dtype)
            eval(k, v)
            let out = Attention.attend(
                queries: x[0], newKeys: x[1], newValues: x[2], relativeStates: x[3], proj: proj,
                sinks: x[4], cache: .init(keys: k, values: v, tables: table, writeFence: fence),
                rows: rows, window: window, scale: scale, write: true,
                layout: .tokenMajor)
            eval(out.output, out.writeFence)
            digest("write w\(window ?? 0) L\(l)", [out.output, k, v])
        }
        let hex = total.finalize().map { String(format: "%02x", $0) }.joined()
        print("[Inkling2AttnDigest] " + lines.joined(separator: "\n[Inkling2AttnDigest] "))
        print("[Inkling2AttnDigest] TOTAL \(hex)")
    }
}
