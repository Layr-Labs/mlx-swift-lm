import Foundation
import MLX
import Testing

@testable import MLXLMCommon

/// Explicit research gate: the existing paged decode reduction is NOT a
/// qualified substitute for native canvas SDPA. Preserve a failing result;
/// do not substitute the older paged kernel's tolerance for this exact gate.
@Suite("DiffusionGemma paged reduction probe", .serialized)
struct DiffusionGemmaPagedAttentionProbeTests {
    private func values(_ shape: [Int], seed: UInt32, dtype: DType) -> MLXArray {
        var state = seed
        let data = (0 ..< shape.reduce(1, *)).map { _ -> Float in
            state ^= state << 13
            state ^= state >> 17
            state ^= state << 5
            return Float(state & 0x00ff_ffff) / Float(0x0100_0000) * 2 - 1
        }
        return MLXArray(data).reshaped(shape).asType(dtype)
    }

    @Test(
        .enabled(if: ProcessInfo.processInfo.environment["DARKBLOOM_DIFFUSION_PAGED_PROBE"] == "1"))
    func existingPagedReductionAgainstNativeCanvasSDPA() throws {
        _ = try #require(
            Bundle.module.url(forResource: "diffusiongemma-text-config", withExtension: "json"))
        let source = try PagedAttentionResources.loadSourceForCurrentProcess()
        for dtype in [DType.bfloat16, .float32] {
            for (dimension, kvHeads, prefix) in [(256, 8, 1023), (512, 2, 1024)] {
                for canvas in [4, 256] {
                    try compare(
                        dimension: dimension, kvHeads: kvHeads, prefix: prefix,
                        canvas: canvas, dtype: dtype, source: source)
                }
            }
        }
    }

    private func compare(
        dimension: Int, kvHeads: Int, prefix: Int, canvas: Int,
        dtype: DType, source: String
    ) throws {
        try MLX.withError { errors in
            let heads = 16
            let pageSize = 16
            let count = prefix + canvas
            let pages = (count + pageSize - 1) / pageSize
            let q = MLXFast.rmsNorm(
                values([1, heads, canvas, dimension], seed: 7419, dtype: dtype), weight: .mlxNone,
                eps: 1e-6)
            let k = MLXFast.rmsNorm(
                values([1, kvHeads, count, dimension], seed: 8132, dtype: dtype), weight: .mlxNone,
                eps: 1e-6)
            let v = values([1, kvHeads, count, dimension], seed: 5297, dtype: dtype)
            let kSlab = MLXArray.zeros([pages, kvHeads, pageSize, dimension], dtype: dtype)
            let vSlab = MLXArray.zeros([pages, kvHeads, pageSize, dimension], dtype: dtype)
            // Reverse physical page placement proves real indirection, not a
            // contiguous pointer accidentally hidden behind a paged interface.
            let pageTable = (0 ..< pages).map { Int32(pages - 1 - $0) }
            let slots = (0 ..< count).map {
                pageTable[$0 / pageSize] * Int32(pageSize) + Int32($0 % pageSize)
            }
            let fence = try PagedAttentionKernel.bulkWrite(
                kSlab: kSlab, vSlab: vSlab,
                keys: k.squeezed(axis: 0), values: v.squeezed(axis: 0), slots: MLXArray(slots),
                prevFence: MLXArray([Int32(0)]), pageSize: pageSize, kernelSource: source)
            try errors.check()
            eval(fence)
            try errors.check()
            let beforeK = kSlab.asArray(Float.self).map(\.bitPattern)
            let beforeV = vSlab.asArray(Float.self).map(\.bitPattern)
            var padded = pageTable
            while padded.count < 8 { padded.append(-1) }
            let tables = broadcast(
                MLXArray(padded).reshaped(1, padded.count), to: [canvas, padded.count])
            let descriptors = Array(
                repeating: PagedAttentionKernel.SeqInfoRow(
                    attendStart: 0, attendLength: count, tableLength: pages), count: canvas)
            let (seqinfo, maxLength) = PagedAttentionKernel.seqinfo(descriptors)
            let result = try PagedAttentionKernel.decode(
                queries: q.transposed(0, 2, 1, 3).reshaped(canvas, heads, dimension),
                kSlab: kSlab, vSlab: vSlab, tables: tables, seqinfo: seqinfo,
                maxAttendLength: maxLength, sinks: nil,
                params: MLXArray([Float(0), 1, 0, 0, 0, 0, 0, 0]), softcap: false,
                pageSize: pageSize, writeFence: fence, kernelSource: source)
            let candidate = result.out.reshaped(1, canvas, heads, dimension).transposed(0, 2, 1, 3)
            let reference = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: 1, mask: .none)
            try errors.check()
            eval(candidate, reference)
            try errors.check()
            let actual = candidate.asArray(Float.self)
            let expected = reference.asArray(Float.self)
            let mismatches = zip(actual, expected).filter { $0.bitPattern != $1.bitPattern }.count
            let largest = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
            let stateUnchanged =
                beforeK == kSlab.asArray(Float.self).map(\.bitPattern)
                && beforeV == vSlab.asArray(Float.self).map(\.bitPattern)
            let row: [String: Any] = [
                "headDim": dimension, "kvHeads": kvHeads, "queryHeads": heads,
                "prefix": prefix, "canvas": canvas, "dtype": String(describing: dtype),
                "mismatches": mismatches, "elements": actual.count, "maxAbs": Double(largest),
                "stateUnchanged": stateUnchanged, "qualified": mismatches == 0 && stateUnchanged,
                "scope":
                    "existing paged scalar reduction versus native full-canvas SDPA; not a performance result",
            ]
            print(
                "DIFFUSION_PAGED_PROBE "
                    + String(
                        decoding: try JSONSerialization.data(
                            withJSONObject: row, options: [.sortedKeys]), as: UTF8.self))
            #expect(stateUnchanged && result.nextWriteFence == nil)
            #expect(
                mismatches == 0,
                "Existing paged reduction is not lossless for this native canvas shape; do not enable it"
            )
        }
    }
}
