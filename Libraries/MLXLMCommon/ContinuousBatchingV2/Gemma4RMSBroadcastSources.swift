// Copyright © 2026 Eigen Labs.
// NORM-TB1 from challenge27c821c4: retain the reduction, replace its broadcast.
import Foundation

enum Gemma4RMSBroadcastSources {
    /// Every reduction gets a distinct scratch plane, including each member
    /// of a paired reduction. No reader can race a later reduction's writer.
    /// This also preserves the current paired-RMS option rather than replacing it.
    static func transform(_ original: String) -> String? {
        guard !original.contains("tb_sums_"),
            original.components(separatedBy: "threadgroup float local_sums[32];").count == 2,
            original.components(separatedBy: "threadgroup float local_inv[1];").count
                + original.components(separatedBy: "threadgroup float local_inv[2];").count == 3
        else { return nil }
        struct Replacement {
            let range: Range<String.Index>
            let src: String
            let slot: Int
            let local: Bool
            let paired: Bool
        }
        var matches: [Replacement] = []
        for (src, slot, local) in [("x", 0, false), ("a", 0, false), ("b", 1, false),
                                   ("sv", 0, true), ("outv", 0, true)] {
            var old = Gemma4DecodeGlueSources.rmsReduce(src, into: "local_inv[\(slot)]")
            if local { old = old.replacingOccurrences(of: "(float)\(src)[base + i]", with: "(float)\(src)[i]") }
            guard original.components(separatedBy: old).count <= 2 else { return nil }
            if let range = original.range(of: old) {
                matches.append(.init(range: range, src: src, slot: slot, local: local, paired: false))
            }
        }
        let paired = Gemma4DecodeGlueSources.pairedRmsSource
        guard original.components(separatedBy: paired).count <= 2 else { return nil }
        if let range = original.range(of: paired) {
            matches.append(.init(range: range, src: "", slot: 0, local: false, paired: true))
        }
        matches.sort { $0.range.lowerBound < $1.range.lowerBound }
        let planes = matches.reduce(0) { $0 + ($1.paired ? 2 : 1) }
        guard (1...4).contains(planes),
            original.components(separatedBy: "metal::precise::rsqrt").count == planes + 1
        else { return nil }
        var result = "", cursor = original.startIndex, plane = 0
        for match in matches {
            guard cursor <= match.range.lowerBound else { return nil }
            result += original[cursor..<match.range.lowerBound]
            result += match.paired ? pairedReduction(plane: plane)
                : reduction(src: match.src, slot: match.slot, local: match.local, plane: plane)
            plane += match.paired ? 2 : 1
            cursor = match.range.upperBound
        }
        result += original[cursor...]
        result = result.replacingOccurrences(of: "threadgroup float local_sums[32];",
            with: (0..<planes).map { "threadgroup float tb_sums_\($0)[32];" }.joined(separator: "\n"))
        for count in [1, 2] {
            result = result.replacingOccurrences(of: "threadgroup float local_inv[\(count)];",
                                                 with: "float local_inv[\(count)];")
        }
        guard !result.contains("threadgroup float local_inv"), !result.contains("local_sums["),
            !result.contains("local_sums_b["), !result.contains("if (simd_group_id == 0)") else { return nil }
        return result
    }

    private static func reduction(src: String, slot: Int, local: Bool, plane: Int) -> String {
        """
        {
            float acc = 0;
            for (int i = 0; i < 4; i++) {
                float xi = (float)\(src)[\(local ? "i" : "base + i")];
                acc += xi * xi;
            }
            acc = simd_sum(acc);
            if (simd_lane_id == 0) tb_sums_\(plane)[simd_group_id] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            acc = simd_sum(simd_lane_id < 22 ? tb_sums_\(plane)[simd_lane_id] : 0.0f);
            local_inv[\(slot)] = metal::precise::rsqrt(fma(acc, (1.0f / 2816.0f), 1e-06f));
        }
        """
    }

    private static func pairedReduction(plane: Int) -> String {
        """
        float av[4];
        float bv[4];
        {
            float acc_a = 0;
            float acc_b = 0;
            for (int i = 0; i < 4; i++) {
                av[i] = (float)a[base + i];
                bv[i] = (float)b[base + i];
                acc_a += av[i] * av[i];
                acc_b += bv[i] * bv[i];
            }
            acc_a = simd_sum(acc_a);
            acc_b = simd_sum(acc_b);
            if (simd_lane_id == 0) {
                tb_sums_\(plane)[simd_group_id] = acc_a;
                tb_sums_\(plane + 1)[simd_group_id] = acc_b;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            acc_a = simd_sum(simd_lane_id < 22 ? tb_sums_\(plane)[simd_lane_id] : 0.0f);
            acc_b = simd_sum(simd_lane_id < 22 ? tb_sums_\(plane + 1)[simd_lane_id] : 0.0f);
            local_inv[0] = metal::precise::rsqrt(fma(acc_a, (1.0f / 2816.0f), 1e-06f));
            local_inv[1] = metal::precise::rsqrt(fma(acc_b, (1.0f / 2816.0f), 1e-06f));
        }
        """
    }
}
