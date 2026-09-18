// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import Testing

/// Keep the strict oracle while making a failed large-tensor comparison
/// actionable. Only synthetic test values are printed, never model weights.
func gemma4ExpectExactBytes(_ actual: MLXArray, _ expected: MLXArray, label: String,
                           sourceLocation: SourceLocation = #_sourceLocation) {
    #expect(actual.shape == expected.shape && actual.dtype == expected.dtype,
            Comment(rawValue: label), sourceLocation: sourceLocation)
    let a = actual.asData(access: .copy).data
    let b = expected.asData(access: .copy).data
    if a != b {
        var mismatches = 0
        var samples: [[String: Int]] = []
        a.withUnsafeBytes { (ap: UnsafeRawBufferPointer) in
            b.withUnsafeBytes { (bp: UnsafeRawBufferPointer) in
                for byte in stride(from: 0, to: min(ap.count, bp.count) - 1, by: 2) {
                    let av = ap.loadUnaligned(fromByteOffset: byte, as: UInt16.self)
                    let bv = bp.loadUnaligned(fromByteOffset: byte, as: UInt16.self)
                    if av != bv {
                        mismatches += 1
                        if samples.count < 8 {
                            samples.append(["element": byte / 2, "actual_bits": Int(av), "expected_bits": Int(bv)])
                        }
                    }
                }
            }
        }
        let detail: [String: Any] = ["case": label, "shape": actual.shape,
            "mismatching_16bit_words": mismatches, "first_mismatches": samples]
        if let data = try? JSONSerialization.data(withJSONObject: detail, options: [.sortedKeys]),
           let json = String(data: data, encoding: .utf8) {
            print("GEMMA_NATIVE_MISMATCH " + json)
        }
    }
    #expect(a == b, Comment(rawValue: label), sourceLocation: sourceLocation)
}
