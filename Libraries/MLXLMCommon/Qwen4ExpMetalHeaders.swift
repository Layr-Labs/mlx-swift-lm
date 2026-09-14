import Foundation

/// Native Qwen4 kernels use the exact generated MLX preambles distributed
/// with this package, without consulting a developer source checkout.
/// Sources: mlx-swift 6d6796d7a81b656d2749d39067e0a6bea2bc2986,
/// `Source/Cmlx/mlx-generated/{gemm,quantized_utils,quantized}.cpp`.
/// These preambles are byte-identical to the qualified private dependency.
/// Their Apple copyright notices and MIT license accompany the resources.
public enum Qwen4ExpMetalHeaders {
    public static let gemm = load("gemm")
    public static let quantizedUtils = load("quantized_utils")
    public static let quantized = load("quantized")

    private static func load(_ name: String) -> String {
        guard let url = Bundle.module.url(
            forResource: name, withExtension: "metal", subdirectory: "Qwen4Metal"),
            let source = try? String(contentsOf: url, encoding: .utf8),
            !source.isEmpty
        else {
            preconditionFailure("Required native Qwen4 Metal resource is unavailable: \(name)")
        }
        return source
    }
}
