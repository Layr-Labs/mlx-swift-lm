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

    /// Release/installer smoke must exercise this lookup from the packaged
    /// executable before an inference process reaches a static kernel header.
    public static func validateResources() throws {
        for name in Qwen4ExpMetalResources.names {
            _ = try Qwen4ExpMetalResources.load(name)
        }
    }

    private static func load(_ name: String) -> String {
        do {
            return try Qwen4ExpMetalResources.load(name)
        } catch {
            preconditionFailure("\(error)")
        }
    }
}
