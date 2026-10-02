import Foundation
import Testing

/// Runs a test only on reference hardware.
///
/// Some tests compare the raw bits of floating-point results with frozen
/// references. The GPU and its Metal compiler can change the order of
/// floating-point operations, so other hardware can give results that differ
/// in the last bits. Set `MLX_REFERENCE_HARDWARE=1` on a Mac of the kind
/// that recorded the references. On other machines, these tests are skipped.
extension Trait where Self == ConditionTrait {
    static var referenceHardware: Self {
        .enabled(
            if: ProcessInfo.processInfo.environment["MLX_REFERENCE_HARDWARE"] == "1",
            "Compares float bits with references from other hardware. Set MLX_REFERENCE_HARDWARE=1 on reference hardware to run it."
        )
    }
}
