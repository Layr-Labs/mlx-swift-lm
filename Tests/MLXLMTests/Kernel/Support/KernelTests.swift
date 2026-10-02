import Testing

/// The parent suite of the `kernel` tests under `Kernel/`.
///
/// Declare each suite under `Kernel/<Area>/` in an extension of this type.
/// The suite then has the tag `kernel`, and its test IDs start with
/// `MLXLMTests.KernelTests/`. This command runs only these tests:
///
/// ```bash
/// swift test --filter '\.KernelTests/'
/// ```
@Suite(.tags(.kernel))
enum KernelTests {}
