import Testing

/// The parent suite of the `kernel` tests of this test target, under
/// `Kernel/`. It is the same as the `KernelTests` suite of `MLXLMTests`.
///
/// Declare each suite under `Kernel/<Area>/` in an extension of this type.
/// The suite then has the tag `kernel`, and its test IDs start with
/// `MLXLMServerTests.KernelTests/`.
@Suite(.tags(.kernel))
enum KernelTests {}
