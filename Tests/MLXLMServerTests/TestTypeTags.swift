import Testing

/// The test types of this test target. They are the same as in
/// `Tests/MLXLMTests/TestTypeTags.swift`; `CONTRIBUTING.md` ("Running Tests")
/// gives the rules for each type.
extension Tag {
    /// Swift logic only. Uses no MLX arrays, no GPU, no Metal library, no
    /// model weights and no network. Lives under `Unit/<Area>/`. Do not add
    /// this tag directly: declare the suite in an extension of `UnitTests`.
    @Tag static var unit: Self

    /// Needs the Metal device and the Metal library.
    @Tag static var kernel: Self

    /// Runs an engine or a model with more than one component together.
    @Tag static var integration: Self

    /// Compares results bit for bit with frozen references.
    @Tag static var reference: Self
}

/// The parent suite of every `unit` test of this test target.
///
/// Declare each unit suite in an extension of this type. Its test IDs start
/// with `MLXLMServerTests.UnitTests/`, and CI selects them with
/// `swift test --filter '\.UnitTests/'`.
@Suite(.tags(.unit))
enum UnitTests {}
