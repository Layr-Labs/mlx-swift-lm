import Testing

/// The test types of this package.
///
/// Each new test gets exactly one of these tags, on its `@Suite` or through
/// its parent suite.
/// `CONTRIBUTING.md` ("Running Tests") gives the rules for each type and how
/// CI runs it.
extension Tag {
    /// Swift logic only. Uses no MLX arrays, no GPU, no Metal library, no
    /// model weights and no network. Lives under `Unit/<Area>/`. Do not add
    /// this tag directly: declare the suite in an extension of `UnitTests`.
    @Tag static var unit: Self

    /// Needs the Metal device and the Metal library. Lives under
    /// `Kernel/<Area>/`.
    @Tag static var kernel: Self

    /// Runs an engine or a model with more than one component together, for
    /// example a scheduler with a tiny model. Lives under
    /// `Integration/<Area>/`.
    @Tag static var integration: Self

    /// Compares results bit for bit with frozen references. Also needs the
    /// `.referenceHardware` trait.
    @Tag static var reference: Self
}

/// The parent suite of every `unit` test.
///
/// Declare each unit suite in an extension of this type. The suite then has
/// the tag `unit`, and its test IDs start with `MLXLMTests.UnitTests/`. CI
/// selects the unit tests with `swift test --filter 'MLXLMTests\.UnitTests/'`,
/// because `swift test` cannot select tests by tag.
@Suite(.tags(.unit))
enum UnitTests {}
