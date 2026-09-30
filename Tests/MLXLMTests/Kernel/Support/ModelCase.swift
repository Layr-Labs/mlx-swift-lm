import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

/// One tiny model for the table-driven forward-pass suites.
///
/// A suite passes a list of cases to `@Test(arguments:)`, so each model
/// runs the same checks with its own configuration.
struct ModelCase: Sendable, CustomTestStringConvertible {
    /// The standard checks of `ModelCaseChecks`.
    enum Check: Sendable {
        case shape, determinism, cache, batch, causality, loading

        /// Text in the comment of each expectation that a defect in the
        /// model can fail. `run` records only these as the known issue.
        /// The control expectations of a check, for example that another
        /// seed changes the logits, fail the test also when the check has
        /// a known issue.
        var failingExpectations: [String] {
            switch self {
            case .shape: ["batch size"]
            case .determinism: ["same model, same input", "same seed, new model"]
            case .cache: ["cached logits differ"]
            case .batch: ["row 0 differs by", "row 1 differs by"]
            case .causality: ["positions before"]
            case .loading: ["loaded logits"]
            }
        }
    }

    /// The name in the test report.
    let name: String
    let vocabularySize: Int
    /// Builds the model with seeded random weights.
    let make: @Sendable (_ seed: UInt64) throws -> any LanguageModel
    /// Keys and shapes that a checkpoint of this model has.
    let expectedShapes: [String: [Int]]
    /// Extra checkpoint keys that `sanitize(weights:)` must drop.
    let droppedKeys: [String: [Int]]
    /// Converts the model's own parameters to the layout of the original
    /// checkpoint (for example one tensor per expert), which
    /// `sanitize(weights:)` must convert back.
    let checkpoint: @Sendable ([String: MLXArray]) -> [String: MLXArray]
    /// Chunks for the cache test. They cover 11 tokens.
    let chunks: [Int]
    /// Tolerance of the float32 comparisons.
    let tolerance: Float
    /// Tolerance of the loaded logits against the reference logits. It is
    /// 0 unless `sanitize(weights:)` changes values.
    let loadTolerance: Float
    /// Checks that fail because of a known production defect, with the
    /// defect. These checks run inside `withKnownIssue`, so they stay red
    /// in the report but do not fail CI, and they fail when the defect is
    /// fixed. Only the expectations in `Check.failingExpectations` are the
    /// known issue; any other failure in the check fails the test.
    let knownIssues: [Check: String]

    init(
        _ name: String, vocabularySize: Int = 64,
        expectedShapes: [String: [Int]], droppedKeys: [String: [Int]] = [:],
        chunks: [Int] = [5, 3, 1, 1, 1], tolerance: Float = 1e-4, loadTolerance: Float = 0,
        knownIssues: [Check: String] = [:],
        checkpoint: @escaping @Sendable ([String: MLXArray]) -> [String: MLXArray] = { $0 },
        make: @escaping @Sendable (_ seed: UInt64) throws -> any LanguageModel
    ) {
        self.name = name
        self.vocabularySize = vocabularySize
        self.make = make
        self.expectedShapes = expectedShapes
        self.droppedKeys = droppedKeys
        self.chunks = chunks
        self.tolerance = tolerance
        self.loadTolerance = loadTolerance
        self.knownIssues = knownIssues
        self.checkpoint = checkpoint
    }

    /// Runs `body`, inside `withKnownIssue` when `check` has a known
    /// defect. Only a failure of the expectations in
    /// `Check.failingExpectations` is the known issue. For `.loading`, a
    /// thrown error of the load is also the known issue, because a wrong
    /// `sanitize(weights:)` makes the strict update throw.
    func run(_ check: Check, _ body: () throws -> Void) rethrows {
        if let issue = knownIssues[check] {
            try withKnownIssue(Comment(rawValue: "\(name): \(issue)")) {
                try body()
            } matching: { recorded in
                (check == .loading && recorded.error != nil)
                    || recorded.isFailedExpectation(check.failingExpectations)
            }
        } else {
            try body()
        }
    }

    var testDescription: String { name }

    func row(_ seed: Int, count: Int = 11) -> [Int] {
        SyntheticModel.tokens(count: count, vocabularySize: vocabularySize, seed: seed)
    }

    /// Builds a model from a configuration dictionary and randomizes it.
    static func build<C: Decodable>(
        _ type: C.Type, _ configuration: [String: Any], seed: UInt64,
        _ initializer: (C) -> any LanguageModel
    ) throws -> any LanguageModel {
        let model = initializer(try SyntheticModel.configuration(type, configuration))
        SyntheticModel.randomize(model, seed: seed)
        return model
    }
}

/// The standard checks for a list of `ModelCase` values.
///
/// Each check is its own test function, so a failure names the check and
/// the model.
enum ModelCaseChecks {

    static func shapeAndFinite(_ c: ModelCase) throws {
        try c.run(.shape) {
            ForwardPassChecks.checkShapeDTypeAndFinite(
                try c.make(1), vocabularySize: c.vocabularySize, length: 7)
        }
    }

    static func determinism(_ c: ModelCase) throws {
        try c.run(.determinism) {
            try ForwardPassChecks.checkDeterminism(
                make: { try c.make(0) }, seed: 3, vocabularySize: c.vocabularySize)
        }
    }

    static func cacheConsistency(_ c: ModelCase) throws {
        try c.run(.cache) {
            let model = try c.make(1)
            ForwardPassChecks.checkCacheConsistency(
                model, rows: [c.row(1)], chunks: c.chunks, tolerance: c.tolerance)
            ForwardPassChecks.checkCacheConsistency(
                model, rows: [c.row(1), c.row(2)], chunks: c.chunks, tolerance: c.tolerance)
        }
    }

    static func batchInvariance(_ c: ModelCase) throws {
        try c.run(.batch) {
            ForwardPassChecks.checkBatchInvariance(
                try c.make(1), rowA: c.row(1), rowB: c.row(2), tolerance: c.tolerance)
        }
    }

    static func causality(_ c: ModelCase) throws {
        try c.run(.causality) {
            ForwardPassChecks.checkCausality(
                try c.make(1), row: c.row(1), position: 6, vocabularySize: c.vocabularySize,
                tolerance: c.tolerance)
        }
    }

    /// Checks the parameter keys and shapes, loads the model's parameters
    /// in the checkpoint layout, plus the keys that `sanitize(weights:)`
    /// must drop, through `loadWeights`, and checks that a wrong shape is
    /// rejected. Only the load and the loaded logits can be a known issue.
    static func loading(_ c: ModelCase) throws {
        let reference = try c.make(5)
        let parameters = SyntheticModel.flatParameters(reference)
        for (key, shape) in c.expectedShapes {
            #expect(parameters[key]?.shape == shape, "\(c.name): \(key)")
        }

        var checkpoint = c.checkpoint(parameters)
        for (key, shape) in c.droppedKeys {
            checkpoint[key] = MLXArray.zeros(shape)
        }
        let loaded = try c.make(6)
        let rows = [c.row(3)]
        try c.run(.loading) {
            try SyntheticModel.load(checkpoint, into: loaded)
            #expect(
                SyntheticModel.maxAbsDifference(
                    ForwardPassChecks.logits(reference, rows),
                    ForwardPassChecks.logits(loaded, rows))
                    <= c.loadTolerance, "\(c.name): loaded logits")
        }

        let (key, value) = parameters.sorted { $0.key < $1.key }.first { $0.value.ndim == 2 }!
        var wrong = parameters
        wrong[key] = MLXArray.zeros([value.dim(0) + 1, value.dim(1)])
        #expect(throws: (any Error).self, "\(c.name): \(key) with a wrong shape") {
            try SyntheticModel.load(wrong, into: try c.make(6))
        }
    }
}

extension Issue {
    /// True when this issue is a failed expectation with a comment that
    /// contains one of `texts`.
    ///
    /// Pass it as the `matching:` predicate of `withKnownIssue`. Only the
    /// named expectations are then the known issue. Another failed
    /// expectation in the block, or a thrown error, fails the test.
    func isFailedExpectation(_ texts: [String]) -> Bool {
        guard case .expectationFailed = kind else { return false }
        return comments.contains { comment in texts.contains { comment.rawValue.contains($0) } }
    }
}

/// Checkpoint layout conversions for `ModelCase.checkpoint`.
enum CheckpointLayout {

    /// Splits each stacked expert tensor `<prefix><stacked>.<name>.<suffix>`
    /// with shape `[E, ...]` into `E` tensors
    /// `<prefix><perExpert>.<e>.<newName>.<suffix>`. `names` maps the
    /// stacked projection names to the per-expert names.
    static func splitExperts(
        _ weights: [String: MLXArray], stacked: String, perExpert: String,
        names: [String: String] = [
            "gate_proj": "gate_proj", "up_proj": "up_proj", "down_proj": "down_proj",
        ]
    ) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            var matched = false
            for (name, newName) in names where key.contains(".\(stacked).\(name).") {
                for expert in 0 ..< value.dim(0) {
                    let newKey = key.replacingOccurrences(
                        of: ".\(stacked).\(name).", with: ".\(perExpert).\(expert).\(newName).")
                    result[newKey] = value[expert]
                }
                matched = true
            }
            if !matched {
                result[key] = value
            }
        }
        return result
    }
}
