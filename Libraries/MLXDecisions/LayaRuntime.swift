// Copyright © 2026 Eigen Labs Inc.
import Foundation
import MLX

/// CPU copies of one inference, for local qualification. Never included in HTTP answers.
public struct LayaDiagnostics: Codable, Sendable {
    public let tokenIDs: [[Int32]]
    public let markerPositions: [[Int32]]
    public let questionTypes: [Int32]
    public let logits: [[Float]]
    public let actionLogits: [[Float]]
}

/// Owns an immutable Laya checkpoint. Calls are serialized, including synchronous MLX evaluation.
/// Laya is a bidirectional decision model: it does not generate text or allocate a KV cache.
public actor LayaRuntime {
    private var model: LayaModel?
    private let prompt: LayaPrompt
    public static let maximumBatchSize = 16

    private init(directory: URL, prompt: LayaPrompt) throws {
        self.prompt = prompt
        model = try MLX.withError { try LayaModel(directory: directory) }
    }

    public static func load(directory: URL) async throws -> LayaRuntime {
        try Task.checkCancellation()
        let prompt = try await LayaPrompt(directory: directory)
        try Task.checkCancellation()
        return try LayaRuntime(directory: directory, prompt: prompt)
    }

    public func predict(data: Data) throws -> Data {
        try predict(SystemOneRequest(data: data))
    }

    public func predict(_ request: SystemOneRequest) throws -> Data {
        guard let model else { throw LayaError.invalidCheckpoint("Laya runtime is shut down") }
        let result = try evaluate(request, model: model)
        let answers = try request.questions.enumerated().map { index, question in
            DecisionJSON.Field(
                question.id,
                try LayaResponse.answer(
                    question: question,
                    logits: result.logits[index], actionLogits: result.actionLogits[index],
                    configuration: model.agent))
        }
        return Data(
            DecisionJSON.object([
                .init("model", .string(request.model)),
                .init("answers", .object(answers)),
                .init(
                    "usage",
                    .object([
                        .init(
                            "input_tokens",
                            .number(String(result.tokenIDs.reduce(0) { $0 + $1.count }))),
                        .init("output_tokens", .number("0")),
                    ])),
            ]).rendered().utf8)
    }

    public func diagnostics(data: Data) throws -> LayaDiagnostics {
        guard let model else { throw LayaError.invalidCheckpoint("Laya runtime is shut down") }
        return try evaluate(SystemOneRequest(data: data), model: model)
    }

    /// Waits behind any active evaluation and releases all checkpoint arrays before returning.
    public func shutdown() { model = nil }

    private func evaluate(_ request: SystemOneRequest, model: LayaModel) throws -> LayaDiagnostics {
        try Task.checkCancellation()
        let items = try prompt.prepare(request, config: model.agent)
        var logits: [[Float]] = []
        var actions: [[Float]] = []
        for start in stride(from: 0, to: items.count, by: Self.maximumBatchSize) {
            try Task.checkCancellation()
            let batch = Array(items[start ..< min(items.count, start + Self.maximumBatchSize)])
            let result = try forward(batch, model: model)
            logits.append(contentsOf: result.0)
            actions.append(contentsOf: result.1)
        }
        try Task.checkCancellation()
        return LayaDiagnostics(
            tokenIDs: items.map(\.ids), markerPositions: items.map(\.markers),
            questionTypes: items.map(\.type), logits: logits, actionLogits: actions)
    }

    private func forward(_ items: [LayaPreparedQuestion], model: LayaModel) throws -> (
        [[Float]], [[Float]]
    ) {
        let length = items.map { $0.ids.count }.max()!
        let count = max(2, items.map { $0.markers.count }.max()!)
        var ids = Array(repeating: Int32(prompt.padID), count: items.count * length)
        var valid = Array(repeating: false, count: ids.count)
        var markers = Array(repeating: Int32(0), count: items.count * count)
        var markerMask = Array(repeating: false, count: markers.count)
        for (row, item) in items.enumerated() {
            ids.replaceSubrange(row * length ..< row * length + item.ids.count, with: item.ids)
            valid.replaceSubrange(
                row * length ..< row * length + item.ids.count,
                with: repeatElement(true, count: item.ids.count))
            markers.replaceSubrange(
                row * count ..< row * count + item.markers.count, with: item.markers)
            markerMask.replaceSubrange(
                row * count ..< row * count + item.markers.count,
                with: repeatElement(true, count: item.markers.count))
        }
        return try MLX.withError { errors in
            let (logits, actions) = model(
                ids: MLXArray(ids, [items.count, length]),
                valid: MLXArray(valid, [items.count, length]),
                markers: MLXArray(markers, [items.count, count]),
                markerMask: MLXArray(markerMask, [items.count, count]),
                types: MLXArray(items.map(\.type)))
            try errors.check()
            eval(logits, actions)
            // MLX's handler records errors and returns; check before any subsequent readback.
            try errors.check()
            let flatLogits = logits.asArray(Float.self)
            let flatActions = actions.asArray(Float.self)
            try errors.check()
            guard flatLogits.allSatisfy(\.isFinite), flatActions.allSatisfy(\.isFinite) else {
                throw LayaError.nonFiniteOutput
            }
            let actionCount = actions.dim(1)
            return (
                items.indices.map { Array(flatLogits[$0 * count ..< ($0 + 1) * count]) },
                items.indices.map {
                    Array(flatActions[$0 * actionCount ..< ($0 + 1) * actionCount])
                }
            )
        }
    }
}
