// Copyright © 2026 Eigen Labs Inc.
import Foundation

/// The checkpoint's calibrated answer mapping, separate from tensor execution.
enum LayaResponse {
    static func answer(
        question: SystemOneRequest.Question, logits: [Float],
        actionLogits: [Float], configuration: LayaAgentConfiguration
    ) throws -> DecisionJSON {
        let count = question.labels.count
        guard logits.count >= count, !actionLogits.isEmpty,
            logits.allSatisfy(\.isFinite), actionLogits.allSatisfy(\.isFinite)
        else { throw LayaError.nonFiniteOutput }
        let bucket = count <= 2 ? "2" : count <= 5 ? "3-5" : count <= 10 ? "6-10" : "11+"
        let temperature =
            configuration.temperature_by_options[question.type + ":" + bucket]
            ?? configuration.temperature[question.typeIndex]
        let probabilities = softmax(
            logits.prefix(count).map { Double($0) / max(1e-3, Double(temperature)) })
        let actions = softmax(actionLogits.map(Double.init))
        let entropy = -probabilities.reduce(0) { $0 + $1 * log(max($1, 1e-12)) }
        let confidence = count < 2 ? 1 : min(1, max(0, 1 - entropy / log(Double(count))))
        var fields: [DecisionJSON.Field] = [
            .init("type", .string(question.type)),
            .init(
                "confidence",
                number(
                    question.type == "noul"
                        ? max(probabilities[1], 1 - probabilities[1]) : confidence)),
            .init("action", .object([.init("act_probability", number(actions[0]))])),
        ]
        switch question.type {
        case "choice":
            // max(by:) chooses the final tied value; upstream argmax chooses the first.
            let winner = probabilities.indices.reduce(0) {
                probabilities[$1] > probabilities[$0] ? $1 : $0
            }
            fields.append(.init("choice", .string(question.labels[winner])))
        case "score":
            fields.append(
                .init(
                    "score",
                    number(
                        probabilities.enumerated().reduce(0) {
                            $0 + Double($1.offset) * $1.element
                        })))
            fields.append(
                .init(
                    "legend",
                    .object(
                        question.legend.enumerated().map {
                            .init(String($0.offset), $0.element)
                        })))
        default:
            fields.append(.init("noul", number(probabilities[1])))
        }
        if question.type != "noul" {
            fields.append(
                .init(
                    "probabilities",
                    .object(
                        zip(question.labels, probabilities).map {
                            .init($0.0, number($0.1))
                        })))
        }
        return .object(fields)
    }

    static func softmax(_ values: [Double]) -> [Double] {
        let maximum = values.max()!
        let exponents = values.map { exp($0 - maximum) }
        let sum = exponents.reduce(0, +)
        return exponents.map { $0 / sum }
    }

    private static func number(_ value: Double) -> DecisionJSON {
        .number(String((value * 10000).rounded(.toNearestOrEven) / 10000))
    }
}
