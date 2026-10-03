// Copyright © 2026 Eigen Labs Inc.
import Foundation

public struct SystemOneRequest: Sendable {
    public let model: String
    public let state: DecisionJSON
    public let questions: [Question]
    public static let maximumQuestions = 64

    public struct Question: Sendable {
        public let id: String
        public let type: String
        public let instructions: String
        public let labels: [String]
        public let options: [String]
        public let legend: [DecisionJSON]
        public var typeIndex: Int { ["choice", "score", "noul"].firstIndex(of: type)! }
    }

    public init(data: Data) throws {
        let json = try DecisionJSON.parse(data)
        let unsupported = [
            "messages", "input", "prompt", "tools", "tool_choice", "max_tokens",
            "max_output_tokens", "max_completion_tokens", "temperature", "top_p", "top_k",
            "stop", "n", "response_format",
        ]
        guard unsupported.allSatisfy({ json[$0] == nil }) else {
            throw LayaError.invalidRequest("System One does not accept generation controls")
        }
        guard let model = json["model"]?.string, !model.isEmpty,
            let state = json["state"], Self.isStructuredText(state),
            let fields = json["questions"]?.fields,
            !fields.isEmpty, fields.count <= Self.maximumQuestions,
            json["stream"] == nil || json["stream"] == .bool(false)
        else {
            throw LayaError.invalidRequest(
                "Expected model, state and 1...64 typed questions; streaming is unsupported")
        }
        self.model = model
        self.state = state
        self.questions = try fields.map { field in
            let value = field.value
            guard let kind = value["type"]?.string, ["choice", "score", "noul"].contains(kind),
                let instructions = value["instructions"], Self.isStructuredText(instructions)
            else { throw LayaError.invalidRequest("Each question requires type and instructions") }
            let criteria = value["criteria"]
            var labels: [String] = []
            var options: [String] = []
            var legend: [DecisionJSON] = []
            switch kind {
            case "choice":
                guard let choices = criteria?.fields, !choices.isEmpty, choices.count <= 255,
                    choices.allSatisfy({ Self.isStructuredText($0.value) || $0.value == .null })
                else {
                    throw LayaError.invalidRequest("Choice criteria must contain 1...255 options")
                }
                labels = choices.map(\.key)
                options = choices.map { option in
                    option.value == .null || option.value == .string("")
                        ? option.key
                        : option.key + ": " + Self.text(option.value)
                }
            case "score":
                guard let levels = criteria?.elements, (2 ... 10).contains(levels.count),
                    levels.allSatisfy(Self.isStructuredText)
                else { throw LayaError.invalidRequest("Score criteria must contain 2...10 levels") }
                legend = levels
                labels = levels.indices.map(String.init)
                options = levels.enumerated().map { "level \($0.offset): " + Self.text($0.element) }
            default:
                guard criteria == nil || criteria == .null || criteria?.fields != nil else {
                    throw LayaError.invalidRequest("Noul criteria must be an object")
                }
                if let entries = criteria?.fields {
                    guard
                        entries.allSatisfy({
                            ["false", "true"].contains($0.key) && Self.isStructuredText($0.value)
                        })
                    else {
                        throw LayaError.invalidRequest(
                            "Noul criteria accepts true and false descriptions")
                    }
                }
                labels = ["false", "true"]
                options = labels.map { label in
                    let value = criteria?[label]
                    let fallback =
                        label == "false"
                        ? "no, the statement does not hold" : "yes, the statement holds"
                    return label + ": "
                        + ((value == nil || value == .null || value == .string(""))
                            ? fallback : Self.text(value!))
                }
            }
            return Question(
                id: field.key, type: kind,
                instructions: instructions.string ?? instructions.rendered(ascii: true),
                labels: labels, options: options, legend: legend)
        }
    }
    private static func isStructuredText(_ value: DecisionJSON) -> Bool {
        switch value {
        case .string, .object, .array: true
        default: false
        }
    }
    static func text(_ value: DecisionJSON) -> String { value.string ?? value.rendered() }
}
