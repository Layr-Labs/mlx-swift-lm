import Foundation

/// Lowers Responses history at the request boundary. Arguments remain opaque
/// JSON strings; only the existing template translator decodes their contents.
struct OpenAIResponseInputMessage: Decodable {
    enum Kind {
        case message, functionCall, functionCallOutput, reasoning
    }

    let kind: Kind
    let message: OpenAIChatMessage

    private enum CodingKeys: String, CodingKey {
        case type, role, content, name, arguments, output, summary
        case callID = "call_id"
    }

    init(from decoder: Decoder) throws {
        let object = try decoder.container(keyedBy: CodingKeys.self)
        switch try object.decodeIfPresent(String.self, forKey: .type) {
        case nil, "message":
            kind = .message
            message = try OpenAIChatMessage(from: decoder)
        case "function_call":
            kind = .functionCall
            let callID = try Self.nonempty(.callID, in: object)
            let name = try Self.nonempty(.name, in: object)
            let arguments = try object.decode(String.self, forKey: .arguments)
            message = .init(role: .assistant, content: .text(""), toolCalls: [
                .init(id: callID, function: .init(name: name, arguments: arguments))
            ])
        case "function_call_output":
            kind = .functionCallOutput
            let callID = try Self.nonempty(.callID, in: object)
            message = .init(role: .tool,
                content: try object.decode(OpenAIMessageContent.self, forKey: .output),
                toolCallID: callID)
        case "reasoning":
            kind = .reasoning
            let summary = try object.decodeIfPresent([OpenAIResponseOutputContent].self, forKey: .summary) ?? []
            message = .init(role: .assistant, content: .text(""),
                reasoningContent: summary.map(\.text).joined())
        default:
            throw DecodingError.dataCorruptedError(forKey: .type, in: object,
                debugDescription: "Unsupported Responses input item type")
        }
    }

    private static func nonempty(_ key: CodingKeys,
        in object: KeyedDecodingContainer<CodingKeys>) throws -> String
    {
        let value = try object.decode(String.self, forKey: key)
        guard !value.isEmpty else {
            throw DecodingError.dataCorruptedError(forKey: key, in: object,
                debugDescription: "Responses function history requires a nonempty \(key.rawValue)")
        }
        return value
    }
}
