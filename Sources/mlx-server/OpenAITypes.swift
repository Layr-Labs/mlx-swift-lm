// OpenAI-compatible request and response types for mlx-server.

import Foundation

// MARK: - Shared

enum JSONValue: Codable, Sendable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
        } else if let value = try? c.decode(String.self) {
            self = .string(value)
        } else if let value = try? c.decode(Int.self) {
            self = .int(value)
        } else if let value = try? c.decode(Double.self) {
            self = .double(value)
        } else if let value = try? c.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? c.decode([String: JSONValue].self) {
            self = .object(value)
        } else {
            self = .array(try c.decode([JSONValue].self))
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let value):
            try c.encode(value)
        case .int(let value):
            try c.encode(value)
        case .double(let value):
            try c.encode(value)
        case .bool(let value):
            try c.encode(value)
        case .object(let value):
            try c.encode(value)
        case .array(let value):
            try c.encode(value)
        case .null:
            try c.encodeNil()
        }
    }

    var sendableValue: any Sendable {
        switch self {
        case .string(let value):
            value
        case .int(let value):
            value
        case .double(let value):
            value
        case .bool(let value):
            value
        case .object(let value):
            value.mapValues { $0.sendableValue }
        case .array(let value):
            value.map { $0.sendableValue }
        case .null:
            NSNull()
        }
    }
}

/// `stop` can be a string or array of strings in the OpenAI spec.
struct StopParam: Codable, Sendable {
    let strings: [String]

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) {
            strings = [s]
        } else {
            strings = try c.decode([String].self)
        }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        try c.encode(strings)
    }
}

struct Usage: Codable, Sendable {
    let prompt_tokens: Int
    let completion_tokens: Int
    let total_tokens: Int
}

// MARK: - Chat Completions

struct ChatMessage: Codable, Sendable {
    let role: String
    let content: JSONValue?
    let name: String?
    let tool_call_id: String?
    let tool_calls: [JSONValue]?

    init(
        role: String,
        content: String?,
        name: String? = nil,
        tool_call_id: String? = nil,
        tool_calls: [JSONValue]? = nil
    ) {
        self.role = role
        self.content = content.map(JSONValue.string)
        self.name = name
        self.tool_call_id = tool_call_id
        self.tool_calls = tool_calls
    }

    init(
        role: String,
        content: JSONValue?,
        name: String? = nil,
        tool_call_id: String? = nil,
        tool_calls: [JSONValue]? = nil
    ) {
        self.role = role
        self.content = content
        self.name = name
        self.tool_call_id = tool_call_id
        self.tool_calls = tool_calls
    }

    var promptMessage: [String: any Sendable] {
        var result: [String: any Sendable] = ["role": role]
        if let content {
            result["content"] = content.sendableValue
        } else {
            result["content"] = ""
        }
        if let name {
            result["name"] = name
        }
        if let tool_call_id {
            result["tool_call_id"] = tool_call_id
        }
        if let tool_calls {
            result["tool_calls"] = tool_calls.map { $0.sendableValue }
        }
        return result
    }
}

struct ChatCompletionRequest: Codable, Sendable {
    let model: String?
    let messages: [ChatMessage]
    let tools: [JSONValue]?
    let max_tokens: Int?
    let temperature: Float?
    let top_p: Float?
    let top_k: Int?
    let min_p: Float?
    let repetition_penalty: Float?
    let presence_penalty: Float?
    let frequency_penalty: Float?
    let stream: Bool?
    let stop: StopParam?
    let seed: Int?

    var promptTools: [[String: any Sendable]]? {
        guard let tools else { return nil }
        return tools.compactMap { $0.sendableValue as? [String: any Sendable] }
    }
}

struct ChatCompletionResponse: Codable, Sendable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [ChatChoice]
    let usage: Usage
}

struct ChatChoice: Codable, Sendable {
    let index: Int
    let message: ChatMessage
    let finish_reason: String?
}

// Streaming chunk types

struct ChatCompletionChunk: Codable, Sendable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [ChunkChoice]
}

struct ChunkChoice: Codable, Sendable {
    let index: Int
    let delta: ChunkDelta
    let finish_reason: String?
}

struct ChunkDelta: Codable, Sendable {
    let role: String?
    let content: String?
    let tool_calls: [JSONValue]?

    init(role: String? = nil, content: String? = nil, tool_calls: [JSONValue]? = nil) {
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
    }
}

// MARK: - Text Completions

struct CompletionRequest: Codable, Sendable {
    let model: String?
    let prompt: String
    let max_tokens: Int?
    let temperature: Float?
    let top_p: Float?
    let stream: Bool?
    let stop: StopParam?
    let seed: Int?
}

struct CompletionResponse: Codable, Sendable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [CompletionChoice]
    let usage: Usage
}

struct CompletionChoice: Codable, Sendable {
    let index: Int
    let text: String
    let finish_reason: String?
}

struct CompletionChunk: Codable, Sendable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [CompletionChunkChoice]
}

struct CompletionChunkChoice: Codable, Sendable {
    let index: Int
    let text: String
    let finish_reason: String?
}

// MARK: - Models

struct ModelsResponse: Codable, Sendable {
    let object: String
    let data: [ModelData]
}

struct ModelData: Codable, Sendable {
    let id: String
    let object: String
    let created: Int
    let owned_by: String
}

// MARK: - Health

struct HealthResponse: Codable, Sendable {
    let status: String
    let model: String
}
