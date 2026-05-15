// OpenAI-compatible request and response types for mlx-server.

import Foundation

// MARK: - Shared

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
    let content: String
}

struct ChatCompletionRequest: Codable, Sendable {
    let model: String?
    let messages: [ChatMessage]
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
