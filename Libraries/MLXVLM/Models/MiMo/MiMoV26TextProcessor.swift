// Copyright © 2026 Eigen Labs.
import Foundation
import MLX
import MLXLMCommon

/// Native template-input boundary only. HTTP/OpenAI normalization, tool-history
/// repair and effort aliases belong upstream. This processor never silently
/// drops media or rewrites native thinking defaults.
public struct MiMoV26TextProcessor: UserInputProcessor {
    public let chatTemplate: String
    public let vocabularySize: Int
    public let maximumSequenceLength: Int
    private let tokenizer: any Tokenizer
    private static let mediaKeys: Set<String> = [
        "image", "images", "image_url", "video", "videos", "video_url",
        "audio", "audio_url", "input_audio", "audio_chunks",
    ]

    public init(
        tokenizer: any Tokenizer, chatTemplate: String,
        vocabularySize: Int, maximumSequenceLength: Int
    ) throws {
        guard !chatTemplate.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            vocabularySize > 0, vocabularySize <= Int(Int32.max),
            maximumSequenceLength > 1, maximumSequenceLength <= Int(Int32.max)
        else {
            throw MiMoV26FactoryError.invalidMetadata(
                "empty template or invalid token/context bounds")
        }
        self.tokenizer = tokenizer
        self.chatTemplate = chatTemplate
        self.vocabularySize = vocabularySize
        self.maximumSequenceLength = maximumSequenceLength
    }

    /// Returns host token IDs, allowing template/tokenizer preflight before any
    /// native model/input allocation. Tools and reasoning fields pass unchanged.
    public func renderTokens(input: UserInput) throws -> [Int] {
        guard input.images.isEmpty, input.videos.isEmpty else {
            throw MiMoV26FactoryError.unsupportedInput(
                "native media processing is not qualified by the text entrypoint")
        }
        guard input.additionalContext?.keys.contains(where: Self.mediaKeys.contains) != true else {
            throw MiMoV26FactoryError.unsupportedInput("render context contains unsupported media")
        }
        if let thinking = input.additionalContext?["enable_thinking"],
            type(of: thinking) != Bool.self
        {
            throw MiMoV26FactoryError.unsupportedInput(
                "enable_thinking must be a native Boolean; API aliases must be resolved upstream")
        }
        if let generation = input.additionalContext?["add_generation_prompt"],
            type(of: generation) != Bool.self
        {
            throw MiMoV26FactoryError.unsupportedInput(
                "add_generation_prompt must be a native Boolean")
        }
        let messages: [Message]
        switch input.prompt {
        case .text(let text): messages = [["role": "user", "content": text]]
        case .messages(let raw): messages = raw
        case .chat(let chat):
            messages = try chat.map { message in
                guard message.images.isEmpty, message.videos.isEmpty,
                    message.templateFields["role"] == nil, message.templateFields["content"] == nil
                else {
                    throw MiMoV26FactoryError.unsupportedInput(
                        "chat contains media or overrides structured role/content")
                }
                // Preserve native per-message tools and future template fields;
                // generic whitelist projection would silently remove them.
                var raw = message.templateFields
                raw["role"] = message.role.rawValue
                raw["content"] = message.content
                return raw
            }
        }
        guard !messages.isEmpty else {
            throw MiMoV26FactoryError.unsupportedInput("empty message list")
        }
        let textMessages = try messages.map(Self.prepareTextMessage)
        let tokens = try tokenizer.applyChatTemplate(
            messages: textMessages, chatTemplate: chatTemplate,
            tools: input.tools, additionalContext: input.additionalContext)
        guard !tokens.isEmpty, tokens.count < maximumSequenceLength,
            tokens.allSatisfy({ $0 >= 0 && $0 < vocabularySize })
        else {
            throw MiMoV26FactoryError.unsupportedInput(
                "rendered tokens exceed native vocabulary/context")
        }
        return tokens
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        try Task.checkCancellation()
        let tokens = try renderTokens(input: input)
        try Task.checkCancellation()
        return LMInput(tokens: MLXArray(tokens.map(Int32.init)))
    }

    private static func prepareTextMessage(_ message: Message) throws -> Message {
        guard let role = message["role"] as? String,
            ["system", "user", "assistant", "tool"].contains(role)
        else {
            throw MiMoV26FactoryError.unsupportedInput(
                "native text message requires a supported role")
        }
        guard !mediaKeys.contains(where: { message[$0] != nil }) else {
            throw MiMoV26FactoryError.unsupportedInput("message has image/video/audio fields")
        }
        let content = message["content"]
        if content == nil || content is NSNull || (content as? MLXLMCommon.JSONValue) == .null {
            // The selected native render_content macro renders absent/null
            // bodies exactly like "". Canonicalize ONLY that body slot for the
            // Swift-Jinja bridge; tool calls, arguments and reasoning remain
            // untouched. Real pinned-template parity is a separate test gate.
            var emptyBody = message
            emptyBody["content"] = ""
            return emptyBody
        }
        if content is String { return message }
        // Preserve text content-part arrays without flattening/reordering them.
        // Unknown parts are refused because the native template may omit them.
        guard let parts = content as? [[String: any Sendable]] else {
            throw MiMoV26FactoryError.unsupportedInput(
                "content must be text or text-only native parts")
        }
        for part in parts {
            guard part["type"] as? String == "text", part["text"] is String,
                !mediaKeys.contains(where: { part[$0] != nil })
            else {
                throw MiMoV26FactoryError.unsupportedInput(
                    "image/video/audio or unknown content part")
            }
        }
        return message
    }
}
