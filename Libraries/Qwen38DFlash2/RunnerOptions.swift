import Foundation
import Qwen38FastCore

public func decodeQwen38TokenIDs(_ data: Data) throws -> [Int] {
    if let tokenIDs = try? JSONDecoder().decode([Int].self, from: data) {
        return tokenIDs
    }
    let text = String(decoding: data, as: UTF8.self)
    let fields = text.split { $0.isWhitespace || $0 == "," }
    guard !fields.isEmpty else {
        throw Qwen38RunnerOptionError.invalid("tokens file contains no token IDs")
    }
    return try fields.map { field in
        guard let tokenID = Int(field) else {
            throw Qwen38RunnerOptionError.invalid(
                "tokens file contains a non-integer field: \(field)")
        }
        return tokenID
    }
}

public func validateQwen38PromptTokenIDs(_ tokenIDs: [Int]) throws {
    let vocabularySize = Qwen38TargetContract.production.vocabularySize
    if let invalid = tokenIDs.first(where: { $0 < 0 || $0 >= vocabularySize }) {
        throw Qwen38RunnerOptionError.invalid(
            "prompt token ID \(invalid) is outside 0..<\(vocabularySize)")
    }
}

public enum Qwen38RunnerMode: String, Equatable, Sendable {
    case autoregressive = "ar"
    case dflash2
    case benchmark
}

public enum Qwen38RunnerOptionError: Error, Equatable, CustomStringConvertible {
    case invalid(String)

    public var description: String {
        switch self {
        case .invalid(let message): message
        }
    }
}

public struct Qwen38RunnerOptions: Equatable, Sendable {
    public let targetPath: String
    public let draftPath: String?
    public let mode: Qwen38RunnerMode
    public let prompt: String?
    public let promptFile: String?
    public let tokensFile: String?
    public let maxTokens: Int
    public let conditionerTokens: Int
    public let receiptPath: String?
    public let useChatTemplate: Bool
    public let dflashPhysicalWidth: Int?
    public let diagnosticCycles: Bool
    public let diagnosticPrefetch: Bool

    public static let usage = """
        qwen38-dflash2-runner \\
          --target <local-target-directory> \\
          --mode <dflash2|ar|benchmark> \\
          [--draft <local-dflash2-directory>] \\
          (--prompt <text> | --prompt-file <path> | --tokens-file <path>) \\
          [--max-tokens 1024] [--conditioner-tokens 1024] \
          [--receipt <path>] [--no-chat-template] \
          [--dflash-width <1...8>] \
          [--diagnostic-cycles | --diagnostic-prefetch]
        """

    public static func parse(_ arguments: [String]) throws -> Qwen38RunnerOptions {
        var values = [String: String]()
        var noChatTemplate = false
        var diagnosticCycles = false
        var diagnosticPrefetch = false
        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            if flag == "--no-chat-template" {
                noChatTemplate = true
                index += 1
                continue
            }
            if flag == "--diagnostic-cycles" {
                diagnosticCycles = true
                index += 1
                continue
            }
            if flag == "--diagnostic-prefetch" {
                diagnosticPrefetch = true
                index += 1
                continue
            }
            guard flag.hasPrefix("--"), index + 1 < arguments.count else {
                throw Qwen38RunnerOptionError.invalid("missing value for \(flag)")
            }
            guard values[flag] == nil else {
                throw Qwen38RunnerOptionError.invalid("duplicate option \(flag)")
            }
            values[flag] = arguments[index + 1]
            index += 2
        }

        let known = Set([
            "--target", "--draft", "--mode", "--prompt", "--prompt-file",
            "--tokens-file", "--max-tokens", "--receipt", "--dflash-width",
            "--conditioner-tokens",
        ])
        if let unknown = values.keys.first(where: { !known.contains($0) }) {
            throw Qwen38RunnerOptionError.invalid("unknown option \(unknown)")
        }
        guard let targetPath = values["--target"], !targetPath.isEmpty else {
            throw Qwen38RunnerOptionError.invalid("--target is required")
        }
        let modeRaw = values["--mode"] ?? Qwen38RunnerMode.dflash2.rawValue
        guard let mode = Qwen38RunnerMode(rawValue: modeRaw) else {
            throw Qwen38RunnerOptionError.invalid("invalid --mode \(modeRaw)")
        }
        if diagnosticCycles && diagnosticPrefetch {
            throw Qwen38RunnerOptionError.invalid(
                "choose only one diagnostic route")
        }
        if diagnosticCycles || diagnosticPrefetch {
            guard mode == .dflash2 else {
                throw Qwen38RunnerOptionError.invalid(
                    "cycle diagnostics require --mode dflash2")
            }
            guard values["--receipt"] != nil else {
                throw Qwen38RunnerOptionError.invalid(
                    "cycle diagnostics require --receipt")
            }
        }
        let draftPath = values["--draft"]
        if mode != .autoregressive, draftPath?.isEmpty != false {
            throw Qwen38RunnerOptionError.invalid(
                "--draft is required for \(mode.rawValue)")
        }
        let promptSources = [
            values["--prompt"], values["--prompt-file"], values["--tokens-file"],
        ].compactMap { $0 }
        guard promptSources.count == 1 else {
            throw Qwen38RunnerOptionError.invalid(
                "provide exactly one of --prompt, --prompt-file, or --tokens-file")
        }
        let maxTokens = Int(values["--max-tokens"] ?? "1024") ?? 0
        guard maxTokens > 0 else {
            throw Qwen38RunnerOptionError.invalid("--max-tokens must be positive")
        }
        let conditionerTokens: Int
        if let rawConditionerTokens = values["--conditioner-tokens"] {
            guard let parsed = Int(rawConditionerTokens), parsed > 0 else {
                throw Qwen38RunnerOptionError.invalid(
                    "--conditioner-tokens must be positive")
            }
            conditionerTokens = parsed
        } else {
            conditionerTokens = 0
        }
        let dflashPhysicalWidth: Int?
        if let rawWidth = values["--dflash-width"] {
            guard let width = Int(rawWidth), (1 ... 8).contains(width) else {
                throw Qwen38RunnerOptionError.invalid(
                    "--dflash-width must be in 1...8")
            }
            guard mode != .autoregressive else {
                throw Qwen38RunnerOptionError.invalid(
                    "--dflash-width is unavailable in ar mode")
            }
            dflashPhysicalWidth = width
        } else {
            dflashPhysicalWidth = nil
        }
        return Qwen38RunnerOptions(
            targetPath: targetPath,
            draftPath: draftPath,
            mode: mode,
            prompt: values["--prompt"],
            promptFile: values["--prompt-file"],
            tokensFile: values["--tokens-file"],
            maxTokens: maxTokens,
            conditionerTokens: conditionerTokens,
            receiptPath: values["--receipt"],
            useChatTemplate: !noChatTemplate,
            dflashPhysicalWidth: dflashPhysicalWidth,
            diagnosticCycles: diagnosticCycles,
            diagnosticPrefetch: diagnosticPrefetch)
    }
}
