// mlx-server — minimal OpenAI-compatible HTTP server for MLX models.
//
// Loads a model from a local directory and serves it over HTTP.
// For downloading models from HuggingFace, use:
//   python -m mlx_lm.convert --hf-path <model-id> --mlx-path <local-dir>
//
// Usage:
//   mlx-server --model /path/to/model [--port 8080] [--host 127.0.0.1]

import Foundation
import Hummingbird
import MLXLLM
import MLXHuggingFace
import MLXLMCommon
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

// MARK: - CLI args

struct ServerArgs: Sendable {
    var model: String = ""
    var port: Int = 8080
    var host: String = "127.0.0.1"
    var maxTokens: Int = 4096
    var prefixCache: Bool = true
    var maxKVCacheTokens: Int = 0
    /// Enable internal MTP head (Qwen3.5/3.6, DeepSeek-V4). Set before model load.
    var mtp: Bool = false
    /// Path to an external Gemma 4 assistant/drafter model directory.
    var draftModel: String = ""

    static func parse() throws -> ServerArgs {
        var args = ServerArgs()
        let argv = CommandLine.arguments
        var i = 1
        while i < argv.count {
            switch argv[i] {
            case "--model", "-m":
                i += 1
                guard i < argv.count else { throw CLIError("--model requires a value") }
                args.model = argv[i]
            case "--port", "-p":
                i += 1
                guard i < argv.count, let p = Int(argv[i]) else {
                    throw CLIError("--port requires an integer")
                }
                args.port = p
            case "--host":
                i += 1
                guard i < argv.count else { throw CLIError("--host requires a value") }
                args.host = argv[i]
            case "--max-tokens":
                i += 1
                guard i < argv.count, let t = Int(argv[i]) else {
                    throw CLIError("--max-tokens requires an integer")
                }
                args.maxTokens = t
            case "--no-prefix-cache":
                args.prefixCache = false
            case "--max-kv-tokens":
                i += 1
                guard i < argv.count, let t = Int(argv[i]) else {
                    throw CLIError("--max-kv-tokens requires an integer")
                }
                args.maxKVCacheTokens = t
            case "--mtp":
                args.mtp = true
            case "--draft-model":
                i += 1
                guard i < argv.count else { throw CLIError("--draft-model requires a value") }
                args.draftModel = argv[i]
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                // Accept bare positional arg as the model path.
                if !argv[i].hasPrefix("-"), args.model.isEmpty {
                    args.model = argv[i]
                }
            }
            i += 1
        }
        if args.model.isEmpty {
            printUsage()
            exit(1)
        }
        return args
    }

    static func printUsage() {
        print("""
        mlx-server — OpenAI-compatible inference server for MLX models

        USAGE:
          mlx-server --model <local-path> [options]

        OPTIONS:
          --model, -m <path>    Path to a local MLX model directory (required)
          --port, -p <int>      Port to listen on (default: 8080)
          --host <string>       Host to bind to (default: 127.0.0.1)
          --max-tokens <int>    Default max tokens per request (default: 4096)
          --no-prefix-cache     Disable KV prefix caching
          --max-kv-tokens <int> Max total KV-cache tokens across running requests (0=unlimited)
          --mtp                 Enable internal MTP speculative decoding (Qwen3.5/3.6, DeepSeek-V4)
          --draft-model <path>  Path to an external Gemma 4 assistant/drafter model directory
          --help, -h            Show this help

        DOWNLOAD MODELS:
          python -m mlx_lm.convert --hf-path mlx-community/Qwen3-4B-4bit \\
                                   --mlx-path ~/models/Qwen3-4B-4bit

        EXAMPLES:
          mlx-server --model ~/models/Qwen3-4B-4bit
          mlx-server --model ~/models/Llama-3.2-3B-Instruct-4bit --port 8080 --host 0.0.0.0
        """)
    }
}

struct CLIError: Error, CustomStringConvertible {
    let description: String
    init(_ msg: String) { description = msg }
}

// MARK: - Gemma 4 MTP server context

/// Holds the target model context and loaded Gemma 4 assistant drafter for MTP generation.
/// Non-nil only when `--draft-model` is supplied.
struct Gemma4ServerContext: @unchecked Sendable {
    let target: ModelContext
    let drafter: Gemma4AssistantDraftModel
}

// MARK: - Model loading

struct ServerSetup: @unchecked Sendable {
    let engine: BatchedEngine
    let modelName: String
    let gemma4Context: Gemma4ServerContext?
}

func loadEngine(args: ServerArgs) async throws -> ServerSetup {
    let tokenizerLoader = #huggingFaceTokenizerLoader()

    // Expand ~ and resolve to an absolute URL.
    let expanded = (args.model as NSString).expandingTildeInPath
    let url = URL(fileURLWithPath: expanded)

    guard FileManager.default.fileExists(atPath: url.path) else {
        throw CLIError("""
        Model directory not found: \(url.path)
        Download models with: hf download <model-id>
        """)
    }

    // Enable internal MTP heads before loading (Qwen3.5/3.6, DeepSeek-V4).
    let hasExternalDrafter = !args.draftModel.isEmpty
    if args.mtp && !hasExternalDrafter {
        _qwen35MTPEnabled = true
        _deepseekV4MTPEnabled = true
    }

    print("Loading model from: \(url.path)")
    let context = try await loadModel(from: url, using: tokenizerLoader)
    let modelName = url.deletingLastPathComponent().lastPathComponent + "/" + url.lastPathComponent
    print("Model loaded: \(modelName)")

    let cbConfig = ContinuousBatchingConfig(
        schedulerConfig: SchedulerConfig(maxKVCacheTokens: args.maxKVCacheTokens),
        prefixCacheConfig: args.prefixCache ? PrefixCacheConfig() : nil,
        mtpEnabled: args.mtp && !hasExternalDrafter
    )
    let engine = BatchedEngine(context: context, config: cbConfig)
    await engine.start()

    var gemma4Context: Gemma4ServerContext? = nil
    if hasExternalDrafter {
        let draftExpanded = (args.draftModel as NSString).expandingTildeInPath
        let draftURL = URL(fileURLWithPath: draftExpanded)
        guard FileManager.default.fileExists(atPath: draftURL.path) else {
            throw CLIError("Draft model directory not found: \(draftURL.path)")
        }
        print("Loading Gemma 4 drafter from: \(draftURL.path)")
        let drafter = try await Gemma4AssistantDraftModel.load(from: draftURL)
        print("Drafter loaded")
        gemma4Context = Gemma4ServerContext(target: context, drafter: drafter)
    }

    return ServerSetup(engine: engine, modelName: modelName, gemma4Context: gemma4Context)
}

// MARK: - Entry point

@main
struct MLXServer {
    static func main() async throws {
        let args = try ServerArgs.parse()
        let setup = try await loadEngine(args: args)

        let router = buildRouter(
            engine: setup.engine,
            gemma4Context: setup.gemma4Context,
            modelName: setup.modelName,
            defaultMaxTokens: args.maxTokens
        )

        let app = Application(
            router: router,
            configuration: .init(address: .hostname(args.host, port: args.port))
        )

        let modeLabel: String
        if setup.gemma4Context != nil {
            modeLabel = "Gemma4 MTP"
        } else if args.mtp {
            modeLabel = "internal MTP"
        } else {
            modeLabel = "standard"
        }
        print("Server listening at http://\(args.host):\(args.port) [\(modeLabel)]")
        print("  POST /v1/chat/completions")
        print("  POST /v1/completions")
        print("  GET  /v1/models")
        print("  GET  /health")

        try await app.runService()
    }
}
