// Copyright © 2026 Apple Inc.

import Darwin
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXSpeculative
import Tokenizers  // required for #huggingFaceTokenizerLoader() macro expansion

enum BenchMode: String {
    case mtp
    case dflash

    var envPrefix: String {
        switch self {
        case .mtp: "MTP_BENCH"
        case .dflash: "DFLASH_BENCH"
        }
    }
}

struct CLIError: Error, CustomStringConvertible {
    let description: String

    init(_ description: String) {
        self.description = description
    }
}

struct BenchArguments {
    var mode: BenchMode?
    var targetPath: String?
    var drafterPath: String?
    var promptsJSONPath: String?
    var promptTokens: String?
    var prompt: String?
    var maxTokens: Int?
    var warmupTokens: Int?
    var blockSizes: [Int]?
    var useChatTemplate = true
    var phaseTimings = false
    var verifySubphaseTimings = false
    var disableVerifyQMM = false

    static func parse() throws -> BenchArguments {
        var args = BenchArguments()
        let argv = CommandLine.arguments
        var i = 1

        while i < argv.count {
            let arg = argv[i]
            switch arg {
            case "mtp", "dflash":
                args.mode = BenchMode(rawValue: arg)
            case "--mode":
                args.mode = BenchMode(rawValue: try value(after: arg, argv: argv, index: &i))
            case "--target", "-t":
                args.targetPath = try value(after: arg, argv: argv, index: &i)
            case "--drafter", "-d":
                args.drafterPath = try value(after: arg, argv: argv, index: &i)
            case "--prompts-json":
                args.promptsJSONPath = try value(after: arg, argv: argv, index: &i)
            case "--prompt-tokens":
                args.promptTokens = try value(after: arg, argv: argv, index: &i)
            case "--prompt":
                args.prompt = try value(after: arg, argv: argv, index: &i)
            case "--max-tokens":
                let value = try value(after: arg, argv: argv, index: &i)
                guard let parsed = Int(value), parsed > 0 else {
                    throw CLIError("--max-tokens requires a positive integer")
                }
                args.maxTokens = parsed
            case "--warmup-tokens":
                let value = try value(after: arg, argv: argv, index: &i)
                guard let parsed = Int(value), parsed > 0 else {
                    throw CLIError("--warmup-tokens requires a positive integer")
                }
                args.warmupTokens = parsed
            case "--block-sizes":
                args.blockSizes = try parsePositiveIntList(
                    try value(after: arg, argv: argv, index: &i),
                    minimum: 2,
                    optionName: "--block-sizes")
            case "--no-chat-template":
                args.useChatTemplate = false
            case "--phase-timings":
                args.phaseTimings = true
            case "--verify-subphases":
                args.verifySubphaseTimings = true
            case "--no-verify-qmm":
                args.disableVerifyQMM = true
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                if !arg.hasPrefix("-"), args.mode == nil {
                    args.mode = BenchMode(rawValue: arg)
                } else {
                    throw CLIError("Unknown argument: \(arg)")
                }
            }
            i += 1
        }

        return args
    }

    static func value(after option: String, argv: [String], index: inout Int) throws -> String {
        index += 1
        guard index < argv.count else {
            throw CLIError("\(option) requires a value")
        }
        return argv[index]
    }

    static func printUsage() {
        print("""
        mlx-bench - local MLX speculative decoding throughput benchmark

        USAGE:
          mlx-bench mtp    --target <target-dir> --drafter <assistant-dir> [options]
          mlx-bench dflash --target <target-dir> --drafter <dflash-dir> [options]

        OPTIONS:
          --target, -t <path>       Target model directory
          --drafter, -d <path>      Assistant or DFlash drafter directory
          --prompts-json <path>     JSON file containing [[Int]] token IDs
          --prompt-tokens <ids>     Comma-separated token IDs for one prompt
          --prompt <text>           Prompt text to tokenize with target tokenizer
          --max-tokens <int>        Generated tokens per prompt
          --warmup-tokens <int>     Generated tokens for warmup
          --block-sizes <list>      Comma-separated speculative block sizes
                                  (DFlash default sweeps 4,5,6,8 plus checkpoint size)
          --no-chat-template        Encode --prompt as plain text
          --phase-timings           Print DFlash diagnostic phase timings
          --verify-subphases        Split DFlash target verify into diagnostic subphases
          --no-verify-qmm           Disable DFlash target M=16 quantized-matmul fast path
          --help, -h                Show this help

        ENVIRONMENT:
          TARGET_DIR, DRAFTER_DIR
          MTP_BENCH_TARGET_DIR, MTP_BENCH_DRAFTER_DIR
          DFLASH_BENCH_TARGET_DIR, DFLASH_BENCH_DRAFTER_DIR
          PROMPTS_JSON, PROMPT_TOKENS, PROMPT
          MAX_TOKENS, WARMUP_TOKENS, BLOCK_SIZES
          DFLASH_BENCH_PHASES=1
          DFLASH_BENCH_VERIFY_SUBPHASES=1
          DFLASH_BENCH_VERIFY_QMM=0
        """)
    }
}

@main
struct MLXBench {
    static func main() async {
        do {
            let args = try BenchArguments.parse()
            guard let mode = args.mode else {
                BenchArguments.printUsage()
                exit(1)
            }

            switch mode {
            case .mtp:
                try await runMTPBenchmark(args: args, mode: mode)
            case .dflash:
                try await runDFlashBenchmark(args: args, mode: mode)
            }
        } catch {
            eprint("error: \(error)")
            exit(1)
        }
    }

    private static func runMTPBenchmark(args: BenchArguments, mode: BenchMode) async throws {
        let targetURL = try requiredURL(
            explicit: args.targetPath,
            envNames: ["TARGET_DIR", "\(mode.envPrefix)_TARGET_DIR"],
            description: "target model directory")
        let drafterURL = try requiredURL(
            explicit: args.drafterPath,
            envNames: ["DRAFTER_DIR", "\(mode.envPrefix)_DRAFTER_DIR"],
            description: "MTP assistant drafter directory")

        print("loading target: \(targetURL.path)")
        let context = try await LLMModelFactory.shared.load(
            from: targetURL,
            using: #huggingFaceTokenizerLoader())
        let target = try gemma4TextModel(from: context.model)

        print("loading MTP drafter: \(drafterURL.path)")
        let drafter = try await Gemma4AssistantDraftModel.load(from: drafterURL)
        eval(context.model, drafter)

        let prompts = try benchmarkPrompts(args: args, mode: mode, tokenizer: context.tokenizer)
        let maxTokens = args.maxTokens
            ?? envInt(names: ["MAX_TOKENS", "\(mode.envPrefix)_MAX_TOKENS"], default: 128)
        let warmupTokens = min(
            args.warmupTokens
                ?? envInt(
                    names: ["WARMUP_TOKENS", "\(mode.envPrefix)_WARMUP_TOKENS"],
                    default: min(16, maxTokens)),
            maxTokens)
        let blockSizes = args.blockSizes
            ?? envIntList(
                names: ["BLOCK_SIZES", "\(mode.envPrefix)_BLOCK_SIZES"],
                default: [2, 3, 4, 5],
                minimum: 2)

        print("")
        print("=== Gemma 4 MTP benchmark ===")
        print("target=\(targetURL.lastPathComponent)")
        print("drafter=\(drafterURL.lastPathComponent)")
        print("prompts=\(prompts.count), max_tokens=\(maxTokens), warmup=\(warmupTokens)")
        print("K  base tok/s  mtp tok/s  speedup  accept_avg")

        let warmupPrompt = MLXArray(prompts[0])
        _ = measureBaselineThroughput(
            target: target,
            promptTokens: warmupPrompt,
            maxTokens: warmupTokens)
        _ = try measureMTPThroughput(
            target: target,
            drafter: drafter,
            promptTokens: warmupPrompt,
            maxTokens: warmupTokens,
            blockSize: blockSizes.max() ?? 2)
        MLX.Memory.clearCache()

        var baselineRates = [Double]()
        for prompt in prompts {
            let result = measureBaselineThroughput(
                target: target,
                promptTokens: MLXArray(prompt),
                maxTokens: maxTokens)
            baselineRates.append(result.tokensPerSecond)
            MLX.Memory.clearCache()
        }
        let baselineAverage = average(baselineRates)

        for blockSize in blockSizes {
            var mtpRates = [Double]()
            var accepts = [Int]()
            for prompt in prompts {
                let result = try measureMTPThroughput(
                    target: target,
                    drafter: drafter,
                    promptTokens: MLXArray(prompt),
                    maxTokens: maxTokens,
                    blockSize: blockSize)
                mtpRates.append(result.tokensPerSecond)
                accepts.append(contentsOf: result.acceptLengths ?? [])
                MLX.Memory.clearCache()
            }

            let mtpAverage = average(mtpRates)
            let speedup = mtpAverage / max(baselineAverage, 1e-9)
            let acceptAverage = average(accepts.map(Double.init))
            print(
                "\(blockSize)  "
                    + "\(String(format: "%10.1f", baselineAverage)) "
                    + "\(String(format: "%9.1f", mtpAverage))   "
                    + "\(String(format: "%.2fx", speedup))   "
                    + "\(String(format: "%.2f", acceptAverage))/\(blockSize - 1)"
            )
        }
    }

    private static func runDFlashBenchmark(args: BenchArguments, mode: BenchMode) async throws {
        let targetURL = try requiredURL(
            explicit: args.targetPath,
            envNames: ["TARGET_DIR", "\(mode.envPrefix)_TARGET_DIR", "MLX_SWIFT_LM_DFLASH_TARGET_DIR"],
            description: "target model directory")
        let drafterURL = try requiredURL(
            explicit: args.drafterPath,
            envNames: ["DRAFTER_DIR", "\(mode.envPrefix)_DRAFTER_DIR", "MLX_SWIFT_LM_DFLASH_DRAFTER_DIR"],
            description: "DFlash drafter directory")

        print("loading target: \(targetURL.path)")
        let context = try await LLMModelFactory.shared.load(
            from: targetURL,
            using: #huggingFaceTokenizerLoader())
        guard let target = context.model as? any DFlashTargetModel else {
            throw CLIError("Target model does not conform to DFlashTargetModel: \(type(of: context.model))")
        }
        let enableVerifyQMM =
            !args.disableVerifyQMM
            && envBool(names: ["DFLASH_BENCH_VERIFY_QMM", "\(mode.envPrefix)_VERIFY_QMM"], default: true)
        let verifyQMMCount =
            enableVerifyQMM
            ? DFlashVerifyLinear.install(on: context.model, enableQMM: true)
            : 0

        print("loading DFlash drafter: \(drafterURL.path)")
        let drafter = try await DFlashDraftModel.load(
            from: drafterURL,
            bindTo: target)
        eval(context.model, drafter)

        let prompts = try benchmarkPrompts(args: args, mode: mode, tokenizer: context.tokenizer)
        let maxTokens = args.maxTokens
            ?? envInt(names: ["MAX_TOKENS", "\(mode.envPrefix)_MAX_TOKENS"], default: 128)
        let warmupTokens = min(
            args.warmupTokens
                ?? envInt(
                    names: ["WARMUP_TOKENS", "\(mode.envPrefix)_WARMUP_TOKENS"],
                    default: min(16, maxTokens)),
            maxTokens)
        let blockSizes = args.blockSizes
            ?? envIntList(
                names: ["BLOCK_SIZES", "\(mode.envPrefix)_BLOCK_SIZES"],
                default: defaultDFlashBlockSizes(
                    configured: drafter.config.blockSize,
                    recommended: drafter.config.recommendedBlockSize),
                minimum: 2)
        let collectPhaseTimings = args.phaseTimings
            || envBool(names: ["DFLASH_BENCH_PHASES", "\(mode.envPrefix)_PHASES"], default: false)
        let collectVerifySubphaseTimings = args.verifySubphaseTimings
            || envBool(
                names: ["DFLASH_BENCH_VERIFY_SUBPHASES", "\(mode.envPrefix)_VERIFY_SUBPHASES"],
                default: false)

        print("")
        print("=== DFlash benchmark ===")
        print("target=\(targetURL.lastPathComponent)")
        print("drafter=\(drafterURL.lastPathComponent)")
        print("prompts=\(prompts.count), max_tokens=\(maxTokens), warmup=\(warmupTokens)")
        print("verify_qmm=\(enableVerifyQMM ? "enabled" : "disabled") linears=\(verifyQMMCount)")
        if collectPhaseTimings || collectVerifySubphaseTimings {
            print("phase_timing=enabled (diagnostic wall-clock phases)")
        }
        if collectVerifySubphaseTimings {
            print("verify_subphases=enabled (adds target eval barriers)")
        }
        print("K  base tok/s  dflash tok/s  speedup  accept_avg  emit_avg  generated")

        let warmupPrompt = MLXArray(prompts[0])
        _ = measureDFlashBaselineThroughput(
            target: target,
            promptTokens: warmupPrompt,
            maxTokens: warmupTokens)
        _ = try measureDFlashThroughput(
            target: target,
            drafter: drafter,
            promptTokens: warmupPrompt,
            maxTokens: warmupTokens,
            blockSize: blockSizes.max())
        MLX.Memory.clearCache()

        var baselineRates = [Double]()
        for prompt in prompts {
            let result = measureDFlashBaselineThroughput(
                target: target,
                promptTokens: MLXArray(prompt),
                maxTokens: maxTokens)
            baselineRates.append(result.tokensPerSecond)
            MLX.Memory.clearCache()
        }
        let baselineAverage = average(baselineRates)

        for blockSize in blockSizes {
            var dflashRates = [Double]()
            var dflashSeconds = 0.0
            var generated = [String]()
            var accepts = [Int]()
            var phaseTotals = DFlashPhaseTotals()
            for prompt in prompts {
                let result = try measureDFlashThroughput(
                    target: target,
                    drafter: drafter,
                    promptTokens: MLXArray(prompt),
                    maxTokens: maxTokens,
                    blockSize: blockSize,
                    collectPhaseTimings: collectPhaseTimings,
                    collectVerifySubphaseTimings: collectVerifySubphaseTimings)
                dflashRates.append(result.tokensPerSecond)
                dflashSeconds += result.generationSeconds
                generated.append(String(result.generatedTokens))
                accepts.append(contentsOf: result.acceptLengths ?? [])
                if let phases = result.phaseTimings {
                    phaseTotals.add(phases)
                }
                MLX.Memory.clearCache()
            }

            let dflashAverage = average(dflashRates)
            let speedup = dflashAverage / max(baselineAverage, 1e-9)
            let acceptAverage = average(accepts.map(Double.init))
            let emitAverage = acceptAverage + 1
            print(
                "\(blockSize)  "
                    + "\(String(format: "%10.1f", baselineAverage)) "
                    + "\(String(format: "%12.1f", dflashAverage))   "
                    + "\(String(format: "%.2fx", speedup))   "
                    + "\(String(format: "%.2f", acceptAverage))/\(blockSize - 1)   "
                    + "\(String(format: "%.2f", emitAverage))/\(blockSize)   "
                    + generated.joined(separator: ",")
            )
            if collectPhaseTimings || collectVerifySubphaseTimings, phaseTotals.rounds > 0 {
                print("   " + phaseTotals.summary(generationSeconds: dflashSeconds))
            }
        }
    }

    private static func defaultDFlashBlockSizes(configured: Int, recommended: Int) -> [Int] {
        var seen = Set<Int>()
        var values = [Int]()
        let candidates =
            recommended == configured
            ? [4, 5, 6, 8, configured]
            : [recommended, 4, 5, 6, 8, configured]
        for candidate in candidates {
            guard candidate >= 2, candidate <= configured, !seen.contains(candidate) else {
                continue
            }
            seen.insert(candidate)
            values.append(candidate)
        }
        if values.isEmpty, configured >= 2 {
            values.append(configured)
        }
        return values
    }
}

private func gemma4TextModel(from model: any LanguageModel) throws -> Gemma4TextModel {
    if let textModel = model as? Gemma4TextModel {
        return textModel
    }
    if let wrapper = model as? Gemma4Model {
        return wrapper.textModel
    }
    throw CLIError("MTP benchmark requires Gemma4Model or Gemma4TextModel; got \(type(of: model))")
}

private func benchmarkPrompts(
    args: BenchArguments,
    mode: BenchMode,
    tokenizer: any MLXLMCommon.Tokenizer
) throws -> [[Int32]] {
    if let path = args.promptsJSONPath
        ?? envString(names: ["PROMPTS_JSON", "\(mode.envPrefix)_PROMPTS", "\(mode.envPrefix)_PROMPTS_JSON"])
    {
        let url = URL(fileURLWithPath: expandPath(path))
        let data = try Data(contentsOf: url)
        let decoded = try JSONDecoder().decode([[Int]].self, from: data)
        return try decoded.map { try int32Tokens($0) }.filter { !$0.isEmpty }
    }

    if let tokenList = args.promptTokens
        ?? envString(names: ["PROMPT_TOKENS", "\(mode.envPrefix)_PROMPT_TOKENS"])
    {
        return [try parseTokenList(tokenList)]
    }

    let prompt = args.prompt
        ?? envString(names: ["PROMPT", "\(mode.envPrefix)_PROMPT"])
        ?? "Write a concise explanation of speculative decoding and why accept rate matters."

    if args.useChatTemplate && envString(names: ["BENCH_CHAT_TEMPLATE"]) != "0" {
        do {
            let messages: [[String: any Sendable]] = [
                ["role": "user", "content": prompt]
            ]
            let tokenIDs = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: nil as [[String: any Sendable]]?,
                additionalContext: nil as [String: any Sendable]?)
            return [try int32Tokens(tokenIDs)]
        } catch {
            eprint("warning: chat template failed, falling back to plain encode: \(error)")
        }
    }

    return [try int32Tokens(tokenizer.encode(text: prompt, addSpecialTokens: true))]
}

private func requiredURL(
    explicit: String?,
    envNames: [String],
    description: String
) throws -> URL {
    guard let path = explicit ?? envString(names: envNames) else {
        throw CLIError("Missing \(description). Pass an argument or set one of: \(envNames.joined(separator: ", "))")
    }
    let url = URL(fileURLWithPath: expandPath(path))
    guard FileManager.default.fileExists(atPath: url.path) else {
        throw CLIError("\(description) not found: \(url.path)")
    }
    return url
}

private func expandPath(_ path: String) -> String {
    (path as NSString).expandingTildeInPath
}

private func envString(names: [String]) -> String? {
    let env = ProcessInfo.processInfo.environment
    for name in names {
        if let value = env[name], !value.isEmpty {
            return value
        }
    }
    return nil
}

private func envInt(names: [String], default defaultValue: Int) -> Int {
    guard let value = envString(names: names), let parsed = Int(value), parsed > 0 else {
        return defaultValue
    }
    return parsed
}

private func envIntList(names: [String], default defaultValue: [Int], minimum: Int) -> [Int] {
    guard let value = envString(names: names),
        let parsed = try? parsePositiveIntList(value, minimum: minimum, optionName: names[0]),
        !parsed.isEmpty
    else {
        return defaultValue
    }
    return parsed
}

private func parsePositiveIntList(_ value: String, minimum: Int, optionName: String) throws -> [Int] {
    let parsed = value
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .compactMap(Int.init)
        .filter { $0 >= minimum }
    guard !parsed.isEmpty else {
        throw CLIError("\(optionName) requires comma-separated integers >= \(minimum)")
    }
    return parsed
}

private func envBool(names: [String], default defaultValue: Bool) -> Bool {
    guard let value = envString(names: names)?.lowercased() else {
        return defaultValue
    }
    switch value {
    case "1", "true", "yes", "on":
        return true
    case "0", "false", "no", "off":
        return false
    default:
        return defaultValue
    }
}

private func parseTokenList(_ value: String) throws -> [Int32] {
    let parsed = value
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .compactMap(Int.init)
    return try int32Tokens(parsed)
}

private func int32Tokens(_ values: [Int]) throws -> [Int32] {
    var output = [Int32]()
    output.reserveCapacity(values.count)
    for value in values {
        guard value >= Int(Int32.min), value <= Int(Int32.max) else {
            throw CLIError("Token ID is outside Int32 range: \(value)")
        }
        output.append(Int32(value))
    }
    guard !output.isEmpty else {
        throw CLIError("Prompt tokens must not be empty")
    }
    return output
}

private func average(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    return values.reduce(0, +) / Double(values.count)
}

private struct DFlashPhaseTotals {
    var rounds = 0
    var cacheSnapshotSeconds = 0.0
    var draftLaunchSeconds = 0.0
    var draftCacheTrimSeconds = 0.0
    var verifyAndWaitSeconds = 0.0
    var targetTrunkSeconds = 0.0
    var targetHiddenConcatSeconds = 0.0
    var targetLMHeadSeconds = 0.0
    var targetSoftcapArgmaxSeconds = 0.0
    var acceptWalkSeconds = 0.0
    var cacheRollbackSeconds = 0.0
    var roundSeconds = 0.0

    mutating func add(_ phases: DFlashBenchmarkPhaseTimings) {
        rounds += phases.rounds
        cacheSnapshotSeconds += phases.cacheSnapshotSeconds
        draftLaunchSeconds += phases.draftLaunchSeconds
        draftCacheTrimSeconds += phases.draftCacheTrimSeconds
        verifyAndWaitSeconds += phases.verifyAndWaitSeconds
        targetTrunkSeconds += phases.targetTrunkSeconds
        targetHiddenConcatSeconds += phases.targetHiddenConcatSeconds
        targetLMHeadSeconds += phases.targetLMHeadSeconds
        targetSoftcapArgmaxSeconds += phases.targetSoftcapArgmaxSeconds
        acceptWalkSeconds += phases.acceptWalkSeconds
        cacheRollbackSeconds += phases.cacheRollbackSeconds
        roundSeconds += phases.roundSeconds
    }

    func summary(generationSeconds: Double) -> String {
        "phase ms/round: "
            + "snapshot=\(msPerRound(cacheSnapshotSeconds)) "
            + "draft_launch=\(msPerRound(draftLaunchSeconds)) "
            + "draft_trim=\(msPerRound(draftCacheTrimSeconds)) "
            + "verify_wait=\(msPerRound(verifyAndWaitSeconds)) "
            + "accept=\(msPerRound(acceptWalkSeconds)) "
            + "rollback=\(msPerRound(cacheRollbackSeconds)) "
            + "round=\(msPerRound(roundSeconds)); "
            + "%gen verify_wait=\(percent(verifyAndWaitSeconds, of: generationSeconds)) "
            + "accept=\(percent(acceptWalkSeconds, of: generationSeconds)) "
            + "rollback=\(percent(cacheRollbackSeconds, of: generationSeconds))"
            + verifySubphaseSummary()
    }

    private func verifySubphaseSummary() -> String {
        let total = targetTrunkSeconds + targetHiddenConcatSeconds + targetLMHeadSeconds
            + targetSoftcapArgmaxSeconds
        guard total > 0 else { return "" }
        return "; target_verify ms/round: "
            + "trunk=\(msPerRound(targetTrunkSeconds)) "
            + "hidden_concat=\(msPerRound(targetHiddenConcatSeconds)) "
            + "lm_head=\(msPerRound(targetLMHeadSeconds)) "
            + "softcap_argmax=\(msPerRound(targetSoftcapArgmaxSeconds))"
    }

    private func msPerRound(_ seconds: Double) -> String {
        String(format: "%.2f", seconds * 1000 / Double(max(rounds, 1)))
    }

    private func percent(_ seconds: Double, of totalSeconds: Double) -> String {
        guard totalSeconds > 0 else { return "0.0%" }
        return String(format: "%.1f%%", seconds * 100 / totalSeconds)
    }
}

private func eprint(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}
