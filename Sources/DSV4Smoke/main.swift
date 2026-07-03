// DSV4Smoke — minimal end-to-end smoke harness for DeepSeek-V4 checkpoints.
//
// Loads a local model directory (full or truncated checkpoint), applies the
// chat template, greedy-generates, and prints text + timing + memory. Used to
// validate the DeepseekV4 port against real weights on a dev box before any
// provider integration.
//
// usage: DSV4Smoke <model-directory> [--prompt "..."] [--max-tokens N] [--raw]
//        --raw skips the chat template (plain completion of the prompt)
//        --logits "id1,id2,..." forwards exact token ids and prints the
//        last-position top-10 logits (for cross-implementation parity checks)

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Tokenizers

@main
struct DSV4Smoke {
    static func main() async {
        var args = Array(CommandLine.arguments.dropFirst())
        guard !args.isEmpty else {
            print("usage: DSV4Smoke <model-directory> [--prompt \"...\"] [--max-tokens N] [--raw]")
            exit(64)
        }
        let directory = URL(fileURLWithPath: args.removeFirst())
        var prompt = "Give me three facts about the Apple M5 Max."
        var maxTokens = 64
        var raw = false
        var logitTokens: [Int]? = nil
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--prompt" where i + 1 < args.count:
                prompt = args[i + 1]
                i += 2
            case "--max-tokens" where i + 1 < args.count:
                maxTokens = Int(args[i + 1]) ?? maxTokens
                i += 2
            case "--raw":
                raw = true
                i += 1
            case "--logits" where i + 1 < args.count:
                logitTokens = args[i + 1].split(separator: ",").compactMap { Int($0) }
                i += 2
            default:
                print("unknown arg: \(args[i])")
                exit(64)
            }
        }

        do {
            let loadStart = CFAbsoluteTimeGetCurrent()
            let container = try await LLMModelFactory.shared.loadContainer(
                from: directory, using: #huggingFaceTokenizerLoader())
            let loadSecs = CFAbsoluteTimeGetCurrent() - loadStart
            print(String(format: "[load] %.1fs, active=%.1f GiB, peak=%.1f GiB",
                loadSecs,
                Double(Memory.activeMemory) / 1073741824,
                Double(Memory.peakMemory) / 1073741824))

            // Logit-parity mode: forward exact ids (no cache), print
            // per-position top-10 (token, logit) of the LAST position. The
            // Python reference (mlx-lm PR #1192) run on the same ids must
            // produce matching values within quantization noise.
            if let ids = logitTokens {
                try await container.perform { (context: ModelContext) async throws in
                    print("[model] class=\(type(of: context.model))")
                    let model = context.model
                    guard let lm = model as? DeepseekV4Model else {
                        print("not a DeepseekV4Model: \(type(of: model))")
                        exit(64)
                    }
                    let cache = lm.makeCache(parameters: GenerateParameters())
                    let logits = lm(MLXArray(ids.map(Int32.init), [1, ids.count]), cache: cache)
                    let last = logits[0, -1, 0...].asType(.float32)
                    eval(last)
                    let vals = last.asArray(Float.self)
                    let top = vals.enumerated().sorted { $0.element > $1.element }.prefix(10)
                    print("[logits] last position top-10:")
                    for (tok, v) in top {
                        print(String(format: "  %6d  %+.4f", tok, v))
                    }
                    let mean = vals.reduce(0, +) / Float(vals.count)
                    let l2 = vals.map { Double($0) * Double($0) }.reduce(0, +).squareRoot()
                    print(String(format: "[logits] mean=%+.5f l2=%.3f vocab=%d", mean, l2, vals.count))
                }
                return
            }

            let (promptCopy, maxTokensCopy, rawCopy) = (prompt, maxTokens, raw)
            try await container.perform { (context: ModelContext) async throws in
                print("[model] class=\(type(of: context.model))")

                let tokens: [Int]
                if rawCopy {
                    tokens = context.tokenizer.encode(text: promptCopy, addSpecialTokens: true)
                } else {
                    tokens = try context.tokenizer.applyChatTemplate(
                        messages: [["role": "user", "content": promptCopy]],
                        tools: nil, additionalContext: nil)
                }
                print("[prompt] \(tokens.count) tokens")

                let input = LMInput(tokens: MLXArray(tokens))
                let params = GenerateParameters(maxTokens: maxTokensCopy, temperature: 0)

                let genStart = CFAbsoluteTimeGetCurrent()
                var firstTokenAt: CFAbsoluteTime?
                var pieces: [String] = []
                var completionInfo: GenerateCompletionInfo?

                let stream = try MLXLMCommon.generate(
                    input: input, parameters: params, context: context)
                for await gen in stream {
                    switch gen {
                    case .chunk(let text):
                        if firstTokenAt == nil { firstTokenAt = CFAbsoluteTimeGetCurrent() }
                        pieces.append(text)
                        print(text, terminator: "")
                        fflush(stdout)
                    case .info(let info):
                        completionInfo = info
                    case .toolCall(let call):
                        print("\n[toolCall] \(call.function.name)")
                    }
                }
                let done = CFAbsoluteTimeGetCurrent()
                print("\n---")
                let ttft = (firstTokenAt ?? done) - genStart
                print(String(format: "[timing] ttft=%.2fs total=%.2fs", ttft, done - genStart))
                if let info = completionInfo {
                    print(String(format: "[usage] prompt=%d completion=%d decode=%.1f tok/s",
                        info.promptTokenCount, info.generationTokenCount, info.tokensPerSecond))
                }
                print(String(format: "[memory] active=%.1f GiB peak=%.1f GiB",
                    Double(Memory.activeMemory) / 1073741824,
                    Double(Memory.peakMemory) / 1073741824))

                let text = pieces.joined()
                guard !text.isEmpty else {
                    print("[FAIL] no text generated")
                    exit(70)
                }
                print("[OK] generated \(text.count) chars")
            }
        } catch {
            print("error: \(error)")
            exit(70)
        }
    }
}
