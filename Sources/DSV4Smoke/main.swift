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

/// Determinism stress for the primitive ops DeepSeek-V4 leans on that other
/// fleet models don't. Runs each op N times on identical inputs and reports
/// any run-to-run drift (bitwise, via float32 sum + abs-diff vs first run).
func runOpStress(iterations: Int) {
    MLXRandom.seed(42)

    func drift(_ name: String, _ make: () -> MLXArray) {
        var reference: [Float]? = nil
        var maxDrift: Float = 0
        var nanRuns = 0
        for _ in 0 ..< iterations {
            let out = make().asType(.float32)
            eval(out)
            let vals = out.asArray(Float.self)
            if vals.contains(where: { $0.isNaN }) { nanRuns += 1; continue }
            if let ref = reference {
                var d: Float = 0
                for (a, b) in zip(vals, ref) { d = max(d, abs(a - b)) }
                maxDrift = max(maxDrift, d)
            } else {
                reference = vals
            }
        }
        let verdict = (nanRuns > 0 || maxDrift > 0) ? "DRIFT" : "ok"
        print(String(format: "[op-stress] %-28@ maxDrift=%.3e nanRuns=%d/%d  %@",
            name, maxDrift, nanRuns, iterations, verdict))
    }

    // Shapes mirror the real model (hidden 4096, moe 2048, experts 256, top-6).
    let x = MLXRandom.normal([7, 4096]).asType(.bfloat16)
    eval(x)

    // 1. Plain dense matmul (control; steel gemm / nax on M5).
    let denseW = MLXRandom.normal([4096, 1024]).asType(.bfloat16)
    eval(denseW)
    drift("dense matmul (control)") { matmul(x, denseW) }

    // 2. Affine-quantized matmul g64 (attention projections).
    let wq = MLXRandom.normal([1024, 4096])
    let (qw, qs, qb) = MLX.quantized(wq, groupSize: 64, bits: 4, mode: .affine)
    eval(qw, qs, qb ?? MLXArray(0))
    drift("qmm affine g64") {
        quantizedMM(x, qw, scales: qs, biases: qb, transpose: true,
            groupSize: 64, bits: 4, mode: .affine)
    }

    // 3. Batched (3D) affine quantized matmul (wo_a MultiLinear).
    let wq3 = MLXRandom.normal([8, 1024, 4096])
    let (qw3, qs3, qb3) = MLX.quantized(wq3, groupSize: 64, bits: 4, mode: .affine)
    let x4 = MLXRandom.normal([1, 8, 7, 4096]).asType(.bfloat16)
    eval(qw3, qs3, x4)
    drift("qmm batched 3D (wo_a)") {
        quantizedMM(x4, qw3, scales: qs3, biases: qb3, transpose: true,
            groupSize: 64, bits: 4, mode: .affine)
    }

    // 4. mxfp4 gather-quantized matmul (routed experts).
    let we = MLXRandom.normal([256, 2048, 4096])
    let (qwe, qse, qbe) = MLX.quantized(we, groupSize: 32, bits: 4, mode: .mxfp4)
    let inds = MLXArray([Int32(3), 17, 40, 99, 200, 255], [1, 1, 6])
    let xg = MLXRandom.normal([1, 1, 1, 4096]).asType(.bfloat16)
    eval(qwe, qse, xg)
    drift("gather-qmm mxfp4 g32") {
        gatherQuantizedMM(
            xg, qwe, scales: qse, biases: qbe, rhsIndices: inds, transpose: true,
            groupSize: 32, bits: 4, mode: .mxfp4)
    }

    // 5. fast.rope with infinite NOPE frequencies (DeepseekV4RoPE).
    let ropeFreqs = concatenated([
        MLXArray(Array(repeating: Float.infinity, count: 224)),
        MLXArray((0 ..< 32).map { Float(pow(10000.0, Double($0) / 32.0)) }),
    ])
    let ropeX = MLXRandom.normal([1, 64, 7, 512]).asType(.bfloat16)
    eval(ropeFreqs, ropeX)
    drift("fast.rope inf freqs (NOPE)") {
        MLXFast.RoPE(
            ropeX, dimensions: 512, traditional: true, base: nil, scale: 1.0,
            offset: 0, freqs: ropeFreqs)
    }

    // 6. SDPA with sinks + bool array mask (1 KV head).
    let q = MLXRandom.normal([1, 64, 7, 512]).asType(.bfloat16)
    let kv = MLXRandom.normal([1, 1, 7, 512]).asType(.bfloat16)
    let sinks = MLXRandom.normal([64]).asType(.bfloat16)
    let boolMask = MLXArray((0 ..< 49).map { Int32($0 % 7 <= $0 / 7 ? 1 : 0) }, [7, 7])
        .asType(.bool)
    eval(q, kv, sinks, boolMask)
    drift("sdpa sinks + bool mask") {
        MLXFast.scaledDotProductAttention(
            queries: q, keys: kv, values: kv, scale: 0.044,
            mask: .array(boolMask), sinks: sinks)
    }
}

@main
struct DSV4Smoke {
    static func main() async {
        var args = Array(CommandLine.arguments.dropFirst())
        guard !args.isEmpty else {
            print("usage: DSV4Smoke <model-directory> [--prompt \"...\"] [--max-tokens N] [--raw]")
            print("       DSV4Smoke --op-stress [iterations]")
            exit(64)
        }
        if args[0] == "--op-stress" {
            runOpStress(iterations: args.count > 1 ? Int(args[1]) ?? 12 : 12)
            return
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
                    let (hidden, _) = lm.model.forward(
                        MLXArray(ids.map(Int32.init), [1, ids.count]),
                        cache: cache, returnRawHidden: false)
                    let hLast = hidden[0, -1, 0...].asType(.float32)
                    eval(hLast)
                    print(String(format: "[hidden] last absMax=%.5e mean=%+.5e l2=%.4f",
                        MLX.abs(hLast).max().item(Float.self),
                        hLast.mean().item(Float.self),
                        MLX.sqrt(hLast.square().sum()).item(Float.self)))
                    let logits = lm.lmHead(hidden)
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
