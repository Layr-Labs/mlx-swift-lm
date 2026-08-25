import CryptoKit
import Darwin
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import Qwen38DFlash2
import Qwen38FastCore
import Tokenizers

private enum RunnerError: Error, CustomStringConvertible {
    case message(String)

    var description: String {
        switch self {
        case .message(let text): text
        }
    }
}

private final class GPUFileLock {
    private let descriptor: Int32

    init(path: String) throws {
        descriptor = open(path, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else {
            throw RunnerError.message("cannot open GPU lock at \(path)")
        }
        FileHandle.standardError.write(Data("waiting for GPU lock \(path)\n".utf8))
        guard flock(descriptor, LOCK_EX) == 0 else {
            close(descriptor)
            throw RunnerError.message("cannot acquire GPU lock at \(path)")
        }
    }

    deinit {
        flock(descriptor, LOCK_UN)
        close(descriptor)
    }
}

private struct RunReceipt: Codable, Sendable {
    let mode: String
    let promptTokens: Int
    let outputTokens: Int
    let prefillSeconds: Double
    let decodeSeconds: Double
    let decodeTokensPerSecond: Double
    let tokenSHA256: String
}

private struct RunnerReceipt: Codable, Sendable {
    let createdAt: String
    let targetPath: String
    let draftPath: String?
    let targetRepository: String
    let targetRevision: String
    let targetConfigSHA256: String
    let draftRepository: String
    let draftRevision: String
    let draftConfigSHA256: String?
    let swiftBaseRevision: String
    let mlxSwiftRevision: String
    let mlxRevision: String
    let mtplxSourceRevision: String
    let targetSourceRevision: String
    let dflash2SourceRevision: String
    let commandBufferMegabytes: Int
    let commandBufferOperations: Int
    let dflashPhysicalWidth: Int
    let targetOptimizedProjections: Int
    let targetPreservedFusedGDNInputs: Int
    let draftOptimizedProjections: Int
    let activeMemoryAtConstruction: Int
    let wiredMemoryLimit: Int
    let wiredMemoryApplied: Int
    let targetTemperature: Float
    let targetTopP: Float
    let targetTopK: Int
    let targetSeed: UInt64
    let runs: [RunReceipt]
}

private struct RunnerOutcome: Sendable {
    let text: String?
    let receipt: RunnerReceipt
}

private func tokenDigest(_ tokens: [Int]) -> String {
    let data = Data(tokens.map(String.init).joined(separator: ",").utf8)
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func installRuntimeContract() throws {
    let required = [
        "MLX_MAX_MB_PER_BUFFER": "512",
        "MLX_MAX_OPS_PER_BUFFER": "50",
    ]
    for (name, value) in required {
        if let existing = ProcessInfo.processInfo.environment[name], existing != value {
            throw RunnerError.message(
                "\(name)=\(existing) conflicts with required value \(value)")
        }
        if getenv(name) == nil, setenv(name, value, 1) != 0 {
            throw RunnerError.message("cannot install runtime setting \(name)=\(value)")
        }
    }
    _qwen35MTPEnabled = false
}

private func validateLocalArtifacts(_ options: Qwen38RunnerOptions) throws {
    let manager = FileManager.default
    var artifacts = [("target", options.targetPath)]
    if let draftPath = options.draftPath {
        artifacts.append(("draft", draftPath))
    }
    for (label, path) in artifacts {
        var isDirectory: ObjCBool = false
        guard manager.fileExists(atPath: path, isDirectory: &isDirectory), isDirectory.boolValue
        else {
            throw RunnerError.message("\(label) directory does not exist: \(path)")
        }
        guard
            manager.fileExists(
                atPath: URL(fileURLWithPath: path)
                    .appending(component: "config.json").path)
        else {
            throw RunnerError.message("\(label) config.json is missing: \(path)")
        }
    }
}

private func promptTokens(
    options: Qwen38RunnerOptions,
    tokenizer: MLXLMCommon.Tokenizer
) throws -> [Int] {
    if let path = options.tokensFile {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return try decodeQwen38TokenIDs(data)
    }
    let prompt: String
    if let inline = options.prompt {
        prompt = inline
    } else {
        prompt = try String(
            contentsOfFile: options.promptFile!, encoding: .utf8)
    }
    if options.useChatTemplate {
        return try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": prompt]],
            tools: nil,
            additionalContext: nil)
    }
    return tokenizer.encode(text: prompt, addSpecialTokens: true)
}

private func runAutoregressive(
    target: any DFlash2QwenTarget,
    prompt: MLXArray,
    outputBudget: Int
) -> (tokens: [Int], receipt: RunReceipt) {
    let cache = target.newCache(parameters: nil)
    let sampler = Qwen38TargetSampler()
    let prefillStart = CFAbsoluteTimeGetCurrent()
    var prefillLogits: MLXArray?
    for range in dflash2PrefillRanges(tokenCount: prompt.dim(1)) {
        prefillLogits = target.dflashTargetOnlyPrefillChunk(
            input: LMInput.Text(tokens: prompt[0..., range]),
            cache: cache)
        eval(prefillLogits!, cache.flatMap { $0.innerState() })
    }
    var logits = prefillLogits!
    var next = sampler.sample(logits: logits[0..., -1, 0...]).reshaped([-1])
    eval(next, cache.flatMap { $0.innerState() })
    let prefillSeconds = CFAbsoluteTimeGetCurrent() - prefillStart

    let decodeStart = CFAbsoluteTimeGetCurrent()
    var generated = [Int]()
    generated.reserveCapacity(outputBudget)
    while generated.count < outputBudget {
        generated.append(Int(next.item(Int32.self)))
        if generated.count == outputBudget { break }
        logits = target(next.reshaped([1, 1]), cache: cache)
        next = sampler.sample(logits: logits[0..., -1, 0...]).reshaped([-1])
        asyncEval(next, cache.flatMap { $0.innerState() })
    }
    let decodeSeconds = CFAbsoluteTimeGetCurrent() - decodeStart
    return (
        generated,
        RunReceipt(
            mode: Qwen38RunnerMode.autoregressive.rawValue,
            promptTokens: prompt.dim(1),
            outputTokens: generated.count,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds,
            decodeTokensPerSecond: Double(generated.count) / decodeSeconds,
            tokenSHA256: tokenDigest(generated))
    )
}

private func runDFlash2(
    target: any DFlash2QwenTarget,
    draft: DFlash2DraftModel,
    prompt: MLXArray,
    outputBudget: Int,
    fixedPhysicalWidth: Int?
) -> (tokens: [Int], receipt: RunReceipt) {
    let session = DFlash2Session(
        target: target, draft: draft, promptLength: prompt.dim(1))
    let prefillStart = CFAbsoluteTimeGetCurrent()
    session.prefill(promptTokens: prompt)
    let prefillSeconds = CFAbsoluteTimeGetCurrent() - prefillStart

    let decodeStart = CFAbsoluteTimeGetCurrent()
    var generated = [Int]()
    generated.reserveCapacity(outputBudget)
    let nextCycle: () -> DFlash2CycleResult
    if let fixedPhysicalWidth {
        nextCycle = { session.warmStep(physicalWidth: fixedPhysicalWidth) }
    } else {
        nextCycle = { session.step() }
    }
    while generated.count < outputBudget {
        let cycle = nextCycle()
        let cycleTokens = cycle.committedTokens.asArray(Int32.self).map(Int.init)
        generated.append(contentsOf: cycleTokens.prefix(outputBudget - generated.count))
    }
    let decodeSeconds = CFAbsoluteTimeGetCurrent() - decodeStart
    return (
        generated,
        RunReceipt(
            mode: Qwen38RunnerMode.dflash2.rawValue,
            promptTokens: prompt.dim(1),
            outputTokens: generated.count,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds,
            decodeTokensPerSecond: Double(generated.count) / decodeSeconds,
            tokenSHA256: tokenDigest(generated))
    )
}

/// Compile and execute every decode geometry on throwaway state before timing.
/// The 2,048-row seed is the same physical prompt chunk used by production;
/// widths 1...8 cover the direct AR lane and every adaptive/fixed DFlash route.
private func warmRuntime(
    target: any DFlash2QwenTarget,
    draft: DFlash2DraftModel?
) {
    let warmTokens = MLXArray([Int32](repeating: 1, count: 2_048))
        .reshaped([1, 2_048])
    if let draft {
        let session = DFlash2Session(
            target: target,
            draft: draft,
            promptLength: 2_048,
            seed: 0)
        let frontier = session.prefill(promptTokens: warmTokens)
        eval(frontier)
        for width in 1 ... 8 {
            let cycle = session.warmStep(physicalWidth: width)
            eval(cycle.committedTokens, cycle.nextToken)
        }
    } else {
        let cache = target.newCache(parameters: nil)
        let logits = target.dflashTargetOnlyPrefillChunk(
            input: LMInput.Text(tokens: warmTokens),
            cache: cache)
        let sampler = Qwen38TargetSampler(seed: 0)
        let token = sampler.sample(logits: logits[0..., -1, 0...])
            .reshaped([1, 1])
        let nextLogits = target(token, cache: cache)
        let next = sampler.sample(logits: nextLogits[0..., -1, 0...])
        eval(next, cache.flatMap { $0.innerState() })
    }
}

@main
private struct Qwen38DFlash2Runner {
    static func main() async {
        do {
            let options = try Qwen38RunnerOptions.parse(
                Array(CommandLine.arguments.dropFirst()))
            FileHandle.standardError.write(
                Data("Powered by MTPLX\nhttps://github.com/youssofal/mtplx\n".utf8))
            try installRuntimeContract()
            try validateLocalArtifacts(options)
            let manifest = Qwen38ArtifactManifest.production
            let targetDirectory = URL(fileURLWithPath: options.targetPath)
            let targetConfigSHA256 = try Qwen38ArtifactValidator.validate(
                directory: targetDirectory,
                reference: manifest.targetArtifact)
            let draftConfigSHA256: String?
            if let draftPath = options.draftPath {
                draftConfigSHA256 = try Qwen38ArtifactValidator.validate(
                    directory: URL(fileURLWithPath: draftPath),
                    reference: manifest.draftArtifact)
            } else {
                draftConfigSHA256 = nil
            }
            let gpuLock = try GPUFileLock(path: "/tmp/mtplx-gpu-exclusive.lock")
            defer { _ = gpuLock }
            let container = try await LLMModelFactory.shared.loadContainer(
                from: targetDirectory,
                using: #huggingFaceTokenizerLoader())
            let draft: DFlash2DraftModel?
            if let draftPath = options.draftPath {
                draft = try await DFlash2DraftModel.load(
                    from: URL(fileURLWithPath: draftPath))
            } else {
                draft = nil
            }
            let draftProjectionReport = try draft.map {
                try installQwen38ProjectionStack(in: $0)
            }
            let outcome = try await container.perform {
                (context: ModelContext) async throws -> RunnerOutcome in
                guard let target = context.model as? any DFlash2QwenTarget else {
                    throw RunnerError.message(
                        "target model is not the pinned Qwen 3.8 text model: "
                            + "\(type(of: context.model))")
                }
                let targetProjectionReport = try installQwen38ProjectionStack(
                    in: context.model)
                let ids = try promptTokens(options: options, tokenizer: context.tokenizer)
                guard !ids.isEmpty else {
                    throw RunnerError.message("prompt token sequence is empty")
                }
                try validateQwen38PromptTokenIDs(ids)
                let prompt = MLXArray(ids.map(Int32.init)).reshaped([1, ids.count])

                warmRuntime(target: target, draft: draft)
                Memory.clearCache()
                let activeMemoryAtConstruction = Memory.activeMemory
                let wiredMemoryLimit = Qwen38WiredMemoryContract.limit(
                    activeBytes: activeMemoryAtConstruction,
                    recommendedMaximum: GPU.maxRecommendedWorkingSetBytes(),
                    physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)
                let wiredTicket = wiredMemoryLimit.map {
                    WiredFixedPolicy(limit: $0).ticket(size: 0)
                }
                let wiredMemoryApplied: Int
                if let wiredTicket, let wiredMemoryLimit {
                    wiredMemoryApplied = await wiredTicket.start()
                    guard wiredMemoryApplied == wiredMemoryLimit else {
                        _ = await wiredTicket.end()
                        throw RunnerError.message(
                            "cannot install wired residency limit: requested "
                                + "\(wiredMemoryLimit), applied \(wiredMemoryApplied)")
                    }
                } else {
                    wiredMemoryApplied = 0
                }

                let runs: [(tokens: [Int], receipt: RunReceipt)]
                switch options.mode {
                case .autoregressive:
                    runs = [
                        runAutoregressive(
                            target: target, prompt: prompt, outputBudget: options.maxTokens)
                    ]
                case .dflash2:
                    let draft = draft!
                    runs = [
                        runDFlash2(
                            target: target, draft: draft, prompt: prompt,
                            outputBudget: options.maxTokens,
                            fixedPhysicalWidth: options.dflashPhysicalWidth)
                    ]
                case .benchmark:
                    let draft = draft!
                    runs = [
                        runAutoregressive(
                            target: target, prompt: prompt, outputBudget: options.maxTokens),
                        runDFlash2(
                            target: target, draft: draft, prompt: prompt,
                            outputBudget: options.maxTokens,
                            fixedPhysicalWidth: options.dflashPhysicalWidth),
                        runDFlash2(
                            target: target, draft: draft, prompt: prompt,
                            outputBudget: options.maxTokens,
                            fixedPhysicalWidth: options.dflashPhysicalWidth),
                        runAutoregressive(
                            target: target, prompt: prompt, outputBudget: options.maxTokens),
                    ]
                }
                if let wiredTicket { _ = await wiredTicket.end() }

                let text =
                    options.mode == .benchmark
                    ? nil
                    : context.tokenizer.decode(tokenIds: runs[0].tokens)
                let receipt = RunnerReceipt(
                    createdAt: ISO8601DateFormatter().string(from: Date()),
                    targetPath: options.targetPath,
                    draftPath: options.draftPath,
                    targetRepository: manifest.targetArtifact.repository,
                    targetRevision: manifest.targetArtifact.revision,
                    targetConfigSHA256: targetConfigSHA256,
                    draftRepository: manifest.draftArtifact.repository,
                    draftRevision: manifest.draftArtifact.revision,
                    draftConfigSHA256: draftConfigSHA256,
                    swiftBaseRevision: manifest.swiftBaseRevision,
                    mlxSwiftRevision: manifest.mlxSwiftRevision,
                    mlxRevision: manifest.mlxRevision,
                    mtplxSourceRevision: manifest.mtplxSourceRevision,
                    targetSourceRevision: manifest.yukonSourceRevision,
                    dflash2SourceRevision: manifest.dflash2SourceRevision,
                    commandBufferMegabytes: 512,
                    commandBufferOperations: 50,
                    dflashPhysicalWidth: options.dflashPhysicalWidth ?? 0,
                    targetOptimizedProjections: targetProjectionReport.installed,
                    targetPreservedFusedGDNInputs:
                        targetProjectionReport.preservedFusedGDNInputs,
                    draftOptimizedProjections: draftProjectionReport?.installed ?? 0,
                    activeMemoryAtConstruction: activeMemoryAtConstruction,
                    wiredMemoryLimit: wiredMemoryLimit ?? 0,
                    wiredMemoryApplied: wiredMemoryApplied,
                    targetTemperature: Qwen38TargetSampler.temperature,
                    targetTopP: Qwen38TargetSampler.topP,
                    targetTopK: Qwen38TargetSampler.topK,
                    targetSeed: Qwen38TargetSampler.seed,
                    runs: runs.map(\.receipt))
                return RunnerOutcome(text: text, receipt: receipt)
            }

            if let text = outcome.text { print(text) }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let receiptData = try encoder.encode(outcome.receipt)
            FileHandle.standardError.write(receiptData)
            FileHandle.standardError.write(Data("\n".utf8))
            if let path = options.receiptPath {
                try receiptData.write(to: URL(fileURLWithPath: path), options: .atomic)
            }
        } catch let error as Qwen38RunnerOptionError {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            FileHandle.standardError.write(Data("\(Qwen38RunnerOptions.usage)\n".utf8))
            exit(64)
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exit(70)
        }
    }
}
