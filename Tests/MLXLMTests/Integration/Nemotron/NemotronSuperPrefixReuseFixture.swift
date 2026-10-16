import Foundation
import MLX

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// EngineV2 serving of the synthetic Super trunk under the owner's serving
/// policy (fixed prefill chunks, one prompt chunk per step), with or without
/// the resident recurrent checkpoint bank. Tiled products and BF16 KV, as the
/// plain owner runs the full checkpoint.
enum NemotronSuperPrefixReuseFixture {
    typealias Batch = NemotronSuperBatchFixture

    /// A multiple of the attention query block, as the owner's 2048 is.
    static let chunk: Int = 128
    static let hidden: Int = Batch.configuration().hiddenSize

    static func model() throws -> NemotronSuperModel {
        try Batch.model(seed: 23, dtype: .bfloat16, products: .tiled)
    }

    static func bank(
        bytes: Int = 64 << 20, checkpointsPerRequest: Int = 2
    ) -> CBv2HybridPrefixCacheConfig {
        .init(
            maximumBytes: bytes, maximumEntries: 8,
            maximumCheckpointsPerRequest: checkpointsPerRequest,
            modelID: "nemotron-super-fixture", promptContractID: "fixture-template",
            buildID: "fixture-build")
    }

    /// `bank == nil` is the plain owner: no checkpoint capability, no cache.
    /// `rows` above one is the batched owner: row-exact `[B, 1]` decode, one
    /// prompt chunk per step beside every decode row.
    static func engine(
        _ model: NemotronSuperModel, bank: CBv2HybridPrefixCacheConfig?, chunk: Int = chunk,
        rows: Int = 1
    ) throws -> (EngineV2, CBv2ContiguousKVBackend) {
        let adapter = try NemotronSuperRecurrentAdapter(
            model, maximumDecodeRows: rows, decodeProducts: .rowExact,
            exactPrefixCheckpoints: bank != nil)
        var config = CBv2SchedulerConfig(
            maxConcurrentRequests: rows, maxBatchedTokensPerStep: chunk + rows,
            prefillChunkSize: chunk, maxConcurrentPartialPrefills: 1,
            mixedStepPrefillTokenCap: chunk, maxWaiting: 8, enablePrefixCache: bank != nil)
        config.maxPrefillRowsPerStep = 1
        let backend = CBv2ContiguousKVBackend(
            config: .init(bytesCapacity: 256 << 20, kvDType: .bfloat16))
        let engine = EngineV2(
            model: CBv2SteppableLanguageModelAdapter(adapter), layerKinds: adapter.cbv2LayerKinds,
            backend: backend,
            cacheProvider: CBv2LayerCacheBank(
                caches: adapter.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) }),
            sampler: CBv2DefaultSampler(), schedulerConfig: config,
            admissionConfig: .init(watermarkFraction: 0), hybridPrefixCache: bank)
        return (engine, backend)
    }

    /// Token streams that avoid the image placeholder.
    static func prompt(_ count: Int, seed: Int) -> [Int] {
        Batch.tokens(count, seed: seed).map { Int($0) }
    }

    struct Served {
        var tokens: [Int] = []
        /// Float32 bits of the full-vocabulary log-softmax of the raw logits
        /// at every generated position, in token-id order.
        var logprobBits: [[UInt32]] = []
        var finish: CBv2FinishReason?
        var usage: CBv2Usage?
    }

    static func request(
        _ id: UInt64, _ prompt: [Int], maxTokens: Int = 6, salt: String? = nil,
        multimodal: CBv2MultimodalInput? = nil
    ) -> CBv2Request {
        CBv2Request(
            id: CBv2RequestID(id), promptTokens: prompt,
            sampling: .init(temperature: 0, topLogprobs: Batch.vocabulary), maxTokens: maxTokens,
            cacheSalt: salt, multimodal: multimodal)
    }

    static func serve(
        _ engine: EngineV2, _ request: CBv2Request, onTokens: (Int) -> Void = { _ in }
    ) async throws -> Served {
        var served = Served()
        for await event in try engine.submit(request) {
            switch event {
            case .delta(_, let tokens, let logprobs):
                served.tokens += tokens
                for entry in logprobs ?? [] {
                    var row = [UInt32](repeating: 0, count: Batch.vocabulary)
                    for (token, value) in entry.topLogprobs { row[token] = value.bitPattern }
                    served.logprobBits.append(row)
                }
                onTokens(served.tokens.count)
            case .finished(let reason, let usage):
                served.finish = reason
                served.usage = usage
            }
        }
        return served
    }

    /// Bit-for-bit equality of shape, dtype and every element, NaN and signed
    /// zero included. Copies first so strided cache views compare as values.
    static func bitwise(_ a: MLXArray, _ b: MLXArray) -> Bool {
        guard a.shape == b.shape, a.dtype == b.dtype else { return false }
        let view: DType
        switch a.dtype.size {
        case 2: view = .uint16
        case 4: view = .uint32
        default: return false
        }
        let left = MLX.where(MLXArray(true), a, a)
        let right = MLX.where(MLXArray(true), b, b)
        eval(left, right)
        return arrayEqual(left.view(dtype: view), right.view(dtype: view)).item(Bool.self)
    }

    /// The bank's deepest checkpoint below the end of `tokens`, compared
    /// bit for bit with another bank's: conv/SSM state of every Mamba layer
    /// and the attention KV prefix. Releases both pins.
    static func checkpointsMatch(
        _ first: CBv2HybridPrefixCache, _ second: CBv2HybridPrefixCache, tokens: [Int],
        position: Int
    ) -> Bool {
        guard let a = first.lookup(tokens: tokens, cacheSalt: nil, maximumChunkSize: chunk)
        else { return false }
        defer { first.endAdoption(pin: a.pin) }
        guard let b = second.lookup(tokens: tokens, cacheSalt: nil, maximumChunkSize: chunk)
        else { return false }
        defer { second.endAdoption(pin: b.pin) }
        guard a.checkpoint.position == position, b.checkpoint.position == position,
            a.checkpoint.layers.keys.sorted() == b.checkpoint.layers.keys.sorted(),
            !a.checkpoint.layers.isEmpty, a.kvPrefix.count == b.kvPrefix.count
        else { return false }
        for (layer, state) in a.checkpoint.layers {
            guard let other = b.checkpoint.layers[layer], let conv = state.conv,
                let ssm = state.ssm, let otherConv = other.conv, let otherSSM = other.ssm,
                bitwise(conv, otherConv), bitwise(ssm, otherSSM)
            else { return false }
        }
        for (row, otherRow) in zip(a.kvPrefix, b.kvPrefix) {
            guard let row, let otherRow, row.offset == position, otherRow.offset == position,
                bitwise(row.keys, otherRow.keys), bitwise(row.values, otherRow.values)
            else { return false }
        }
        return true
    }

    /// Polls `condition` for up to five seconds.
    static func eventually(_ condition: () -> Bool) async -> Bool {
        for _ in 0 ..< 500 {
            if condition() { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return condition()
    }
}
