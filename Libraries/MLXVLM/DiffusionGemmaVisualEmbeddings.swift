import MLX
import MLXLLM
import MLXLMCommon

/// Immutable evaluated media features owned by one native generation session.
/// There is no autoregressive model/cache adapter and no provisional KV owner.
public struct DiffusionGemmaVisualEmbeddings {
    let spans: [CBv2ImageSpan]
    let blocks: [CBv2ImageSpan]
    let values: [MLXArray]
    public var retainedBytes: Int { values.reduce(0) { $0 + $1.nbytes } }
    var maximumBlockLength: Int { blocks.map(\.length).max() ?? 0 }

    static func validate(spans: [CBv2ImageSpan], promptCount: Int) throws {
        guard !spans.isEmpty, promptCount > 0 else {
            throw DiffusionGemmaModelError.invalidInput("empty visual spans")
        }
        var end = 0
        for span in spans {
            guard span.tokenOffset >= end, span.tokenOffset < promptCount,
                span.length > 0, span.length <= 1120,
                span.length <= promptCount - span.tokenOffset
            else { throw DiffusionGemmaModelError.invalidInput("native visual span range/order") }
            end = span.tokenOffset + span.length
        }
        guard coalesced(spans).allSatisfy({ $0.length <= 1120 }) else {
            throw DiffusionGemmaModelError.invalidInput("native visual block budget")
        }
    }

    static func coalesced(_ spans: [CBv2ImageSpan]) -> [CBv2ImageSpan] {
        var blocks = [CBv2ImageSpan]()
        for span in spans {
            if let last = blocks.indices.last, blocks[last].tokenOffset + blocks[last].length == span.tokenOffset {
                blocks[last].length += span.length
            } else { blocks.append(span) }
        }
        return blocks
    }

    public init(input: CBv2MultimodalInput, promptCount: Int, hiddenSize: Int) throws {
        guard input.attention == .bidirectionalSpans, input.positionState == nil, input.deepstackEmbeddings == nil else {
            throw DiffusionGemmaModelError.invalidInput("non-native visual attention/positions")
        }
        try Self.validate(spans: input.spans, promptCount: promptCount)
        let provided = try input.embeddings()
        guard provided.count == input.spans.count else {
            throw DiffusionGemmaModelError.invalidInput("native visual feature count")
        }
        self.spans = input.spans
        self.blocks = Self.coalesced(input.spans)
        self.values = try zip(input.spans, provided).map { span, value in
            let shaped = value.ndim == 2 ? value.expandedDimensions(axis: 0) : value
            guard shaped.shape == [1, span.length, hiddenSize],
                [.float16, .bfloat16, .float32].contains(shaped.dtype) else {
                throw DiffusionGemmaModelError.invalidInput("native visual feature dimensions")
            }
            eval(shaped)
            return shaped[0..., 0..., 0...]
        }
    }

    func chunkLength(start: Int, requested: Int, promptCount: Int) throws -> Int {
        guard start >= 0, requested > 0, start < promptCount else {
            throw DiffusionGemmaModelError.invalidInput("media chunk range")
        }
        var end = start + min(requested, promptCount - start)
        for block in blocks where block.tokenOffset < end && block.tokenOffset + block.length > end {
            end = block.tokenOffset > start ? block.tokenOffset : block.tokenOffset + block.length
        }
        guard end > start, end <= promptCount else {
            throw DiffusionGemmaModelError.invalidInput("media chunk progress")
        }
        return end - start
    }

    func encode(model: DiffusionGemma, tokens: MLXArray, cache: DiffusionGemmaRequestCache, start: Int) throws {
        let count = tokens.dim(1), end = start + tokens.dim(1)
        guard cache.position == start else { throw DiffusionGemmaModelError.invalidInput("media chunk cache position") }
        let local = spans.indices.filter { spans[$0].tokenOffset < end && spans[$0].tokenOffset + spans[$0].length > start }
        guard local.allSatisfy({ spans[$0].tokenOffset >= start && spans[$0].tokenOffset + spans[$0].length <= end }) else {
            throw DiffusionGemmaModelError.invalidInput("split visual span")
        }
        if local.isEmpty { _ = try model.encode(tokenIds: tokens, cache: cache); return }
        var blockIDs = Array(repeating: Int32(-1), count: count)
        for (index, block) in blocks.enumerated() where block.tokenOffset < end && block.tokenOffset + block.length > start {
            guard block.tokenOffset >= start, block.tokenOffset + block.length <= end else {
                throw DiffusionGemmaModelError.invalidInput("split visual block")
            }
            for position in block.tokenOffset..<(block.tokenOffset + block.length) { blockIDs[position - start] = Int32(index) }
        }
        let blocks = MLXArray(blockIDs).reshaped(1, count)
        let embeddedIDs = which(blocks .>= 0, MLXArray(Int32(model.configuration.textConfig.padTokenId ?? 0)), tokens)
        let embedded = try model.model.decoder.scaledEmbeddings(embeddedIDs)
        for index in local {
            let span = spans[index]
            embedded[0..., (span.tokenOffset - start)..<(span.tokenOffset + span.length - start), 0...] = values[index].asType(embedded.dtype)
        }
        _ = try model.model.decoder.encode(tokenIds: tokens, cache: cache,
            encoderParameters: model.model.encoder.languageModel,
            preparedEmbeddings: embedded, visualBlockIds: blocks)
    }
}
