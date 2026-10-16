import Foundation
import MLX
import Testing

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Exact prefix reuse of Super image requests. An image is a run of causal
/// embeddings at scalar positions, so a request bound to an identity of its
/// evaluated embeddings and spans checkpoints like text, in that identity's
/// own scope: a later turn with the same image restores and continues the
/// cold prefill bit for bit; another identity, or text, never sees it.
/// Synthetic weights, tiled BF16 products, 128-token chunks.
@Suite(.tags(.integration))
struct NemotronSuperMediaPrefixReuseTests {
    typealias Fixture = NemotronSuperPrefixReuseFixture

    static let chunk: Int = Fixture.chunk
    static let imageAt: Int = 100
    static let imageTokens: Int = 6
    static let placeholder: Int = Int(NemotronSuperBatchFixture.imageToken)
    /// One turn: text, the image span (crossing no chunk end), more text.
    static let turn: [Int] =
        Fixture.prompt(imageAt, seed: 31) + [Int](repeating: placeholder, count: imageTokens)
        + Fixture.prompt(2 * chunk + 20 - imageAt - imageTokens, seed: 37)
    /// The next turn repeats it and adds text.
    static let nextTurn: [Int] = turn + Fixture.prompt(chunk + 40, seed: 43)

    static func identity(_ byte: UInt8) throws -> CBv2HybridPrefixIdentity {
        try CBv2HybridPrefixIdentity(digest: Data(repeating: byte, count: 32))
    }

    /// The same embeddings for the same `scale`; the identity names them.
    static func image(scale: Float) -> CBv2MultimodalInput {
        let count: Int = imageTokens * Fixture.hidden
        let values: [Float] = (0 ..< count).map { (index: Int) -> Float in
            Float(index % 97) * scale / 1000
        }
        let embedding = MLXArray(values).reshaped([1, imageTokens, Fixture.hidden])
            .asType(.bfloat16)
        return CBv2MultimodalInput(
            spans: [CBv2ImageSpan(tokenOffset: imageAt, length: imageTokens)], attention: .causal
        ) { [embedding] }
    }

    static func request(
        _ id: UInt64, _ prompt: [Int], scale: Float = 1, identity: CBv2HybridPrefixIdentity?
    ) -> CBv2Request {
        var request = Fixture.request(id, prompt, multimodal: image(scale: scale))
        request.hybridPrefixIdentity = identity
        return request
    }

    /// The donor turn publishes C and 2C, both after the image. The next turn with the
    /// same identity restores 2C and gives the cold tokens and full-vocabulary
    /// log-softmax bit for bit. Other embeddings under another identity, the
    /// same image without an identity, and text with the same tokens restore
    /// nothing.
    @Test(.timeLimit(.minutes(5)))
    func sameImageRestoresExactlyAndNothingElseSeesIt() async throws {
        let model = try Fixture.model()
        let x = try Self.identity(0x58)
        let y = try Self.identity(0x59)

        let (plain, _) = try Fixture.engine(model, bank: nil)
        let reference = try await Fixture.serve(
            plain, Self.request(1, Self.nextTurn, identity: nil))
        let otherReference = try await Fixture.serve(
            plain, Self.request(2, Self.nextTurn, scale: 2, identity: nil))
        await plain.shutdown()
        #expect(
            reference.tokens != otherReference.tokens
                || reference.logprobBits != otherReference.logprobBits)

        let (engine, backend) = try Fixture.engine(model, bank: Fixture.bank())
        let bank = try #require(engine.hybridPrefixCache)
        let donor = try await Fixture.serve(engine, Self.request(3, Self.turn, identity: x))
        #expect(donor.finish == .length)
        let published = await Fixture.eventually { bank.stats.entries == 1 }
        #expect(published)

        let other = try await Fixture.serve(
            engine, Self.request(4, Self.nextTurn, scale: 2, identity: y))
        #expect(other.usage?.prefixCachePrefillTokensSaved == 0)
        #expect(other.tokens == otherReference.tokens)
        #expect(other.logprobBits == otherReference.logprobBits)
        let unbound = try await Fixture.serve(engine, Self.request(5, Self.nextTurn, identity: nil))
        #expect(unbound.usage?.prefixCachePrefillTokensSaved == 0)
        let text = try await Fixture.serve(engine, Fixture.request(6, Self.nextTurn))
        #expect(text.usage?.prefixCachePrefillTokensSaved == 0)

        let restored = try await Fixture.serve(engine, Self.request(7, Self.nextTurn, identity: x))
        let restoredAt: Int = 2 * Self.chunk
        #expect(restored.usage?.prefixCacheOutcome == .hit)
        #expect(restored.usage?.prefixCachePrefillTokensSaved == restoredAt)
        #expect(restored.tokens == reference.tokens)
        #expect(restored.logprobBits == reference.logprobBits)
        await engine.shutdown()
        #expect(backend.bytesReserved == 0)
        #expect(bank.stats.retainedBytes == 0)
    }
}
