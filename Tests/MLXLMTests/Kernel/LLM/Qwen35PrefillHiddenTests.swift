import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom
import Testing

@testable import MLXLLM

extension KernelTests {

    /// The prompt entry that Qwen3.5 gives the engine while an inline MTP
    /// assistant keeps a hidden history for the request
    /// (`cbv2ForwardWithHiddenForPrefill`).
    ///
    /// The assistant needs the pre-norm hidden row of every prompt position.
    /// The engine needs vocabulary logits for the last position only, or just
    /// a handle. The entry must therefore return the hidden history of
    /// `cbv2ForwardWithHidden`, leave the same K/V rows and recurrent state
    /// behind, and give the last logits row that the prompt path without MTP
    /// (`cbv2RecurrentPrefill`) gives.
    @Suite(.serialized)
    struct Qwen35PrefillHiddenTests {

        static let vocabularySize = 64
        static let hiddenSize = 64

        /// Four layers of width 64: gated delta net layers with a full
        /// attention layer every second layer. `experts == 0` is the dense
        /// model; otherwise each layer has a sparse MoE block.
        static func makeModel(experts: Int, quantized: Bool) throws -> Qwen35TextModel {
            let json = """
                {"model_type":"qwen3_5_moe_text","hidden_size":\(hiddenSize),"num_hidden_layers":4,
                 "intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":64,
                 "linear_num_value_heads":1,"linear_num_key_heads":1,"linear_key_head_dim":64,
                 "linear_value_head_dim":64,"linear_conv_kernel_dim":4,"full_attention_interval":2,
                 "vocab_size":\(vocabularySize),"num_experts":\(experts),
                 "num_experts_per_tok":\(experts > 0 ? 2 : 0),
                 "moe_intermediate_size":32,"shared_expert_intermediate_size":32,"norm_topk_prob":true}
                """
            MLXRandom.seed(7029)
            let model = Qwen35TextModel(
                try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8)))
            // bfloat16 as the served checkpoints are; quantization comes after
            // the cast because it packs the weights into integers.
            model.update(
                parameters: ModuleParameters.unflattened(
                    model.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }))
            if quantized {
                quantize(model: model, groupSize: 32, bits: 4)
            }
            eval(model)
            return model
        }

        static func tokens(_ values: [Int]) -> MLXArray {
            MLXArray(values.map { Int32($0) }).reshaped([1, values.count])
        }

        static func difference(_ a: MLXArray, _ b: MLXArray) -> Float {
            guard a.shape == b.shape else { return .infinity }
            return abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
        }

        /// One request: its attention rows and its recurrent state.
        final class Request {
            let caches: [KVCache]
            let state: CBv2RecurrentRequestState

            init(_ model: Qwen35TextModel, promptLength: Int) throws {
                let kinds = model.cbv2LayerKinds
                let backend = CBv2ContiguousKVBackend(
                    config: .init(bytesCapacity: 64 << 20, kvDType: .bfloat16))
                let row = try backend.makeSequenceState(
                    layerKinds: kinds, promptLength: promptLength, maxLength: promptLength + 16)
                caches = CBv2LayerCacheBank(layerKinds: kinds).layerCaches(rowStates: [row]).map {
                    guard let cache = $0 as? KVCache else {
                        fatalError("CBv2 layer cache \(type(of: $0)) must conform to KVCache")
                    }
                    return cache
                }
                state = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
            }

            /// Runs one forward against this request, evaluates it with its
            /// staged recurrent state, and commits the state.
            func commit<Output>(
                _ forward: ([KVCache], CBv2RecurrentStateEvaluation) -> Output,
                arrays: (Output) -> [MLXArray]
            ) throws -> Output {
                let binding = try state.bind()
                let output = forward(caches, binding)
                eval(arrays(output) + (try binding.evaluate()))
                try binding.commit()
                return output
            }
        }

        // Tolerance of the comparison of one projected row with the same row
        // of a projection of the whole window: the same equations run with
        // another matrix shape, so sums can run in another order and the
        // bfloat16 result can round the other way. 0.0625 is four units in
        // the last place of a bfloat16 value between 2 and 4. A wrong row or
        // a wrong state gives differences near 1.
        static let tolerance: Float = 0.0625

        static let prompt = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 58]

        @Test(arguments: [0, 4], [false, true])
        func prefillEntryKeepsTheHistoryAndProjectsOnlyTheLastRow(
            experts: Int, quantized: Bool
        ) throws {
            let model = try Self.makeModel(experts: experts, quantized: quantized)
            let input = Self.tokens(Self.prompt)
            let length = Self.prompt.count
            let label = "experts=\(experts), quantized=\(quantized)"

            // The entry the engine used before: logits for every position.
            let full = try Request(model, promptLength: length)
            let reference = try full.commit(
                {
                    model.cbv2ForwardWithHidden(
                        input, caches: $0, recurrentState: [$1], positionIds: nil)
                },
                arrays: { [$0.logits, $0.lastHidden] })
            #expect(reference.logits.shape == [1, length, Self.vocabularySize], "\(label)")
            #expect(reference.lastHidden.shape == [1, length, Self.hiddenSize], "\(label)")

            // The prompt path without MTP: the last row only.
            let plain = try Request(model, promptLength: length)
            let plainLast = try plain.commit(
                {
                    model.cbv2RecurrentPrefill(
                        input, inputEmbedding: nil, cache: $0, recurrentState: [$1],
                        positionIds: nil, requirement: .lastPositionLogits)
                },
                arrays: { [$0] })
            #expect(plainLast.shape == [1, Self.vocabularySize], "\(label)")

            // The new entry, frontier chunk.
            let frontier = try Request(model, promptLength: length)
            let narrowed = try frontier.commit(
                {
                    model.cbv2ForwardWithHiddenForPrefill(
                        input, caches: $0, recurrentState: [$1], positionIds: nil,
                        requirement: .lastPositionLogits)
                },
                arrays: { [$0.logits, $0.lastHidden] })
            #expect(narrowed.logits.shape == [1, 1, Self.vocabularySize], "\(label)")
            // Same trunk call on the same data: the history is exact.
            #expect(Self.difference(narrowed.lastHidden, reference.lastHidden) == 0, "\(label)")
            // Same operations as the prompt path without MTP: exact.
            #expect(
                Self.difference(narrowed.logits[0..., 0, 0...], plainLast) == 0, "\(label)")
            // The row of the whole-window projection, up to summation order.
            #expect(
                Self.difference(
                    narrowed.logits, reference.logits[0..., (length - 1)..., 0...])
                    <= Self.tolerance, "\(label)")

            // The new entry, a chunk before the frontier: a handle only.
            let earlier = try Request(model, promptLength: length)
            let handle = try earlier.commit(
                {
                    model.cbv2ForwardWithHiddenForPrefill(
                        input, caches: $0, recurrentState: [$1], positionIds: nil,
                        requirement: .evaluationOnly)
                },
                arrays: { [$0.logits, $0.lastHidden] })
            #expect(handle.logits.shape == [1, 1, 1], "\(label)")
            #expect(Self.difference(handle.lastHidden, reference.lastHidden) == 0, "\(label)")
            #expect(
                Self.difference(
                    handle.logits, reference.lastHidden[0..., (length - 1)..., 0 ..< 1]) == 0,
                "\(label)")

            // The K/V rows and the recurrent state that each entry left behind
            // are the same: the next token gives the same logits and hidden.
            let next = Self.tokens([7])
            func step(_ request: Request) throws -> (logits: MLXArray, lastHidden: MLXArray) {
                try request.commit(
                    {
                        model.cbv2ForwardWithHidden(
                            next, caches: $0, recurrentState: [$1], positionIds: nil)
                    },
                    arrays: { [$0.logits, $0.lastHidden] })
            }
            let expected = try step(full)
            for (name, request) in [("frontier", frontier), ("earlier", earlier)] {
                let actual = try step(request)
                #expect(
                    Self.difference(actual.logits, expected.logits) == 0, "\(label), \(name)")
                #expect(
                    Self.difference(actual.lastHidden, expected.lastHidden) == 0,
                    "\(label), \(name)")
            }
        }

        /// The engine reaches the model through the steppable adapter, which
        /// falls back to the whole-window projection for a model without the
        /// entry. The narrowed shape shows that the adapter takes the entry.
        @Test func adapterTakesThePrefillEntry() throws {
            let model = try Self.makeModel(experts: 0, quantized: false)
            let adapter = CBv2SteppableLanguageModelAdapter(model)
            let request = try Request(model, promptLength: Self.prompt.count)
            let attending = request.caches.map { $0 as! any CBv2AttendingLayerCache }
            let binding = try request.state.bind()
            let output = adapter.forwardWithHiddenForPrefill(
                tokens: Self.tokens(Self.prompt), caches: attending, recurrentState: [binding],
                positionIds: nil, requirement: .lastPositionLogits)
            eval([output.logits, output.lastHidden] + (try binding.evaluate()))
            #expect(output.logits.shape == [1, 1, Self.vocabularySize])
            #expect(output.lastHidden.shape == [1, Self.prompt.count, Self.hiddenSize])
        }

        /// The provider serves the dense, MoE and Prism Hadamard checkpoints
        /// through `Qwen35Model` and its subclasses.
        @Test func wrappersConform() {
            #expect((Qwen35Model.self as Any) is any CBv2RecurrentPrefillHiddenForwardable.Type)
            #expect((Qwen35MoEModel.self as Any) is any CBv2RecurrentPrefillHiddenForwardable.Type)
            #expect(
                (PrismHadamardQwen35TextModel.self as Any)
                    is any CBv2RecurrentPrefillHiddenForwardable.Type)
        }
    }
}
