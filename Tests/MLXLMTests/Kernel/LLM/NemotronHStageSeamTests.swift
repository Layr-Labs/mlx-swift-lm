import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// The layer-stage seam of `NemotronHModel`: two models constructed over
    /// the two halves of a block pattern, joined only by the residual at the
    /// cut, against the complete model on the same tokens.
    ///
    /// Both sides run the same operations on the same values, so every
    /// comparison is exact. The complete model is driven through the serving
    /// entry points, which the seam must not change.
    @Suite
    struct NemotronHStageSeamTests {

        /// Mamba, expert and attention blocks on both sides of either cut.
        static let pattern = [
            "mamba", "moe", "mamba", "attention", "moe", "mamba", "moe", "attention", "moe",
        ]

        /// One request's attention rows and recurrent state for one model.
        final class RequestState {
            let model: NemotronHModel
            let backend: CBv2ContiguousKVBackend
            let bank: CBv2LayerCacheBank
            let rows: [CBv2SequenceKV?]
            let recurrent: CBv2RecurrentRequestState

            init(_ model: NemotronHModel, maximumTokens: Int) throws {
                self.model = model
                let kinds = model.cbv2LayerKinds
                backend = CBv2ContiguousKVBackend(
                    config: .init(bytesCapacity: 64 << 20, kvDType: .float32))
                rows = try backend.makeSequenceState(
                    layerKinds: kinds, promptLength: maximumTokens, maxLength: maximumTokens)
                bank = CBv2LayerCacheBank(
                    caches: model.newCacheV2 { CBv2LayerCache(layerIndex: $0, kind: $1) })
                recurrent = try CBv2RecurrentRequestState(spec: model.cbv2RecurrentStateSpec)
            }

            /// One committed forward: binds the state, evaluates the output
            /// with every staged state root, and commits.
            func run(_ forward: ([KVCache], CBv2RecurrentStateEvaluation) -> MLXArray) throws
                -> MLXArray
            {
                let caches = bank.layerCaches(rowStates: [rows])
                let binding = try recurrent.bind()
                let output = forward(caches.map { $0 as! KVCache }, binding)
                let roots = try binding.evaluate()
                eval([output] + roots + caches.flatMap { ($0 as! KVCache).innerState() })
                try binding.commit()
                return output
            }

            deinit {
                bank.releaseBoundRows()
                backend.release(rows)
                try? recurrent.release()
            }
        }

        /// A model over `blocks[range]` that shares the complete model's
        /// parameters for those blocks, and for whichever end it keeps.
        static func stage(
            of full: NemotronH35Model, range: Range<Int>
        ) throws -> NemotronH35Model {
            let model = try NemotronHRuntimeTests.makeLightning(
                Array(pattern[range]), seed: 99)
            var shared: [String: MLXArray] = [:]
            for (name, value) in full.parameters().flattened() {
                let prefix = "backbone.layers."
                if name.hasPrefix(prefix) {
                    let rest = name.dropFirst(prefix.count)
                    let index = Int(rest.prefix(while: { $0 != "." }))!
                    guard range.contains(index) else { continue }
                    let local = prefix + String(index - range.lowerBound)
                        + String(rest.drop(while: { $0 != "." }))
                    shared[local] = value
                } else if name.hasPrefix("backbone.embeddings.") {
                    if range.lowerBound == 0 { shared[name] = value }
                } else if range.upperBound == pattern.count {
                    shared[name] = value  // norm_f and the head
                }
            }
            try model.update(
                parameters: ModuleParameters.unflattened(shared), verify: [.noUnusedKeys])
            eval(model)
            return model
        }

        static func rows(_ values: [Int]) -> MLXArray { NemotronHRuntimeTests.tokens(values) }

        /// Prompt chunks of 5 and 3 tokens, then two decode tokens.
        static let frames: [[Int]] = [[3, 1, 4, 1, 5], [9, 2, 6], [5], [3]]

        /// The complete model through its serving entry points.
        static func serve(_ full: NemotronH35Model) throws -> ([MLXArray], RequestState) {
            let state = try RequestState(full, maximumTokens: frames.joined().count)
            var outputs: [MLXArray] = []
            for (index, frame) in frames.enumerated() {
                let input = rows(frame)
                outputs.append(
                    try state.run { caches, binding in
                        switch index {
                        case 0:
                            return full.cbv2RecurrentPrefill(
                                input, inputEmbedding: nil, cache: caches,
                                recurrentState: [binding], positionIds: nil,
                                requirement: .evaluationOnly)
                        case 1:
                            return full.cbv2RecurrentPrefill(
                                input, inputEmbedding: nil, cache: caches,
                                recurrentState: [binding], positionIds: nil,
                                requirement: .lastPositionLogits)
                        default:
                            return full.cbv2Forward(
                                input, caches: caches, recurrentState: [binding])
                        }
                    })
            }
            return (outputs, state)
        }

        /// The same frames through two stages joined by the residual.
        static func staged(
            _ full: NemotronH35Model, cut: Int
        ) throws -> ([MLXArray], RequestState, RequestState) {
            let first = try stage(of: full, range: 0 ..< cut)
            let second = try stage(of: full, range: cut ..< pattern.count)
            let total = frames.joined().count
            let firstState = try RequestState(first, maximumTokens: total)
            let secondState = try RequestState(second, maximumTokens: total)
            var outputs: [MLXArray] = []
            for (index, frame) in frames.enumerated() {
                let input = rows(frame)
                let residual = try firstState.run { caches, binding in
                    first.cbv2StageResidual(
                        input, inputEmbedding: nil, caches: caches, recurrentState: [binding])
                }
                #expect(residual.shape == [1, frame.count, 64])
                let requirement: CBv2PrefillRequirement? =
                    index == 0 ? .evaluationOnly : index == 1 ? .lastPositionLogits : nil
                outputs.append(
                    try secondState.run { caches, binding in
                        second.cbv2StageLogits(
                            input, inputEmbedding: residual, caches: caches,
                            recurrentState: [binding], requirement: requirement)
                    })
            }
            return (outputs, firstState, secondState)
        }

        static func exact(_ a: MLXArray, _ b: MLXArray) -> Bool {
            a.shape == b.shape && a.dtype == b.dtype
                && SyntheticModel.maxAbsDifference(a, b) == 0
        }

        @Test func twoStagesEqualTheCompleteModelAtEveryFrame() throws {
            let full = try NemotronHRuntimeTests.makeLightning(Self.pattern, seed: 7)
            let (served, servedState) = try Self.serve(full)
            #expect(served.map(\.shape) == [[1, 1], [1, 64], [1, 1, 64], [1, 1, 64]])

            for cut in [4, 6] {
                let (outputs, firstState, secondState) = try Self.staged(full, cut: cut)
                #expect(outputs.count == served.count)
                for (staged, complete) in zip(outputs, served) {
                    #expect(Self.exact(staged, complete))
                }

                // Request state: every Mamba block's convolution history and
                // SSM state and every attention block's keys and values, by
                // the block's index in the complete pattern.
                var attention = 0
                var stageAttention = [0, 0]
                for (index, block) in Self.pattern.enumerated() {
                    let stageIndex = index < cut ? 0 : 1
                    let state = stageIndex == 0 ? firstState : secondState
                    let local = index - (stageIndex == 0 ? 0 : cut)
                    switch block {
                    case "mamba":
                        let complete = try #require(
                            servedState.recurrent.state(modelLayerIndex: index))
                        let part = try #require(state.recurrent.state(modelLayerIndex: local))
                        #expect(Self.exact(try #require(part.conv), try #require(complete.conv)))
                        #expect(Self.exact(try #require(part.ssm), try #require(complete.ssm)))
                    case "attention":
                        let complete = try #require(servedState.rows[attention]).snapshot()
                        let part = try #require(state.rows[stageAttention[stageIndex]])
                            .snapshot()
                        #expect(part.offset == complete.offset)
                        #expect(Self.exact(part.keys, complete.keys))
                        #expect(Self.exact(part.values, complete.values))
                        attention += 1
                        stageAttention[stageIndex] += 1
                    default:
                        #expect(state.recurrent.state(modelLayerIndex: local) == nil)
                    }
                }
            }
        }

        @Test func aStageKeepsOnlyItsOwnBlocksState() throws {
            let full = try NemotronHRuntimeTests.makeLightning(Self.pattern, seed: 7)
            let first = try Self.stage(of: full, range: 0 ..< 4)
            let second = try Self.stage(of: full, range: 4 ..< Self.pattern.count)
            // Indices are the stage's own: the runtime maps them to the
            // complete pattern through its plan.
            #expect(first.cbv2LayerKinds.map(\.modelLayerIndex) == [3])
            #expect(second.cbv2LayerKinds.map(\.modelLayerIndex) == [3])
            #expect(first.cbv2RecurrentStateSpec.layers.map(\.modelLayerIndex) == [0, 2])
            #expect(second.cbv2RecurrentStateSpec.layers.map(\.modelLayerIndex) == [1])
        }

        @Test func aResidualThroughEveryBlockEqualsTheTokenForward() throws {
            // The complete model as its own last stage: its embedding rows in
            // through the residual entry, against the token entry.
            let full = try NemotronHRuntimeTests.makeLightning(Self.pattern, seed: 7)
            let input = Self.rows([3, 1, 4, 1, 5])
            let byTokens = try RequestState(full, maximumTokens: 5).run { caches, binding in
                full.cbv2Forward(input, caches: caches, recurrentState: [binding])
            }
            let embedded = full.mtpEmbedding(input)
            let byResidual = try RequestState(full, maximumTokens: 5).run { caches, binding in
                full.cbv2StageLogits(
                    input, inputEmbedding: embedded, caches: caches,
                    recurrentState: [binding], requirement: nil)
            }
            #expect(Self.exact(byResidual, byTokens))
        }
    }
}
