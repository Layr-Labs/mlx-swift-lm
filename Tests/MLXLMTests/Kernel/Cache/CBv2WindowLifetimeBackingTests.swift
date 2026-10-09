import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {
    @Suite("Window request-lifetime backing hints", .serialized)
    struct CBv2WindowLifetimeBackingTests {
        private func tensor(_ start: Int, _ count: Int, heads: Int = 1, width: Int = 64) -> MLXArray
        {
            sin(
                MLXArray(start * heads * width ..< (start + count) * heads * width).asType(.float32)
                    * 0.17
            )
            .reshaped([1, heads, count, width]).asType(.bfloat16)
        }
        private func equal(_ a: CBv2WindowedSequenceKV, _ b: CBv2WindowedSequenceKV) {
            let x = a.snapshot()
            let y = b.snapshot()
            #expect(x.offset == y.offset)
            #expect(x.keys.asData(access: .copy).data == y.keys.asData(access: .copy).data)
            #expect(x.values.asData(access: .copy).data == y.values.asData(access: .copy).data)
        }
        @Test func boundedHorizonClampsOnlyGeometricSlack() throws {
            for (promptLength, completion) in [(512, 32), (512, 64), (32, 512)] {
                let horizon = promptLength + completion
                let hinted = CBv2WindowedSequenceKV(
                    window: 1_024, kvHeads: 8, headDim: 256,
                    elasticStorage: true, maximumSequenceLength: horizon)
                let parent = CBv2WindowedSequenceKV(
                    window: 1_024, kvHeads: 8, headDim: 256,
                    elasticStorage: true)
                let prompt = tensor(0, promptLength, heads: 8, width: 256)
                _ = hinted.update(keys: prompt, values: prompt)
                _ = parent.update(keys: prompt, values: prompt)
                let tail = tensor(promptLength, completion, heads: 8, width: 256)
                _ = hinted.update(keys: tail, values: tail)
                _ = parent.update(keys: tail, values: tail)
                eval(hinted.cbv2InnerState() + parent.cbv2InnerState())
                equal(hinted, parent)
                #expect(hinted.cbv2InnerState()[0].dim(2) == horizon)
                #expect(parent.cbv2InnerState()[0].dim(2) == 1_024)
                #expect(hinted.byteCount == horizon * 8 * 256 * 2 * 2)
                for array in hinted.cbv2InnerState() {
                    let info = try array.evaluatedBufferInfo()
                    let backing = try #require(info)
                    #expect(backing.allocatedBytes == array.nbytes)
                }
            }
        }
        @Test func backendPassesHorizonWithoutReducingConservativeReservation() throws {
            let kind = CBv2LayerKind(
                attention: .slidingWindow(1_024), headDim: 64, kvHeads: 1, queryHeads: 1)
            let elastic = CBv2ContiguousKVBackend(
                config: .init(
                    bytesCapacity: 1 << 20, kvDType: .bfloat16, elasticWindowStorage: true))
            let fixed = CBv2ContiguousKVBackend(
                config: .init(bytesCapacity: 1 << 20, kvDType: .bfloat16))
            let hintedState = try elastic.makeSequenceState(
                layerKinds: [kind], promptLength: 512, maxLength: 544)
            let fixedState = try fixed.makeSequenceState(
                layerKinds: [kind], promptLength: 512, maxLength: 544)
            defer {
                elastic.release(hintedState)
                fixed.release(fixedState)
            }
            let reservation = 1_024 * 64 * 2 * DType.bfloat16.size
            #expect(elastic.bytesReserved == reservation)
            #expect(fixed.bytesReserved == reservation)
            let data = tensor(0, 544)
            _ = hintedState[0]!.update(keys: data, values: data)
            _ = fixedState[0]!.update(keys: data, values: data)
            #expect(elastic.bytesInUse == 544 * 64 * 2 * DType.bfloat16.size)
            #expect(fixed.bytesInUse == reservation)
            #expect(elastic.bytesReserved == fixed.bytesReserved)
            equal(
                try #require(hintedState[0] as? CBv2WindowedSequenceKV),
                try #require(fixedState[0] as? CBv2WindowedSequenceKV))
        }
        @Test func replayOriginRemapsNonPowerOfTwoCapacityExactly() {
            let origin = 997
            let end = origin + 544
            let hinted = CBv2WindowedSequenceKV(
                window: 1_024, kvHeads: 1, headDim: 64,
                initialOffset: origin, elasticStorage: true, maximumSequenceLength: end)
            let parent = CBv2WindowedSequenceKV(
                window: 1_024, kvHeads: 1, headDim: 64,
                initialOffset: origin, elasticStorage: true)
            var position = origin
            for count in [7, 10, 64, 200, 263] {
                let data = tensor(position, count)
                _ = hinted.update(keys: data, values: data)
                _ = parent.update(keys: data, values: data)
                equal(hinted, parent)
                position += count
            }
            #expect(hinted.cbv2InnerState()[0].dim(2) == 544)
            #expect(hinted.absoluteOffset == end)
        }
        @Test func plannedPrefixReplayKeepsItsFullWindowReservation() throws {
            let full = CBv2LayerKind(
                attention: .full, headDim: 64, kvHeads: 1, queryHeads: 1)
            let window = CBv2LayerKind(
                attention: .slidingWindow(33), headDim: 64, kvHeads: 1, queryHeads: 1)
            // A real replay plan already covers the dependency window. Its
            // remaining request horizon therefore cannot shrink this ring.
            for kinds in [[full, window], [window, full]] {
                let maximumLength = 1_070
                let capability = CBv2PrefixReuseCapability.derive(
                    layerKinds: kinds, backend: .contiguousUnquantized)
                let plan = try #require(
                    capability.plan(
                        matchedBoundary: 1_030, maximumSequenceLength: maximumLength))
                #expect(plan.replayStart == 997)
                let prefix: [(keys: MLXArray, values: MLXArray, offset: Int)?] = kinds.map {
                    kind in
                    guard case .full = kind.attention else { return nil }
                    let data = tensor(0, plan.restoredFullTokens)
                    return (data, data, plan.restoredFullTokens)
                }
                let backend = CBv2ContiguousKVBackend(
                    config: .init(
                        bytesCapacity: 16 << 20, kvDType: .bfloat16, elasticWindowStorage: true))
                let state = try backend.makeSequenceState(
                    adopting: prefix, plan: plan, layerKinds: kinds, maxLength: maximumLength)
                defer { backend.release(state) }
                let windowIndex = try #require(
                    kinds.firstIndex {
                        if case .slidingWindow = $0.attention { return true }
                        return false
                    })
                let row = try #require(state[windowIndex] as? CBv2WindowedSequenceKV)
                let reference = CBv2WindowedSequenceKV(
                    window: 33, kvHeads: 1, headDim: 64,
                    initialOffset: plan.replayStart, elasticStorage: true)
                #expect(row.absoluteOffset == plan.replayStart)
                let reserved = (maximumLength + 33) * 64 * 2 * DType.bfloat16.size
                #expect(backend.bytesReserved == reserved)
                var position = plan.replayStart
                for count in [7, 10, 16, 40] {
                    let data = tensor(position, count)
                    _ = row.update(keys: data, values: data)
                    _ = reference.update(keys: data, values: data)
                    equal(row, reference)
                    position += count
                }
                #expect(row.absoluteOffset == maximumLength)
                #expect(row.cbv2InnerState()[0].dim(2) == 33)
                #expect(backend.bytesReserved == reserved)
                backend.release(state)
                #expect(backend.bytesReserved == 0)
            }
        }
        @Test func underestimatedHintPreservesValidWritesAndWindowRetention() {
            let hinted = CBv2WindowedSequenceKV(
                window: 33, kvHeads: 1, headDim: 64,
                elasticStorage: true, maximumSequenceLength: 17)
            let parent = CBv2WindowedSequenceKV(
                window: 33, kvHeads: 1, headDim: 64,
                elasticStorage: true)
            var position = 0
            for count in [17, 1, 20, 40] {
                let data = tensor(position, count)
                _ = hinted.update(keys: data, values: data)
                _ = parent.update(keys: data, values: data)
                equal(hinted, parent)
                position += count
            }
            #expect(hinted.cbv2InnerState()[0].dim(2) == 33)
            #expect(hinted.retainedCount == 33)
        }
        @Test func pendingOverspeculationDoesNotGrowUntilConfirmed() {
            let hinted = CBv2WindowedSequenceKV(
                window: 1_024, kvHeads: 1, headDim: 64,
                elasticStorage: true, maximumSequenceLength: 40)
            let parent = CBv2WindowedSequenceKV(
                window: 1_024, kvHeads: 1, headDim: 64,
                elasticStorage: true)
            let base = tensor(0, 32)
            _ = hinted.update(keys: base, values: base)
            _ = parent.update(keys: base, values: base)
            hinted.beginSpeculativeWrite()
            let draft = tensor(32, 17)
            _ = hinted.update(keys: draft, values: draft)
            #expect(hinted.cbv2InnerState()[0].dim(2) == 32)
            hinted.rollback(9)
            hinted.commitSpeculativeWrite()
            let accepted = tensor(32, 8)
            _ = parent.update(keys: accepted, values: accepted)
            equal(hinted, parent)
            #expect(hinted.cbv2InnerState()[0].dim(2) == 40)
            let before = hinted.byteCount
            hinted.beginSpeculativeWrite()
            _ = hinted.update(keys: tensor(40, 20), values: tensor(40, 20))
            hinted.rollback(20)
            hinted.commitSpeculativeWrite()
            #expect(hinted.byteCount == before)
            equal(hinted, parent)
            hinted.beginSpeculativeWrite()
            let overrun = tensor(40, 20)
            _ = hinted.update(keys: overrun, values: overrun)
            hinted.commitSpeculativeWrite()
            _ = parent.update(keys: overrun, values: overrun)
            equal(hinted, parent)
            #expect(hinted.cbv2InnerState()[0].dim(2) >= hinted.retainedCount)
            #expect(hinted.cbv2InnerState()[0].dim(2) <= hinted.window)
        }
        @Test func nonElasticAndUnusableHintsPreserveEstablishedAllocation() {
            let fixed = CBv2WindowedSequenceKV(
                window: 33, kvHeads: 1, headDim: 64,
                maximumSequenceLength: 4)
            let invalidHint = CBv2WindowedSequenceKV(
                window: 33, kvHeads: 1, headDim: 64,
                initialOffset: 99, elasticStorage: true, maximumSequenceLength: 4)
            let parent = CBv2WindowedSequenceKV(
                window: 33, kvHeads: 1, headDim: 64,
                initialOffset: 99, elasticStorage: true)
            let data = tensor(0, 20)
            _ = fixed.update(keys: data, values: data)
            _ = invalidHint.update(keys: data, values: data)
            _ = parent.update(keys: data, values: data)
            #expect(fixed.cbv2InnerState()[0].dim(2) == 33)
            #expect(invalidHint.byteCount == parent.byteCount)
            equal(invalidHint, parent)
        }
    }
}
