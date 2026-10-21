import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextPagedSparseStateTests {
        private final class Identity {}
        private struct Fixture {
            let identity = Identity()
            let admission: AdmissionV2
            let store: GLM5NextPagedSparseStore
            var binding: GLM5NextSparseStateBinding {
                .init(
                    modelIdentity: ObjectIdentifier(identity), cacheSalt: "tenant",
                    modelLayerIndex: 3)
            }
            init() throws {
                admission = AdmissionV2(
                    layerKinds: [], bytesCapacity: 32 << 20,
                    config: .init(watermarkFraction: 0))
                let geometry = try GLM5NextSparseStateGeometry(
                    maximumTokens: 64, maximumStagedTokens: 64,
                    indexDimension: 2, latentDimension: 4,
                    indexDType: .float32, latentDType: .float32)
                store = try GLM5NextPagedSparseStore(
                    geometry: geometry, admission: admission, capacityBytes: 8 << 20,
                    segmentBytes: 64 << 10, maximumBufferBytes: 1 << 20)
            }
        }

        private func values(_ range: Range<Int>, width: Int, marker: Float = 0) -> [Float] {
            range.flatMap { position in (0 ..< width).map { marker + Float(position * 10 + $0) } }
        }

        private func append(
            _ state: GLM5NextPagedSparseState, start: Int, count: Int,
            marker: Float = 0, accepting: Int? = nil
        ) throws {
            let range = start ..< (start + count)
            let transaction = try state.begin(at: start)
            try state.stage(
                transaction, at: start,
                latents: MLXArray(values(range, width: 4, marker: marker)).reshaped(1, count, 4),
                indexKeys: MLXArray(values(range, width: 2, marker: marker)).reshaped(1, count, 2),
                poolGates: MLXArray(values(range, width: 2, marker: marker + 100)).reshaped(
                    1, count, 2),
                validity: MLXArray(range.map { $0 % 3 != 1 }).reshaped(1, count))
            try state.commit(transaction, accepting: accepting ?? count)
        }

        private func expectPrefix(_ state: GLM5NextPagedSparseState, count: Int) throws {
            let view = try state.indexView()
            #expect(view.tokenCount == count)
            #expect(view.indexKeys.asArray(Float.self) == values(0 ..< count, width: 2))
            #expect(
                view.poolGates.asArray(Float.self) == values(0 ..< count, width: 2, marker: 100))
            #expect(view.positions.asArray(Int32.self) == (0 ..< count).map(Int32.init))
            #expect(view.validity.asArray(Bool.self) == (0 ..< count).map { $0 % 3 != 1 })
            let selected = try state.gatherLatents(MLXArray((0 ..< count).map(Int32.init)))
            #expect(selected.latents.asArray(Float.self) == values(0 ..< count, width: 4))
        }

        @Test(arguments: [3, 4, 5, 15, 16, 17, 31, 32, 33])
        func snapshotsSurviveSourceTrimRegrowAndForkWrites(boundary: Int) throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try Fixture()
                let source = try fixture.store.makeState(binding: fixture.binding)
                try append(source, start: 0, count: boundary)
                #expect(fixture.store.bytesMaterialized > 0)
                #expect(fixture.store.bytesInUse > 0)
                var checkpoint: GLM5NextPagedSparseState.Checkpoint? = try source.checkpoint()
                let fork = try source.fork()
                try source.trim(to: boundary - 1)
                try append(source, start: boundary - 1, count: 2, marker: 1000)
                try expectPrefix(fork, count: boundary)

                // Evict the explicit cache handle. The two live rows still
                // share older pages, so COW must also work without that handle.
                checkpoint = nil
                #expect(checkpoint == nil)
                #expect(fixture.store.pool.immutableSnapshotReservedPages == 0)
                try fork.trim(to: boundary - 1)
                try append(fork, start: boundary - 1, count: 1, marker: 2000)
                let sourceTail = try source.gatherLatents(MLXArray([Int32(boundary - 1)]))
                #expect(
                    sourceTail.latents.asArray(Float.self)
                        == values((boundary - 1) ..< boundary, width: 4, marker: 1000))
                let forkTail = try fork.gatherLatents(MLXArray([Int32(boundary - 1)]))
                #expect(
                    forkTail.latents.asArray(Float.self)
                        == values((boundary - 1) ..< boundary, width: 4, marker: 2000))
                try source.close()
                try fork.close()
            }
        }

        @Test(arguments: [0, 2, 4])
        func speculativeSuffixAcceptsExactlyTheChosenPrefix(accepted: Int) throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try Fixture()
                let state = try fixture.store.makeState(binding: fixture.binding)
                try append(state, start: 0, count: 3)
                try append(state, start: 3, count: 4, accepting: accepted)
                #expect(state.committedTokenCount == 3 + accepted)
                #expect(state.stagedTokenCount == 0)
                try expectPrefix(state, count: 3 + accepted)
                try state.close()
            }
        }

        @Test func aCommittedCheckpointNeverExposesAnInFlightSuffix() throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try Fixture()
                let state = try fixture.store.makeState(binding: fixture.binding)
                try append(state, start: 0, count: 5)
                let transaction = try state.begin(at: 5)
                try state.stage(
                    transaction, at: 5,
                    latents: .ones([1, 3, 4]), indexKeys: .ones([1, 3, 2]),
                    poolGates: .ones([1, 3, 2]), validity: .ones([1, 3], dtype: .bool))
                let checkpoint = try state.checkpoint()
                #expect(checkpoint.tokenCount == 5)
                let restored = try fixture.store.makeState(binding: state.binding)
                try restored.restore(checkpoint)
                try expectPrefix(restored, count: 5)
                try state.rollback(transaction)
                try expectPrefix(state, count: 5)
                try state.close()
                try restored.close()
            }
        }

        @Test(arguments: [3, 4, 5, 15, 16, 17])
        func rollbackAcrossPoolAndPageBoundariesKeepsTheAcceptedPrefix(boundary: Int) throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try Fixture()
                let state = try fixture.store.makeState(binding: fixture.binding)
                try append(state, start: 0, count: boundary)
                let checkpoint = try state.checkpoint()
                try append(state, start: boundary, count: 4, marker: 9000, accepting: 0)
                try expectPrefix(state, count: boundary)
                try append(state, start: boundary, count: 4)
                try expectPrefix(state, count: boundary + 4)
                try state.restore(checkpoint)
                try expectPrefix(state, count: boundary)
                try state.close()
            }
        }

        @Test func evaluatedReadersSurviveTeardownAndPhysicalSlotReuse() throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try Fixture()
                let state = try fixture.store.makeState(binding: fixture.binding)
                try append(state, start: 0, count: 17)
                let reader = try state.gatherLatents(MLXArray([Int32(16), 0, -1, 40]))
                let expected =
                    values(16 ..< 17, width: 4) + values(0 ..< 1, width: 4)
                    + Array(repeating: Float(0), count: 8)
                try state.close()
                let replacement = try fixture.store.makeState(binding: fixture.binding)
                try append(replacement, start: 0, count: 33, marker: 9000)
                #expect(reader.latents.asArray(Float.self) == expected)
                try replacement.close()
                #expect(fixture.store.bytesInUse == 0)
                #expect(fixture.store.bytesReserved == 0)
            }
        }

        @Test func pageHandleGenerationsAndPartialPageCOWRemainIndependent() throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try Fixture()
                let rows = try fixture.store.makeRows()
                let source = rows[0]
                source.write(
                    keys: MLXArray(values(0 ..< 17, width: 4)).reshaped(1, 17, 4),
                    values: MLXArray(values(0 ..< 17, width: 4)).reshaped(1, 17, 4))
                try fixture.store.finish()
                var snapshot: PagedImmutableSnapshot? = try source.immutableSnapshot(through: 17)
                let handles = try #require(snapshot).handles
                source.rollback(2)
                source.write(
                    keys: MLXArray(values(15 ..< 18, width: 4, marker: 1000)).reshaped(1, 3, 4),
                    values: MLXArray(values(15 ..< 18, width: 4, marker: 1000)).reshaped(1, 3, 4))
                try fixture.store.finish()
                #expect(handles.allSatisfy { fixture.store.pool.isValid($0) })
                let original = fixture.store.pool.gather(
                    group: source.groupKey, pages: handles.map(\.page), firstSlot: 0, count: 17)
                try fixture.store.finish([original.keys, original.values])
                #expect(original.keys.asArray(Float.self) == values(0 ..< 17, width: 4))
                #expect(source.table[0] != handles[0].page)
                snapshot = nil
                rows.forEach { $0.releaseStorage() }
                #expect(fixture.store.pool.immutableSnapshotReservedPages == 0)
                #expect(fixture.store.bytesReserved == 0)
            }
        }

        @Test func foreignCheckpointRefusesBeforeMutation() throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try Fixture()
                let state = try fixture.store.makeState(binding: fixture.binding)
                try append(state, start: 0, count: 3)
                let checkpoint = try state.checkpoint()
                let other = try fixture.store.makeState(binding: fixture.binding)
                #expect(throws: GLM5NextSparseStateError.incompatibleCheckpoint) {
                    try other.restore(checkpoint)
                }
                #expect(other.committedTokenCount == 0)
                try other.close()
                try state.close()
            }
        }

        @Test func refusedSnapshotKeepsTheSourceAndExistingPageClaims() throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try Fixture()
                let state = try fixture.store.makeState(binding: fixture.binding)
                try append(state, start: 0, count: 5)
                let pageClaims = fixture.store.bytesReserved
                fixture.admission.updateBytesCapacity(1)
                #expect(throws: (any Error).self) { try state.checkpoint() }
                #expect(state.committedTokenCount == 5)
                #expect(fixture.store.bytesReserved == pageClaims)
                fixture.admission.updateBytesCapacity(32 << 20)
                try expectPrefix(state, count: 5)
                try state.close()
            }
        }

        @Test func longNarrowIndexReadPricesTokenRecordsBeforeReservation() throws {
            // Size-only calculation: no long history, native pool, or device
            // tensor is allocated to reproduce the reviewed undercharge.
            let count = 262_144
            let plan = try GLM5NextPagedIndexReadFootprint.plan(
                tokens: count, indexDimension: 1, dtype: .float16,
                pages: count / 16, tokensPerSegment: [count], bound: { $0 })
            #expect(plan.transferRecordBytes == count * 3 * MemoryLayout<Int32>.stride)
            #expect(plan.destinationBytes == count * (4 * DType.float16.size + 1 + 4))
            #expect(plan.hostBytes >= count * (3 * 4 + 4))
            #expect(
                plan.totalBytes == plan.destinationBytes + plan.transferRecordBytes
                    + plan.transferSupportBytes + plan.hostBytes)
            let short = AdmissionV2(
                layerKinds: [], bytesCapacity: plan.totalBytes - 1,
                config: .init(watermarkFraction: 0))
            #expect(throws: (any Error).self) { try short.reserveTransient(bytes: plan.totalBytes) }
            #expect(short.bytesReserved == 0)
            let enough = AdmissionV2(
                layerKinds: [], bytesCapacity: plan.totalBytes,
                config: .init(watermarkFraction: 0))
            let reservation = try enough.reserveTransient(bytes: plan.totalBytes)
            #expect(enough.bytesReserved == plan.totalBytes)
            reservation.release()
            #expect(enough.bytesReserved == 0)
        }

        @Test func indexReadPricesEachSegmentsPaddedRecordsAndFence() throws {
            let plan = try GLM5NextPagedIndexReadFootprint.plan(
                tokens: 5, indexDimension: 2, dtype: .float32, pages: 2,
                tokensPerSegment: [3, 2], bound: { $0 })
            #expect(plan.transferRecordBytes == 2 * 24 * MemoryLayout<Int32>.stride)
            // Selector, one fence + five shape ints per transfer, completion.
            #expect(plan.transferSupportBytes == 1 + 2 * (4 + 5 * 4) + 4)
            #expect(throws: GLM5NextSparseStateError.invalidGeometry) {
                try GLM5NextPagedIndexReadFootprint.plan(
                    tokens: 5, indexDimension: 2, dtype: .float32, pages: 2,
                    tokensPerSegment: [2, 2], bound: { $0 })
            }
        }

        @Test func checkedBackendConstructionRejectsRebindingBeforeCallback() throws {
            let admission = AdmissionV2(layerKinds: [], bytesCapacity: 1 << 20)
            let first = try admission.constructWithBackendPhysicalFloor { 1 }
            var invoked = false
            #expect(throws: CBv2BackendConstructionError.physicalFloorAlreadyBound) {
                try admission.constructWithBackendPhysicalFloor {
                    invoked = true
                    return 2
                }
            }
            #expect(!invoked && first.value == 1)
            first.lease.close()
        }

        @Test func checkedBackendConstructionRejectsBusyLedgerAndRollsBackThrownConstruction()
            throws
        {
            let admission = AdmissionV2(layerKinds: [], bytesCapacity: 1 << 20)
            let request = try admission.reserveTransient(bytes: 1)
            var invoked = false
            #expect(throws: CBv2BackendConstructionError.liveReservations) {
                try admission.constructWithBackendPhysicalFloor { invoked = true }
            }
            #expect(!invoked && admission.bytesReserved == 1)
            request.release()
            enum Refused: Error { case construction }
            #expect(throws: Refused.self) {
                try admission.constructWithBackendPhysicalFloor { () throws -> Void in
                    throw Refused.construction
                }
            }
            let successful = try admission.constructWithBackendPhysicalFloor { 7 }
            #expect(successful.value == 7 && admission.bytesReserved == 0)
            successful.lease.close()
        }

        @Test func secondGLMStoreFailsWithoutBreakingTheExistingStore() throws {
            try Device.withDefaultDevice(.gpu) {
                let fixture = try Fixture()
                #expect(throws: CBv2BackendConstructionError.physicalFloorAlreadyBound) {
                    try GLM5NextPagedSparseStore(
                        geometry: fixture.store.geometry, admission: fixture.admission,
                        capacityBytes: 8 << 20, segmentBytes: 64 << 10, maximumBufferBytes: 1 << 20)
                }
                let state = try fixture.store.makeState(binding: fixture.binding)
                try append(state, start: 0, count: 5)
                try expectPrefix(state, count: 5)
                try state.close()
            }
        }
    }
}
