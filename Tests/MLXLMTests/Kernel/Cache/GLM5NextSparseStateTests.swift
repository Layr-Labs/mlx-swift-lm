import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextSparseStateTests {
        private final class ModelIdentity {}

        private struct Fixture {
            let model = ModelIdentity()
            let admission: AdmissionV2
            let geometry: GLM5NextSparseStateGeometry
            init(capacity: Int = 16 << 20) throws {
                admission = AdmissionV2(layerKinds: [], bytesCapacity: capacity,
                                        config: .init(watermarkFraction: 0))
                geometry = try GLM5NextSparseStateGeometry(
                    maximumTokens: 32, maximumStagedTokens: 8,
                    indexDimension: 2, latentDimension: 3,
                    indexDType: .float32, latentDType: .float32)
            }
            func state(salt: String = "tenant-a", layer: Int = 3) -> GLM5NextSparseState {
                GLM5NextSparseState(
                    geometry: geometry,
                    binding: .init(modelIdentity: ObjectIdentifier(model), cacheSalt: salt, modelLayerIndex: layer),
                    admission: admission)
            }
        }

        private func stage(
            _ state: GLM5NextSparseState, _ transaction: GLM5NextSparseState.Transaction,
            start: Int, count: Int, marker: Float = 0
        ) throws {
            func rows(width: Int, scale: Int) -> MLXArray {
                let values = (start..<(start + count)).flatMap { position in
                    (0..<width).map { marker + Float(position * scale + $0) }
                }
                return MLXArray(values).reshaped(1, count, width)
            }
            try state.stage(
                transaction, at: start, latents: rows(width: 3, scale: 10),
                indexKeys: rows(width: 2, scale: 20), poolGates: rows(width: 2, scale: 30),
                validity: MLXArray((start..<(start + count)).map { $0 % 3 != 1 }).reshaped(1, count))
        }

        private func expect(
            _ view: GLM5NextSparseState.View, positions: Range<Int>, marker: Float = 0
        ) {
            eval(view.evaluationTargets)
            #expect(view.tokenCount == positions.count)
            #expect(view.positions.dtype == .int32)
            if positions.isEmpty {
                // MLX's host-copy helper derives a nonzero physical span from
                // [1, 0, D]'s trailing dimension even though no elements exist.
                // Check the complete empty schema without reading a nil buffer.
                #expect(view.latents.shape == [1, 0, 3] && view.latents.dtype == .float32)
                #expect(view.indexKeys.shape == [1, 0, 2] && view.indexKeys.dtype == .float32)
                #expect(view.poolGates.shape == [1, 0, 2] && view.poolGates.dtype == .float32)
                #expect(view.validity.shape == [1, 0] && view.validity.dtype == .bool)
                #expect(view.positions.shape == [1, 0])
                #expect(view.evaluationTargets.allSatisfy { $0.size == 0 && $0.nbytes == 0 })
                return
            }
            #expect(view.positions.asArray(Int32.self) == positions.map(Int32.init))
            #expect(view.validity.asArray(Bool.self) == positions.map { $0 % 3 != 1 })
            #expect(view.latents.asArray(Float.self) == positions.flatMap { position in
                (0..<3).map { marker + Float(position * 10 + $0) }
            })
            #expect(view.indexKeys.asArray(Float.self) == positions.flatMap { position in
                (0..<2).map { marker + Float(position * 20 + $0) }
            })
            #expect(view.poolGates.asArray(Float.self) == positions.flatMap { position in
                (0..<2).map { marker + Float(position * 30 + $0) }
            })
        }

        @Test(arguments: [0, 2, 5])
        func acceptsExactlyZeroPartialOrAll(accepted: Int) throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try Fixture()
                let state = fixture.state()
                let transaction = try state.begin(at: 0)
                try stage(state, transaction, start: 0, count: 5)
                let speculative = try state.view(transaction)
                expect(speculative, positions: 0..<5)
                expect(try state.view(), positions: 0..<0)
                #expect(try state.checkpoint().tokenCount == 0)
                let charged = state.retainedAllocationBytes
                try state.commit(transaction, accepting: accepted)
                #expect(state.stagedTokenCount == 0)
                expect(try state.view(), positions: 0..<accepted)
                #expect(state.retainedAllocationBytes == (accepted == 0 ? 0 : charged))
                // Already submitted readers keep the full immutable suffix and its lease.
                expect(speculative, positions: 0..<5)
                #expect(throws: GLM5NextSparseStateError.invalidTransaction) {
                    try state.commit(transaction, accepting: 0)
                }
            }
        }

        @Test func partialAcceptanceSpansMultipleStagedChunks() throws {
            try Device.withDefaultDevice(.cpu) {
                let state = try Fixture().state()
                let transaction = try state.begin(at: 0)
                try stage(state, transaction, start: 0, count: 3)
                try stage(state, transaction, start: 3, count: 2)
                expect(try state.view(transaction), positions: 0..<5)
                try state.commit(transaction, accepting: 4)
                expect(try state.view(), positions: 0..<4)
            }
        }

        @Test(arguments: [3, 4, 5])
        func trimsRawStateAtPoolBoundariesAndForksIndependently(boundary: Int) throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try Fixture()
                let original = fixture.state()
                let fill = try original.begin(at: 0)
                try stage(original, fill, start: 0, count: 6)
                try original.commit(fill, accepting: 6)
                let checkpoint = try original.checkpoint()
                let fork = try original.fork()
                try fork.trim(to: boundary)
                expect(try fork.view(), positions: 0..<boundary)
                let append = try fork.begin(at: boundary)
                try stage(fork, append, start: boundary, count: 1, marker: 1000)
                try fork.commit(append, accepting: 1)
                let forkView = try fork.view()
                eval(forkView.evaluationTargets)
                #expect(forkView.positions.asArray(Int32.self) == (0...boundary).map(Int32.init))
                #expect(Array(forkView.indexKeys.asArray(Float.self).suffix(2)) == [1000 + Float(boundary * 20), 1001 + Float(boundary * 20)])
                expect(try original.view(), positions: 0..<6)
                try fork.restore(checkpoint)
                expect(try fork.view(), positions: 0..<6)
            }
        }

        @Test func checkpointAndForkExcludeUnconfirmedSuffix() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try Fixture()
                let state = fixture.state()
                let first = try state.begin(at: 0)
                try stage(state, first, start: 0, count: 3)
                try state.commit(first, accepting: 3)
                let suffix = try state.begin(at: 3)
                try stage(state, suffix, start: 3, count: 2, marker: 1000)
                let snapshot = try state.checkpoint()
                #expect(snapshot.tokenCount == 3)
                let fork = try state.fork()
                expect(try fork.view(), positions: 0..<3)
                try state.rollback(suffix)
                expect(try state.view(), positions: 0..<3)
                let append = try fork.begin(at: 3)
                try stage(fork, append, start: 3, count: 1)
                try fork.commit(append, accepting: 1)
                expect(try fork.view(), positions: 0..<4)
                expect(try state.view(), positions: 0..<3)
            }
        }

        @Test func cancellationRetainsReadersUntilTheyRetire() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try Fixture()
                let state = fixture.state()
                let transaction = try state.begin(at: 0)
                try stage(state, transaction, start: 0, count: 5)
                var reader: GLM5NextSparseState.View? = try state.view(transaction)
                let charge = fixture.admission.bytesReserved
                #expect(charge > 0)
                state.close()
                state.close()
                #expect(fixture.admission.bytesReserved == charge)
                expect(try #require(reader), positions: 0..<5)
                reader = nil
                #expect(fixture.admission.bytesReserved == 0)
                #expect(throws: GLM5NextSparseStateError.closed) { try state.view() }
            }
        }

        @Test func rejectedPayloadAndCapacityDoNotAdvanceState() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try Fixture()
                let state = fixture.state()
                #expect(throws: GLM5NextSparseStateError.invalidPosition) { try state.begin(at: 1) }
                let transaction = try state.begin(at: 0)
                #expect(throws: GLM5NextSparseStateError.invalidPosition) {
                    try stage(state, transaction, start: 1, count: 1)
                }
                #expect(throws: GLM5NextSparseStateError.invalidPayload("index keys")) {
                    try state.stage(transaction, at: 0,
                        latents: .zeros([1, 1, 3]), indexKeys: .zeros([1, 1, 2], dtype: .float16),
                        poolGates: .zeros([1, 1, 2]), validity: .zeros([1, 1], dtype: .bool))
                }
                #expect(throws: GLM5NextSparseStateError.invalidPayload("validity")) {
                    try state.stage(transaction, at: 0,
                        latents: .zeros([1, 1, 3]), indexKeys: .zeros([1, 1, 2]),
                        poolGates: .zeros([1, 1, 2]), validity: .zeros([1, 1], dtype: .int32))
                }
                #expect(state.stagedTokenCount == 0)
                #expect(fixture.admission.bytesReserved == 0)
                let starved = try Fixture(capacity: 1)
                let denied = starved.state()
                let deniedTransaction = try denied.begin(at: 0)
                #expect(throws: (any Error).self) {
                    try stage(denied, deniedTransaction, start: 0, count: 1)
                }
                #expect(denied.stagedTokenCount == 0)
                #expect(starved.admission.bytesReserved == 0)
            }
        }

        @Test func transactionsAreOwnerBoundAndCheckpointsAreTenantBound() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try Fixture()
                let state = fixture.state()
                let transaction = try state.begin(at: 0)
                let foreign = fixture.state()
                #expect(throws: GLM5NextSparseStateError.invalidTransaction) {
                    try stage(foreign, transaction, start: 0, count: 1)
                }
                try stage(state, transaction, start: 0, count: 3)
                try state.commit(transaction, accepting: 3)
                let checkpoint = try state.checkpoint()
                let adopter = GLM5NextSparseState(
                    geometry: fixture.geometry, binding: state.binding, admission: fixture.admission)
                try adopter.restore(checkpoint)
                expect(try adopter.view(), positions: 0..<3)
                for other in [fixture.state(salt: "tenant-b"), fixture.state(layer: 7), try Fixture().state()] {
                    #expect(throws: GLM5NextSparseStateError.incompatibleCheckpoint) {
                        try other.restore(checkpoint)
                    }
                }
                let otherPrecision = try GLM5NextSparseStateGeometry(
                    maximumTokens: 32, maximumStagedTokens: 8, indexDimension: 2,
                    latentDimension: 3, indexDType: .bfloat16, latentDType: .float32)
                let incompatible = GLM5NextSparseState(
                    geometry: otherPrecision, binding: state.binding, admission: fixture.admission)
                #expect(throws: GLM5NextSparseStateError.incompatibleCheckpoint) {
                    try incompatible.restore(checkpoint)
                }
            }
        }

        @Test func geometryPricesAllPayloadsAndBoundsTheSuffix() throws {
            try Device.withDefaultDevice(.cpu) {
                let fixture = try Fixture()
                #expect(try fixture.geometry.logicalBytes(tokenCount: 5) == 165)
                #expect(try fixture.geometry.allocationBytes(tokenCount: 5) >= 165)
                #expect(throws: GLM5NextSparseStateError.invalidPosition) {
                    try fixture.geometry.logicalBytes(tokenCount: 33)
                }
                #expect(throws: GLM5NextSparseStateError.invalidGeometry) {
                    try GLM5NextSparseStateGeometry(maximumTokens: 32, maximumStagedTokens: 8,
                        indexDimension: 2, latentDimension: 3, indexDType: .float32, latentDType: .float32, poolSize: 3)
                }
                let state = fixture.state()
                let transaction = try state.begin(at: 0)
                try stage(state, transaction, start: 0, count: 8)
                #expect(throws: GLM5NextSparseStateError.invalidPosition) {
                    try stage(state, transaction, start: 8, count: 1)
                }
            }
        }

        @Test func existingRecurrentTransactionsComposeAtTheCallersFence() throws {
            try Device.withDefaultDevice(.cpu) {
                let sparse = try Fixture().state()
                let recurrence = try CBv2RecurrentRequestState(spec: .init(layers: [
                    .init(modelLayerIndex: 0, convShape: [1, 3, 2], convDType: .float32,
                          ssmShape: [1, 1, 2, 2], ssmDType: .float32),
                ]))
                let sparseTransaction = try sparse.begin(at: 0)
                try stage(sparse, sparseTransaction, start: 0, count: 3)
                let recurrentTransaction = try recurrence.bind()
                try recurrentTransaction.stage(modelLayerIndex: 0,
                    conv: .ones([1, 3, 2]), ssm: .ones([1, 1, 2, 2]))
                let roots = try recurrentTransaction.evaluate()
                let view = try sparse.view(sparseTransaction)
                eval(roots + view.evaluationTargets)
                try recurrentTransaction.commit()
                try sparse.commit(sparseTransaction, accepting: 3)
                #expect(recurrence.confirmedStateSnapshot()?[0]?.ssm?.shape == [1, 1, 2, 2])
                expect(try sparse.view(), positions: 0..<3)
                try recurrence.release()
            }
        }
    }
}
