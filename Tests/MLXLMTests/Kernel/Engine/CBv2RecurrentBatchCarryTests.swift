import Foundation
import MLX
import Testing

@testable import MLXLMCommon

extension KernelTests {

    /// The recurrent batch carry (`CBv2RecurrentBatchCarry`) and the rule that
    /// a request paused by backpressure owns its recurrent rows
    /// (`CBv2RecurrentRowOwnership`).
    @Suite(.serialized)
    struct CBv2RecurrentBatchCarryTests {
        /// `[1, 2, 256, 256]` float32: 512 KiB per row, large enough that a
        /// held batch is visible in MLX's active memory.
        private static let side = 256
        private static let rowBytes = 2 * side * side * 4

        private var spec: CBv2RecurrentStateSpec {
            CBv2RecurrentStateSpec(layers: [
                CBv2RecurrentLayerStateSpec(
                    modelLayerIndex: 0,
                    convShape: [1, 3, 8], convDType: .float16,
                    ssmShape: [1, 2, Self.side, Self.side], ssmDType: .float32)
            ])
        }

        private func makeStates(_ count: Int) throws -> [CBv2RecurrentRequestState] {
            try (0 ..< count).map { _ in try CBv2RecurrentRequestState(spec: spec) }
        }

        /// One batched step. Every request stages row `i` of one batched tensor
        /// pair, as a recurrent layer does, and commits. Row `i` holds
        /// `seed + i` in every element.
        @discardableResult
        private func step(
            _ states: [CBv2RecurrentRequestState], seed: Float
        ) throws -> CBv2RecurrentBatchOrigin {
            let rows = states.count
            // Built from host data, so the batch occupies `rows` whole rows. A
            // broadcast plus a scalar keeps the broadcast's compact storage in
            // MLX and would make a batch of a few bytes.
            let ssmRow = 2 * Self.side * Self.side
            let conv = MLXArray(
                (0 ..< rows).flatMap { [Float](repeating: Float($0) + seed, count: 3 * 8) },
                [rows, 3, 8]
            ).asType(.float16)
            let ssm = MLXArray(
                (0 ..< rows).flatMap { [Float](repeating: Float($0) + seed, count: ssmRow) },
                [rows, 2, Self.side, Self.side])
            eval(conv, ssm)
            let origin = CBv2RecurrentBatchOrigin(conv: conv, ssm: ssm)
            let evaluations = try states.enumerated().map { row, state in
                let evaluation = try state.bind()
                try evaluation.stage(
                    modelLayerIndex: 0, state: CBv2RecurrentLayerState(origin: origin, row: row))
                return evaluation
            }
            for evaluation in evaluations { _ = try evaluation.evaluate() }
            for evaluation in evaluations { try evaluation.commit() }
            return origin
        }

        private func layerStates(
            _ states: [CBv2RecurrentRequestState]
        ) -> [CBv2RecurrentLayerState?] {
            states.map { $0.state(modelLayerIndex: 0) }
        }

        /// The row's bits, read without keeping any array of the state alive.
        private func bits(_ state: CBv2RecurrentRequestState) -> (conv: [Float], ssm: [UInt32]) {
            let layer = state.state(modelLayerIndex: 0)!
            let conv = layer.conv!.asType(.float32).asArray(Float.self)
            let ssm = layer.ssm!.asArray(Float.self).map(\.bitPattern)
            return (conv, ssm)
        }

        @Test(.enabled(if: CBv2RecurrentBatchCarry.enabled, "the carry is switched off"))
        func carryTakesTheBatchOnlyForTheSameRowsInTheSameOrder() throws {
            let states = try makeStates(3)
            let origin = try step(states, seed: 10)

            #expect(CBv2RecurrentBatchCarry.carried(layerStates(states)) === origin)
            // Another order, a row that left, a row that joined: join as before.
            #expect(
                CBv2RecurrentBatchCarry.carried(layerStates([states[1], states[0], states[2]]))
                    == nil)
            #expect(CBv2RecurrentBatchCarry.carried(layerStates([states[0], states[1]])) == nil)
            let joiner = try makeStates(1)
            try step(joiner, seed: 50)
            #expect(CBv2RecurrentBatchCarry.carried(layerStates(states + joiner)) == nil)
            // A state staged from plain tensors has no origin.
            let plain = try makeStates(2)
            for (row, state) in plain.enumerated() {
                let evaluation = try state.bind()
                try evaluation.stage(
                    modelLayerIndex: 0,
                    conv: origin.conv[row ..< row + 1], ssm: origin.ssm[row ..< row + 1])
                _ = try evaluation.evaluate()
                try evaluation.commit()
            }
            #expect(CBv2RecurrentBatchCarry.carried(layerStates(plain)) == nil)

            // The next step of the same rows carries again, on the new batch.
            let next = try step(states, seed: 20)
            #expect(CBv2RecurrentBatchCarry.carried(layerStates(states)) === next)
            #expect(next !== origin)
            for state in states + joiner + plain { try state.release() }
        }

        @Test(.enabled(if: CBv2RecurrentBatchCarry.enabled, "the carry is switched off"))
        func carryCountersCountCarriedAndJoinedCalls() throws {
            let states = try makeStates(2)
            try step(states, seed: 1)
            _ = CBv2RecurrentBatchCarry.snapshotCounts(reset: true)
            _ = CBv2RecurrentBatchCarry.carried(layerStates(states))
            _ = CBv2RecurrentBatchCarry.carried(layerStates(states.reversed()))
            let counts = CBv2RecurrentBatchCarry.snapshotCounts(reset: true)
            #expect(counts.carried == 1)
            #expect(counts.joined == 1)
            for state in states { try state.release() }
        }

        @Test(
            .enabled(
                if: CBv2RecurrentRowOwnership.mode == .onPause,
                "the re-own rule is not in its default mode"))
        func pausedRequestOwnsItsRowWithTheSameBits() throws {
            let states = try makeStates(4)
            try step(states, seed: 100)
            let paused = states[2]
            #expect(paused.state(modelLayerIndex: 0)?.batchOrigin != nil)
            let before = bits(paused)
            #expect(before.conv.first == 102)

            _ = CBv2RecurrentRowOwnership.snapshotCount(reset: true)
            _ = CBv2RecurrentRowOwnership.snapshotPauses(reset: true)
            paused.setPaused(true)
            #expect(CBv2RecurrentRowOwnership.snapshotPauses() == 1)
            #expect(CBv2RecurrentRowOwnership.snapshotCount() == 1)

            let owned = try #require(paused.state(modelLayerIndex: 0))
            #expect(owned.batchOrigin == nil)
            #expect(owned.conv?.shape == [1, 3, 8])
            #expect(owned.conv?.dtype == .float16)
            #expect(owned.ssm?.shape == [1, 2, Self.side, Self.side])
            #expect(owned.ssm?.dtype == .float32)
            let after = bits(paused)
            #expect(after.conv == before.conv)
            #expect(after.ssm == before.ssm)
            #expect(paused.materializedByteCount == paused.byteCount)

            // The other rows are untouched, and with one row gone the step joins.
            #expect(states[0].state(modelLayerIndex: 0)?.batchOrigin != nil)
            #expect(CBv2RecurrentBatchCarry.carried(layerStates(states)) == nil)

            // A generation that settles while the request stays paused is owned too.
            try step(states, seed: 200)
            #expect(paused.state(modelLayerIndex: 0)?.batchOrigin == nil)
            #expect(states[0].state(modelLayerIndex: 0)?.batchOrigin != nil)
            #expect(bits(paused).conv.first == 202)

            // After the resume the request is an ordinary row again.
            paused.setPaused(false)
            let resumed = try step(states, seed: 300)
            #expect(paused.state(modelLayerIndex: 0)?.batchOrigin === resumed)
            if CBv2RecurrentBatchCarry.enabled {
                #expect(CBv2RecurrentBatchCarry.carried(layerStates(states)) === resumed)
            }
            for state in states { try state.release() }
        }

        /// A one-row view keeps its whole batch allocated. Without the rule a
        /// paused request would hold every lane's row of its last step. MLX's
        /// active memory is global to the process and other suites may run at
        /// the same time, so the scenario gets three attempts.
        @Test(
            .enabled(
                if: CBv2RecurrentRowOwnership.mode == .onPause,
                "the re-own rule is not in its default mode"))
        func pausedRequestDoesNotKeepItsLastBatchAllocated() throws {
            let rows = 16
            let wanted = (rows - 2) * Self.rowBytes
            var best = 0
            for attempt in 0 ..< 3 where best < wanted {
                var states = try makeStates(rows)
                try step(states, seed: Float(7 + 100 * attempt))
                let paused = states[5]
                for (index, state) in states.enumerated() where index != 5 {
                    try state.release()
                }
                states.removeAll()

                // Every other request is gone, and the batch is still allocated.
                let held = Memory.activeMemory
                paused.setPaused(true)
                // The copy's inputs may be released a moment after the pause
                // returns. Wait for that, with a bound.
                let deadline = Date().addingTimeInterval(2)
                var freed = held - Memory.activeMemory
                while freed < wanted, Date() < deadline {
                    Thread.sleep(forTimeInterval: 0.002)
                    freed = held - Memory.activeMemory
                }
                best = max(best, freed)
                #expect(bits(paused).conv.first == Float(12 + 100 * attempt))
                try paused.release()
            }
            #expect(best >= wanted, "pausing must free the other rows of the last batch")
        }
    }
}
