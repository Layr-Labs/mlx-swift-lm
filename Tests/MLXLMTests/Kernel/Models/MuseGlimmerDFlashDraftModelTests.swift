// Copyright © 2026 Eigen Labs.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// The drafter module and its context ring on tiny random weights,
    /// against `MuseGlimmerDFlashReference`. Float32, so the tolerance only
    /// covers the different order of the same sums (fused attention and norm
    /// kernels against plain products): measured differences are below 2e-6
    /// on outputs of unit scale, and the bound is 5e-5. A wrong position,
    /// mask or row measured 0.24 or more; the tests require 0.1.
    @Suite(.serialized)
    struct MuseGlimmerDFlashDraftModelTests {
        typealias Fixture = MuseGlimmerDFlashFixture
        typealias Reference = MuseGlimmerDFlashReference
        static let tolerance: Float = 5e-5
        static let discrimination: Float = 0.1

        /// Commit `rows` feature rows from `start` in pieces of `chunks`.
        private func commit(
            _ model: MuseGlimmerDFlashDraftModel, _ state: MuseGlimmerDFlashContextState,
            _ features: MLXArray, start: Int, chunks: [Int]
        ) {
            var row = 0
            var index = 0
            while row < features.dim(0) {
                let count = min(chunks[index % chunks.count], features.dim(0) - row)
                model.commit(
                    features: features[row ..< (row + count), 0...], at: start + row, to: state)
                eval(state.arrays)
                row += count
                index += 1
            }
        }

        @Test func blockWithoutContextMatchesReference() throws {
            let c = try Fixture.configuration()
            let model = try Fixture.model(c)
            let embeddings = Fixture.embeddings(configuration: c, seed: 3)
            for anchor in [0, 5, 700] {
                let state = model.makeContextState()
                state.restart(at: anchor)
                let drafted = model.draftBlock(embeddings: embeddings, state: state)
                let expected = Reference.block(
                    model: model, features: nil, positions: [], embeddings: embeddings,
                    anchor: anchor)
                #expect(drafted.shape == [c.blockSize, c.hiddenSize])
                #expect(Fixture.maxAbsDifference(drafted, expected) < Self.tolerance)
            }
        }

        @Test func cachedContextMatchesUncachedRecompute() throws {
            let c = try Fixture.configuration()
            let model = try Fixture.model(c)
            let embeddings = Fixture.embeddings(configuration: c, seed: 4)
            for (rows, chunks) in [(1, [1]), (5, [5]), (7, [1, 2, 4]), (9, [3]), (12, [5, 7])] {
                let features = Fixture.features(rows: rows, configuration: c, seed: UInt64(rows))
                let state = model.makeContextState()
                commit(model, state, features, start: 0, chunks: chunks)
                #expect(state.nextPosition == rows && state.firstPosition == 0)
                let drafted = model.draftBlock(embeddings: embeddings, state: state)
                let expected = Reference.block(
                    model: model, features: features, positions: Array(0 ..< rows),
                    embeddings: embeddings, anchor: rows)
                #expect(
                    Fixture.maxAbsDifference(drafted, expected) < Self.tolerance,
                    "rows \(rows) chunks \(chunks)")
                // RoPE is relative: moving every position by one changes nothing.
                let shifted = Reference.block(
                    model: model, features: features, positions: Array(1 ... rows),
                    embeddings: embeddings, anchor: rows + 1)
                #expect(Fixture.maxAbsDifference(drafted, shifted) < Self.tolerance)
                // The anchor one position off its context is a different block.
                let misplaced = Reference.block(
                    model: model, features: features, positions: Array(0 ..< rows),
                    embeddings: embeddings, anchor: rows + 1)
                #expect(Fixture.maxAbsDifference(drafted, misplaced) > Self.discrimination)
            }
        }

        /// Past the window the ring overwrites its oldest rows, and each block
        /// query loses the keys further than `window` behind it.
        @Test func ringWrapsAndKeepsThePerQueryWindow() throws {
            let c = try Fixture.configuration(window: 12, blockSize: 4)
            let model = try Fixture.model(c)
            let embeddings = Fixture.embeddings(configuration: c, seed: 5)
            let total = 41
            let features = Fixture.features(rows: total, configuration: c, seed: 77)
            for chunks in [[1], [5], [12], [13], [40], [3, 1, 17]] {
                let state = model.makeContextState()
                var committed = 0
                var index = 0
                while committed < total {
                    let count = min(chunks[index % chunks.count], total - committed)
                    model.commit(
                        features: features[committed ..< (committed + count), 0...],
                        at: committed, to: state)
                    eval(state.arrays)
                    committed += count
                    index += 1
                    #expect(state.nextPosition == committed)
                    #expect(state.firstPosition == max(0, committed - c.slidingWindow))
                    #expect(Set(state.heldPositions) == Set(state.firstPosition ..< committed))
                    let drafted = model.draftBlock(embeddings: embeddings, state: state)
                    // The reference sees every committed row and masks by distance.
                    let expected = Reference.block(
                        model: model, features: features[0 ..< committed, 0...],
                        positions: Array(0 ..< committed), embeddings: embeddings,
                        anchor: committed)
                    #expect(
                        Fixture.maxAbsDifference(drafted, expected) < Self.tolerance,
                        "chunks \(chunks) committed \(committed)")
                }
            }
        }

        /// The reference mask keeps a key exactly `window` positions behind a
        /// query. Other implementations stop one position earlier; that is a
        /// different result once the context reaches the window.
        @Test func windowBoundaryIsInclusive() throws {
            let c = try Fixture.configuration(window: 12, blockSize: 4)
            let model = try Fixture.model(c)
            let embeddings = Fixture.embeddings(configuration: c, seed: 6)
            for rows in [8, 9, 11, 12, 13, 30] {
                let features = Fixture.features(rows: rows, configuration: c, seed: UInt64(rows))
                let state = model.makeContextState()
                commit(model, state, features, start: 0, chunks: [4])
                let drafted = model.draftBlock(embeddings: embeddings, state: state)
                let positions = Array(0 ..< rows)
                let inclusive = Reference.block(
                    model: model, features: features, positions: positions,
                    embeddings: embeddings, anchor: rows)
                let strict = Reference.block(
                    model: model, features: features, positions: positions,
                    embeddings: embeddings, anchor: rows, strictWindow: true)
                #expect(Fixture.maxAbsDifference(drafted, inclusive) < Self.tolerance)
                // The last query, at `rows + blockSize - 1`, is the first to reach
                // a key exactly `window` behind it.
                let boundaryInUse = rows + c.blockSize - 1 >= c.slidingWindow
                let gap = Fixture.maxAbsDifference(inclusive, strict)
                #expect(
                    boundaryInUse ? gap > Self.discrimination : gap == 0, "rows \(rows) gap \(gap)")
                if boundaryInUse {
                    #expect(Fixture.maxAbsDifference(drafted, strict) > Self.discrimination)
                }
            }
        }

        /// A block's own keys and values never enter the state: drafting
        /// changes no byte of it, and drafting again gives the same block.
        @Test func blockNeverEntersTheState() throws {
            let c = try Fixture.configuration(window: 12, blockSize: 4)
            let model = try Fixture.model(c)
            let embeddings = Fixture.embeddings(configuration: c, seed: 8)
            for rows in [3, 12, 29] {
                let features = Fixture.features(rows: rows, configuration: c, seed: 90)
                let state = model.makeContextState()
                commit(model, state, features, start: 0, chunks: [7])
                let before = state.arrays.map { $0.asData().data }
                let positions = (state.firstPosition, state.nextPosition)
                let first = model.draftBlock(embeddings: embeddings, state: state)
                eval(first)
                let second = model.draftBlock(embeddings: embeddings, state: state)
                eval(second)
                #expect(state.arrays.map { $0.asData().data } == before)
                #expect((state.firstPosition, state.nextPosition) == positions)
                #expect(first.asData().data == second.asData().data)
                // A later commit does not change an already built block.
                let third = model.draftBlock(embeddings: embeddings, state: state)
                model.commit(
                    features: Fixture.features(rows: 2, configuration: c, seed: 91),
                    at: rows, to: state)
                eval(third, state.arrays[0], state.arrays[1])
                #expect(third.asData().data == first.asData().data)
            }
        }

        /// A state that starts at a later position, as after a restored
        /// prefix, equals a reference that only has those rows at those
        /// absolute positions.
        @Test func stateStartingMidSequenceKeepsAbsolutePositions() throws {
            let c = try Fixture.configuration(window: 12, blockSize: 4)
            let model = try Fixture.model(c)
            let embeddings = Fixture.embeddings(configuration: c, seed: 9)
            for start in [1, 37, 1000] {
                for rows in [1, 6, 12, 20] {
                    let features = Fixture.features(rows: rows, configuration: c, seed: 12)
                    let state = model.makeContextState()
                    commit(model, state, features, start: start, chunks: [5])
                    #expect(state.nextPosition == start + rows)
                    #expect(state.firstPosition == max(start, start + rows - c.slidingWindow))
                    let drafted = model.draftBlock(embeddings: embeddings, state: state)
                    let expected = Reference.block(
                        model: model, features: features,
                        positions: Array(start ..< (start + rows)), embeddings: embeddings,
                        anchor: start + rows)
                    #expect(
                        Fixture.maxAbsDifference(drafted, expected) < Self.tolerance,
                        "start \(start) rows \(rows)")
                }
            }
        }

        /// Rows already held are skipped; a gap restarts the context there.
        @Test func duplicatesAreSkippedAndGapsRestart() throws {
            let c = try Fixture.configuration(window: 12, blockSize: 4)
            let model = try Fixture.model(c)
            let embeddings = Fixture.embeddings(configuration: c, seed: 10)
            let features = Fixture.features(rows: 10, configuration: c, seed: 13)
            let state = model.makeContextState()
            commit(model, state, features[0 ..< 6, 0...], start: 0, chunks: [6])
            // Rows 4 and 5 again, then 6 and 7.
            model.commit(features: features[4 ..< 8, 0...], at: 4, to: state)
            model.commit(features: features[2 ..< 5, 0...], at: 2, to: state)
            #expect(state.nextPosition == 8 && state.firstPosition == 0)
            var drafted = model.draftBlock(embeddings: embeddings, state: state)
            var expected = Reference.block(
                model: model, features: features[0 ..< 8, 0...], positions: Array(0 ..< 8),
                embeddings: embeddings, anchor: 8)
            #expect(Fixture.maxAbsDifference(drafted, expected) < Self.tolerance)
            // Position 8 never arrives: only row 9 remains.
            model.commit(features: features[9 ..< 10, 0...], at: 9, to: state)
            #expect(state.nextPosition == 10 && state.firstPosition == 9)
            drafted = model.draftBlock(embeddings: embeddings, state: state)
            expected = Reference.block(
                model: model, features: features[9 ..< 10, 0...], positions: [9],
                embeddings: embeddings, anchor: 10)
            #expect(Fixture.maxAbsDifference(drafted, expected) < Self.tolerance)
        }

        /// Half-width weights and activations stay close to the float32
        /// reference. The bound is the rounding of 16-bit activations through
        /// two layers (measured 0.025 for BF16 and 0.003 for F16 on outputs
        /// of mean magnitude 0.79), not an exactness claim.
        @Test(arguments: [DType.bfloat16, .float16])
        func halfPrecisionTracksTheReference(dtype: DType) throws {
            let c = try Fixture.configuration(window: 12, blockSize: 4)
            let model = try Fixture.model(c, dtype: dtype)
            let embeddings = Fixture.embeddings(configuration: c, seed: 14)
            let features = Fixture.features(rows: 20, configuration: c, seed: 15)
            let state = model.makeContextState()
            commit(model, state, features, start: 0, chunks: [8])
            let drafted = model.draftBlock(embeddings: embeddings, state: state)
            #expect(drafted.dtype == dtype)
            #expect(state.arrays.allSatisfy { $0.dtype == dtype })
            let expected = Reference.block(
                model: model, features: features, positions: Array(0 ..< 20),
                embeddings: embeddings, anchor: 20)
            #expect(
                Fixture.maxAbsDifference(drafted, expected) < (dtype == .bfloat16 ? 0.08 : 0.02))
        }

        @Test func stateAllocatesOnceAtItsDeclaredSize() throws {
            let c = try Fixture.configuration(window: 12, blockSize: 4)
            let model = try Fixture.model(c)
            let state = model.makeContextState()
            #expect(!state.isAllocated && state.arrays.isEmpty)
            let features = Fixture.features(rows: 30, configuration: c, seed: 16)
            commit(model, state, features, start: 0, chunks: [1])
            let bytes = state.arrays.reduce(0) { $0 + $1.nbytes }
            #expect(
                bytes
                    == MuseGlimmerDFlashContextState.residentBytes(
                        configuration: c, dtype: .float32))
            #expect(
                state.arrays.allSatisfy {
                    $0.shape == [c.hiddenLayers, c.kvHeads, c.slidingWindow, c.headDimensions]
                })
            state.release()
            #expect(!state.isAllocated && state.count == 0)
        }
    }
}
