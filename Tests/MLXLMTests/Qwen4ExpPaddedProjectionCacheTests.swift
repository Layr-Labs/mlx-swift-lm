import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM

final class Qwen4ExpPaddedProjectionCacheTests: XCTestCase {
    private final class Owner {}

    private func sources(_ value: UInt32) -> [MLXArray] {
        [MLXArray([value, value], [2, 1]),
         MLXArray([Float(value), Float(value)], [2, 1]),
         MLXArray([-Float(value), -Float(value)], [2, 1])]
    }

    private func bank(_ cache: Qwen4ExpPaddedProjectionCache, _ owner: AnyObject,
                      _ inputs: [MLXArray], compute: DType = .float32, rows: Int = 4)
        -> Qwen4ExpPaddedProjectionCache.Bank {
        cache.bank(owner: owner, weight: inputs[0], scales: inputs[1], biases: inputs[2],
                   compute: compute, targetRows: rows)
    }

    func testDeterministicOwnerKeyCollisionNeverReturnsOtherLayersWeights() {
        Device.withDefaultDevice(.cpu) {
            let firstOwner = Owner()
            let secondOwner = Owner()
            let collisionKey = ObjectIdentifier(firstOwner)
            let cache = Qwen4ExpPaddedProjectionCache(keyForOwner: { _ in collisionKey })
            let firstInputs = sources(3)
            let first = bank(cache, firstOwner, firstInputs)
            XCTAssertTrue(first.weight === bank(cache, firstOwner, firstInputs).weight)

            let second = bank(cache, secondOwner, sources(9))
            XCTAssertFalse(first.weight === second.weight)
            XCTAssertEqual(first.weight.asArray(UInt32.self), [3, 3, 0, 0])
            XCTAssertEqual(second.weight.asArray(UInt32.self), [9, 9, 0, 0])
            XCTAssertEqual(second.scales.asArray(Float.self), [9, 9, 0, 0])
            XCTAssertEqual(second.biases.asArray(Float.self), [-9, -9, 0, 0])
        }
    }

    func testOwnerDeallocatesAndPruningReleasesItsCachedBank() {
        Device.withDefaultDevice(.cpu) {
            let cache = Qwen4ExpPaddedProjectionCache()
            weak var releasedOwner: Owner?
            weak var releasedBank: MLXArray?
            do {
                let owner = Owner()
                releasedOwner = owner
                releasedBank = bank(cache, owner, sources(4)).weight
                XCTAssertEqual(cache.retainedEntryCount, 1)
            }
            XCTAssertNil(releasedOwner, "Padding must not pin the unloaded model's layer")
            XCTAssertNotNil(releasedBank)
            cache.pruneDeadOwners()
            XCTAssertNil(releasedBank)
            XCTAssertEqual(cache.retainedEntryCount, 0)
            XCTAssertEqual(cache.retainedBytes, 0)

            // Normal access also prunes without requiring an explicit unload hook.
            do {
                let owner = Owner()
                _ = bank(cache, owner, sources(6))
            }
            let next = Owner()
            _ = bank(cache, next, sources(7))
            XCTAssertEqual(cache.retainedEntryCount, 1)
        }
    }

    func testReplacingEverySourceDescriptorInvalidatesWithoutChangingOldBank() {
        Device.withDefaultDevice(.cpu) {
            for parameter in 0..<3 {
                let cache = Qwen4ExpPaddedProjectionCache()
                let owner = Owner()
                let inputs = sources(2)
                let before = bank(cache, owner, inputs)
                let replacement = sources(7)
                // Module.update/indexed assignment can preserve the Swift
                // MLXArray object while replacing its underlying descriptor.
                let sameWrapper = inputs[parameter]
                sameWrapper._updateInternal(replacement[parameter])
                let after = bank(cache, owner, inputs)
                XCTAssertFalse(before.weight === after.weight)
                XCTAssertTrue(after.weight === bank(cache, owner, inputs).weight)
                XCTAssertEqual(before.weight.asArray(UInt32.self), [2, 2, 0, 0])
                XCTAssertEqual(before.scales.asArray(Float.self), [2, 2, 0, 0])
                XCTAssertEqual(before.biases.asArray(Float.self), [-2, -2, 0, 0])
                XCTAssertEqual(after.weight.asArray(UInt32.self), parameter == 0 ? [7, 7, 0, 0] : [2, 2, 0, 0])
                XCTAssertEqual(after.scales.asArray(Float.self), parameter == 1 ? [7, 7, 0, 0] : [2, 2, 0, 0])
                XCTAssertEqual(after.biases.asArray(Float.self), parameter == 2 ? [-7, -7, 0, 0] : [-2, -2, 0, 0])
            }
        }
    }

    func testComputeDTypeTargetRowsAndExecutionStreamArePartOfReuse() {
        Device.withDefaultDevice(.cpu) {
            let cache = Qwen4ExpPaddedProjectionCache()
            let owner = Owner()
            let inputs = sources(5)
            let first = bank(cache, owner, inputs, compute: .float32)
            let differentCompute = bank(cache, owner, inputs, compute: .float16)
            XCTAssertFalse(first.weight === differentCompute.weight)
            XCTAssertTrue(differentCompute.weight === bank(cache, owner, inputs, compute: .float16).weight)
            // Match the original concatenation/promotion, without changing math.
            let expected = concatenated([inputs[1], MLXArray.zeros([2, 1], dtype: .float16)], axis: 0)
            XCTAssertEqual(differentCompute.scales.dtype, expected.dtype)
            XCTAssertEqual(differentCompute.scales.asArray(Float.self), expected.asArray(Float.self))
            let wider = bank(cache, owner, inputs, rows: 8)
            XCTAssertEqual(wider.weight.asArray(UInt32.self), [5, 5, 0, 0, 0, 0, 0, 0])
            Stream.withNewDefaultStream(device: .cpu) {
                let otherStream = bank(cache, owner, inputs, rows: 8)
                XCTAssertFalse(wider.weight === otherStream.weight)
                XCTAssertEqual(otherStream.weight.asArray(UInt32.self), wider.weight.asArray(UInt32.self))
            }
        }
    }

    func testCacheHasEntryAndByteBoundsForLiveOwners() {
        Device.withDefaultDevice(.cpu) {
            let owner = Owner()
            let other = Owner()
            let inputs = sources(1)
            let cache = Qwen4ExpPaddedProjectionCache(maximumEntries: 1)
            let first = bank(cache, owner, inputs)
            _ = bank(cache, other, inputs)
            XCTAssertEqual(cache.retainedEntryCount, 1)
            XCTAssertFalse(first.weight === bank(cache, owner, inputs).weight)
            let uncached = Qwen4ExpPaddedProjectionCache(maximumBytes: 0)
            let one = bank(uncached, owner, inputs)
            let two = bank(uncached, owner, inputs)
            XCTAssertFalse(one.weight === two.weight)
            XCTAssertEqual(one.weight.asArray(UInt32.self), two.weight.asArray(UInt32.self))
            XCTAssertEqual(uncached.retainedEntryCount, 0)
            XCTAssertEqual(uncached.retainedBytes, 0)
        }
    }

    func testProductionPaddingSeamInvalidatesQuantizedLinearParameterUpdate() {
        Device.withDefaultDevice(.cpu) {
            let weight = MLXArray(Array(repeating: UInt32(3), count: 32), [4, 8])
            let scales = MLXArray(Array(repeating: Float(2), count: 4), [4, 1])
            let biases = MLXArray(Array(repeating: Float(-1), count: 4), [4, 1])
            let layer = QuantizedLinear(weight: weight, scales: scales, biases: biases,
                                        groupSize: 64, bits: 4)
            let first = Qwen4ExpAffineQMM.paddedInjectBank(layer, compute: .float32, targetRows: 32)!
            layer.update(parameters: ModuleParameters.unflattened([
                ("scales", MLXArray(Array(repeating: Float(9), count: 4), [4, 1]))
            ]))
            let second = Qwen4ExpAffineQMM.paddedInjectBank(layer, compute: .float32, targetRows: 32)!
            XCTAssertFalse(first.scales === second.scales)
            XCTAssertEqual(Array(first.scales.asArray(Float.self).prefix(4)), [2, 2, 2, 2])
            XCTAssertEqual(Array(second.scales.asArray(Float.self).prefix(4)), [9, 9, 9, 9])
            XCTAssertEqual(second.weight.asArray(UInt32.self), first.weight.asArray(UInt32.self))
            XCTAssertEqual(second.biases.asArray(Float.self), first.biases.asArray(Float.self))
        }
    }
}
