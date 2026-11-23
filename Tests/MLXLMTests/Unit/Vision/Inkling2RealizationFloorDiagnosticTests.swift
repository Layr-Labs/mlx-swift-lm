// Copyright © 2026 Eigen Labs.

import Foundation
import Testing

extension UnitTests {

    /// The rule of the Inkling 2 vision diagnostic (b) ("vision gate
    /// v2.2"), `Inkling2VisionPrecisionFixture.beyondTheHeldOutReference`, on
    /// the checked-in reference data, without the checkpoint.
    @Suite
    struct Inkling2RealizationFloorDiagnosticTests {
        typealias Check = Inkling2VisionPrecisionFixture.FloorCheck

        /// The 24 checks with every value at its floor.
        static func checksAtTheFloors(_ fixture: Inkling2VisionPrecisionFixture) -> [Check] {
            fixture.images.flatMap { image in
                [
                    ("tower output", image.realizationFloor.tower, image.heldOutShortfall.tower),
                    ("image embeds", image.realizationFloor.embeds, image.heldOutShortfall.embeds),
                ].flatMap { label, floor, heldOut in
                    let (tail, heldOutTail) =
                        image.tail == "min" ? (floor.min, heldOut.min) : (floor.p01, heldOut.p01)
                    return [
                        Check(
                            name: "image\(image.index) \(label) mean", value: floor.mean,
                            floor: floor.mean, allowedShortfall: heldOut.mean),
                        Check(
                            name: "image\(image.index) \(label) \(image.tail)", value: tail,
                            floor: tail, allowedShortfall: heldOutTail),
                    ]
                }
            }
        }

        /// `checks` with those at `indices` lowered by `fraction` of their
        /// largest held-out shortfall.
        static func lowered(_ checks: [Check], _ indices: [Int], fraction: Double) -> [Check] {
            var result = checks
            for index in indices {
                let check = checks[index]
                result[index] = Check(
                    name: check.name, value: check.floor - check.allowedShortfall * fraction,
                    floor: check.floor, allowedShortfall: check.allowedShortfall)
            }
            return result
        }

        @Test func heldOutDataCountsTheUnshuffledRunOnce() throws {
            let fixture = try Inkling2VisionPrecisionFixture.load()
            #expect(fixture.images.allSatisfy { $0.realizations == 19 })
            #expect(fixture.images.map(\.tail) == ["p01", "min", "min", "min", "p01", "p01"])
            #expect(fixture.heldOut.checks == 24)
            #expect(fixture.heldOut.misses.count == 19)
            #expect(fixture.heldOut.misses["row0"] != nil)
            #expect(fixture.heldOut.misses["channel0"] == nil)
            #expect(fixture.heldOut.worstMisses == fixture.heldOut.misses.values.max())
            #expect(fixture.heldOut.worstMisses == 3)
            let checks = Self.checksAtTheFloors(fixture)
            #expect(checks.count == fixture.heldOut.checks)
            #expect(checks.allSatisfy { $0.allowedShortfall >= 0 })
            #expect(fixture.beyondTheHeldOutReference(checks).isEmpty)
        }

        @Test func reportsMissesBeyondTheHeldOutReference() throws {
            let fixture = try Inkling2VisionPrecisionFixture.load()
            let checks = Self.checksAtTheFloors(fixture)
            let missable = checks.indices.filter { checks[$0].allowedShortfall > 0 }
            let worst = fixture.heldOut.worstMisses
            try #require(missable.count > worst)

            let atWorst = Self.lowered(checks, Array(missable.prefix(worst)), fraction: 0.99)
            #expect(atWorst.filter(\.missed).count == worst)
            #expect(atWorst.filter(\.beyondTheReference).isEmpty)
            #expect(fixture.beyondTheHeldOutReference(atWorst).isEmpty)

            let tooFar = Self.lowered(checks, [missable[0]], fraction: 1.01)
            #expect(tooFar.filter(\.beyondTheReference).map(\.name) == [checks[missable[0]].name])
            #expect(fixture.beyondTheHeldOutReference(tooFar).count == 1)

            let tooMany = Self.lowered(checks, Array(missable.prefix(worst + 1)), fraction: 0.5)
            #expect(tooMany.filter(\.beyondTheReference).isEmpty)
            #expect(fixture.beyondTheHeldOutReference(tooMany).count == 1)

            let noAllowance = Check(
                name: "no held-out miss", value: 0.5 - 1e-9, floor: 0.5, allowedShortfall: 0)
            #expect(noAllowance.beyondTheReference)
            #expect(fixture.beyondTheHeldOutReference([noAllowance]).count == 1)
            let atFloor = Check(
                name: "at the floor", value: 0.5, floor: 0.5, allowedShortfall: 0)
            #expect(!atFloor.missed && !atFloor.beyondTheReference)
        }
    }
}
