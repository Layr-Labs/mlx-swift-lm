import Foundation
import Dispatch
import MLXLMCommon
import Testing

extension UnitTests {
    struct NemotronSuperResidencyTests {
        @Test func additiveAccountingAtomicFailureAndIdempotentRelease() throws {
            let ledger = try NemotronSuperResidency(capacityBytes: 100, maximumRequests: 2)
            let weights = try ledger.reserve(.weights, bytes: 50)
            let staging = try ledger.reserve(.staging, components: [5, 5])
            let request = try ledger.reserve(.request, components: [10, 20])
            let before = ledger.snapshot
            #expect(before.usedBytes == 90 && before.availableBytes == 10 && before.requestCount == 1)
            #expect(throws: NemotronSuperResidency.Failure.capacity) { try ledger.reserve(.request, bytes: 11) }
            #expect(ledger.snapshot == before)
            #expect(request.release())
            #expect(!request.release())
            #expect(ledger.snapshot.usedBytes == 60 && ledger.snapshot.requestCount == 0)
            #expect(staging.release() && weights.release())
            #expect(ledger.snapshot.usedBytes == 0)
        }

        @Test func malformedOverflowAndConcurrencyFailClosed() throws {
            #expect(throws: NemotronSuperResidency.Failure.invalidBudget) {
                try NemotronSuperResidency(capacityBytes: 0, maximumRequests: 1)
            }
            let ledger = try NemotronSuperResidency(capacityBytes: Int.max, maximumRequests: 1, maximumReservations: 2)
            #expect(throws: NemotronSuperResidency.Failure.overflow) {
                try ledger.reserve(.weights, components: [Int.max, 1])
            }
            #expect(throws: NemotronSuperResidency.Failure.invalidBytes) { try ledger.reserve(.staging, bytes: -1) }
            #expect(throws: NemotronSuperResidency.Failure.invalidBytes) { try ledger.reserve(.request, bytes: 0) }
            let request = try ledger.reserve(.request, bytes: 1)
            #expect(throws: NemotronSuperResidency.Failure.concurrency) { try ledger.reserve(.request, bytes: 1) }
            let weights = try ledger.reserve(.weights, bytes: Int.max - 1)
            #expect(ledger.snapshot.usedBytes == Int.max)
            #expect(throws: NemotronSuperResidency.Failure.reservationLimit) { try ledger.reserve(.staging, bytes: 1) }
            request.release(); weights.release()
        }

        @Test func drainIsTerminalAndRetainsLiveCharges() throws {
            let ledger = try NemotronSuperResidency(capacityBytes: 100, maximumRequests: 1)
            let weights = try ledger.reserve(.weights, bytes: 60)
            let request = try ledger.reserve(.request, bytes: 30)
            #expect(ledger.beginDrain().usedBytes == 90)
            #expect(!ledger.beginDrain().isDrained)
            for kind in [NemotronSuperResidency.Kind.weights, .staging, .request] {
                #expect(throws: NemotronSuperResidency.Failure.draining) { try ledger.reserve(kind, bytes: 1) }
            }
            request.release()
            #expect(!ledger.snapshot.isDrained)
            weights.release()
            #expect(ledger.snapshot.isDrained)
            #expect(throws: NemotronSuperResidency.Failure.draining) { try ledger.reserve(.request, bytes: 1) }
        }

        private final class Held: @unchecked Sendable {
            let lock = NSLock()
            var values: [NemotronSuperResidency.Reservation] = []
            func append(_ value: NemotronSuperResidency.Reservation) { lock.withLock { values.append(value) } }
        }

        @Test func concurrentAdmissionNeverOverbooksAndConcurrentReleaseIsIdempotent() throws {
            let ledger = try NemotronSuperResidency(capacityBytes: 100, maximumRequests: 10)
            let held = Held()
            DispatchQueue.concurrentPerform(iterations: 100) { _ in
                if let reservation = try? ledger.reserve(.request, bytes: 10) { held.append(reservation) }
            }
            #expect(held.values.count == 10)
            #expect(ledger.snapshot.usedBytes == 100 && ledger.snapshot.requestCount == 10)
            ledger.beginDrain()
            let reservations = held.values
            DispatchQueue.concurrentPerform(iterations: 100) { index in
                reservations[index % reservations.count].release()
            }
            #expect(ledger.snapshot.isDrained && ledger.snapshot.usedBytes == 0)
        }

        @Test func droppedHandleDoesNotReleaseLiveAllocationCharge() throws {
            let ledger = try NemotronSuperResidency(capacityBytes: 10, maximumRequests: 1)
            _ = try ledger.reserve(.weights, bytes: 10)
            #expect(ledger.snapshot.usedBytes == 10)
            #expect(!ledger.beginDrain().isDrained)
        }
    }
}
