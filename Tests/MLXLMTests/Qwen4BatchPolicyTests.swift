import Foundation
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon

final class Qwen4BatchPolicyTests: XCTestCase {
    private final class Fixture: @unchecked Sendable {
        let policy = Qwen4ExpBatchedQSAPolicyState()
        var enabled = false
        var writes = 0
        let resultsLock = NSLock()
        var installed: [Bool] = []
        var refused = 0

        func install(_ value: Bool) throws {
            try policy.configure(enabled: value, seal: true, current: { enabled }) {
                enabled = value
                writes += 1
            }
        }
    }

    func testCompetingInstallationsSealOneValueAndSameValueDoesNotRewrite() throws {
        let fixture = Fixture()
        DispatchQueue.concurrentPerform(iterations: 16) { index in
            let value = index.isMultiple(of: 2)
            do {
                try fixture.install(value)
                fixture.resultsLock.withLock { fixture.installed.append(value) }
            } catch {
                XCTAssertEqual(error as? CBv2Qwen4BatchPolicyError, .alreadySealed)
                fixture.resultsLock.withLock { fixture.refused += 1 }
            }
        }
        XCTAssertEqual(fixture.installed.count, 8)
        XCTAssertEqual(fixture.refused, 8)
        XCTAssertTrue(fixture.installed.allSatisfy { $0 == fixture.enabled })
        let writes = fixture.writes
        for _ in 0..<8 { try fixture.install(fixture.enabled) }
        XCTAssertEqual(fixture.writes, writes)
        XCTAssertThrowsError(try fixture.install(!fixture.enabled))
        XCTAssertEqual(fixture.writes, writes)
    }

    func testFirstForwardSealsCurrentChoiceWithoutMutation() throws {
        let fixture = Fixture()
        fixture.policy.sealOnForward { fixture.enabled }
        XCTAssertThrowsError(try fixture.install(true))
        try fixture.install(false)
        XCTAssertEqual(fixture.writes, 0)
        XCTAssertFalse(fixture.policy.read { fixture.enabled })
    }
}
