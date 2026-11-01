// Copyright © 2026 Eigen Labs.

import Foundation

@testable import MLXLMCommon

/// Process-memory owner for GLM5Next engine tests: a 512 MiB charge limit with
/// exact coverage accounting, refusing growth after retirement.
final class GLM5NextTestProcessBudget: CBv2ProcessMemoryOwner, @unchecked Sendable {
    enum Failure: Error { case exhausted, invalidCoverage }
    private let lock = NSLock()
    private var charge: UInt64 = 0
    private var coverage: UInt64 = 0
    private var retired = false
    var bytes: UInt64 { lock.withLock { charge } }
    func replaceCharge(_ bytes: UInt64) throws {
        try lock.withLock {
            guard bytes <= (512 << 20), !retired || bytes <= charge else {
                throw Failure.exhausted
            }
            guard bytes >= coverage else { throw Failure.invalidCoverage }
            charge = bytes
        }
    }
    func recordMaterialization(_ bytes: UInt64) throws {
        try lock.withLock {
            guard bytes >= coverage && bytes <= charge else {
                throw Failure.invalidCoverage
            }
            coverage = bytes
        }
    }
    func withdrawCoverage(_ bytes: UInt64) throws {
        try lock.withLock {
            guard bytes <= coverage else { throw Failure.invalidCoverage }
            coverage -= bytes
        }
    }
    func retire() { lock.withLock { retired = true } }
}
