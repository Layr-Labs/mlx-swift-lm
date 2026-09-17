// Copyright © 2026 Eigen Labs Inc.

import Foundation
import XCTest

@testable import MLXLLM

final class Qwen4ExpPLEResidencyTests: XCTestCase {
    func testFactoryLeaseTransfersToModelOwnership() throws {
        guard Qwen4ExpPLEResidency.useMmap else { throw XCTSkip("Requires default mmap policy") }
        let dir = try makeSnapshot(modelType: "qwen4_exp")
        let loader = try XCTUnwrap(Qwen4ExpPLEResidency.acquireLoadLease(
            directory: dir, modelType: "qwen4_exp"))
        let model = try XCTUnwrap(Qwen4ExpPLEResidency.retainCurrentDirectory())
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 2)
        loader.release()
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 1)
        XCTAssertEqual(Qwen4ExpPLEResidency.modelDirectory, dir.standardizedFileURL)
        model.release()
        XCTAssertNil(Qwen4ExpPLEResidency.modelDirectory)
    }

    func testFactoryLeaseFailureAndNonQwenLeaveNoBinding() throws {
        guard Qwen4ExpPLEResidency.useMmap else { throw XCTSkip("Requires default mmap policy") }
        let dir = try makeSnapshot(modelType: "qwen4_exp")
        XCTAssertNil(try Qwen4ExpPLEResidency.acquireLoadLease(directory: dir, modelType: "gemma4"))
        enum Failure: Error { case fixture }
        func failedLoad() throws {
            let loader = try Qwen4ExpPLEResidency.acquireLoadLease(directory: dir, modelType: "qwen4_exp_text")
            defer { loader?.release() }
            throw Failure.fixture
        }
        XCTAssertThrowsError(try failedLoad())
        XCTAssertNil(Qwen4ExpPLEResidency.modelDirectory)
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 0)
    }

    func testModelLeaseOutlivesLoaderAndDeinitReleasesBinding() throws {
        let dir = try makeSnapshot(modelType: "qwen4_exp")
        XCTAssertTrue(Qwen4ExpPLEResidency.adoptIfQwen4Exp(directory: dir))
        var lease = Qwen4ExpPLEResidency.retainCurrentDirectory()
        XCTAssertNotNil(lease)
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 2)
        Qwen4ExpPLEResidency.release(directory: dir)
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 1)
        XCTAssertEqual(Qwen4ExpPLEResidency.modelDirectory, dir.standardizedFileURL)
        lease = nil
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 0)
        XCTAssertNil(Qwen4ExpPLEResidency.modelDirectory)
    }

    func testExplicitLeaseReleaseCannotReleaseLaterModelsBinding() throws {
        let first = try makeSnapshot(modelType: "qwen4_exp")
        let second = try makeSnapshot(modelType: "qwen4_exp")
        XCTAssertTrue(Qwen4ExpPLEResidency.adoptIfQwen4Exp(directory: first))
        let lease = try XCTUnwrap(Qwen4ExpPLEResidency.retainCurrentDirectory())
        Qwen4ExpPLEResidency.release(directory: first)
        lease.release()
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 0)
        XCTAssertTrue(Qwen4ExpPLEResidency.adoptIfQwen4Exp(directory: second))
        lease.release()
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 1)
        XCTAssertEqual(Qwen4ExpPLEResidency.modelDirectory, second.standardizedFileURL)
        Qwen4ExpPLEResidency.release(directory: second)
    }

    override func setUp() {
        super.setUp()
        Qwen4ExpPLEResidency.reset()
    }

    override func tearDown() {
        Qwen4ExpPLEResidency.reset()
        super.tearDown()
    }

    func testNonQwen4ConfigDoesNotAdopt() throws {
        let dir = try makeSnapshot(modelType: "gemma4")
        XCTAssertFalse(Qwen4ExpPLEResidency.adoptIfQwen4Exp(directory: dir))
        XCTAssertNil(Qwen4ExpPLEResidency.modelDirectory)
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 0)
    }

    func testQwen4AdoptReleaseClearsProcessRoot() throws {
        let dir = try makeSnapshot(modelType: "qwen4_exp")
        XCTAssertTrue(Qwen4ExpPLEResidency.adoptIfQwen4Exp(directory: dir))
        XCTAssertEqual(
            Qwen4ExpPLEResidency.modelDirectory?.standardizedFileURL,
            dir.standardizedFileURL)
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 1)

        Qwen4ExpPLEResidency.release(directory: dir)
        XCTAssertNil(Qwen4ExpPLEResidency.modelDirectory)
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 0)
    }

    func testRetainCountKeepsRootUntilLastRelease() throws {
        let dir = try makeSnapshot(modelType: "qwen4_exp")
        XCTAssertTrue(Qwen4ExpPLEResidency.adoptIfQwen4Exp(directory: dir))
        XCTAssertTrue(Qwen4ExpPLEResidency.adoptIfQwen4Exp(directory: dir))
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 2)

        Qwen4ExpPLEResidency.release(directory: dir)
        XCTAssertEqual(
            Qwen4ExpPLEResidency.modelDirectory?.standardizedFileURL,
            dir.standardizedFileURL)
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 1)

        Qwen4ExpPLEResidency.release(directory: dir)
        XCTAssertNil(Qwen4ExpPLEResidency.modelDirectory)
    }

    func testLiveTreeIsNotOverwrittenByADifferentQwen4Path() throws {
        let first = try makeSnapshot(modelType: "qwen4_exp")
        let second = try makeSnapshot(modelType: "qwen4_exp")
        XCTAssertTrue(Qwen4ExpPLEResidency.adoptIfQwen4Exp(directory: first))
        XCTAssertFalse(Qwen4ExpPLEResidency.adoptIfQwen4Exp(directory: second))
        XCTAssertEqual(
            Qwen4ExpPLEResidency.modelDirectory?.standardizedFileURL,
            first.standardizedFileURL)

        Qwen4ExpPLEResidency.release(directory: second)
        XCTAssertEqual(
            Qwen4ExpPLEResidency.modelDirectory?.standardizedFileURL,
            first.standardizedFileURL)

        Qwen4ExpPLEResidency.release(directory: first)
        XCTAssertNil(Qwen4ExpPLEResidency.modelDirectory)
    }

    func testProviderLoadSeamRejectsDifferentLiveQwen4Tree() throws {
        let first = try makeSnapshot(modelType: "qwen4_exp")
        let second = try makeSnapshot(modelType: "qwen4_exp")
        XCTAssertTrue(
            try Qwen4ExpPLEResidency.adoptForLoadIfQwen4Exp(directory: first))
        XCTAssertThrowsError(
            try Qwen4ExpPLEResidency.adoptForLoadIfQwen4Exp(directory: second)
        ) { error in
            XCTAssertEqual(
                error as? Qwen4ExpPLEResidencyError,
                .conflictingActiveModel)
        }
        XCTAssertEqual(
            Qwen4ExpPLEResidency.modelDirectory?.standardizedFileURL,
            first.standardizedFileURL)
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 1)
    }

    func testDirectNilAssignmentClearsRetainCount() throws {
        let dir = try makeSnapshot(modelType: "qwen4_exp")
        Qwen4ExpPLEResidency.modelDirectory = dir
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 1)
        Qwen4ExpPLEResidency.modelDirectory = nil
        XCTAssertEqual(Qwen4ExpPLEResidency.retainCount, 0)
        XCTAssertNil(Qwen4ExpPLEResidency.modelDirectory)
    }

    private func makeSnapshot(modelType: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen4-ple-residency-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let json = #"{"model_type":"\#(modelType)","vision_config":{}}"#
        try Data(json.utf8).write(to: dir.appendingPathComponent("config.json"))
        addTeardownBlock {
            try? FileManager.default.removeItem(at: dir)
        }
        return dir
    }
}
