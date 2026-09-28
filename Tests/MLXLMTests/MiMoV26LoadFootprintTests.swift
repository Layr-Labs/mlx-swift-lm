import Foundation
@testable import MLXVLM
import XCTest

/// Actual estimate implementation over authenticated metadata only. These
/// assertions allocate no model arrays and do not measure a full-model peak.
final class MiMoV26LoadFootprintTests: XCTestCase {
    private struct Inventory: Decodable {
        struct File: Decodable { let name: String; let bytes: Int }
        let files: [File]
    }

    func testSelectedBundlePricesEveryComponentAndCopyFamily() throws {
        let (plan, files) = try inputs()
        let estimate = try MiMoV26LoadFootprint.estimateValidatedLayout(
            plan, files: files, allocationPageBytes: 16_384)
        // Independently calculated from the frozen1258-tensor header inventory.
        XCTAssertEqual(estimate.residentBytes, 175_362_473_140)
        XCTAssertEqual(estimate.largestShardBytes, 4_230_398_519)
        XCTAssertEqual(estimate.largestTensorBytes, 1_249_902_592)
        XCTAssertEqual(estimate.largestExpertBlockBytes, 3_422_552_064)
        XCTAssertEqual(estimate.auxiliaryCopyBytes, 2_535_180_672)
        XCTAssertEqual(estimate.pageRoundingBytes, 41_222_144)
        XCTAssertEqual(estimate.metadataAllowanceBytes, 1_073_741_824)
        XCTAssertEqual(estimate.transientBytes, 12_552_997_815)
        XCTAssertEqual(estimate.totalBytes, 187_915_470_955)
        XCTAssertEqual(estimate.tensorCount, 1258)
        XCTAssertEqual(estimate.configSHA256, plan.configSHA256)
        XCTAssertEqual(estimate.indexSHA256, plan.indexSHA256)
        XCTAssertEqual(estimate.descriptorSHA256, plan.descriptorSHA256)
        XCTAssertEqual(estimate.contract, MiMoV26LoadFootprint.contract)
    }

    func testFileOrderCannotChangeEstimateOrDropInventory() throws {
        let (plan, files) = try inputs()
        let original = try MiMoV26LoadFootprint.estimateValidatedLayout(
            plan, files: files, allocationPageBytes: 16_384)
        XCTAssertEqual(original, try MiMoV26LoadFootprint.estimateValidatedLayout(
            plan, files: Array(files.reversed()), allocationPageBytes: 16_384))
        XCTAssertThrowsError(try MiMoV26LoadFootprint.estimateValidatedLayout(
            plan, files: Array(files.dropLast()), allocationPageBytes: 16_384))
        var duplicate = files; duplicate[1] = duplicate[0]
        XCTAssertThrowsError(try MiMoV26LoadFootprint.estimateValidatedLayout(
            plan, files: duplicate, allocationPageBytes: 16_384))
        XCTAssertThrowsError(try MiMoV26LoadFootprint.estimateValidatedLayout(
            plan, files: files + [("unaccounted.safetensors", 1024)], allocationPageBytes: 16_384))
    }

    func testTruncatedOrOversizedFileMetadataCannotObtainAllowance() throws {
        let (plan, files) = try inputs()
        for size in [-1, 0, 8, 1024] {
            var changed = files; changed[0].bytes = size
            XCTAssertThrowsError(try MiMoV26LoadFootprint.estimateValidatedLayout(
                plan, files: changed, allocationPageBytes: 16_384))
        }
        var excessive = files; excessive[0].bytes += 32 * 1_048_576
        XCTAssertThrowsError(try MiMoV26LoadFootprint.estimateValidatedLayout(
            plan, files: excessive, allocationPageBytes: 16_384))
        let overflowing = files.map { (name: $0.name, bytes: Int.max) }
        XCTAssertThrowsError(try MiMoV26LoadFootprint.estimateValidatedLayout(
            plan, files: overflowing, allocationPageBytes: 16_384)) {
            XCTAssertEqual($0 as? MiMoV26LoadFootprintError, .overflow)
        }
    }

    func testOtherPageSizeAndUnqualifiedPrecisionRemainIneligible() throws {
        let (plan, files) = try inputs()
        for page in [0, 4096, 8192, 32768] {
            XCTAssertThrowsError(try MiMoV26LoadFootprint.estimateValidatedLayout(
                plan, files: files, allocationPageBytes: page)) {
                XCTAssertEqual($0 as? MiMoV26LoadFootprintError, .unsupportedProfile)
            }
        }
        let tiny = try convertedPlan("tiny") // Existing validFP32 component fixture.
        let tinyFiles = tiny.rootFiles.sorted().map { (name: $0, bytes: 1_048_576) }
        XCTAssertThrowsError(try MiMoV26LoadFootprint.estimateValidatedLayout(
            tiny, files: tinyFiles, allocationPageBytes: 16_384)) {
            XCTAssertEqual($0 as? MiMoV26LoadFootprintError, .unsupportedProfile)
        }
    }

    private func inputs() throws -> (MiMoV26ConvertedLoadPlan, [(name: String, bytes: Int)]) {
        guard let file = ProcessInfo.processInfo.environment["MIMO_V26_FOOTPRINT_FILE_INVENTORY"] else {
            throw XCTSkip("Set MIMO_V26_FOOTPRINT_FILE_INVENTORY to frozen header file inventory")
        }
        let inventory = try JSONDecoder().decode(Inventory.self, from: Data(contentsOf: URL(fileURLWithPath: file)))
        return (try convertedPlan("actual"), inventory.files.map { (name: $0.name, bytes: $0.bytes) })
    }
    private func convertedPlan(_ prefix: String) throws -> MiMoV26ConvertedLoadPlan {
        guard let root = ProcessInfo.processInfo.environment["MIMO_V26_CONVERTED_LOAD_FIXTURES"] else {
            throw XCTSkip("Set MIMO_V26_CONVERTED_LOAD_FIXTURES to owned metadata fixture directory")
        }
        let directory = URL(fileURLWithPath: root)
        func data(_ name: String) throws -> Data { try Data(contentsOf: directory.appendingPathComponent(name)) }
        let provenanceObject = try JSONSerialization.jsonObject(with: data("provenance.json")) as? [String: Any]
        let provenance = try XCTUnwrap(provenanceObject)
        return try MiMoV26ConvertedLoadPlan.make(configurationData: data(prefix + "-config.json"),
            indexData: data(prefix + "-index.json"),
            descriptors: JSONDecoder().decode([String: MiMoV26ConvertedTensorDescriptor].self,
                from: data(prefix + "-descriptors.json")),
            provenance: .init(artifactID: XCTUnwrap(provenance["artifactID"] as? String),
                sourceRepository: XCTUnwrap(provenance["sourceRepository"] as? String),
                sourceRevision: XCTUnwrap(provenance["sourceRevision"] as? String),
                conversionManifestSHA256: XCTUnwrap(provenance["conversionManifestSHA256"] as? String)))
    }
}
