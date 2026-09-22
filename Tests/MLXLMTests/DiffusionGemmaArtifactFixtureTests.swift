import CryptoKit
import Foundation
import Testing

@Suite("Pinned DiffusionGemma artifact integrity")
struct DiffusionGemmaArtifactFixtureTests {
    private func withFixture(_ body: (URL, DiffusionGemmaArtifactFixture.File) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let directory = root.appendingPathComponent("snapshot")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let bytes = Data("original".utf8)
        let blob = root.appendingPathComponent("blob")
        try bytes.write(to: blob)
        try FileManager.default.createSymbolicLink(at: directory.appendingPathComponent("model.safetensors"),
            withDestinationURL: blob)
        let entry = DiffusionGemmaArtifactFixture.File(name: "model.safetensors", size: UInt64(bytes.count),
            sha256: SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined())
        try body(directory, entry)
    }

    @Test func hfSymlinkTargetsAreHashedAndSameSizeTamperingIsRejected() throws {
        try withFixture { directory, entry in
            let first = try DiffusionGemmaArtifactFixture.verify(directory: directory, expected: [entry])
            #expect(first.count == 64)
            try Data("modified".utf8).write(to: directory.appendingPathComponent(entry.name).resolvingSymlinksInPath())
            #expect(throws: (any Error).self) {
                try DiffusionGemmaArtifactFixture.verify(directory: directory, expected: [entry])
            }
        }
    }

    @Test(arguments: ["missing", "extra", "truncated"])
    func rejectsChangedFileInventoryOrLength(_ change: String) throws {
        try withFixture { directory, entry in
            switch change {
            case "missing": try FileManager.default.removeItem(at: directory.appendingPathComponent(entry.name))
            case "extra": try Data().write(to: directory.appendingPathComponent("extra.safetensors"))
            default: try Data().write(to: directory.appendingPathComponent(entry.name).resolvingSymlinksInPath())
            }
            #expect(throws: (any Error).self) {
                try DiffusionGemmaArtifactFixture.verify(directory: directory, expected: [entry])
            }
        }
    }

    @Test func rejectsUnsafeManifestPathsAndDuplicateEntries() throws {
        try withFixture { directory, entry in
            for name in ["", ".", "..", "../blob", "/absolute"] {
                let unsafe = DiffusionGemmaArtifactFixture.File(name: name, size: entry.size, sha256: entry.sha256)
                #expect(throws: (any Error).self) {
                    try DiffusionGemmaArtifactFixture.verify(directory: directory, expected: [unsafe])
                }
            }
            #expect(throws: (any Error).self) {
                try DiffusionGemmaArtifactFixture.verify(directory: directory, expected: [entry, entry])
            }
        }
    }
}
