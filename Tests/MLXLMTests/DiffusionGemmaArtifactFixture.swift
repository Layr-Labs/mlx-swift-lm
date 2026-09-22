import CryptoKit
import Foundation

/// Test-only identity for the public, immutable MLX checkpoint. Verifies actual
/// bytes (including processor/generation metadata), not a locally minted receipt.
/// Shared with the provider tests through a source symlink.
enum DiffusionGemmaArtifactFixture {
    static let modelID = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
    static let revision = "a7a81407613811e8ba63af92ac0d852b809e191f"
    struct File {
        let name: String
        let size: UInt64
        let sha256: String
    }
    enum Failure: Error { case invalidManifest, unexpectedFiles, mismatchedFile(String) }
    static let files: [File] = [
        .init(name: ".gitattributes", size: 1570, sha256: "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930"),
        .init(name: "README.md", size: 779, sha256: "ad3262ee1b00cc856307a469e710991295712fee20480789aa522152785866c8"),
        .init(name: "chat_template.jinja", size: 17336, sha256: "2f1b4d75d067bae3fe44e676721c7f077d243bc007156cb9c2f8b5836613d082"),
        .init(name: "config.json", size: 58854, sha256: "b41320c97651075363f2895e2cbb3d1580670ee11edb653a14290a35bbf7cac5"),
        .init(name: "generation_config.json", size: 357, sha256: "99334f763c3dbe8b161aeaca1c150a05344299fda2d2e4a0e1d342c744461200"),
        .init(name: "model-00001-of-00004.safetensors", size: 5215121220, sha256: "4e9e90d11166740e647ab81edc7eb85ce93fbd97bd6a3b352407224b6337833a"),
        .init(name: "model-00002-of-00004.safetensors", size: 5358706305, sha256: "67508ed60a26b9e725be0fb4c2e5bb8f0b76f02ecd33650c66c3330164257ad2"),
        .init(name: "model-00003-of-00004.safetensors", size: 5367044182, sha256: "a360651889edf28e7cc50e6d0e30f617b052d8b998cd9b826394f0325cfe9bad"),
        .init(name: "model-00004-of-00004.safetensors", size: 602183698, sha256: "ead1033d8d0907f5f7820efb5292a87359ba4683ce9b947fddf5e2e57792ae5f"),
        .init(name: "model.safetensors.index.json", size: 165362, sha256: "9af3e8635b62d12870e3fe0cc63eafb4bc42d9e031eac409eb5f32cdc974e902"),
        .init(name: "processor_config.json", size: 911, sha256: "b113d8d45d04095c078674b92ec5f32b2857abee2c99c6993e1e557affd69cf9"),
        .init(name: "tokenizer.json", size: 32169626, sha256: "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"),
        .init(name: "tokenizer_config.json", size: 2749, sha256: "c1112569480ce4f86410f29131ff779f8944097f118f5524f4762ca85a32aa5b"),
    ]

    @discardableResult
    static func verify(directory: URL, expected: [File] = files) throws -> String {
        let names = Set(expected.map(\.name))
        guard !expected.isEmpty, names.count == expected.count,
            expected.allSatisfy({ !$0.name.isEmpty && $0.name != "." && $0.name != ".."
                && !$0.name.contains("/") && $0.sha256.count == 64
                && $0.sha256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) })
        else { throw Failure.invalidManifest }
        let contents = try FileManager.default.contentsOfDirectory(at: directory,
            includingPropertiesForKeys: [.isDirectoryKey])
        // hf download --local-dir owns .cache; no other unexpected files or
        // weight shards may silently influence the loader.
        let actual = try contents.filter {
            if $0.lastPathComponent == ".cache" {
                return try $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory != true
            }
            return true
        }
        guard Set(actual.map(\.lastPathComponent)) == names else { throw Failure.unexpectedFiles }
        var aggregate = SHA256()
        for entry in expected.sorted(by: { $0.name < $1.name }) {
            let path = directory.appendingPathComponent(entry.name)
            let stream = try FileHandle(forReadingFrom: path)
            defer { try? stream.close() }
            var hash = SHA256(), count: UInt64 = 0
            while let data = try stream.read(upToCount: 1 << 20), !data.isEmpty {
                count += UInt64(data.count)
                guard count <= entry.size else { throw Failure.mismatchedFile(entry.name) }
                hash.update(data: data)
            }
            let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
            guard count == entry.size, digest == entry.sha256 else { throw Failure.mismatchedFile(entry.name) }
            aggregate.update(data: Data((entry.name + "\0" + digest + "\n").utf8))
        }
        return aggregate.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
