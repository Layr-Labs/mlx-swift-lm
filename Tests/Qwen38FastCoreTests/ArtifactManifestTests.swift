import Foundation
import Testing

@testable import Qwen38FastCore

@Suite("Qwen 3.8 pinned artifact manifest")
struct ArtifactManifestTests {
    @Test("production manifest pins every source and artifact revision")
    func pinsEveryRevision() {
        let manifest = Qwen38ArtifactManifest.production

        #expect(manifest.swiftBaseRevision == "ab73a827c9dde6f8802507003aa0be71605aab8e")
        #expect(manifest.mlxSwiftRevision == "606d28cfa8c1d66b2975d3162a4aac9756835c5f")
        #expect(manifest.mlxRevision == "0a725e3000edabc4911cde345270ca950bfa152f")
        #expect(manifest.mtplxSourceRevision == "26e27b78d5299dcafb319844283ac50a137bfee5")
        #expect(manifest.yukonSourceRevision == "eb5eadc7a165047d4321ce883b9ff30894d8bd19")
        #expect(manifest.dflash2SourceRevision == "c5b76ddb62bdefb6eeef1282641842edcf23a1b8")
        #expect(manifest.targetArtifact.repository == "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed")
        #expect(manifest.targetArtifact.revision == "123db8bcc7101455b00d9aad36c0e760c6e7de02")
        #expect(manifest.draftArtifact.repository == "z-lab/Qwen3.8-27B-DFlash2")
        #expect(manifest.draftArtifact.revision == "50307d4c4cde6860d4eee73e2547cd786fe8e8a4")
        #expect(
            manifest.targetArtifact.configSHA256s
                == Set([
                    "533e833dedb9e7b6a8ee22ab4f2fc034bcf6ded9d8693e5ebcc9d5f159b62a3b",
                    "913b57d11eb131d9e1bb7316ea8729b6bcd110b59f8a08840a83ba2e524f370d",
                ]))
        #expect(manifest.targetArtifact.requiredFileSHA256.count == 10)
        #expect(
            manifest.targetArtifact.requiredFileSHA256["model-00001-of-00004.safetensors"]
                == "aed435f4011667af7772fee7ccec90c86f5a6c85649ef74fe6e397a7504b4578")
        #expect(
            manifest.targetArtifact.requiredFileSHA256["tokenizer.json"]
                == "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523")
        #expect(
            manifest.draftArtifact.configSHA256s
                == Set(["873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980"]))
        #expect(
            manifest.draftArtifact.requiredFileSHA256["model.safetensors"]
                == "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c")
    }

    @Test("artifact validator hashes bytes and rejects a mismatch")
    func artifactValidation() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: UUID().uuidString, directoryHint: .isDirectory)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("abc".utf8).write(to: directory.appending(component: "config.json"))
        try Data("weights".utf8).write(
            to: directory.appending(component: "model.safetensors"))

        let reference = Qwen38ArtifactReference(
            repository: "fixture/model",
            revision: "fixture-revision",
            configSHA256s: [
                "0000000000000000000000000000000000000000000000000000000000000000",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ],
            requiredFileSHA256: [
                "model.safetensors":
                    "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"
            ])
        let configSHA256 = try Qwen38ArtifactValidator.validate(
            directory: directory, reference: reference)
        #expect(
            configSHA256
                == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

        try Data("changed config".utf8).write(
            to: directory.appending(component: "config.json"))
        #expect(throws: Qwen38ArtifactValidationError.self) {
            try Qwen38ArtifactValidator.validate(directory: directory, reference: reference)
        }
        try Data("abc".utf8).write(to: directory.appending(component: "config.json"))

        try Data("changed".utf8).write(
            to: directory.appending(component: "model.safetensors"))
        #expect(throws: Qwen38ArtifactValidationError.self) {
            try Qwen38ArtifactValidator.validate(directory: directory, reference: reference)
        }
    }
}
