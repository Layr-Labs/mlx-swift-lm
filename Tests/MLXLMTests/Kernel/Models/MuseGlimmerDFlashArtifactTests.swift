// Copyright © 2026 Eigen Labs.

import CryptoKit
import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM

extension KernelTests {

    /// The strict drafter loader and the artifact writer, on tiny models.
    @Suite(.serialized)
    struct MuseGlimmerDFlashArtifactTests {
        typealias Fixture = MuseGlimmerDFlashFixture
        typealias Pin = MuseGlimmerDFlashArtifactPin

        private func temporaryDirectory() throws -> URL {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent(
                "muse-dflash-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            return url
        }

        private func sha256(_ url: URL) throws -> String {
            SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }
                .joined()
        }

        private func bytes(_ model: MuseGlimmerDFlashDraftModel) -> [String: Data] {
            Dictionary(
                uniqueKeysWithValues: model.parameters().flattened().map {
                    ($0.0, $0.1.asData().data)
                })
        }

        /// One block over a short context, as bytes.
        private func probe(_ model: MuseGlimmerDFlashDraftModel) -> Data {
            let c = model.configuration
            let state = model.makeContextState()
            model.commit(
                features: Fixture.features(rows: 9, configuration: c, seed: 31), at: 0, to: state)
            return model.draftBlock(
                embeddings: Fixture.embeddings(configuration: c, seed: 32), state: state
            ).asData().data
        }

        /// A directory with `config.json` for `c` and the given tensors.
        private func artifact(
            _ c: MuseGlimmerDFlashConfiguration, tensors: [String: MLXArray]
        ) throws -> (URL, Pin) {
            let directory = try temporaryDirectory()
            let weights = directory.appendingPathComponent(Pin.weightsFileName)
            try save(arrays: tensors, url: weights)
            let configuration = directory.appendingPathComponent(Pin.configurationFileName)
            try JSONEncoder().encode(c).write(to: configuration)
            return (
                directory,
                Pin(
                    weightsSHA256: try sha256(weights),
                    configurationSHA256: try sha256(configuration), configuration: c)
            )
        }

        @Test func publishedPinIsTheVerifiedRevision() {
            let pin = Pin.museGlimmer30BBF16
            #expect(
                pin.weightsSHA256
                    == "fd88d337eb84f8d0e6ba33a7684d7efa6722d4460ba4d6badca9699418392a84")
            #expect(
                pin.configurationSHA256
                    == "38915167b64b1e6405492aacae5b1b4511b6431163d2960b9bd25821df6fa30a")
            #expect(pin.configuration == .museGlimmer30B)
        }

        @Test func denseArtifactRoundTripsAndIsDeterministic() throws {
            let c = try Fixture.configuration()
            let model = try Fixture.model(c, dtype: .bfloat16)
            let first = try temporaryDirectory()
            let second = try temporaryDirectory()
            defer { [first, second].forEach { try? FileManager.default.removeItem(at: $0) } }
            let written = try model.writeArtifact(to: first, source: "test")
            let again = try model.writeArtifact(to: second, source: "test")
            #expect(written.manifest == again.manifest && written.pin == again.pin)
            for name in [Pin.weightsFileName, Pin.configurationFileName, "manifest.json"] {
                #expect(
                    try Data(contentsOf: first.appendingPathComponent(name))
                        == Data(contentsOf: second.appendingPathComponent(name)))
            }
            #expect(written.manifest.tensorCount == model.parameters().flattened().count)
            #expect(written.manifest.projections.allSatisfy { $0.bits == 0 })
            #expect(written.manifest.pin(configuration: c) == written.pin)
            #expect(
                written.manifest.files[Pin.weightsFileName]?.sha256
                    == (try sha256(first.appendingPathComponent(Pin.weightsFileName))))
            // An existing artifact is never overwritten.
            #expect(throws: MuseGlimmerDFlashError.self) {
                try model.writeArtifact(to: first, source: "test")
            }

            let loaded = try MuseGlimmerDFlashDraftModel.load(
                directory: first, pin: written.pin, maximumBytes: model.artifactPayloadBytes)
            #expect(loaded.activationDType == .bfloat16)
            #expect(bytes(loaded) == bytes(model))
            #expect(probe(loaded) == probe(model))
            #expect(loaded.projectionFormats.values.allSatisfy { $0 == .dense })
        }

        @Test func quantizedArtifactRoundTripsWithFormatsFromTheHeader() throws {
            let c = try Fixture.configuration()
            let model = try Fixture.model(c, dtype: .bfloat16)
            let dense = probe(model)
            var plan: [String: MuseGlimmerDFlashProjectionFormat] = [:]
            for projection in MuseGlimmerDFlashInventory(c).projections {
                if projection.path == "encoder.fc" {
                    plan[projection.path] = .affine(bits: 8, groupSize: 32)
                } else if projection.path.contains(".mlp.") {
                    plan[projection.path] = .affine(bits: 4, groupSize: 32)
                } else if projection.path.hasSuffix("o_proj") {
                    plan[projection.path] = .affine(bits: 6, groupSize: 32)
                }
            }
            try model.quantize(plan)
            let formats = model.projectionFormats
            #expect(formats["encoder.fc"] == .affine(bits: 8, groupSize: 32))
            #expect(formats["layers.1.mlp.down_proj"] == .affine(bits: 4, groupSize: 32))
            #expect(formats["layers.0.self_attn.o_proj"] == .affine(bits: 6, groupSize: 32))
            #expect(formats["layers.0.self_attn.q_proj"] == .dense)
            let quantized = probe(model)
            #expect(quantized != dense)
            // A quantized projection is never quantized again.
            #expect(throws: MuseGlimmerDFlashError.self) {
                try model.quantize(["encoder.fc": .affine(bits: 4, groupSize: 32)])
            }
            #expect(throws: MuseGlimmerDFlashError.self) {
                try model.quantize(["layers.9.mlp.up_proj": .affine(bits: 4, groupSize: 32)])
            }

            let directory = try temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            let written = try model.writeArtifact(to: directory, source: "test")
            #expect(written.manifest.totalBytes < 2 << 30)
            let loaded = try MuseGlimmerDFlashDraftModel.load(
                directory: directory, pin: written.pin, maximumBytes: model.artifactPayloadBytes)
            #expect(loaded.projectionFormats == formats)
            #expect(bytes(loaded) == bytes(model))
            #expect(probe(loaded) == quantized)
        }

        @Test func loaderRefusesWrongHashesAndConfigurations() throws {
            let c = try Fixture.configuration()
            let model = try Fixture.model(c, dtype: .bfloat16)
            let directory = try temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            let written = try model.writeArtifact(to: directory, source: "test")
            let budget = model.artifactPayloadBytes
            let zeros = String(repeating: "0", count: 64)

            var pin = written.pin
            pin.weightsSHA256 = zeros
            #expect(throws: MuseGlimmerDFlashError.checksum("model.safetensors")) {
                try MuseGlimmerDFlashDraftModel.load(
                    directory: directory, pin: pin, maximumBytes: budget)
            }
            pin = written.pin
            pin.configurationSHA256 = zeros
            #expect(throws: MuseGlimmerDFlashError.checksum("config.json")) {
                try MuseGlimmerDFlashDraftModel.load(
                    directory: directory, pin: pin, maximumBytes: budget)
            }
            // The right bytes for a configuration the caller did not pin.
            pin = written.pin
            pin.configuration = try Fixture.configuration(window: 13)
            #expect(throws: MuseGlimmerDFlashError.self) {
                try MuseGlimmerDFlashDraftModel.load(
                    directory: directory, pin: pin, maximumBytes: budget)
            }
            #expect(throws: MuseGlimmerDFlashError.budget) {
                try MuseGlimmerDFlashDraftModel.load(
                    directory: directory, pin: written.pin, maximumBytes: budget - 1)
            }
            // The published pin never accepts another artifact.
            #expect(throws: MuseGlimmerDFlashError.self) {
                try MuseGlimmerDFlashDraftModel.load(
                    directory: directory, pin: .museGlimmer30BBF16, maximumBytes: budget)
            }
            // A symbolic link is not the pinned file.
            let linked = try temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: linked) }
            for name in [Pin.weightsFileName, Pin.configurationFileName] {
                try FileManager.default.createSymbolicLink(
                    at: linked.appendingPathComponent(name),
                    withDestinationURL: directory.appendingPathComponent(name))
            }
            #expect(throws: MuseGlimmerDFlashError.self) {
                try MuseGlimmerDFlashDraftModel.load(
                    directory: linked, pin: written.pin, maximumBytes: budget)
            }
            #expect(throws: MuseGlimmerDFlashError.self) {
                try MuseGlimmerDFlashDraftModel.load(
                    directory: directory.appendingPathComponent("missing"), pin: written.pin,
                    maximumBytes: budget)
            }
        }

        @Test func loaderRefusesIncompleteExtraAndMisshapenTensors() throws {
            let c = try Fixture.configuration()
            let model = try Fixture.model(c, dtype: .bfloat16)
            let good = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            let budget = model.artifactPayloadBytes * 2

            var missing = good
            missing.removeValue(forKey: "layers.1.self_attn.q_norm.weight")
            var extra = good
            extra["embed_tokens.weight"] = MLXArray.zeros([4, c.hiddenSize], dtype: .bfloat16)
            var misshapen = good
            misshapen["layers.0.mlp.up_proj.weight"] = MLXArray.zeros(
                [c.hiddenSize, c.intermediateSize], dtype: .bfloat16)
            var mixed = good
            mixed["norm.weight"] = good["norm.weight"]!.asType(.float32)
            var transposedNorm = good
            transposedNorm["norm.weight"] = good["norm.weight"]!.reshaped([1, c.hiddenSize])

            for (name, tensors) in [
                ("good", good), ("missing", missing), ("extra", extra),
                ("misshapen", misshapen), ("mixed", mixed), ("transposedNorm", transposedNorm),
            ] {
                let (directory, pin) = try artifact(c, tensors: tensors)
                defer { try? FileManager.default.removeItem(at: directory) }
                if name == "good" {
                    let loaded = try MuseGlimmerDFlashDraftModel.load(
                        directory: directory, pin: pin, maximumBytes: budget)
                    #expect(bytes(loaded) == bytes(model))
                } else {
                    #expect(throws: MuseGlimmerDFlashError.self, "\(name)") {
                        try MuseGlimmerDFlashDraftModel.load(
                            directory: directory, pin: pin, maximumBytes: budget)
                    }
                }
            }
        }
    }
}
