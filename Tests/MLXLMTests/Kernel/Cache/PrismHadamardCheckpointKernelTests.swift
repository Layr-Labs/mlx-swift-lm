import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

extension KernelTests {

    /// Kernel tests of the Prism Hadamard checkpoint in
    /// `PrismHadamardCheckpoint.swift`: the tensor checks of
    /// `PrismHadamardCheckpoint.init`, the module install and the exact
    /// transforms of the installed modules.
    ///
    /// The synthetic checkpoint has 2 modules, `model.embed_tokens` and
    /// `lm_head`, each with 4 rows and width 512 (one Hadamard block). Each
    /// packed 2-bit weight is 0, each scale is 1 and row r has bias r + 1,
    /// so the dequantized row r has the value r + 1 at all 512 columns.
    @Suite
    struct PrismHadamardCheckpointKernelTests {

        static let width = 512
        static let rows = 4
        /// Sign i is -1 when i is a multiple of 3, else 1.
        static let signs: [Float] = (0 ..< 512).map { $0 % 3 == 0 ? -1 : 1 }

        static let normalizerDetail =
            "schema-2 native state requires the published FP32 normalizers"
        static let manifestDetail = "transform manifest disagrees with tensors"
        static let packedDetail = "packed tensor shape/dtype mismatch"

        // MARK: - Fixtures

        /// The packed artifact declaration, as `config.json` holds it.
        static func configurationJSON(blocks: [Int] = [512, 512]) -> [String: Any] {
            [
                "schema_version": 2, "model_type": "prism_hadamard_qwen35",
                "base_model_type": "qwen3_5", "gdn_activation_layout": "grouped",
                "tensor_namespace": "mlx-vlm-qwen3_5", "hadamard_config": "hadamard.json",
                "components": ["text": true, "vision": false, "mtp": false],
                "quantization": ["bits": 2, "group_size": 128, "mode": "affine"],
                "text_config": ["num_experts": 0],
                "modules": [
                    [
                        "path": "model.embed_tokens", "block": blocks[0], "embedding": true,
                        "dtype": "float16",
                    ],
                    [
                        "path": "lm_head", "block": blocks[1], "embedding": false,
                        "dtype": "float16",
                    ],
                ],
            ]
        }

        static func decodeConfiguration(_ value: [String: Any]) throws
            -> PrismHadamardCheckpointConfiguration
        {
            try JSONDecoder().decode(
                PrismHadamardCheckpointConfiguration.self,
                from: JSONSerialization.data(withJSONObject: value))
        }

        /// The `hadamard.json` transform metadata.
        static func metadata(_ edit: (inout [String: Any]) -> Void = { _ in }) -> [String: Any] {
            var value: [String: Any] = [
                "prism.hadamard.version": 1, "prism.hadamard.block_size": 512,
                "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
                "prism.hadamard.axis": "input-last-dimension",
                "prism.hadamard.sign_mode": "explicit",
                "prism.hadamard.weight_names": ["language_model.lm_head.weight"],
                "prism.hadamard.inverse_weight_names": ["language_model.model.embed_tokens.weight"],
                "prism.hadamard.sign_widths": [512],
                "prism.hadamard.sign_values": signs.map(Double.init),
                "prism.hadamard.gdn_v_grouped": true,
            ]
            edit(&value)
            return value
        }

        /// The packed tensors of both modules and two float32 normalizers.
        static func weights() -> [String: MLXArray] {
            var weights: [String: MLXArray] = [
                "language_model.model.layers.0.input_layernorm.weight": MLXArray.ones([8]),
                "language_model.model.layers.0.post_attention_layernorm.weight": MLXArray.ones([8]),
            ]
            let biases = MLXArray((0 ..< 16).map { Float($0 / 4 + 1) }, [4, 4]).asType(.float16)
            for path in ["model.embed_tokens", "lm_head"] {
                let name = "language_model.\(path)"
                weights[name + ".weight"] = MLXArray.zeros([4, 32], dtype: .uint32)
                weights[name + ".scales"] = MLXArray.ones([4, 4], dtype: .float16)
                weights[name + ".biases"] = biases
                weights[name + ".signs"] = MLXArray(signs)
            }
            return weights
        }

        /// A new directory that holds `hadamard.json`. The caller removes it.
        static func directory(metadata: [String: Any]) throws -> URL {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent(
                "prism-checkpoint-kernel-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: metadata).write(
                to: url.appendingPathComponent("hadamard.json"))
            return url
        }

        /// The detail of a `PrismCheckpointError`, or a description of
        /// another outcome.
        static func failure(_ body: () throws -> Void) -> String {
            do {
                try body()
                return "no error"
            } catch PrismCheckpointError.invalid(let detail) {
                return detail
            } catch {
                return "other error: \(error)"
            }
        }

        // MARK: - Valid checkpoint

        /// A valid checkpoint passes the checks and installs both modules.
        ///
        /// The embedding returns `(x H) * signs` for the dequantized row x. A
        /// row of equal values v gives `v * sqrt(512)` at column 0 and 0 at
        /// the other columns, then sign 0 (-1) applies. The linear layer
        /// computes `((x * signs) H) W^T`. With x = signs, `x * signs` is all
        /// ones, so output o is `(o + 1) * sqrt(512)`.
        @Test func validCheckpointInstallsModulesWithExactTransforms() throws {
            let directory = try Self.directory(metadata: Self.metadata())
            defer { try? FileManager.default.removeItem(at: directory) }
            let configuration = try Self.decodeConfiguration(Self.configurationJSON())
            #expect(!configuration.hasVision)
            let weights = Self.weights()
            let checkpoint = try PrismHadamardCheckpoint(
                directory: directory, configuration: configuration, weights: weights)
            #expect(checkpoint.transforms.blockSize == 512)
            #expect(checkpoint.transforms.gdnVGrouped)
            #expect(
                checkpoint.configuration.modules.map(\.path) == ["model.embed_tokens", "lm_head"])

            let model = PrismCheckpointKernelRoot()
            try checkpoint.install(model: model, weights: weights)
            let embedding = try #require(
                model.languageModel.model.embedTokens as? HadamardQuantizedEmbedding)
            let head = try #require(model.languageModel.lmHead as? HadamardQuantizedLinear)
            #expect(embedding.shape.0 == 4 && embedding.shape.1 == 512)
            #expect(embedding.bits == 2 && embedding.groupSize == 128)
            #expect(head.transform.width == 512 && head.transform.blockSize == 512)

            let root = Float(512).squareRoot()
            // Tokens 2 and 0 have biases 3 and 1.
            let rows = embedding(MLXArray([Int32(2), 0], [1, 2]))
            #expect(rows.shape == [1, 2, 512])
            #expect(rows.dtype == .float16)
            let values = rows.asType(.float32).asArray(Float.self)
            // Tolerance 0.05: the result is float16, with a spacing of
            // 0.0625 between 64 and 128.
            #expect(abs(values[0] + 3 * root) < 0.05)
            #expect(abs(values[512] + root) < 0.05)
            // Exact: the transform of equal values adds and subtracts equal
            // numbers, so the other columns are exactly 0.
            let others = values.enumerated().filter { $0.offset % 512 != 0 }.map(\.element)
            #expect(others.allSatisfy { $0 == 0 })

            let output = head(MLXArray(Self.signs, [1, 512]))
            #expect(output.shape == [1, 4])
            let outputValues = output.asType(.float32).asArray(Float.self)
            // Tolerance 0.05: covers a float16 output (spacing 0.0625 near 90).
            for (index, value) in outputValues.enumerated() {
                #expect(abs(value - Float(index + 1) * root) < 0.05, "output \(index)")
            }
        }

        /// The metadata file can be a symbolic link, as in a Hugging Face
        /// snapshot. Signs in float16 are accepted.
        @Test func symbolicLinkMetadataAndFloat16SignsAreAccepted() throws {
            let directory = try Self.directory(metadata: Self.metadata())
            defer { try? FileManager.default.removeItem(at: directory) }
            let file = directory.appendingPathComponent("hadamard.json")
            let blob = directory.appendingPathComponent("blob-metadata")
            try FileManager.default.moveItem(at: file, to: blob)
            try FileManager.default.createSymbolicLink(at: file, withDestinationURL: blob)
            var weights = Self.weights()
            weights["language_model.lm_head.signs"] = MLXArray(Self.signs).asType(.float16)
            let configuration = try Self.decodeConfiguration(Self.configurationJSON())
            #expect(
                Self.failure {
                    _ = try PrismHadamardCheckpoint(
                        directory: directory, configuration: configuration, weights: weights)
                } == "no error")
        }

        // MARK: - Error paths

        /// A missing, empty, oversized or non-regular metadata file fails.
        @Test func unsafeMetadataFilesAreRejected() throws {
            let configuration = try Self.decodeConfiguration(Self.configurationJSON())
            let weights = Self.weights()
            let directory = try Self.directory(metadata: Self.metadata())
            defer { try? FileManager.default.removeItem(at: directory) }
            let file = directory.appendingPathComponent("hadamard.json")
            let load = {
                _ = try PrismHadamardCheckpoint(
                    directory: directory, configuration: configuration, weights: weights)
            }

            try FileManager.default.removeItem(at: file)
            #expect(Self.failure(load).hasPrefix("other error"), "missing file")

            try Data().write(to: file)
            #expect(Self.failure(load) == "unsafe transform metadata", "empty file")

            try Data(count: 8 * 1024 * 1024 + 1).write(to: file)
            #expect(Self.failure(load) == "unsafe transform metadata", "oversized file")

            try FileManager.default.removeItem(at: file)
            try FileManager.default.createDirectory(at: file, withIntermediateDirectories: false)
            #expect(Self.failure(load) == "unsafe transform metadata", "directory")
        }

        /// A transform width that the metadata does not declare fails with
        /// the transform error of MLXNN.
        @Test func undeclaredTransformWidthIsRejected() throws {
            let directory = try Self.directory(
                metadata: Self.metadata { value in
                    value["prism.hadamard.sign_widths"] = [1024]
                    value["prism.hadamard.sign_values"] = Array(repeating: 1.0, count: 1024)
                })
            defer { try? FileManager.default.removeItem(at: directory) }
            let configuration = try Self.decodeConfiguration(Self.configurationJSON())
            var missingWidth: Int?
            do {
                _ = try PrismHadamardCheckpoint(
                    directory: directory, configuration: configuration, weights: Self.weights())
            } catch HadamardError.missingSignWidth(let width) {
                missingWidth = width
            }
            #expect(missingWidth == 512)
        }

        /// Each change to a valid checkpoint fails with the expected detail.
        /// All changes are on `lm_head` or global, so the embedding (checked
        /// first) stays valid.
        @Test func invalidTensorsAndManifestsAreRejected() throws {
            let lmHead = "language_model.lm_head"
            var flipped = Self.signs
            flipped[5] = -flipped[5]
            let cases: [PrismCheckpointKernelCase] = [
                .init(
                    name: "no normalizers", detail: Self.normalizerDetail,
                    weights: {
                        weights in weights = weights.filter { !$0.key.contains("layernorm") }
                    }),
                .init(
                    name: "bfloat16 normalizer", detail: Self.normalizerDetail,
                    weights: {
                        $0["language_model.model.layers.0.input_layernorm.weight"] =
                            MLXArray.ones([8], dtype: .bfloat16)
                    }),
                .init(
                    name: "tiled GDN values", detail: Self.manifestDetail,
                    metadata: { $0["prism.hadamard.gdn_v_grouped"] = false }),
                .init(
                    name: "other forward names", detail: Self.manifestDetail,
                    metadata: {
                        $0["prism.hadamard.weight_names"] = ["language_model.model.other.weight"]
                    }),
                .init(
                    name: "no inverse names", detail: Self.manifestDetail,
                    metadata: { $0["prism.hadamard.inverse_weight_names"] = [String]() }),
                .init(
                    name: "extra signs", detail: Self.manifestDetail,
                    weights: {
                        $0["language_model.model.extra.signs"] = MLXArray(Self.signs)
                    }),
                .init(
                    name: "missing signs", detail: Self.manifestDetail,
                    weights: {
                        $0[lmHead + ".signs"] = nil
                    }),
                .init(
                    name: "mtp prefix", detail: Self.manifestDetail,
                    weights: {
                        $0["mtp.fc.weight"] = MLXArray.zeros([1])
                    }),
                .init(
                    name: "mtp infix", detail: Self.manifestDetail,
                    weights: {
                        $0["language_model.model.mtp.norm.weight"] = MLXArray.zeros([1])
                    }),
                .init(
                    name: "float32 scales", detail: Self.packedDetail,
                    weights: {
                        $0[lmHead + ".scales"] = MLXArray.ones([4, 4], dtype: .float32)
                    }),
                .init(
                    name: "missing biases", detail: Self.packedDetail,
                    weights: {
                        $0[lmHead + ".biases"] = nil
                    }),
                .init(
                    name: "3-D weight", detail: Self.packedDetail,
                    weights: {
                        $0[lmHead + ".weight"] = MLXArray.zeros([1, 4, 32], dtype: .uint32)
                    }),
                .init(
                    name: "int32 signs", detail: Self.packedDetail,
                    weights: {
                        $0[lmHead + ".signs"] = MLXArray(Self.signs).asType(.int32)
                    }),
                .init(
                    name: "short signs", detail: Self.packedDetail,
                    weights: {
                        $0[lmHead + ".signs"] = MLXArray(Array(Self.signs[0 ..< 256]))
                    }),
                .init(name: "block 1024", detail: Self.packedDetail, blocks: [512, 1024]),
                .init(
                    name: "flipped sign", detail: "tensor signs disagree with metadata",
                    weights: {
                        $0[lmHead + ".signs"] = MLXArray(flipped)
                    }),
            ]
            for item in cases {
                let directory = try Self.directory(metadata: Self.metadata(item.metadata))
                defer { try? FileManager.default.removeItem(at: directory) }
                let configuration = try Self.decodeConfiguration(
                    Self.configurationJSON(blocks: item.blocks))
                var weights = Self.weights()
                item.weights(&weights)
                let detail = Self.failure {
                    _ = try PrismHadamardCheckpoint(
                        directory: directory, configuration: configuration, weights: weights)
                }
                #expect(detail == item.detail, "\(item.name)")
            }
        }

        /// `install` fails when a declared module is missing or has other
        /// dimensions, and then it changes no module.
        @Test func installRejectsMissingAndMismatchedModules() throws {
            let directory = try Self.directory(metadata: Self.metadata())
            defer { try? FileManager.default.removeItem(at: directory) }
            let configuration = try Self.decodeConfiguration(Self.configurationJSON())
            let weights = Self.weights()
            let checkpoint = try PrismHadamardCheckpoint(
                directory: directory, configuration: configuration, weights: weights)

            var noScales = weights
            noScales["language_model.model.embed_tokens.scales"] = nil
            let cases: [(String, PrismCheckpointKernelRoot, [String: MLXArray], String)] = [
                (
                    "no lm_head", PrismCheckpointKernelRoot(head: nil), weights,
                    "declared module is missing"
                ),
                ("no scales", PrismCheckpointKernelRoot(), noScales, "declared module is missing"),
                (
                    "head rows", PrismCheckpointKernelRoot(head: Linear(512, 8, bias: false)),
                    weights,
                    "linear dimensions differ from model"
                ),
                (
                    "head width", PrismCheckpointKernelRoot(head: Linear(256, 4, bias: false)),
                    weights, "linear dimensions differ from model"
                ),
                (
                    "embedding rows", PrismCheckpointKernelRoot(vocabulary: 8), weights,
                    "embedding dimensions differ from model"
                ),
                (
                    "embedding width", PrismCheckpointKernelRoot(embeddingWidth: 256), weights,
                    "embedding dimensions differ from model"
                ),
            ]
            for (name, model, installWeights, expected) in cases {
                #expect(
                    Self.failure { try checkpoint.install(model: model, weights: installWeights) }
                        == expected, "\(name)")
                #expect(
                    !(model.languageModel.model.embedTokens is HadamardQuantizedEmbedding),
                    "\(name)")
                #expect(!(model.languageModel.lmHead is HadamardQuantizedLinear), "\(name)")
            }
        }

        /// Declarations outside the supported packing contract are rejected.
        /// The existing `PrismHadamardCheckpointTests` cover the schema
        /// version, base type, layout, metadata name, bit width, MTP flag and
        /// MTP layers, and duplicate or `..` paths.
        @Test func configurationRejectsOtherContracts() throws {
            func rejects(_ edit: (inout [String: Any]) -> Void) -> Bool {
                var value = Self.configurationJSON()
                edit(&value)
                return (try? Self.decodeConfiguration(value)) == nil
            }
            func module(
                _ path: String, block: Int = 512, embedding: Bool, dtype: String = "float16"
            )
                -> [String: Any]
            {
                ["path": path, "block": block, "embedding": embedding, "dtype": dtype]
            }
            let embed = module("model.embed_tokens", embedding: true)
            let invalid: [(String, (inout [String: Any]) -> Void)] = [
                ("model type", { $0["model_type"] = "qwen3_5" }),
                ("namespace", { $0["tensor_namespace"] = "mlx-lm-qwen3_5" }),
                ("experts", { $0["text_config"] = ["num_experts": 8] }),
                ("no text", { $0["components"] = ["text": false, "vision": false, "mtp": false] }),
                (
                    "group size",
                    { $0["quantization"] = ["bits": 2, "group_size": 64, "mode": "affine"] }
                ),
                ("mode", { $0["quantization"] = ["bits": 2, "group_size": 128, "mode": "mxfp4"] }),
                ("no modules", { $0["modules"] = [[String: Any]]() }),
                (
                    "block 256",
                    { $0["modules"] = [embed, module("lm_head", block: 256, embedding: false)] }
                ),
                (
                    "bfloat16",
                    {
                        $0["modules"] = [
                            embed, module("lm_head", embedding: false, dtype: "bfloat16"),
                        ]
                    }
                ),
                (
                    "embedding flag",
                    { $0["modules"] = [embed, module("lm_head", embedding: true)] }
                ),
                ("no embedding", { $0["modules"] = [module("lm_head", embedding: false)] }),
                (
                    "empty part",
                    { $0["modules"] = [embed, module("model..proj", embedding: false)] }
                ),
                ("dash", { $0["modules"] = [embed, module("lm-head", embedding: false)] }),
            ]
            for (name, edit) in invalid {
                #expect(rejects(edit), "\(name)")
            }
            // The 4 block sizes and a missing MTP layer count are accepted.
            for block in [512, 1024, 2048, 4096] {
                #expect(
                    !rejects {
                        $0["modules"] = [embed, module("lm_head", block: block, embedding: false)]
                    })
            }
            #expect(
                PrismCheckpointError.invalid("x").errorDescription
                    == "Invalid Prism Hadamard checkpoint: x")
        }
    }

    /// Kernel tests of the packed prefill carry scope in
    /// `PrismHadamardPrefillCarry.swift`, with contiguous `CBv2LayerCache`
    /// caches. The existing `PrismPrefillCarrySubmissionTests` use paged
    /// caches and run only on the exclusive GPU lane.
    @Suite(.enabled(if: PrismHadamardPrefillCarry.enabled))
    struct PrismHadamardPrefillCarryKernelTests {

        static func cache(rows: Int) -> CBv2LayerCache {
            let kind = CBv2LayerKind(attention: .full, headDim: 8, kvHeads: 1, queryHeads: 1)
            let sequences: [any CBv2SequenceKV] = (0 ..< rows).map { _ in
                CBv2FullSequenceKV(promptLength: 0, maxLength: 16, kvHeads: 1, headDim: 8)
            }
            return CBv2LayerCache(layerIndex: 0, kind: kind, rows: sequences)
        }

        static func scope<T>(
            batch: Int = 1, caches: [any CBv2AttendingLayerCache], body: () -> T
        ) -> T {
            PrismHadamardPrefillCarry.withScope(
                isPacked: true, batch: batch, width: 128, hasEmbeddings: false,
                hasPositions: false, capturesState: false, caches: caches, body: body)
        }

        /// Only a packed prefill of width 128 or more, with no embeddings,
        /// positions or state capture and with caches, opens a context. The
        /// scope returns the body value and does not leak the context.
        @Test func scopeOpensAContextOnlyForEligiblePrefill() {
            let caches: [any CBv2AttendingLayerCache] = [Self.cache(rows: 2)]
            let cases:
                [(
                    packed: Bool, batch: Int, width: Int, embeds: Bool, positions: Bool,
                    captures: Bool, hasCaches: Bool, eligible: Bool
                )] = [
                    (true, 2, 128, false, false, false, true, true),
                    (true, 2, 4096, false, false, false, true, true),
                    (false, 2, 128, false, false, false, true, false),
                    (true, 0, 128, false, false, false, true, false),
                    (true, 2, 127, false, false, false, true, false),
                    (true, 2, 128, true, false, false, true, false),
                    (true, 2, 128, false, true, false, true, false),
                    (true, 2, 128, false, false, true, true, false),
                    (true, 2, 128, false, false, false, false, false),
                ]
            for item in cases {
                let seen = PrismHadamardPrefillCarry.withScope(
                    isPacked: item.packed, batch: item.batch, width: item.width,
                    hasEmbeddings: item.embeds, hasPositions: item.positions,
                    capturesState: item.captures, caches: item.hasCaches ? caches : []
                ) { PrismHadamardPrefillCarry.context.map { [$0.batch, $0.caches.count] } }
                let expected: [Int]? = item.eligible ? [item.batch, 1] : nil
                #expect(seen == expected, "\(item)")
                #expect(PrismHadamardPrefillCarry.context?.batch == nil)
            }

            // A nested ineligible scope hides the outer context only inside it.
            Self.scope(batch: 2, caches: caches) {
                #expect(PrismHadamardPrefillCarry.context?.permitsSubmission == true)
                PrismHadamardPrefillCarry.withScope(
                    isPacked: false, batch: 2, width: 128, hasEmbeddings: false,
                    hasPositions: false, capturesState: false, caches: caches
                ) {
                    #expect(PrismHadamardPrefillCarry.context?.batch == nil)
                }
                #expect(PrismHadamardPrefillCarry.context?.batch == 2)
            }
        }

        /// `submit` accepts only a non-empty rank-3 carry with the scope batch
        /// inside a context whose caches all have that many rows. A submitted
        /// carry keeps its values.
        @Test func submitChecksTheCarryAndTheCacheRows() {
            let carry = MLXArray(0 ..< 12).asType(.float32).reshaped(1, 3, 4)
            let diagnose =
                ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_PREFILL_CARRY_DIAGNOSTICS"]
                == "1"
            let before = PrismHadamardPrefillCarry.submittedCount
            #expect(!PrismHadamardPrefillCarry.submit(carry), "no scope")

            let results = Self.scope(caches: [Self.cache(rows: 1), Self.cache(rows: 1)]) {
                [
                    PrismHadamardPrefillCarry.submit(carry),
                    PrismHadamardPrefillCarry.submit(MLXArray.ones([2, 3, 4])),
                    PrismHadamardPrefillCarry.submit(MLXArray.ones([1, 12])),
                    PrismHadamardPrefillCarry.submit(MLXArray.ones([1, 0, 4])),
                    PrismHadamardPrefillCarry.submit(MLXArray.ones([1, 3, 0])),
                ]
            }
            #expect(results == [true, false, false, false, false])
            // Exact: an asynchronous evaluation does not change the values.
            #expect(carry.asArray(Float.self) == (0 ..< 12).map(Float.init))

            // A cache with another row count blocks the submission.
            for caches: [any CBv2AttendingLayerCache] in [
                [Self.cache(rows: 0)], [Self.cache(rows: 1), Self.cache(rows: 2)],
            ] {
                let submitted = Self.scope(caches: caches) {
                    (
                        PrismHadamardPrefillCarry.context?.permitsSubmission,
                        PrismHadamardPrefillCarry.submit(carry)
                    )
                }
                #expect(submitted.0 == false)
                #expect(submitted.1 == false)
            }
            if !diagnose {
                // The counter moves only when diagnostics are on.
                #expect(PrismHadamardPrefillCarry.submittedCount == before)
            }
        }
    }
}

/// One invalid change to the valid synthetic checkpoint.
private struct PrismCheckpointKernelCase {
    let name: String
    let detail: String
    var blocks: [Int] = [512, 512]
    var metadata: (inout [String: Any]) -> Void = { _ in }
    var weights: (inout [String: MLXArray]) -> Void = { _ in }
}

/// The module tree `language_model.model.embed_tokens` and
/// `language_model.lm_head` that the checkpoint declares.
private final class PrismCheckpointKernelBody: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    init(vocabulary: Int, width: Int) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: vocabulary, dimensions: width)
        super.init()
    }
}

private final class PrismCheckpointKernelLanguageModel: Module {
    @ModuleInfo(key: "model") var model: PrismCheckpointKernelBody
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(body: PrismCheckpointKernelBody, head: Linear?) {
        _model.wrappedValue = body
        _lmHead.wrappedValue = head
        super.init()
    }
}

private final class PrismCheckpointKernelRoot: Module {
    @ModuleInfo(key: "language_model") var languageModel: PrismCheckpointKernelLanguageModel

    init(
        vocabulary: Int = 4, embeddingWidth: Int = 512,
        head: Linear? = Linear(512, 4, bias: false)
    ) {
        _languageModel.wrappedValue = PrismCheckpointKernelLanguageModel(
            body: PrismCheckpointKernelBody(vocabulary: vocabulary, width: embeddingWidth),
            head: head)
        super.init()
    }
}
