// Copyright © 2026 Eigen Labs. SPDX-License-Identifier: Apache-2.0

import CoreGraphics
import CoreImage
import CoreText
import Foundation
import MLX
import MLXHuggingFace
import Testing
import Tokenizers

@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM

/// Explicit offline real-checkpoint image correctness. Runs only when
/// `GLM53_REAL_VISION=1`, `GLM53_REAL_MODEL_DIR` names the staged artifact and
/// `GLM53_REAL_PROFILE_DIR` holds its `identity.json`; downloads nothing.
///
/// Images are generated in-process (shapes, counting, rendered text, a bar
/// chart, two interleaved images, one 8000-token image), so their content is
/// known exactly. Each case is answered greedily through:
/// - the direct multimodal forward (`GLM5NextModel.forward(prepared)`), and
/// - EngineV2 on the native paged backend with the `.serving` policy, the
///   provider's step/chunk bounds and a concurrent text request.
/// The vision tower runs one image at a time with query-tiled attention.
///
/// Optional:
/// - `GLM53_VISION_REFERENCE_DIR`: output of `make_vision_reference.py` (official
///   Transformers processor + tower on the same artifact's BF16 weights).
///   Requires exact processor pixels, and tower features within the official
///   BF16 tower's own error against its FP32 run.
/// - `GLM53_REAL_STORAGE`: `nvfp4` (default) or `q4`.
/// - `GLM53_REAL_VISION_TOKENS`: output budget per answer (default 384).
/// - `GLM53_REAL_VISION_LARGE=0`: skip the 8000-token image.
extension KernelTests {
    @Suite(.serialized)
    struct GLM5NextRealVisionTests {
        private struct Identity: Decodable {
            struct File: Decodable {
                let bytes: Int
                let sha256: String
            }
            let repository: String
            let revision: String
            let files: [String: File]
        }

        private struct Case {
            let name: String
            /// Ordered message parts: text or an image index into `images`.
            let parts: [Part]
            let images: [CIImage]
            /// Every group must have at least one member in the answer.
            let expected: [[String]]
            var large = false
            enum Part {
                case text(String)
                case image
            }
        }

        private static var environment: [String: String] { ProcessInfo.processInfo.environment }
        private static var enabled: Bool {
            environment["GLM53_REAL_VISION"] == "1" && environment["GLM53_REAL_MODEL_DIR"] != nil
                && environment["GLM53_REAL_PROFILE_DIR"] != nil
        }

        // MARK: Generated images

        private static func canvas(
            width: Int, height: Int, draw: (CGContext) -> Void
        ) -> CIImage {
            let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            draw(context)
            return CIImage(cgImage: context.makeImage()!)
        }

        private static func text(
            _ string: String, at point: CGPoint, size: CGFloat, in context: CGContext
        ) {
            let font = CTFontCreateWithName("Helvetica-Bold" as CFString, size, nil)
            let attributes: [NSAttributedString.Key: Any] = [
                NSAttributedString.Key(kCTFontAttributeName as String): font,
                NSAttributedString.Key(kCTForegroundColorAttributeName as String):
                    CGColor(red: 0, green: 0, blue: 0, alpha: 1),
            ]
            let line = CTLineCreateWithAttributedString(
                NSAttributedString(string: string, attributes: attributes))
            context.textPosition = point
            CTLineDraw(line, context)
        }

        private static let red = CGColor(red: 0.86, green: 0.1, blue: 0.1, alpha: 1)
        private static let blue = CGColor(red: 0.1, green: 0.25, blue: 0.85, alpha: 1)
        private static let green = CGColor(red: 0.1, green: 0.65, blue: 0.2, alpha: 1)

        private static var cases: [Case] {
            let square = canvas(width: 448, height: 448) {
                $0.setFillColor(red)
                $0.fill(CGRect(x: 124, y: 124, width: 200, height: 200))
            }
            let circles = canvas(width: 560, height: 280) { context in
                context.setFillColor(blue)
                for x in [40, 220, 400] {
                    context.fillEllipse(in: CGRect(x: x, y: 70, width: 120, height: 120))
                }
            }
            let words = canvas(width: 700, height: 220) {
                text("DARKBLOOM 42", at: CGPoint(x: 30, y: 80), size: 72, in: $0)
            }
            let chart = canvas(width: 600, height: 420) { context in
                context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
                context.fill(CGRect(x: 40, y: 60, width: 520, height: 3))
                for (index, (label, height)) in [("A", 90), ("B", 180), ("C", 320), ("D", 140)]
                    .enumerated()
                {
                    let x = 70 + index * 125
                    context.setFillColor(blue)
                    context.fill(CGRect(x: x, y: 63, width: 70, height: height))
                    text(label, at: CGPoint(x: x + 22, y: 20), size: 34, in: context)
                }
            }
            let blueSquare = canvas(width: 336, height: 336) {
                $0.setFillColor(blue)
                $0.fill(CGRect(x: 68, y: 68, width: 200, height: 200))
            }
            let redCircle = canvas(width: 336, height: 336) {
                $0.setFillColor(red)
                $0.fillEllipse(in: CGRect(x: 68, y: 68, width: 200, height: 200))
            }
            // Large enough to hit the processor's 8000-token image cap.
            let large = canvas(width: 2800, height: 2800) { context in
                context.setFillColor(green)
                context.move(to: CGPoint(x: 1400, y: 2300))
                context.addLine(to: CGPoint(x: 500, y: 500))
                context.addLine(to: CGPoint(x: 2300, y: 500))
                context.closePath()
                context.fillPath()
            }
            let brief = " Answer with only the answer, no explanation."
            return [
                .init(
                    name: "color", parts: [.image, .text("What color is the square?" + brief)],
                    images: [square], expected: [["red"]]),
                .init(
                    name: "count",
                    parts: [.image, .text("How many circles are in this image?" + brief)],
                    images: [circles], expected: [["3", "three"]]),
                .init(
                    name: "text",
                    parts: [.image, .text("What text is written in this image?" + brief)],
                    images: [words], expected: [["darkbloom"], ["42"]]),
                .init(
                    name: "chart",
                    parts: [
                        .image,
                        .text(
                            "This bar chart has bars labeled A, B, C and D. Which label has the "
                                + "tallest bar? Answer with only the letter."),
                    ], images: [chart], expected: [["c"]]),
                .init(
                    name: "interleaved",
                    parts: [
                        .text("First image:"), .image, .text("Second image:"), .image,
                        .text("Which image shows a red shape, the first or the second?" + brief),
                    ], images: [blueSquare, redCircle], expected: [["second", "2"]]),
                .init(
                    name: "large", parts: [.image, .text("What color is the triangle?" + brief)],
                    images: [large], expected: [["green"]], large: true),
            ]
        }

        private static func answer(_ text: String) -> String {
            let tail = text.components(separatedBy: "</think>").last ?? text
            return tail.lowercased()
        }

        private static func matches(_ text: String, _ expected: [[String]]) -> Bool {
            let answer = Self.answer(text)
            let words = Set(
                answer.components(separatedBy: CharacterSet.alphanumerics.inverted).filter {
                    !$0.isEmpty
                })
            return expected.allSatisfy { group in
                group.contains { $0.count <= 2 ? words.contains($0) : answer.contains($0) }
            }
        }

        // MARK: CBv2 target-only adapter (mirrors the provider's media seam)

        private final class Steppable: CBv2RecurrentSteppableModel,
            CBv2RecurrentPrefillSteppableModel,
            CBv2PositionedMultimodalSteppableModel, CBv2CompleteCheckpointKVTypeProviding,
            CBv2TargetAuxiliaryAllocationProviding
        {
            let target: GLM5NextTextModel
            init(_ target: GLM5NextTextModel) { self.target = target }
            var cbv2Capabilities: CBv2ModelCapabilities { target.cbv2Capabilities }
            var recurrentStateSpec: CBv2RecurrentStateSpec? { target.cbv2RecurrentStateSpec }
            var cbv2CompleteCheckpointKVDTypes: [DType]? { target.cbv2CompleteCheckpointKVDTypes }
            var cbv2TargetAuxiliaryAllocationSpecs: [CBv2AuxiliaryAllocationSpec]? {
                target.cbv2TargetAuxiliaryAllocationSpecs
            }
            var causalPositionRequirement: CBv2CausalPositionRequirement { .scalarCacheOffset }
            func supportsMultimodalPrefill(attention: CBv2MultimodalAttention) -> Bool {
                attention == .causal
            }
            func embedPromptTokens(_ tokens: MLXArray) -> MLXArray { target.embed(tokens) }
            func forward(tokens: MLXArray, caches: [any CBv2AttendingLayerCache]) -> MLXArray {
                preconditionFailure("GLM requires request-owned recurrent state")
            }
            func forward(
                tokens: MLXArray, inputEmbeddings: MLXArray, caches: [any CBv2AttendingLayerCache]
            ) -> MLXArray {
                preconditionFailure("GLM media requires request-owned recurrent state")
            }
            func forward(
                tokens: MLXArray, caches: [any CBv2AttendingLayerCache],
                recurrentState: [CBv2RecurrentStateEvaluation]
            ) -> MLXArray {
                target.cbv2Forward(tokens, caches: kv(caches), recurrentState: recurrentState)
            }
            func recurrentPrefill(
                tokens: MLXArray, inputEmbeddings: MLXArray?,
                caches: [any CBv2AttendingLayerCache],
                recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
                requirement: CBv2PrefillRequirement
            ) -> MLXArray {
                target.cbv2RecurrentPrefill(
                    tokens, inputEmbedding: inputEmbeddings, cache: kv(caches),
                    recurrentState: recurrentState, positionIds: positionIds,
                    requirement: requirement)
            }
            func forward(
                tokens: MLXArray, inputEmbeddings: MLXArray, caches: [any CBv2AttendingLayerCache],
                recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
            ) -> MLXArray {
                do {
                    return try target.prepareCBv2Forward(
                        tokens: tokens, caches: kv(caches), recurrentState: recurrentState,
                        positionIds: positionIds, inputEmbeddings: inputEmbeddings
                    ).logits
                } catch { preconditionFailure("GLM embedding forward failed: \(error)") }
            }
            private func kv(_ caches: [any CBv2AttendingLayerCache]) -> [any KVCache] {
                caches.map { $0 as! any KVCache }
            }
        }

        /// GLM's step-owned pages support causal substitution only.
        private final class CausalBank: CBv2MultimodalAttentionCapabilityProviding,
            CBv2CompositionInvalidating
        {
            let bank: CBv2LayerCacheBank
            init(_ caches: [any CBv2AttendingLayerCache]) {
                bank = CBv2LayerCacheBank(caches: caches)
            }
            var supportsMultimodalSpans: Bool { false }
            var supportsPackedPrefill: Bool { false }
            var supportsPackedMultimodalSpans: Bool { false }
            var supportsMTPRectangularVerification: Bool { false }
            func supportsMultimodalPrefill(attention: CBv2MultimodalAttention) -> Bool {
                attention == .causal
            }
            func layerCaches(rowStates: [[(any CBv2SequenceKV)?]]) -> [any CBv2AttendingLayerCache]
            {
                bank.layerCaches(rowStates: rowStates)
            }
            func invalidateBoundComposition() { bank.invalidateBoundComposition() }
            func releaseBoundRows() { bank.releaseBoundRows() }
        }

        private final class ProcessBudget: CBv2ProcessMemoryOwner, @unchecked Sendable {
            enum Failure: Error { case exhausted, invalidCoverage }
            private let lock = NSLock()
            private var charge: UInt64 = 0
            private var coverage: UInt64 = 0
            func replaceCharge(_ bytes: UInt64) throws {
                try lock.withLock {
                    guard bytes <= (192 << 30) else { throw Failure.exhausted }
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
            func retire() {}
        }

        private static func makeEngine(_ model: GLM5NextTextModel, rows: Int) throws -> EngineV2 {
            let maximumBuffer = MLX.GPU.deviceInfo().maxBufferSize
            let context = 16_384
            let kvBytes = 48 << 30
            // The provider's GLM bounds (EngineV2Factory.glm5NextServingScheduler).
            let backend = try model.makeCBv2PagedBackend(
                kvBytesCapacity: kvBytes, maximumBatchSize: rows, maximumQueryTokens: 512,
                maximumContextTokens: context,
                maximumScratchBytes: min(kvBytes, maximumBuffer),
                segmentBytes: 64 << 20, maximumBufferBytes: maximumBuffer)
            let engine = EngineV2(
                model: Steppable(model), layerKinds: model.cbv2LayerKinds, backend: backend,
                cacheProvider: CausalBank(backend.makeLayerCaches()),
                sampler: CBv2DefaultSampler(),
                schedulerConfig: .init(
                    maxConcurrentRequests: rows, maxBatchedTokensPerStep: 512,
                    prefillChunkSize: 512, maxWaiting: 4 * rows, enablePrefixCache: false),
                admissionConfig: model.cbv2NativeAdmissionConfig,
                processMemoryOwner: ProcessBudget())
            if let reason = engine.pagedAttentionWorkInactiveReason {
                throw CBv2KVError.backendIneligible(reason: reason)
            }
            return engine
        }

        private static func collect(_ stream: AsyncStream<CBv2Event>) async -> (
            tokens: [Int], finish: String
        ) {
            var tokens: [Int] = []
            var finish = "none"
            for await event in stream {
                switch event {
                case .delta(_, let delta, _): tokens += delta
                case .finished(let reason, _): finish = "\(reason)"
                }
            }
            return (tokens, finish)
        }

        // MARK: Test

        @Test(.enabled(if: enabled))
        func realImagesAnswerCorrectlyThroughDirectAndPagedCBv2Paths() async throws {
            let model = URL(
                fileURLWithPath: Self.environment["GLM53_REAL_MODEL_DIR"]!, isDirectory: true)
            let profile = URL(
                fileURLWithPath: Self.environment["GLM53_REAL_PROFILE_DIR"]!, isDirectory: true)
            let identityFile = try JSONDecoder().decode(
                Identity.self,
                from: Data(contentsOf: profile.appendingPathComponent("identity.json")))
            let identity = GLM5NextArtifactIdentity(
                repository: identityFile.repository, revision: identityFile.revision,
                files: identityFile.files.mapValues { .init(bytes: $0.bytes, sha256: $0.sha256) })
            let storage: GLM5NextWeightInventory.Storage =
                Self.environment["GLM53_REAL_STORAGE"] == "q4" ? .vontraQ4 : .nvfp4Hybrid
            let outputTokens = Self.environment["GLM53_REAL_VISION_TOKENS"].flatMap(Int.init) ?? 384
            let runLarge = Self.environment["GLM53_REAL_VISION_LARGE"] != "0"

            let roots = GLM5NextArtifactLoadRoots()
            let artifact = try GLM5NextArtifactLoader(
                directory: model, expectedIdentity: identity, storage: storage,
                constructionRoots: roots, executionPolicy: .serving)
            let glm = try GLM5NextModel(artifact: artifact, loadMTP: false)
            try roots.evaluate(glm.parameters().flattened().map(\.1))
            GLM5NextRealResidency.applyIfRequested(ProcessInfo.processInfo.environment)
            let tokenizer = #adaptHuggingFaceTokenizer(
                try await AutoTokenizer.from(modelFolder: model))
            let processor = try GLM5NextProcessor(configuration: .init(), tokenizer: tokenizer)
            var eos = Set(glm.configuration.textConfig.eosTokenIds?.values ?? [])
            if let data = try? Data(
                contentsOf: model.appendingPathComponent("generation_config.json")),
                let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            {
                if let id = object["eos_token_id"] as? Int { eos.insert(id) }
                if let ids = object["eos_token_id"] as? [Int] { eos.formUnion(ids) }
            }
            print(
                "GLM53_VISION loaded storage=\(storage) eos=\(eos.sorted()) "
                    + "active_gb=\(Double(Memory.activeMemory) / 1e9)")
            var failures: [String] = []

            // Official reference: processor pixels and tower features.
            if let directory = Self.environment["GLM53_VISION_REFERENCE_DIR"] {
                let root = URL(fileURLWithPath: directory, isDirectory: true)
                let reference = try loadArrays(
                    url: root.appendingPathComponent("reference.safetensors"))
                for name in ["shapes", "text"] {
                    let pixels = try #require(reference["\(name).pixel_values"])
                    let grid = try #require(reference["\(name).grid_thw"]).asArray(Int64.self).map(
                        Int.init)
                    let expected = try #require(reference["\(name).features"])
                    let image = try #require(
                        CIImage(contentsOf: root.appendingPathComponent("\(name).png")))
                    let prepared = try await processor.prepareDetailed(
                        input: UserInput(
                            messages: [
                                [
                                    "role": "user",
                                    "content": [["type": "image"], ["type": "text", "text": "x"]],
                                ]
                            ],
                            images: [.ciImage(image)]))
                    let native = try #require(prepared.input.image)
                    let pixelError =
                        native.pixels.shape == pixels.shape
                        ? abs(native.pixels.asType(.float32) - pixels).max().item(Float.self)
                        : .infinity
                    let visionGrid = try GLM5NextVisionGrid(
                        temporal: grid[0], height: grid[1], width: grid[2])
                    // The default (tiled) kernel is the serving kernel; the fused
                    // kernel is also measured so precision can be compared.
                    let features = try glm.visionTower.forward(patches: pixels, grids: [visionGrid])
                        .asType(.float32)
                    let tiled = try glm.visionTower.forward(
                        patches: pixels, grids: [visionGrid], limits: .init(attention: .tiled)
                    ).asType(.float32)
                    let fused = try glm.visionTower.forward(
                        patches: pixels, grids: [visionGrid], limits: .init(attention: .fused)
                    ).asType(.float32)
                    // The stored tower is BF16. The official tower run in BF16 bounds
                    // the precision error a BF16 implementation can be expected to have
                    // against the FP32 reference; low-norm rows dominate the minimum.
                    let officialBF16 = try #require(reference["\(name).features_bf16"])
                    func compare(
                        _ value: MLXArray, against target: MLXArray? = nil, rows: MLXArray? = nil
                    ) -> (minimum: Float, mean: Float, relative: Float) {
                        var value = value
                        var target = target ?? expected
                        if let rows {
                            value = value.take(rows, axis: 0)
                            target = target.take(rows, axis: 0)
                        }
                        let cosine =
                            (value * target).sum(axis: -1)
                            / (sqrt((value * value).sum(axis: -1))
                                * sqrt((target * target).sum(axis: -1)))
                        let relative =
                            sqrt(((value - target) * (value - target)).sum())
                            / sqrt((target * target).sum())
                        return (
                            cosine.min().item(Float.self), cosine.mean().item(Float.self),
                            relative.item(Float.self)
                        )
                    }
                    // Print-only data: rows whose official FP32 norm is at or
                    // above the median, and comparisons against official BF16.
                    let norms = sqrt((expected * expected).sum(axis: -1)).asArray(Float.self)
                    let median = norms.sorted()[norms.count / 2]
                    let strongRows = MLXArray(
                        norms.indices.filter { norms[$0] >= median }.map(Int32.init))
                    func line(_ value: (minimum: Float, mean: Float, relative: Float)) -> String {
                        "min_cos=\(value.minimum) mean_cos=\(value.mean) rel_l2=\(value.relative)"
                    }
                    for (label, value) in [
                        ("default", features), ("tiled", tiled), ("fused", fused),
                    ] {
                        print(
                            "GLM53_VISION precision \(name) kernel=\(label) "
                                + "vs_fp32[\(line(compare(value)))] "
                                + "vs_bf16[\(line(compare(value, against: officialBF16)))] "
                                + "vs_fp32_strong_rows[\(line(compare(value, rows: strongRows)))] "
                                + "vs_bf16_strong_rows[\(line(compare(value, against: officialBF16, rows: strongRows)))] "
                                + "strong_rows=\(strongRows.dim(0))/\(norms.count)")
                    }
                    print(
                        "GLM53_VISION precision \(name) kernel=official_bf16 "
                            + "vs_fp32[\(line(compare(officialBF16)))] "
                            + "vs_fp32_strong_rows[\(line(compare(officialBF16, rows: strongRows)))]"
                    )
                    let nativeError = compare(features)
                    let envelope = compare(officialBF16)
                    print(
                        "GLM53_VISION reference \(name) grid=\(grid) pixel_max_abs=\(pixelError) "
                            + "native_vs_fp32 min_cos=\(nativeError.minimum) mean_cos=\(nativeError.mean) "
                            + "rel_l2=\(nativeError.relative) official_bf16_vs_fp32 min_cos=\(envelope.minimum) "
                            + "mean_cos=\(envelope.mean) rel_l2=\(envelope.relative)")
                    if nativeError.relative > envelope.relative * 1.25 + 0.01
                        || nativeError.mean < envelope.mean - 0.01
                    {
                        failures.append(
                            "reference \(name): tower features exceed the BF16 envelope")
                    }
                    if pixelError > 1e-4 {
                        failures.append("reference \(name): processor pixels differ")
                    }
                }
            }

            // Prepare every case once; both paths consume the same prompt/pixels.
            struct Prepared {
                let name: String
                let detailed: GLM5NextPreparedInput
                let tokens: [Int]
                let expected: [[String]]
                let large: Bool
            }
            var prepared: [Prepared] = []
            for item in Self.cases where runLarge || !item.large {
                var content: [[String: any Sendable]] = []
                for part in item.parts {
                    switch part {
                    case .text(let text): content.append(["type": "text", "text": text])
                    case .image: content.append(["type": "image"])
                    }
                }
                let input = UserInput(
                    messages: [["role": "user", "content": content]],
                    images: item.images.map { .ciImage($0) },
                    additionalContext: ["reasoning_effort": "low", "clear_thinking": false])
                let detailed = try await processor.prepareDetailed(input: input)
                let tokens = detailed.input.text.tokens.asArray(Int32.self).map(Int.init)
                let visual = detailed.placements.map(\.tokenRange.count).reduce(0, +)
                print(
                    "GLM53_VISION prepared \(item.name) prompt_tokens=\(tokens.count) "
                        + "visual_tokens=\(visual) grids=\(detailed.input.image?.frames?.map { [$0.t, $0.h, $0.w] } ?? [])"
                )
                if item.large, visual < 7_000 {
                    failures.append(
                        "large: expected an image near the 8000-token cap, got \(visual)")
                }
                prepared.append(
                    .init(
                        name: item.name, detailed: detailed, tokens: tokens,
                        expected: item.expected,
                        large: item.large))
            }

            // Path 1: direct multimodal forward (single-shot prefill).
            var direct: [String: [Int]] = [:]
            for item in prepared where !item.large {
                let start = ContinuousClock.now
                let generated: [Int] = try MLX.withError { error in
                    let cache = glm.newTextCache()
                    var logits = try glm.forward(item.detailed, cache: cache).logits
                    var next = argMax(logits[0, -1], axis: -1).asType(.int32)
                    eval(next)
                    try error.check()
                    var out = [Int(next.item(Int32.self))]
                    while out.count < outputTokens, !eos.contains(out.last!) {
                        logits = try glm.forward(
                            LMInput(text: .init(tokens: next.reshaped(1, 1))), cache: cache
                        ).logits
                        next = argMax(logits[0, -1], axis: -1).asType(.int32)
                        eval(next)
                        try error.check()
                        out.append(Int(next.item(Int32.self)))
                    }
                    return out
                }
                direct[item.name] = generated
                let text = tokenizer.decode(tokenIds: generated, skipSpecialTokens: false)
                let ok = Self.matches(text, item.expected)
                print(
                    "GLM53_VISION direct \(item.name) ok=\(ok) tokens=\(generated.count) "
                        + "seconds=\(start.duration(to: .now)) answer=\(Self.answer(text).debugDescription)"
                )
                if !ok { failures.append("direct \(item.name): \(Self.answer(text).prefix(160))") }
            }

            // Path 2: EngineV2 paged CBv2 (.serving), media rows next to text.
            let engine = try Self.makeEngine(glm.textTower, rows: 4)
            let textPrompt = try tokenizer.applyChatTemplate(
                messages: [
                    [
                        "role": "user",
                        "content": "What is the capital of France? Answer in one word.",
                    ]
                ],
                tools: nil, additionalContext: ["reasoning_effort": "low", "clear_thinking": false])
            var nextID: UInt64 = 1
            for item in prepared {
                let towerStart = ContinuousClock.now
                let image = try #require(item.detailed.input.image)
                let grids = try #require(image.frames)
                var row = 0
                var features: [MLXArray] = []
                for grid in grids {
                    // One image per tower call; query-tiled attention.
                    let count = grid.t * grid.h * grid.w
                    let value = try glm.visionTower.forward(
                        patches: image.pixels[row ..< row + count],
                        grids: [
                            try GLM5NextVisionGrid(temporal: grid.t, height: grid.h, width: grid.w)
                        ])
                    eval(value)
                    features.append(value.expandedDimensions(axis: 0))
                    row += count
                }
                let towerSeconds = towerStart.duration(to: .now)
                let placements = item.detailed.placements.sorted {
                    $0.tokenRange.lowerBound < $1.tokenRange.lowerBound
                }
                let embeddings = placements.map { placement in
                    features[placement.itemIndex]
                }
                let media = CBv2Request(
                    id: .init(nextID), promptTokens: item.tokens, sampling: .init(temperature: 0),
                    maxTokens: outputTokens, stopTokens: eos, prefixCacheEnabled: false,
                    multimodal: CBv2MultimodalInput(
                        spans: placements.map {
                            CBv2ImageSpan(
                                tokenOffset: $0.tokenRange.lowerBound, length: $0.tokenRange.count)
                        }, attention: .causal
                    ) { embeddings })
                let text = CBv2Request(
                    id: .init(nextID + 1), promptTokens: textPrompt,
                    sampling: .init(temperature: 0),
                    maxTokens: outputTokens, stopTokens: eos, prefixCacheEnabled: false)
                nextID += 2
                let start = ContinuousClock.now
                async let mediaRun = Self.collect(try engine.submit(media))
                async let textRun = Self.collect(try engine.submit(text))
                let (mediaResult, textResult) = try await (mediaRun, textRun)
                let mediaText = tokenizer.decode(
                    tokenIds: mediaResult.tokens, skipSpecialTokens: false)
                let textAnswer = tokenizer.decode(
                    tokenIds: textResult.tokens, skipSpecialTokens: false)
                let ok = Self.matches(mediaText, item.expected)
                let textOK = Self.answer(textAnswer).contains("paris")
                let agreement = direct[item.name].map { reference in
                    zip(reference, mediaResult.tokens).prefix { $0 == $1 }.count
                }
                print(
                    "GLM53_VISION cbv2 \(item.name) ok=\(ok) finish=\(mediaResult.finish) "
                        + "tokens=\(mediaResult.tokens.count) tower_s=\(towerSeconds) "
                        + "seconds=\(start.duration(to: .now)) direct_prefix_agreement=\(agreement.map(String.init) ?? "n/a") "
                        + "concurrent_text_ok=\(textOK) answer=\(Self.answer(mediaText).debugDescription)"
                )
                if !ok {
                    failures.append("cbv2 \(item.name): \(Self.answer(mediaText).prefix(160))")
                }
                if !textOK {
                    failures.append(
                        "cbv2 \(item.name): concurrent text answer \(Self.answer(textAnswer).prefix(80))"
                    )
                }
            }
            await engine.shutdown()
            print("GLM53_VISION failures=\(failures)")
            #expect(failures.isEmpty, "\(failures)")
        }
    }
}
