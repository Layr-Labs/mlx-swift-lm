// Copyright © 2026 Eigen Labs.

import CoreGraphics
import CoreImage
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXNN
import Testing
import Tokenizers

@testable import MLXLMCommon
@testable import MLXVLM

/// The reference side of the Inkling 2 vision gate v2.2 (release gates "Tolerances",
/// amended v2, and "vision gate v2.2"), from `make_vision_precision_fixture.py`:
/// per image, the error each transformers bf16 op adds (1 - mean row cosine
/// against float32, the op alone on the float32 run's inputs rounded to bf16;
/// gate (a)); the floor of 19 distinct equally valid bf16 runs' distance to
/// float32, and each run held out against the floor of the other 18
/// (diagnostic (b)); for oracle prompt `e`, the top-1 range of the same 19
/// reference embedding runs through our text model (gate (c)).
struct Inkling2VisionPrecisionFixture: Decodable {
    struct Stats: Decodable {
        let mean: Double
        let p01: Double
        let min: Double
    }

    struct Floor: Decodable {
        let tower: Stats
        let embeds: Stats
    }

    struct Image: Decodable {
        let index: Int
        let grid: [Int]
        let realizations: Int
        /// The second statistic (b) checks: `min` below 25 tokens, else `p01`.
        let tail: String
        let addedError: [String: Double]
        let realizationFloor: Floor
        /// Per statistic, the largest shortfall below the other 18 runs' floor
        /// that any held-out reference run shows (0 when none falls below).
        let heldOutShortfall: Floor

        enum CodingKeys: String, CodingKey {
            case index, grid, realizations, tail
            case addedError = "added_error"
            case realizationFloor = "realization_floor"
            case heldOutShortfall = "held_out_shortfall"
        }
    }

    /// Each reference run held out against the floors of the other 18, on
    /// (b)'s checks: how many of them it misses.
    struct HeldOut: Decodable {
        let checks: Int
        let worstMisses: Int
        let misses: [String: Int]

        enum CodingKeys: String, CodingKey {
            case checks, misses
            case worstMisses = "worst_misses"
        }
    }

    struct TopOneRange: Decodable {
        let low: Int
        let high: Int
    }

    /// One (b) check: our statistic, its floor and the largest held-out
    /// reference shortfall on it.
    struct FloorCheck {
        let name: String
        let value: Double
        let floor: Double
        let allowedShortfall: Double

        var missed: Bool { value < floor }
        /// Further below the floor than any held-out reference run falls here.
        var beyondTheReference: Bool { floor - value > allowedShortfall }
    }

    let images: [Image]
    let heldOut: HeldOut
    let promptETop1: TopOneRange

    enum CodingKeys: String, CodingKey {
        case images
        case heldOut = "held_out"
        case promptETop1 = "prompt_e_top1"
    }

    /// Where `checks` go beyond the reference's own held-out runs (see
    /// "vision gate v2.2", diagnostic (b), reported and never asserted): more
    /// missed checks than the worst held-out run misses, and each miss
    /// further below its floor than any held-out run falls on that check.
    /// Empty when neither happens.
    func beyondTheHeldOutReference(_ checks: [FloorCheck]) -> [String] {
        let misses = checks.filter(\.missed)
        var findings: [String] = []
        if misses.count > heldOut.worstMisses {
            findings.append(
                "\(misses.count) of \(checks.count) checks below their floors; "
                    + "the worst held-out reference run misses \(heldOut.worstMisses)")
        }
        for miss in misses where miss.beyondTheReference {
            findings.append(
                "\(miss.name) \(miss.floor - miss.value) below its floor; the largest "
                    + "held-out reference shortfall there is \(miss.allowedShortfall)")
        }
        return findings
    }

    static func load() throws -> Self {
        let url = try #require(
            Bundle.module.url(
                forResource: "inkling2-vision-reference-precision", withExtension: "json"))
        return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }
}

extension KernelTests {
    /// Real-checkpoint checks of the Inkling 2 vision tower. They read only the
    /// `model.visual.*` tensors (about 0.65 GB) and run when
    /// `INKLING2_MODEL_DIR` names a verified `thinkingmachines/Inkling-2-Mini`
    /// snapshot.
    @Suite(.serialized)
    struct Inkling2CheckpointTests {
        static let environment = ProcessInfo.processInfo.environment
        static var modelDirectory: URL? {
            environment["INKLING2_MODEL_DIR"].map { URL(fileURLWithPath: $0, isDirectory: true) }
        }

        /// The checkpoint's tower, its weights cast to `dtype` when given.
        static func loadTower(_ directory: URL, dtype: DType? = nil) throws -> Inkling2VisionModel {
            struct Root: Decodable {
                let vision_config: Inkling2VisionConfiguration
            }
            let root = try JSONDecoder().decode(
                Root.self, from: Data(contentsOf: directory.appending(component: "config.json")))
            try root.vision_config.validate()
            let model = Inkling2VisionModel(root.vision_config)
            var weights: [String: MLXArray] = [:]
            let files = try FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil
            ).filter {
                $0.lastPathComponent.hasPrefix("model-") && $0.pathExtension == "safetensors"
            }
            for file in files {
                for (key, value) in try loadArrays(url: file) where key.hasPrefix("model.visual.") {
                    weights[String(key.dropFirst("model.visual.".count))] =
                        dtype.map { value.asType($0) } ?? value
                }
            }
            try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
            eval(model)
            return model
        }

        static func pattern(side: Int) -> CIImage {
            var rgba = [UInt8](repeating: 255, count: side * side * 4)
            for y in 0 ..< side {
                for x in 0 ..< side {
                    for c in 0 ..< 3 {
                        rgba[(y * side + x) * 4 + c] = UInt8(
                            (x * 7 + y * 13 + c * 101 + (x * y) % 97) % 256)
                    }
                }
            }
            let image = CGImage(
                width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: side * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: CGDataProvider(data: Data(rgba) as CFData)!, decode: nil,
                shouldInterpolate: false, intent: .defaultIntent)!
            return CIImage(cgImage: image)
        }

        /// Mean, first percentile (linear interpolation, as numpy) and minimum
        /// of the per-row cosines between two `[rows, width]` arrays.
        static func cosineStats(_ actual: MLXArray, _ expected: MLXArray) -> (
            mean: Float, p01: Float, min: Float
        ) {
            let cosines = rowCosines(actual, expected).asArray(Float.self).sorted()
            let position = 0.01 * Float(cosines.count - 1)
            let lower = Int(position.rounded(.down))
            let upper = min(lower + 1, cosines.count - 1)
            let p01 = cosines[lower] + (cosines[upper] - cosines[lower]) * (position - Float(lower))
            return (cosines.reduce(0, +) / Float(cosines.count), p01, cosines[0])
        }

        /// Vision gate v2.2, diagnostic (b): reported, never asserted.
        /// A correct implementation's equally valid bf16 runs fail a floor test
        /// like this most of the time, so it measures rounding noise; gates (a)
        /// and (c) carry the precision bar. 24 checks: on each of the six
        /// reference images, our tower output (with the class row) and
        /// projected embeddings against the transformers float32 run, per-row
        /// cosine mean and p01, or the minimum on images with fewer than 25
        /// tokens (at most 97 tower rows, where p01 falls between the two lowest
        /// rows). A check misses when it falls below the worst of 19 distinct
        /// equally valid transformers bf16 runs (10 encoder row orders, 9
        /// channel relabellings); it is over when it falls further below than
        /// any of those runs, held out against the other 18, falls there. The
        /// test prints every check, the misses and the overs.
        /// Floors and held-out data: `inkling2-vision-reference-precision.json`
        /// (make_vision_precision_fixture.py). Reference outputs:
        /// `make_vision_real_reference.py`, path in
        /// `INKLING2_VISION_REAL_REFERENCE`. With `INKLING2_ORACLE_MODEL_DIR` it
        /// also pins the first image to oracle prompt `e`'s pixels and prints
        /// our distance to the oracle fixture.
        @Test(
            .enabled(
                if: modelDirectory != nil && environment["INKLING2_VISION_REAL_REFERENCE"] != nil))
        func towerRealizationFloorDiagnostic() throws {
            let oldCache = Memory.cacheLimit
            Memory.cacheLimit = 2 << 30
            defer {
                Memory.clearCache()
                Memory.cacheLimit = oldCache
            }
            let fixture = try Inkling2VisionPrecisionFixture.load()
            let path = try #require(Self.environment["INKLING2_VISION_REAL_REFERENCE"])
            let reference = try loadArrays(url: URL(fileURLWithPath: path))
            let tower = try Self.loadTower(Self.modelDirectory!)
            var checks: [Inkling2VisionPrecisionFixture.FloorCheck] = []
            for image in fixture.images {
                let index = image.index
                let pixels = try #require(reference["image\(index).pixel_values"])
                let grid = try #require(reference["image\(index).grid"]).asArray(Int32.self)
                let (rows, columns) = (Int(grid[1]), Int(grid[2]))
                #expect([1, rows, columns] == image.grid)
                let features = try tower.imageFeatures(
                    pixels: pixels, grids: [THW(1, rows, columns)])[0]
                let states = tower.backboneStates(pixels, tokenRows: rows, tokenColumns: columns)
                for (label, ours, key, floor, heldOut) in [
                    (
                        "tower output", states, ".tower", image.realizationFloor.tower,
                        image.heldOutShortfall.tower
                    ),
                    (
                        "image embeds", features, "", image.realizationFloor.embeds,
                        image.heldOutShortfall.embeds
                    ),
                ] {
                    let fp32 = try #require(reference["image\(index).fp32\(key)"])
                    let mine = Self.cosineStats(ours, fp32)
                    let small = rows * columns < 25
                    let (tail, mineTail, floorTail, heldOutTail) =
                        small
                        ? ("min", mine.min, floor.min, heldOut.min)
                        : ("p01", mine.p01, floor.p01, heldOut.p01)
                    #expect(image.tail == tail)
                    print(
                        "INKLING2_VISION_DIAGNOSTIC_B image\(index) \(rows)x\(columns) \(label): "
                            + "ours_vs_fp32 mean=\(mine.mean) p01=\(mine.p01) min=\(mine.min) | "
                            + "floor over \(image.realizations) reference runs mean=\(floor.mean) "
                            + "\(tail)=\(floorTail) | largest held-out shortfall "
                            + "mean=\(heldOut.mean) \(tail)=\(heldOutTail)")
                    checks += [
                        .init(
                            name: "image\(index) \(label) mean", value: Double(mine.mean),
                            floor: floor.mean, allowedShortfall: heldOut.mean),
                        .init(
                            name: "image\(index) \(label) \(tail)", value: Double(mineTail),
                            floor: floorTail, allowedShortfall: heldOutTail),
                    ]
                }
            }
            #expect(fixture.images.count == 6)
            #expect(checks.count == fixture.heldOut.checks)
            let misses = checks.filter(\.missed)
            for miss in misses {
                print(
                    "INKLING2_VISION_DIAGNOSTIC_B miss \(miss.name): shortfall "
                        + "\(miss.floor - miss.value), largest held-out \(miss.allowedShortfall), "
                        + (miss.beyondTheReference ? "over" : "within"))
            }
            let findings = fixture.beyondTheHeldOutReference(checks)
            let overs = misses.filter(\.beyondTheReference).count
            print(
                "INKLING2_VISION_DIAGNOSTIC_B (reported, not asserted) misses="
                    + "\(misses.count)/\(checks.count) "
                    + "worst_held_out=\(fixture.heldOut.worstMisses) over on \(overs) check(s)"
                    + (findings.isEmpty ? "" : ": " + findings.joined(separator: "; ")))

            guard let oracleDirectory = Self.environment["INKLING2_ORACLE_MODEL_DIR"] else {
                return
            }
            let oracle = try loadArrays(
                url: URL(fileURLWithPath: oracleDirectory, isDirectory: true)
                    .appending(component: "e").appending(component: "vision.safetensors"))
            let pixels = try #require(oracle["pixel_values"])
            #expect(
                pixels.asArray(Float.self).map(\.bitPattern)
                    == (try #require(reference["image0.pixel_values"])).asArray(Float.self)
                    .map(\.bitPattern))
            let states = tower.backboneStates(pixels, tokenRows: 6, tokenColumns: 8)
            let features = try tower.imageFeatures(pixels: pixels, grids: [THW(1, 6, 8)])[0]
            let output = Self.cosineStats(states, try #require(oracle["vision_last_hidden_state"]))
            let embeds = Self.cosineStats(features, try #require(oracle["image_embeds"]))
            print(
                "INKLING2_VISION_ORACLE (measured, no bar) tower_output mean=\(output.mean) "
                    + "p01=\(output.p01) min=\(output.min) | image_embeds mean=\(embeds.mean) "
                    + "p01=\(embeds.p01) min=\(embeds.min)")
        }

        /// The layer's ops one at a time, as `Inkling2VisionLayer` runs them;
        /// named as in `vision_layer_trace.py`.
        static func layerOps(
            _ layer: Inkling2VisionLayer, _ x: MLXArray, cos: MLXArray, sin: MLXArray
        ) -> [String: MLXArray] {
            let a = layer.attention
            func heads(_ y: MLXArray) -> MLXArray { y.reshaped(x.dim(0), a.heads, a.headDim) }
            var p: [String: MLXArray] = [:]
            p["ln1"] = layer.norm1(x)
            p["q"] = a.queryProjection(p["ln1"]!)
            p["k"] = a.keyProjection(p["ln1"]!)
            p["v"] = a.valueProjection(p["ln1"]!)
            p["rq"] = inkling2ApplyRotary(heads(p["q"]!), cos: cos, sin: sin).reshaped(x.dim(0), -1)
            p["rk"] = inkling2ApplyRotary(heads(p["k"]!), cos: cos, sin: sin).reshaped(x.dim(0), -1)
            p["sdpa"] = a.attend(heads(p["rq"]!), heads(p["rk"]!), heads(p["v"]!))
            p["attn"] = a.outputProjection(p["sdpa"]!)
            p["h1"] = x + layer.scale1.gamma * p["attn"]!
            p["ln2"] = layer.norm2(p["h1"]!)
            p["fc1"] = layer.mlp.fc(p["ln2"]!)
            p["gelu"] = inkling2Gelu(p["fc1"]!)
            p["mlp"] = layer.mlp.projection(p["gelu"]!)
            p["out"] = p["h1"]! + layer.scale2.gamma * p["mlp"]!
            return p
        }

        /// Each op of `layer` alone on the float32 run's inputs (`input`, `t`)
        /// rounded to the layer's dtype.
        static func layerOpsAlone(
            _ layer: Inkling2VisionLayer, input: MLXArray, truth t: [String: MLXArray],
            cos: MLXArray, sin: MLXArray
        ) -> [String: MLXArray] {
            let a = layer.attention
            let dtype = layer.attention.queryProjection.weight.dtype
            func low(_ y: MLXArray) -> MLXArray { y.asType(dtype) }
            func heads(_ y: MLXArray) -> MLXArray {
                low(y).reshaped(input.dim(0), a.heads, a.headDim)
            }
            var p: [String: MLXArray] = [:]
            p["ln1"] = layer.norm1(low(input))
            p["q"] = a.queryProjection(low(t["ln1"]!))
            p["k"] = a.keyProjection(low(t["ln1"]!))
            p["v"] = a.valueProjection(low(t["ln1"]!))
            p["rq"] = inkling2ApplyRotary(heads(t["q"]!), cos: cos, sin: sin)
                .reshaped(input.dim(0), -1)
            p["rk"] = inkling2ApplyRotary(heads(t["k"]!), cos: cos, sin: sin)
                .reshaped(input.dim(0), -1)
            p["sdpa"] = a.attend(heads(t["rq"]!), heads(t["rk"]!), heads(t["v"]!))
            p["attn"] = a.outputProjection(low(t["sdpa"]!))
            p["h1"] = low(input) + layer.scale1.gamma * low(t["attn"]!)
            p["ln2"] = layer.norm2(low(t["h1"]!))
            p["fc1"] = layer.mlp.fc(low(t["ln2"]!))
            p["gelu"] = inkling2Gelu(low(t["fc1"]!))
            p["mlp"] = layer.mlp.projection(low(t["gelu"]!))
            p["out"] = low(t["h1"]!) + layer.scale2.gamma * low(t["mlp"]!)
            return p
        }

        /// 1 - mean row cosine, as half the squared distance of the
        /// normalized rows (exact in float32 for errors near 1e-6).
        static func addedError(_ actual: MLXArray, _ expected: MLXArray) -> Double {
            let a = actual.asType(.float32)
            let b = expected.asType(.float32)
            let d =
                a / sqrt((a * a).sum(axis: -1, keepDims: true))
                - b / sqrt((b * b).sum(axis: -1, keepDims: true))
            return Double((0.5 * (d * d).sum(axis: -1)).mean().item(Float.self))
        }

        /// Vision gate v2.2 (a), required on every host: each op of our bf16 tower adds
        /// at most 1.05x the error the same op adds in the transformers bf16
        /// tower, at every op (patch embedding, position add, all 47 norms,
        /// projections, rotary, attention, residual adds, MLP, merger) on all
        /// six reference images. An op's added error is 1 - mean row cosine
        /// between the op on the float32 run's inputs rounded to bf16 and the
        /// float32 op output. Ours uses our float32 tower as the float32 run (it
        /// equals transformers float32 to cosine 1.00000000); the reference
        /// values are in `inkling2-vision-reference-precision.json`.
        @Test(
            .enabled(
                if: modelDirectory != nil && environment["INKLING2_VISION_REAL_REFERENCE"] != nil))
        func towerOpsAddNoMoreErrorThanTheReference() throws {
            let oldCache = Memory.cacheLimit
            Memory.cacheLimit = 2 << 30
            defer {
                Memory.clearCache()
                Memory.cacheLimit = oldCache
            }
            let fixture = try Inkling2VisionPrecisionFixture.load()
            let path = try #require(Self.environment["INKLING2_VISION_REAL_REFERENCE"])
            let reference = try loadArrays(url: URL(fileURLWithPath: path))
            let low = try Self.loadTower(Self.modelDirectory!)
            let high = try Self.loadTower(Self.modelDirectory!, dtype: .float32)
            let dtype = low.computeDType
            var worst: [String: (ratio: Double, at: String)] = [:]
            var checked = 0
            for image in fixture.images {
                let index = image.index
                let pixels = try #require(reference["image\(index).pixel_values"])
                let grid = try #require(reference["image\(index).grid"]).asArray(Int32.self)
                let (rows, columns) = (Int(grid[1]), Int(grid[2]))
                let geometry = Inkling2VisionGeometry(
                    tokenRows: rows, tokenColumns: columns, mergeSize: low.configuration.mergeSize)
                let (cos, sin) = low.rotaryTables(geometry)
                func check(_ name: String, _ ours: MLXArray, _ truth: MLXArray) throws {
                    let mine = Self.addedError(ours, truth)
                    let theirs = try #require(image.addedError[name], "image\(index) \(name)")
                    let ratio = mine / theirs
                    #expect(
                        ratio <= 1.05,
                        "image\(index) \(name): added error \(mine) vs reference \(theirs)")
                    let kind = String(name.split(separator: ".").last!)
                    if ratio > worst[kind]?.ratio ?? 0 {
                        worst[kind] = (ratio, "image\(index) \(name)")
                    }
                    checked += 1
                }

                let embeddings = low.backbone.embeddings
                let patchTruth = high.backbone.embeddings.patchEmbedding(
                    high.patchFeatures(pixels, geometry))
                try check(
                    "patch", embeddings.patchEmbedding(low.patchFeatures(pixels, geometry)),
                    patchTruth)
                let embedTruth = concatenated(
                    [
                        high.classToken(dtype: .float32),
                        patchTruth + high.positionTerm(geometry, dtype: .float32),
                    ], axis: 0)
                try check(
                    "embed",
                    concatenated(
                        [
                            low.classToken(dtype: dtype),
                            patchTruth.asType(dtype) + low.positionTerm(geometry, dtype: dtype),
                        ], axis: 0), embedTruth)
                var truth = high.backbone.preNorm(embedTruth)
                try check("pre_ln", low.backbone.preNorm(embedTruth.asType(dtype)), truth)

                // The op-by-op bf16 path reproduces the production tower.
                var ours = low.backbone.preNorm(
                    concatenated(
                        [
                            low.classToken(dtype: dtype),
                            embeddings.patchEmbedding(low.patchFeatures(pixels, geometry))
                                + low.positionTerm(geometry, dtype: dtype),
                        ], axis: 0))
                for (i, (layer, layer32)) in zip(low.backbone.layers, high.backbone.layers)
                    .enumerated()
                {
                    let exact = Self.layerOps(layer32, truth, cos: cos, sin: sin)
                    let alone = Self.layerOpsAlone(
                        layer, input: truth, truth: exact, cos: cos, sin: sin)
                    for (name, value) in alone.sorted(by: { $0.key < $1.key }) {
                        try check(String(format: "l%02d.", i) + name, value, exact[name]!)
                    }
                    ours = Self.layerOps(layer, ours, cos: cos, sin: sin)["out"]!
                    truth = exact["out"]!
                }
                #expect(
                    arrayEqual(
                        ours, low.backboneStates(pixels, tokenRows: rows, tokenColumns: columns)
                    ).item(Bool.self), "image\(index): op-by-op path == backboneStates")

                let width =
                    low.configuration.mergeSize * low.configuration.mergeSize
                    * low.configuration.hiddenSize
                let merged = truth[1...].reshaped(rows * columns, width)
                let projectionIn = high.projectionIn(merged)
                let projectionGelu = inkling2Gelu(projectionIn)
                try check("proj_in", low.projectionIn(merged.asType(dtype)), projectionIn)
                try check("proj_gelu", inkling2Gelu(projectionIn.asType(dtype)), projectionGelu)
                try check(
                    "proj_out", low.projectionOut(projectionGelu.asType(dtype)),
                    high.projectionOut(projectionGelu))
                Memory.clearCache()
            }
            print(
                "INKLING2_VISION_OPS checked=\(checked) worst ratio per op: "
                    + worst.sorted(by: { $0.key < $1.key }).map {
                        "\($0.key) \(String(format: "%.3f", $0.value.ratio)) (\($0.value.at))"
                    }.joined(separator: ", "))
            #expect(checked == fixture.images.reduce(0) { $0 + $1.addedError.count })
        }

        /// Vision-tower time per image (median of 3 after one warm-up) at
        /// 512, 1024 and 2048 px squares. Writes one JSON line per size to
        /// `INKLING2_BENCH_OUT` when set.
        @Test(.enabled(if: environment["INKLING2_VISION_BENCH"] == "1" && modelDirectory != nil))
        func towerThroughput() throws {
            let oldCache = Memory.cacheLimit
            Memory.cacheLimit = 2 << 30
            defer {
                Memory.clearCache()
                Memory.cacheLimit = oldCache
            }
            let tower = try Self.loadTower(Self.modelDirectory!)
            let processor = Inkling2Processor(
                Inkling2ProcessorConfiguration(), tokenizer: TestTokenizer())
            var lines: [String] = []
            for side in [512, 1024, 2048] {
                let (pixels, grid) = try processor.preprocess(
                    image: Self.pattern(side: side), processing: nil)
                eval(pixels)
                var times: [Double] = []
                for _ in 0 ..< 4 {
                    let start = Date()
                    let features = try tower.imageFeatures(pixels: pixels, grids: [grid])
                    eval(features)
                    times.append(Date().timeIntervalSince(start))
                    #expect(features[0].shape == [grid.h * grid.w, 2304])
                }
                let median = times.dropFirst().sorted()[1]
                let line =
                    "{\"side\":\(side),\"tokens\":\(grid.h * grid.w),\"patches\":\(4 * grid.h * grid.w),"
                    + "\"tower_ms\":\(String(format: "%.1f", median * 1000)),"
                    + "\"tokens_per_s\":\(String(format: "%.0f", Double(grid.h * grid.w) / median))}"
                print("INKLING2_VISION_BENCH", line)
                lines.append(line)
                Memory.clearCache()
            }
            if let path = Self.environment["INKLING2_BENCH_OUT"] {
                try (lines.joined(separator: "\n") + "\n").write(
                    toFile: path, atomically: true, encoding: .utf8)
            }
        }

        static var oraclePrompt: URL? {
            environment["INKLING2_ORACLE_MODEL_DIR"].map {
                URL(fileURLWithPath: $0, isDirectory: true).appending(component: "e")
            }
        }

        /// The whole checkpoint as the `inkling2_loop` VLM (about 20 GB).
        static func loadModel(_ directory: URL) throws -> Inkling2VLM {
            let data = try Data(contentsOf: directory.appending(component: "config.json"))
            let configuration = try JSONDecoder.json5().decode(
                Inkling2VLMConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            let model = Inkling2VLM(configuration)
            try loadWeights(modelDirectory: directory, model: model)
            return model
        }

        static func rowCosines(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
            let a = a.asType(.float32)
            let b = b.asType(.float32)
            return (a * b).sum(axis: -1)
                / sqrt((a * a).sum(axis: -1) * (b * b).sum(axis: -1))
        }

        /// A prompt's floors v3: the top-1 rate and last-8 bar the
        /// reference reaches across its own perturbations (fixture
        /// `noise_floor_v2.json`, key `floors_v3.prompts.<p>`, in
        /// `INKLING2_ORACLE_MODEL_DIR`; `INKLING2_ORACLE_FLOORS` names a newer
        /// floors file of the same layout).
        static func oracleFloors(prompt: String) throws -> (top1: Double, last8: Double) {
            var path = environment["INKLING2_ORACLE_FLOORS"] ?? ""
            if path.isEmpty {
                let directory = try #require(environment["INKLING2_ORACLE_MODEL_DIR"])
                path =
                    URL(fileURLWithPath: directory, isDirectory: true)
                    .appending(component: "noise_floor_v2.json").path
            }
            let root = try JSONSerialization.jsonObject(
                with: Data(contentsOf: URL(fileURLWithPath: path)))
            let floors = (root as? [String: Any])?["floors_v3"] as? [String: Any]
            let entry = (floors?["prompts"] as? [String: Any])?[prompt] as? [String: Any]
            return (
                try #require(entry?["min_top1_rate"] as? Double),
                try #require(entry?["max_last8_abs_logit_diff"] as? Double)
            )
        }

        /// Oracle prompt `e` (one 300x220 image, 48 image tokens, 71 tokens):
        /// spliced input embeddings against the reference embeddings, then
        /// teacher-forced top-1 and last-8 logits of the full-recompute forward
        /// against `e`'s floors v3 (`oracleFloors`), and top-1 within the
        /// range that 19 equally valid reference bf16 runs of the image
        /// embeddings give through our text model (vision gate v2.2 (c),
        /// required on every host; one oracle run is one draw). Needs
        /// `INKLING2_FULL_MODEL=1`.
        @Test(
            .enabled(
                if: modelDirectory != nil && oraclePrompt != nil
                    && environment["INKLING2_FULL_MODEL"] == "1"))
        func imagePromptMatchesTheOracle() throws {
            let (oldCache, oldMemory) = (Memory.cacheLimit, Memory.memoryLimit)
            Memory.cacheLimit = 4 << 30
            Memory.memoryLimit = 40 << 30
            defer {
                Memory.clearCache()
                Memory.cacheLimit = oldCache
                Memory.memoryLimit = oldMemory
            }
            let prompt = Self.oraclePrompt!
            let vision = try loadArrays(url: prompt.appending(component: "vision.safetensors"))
            let forced = try loadArrays(
                url: prompt.appending(component: "teacher_forced.safetensors"))
            let hidden = try loadArrays(url: prompt.appending(component: "hidden.safetensors"))
            let ids = try #require(forced["input_ids"]).asType(.int32).reshaped(1, -1)
            let pixels = try #require(vision["pixel_values"])

            // The processor reproduces the reference pixels from the PNG.
            let png = prompt.deletingLastPathComponent().appending(
                component: "e_synthetic_300x220.png")
            let processor = Inkling2Processor(
                Inkling2ProcessorConfiguration(), tokenizer: TestTokenizer())
            let (processed, grid) = try processor.preprocess(
                image: try #require(CIImage(contentsOf: png)), processing: nil)
            #expect([grid.t, grid.h, grid.w] as [Int] == [1, 6, 8])
            #expect(
                processed.asArray(Float.self).map(\.bitPattern)
                    == pixels.asArray(Float.self).map(\.bitPattern))

            let floors = try Self.oracleFloors(prompt: "e")
            let model = try Self.loadModel(Self.modelDirectory!)
            let image = LMInput.ProcessedImage(pixels: pixels, frames: [grid])
            let embeddings = try model.inputEmbeddings(tokens: ids, image: image)
            let cosines = Self.rowCosines(embeddings[0], try #require(hidden["embeddings"]))
            let tokens = ids[0].asArray(Int32.self)
            let imageRows = tokens.indices.filter { tokens[$0] == 200_054 }
            let imageCosine = cosines[MLXArray(imageRows.map(Int32.init))].min().item(Float.self)
            let textCosine = cosines.min().item(Float.self)

            // Teacher-forced top-1 with our embeddings, and with the oracle's
            // embeddings through the same text model (isolates the tower).
            let expected = try #require(forced["argmax"]).asArray(Int32.self)
            let margins = try #require(forced["top20_values"])
            let gaps = (margins[0..., 0] - margins[0..., 1]).asArray(Float.self)
            func teacherForced(_ inputs: MLXArray) -> (argmax: [Int32], last8: MLXArray) {
                let logits = model.languageModel.forwardNoCache(nil, inputEmbedding: inputs).logits
                return (
                    argMax(logits[0], axis: -1).asType(.int32).asArray(Int32.self),
                    logits[0, (-8)...].asType(.float32)
                )
            }
            let ours = teacherForced(embeddings)
            let reference = teacherForced(
                try #require(hidden["embeddings"]).asType(embeddings.dtype).expandedDimensions(
                    axis: 0))
            let oracleLast8 = try #require(forced["last8_logits"])
            let last8Gap = abs(ours.last8 - oracleLast8).max().item(Float.self)
            let agree = zip(ours.argmax, expected).filter { $0 == $1 }.count
            let agreeReference = zip(reference.argmax, expected).filter { $0 == $1 }.count
            func misses(_ argmax: [Int32]) -> [String] {
                expected.indices.filter { argmax[$0] != expected[$0] }.map {
                    "\($0)\(tokens[$0] == 200_054 ? "img" : "")(gap \(String(format: "%.2f", gaps[$0])))"
                }
            }
            print(
                "INKLING2_IMAGE_PROMPT image_row_min_cosine=\(imageCosine) "
                    + "all_row_min_cosine=\(textCosine) top1=\(agree)/\(expected.count) "
                    + "top1_with_oracle_embeddings=\(agreeReference)/\(expected.count) "
                    + "last8_max_abs_logit_diff=\(last8Gap) misses=\(misses(ours.argmax)) "
                    + "misses_with_oracle_embeddings=\(misses(reference.argmax)) "
                    + "floors_v3 top1=\(floors.top1) last8=\(floors.last8)")
            #expect(imageCosine >= 0.999)
            #expect(Double(agree) / Double(expected.count) >= floors.top1)
            #expect(Double(last8Gap) <= floors.last8)
            let e = try Inkling2VisionPrecisionFixture.load().promptETop1
            #expect(
                (e.low ... e.high).contains(agree),
                "top-1 \(agree) outside the reference realizations' \(e.low)...\(e.high)")
        }

        /// Oracle text prompt `a` through the checkpoint loaded as the VLM
        /// (`VLMTypeRegistry` + `loadWeights`, the factory's path) with the
        /// checkpoint's DSpark drafter bound to its text model: the drafter
        /// runs rounds and the greedy tokens equal the drafter-off tokens.
        /// Needs `INKLING2_FULL_MODEL=1`.
        @Test(
            .enabled(
                if: modelDirectory != nil && oraclePrompt != nil
                    && environment["INKLING2_FULL_MODEL"] == "1"))
        func textPromptDraftsThroughTheVLM() async throws {
            let (oldCache, oldMemory) = (Memory.cacheLimit, Memory.memoryLimit)
            Memory.cacheLimit = 4 << 30
            Memory.memoryLimit = 40 << 30
            defer {
                Memory.clearCache()
                Memory.cacheLimit = oldCache
                Memory.memoryLimit = oldMemory
            }
            let directory = Self.modelDirectory!
            let loaded = try await VLMTypeRegistry.shared.createModel(
                configuration: try Data(contentsOf: directory.appending(component: "config.json")),
                modelType: "inkling2_loop")
            let model = try #require(loaded as? Inkling2VLM)
            try await loadWeights(modelDirectory: directory, model: model)
            let forced = try loadArrays(
                url: Self.oraclePrompt!.deletingLastPathComponent().appending(component: "a")
                    .appending(component: "teacher_forced.safetensors"))
            let prompt = try #require(forced["input_ids"]).asArray(Int32.self).map(Int.init)
            let steps = 48

            func generate(_ drafter: Inkling2DSparkDrafter?) async throws -> (
                [Int], CBv2MTPMetrics?
            ) {
                let kinds = model.cbv2LayerKinds
                let paged = try PagedKVBackend(
                    layerKinds: kinds,
                    config: .init(
                        capacityBytes: 2 << 30, dtype: .bfloat16, maxPrefillChunk: 256,
                        nominalMaxSequenceLength: 1024, segmentSizeBytes: 64 << 20,
                        layerDTypes: Array(repeating: .bfloat16, count: kinds.count)))
                let storage = paged.makeLayerCaches()
                let caches = model.newCacheV2 { index, _ in storage[index] }
                let engine = EngineV2(
                    model: CBv2SteppableLanguageModelAdapter(model), layerKinds: kinds,
                    backend: paged,
                    cacheProvider: CBv2LayerCacheBank(caches: caches, permitsCausalMedia: true),
                    sampler: CBv2GreedySampler(),
                    schedulerConfig: .init(
                        maxConcurrentRequests: 1, maxBatchedTokensPerStep: 256,
                        prefillChunkSize: 256, maxWaiting: 4, enablePrefixCache: false),
                    admissionConfig: .init(watermarkFraction: 0),
                    mtpDrafter: drafter,
                    mtpConfig: drafter == nil ? .init() : Inkling2DSparkDrafter.defaultMTPConfig())
                let stream = try engine.submit(
                    .init(
                        id: .init(1), promptTokens: prompt, sampling: .init(temperature: 0),
                        maxTokens: steps, prefixCacheEnabled: false))
                let tokens = await cbv2SchedCollect(stream, timeoutSeconds: 900).tokens
                let metrics = engine.mtpMetricsSnapshot()
                await engine.shutdown()
                return (tokens, metrics)
            }

            let (off, _) = try await generate(nil)
            let drafter = try Inkling2DSparkDrafter.load(
                directory: directory, target: model.languageModel)
            let (on, metrics) = try await generate(drafter)
            let rounds = metrics?.rounds ?? 0
            print(
                "INKLING2_VLM_MTP rounds=\(rounds) drafted=\(metrics?.draftedTokens ?? 0) "
                    + "accepted=\(metrics?.acceptedTokens ?? 0) identical=\(on == off) "
                    + "tokens=\(on.count)")
            #expect(rounds > 0)
            #expect(on == off)
        }

        /// Greedy continuation of oracle prompt `e` through the CBv2 paged
        /// engine (causal image span, scalar positions) equals the
        /// full-recompute greedy continuation. Needs `INKLING2_FULL_MODEL=1`.
        @Test(
            .enabled(
                if: modelDirectory != nil && oraclePrompt != nil
                    && environment["INKLING2_FULL_MODEL"] == "1"))
        func imagePromptGreedyThroughTheEngineMatchesFullRecompute() async throws {
            let (oldCache, oldMemory) = (Memory.cacheLimit, Memory.memoryLimit)
            Memory.cacheLimit = 4 << 30
            Memory.memoryLimit = 40 << 30
            defer {
                Memory.clearCache()
                Memory.cacheLimit = oldCache
                Memory.memoryLimit = oldMemory
            }
            let prompt = Self.oraclePrompt!
            let vision = try loadArrays(url: prompt.appending(component: "vision.safetensors"))
            let forced = try loadArrays(
                url: prompt.appending(component: "teacher_forced.safetensors"))
            let tokens = try #require(forced["input_ids"]).asArray(Int32.self).map(Int.init)
            let image = LMInput.ProcessedImage(
                pixels: try #require(vision["pixel_values"]), frames: [THW(1, 6, 8)])
            let model = try Self.loadModel(Self.modelDirectory!)
            let features = try model.perImageVisionFeatures(
                pixels: image.pixels, frames: image.frames)
            eval(features)
            let start = try #require(tokens.firstIndex(of: 200_054))
            let span = CBv2ImageSpan(tokenOffset: start, length: features[0].dim(1))
            let steps = 12

            var sequence = tokens
            var expected: [Int] = []
            for _ in 0 ..< steps {
                let logits = try model.forwardNoCache(
                    tokens: MLXArray(sequence.map(Int32.init)).reshaped(1, -1), image: image)
                let next = argMax(logits[0, -1], axis: -1).item(Int.self)
                expected.append(next)
                sequence.append(next)
            }

            let kinds = model.cbv2LayerKinds
            let paged = try PagedKVBackend(
                layerKinds: kinds,
                config: .init(
                    capacityBytes: 1 << 30, dtype: .bfloat16, maxPrefillChunk: 64,
                    nominalMaxSequenceLength: 256, segmentSizeBytes: 16 << 20,
                    layerDTypes: Array(repeating: .bfloat16, count: kinds.count)))
            let storage = paged.makeLayerCaches()
            let caches = model.newCacheV2 { index, _ in storage[index] }
            let engine = EngineV2(
                model: CBv2SteppableLanguageModelAdapter(model), layerKinds: kinds,
                backend: paged,
                cacheProvider: CBv2LayerCacheBank(caches: caches, permitsCausalMedia: true),
                sampler: CBv2GreedySampler(),
                schedulerConfig: .init(
                    maxConcurrentRequests: 2, maxBatchedTokensPerStep: 256,
                    prefillChunkSize: 32, maxWaiting: 4, enablePrefixCache: false))
            let row = features[0][0]
            let stream = try engine.submit(
                .init(
                    id: .init(1), promptTokens: tokens, sampling: .init(temperature: 0),
                    maxTokens: steps, prefixCacheEnabled: false,
                    multimodal: CBv2MultimodalInput(spans: [span], attention: .causal) { [row] }
                ))
            let generated = await cbv2SchedCollect(stream, timeoutSeconds: 600).tokens
            await engine.shutdown()
            print("INKLING2_IMAGE_GREEDY engine=\(generated) full_recompute=\(expected)")
            #expect(generated == expected)
        }
    }

    /// The released processor fixtures from the oracle worker (transformers
    /// e3a65e2c on the checkpoint's own tokenizer, template and processor
    /// config). Runs when `INKLING2_ORACLE_TEMPLATE_DIR` names the fixture folder;
    /// token checks also need `INKLING2_MODEL_DIR` for the tokenizer files.
    @Suite(.serialized)
    struct Inkling2ProcessorFixtureTests {
        struct Fixture: Decodable {
            let name: String
            let messages: [[String: AnyCodableContent]]
            let kwargs: [String: String]
            let input_ids: [Int]
            let image_grid_thw: [[Int]]
            let pixel_values_file: String
            let pixel_values_sha256: String
        }

        /// Content as the fixture stores it: a string or a list of parts.
        enum AnyCodableContent: Decodable {
            case string(String)
            case parts([[String: String]])

            init(from decoder: any Swift.Decoder) throws {
                let container = try decoder.singleValueContainer()
                if let value = try? container.decode(String.self) {
                    self = .string(value)
                } else {
                    self = .parts(try container.decode([[String: String]].self))
                }
            }

            var sendable: any Sendable {
                switch self {
                case .string(let value): value
                case .parts(let parts): parts
                }
            }
        }

        static let directory = ProcessInfo.processInfo.environment["INKLING2_ORACLE_TEMPLATE_DIR"]
            .map { URL(fileURLWithPath: $0, isDirectory: true) }

        static func load(_ name: String) throws -> (Fixture, [CIImage], MLXArray) {
            let directory = try #require(Self.directory)
            let fixture = try JSONDecoder().decode(
                Fixture.self, from: Data(contentsOf: directory.appending(component: name + ".json"))
            )
            var images: [CIImage] = []
            for message in fixture.messages {
                guard case .parts(let parts) = message["content"] else { continue }
                for part in parts where part["type"] == "image" {
                    let url = directory.appending(component: "images").appending(
                        component: try #require(part["image"]))
                    images.append(try #require(CIImage(contentsOf: url)))
                }
            }
            let arrays = try loadArrays(
                url: directory.appending(component: fixture.pixel_values_file))
            return (fixture, images, try #require(arrays["pixel_values"]))
        }

        @Test(
            .enabled(if: directory != nil),
            arguments: ["10_image_single", "11_image_two_padded_rgba"])
        func pixelsAreBitIdentical(_ name: String) throws {
            let (fixture, images, expected) = try Self.load(name)
            let processor = Inkling2Processor(
                Inkling2ProcessorConfiguration(), tokenizer: TestTokenizer())
            let processed = try images.map { try processor.preprocess(image: $0, processing: nil) }
            let pixels = concatenated(processed.map(\.0), axis: 0)
            #expect(processed.map { [$0.1.t, $0.1.h, $0.1.w] } == fixture.image_grid_thw)
            #expect(pixels.shape == expected.shape)
            #expect(
                pixels.asArray(Float.self).map(\.bitPattern)
                    == expected.asArray(Float.self).map(\.bitPattern))
        }

        @Test(
            .enabled(if: directory != nil && Inkling2CheckpointTests.modelDirectory != nil),
            arguments: ["10_image_single", "11_image_two_padded_rgba"])
        func promptTokensMatch(_ name: String) async throws {
            let (fixture, images, _) = try Self.load(name)
            let tokenizer = try await #huggingFaceTokenizerLoader().load(
                from: try #require(Inkling2CheckpointTests.modelDirectory))
            let processor = Inkling2Processor(
                Inkling2ProcessorConfiguration(), tokenizer: tokenizer)
            let messages: [Message] = fixture.messages.map { message in
                message.mapValues(\.sendable)
            }
            var input = UserInput(
                prompt: .messages(messages), images: images.map { .ciImage($0) })
            if !fixture.kwargs.isEmpty {
                input.additionalContext = fixture.kwargs
            }
            let prepared = try await processor.prepare(input: input)
            #expect(prepared.text.tokens.asArray(Int32.self).map(Int.init) == fixture.input_ids)
            #expect(prepared.image?.frames?.map { [$0.t, $0.h, $0.w] } == fixture.image_grid_thw)
        }

        /// The chat-message path (images first, then the text) renders the
        /// image-first fixture exactly as the reference.
        @Test(.enabled(if: directory != nil && Inkling2CheckpointTests.modelDirectory != nil))
        func chatMessagesMatchTheImageFirstFixture() async throws {
            let (fixture, images, _) = try Self.load("10_image_single")
            let tokenizer = try await #huggingFaceTokenizerLoader().load(
                from: try #require(Inkling2CheckpointTests.modelDirectory))
            let processor = Inkling2Processor(
                Inkling2ProcessorConfiguration(), tokenizer: tokenizer)
            let input = UserInput(
                chat: [
                    .user("What shapes are in this image?", images: images.map { .ciImage($0) })
                ])
            let prepared = try await processor.prepare(input: input)
            #expect(prepared.text.tokens.asArray(Int32.self).map(Int.init) == fixture.input_ids)
        }
    }
}
