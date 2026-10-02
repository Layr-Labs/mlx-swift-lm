import CoreGraphics
import Foundation
import ImageIO
import MLXLLM
import MLXLMCommon
import XCTest

@testable import MLXVLM

/// Prepared only. Real ImageIO/CG raster APIs; no CoreImage renderer, model,
/// fake decoder or plaintext fixture file.
final class MiMoV26EncodedVisualDecoderTests: XCTestCase {
    private let mp4Base64 =
        "AAAAHGZ0eXBtcDQyAAAAAWlzb21tcDQxbXA0MgAAAAFtZGF0AAAAAAAAAK4AAAA7BgUyR1ZK3FxMQz+U78URPNFDqAEAAAMAAQMAAAMAAQIAAeYACwAAAwAA"
        + "AwAAAwAUDAOJJAEN/////4AAAAAxJbggH4AuSqwRNmYXSACJwyG5akafRwrPDoFqVCtjHBP+QvRWhyAAGk1PzfAEsEedgAAAABEh4QhfAoAvQrFXFN4ACQ7CtgA"
        + "AABEBqIGK/1jQw/VufW+ACvdnuAAAAvFtb292AAAAbG12aGQAAAAA5lOws+ZTsLMAAAJYAAACWAABAAABAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAA"
        + "AAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAAACfXRyYWsAAABcdGtoZAAAAAHmU7Cz5lOwswAAAAEAAAAAAAACWAAAAAAAAAAA"
        + "AAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAEAAAAAAQAAAAEAAAAAAACRlZHRzAAAAHGVsc3QAAAAAAAAAAQAAAlgAAADIAAEAAAAAAfV"
        + "tZGlhAAAAIG1kaGQAAAAA5lOws+ZTsLMAAAJYAAACWFXEAAAAAAAxaGRscgAAAAAAAAAAdmlkZQAAAAAAAAAAAAAAAENvcmUgTWVkaWEgVmlkZW8AAAABnG1pbm"
        + "YAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAAVxzdGJsAAAAoXN0c2QAAAAAAAAAAQAAAJFhdmMxAAAAAA"
        + "AAAAEAAAAAAAAAAAAAAAAAAAAAAEAAQABIAAAASAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAGP//AAAAJ2F2Y0MBZAAL/+EADCdkAA"
        + "usVlDDeBBhFAEABCjuPLD9+PgAAAAACmZpZWwBAAAAAApjaHJtAAAAAAAYc3R0cwAAAAAAAAABAAAAAwAAAMgAAAAoY3R0cwAAAAAAAAADAAAAAQAAAMgAAAABAA"
        + "ABkAAAAAEAAAAAAAAAFHN0c3MAAAAAAAAAAQAAAAEAAAAPc2R0cAAAAAAgEBgAAAAcc3RzYwAAAAAAAAABAAAAAQAAAAMAAAABAAAAIHN0c3oAAAAAAAAAAAAAAA"
        + "MAAAB0AAAAFQAAABUAAAAUc3RjbwAAAAAAAAABAAAALA=="
    private let limits = MiMoV26EncodedVisualDecoder.Limits(
        maximumPixels: 10000,
        maximumWorkingBytes: 16 << 20, maximumSourceFrames: 10000, maximumSampledFrames: 64)
    private func image() throws -> CGImage {
        let bytes: [UInt8] = [
            1, 41, 81, 0, 2, 42, 82, 127, 3, 43, 83, 255,
            4, 44, 84, 255, 5, 45, 85, 255, 6, 46, 86, 255,
        ]
        let provider = try XCTUnwrap(CGDataProvider(data: Data(bytes) as CFData))
        return try XCTUnwrap(
            CGImage(
                width: 2, height: 3, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: 8, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue).union(
                    .byteOrder32Big),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    }
    func testNativeSmartFramesAndIndexSamplingNotNearestPresentationTime() throws {
        let sampling = try MiMoV26EncodedVisualDecoder.Sampling(
            fps: 1, minimumFrames: 8, maximumFrames: 3600)
        XCTAssertEqual(
            try sampling.indices(totalFrames: 3, averageFPS: 3, admissionLimit: 64), [0, 2])
        XCTAssertEqual(
            try sampling.indices(totalFrames: 9, averageFPS: 30, admissionLimit: 64),
            [0, 1, 2, 3, 4, 5, 6, 8])
        XCTAssertEqual(
            try sampling.indices(totalFrames: 16, averageFPS: 4, admissionLimit: 64),
            [0, 2, 4, 6, 8, 10, 12, 15])
        XCTAssertThrowsError(
            try sampling.indices(totalFrames: 16, averageFPS: 4, admissionLimit: 7)
        ) {
            XCTAssertEqual($0 as? MiMoV26EncodedVisualDecoder.Failure, .limit)
        }
        XCTAssertThrowsError(
            try sampling.indices(totalFrames: 1, averageFPS: 1, admissionLimit: 64))
        XCTAssertThrowsError(
            try sampling.indices(totalFrames: 8, averageFPS: .nan, admissionLimit: 64))
        let odd = try MiMoV26EncodedVisualDecoder.Sampling(
            fps: 1, minimumFrames: 9, maximumFrames: 100)
        XCTAssertEqual(
            try odd.indices(totalFrames: 11, averageFPS: 30, admissionLimit: 64),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 10])
    }
    func testAllEightLosslessOrientationsKeepStraightHiddenRGBAndChannelOrder() throws {
        let expected: [[Float]] = [
            [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5], [6, 5, 4, 3, 2, 1], [5, 6, 3, 4, 1, 2],
            [1, 3, 5, 2, 4, 6], [5, 3, 1, 6, 4, 2], [6, 4, 2, 5, 3, 1], [2, 4, 6, 1, 3, 5],
        ]
        for orientation in 1 ... 8 {
            let rgb = try MiMoV26EncodedVisualDecoder.straightRGB(
                image(), orientation: orientation, limits: limits)
            XCTAssertEqual(rgb.width, orientation >= 5 ? 3 : 2)
            XCTAssertEqual(rgb.height, orientation >= 5 ? 2 : 3)
            XCTAssertEqual(Array(rgb.planarRGB[0 ..< 6]), expected[orientation - 1])
            XCTAssertEqual(
                Array(rgb.planarRGB[6 ..< 12]), expected[orientation - 1].map { $0 + 40 })
            XCTAssertEqual(
                Array(rgb.planarRGB[12 ..< 18]), expected[orientation - 1].map { $0 + 80 })
        }
    }
    func testSamplingRejectsInvertedRoundedBoundsFromBothInitializers() throws {
        // Metadata-only configuration: no weights, native load or fixture-path prerequisite.
        let raw = Data(
            """
            {"model_type":"mimo_v2","architectures":["MiMoV2ForCausalLM"],
             "hidden_size":16,"intermediate_size":32,"moe_intermediate_size":8,
             "vocab_size":64,"num_hidden_layers":2,"max_position_embeddings":64,
             "sliding_window_size":3,"sliding_window":3,"num_nextn_predict_layers":3,
             "hybrid_layer_pattern":[0,1],"moe_layer_freq":[0,1],
             "partial_rotary_factor":0.5,"attention_value_scale":0.707,
             "layernorm_epsilon":0.000001,"attention_projection_layout":"split",
             "moe_router_dtype":"bfloat16","hidden_act":"silu","dtype":"float32",
             "attention_bias":false,"tie_word_embeddings":false,"attention_dropout":0,
             "scoring_func":"sigmoid","topk_method":"noaux_tc","n_routed_experts":4,
             "num_experts_per_tok":2,"n_group":1,"topk_group":1,"norm_topk_prob":true,
             "n_shared_experts":null,"routed_scaling_factor":null,
             "num_attention_heads":4,"num_key_value_heads":1,"head_dim":8,"v_head_dim":4,
             "swa_num_attention_heads":4,"swa_num_key_value_heads":2,"swa_head_dim":8,
             "swa_v_head_dim":4,"rope_theta":10000000,"swa_rope_theta":10000,
             "add_full_attention_sink_bias":false,"add_swa_attention_sink_bias":true,
             "eos_token_id":3,"pad_token_id":0}
            """.utf8)
        var fields = try JSONDecoder().decode([String: MiMoV26JSONValue].self, from: raw)
        for (low, high) in [(10, 8), (9, 9), (1, 1)] {
            XCTAssertThrowsError(
                try MiMoV26EncodedVisualDecoder.Sampling(
                    fps: 1, minimumFrames: low, maximumFrames: high)
            ) {
                XCTAssertEqual($0 as? MiMoV26EncodedVisualDecoder.Failure, .invalidVideo)
            }
            fields["processor_config"] = .object([
                "fps": .number(1), "min_frames": .number(Decimal(low)),
                "max_frames": .number(Decimal(high)),
            ])
            let configuration = try MiMoV26Configuration(rawFields: fields)
            XCTAssertThrowsError(
                try MiMoV26EncodedVisualDecoder.Sampling(configuration: configuration)
            ) {
                XCTAssertEqual($0 as? MiMoV26EncodedVisualDecoder.Failure, .invalidVideo)
            }
        }
        XCTAssertThrowsError(
            try MiMoV26EncodedVisualDecoder.Sampling(
                fps: 1, minimumFrames: Int.max, maximumFrames: Int.max))
        let valid = try MiMoV26EncodedVisualDecoder.Sampling(
            fps: 1, minimumFrames: 9, maximumFrames: 10)
        XCTAssertEqual(
            try valid.indices(totalFrames: 10, averageFPS: 1, admissionLimit: 10), Array(0 ..< 10))
    }
    func testEncodedImageByteCeilingPrecedesPlatformParsingAndIncludesPadding() throws {
        let encoded = NSMutableData()
        let destination = try XCTUnwrap(
            CGImageDestinationCreateWithData(encoded, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, try image(), nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        let data = encoded as Data
        let bounded = MiMoV26EncodedVisualDecoder.Limits(
            maximumPixels: 10000,
            maximumWorkingBytes: 16 << 20, maximumSourceFrames: 10000,
            maximumSampledFrames: 64, maximumEncodedBytes: data.count)
        XCTAssertEqual(try MiMoV26EncodedVisualDecoder.image(data, limits: bounded).width, 2)
        var padded = data
        padded.append(Data(repeating: 0, count: 1024))
        for rejected in [padded, Data(repeating: 0, count: data.count + 1)] {
            XCTAssertThrowsError(try MiMoV26EncodedVisualDecoder.image(rejected, limits: bounded)) {
                XCTAssertEqual($0 as? MiMoV26EncodedVisualDecoder.Failure, .limit)
            }
        }
        XCTAssertEqual(limits.maximumEncodedBytes, limits.maximumWorkingBytes)
    }
    func testEncodedVideoByteCeilingPrecedesAssetReadsAndAppliesToReusedPlan() async throws {
        let data = try XCTUnwrap(Data(base64Encoded: mp4Base64))
        let owner = try MemoryBackedVideoAsset(videoData: data)
        let sampling = try MiMoV26EncodedVisualDecoder.Sampling(
            fps: 1, minimumFrames: 8, maximumFrames: 3600)
        let small = MiMoV26EncodedVisualDecoder.Limits(
            maximumPixels: 10000,
            maximumWorkingBytes: 16 << 20, maximumSourceFrames: 10000,
            maximumSampledFrames: 64, maximumEncodedBytes: data.count - 1)
        XCTAssertEqual(owner.resourceRequestCount, 0)
        do {
            _ = try await MiMoV26EncodedVisualDecoder.inspectVideo(
                owner, sampling: sampling, limits: small)
            XCTFail("oversized encoded video reached AVFoundation")
        } catch { XCTAssertEqual(error as? MiMoV26EncodedVisualDecoder.Failure, .limit) }
        XCTAssertEqual(owner.resourceRequestCount, 0)
        let plan = try await MiMoV26EncodedVisualDecoder.inspectVideo(
            owner, sampling: sampling, limits: limits)
        // The real reader emits zero-sample, zero-duration control markers.
        // They must not become frames or turn valid content into a limit error.
        XCTAssertEqual(plan.codedPixels, 64 * 64)
        XCTAssertEqual(plan.sourceFrameCount, 3)
        XCTAssertEqual(plan.averageFPS, 3)
        XCTAssertEqual(plan.sampledIndices, [0, 2])
        let decoded = try await MiMoV26EncodedVisualDecoder.silentVideo(plan, limits: limits)
        XCTAssertEqual(decoded.frames.count, 2)
        XCTAssertEqual(decoded.frames.map(\.width), [64, 64])
        XCTAssertEqual(decoded.frames.map(\.height), [64, 64])
        XCTAssertEqual(
            decoded.timestamps.map(\.bitPattern),
            [Float(0).bitPattern, (Float(2) / Float(3)).bitPattern])
        // A canceled caller must receive the synchronous resource refusal,
        // not enter a reader and only then notice cancellation. Do not infer
        // AVFoundation quiescence from a sampled resource-request count.
        let task = Task { () -> (Bool, MiMoV26EncodedVisualDecoder.Failure?) in
            withUnsafeCurrentTask { $0?.cancel() }
            let cancelled = Task.isCancelled
            do {
                _ = try await MiMoV26EncodedVisualDecoder.silentVideo(plan, limits: small)
                return (cancelled, nil)
            } catch { return (cancelled, error as? MiMoV26EncodedVisualDecoder.Failure) }
        }
        let result = await task.value
        XCTAssertTrue(result.0)
        XCTAssertEqual(result.1, .limit)
    }
    func testSourceFrameLimitCountsMediaIndependentlyOfControlMarkers() async throws {
        let data = try XCTUnwrap(Data(base64Encoded: mp4Base64))
        let sampling = try MiMoV26EncodedVisualDecoder.Sampling(
            fps: 1, minimumFrames: 8, maximumFrames: 3600)
        let exact = MiMoV26EncodedVisualDecoder.Limits(
            maximumPixels: 10000,
            maximumWorkingBytes: 16 << 20, maximumSourceFrames: 3,
            maximumSampledFrames: 64, maximumEncodedBytes: data.count)
        XCTAssertEqual(exact.maximumSourceFrames, 3)
        XCTAssertEqual(exact.maximumControlMarkers, 4096)
        let plan = try await MiMoV26EncodedVisualDecoder.inspectVideo(
            MemoryBackedVideoAsset(videoData: data), sampling: sampling, limits: exact)
        XCTAssertEqual(plan.sourceFrameCount, 3)
        XCTAssertEqual(plan.averageFPS, 3)
        XCTAssertEqual(plan.sampledIndices, [0, 2])
        let decoded = try await MiMoV26EncodedVisualDecoder.silentVideo(plan, limits: exact)
        XCTAssertEqual(decoded.frames.count, 2)
        XCTAssertEqual(decoded.frames.map(\.width), [64, 64])
        XCTAssertEqual(decoded.frames.map(\.height), [64, 64])
        XCTAssertEqual(
            decoded.timestamps.map(\.bitPattern),
            [Float(0).bitPattern, (Float(2) / Float(3)).bitPattern])

        let lower = MiMoV26EncodedVisualDecoder.Limits(
            maximumPixels: exact.maximumPixels,
            maximumWorkingBytes: exact.maximumWorkingBytes, maximumSourceFrames: 2,
            maximumSampledFrames: exact.maximumSampledFrames,
            maximumEncodedBytes: exact.maximumEncodedBytes)
        do {
            _ = try await MiMoV26EncodedVisualDecoder.inspectVideo(
                MemoryBackedVideoAsset(videoData: data), sampling: sampling, limits: lower)
            XCTFail("three media frames exceeded the unchanged two-frame ceiling")
        } catch {
            XCTAssertEqual(error as? MiMoV26EncodedVisualDecoder.Failure, .limit)
        }
    }

    func testActualRGBAImageIODecodeDoesNotWhiteCompositeOrNormalize() throws {
        let data = NSMutableData()
        let destination = try XCTUnwrap(
            CGImageDestinationCreateWithData(data, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, try image(), nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        let rgb = try MiMoV26EncodedVisualDecoder.image(data as Data, limits: limits)
        XCTAssertEqual(rgb.width, 2)
        XCTAssertEqual(rgb.height, 3)
        XCTAssertEqual(
            rgb.planarRGB, [1, 2, 3, 4, 5, 6, 41, 42, 43, 44, 45, 46, 81, 82, 83, 84, 85, 86])
    }
    func testActualJPEGEXIFOrientationMatchesApplicationPolicy() throws {
        let data = NSMutableData()
        let destination = try XCTUnwrap(
            CGImageDestinationCreateWithData(data, "public.jpeg" as CFString, 1, nil))
        CGImageDestinationAddImage(
            destination, try image(),
            [
                kCGImagePropertyOrientation: 6, kCGImageDestinationLossyCompressionQuality: 1.0,
            ] as [CFString: Any] as CFDictionary)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        let source = try XCTUnwrap(CGImageSourceCreateWithData(data, nil))
        let properties = try XCTUnwrap(
            CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
        XCTAssertEqual((properties[kCGImagePropertyOrientation] as? NSNumber)?.intValue, 6)
        let raw = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        let expected = try MiMoV26EncodedVisualDecoder.straightRGB(
            raw, orientation: 6, limits: limits)
        let actual = try MiMoV26EncodedVisualDecoder.image(data as Data, limits: limits)
        XCTAssertEqual(actual.width, expected.width)
        XCTAssertEqual(actual.height, expected.height)
        XCTAssertEqual(actual.planarRGB, expected.planarRGB)
        // Decoder/orientation evidence only, not JPEG identity with another codec.
    }
    func testBoundsRejectBeforeRasterCopyAndNoHeuristicPremultipliedRecovery() throws {
        let small = MiMoV26EncodedVisualDecoder.Limits(
            maximumPixels: 1,
            maximumWorkingBytes: 128, maximumSourceFrames: 2, maximumSampledFrames: 2)
        XCTAssertThrowsError(
            try MiMoV26EncodedVisualDecoder.straightRGB(image(), orientation: 1, limits: small))
        let provider = try XCTUnwrap(CGDataProvider(data: Data([UInt8(0), 0, 0, 0]) as CFData))
        let premultiplied = try XCTUnwrap(
            CGImage(
                width: 1, height: 1, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
                    .union(.byteOrder32Big),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        XCTAssertThrowsError(
            try MiMoV26EncodedVisualDecoder.straightRGB(
                premultiplied, orientation: 1, limits: limits))
        XCTAssertThrowsError(
            try MiMoV26EncodedVisualDecoder.image(Data("invalid".utf8), limits: limits))
    }
    func testTighterVideoPixelLimitRejectsBeforeCancelledReaderWork() async throws {
        // Reuse the existing real inline MP4 fixture; no generated decoder or
        // public hook. Inspection creates the actual immutable source plan.
        let owner = try MemoryBackedVideoAsset(videoData: XCTUnwrap(Data(base64Encoded: mp4Base64)))
        let sampling = try MiMoV26EncodedVisualDecoder.Sampling(
            fps: 1, minimumFrames: 8, maximumFrames: 3600)
        let plan = try await MiMoV26EncodedVisualDecoder.inspectVideo(
            owner, sampling: sampling, limits: limits)
        XCTAssertFalse(plan.hasAudioTrack)
        XCTAssertGreaterThan(plan.codedPixels, 1)
        XCTAssertLessThanOrEqual(try plan.decodeWorkingByteBound(), limits.maximumWorkingBytes)
        let tighter = MiMoV26EncodedVisualDecoder.Limits(
            maximumPixels: plan.codedPixels - 1,
            maximumWorkingBytes: limits.maximumWorkingBytes,
            maximumSourceFrames: limits.maximumSourceFrames,
            maximumSampledFrames: limits.maximumSampledFrames)
        // Bounds must win synchronously before withAsset/reader work. The old
        // implementation only checked pixels after copyNextSampleBuffer and
        // Task.checkCancellation, so it cannot return .limit in this case.
        let task = Task { () -> (Bool, MiMoV26EncodedVisualDecoder.Failure?) in
            withUnsafeCurrentTask { $0?.cancel() }
            let cancelled = Task.isCancelled
            do {
                _ = try await MiMoV26EncodedVisualDecoder.silentVideo(plan, limits: tighter)
                return (cancelled, nil)
            } catch {
                return (cancelled, error as? MiMoV26EncodedVisualDecoder.Failure)
            }
        }
        let result = await task.value
        XCTAssertTrue(result.0)
        XCTAssertEqual(result.1, .limit)
    }

}
