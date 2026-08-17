import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class CBv2AttentionExecutionCacheTests: XCTestCase {
    private let headDim = 256
    private let queryHeads = 16
    private let kvHeads = 2

    private var scale: Float { 1.0 / Float(headDim).squareRoot() }

    private var qualifiedKind: CBv2LayerKind {
        CBv2LayerKind(
            attention: .full,
            headDim: headDim,
            kvHeads: kvHeads,
            queryHeads: queryHeads,
            attentionExecutionQualification: CBv2AttentionExecutionQualification(
                architecture: .qwenLike))
    }

    private var hardwareQualification: CBv2AttentionHardwareQualification {
        .qwenLikeD256Hq16Hkv2GQA8BF16FullAttentionPrefill
    }

    private struct Run {
        var cache: CBv2LayerCache
        var output: MLXArray
        var observations: [CBv2AttentionExecutionObservation]
    }

    private func tensors(
        queryLength: Int,
        keyValueLength: Int? = nil
    ) -> (queries: MLXArray, keys: MLXArray, values: MLXArray) {
        let kvLength = keyValueLength ?? queryLength
        return (
            MLXArray.zeros(
                [1, queryHeads, queryLength, headDim], dtype: .bfloat16),
            MLXArray.zeros(
                [1, kvHeads, kvLength, headDim], dtype: .bfloat16),
            MLXArray.ones(
                [1, kvHeads, kvLength, headDim], dtype: .bfloat16)
        )
    }

    private func runPrefill(
        queryLength: Int,
        policy: CBv2AttentionExecutionPolicy? = nil,
        spanContext: CBv2SpanChunkContext? = nil,
        serializesQueries: Bool = false
    ) -> Run {
        let row = CBv2FullSequenceKV(
            promptLength: queryLength,
            maxLength: queryLength + 8,
            kvHeads: kvHeads,
            headDim: headDim)
        let cache: CBv2LayerCache
        if let policy {
            cache = CBv2LayerCache(
                layerIndex: 0,
                kind: qualifiedKind,
                rows: [row],
                attentionExecutionPolicy: policy)
        } else {
            cache = CBv2LayerCache(layerIndex: 0, kind: qualifiedKind, rows: [row])
        }

        var observations: [CBv2AttentionExecutionObservation] = []
        cache.attentionExecutionObserver = { observations.append($0) }
        cache.mtpSerializesRectangularAttention = serializesQueries
        cache.bindSpanContext(spanContext)

        let tensors = tensors(queryLength: queryLength)
        let output = cache.updateAndAttend(
            queries: tensors.queries,
            keys: tensors.keys,
            values: tensors.values,
            scale: scale,
            sinks: nil)
        return Run(cache: cache, output: output, observations: observations)
    }

    private func runLastQuery(
        policy: CBv2AttentionExecutionPolicy
    ) -> Run {
        let keyValueLength = 2
        let row = CBv2FullSequenceKV(
            promptLength: keyValueLength,
            maxLength: keyValueLength + 8,
            kvHeads: kvHeads,
            headDim: headDim)
        let cache = CBv2LayerCache(
            layerIndex: 0,
            kind: qualifiedKind,
            rows: [row],
            attentionExecutionPolicy: policy)
        var observations: [CBv2AttentionExecutionObservation] = []
        cache.attentionExecutionObserver = { observations.append($0) }

        let tensors = tensors(queryLength: 1, keyValueLength: keyValueLength)
        let output = cache.updateAndAttendLastQuery(
            queries: tensors.queries,
            keys: tensors.keys,
            values: tensors.values,
            scale: scale,
            sinks: nil)
        return Run(cache: cache, output: output, observations: observations)
    }

    func testDefaultCacheMatchesExplicitFallbackOutputAndGraphRoute() {
        XCTAssertNil(
            ProcessInfo.processInfo.environment[
                CBv2AttentionExecutionPolicy.environmentVariable])

        Device.withDefaultDevice(.cpu) {
            let defaultRun = runPrefill(queryLength: 129)
            let fallbackRun = runPrefill(
                queryLength: 129,
                policy: .fallback)

            XCTAssertEqual(defaultRun.observations, fallbackRun.observations)
            XCTAssertEqual(
                defaultRun.observations,
                [
                    CBv2AttentionExecutionObservation(
                        route: .fallback, path: .queryBlocked, queryLength: 129)
                ])
            XCTAssertEqual(defaultRun.output.shape, fallbackRun.output.shape)
            XCTAssertEqual(defaultRun.output.dtype, fallbackRun.output.dtype)
            XCTAssertEqual(
                defaultRun.cache.innerState().map(\.shape),
                fallbackRun.cache.innerState().map(\.shape))
            XCTAssertTrue(arrayEqual(defaultRun.output, fallbackRun.output).item(Bool.self))
        }
    }

    func testForcedFusedBypassesThe128QueryBlockOnActualCacheRoute() {
        XCTAssertEqual(CBv2AttentionV1.queryBlockSize, 128)

        Device.withDefaultDevice(.cpu) {
            let fallbackRun = runPrefill(
                queryLength: 129,
                policy: .fallback)
            let fusedRun = runPrefill(
                queryLength: 129,
                policy: CBv2AttentionExecutionPolicy(control: .fused))

            XCTAssertEqual(
                fallbackRun.observations,
                [
                    CBv2AttentionExecutionObservation(
                        route: .fallback, path: .queryBlocked, queryLength: 129)
                ])
            XCTAssertEqual(
                fusedRun.observations,
                [
                    CBv2AttentionExecutionObservation(
                        route: .forcedFused, path: .singleCall, queryLength: 129)
                ])
            XCTAssertEqual(fusedRun.output.shape, fallbackRun.output.shape)
        }
    }

    func testAutoCacheRequiresExternalHardwareQualification() {
        Device.withDefaultDevice(.cpu) {
            let unqualified = runPrefill(
                queryLength: 2,
                policy: CBv2AttentionExecutionPolicy(control: .auto))
            let qualified = runPrefill(
                queryLength: 2,
                policy: CBv2AttentionExecutionPolicy(
                    control: .auto,
                    hardwareQualification: hardwareQualification))

            XCTAssertEqual(unqualified.observations.first?.route, .fallback)
            XCTAssertEqual(qualified.observations.first?.route, .forcedFused)
        }
    }

    func testSpanContextExcludesForcedFusedOnActualCacheRoute() {
        Device.withDefaultDevice(.cpu) {
            let run = runPrefill(
                queryLength: 2,
                policy: CBv2AttentionExecutionPolicy(control: .fused),
                spanContext: CBv2SpanChunkContext(
                    chunkEnd: 2,
                    blocks: [CBv2ImageSpan(tokenOffset: 0, length: 2)]))

            XCTAssertEqual(
                run.observations,
                [
                    CBv2AttentionExecutionObservation(
                        route: .fallback, path: .span, queryLength: 2)
                ])
        }
    }

    func testDecodeMTPAndLastQueryExcludeForcedFusedOnActualCacheRoutes() {
        Device.withDefaultDevice(.cpu) {
            let policy = CBv2AttentionExecutionPolicy(control: .fused)
            let decode = runPrefill(queryLength: 1, policy: policy)
            let mtp = runPrefill(
                queryLength: 2,
                policy: policy,
                serializesQueries: true)
            let lastQuery = runLastQuery(policy: policy)

            XCTAssertEqual(
                decode.observations.first,
                CBv2AttentionExecutionObservation(
                    route: .fallback, path: .singleCall, queryLength: 1))
            XCTAssertEqual(
                mtp.observations.first,
                CBv2AttentionExecutionObservation(
                    route: .fallback, path: .serializedQueries, queryLength: 2))
            XCTAssertEqual(
                lastQuery.observations.first,
                CBv2AttentionExecutionObservation(
                    route: .fallback, path: .lastQuery, queryLength: 1))
        }
    }
}
