import Foundation
import MLX
import XCTest
@testable import MLXLLM
@testable import MLXLMCommon
@testable import MLXVLM


private func withMiMoConstructionScope<Value>(
    _ body: (NativeConstructionScope) throws -> Value
) rethrows -> Value {
    let work = NativeConstructionScope()
    defer {
        // Unexpected failed completion is restart-only, including in this
        // dedicated native test process. Never deallocate its sole SDK owner.
        if work.snapshot.isRetainedFault { _ = Unmanaged.passRetained(work) }
    }
    return try body(work)
}

// Direct untracked target/math fixtures. Scoped KV probes below do not issue a
// managed media execution profile or turn Void shutdown into F1 retirement.
final class MiMoV26CBv2MultimodalTests: XCTestCase {
    private func prepare(_ f: MiMoMediaFixture.Models, _ input: MiMoV26MultimodalInput) throws -> MiMoV26PreparedMultimodal {
        try f.processor.prepare(f.processor.plan(input),authorize:{ MiMoMediaFixture.Reservation($0) })
    }
    private static func collect(_ stream: AsyncStream<CBv2Event>) async -> ([Int],CBv2FinishReason?) {
        var tokens: [Int] = [], reason: CBv2FinishReason?
        for await event in stream {
            switch event {
            case .delta(_,let values,_): tokens += values
            case .finished(let value,_): reason = value
            }
        }
        return (tokens,reason)
    }
    /// Independent host span assembly; does not call engine splice/resolve.
    private func embeddings(_ target: MiMoV26TextModel, _ prompt: [Int],
                            _ spans: [MiMoV26MultimodalSpan], _ arrays: [MLXArray]) -> MLXArray {
        let text = target.model.embedTokens(MLXArray(prompt.map(Int32.init),[1,prompt.count]))
        var pieces: [MLXArray] = [], cursor = 0
        for (span,array) in zip(spans,arrays) {
            if cursor < span.tokenOffset { pieces.append(text[0...,cursor..<span.tokenOffset,0...]) }
            pieces.append(array.expandedDimensions(axis:0))
            cursor = span.tokenOffset+span.length
        }
        if cursor < prompt.count { pieces.append(text[0...,cursor..<prompt.count,0...]) }
        return concatenated(pieces,axis:1)
    }
    private func reference(_ f: MiMoMediaFixture.Models, _ prepared: MiMoV26PreparedMultimodal, id: UInt64) throws -> [Int] {
        let request = try prepared.makeRequest(binding:f.binding,id:.init(id),sampling:.init(temperature:0))
        let arrays = try XCTUnwrap(request.multimodal).embeddings(), cache = f.target.newCache()
        var output = try f.target.forward(embeddings:embeddings(f.target,request.promptTokens,prepared.plan.spans,arrays),cache:cache).logits
        var result: [Int] = []
        for step in 0..<request.maxTokens {
            let last = output[0...,-1,0...], next = argMax(last,axis:-1)
            eval(next); let token = Int(next.item(Int32.self)); result.append(token)
            if step+1 < request.maxTokens { output = try f.target.forward(inputIDs:MLXArray([Int32(token)],[1,1]),cache:cache).logits }
        }
        return result
    }
    func testRealEngineMixedImageVideoAndTextMatchOrdinaryNativeReference() async throws {
        let f = try MiMoMediaFixture.models()
        _ = try withMiMoConstructionScope { constructionWork in try f.adapter.probeNativeKVTypes(retaining: constructionWork) }
        XCTAssertTrue(f.adapter.layerKinds.allSatisfy { $0.headDim == 32 && $0.valueHeadDim == 16 })
        let image = MiMoMediaFixture.image(60), video = MiMoV26SilentVideo(frames:[image,MiMoMediaFixture.image(100),image],timestamps:[0,1,2])
        let inputs = [MiMoMediaFixture.request([.text("a"),.image(image),.text("tail")]),
                      MiMoMediaFixture.request([.silentVideo(video),.image(image),.text("b")])]
        var prepared: [MiMoV26PreparedMultimodal] = [], references: [[Int]] = []
        for (i,input) in inputs.enumerated() {
            references.append(try reference(f,prepare(f,input),id:UInt64(100+i)))
            prepared.append(try prepare(f,input))
        }
        let textPrompt = [20,21,22,23,24,25,26,27,28,29,30], textCache = f.target.newCache()
        var logits = try f.target.forward(inputIDs:MLXArray(textPrompt.map(Int32.init),[1,textPrompt.count]),cache:textCache).logits
        var textExpected: [Int] = []
        for i in 0..<3 {
            let next = argMax(logits[0...,-1,0...],axis:-1); eval(next)
            let token = Int(next.item(Int32.self)); textExpected.append(token)
            if i < 2 { logits = try f.target.forward(inputIDs:MLXArray([Int32(token)],[1,1]),cache:textCache).logits }
        }
        let backend = try f.adapter.makeBackend(bytesCapacity:32<<20)
        let engine = EngineV2(model:f.adapter,layerKinds:f.adapter.layerKinds,backend:backend,
            cacheProvider:try f.adapter.makeMultimodalCacheProvider(),
            schedulerConfig:.init(maxConcurrentRequests:3,maxBatchedTokensPerStep:5,prefillChunkSize:3,maxWaiting:8),
            loopConfig:.init(requestTimeout:30,stepTimeout:15,shutdownTimeout:5,useLegacyRequestTimeout:true),
            nativeCompletionTracking:false)
        do {
            let a = try engine.submit(prepared[0].makeRequest(binding:f.binding,id:.init(1),sampling:.init(temperature:0)))
            let b = try engine.submit(prepared[1].makeRequest(binding:f.binding,id:.init(2),sampling:.init(temperature:0)))
            let text = try engine.submit(.init(id:.init(3),promptTokens:textPrompt,sampling:.init(temperature:0),maxTokens:3,prefixCacheEnabled:false))
            async let resultA = Self.collect(a), resultB = Self.collect(b), resultText = Self.collect(text)
            let result = await (resultA,resultB,resultText)
            XCTAssertEqual(result.0.0,references[0]); XCTAssertEqual(result.1.0,references[1]); XCTAssertEqual(result.2.0,textExpected)
            XCTAssertEqual(result.0.1,.length); XCTAssertEqual(result.1.1,.length); XCTAssertEqual(result.2.1,.length)
            XCTAssertFalse(engine.packedPrefillActivity().isSupported)
            await engine.shutdown()
            XCTAssertEqual(backend.bytesReserved,0)
        } catch { await engine.shutdown(); throw error }
    }
    func testCausalChunksAndSWAWrapMatchFP32AndBF16Logits() throws {
        for dtype in ["float32","bfloat16"] {
            let f = try MiMoMediaFixture.models(dtype); _ = try withMiMoConstructionScope { constructionWork in try f.adapter.probeNativeKVTypes(retaining: constructionWork) }
            let value = try prepare(f,MiMoMediaFixture.request([.image(MiMoMediaFixture.image()),.text("0123456789")]))
            let request = try value.makeRequest(binding:f.binding,id:.init(11))
            let arrays = try XCTUnwrap(request.multimodal).embeddings()
            let whole = embeddings(f.target,request.promptTokens,value.plan.spans,arrays)
            let ordinary = f.target.newCache(), backend = try f.adapter.makeBackend(bytesCapacity:16<<20)
            let rows = try backend.makeSequenceState(layerKinds:f.adapter.layerKinds,promptLength:request.promptTokens.count,maxLength:128)
            let caches = f.adapter.makeCaches()
            for start in stride(from:0,to:request.promptTokens.count,by:3) {
                let end = min(start+3,request.promptTokens.count), ids = MLXArray(request.promptTokens[start..<end].map(Int32.init),[1,end-start])
                try f.adapter.bindRows([rows],caches:caches)
                let actual = try f.adapter.forwardValidated(tokens:ids,inputEmbeddings:whole[0...,start..<end,0...],caches:caches)
                let expected = try f.target.forward(embeddings:whole[0...,start..<end,0...],cache:ordinary).logits
                eval(actual,expected)
                let absolute: Float = dtype == "bfloat16" ? 0.02 : 5e-5
                let relative: Float = dtype == "bfloat16" ? 0.02 : 1e-4
                XCTAssertTrue(all(abs(actual-expected) .<= (absolute+relative*abs(expected))).item(Bool.self))
            }
            XCTAssertEqual(rows.compactMap { $0?.absoluteOffset },Array(repeating:request.promptTokens.count,count:rows.count))
            backend.release(rows)
        }
    }
    func testActualEngineCancellationDrainsRequestReservation() async throws {
        let f = try MiMoMediaFixture.models(); _ = try withMiMoConstructionScope { constructionWork in try f.adapter.probeNativeKVTypes(retaining: constructionWork) }
        let input = MiMoV26MultimodalInput(messages:[.init(role:.user,content:[.image(MiMoMediaFixture.image())])],maximumOutputTokens:100)
        let plan = try f.processor.plan(input)
        weak var reservation: MiMoMediaFixture.Reservation?
        var prepared: MiMoV26PreparedMultimodal? = try f.processor.prepare(plan,authorize:{ value in
            let owner = MiMoMediaFixture.Reservation(value); reservation = owner; return owner
        })
        let backend = try f.adapter.makeBackend(bytesCapacity:16<<20)
        let engine = EngineV2(model:f.adapter,layerKinds:f.adapter.layerKinds,backend:backend,
            cacheProvider:try f.adapter.makeMultimodalCacheProvider(),
            schedulerConfig:.init(maxConcurrentRequests:1,maxBatchedTokensPerStep:1,prefillChunkSize:1,maxWaiting:2),
            loopConfig:.init(requestTimeout:30,stepTimeout:15,shutdownTimeout:5,useLegacyRequestTimeout:true),
            nativeCompletionTracking:false)
        do {
            var request: CBv2Request? = try prepared!.makeRequest(binding:f.binding,id:.init(90),sampling:.init(temperature:0))
            let stream = try engine.submit(request!)
            prepared = nil; request = nil
            XCTAssertNotNil(reservation)
            engine.cancel(.init(90))
            let result = await Self.collect(stream)
            XCTAssertEqual(result.1,.cancelled)
            await engine.shutdown()
            XCTAssertNil(reservation); XCTAssertEqual(backend.bytesReserved,0)
        } catch { await engine.shutdown(); throw error }
    }
    func testCausalRefinementDoesNotFakeBidirectionalOrPackedCacheCapabilities() throws {
        let f = try MiMoMediaFixture.models()
        XCTAssertThrowsError(try f.adapter.makeMultimodalCacheProvider())
        _ = try withMiMoConstructionScope { constructionWork in try f.adapter.probeNativeKVTypes(retaining: constructionWork) }
        let provider = try f.adapter.makeMultimodalCacheProvider()
        XCTAssertFalse(provider.supportsMultimodalSpans); XCTAssertFalse(provider.supportsPackedPrefill)
        XCTAssertFalse(provider.supportsPackedMultimodalSpans)
        XCTAssertTrue((provider as! any CBv2MultimodalAttentionCapabilityProviding).supportsMultimodalPrefill(attention:.causal))
        XCTAssertFalse((provider as! any CBv2MultimodalAttentionCapabilityProviding).supportsMultimodalPrefill(attention:.bidirectionalSpans))
        var calls = 0
        let input = CBv2MultimodalInput(spans:[.init(tokenOffset:0,length:1)],attention:.causal) {
            calls += 1; return []
        }
        XCTAssertEqual(try CBv2MultimodalPlan.validate(input,promptTokenCount:2,model:f.adapter,cacheProvider:provider,maxBatchedTokensPerStep:1),[])
        XCTAssertThrowsError(try CBv2MultimodalPlan.validate(input,promptTokenCount:2,model:f.adapter,
            cacheProvider:CBv2LayerCacheBank(caches:f.adapter.makeCaches()),maxBatchedTokensPerStep:1))
        var bidirectional = input; bidirectional.attention = .bidirectionalSpans
        XCTAssertThrowsError(try CBv2MultimodalPlan.validate(bidirectional,promptTokenCount:2,model:f.adapter,cacheProvider:provider,maxBatchedTokensPerStep:1))
        XCTAssertEqual(calls,0)
        XCTAssertTrue(f.adapter.makeCaches().allSatisfy { !($0 is any CBv2SpanMaskBinding) })
    }
    func testForeignRowsWrongEmbeddingTypeAndReleasedRowsRefuseWithoutWrites() throws {
        let a = try MiMoMediaFixture.models(), b = try MiMoMediaFixture.models()
        _ = try withMiMoConstructionScope { constructionWork in try a.adapter.probeNativeKVTypes(retaining: constructionWork) }; _ = try withMiMoConstructionScope { constructionWork in try b.adapter.probeNativeKVTypes(retaining: constructionWork) }
        let backend = try a.adapter.makeBackend(bytesCapacity:16<<20)
        let rows = try backend.makeSequenceState(layerKinds:a.adapter.layerKinds,promptLength:4,maxLength:128)
        let before = backend.bytesReserved, caches = a.adapter.makeCaches()
        XCTAssertThrowsError(try b.adapter.bindRows([rows],caches:b.adapter.makeCaches()))
        try a.adapter.bindRows([rows],caches:caches)
        let tokens = MLXArray([Int32(20),21],[1,2])
        XCTAssertThrowsError(try a.adapter.forwardValidated(tokens:tokens,inputEmbeddings:MLXArray.zeros([1,2,64],dtype:.int32),caches:caches))
        XCTAssertTrue(rows.allSatisfy { $0?.absoluteOffset == 0 }); XCTAssertEqual(backend.bytesReserved,before)
        backend.release(rows)
        XCTAssertThrowsError(try a.adapter.bindRows([rows],caches:caches))
        XCTAssertEqual(backend.bytesReserved,0)
    }
    // Preserve this original test selector. Paging stays refused; the former
    // installed-assistant refusal is replaced by actual successful EngineV2
    // target-only media with the same native trained text assistant installed.
    func testPagedAndInstalledAssistantRemainExplicitlyUnqualified() async throws {
        let f = try MiMoMediaFixture.models()
        let paged = try PagedKVBackend(layerKinds:[.init(attention:.full,headDim:64,kvHeads:1,queryHeads:1)],
            config:.init(capacityBytes:1<<20,maxPrefillChunk:16,nominalMaxSequenceLength:128,maxBufferLength:8<<20))
        XCTAssertNotNil(EngineV2.backendCapabilityViolation(capabilities:f.adapter.cbv2Capabilities,backend:paged))
        let mtp = try MiMoV26MTP(target:f.target)
        try mtp.loadConvertedWeights(Dictionary(uniqueKeysWithValues:mtp.parameters().flattened().map { name,array in
            ("mtp."+name,MiMoMediaFixture.values(name,array.shape,name.hasSuffix("e_score_correction_bias") ? .float32 : name.hasSuffix("mlp.gate.weight") ? .bfloat16 : .float32))
        }))
        let assistant = try MiMoV26MTPAssistant(target:f.target,predictor:mtp)
        let adapter = try MiMoV26CBv2Adapter(target:f.target,assistant:assistant)
        XCTAssertTrue(adapter.supportsMultimodalPrefill(attention:.causal))
        XCTAssertFalse(adapter.cbv2Capabilities.supportsPagedKV)
        let prepared = try MiMoMediaMTPFixture.prepared(f,adapter,output:3)
        let (engine,backend) = try MiMoMediaMTPFixture.engine(adapter)
        do {
            let request = try prepared.makeRequest(binding:MiMoMediaMTPFixture.binding(f,adapter),id:.init(501),sampling:.init(temperature:0))
            let result = await MiMoMediaMTPFixture.collect(try engine.submit(request))
            XCTAssertEqual(result.0.count,3); XCTAssertEqual(result.1,.length)
            let metrics = try XCTUnwrap(engine.mtpMetricsSnapshot())
            XCTAssertTrue(metrics.active); XCTAssertEqual(metrics.draftedTokens,0)
            await engine.shutdown(); XCTAssertEqual(backend.bytesReserved,0)
        } catch { await engine.shutdown(); throw error }
    }
}
