import Foundation
import MLX
import MLXFast
import MLXHuggingFace
import Tokenizers
import XCTest
@testable import MLXLMCommon
@testable import MLXLLM
@testable import MLXVLM

/// Source-prepared only. Arithmetic/scope cells are not native owner proof.
/// Numerical cells require actual NAX hardware and never replace its result.
final class MiMoV26BlockBatchTests: XCTestCase {
    private enum Failure: Error { case injected, fixture, nativeCompletion }
    private func lane() throws {
        guard ProcessInfo.processInfo.environment["MIMO_V26_BLOCKBATCH_NATIVE_TESTS"] == "1",
              MiMoV26NAXGatherQMM.gpuStream(.default),MiMoV26NAXGatherQMM.naxAvailable else {
            throw XCTSkip("Explicit owned NAX native process required")
        }
    }
    func testExactDescriptorVisibilityAndOrderedCoverage() throws {
        for window: Int? in [nil,128] {
            for prefix in [0,1,127,128,129,8193] {
                for count in [256,300,384,511,512,1024,2048,4095] {
                    guard let plan = MiMoV26BlockBatchAttention.layout(queries:count,keys:prefix+count,
                        heads:64,kvHeads:8,window:window,admittedMaximumQueries:4096) else {
                        // Cold SWA256 has one causal + one window variant,
                        // so deliberately no real aggregate; old loop remains.
                        XCTAssertEqual(window,128); XCTAssertTrue([256,300].contains(count)); XCTAssertEqual(prefix,0)
                        continue
                    }
                    let flat = plan.groups.flatMap { $0 }
                    XCTAssertEqual(flat.reduce(0) { $0+$1.queryCount },count)
                    var cursor = 0
                    for d in flat {
                        XCTAssertEqual(d.queryStart,cursor)
                        let end = prefix+cursor+d.queryCount
                        let start = window.map { max(0,prefix+cursor+1-$0) } ?? 0
                        XCTAssertEqual(d.keyStart,start); XCTAssertEqual(d.keyCount,end-start)
                        XCTAssertEqual(d.causal,window == nil || end-start <= window!)
                        if window != nil { XCTAssertLessThanOrEqual(d.keyCount,255) }
                        cursor += d.queryCount
                    }
                    XCTAssertTrue(plan.groups.contains { $0.count > 1 })
                    for group in plan.groups {
                        XCTAssertLessThanOrEqual(group.count,4)
                        XCTAssertTrue(group.allSatisfy { $0.sameVariant(group[0]) })
                    }
                }
            }
        }
    }
    func testMalformedBoundsAndShortTailDeclineBeforeNativeWork() {
        for q in [1,8,9,127,128,129,136,257,264] {
            XCTAssertNil(MiMoV26BlockBatchAttention.layout(queries:q,keys:q,heads:64,kvHeads:4,
                window:nil,admittedMaximumQueries:2048))
        }
        for tuple in [(Int.max,Int.max,8192),(256,255,2048),(256,Int.max,2048),(256,256,Int.max)] {
            XCTAssertNil(MiMoV26BlockBatchAttention.layout(queries:tuple.0,keys:tuple.1,heads:64,kvHeads:4,
                window:nil,admittedMaximumQueries:tuple.2))
        }
        XCTAssertNil(MiMoV26BlockBatchAttention.layout(queries:512,keys:512,heads:64,kvHeads:7,
            window:nil,admittedMaximumQueries:2048))
        XCTAssertNil(MiMoV26BlockBatchAttention.layout(queries:512,keys:512,heads:64,kvHeads:8,
            window:256,admittedMaximumQueries:2048))
    }
    func testAllocatorPartitionBoundOverflowAndUnavailability() throws {
        func bound(_ n: Int) -> Int? { n+16384 }
        let small = try MiMoV26BlockBatchAttention.scratchBytes(maximumQueries:512,layerCount:2,
            maximumExtraBytes:16384,upperBound:bound)
        let wide = try MiMoV26BlockBatchAttention.scratchBytes(maximumQueries:2048,layerCount:2,
            maximumExtraBytes:16384,upperBound:bound)
        XCTAssertGreaterThan(wide,small); XCTAssertGreaterThan(small,0)
        XCTAssertThrowsError(try MiMoV26BlockBatchAttention.scratchBytes(maximumQueries:512,layerCount:2,
            maximumExtraBytes:Int.max,upperBound:bound))
        XCTAssertThrowsError(try MiMoV26BlockBatchAttention.scratchBytes(maximumQueries:512,layerCount:2,
            maximumExtraBytes:0,upperBound:{ _ in nil }))
        XCTAssertThrowsError(try MiMoV26BlockBatchAttention.scratchBytes(maximumQueries:512,layerCount:2,
            maximumExtraBytes:0,upperBound:{ $0-1 }))
        XCTAssertThrowsError(try MiMoV26BlockBatchAttention.scratchBytes(maximumQueries:Int.max,layerCount:2,
            maximumExtraBytes:16384,upperBound:bound))
    }
    func testNestedThrowingScopeRestoresAndRejectsUnarmedCrossEngineIdentity() throws {
        try lane()
        let policy = try XCTUnwrap(Memory.allocationFootprintPolicy())
        let owner = NSObject(),backend = NSObject(),bank = NSObject()
        // Metadata-only scope objects; never a fabricated execution contract,
        // row, loaded resource owner or successful retirement receipt.
        let a = try MiMoV26BlockBatchBudget(engineID:UUID(),model:owner,backend:backend,cacheProvider:bank,
            maximumQueries:512,layerCount:2,policy:policy)
        let b = try MiMoV26BlockBatchBudget(engineID:UUID(),model:owner,backend:backend,cacheProvider:bank,
            maximumQueries:512,layerCount:2,policy:policy)
        XCTAssertFalse(MiMoV26BlockBatchAttention.matchesCurrentBudget(nil))
        XCTAssertFalse(MiMoV26BlockBatchAttention.matchesCurrentBudget(a))
        try MiMoV26BlockBatchAttention.withBudget(a) {
            XCTAssertTrue(MiMoV26BlockBatchAttention.matchesCurrentBudget(a))
            XCTAssertFalse(MiMoV26BlockBatchAttention.matchesCurrentBudget(b))
            XCTAssertThrowsError(try MiMoV26BlockBatchAttention.withBudget(b) {
                XCTAssertTrue(MiMoV26BlockBatchAttention.matchesCurrentBudget(b))
                XCTAssertFalse(MiMoV26BlockBatchAttention.matchesCurrentBudget(a))
                throw Failure.injected
            })
            XCTAssertTrue(MiMoV26BlockBatchAttention.matchesCurrentBudget(a))
            MiMoV26BlockBatchAttention.withBudget(nil) {
                XCTAssertFalse(MiMoV26BlockBatchAttention.matchesCurrentBudget(a))
            }
            XCTAssertTrue(MiMoV26BlockBatchAttention.matchesCurrentBudget(a))
        }
        XCTAssertFalse(MiMoV26BlockBatchAttention.matchesCurrentBudget(a))
    }
    private let scale: Float = Float(1/sqrt(192.0))
    func testUnarmedAndForeignScopedComponentCallsDoNotEncode() throws {
        try lane()
        guard ProcessInfo.processInfo.environment["DARKBLOOM_MIMO_BLOCK_BATCH_PREFILL"] == "1",
              ProcessInfo.processInfo.environment["DARKBLOOM_MIMO_V26_NAX_ATTENTION"] == "1" else {
            throw XCTSkip("Exact process-start route flags required for discriminating fallback test")
        }
        let policy = try XCTUnwrap(Memory.allocationFootprintPolicy())
        let owner = NSObject(),backend = NSObject(),bank = NSObject()
        let a = try MiMoV26BlockBatchBudget(engineID:UUID(),model:owner,backend:backend,cacheProvider:bank,
            maximumQueries:512,layerCount:2,policy:policy)
        let b = try MiMoV26BlockBatchBudget(engineID:UUID(),model:owner,backend:backend,cacheProvider:bank,
            maximumQueries:512,layerCount:2,policy:policy)
        let q = MLXArray.zeros([1,64,256,192],dtype:.bfloat16)
        let k = MLXArray.zeros([1,4,256,192],dtype:.bfloat16)
        let v = MLXArray.zeros([1,4,256,128],dtype:.bfloat16)
        func call(_ budget: MiMoV26BlockBatchBudget?) -> MLXArray? {
            MiMoV26BlockBatchAttention.tryAttention(queries:q,keys:k,values:v,scale:scale,
                sinks:nil,window:nil,queryBlockSize:128,budget:budget)
        }
        let before = MiMoV26BlockBatchAttention.encodedDispatches()
        XCTAssertNil(call(nil)); XCTAssertNil(call(a))
        MiMoV26BlockBatchAttention.withBudget(a) {
            XCTAssertNil(call(nil)); XCTAssertNil(call(b))
        }
        XCTAssertEqual(MiMoV26BlockBatchAttention.encodedDispatches(),before)
        let result = try MiMoV26BlockBatchAttention.withBudget(a) { try XCTUnwrap(call(a)) }
        try withError { error in eval(result); try error.check() }
        XCTAssertEqual(MiMoV26BlockBatchAttention.encodedDispatches(),before+1,
            "positive component control proves negatives did not merely use an unavailable kernel")
    }
    private func values(_ shape: [Int],salt: Int,dtype: DType) -> MLXArray {
        MLXArray((0..<shape.reduce(1,*)).map { sin(Float(($0+salt)*7))*0.12 })
            .reshaped(shape).asType(dtype)
    }
    private func exact(_ a: MLXArray,_ b: MLXArray,_ label: String) throws {
        try withError { error in eval(a,b); try error.check() }
        XCTAssertEqual(a.shape,b.shape,label); XCTAssertEqual(a.dtype,b.dtype,label)
        XCTAssertEqual(a.asData(access:.copy).data,b.asData(access:.copy).data,label)
    }
    private func compare(q: MLXArray,k: MLXArray,v: MLXArray,window: Int?,sinks: MLXArray?) throws -> MLXArray {
        let plan = try XCTUnwrap(MiMoV26BlockBatchAttention.makePlan(queries:q,keys:k,values:v,
            scale:scale,sinks:sinks,window:window,maximumQueries:4096,production:false))
        let candidate = MiMoV26BlockBatchAttention.launch(queries:q,keys:k,values:v,
            scale:scale,sinks:sinks,plan:plan)
        var reference: [MLXArray] = []
        // Baseline descriptors recomputed independently, not replayed from plan.
        let history = k.dim(2)-q.dim(2)
        var offset = 0
        while offset < q.dim(2) {
            let count = min(128,q.dim(2)-offset)
            let end = history+offset+count
            let start = window.map { max(0,history+offset+1-$0) } ?? 0
            let qs = q[0...,0...,offset..<offset+count,0...]
            let ks = k[0...,0...,start..<end,0...],vs = v[0...,0...,start..<end,0...]
            let mask = CBv2AttentionV1.maskMode(L:count,kL:end-start,window:window)
            let baseline = try XCTUnwrap(MiMoV26NAXAttention.makePlan(queries:qs,keys:ks,values:vs,
                scale:scale,mask:mask,sinks:sinks,production:false))
            reference.append(MiMoV26NAXAttention.launch(queries:qs,keys:ks,values:vs,
                scale:scale,plan:baseline))
            offset += count
        }
        try exact(candidate,concatenated(reference,axis:2),"exact ordinary q128 NAX baseline")
        return candidate
    }
    func testExactBytesDistinctKeyLengthsRaggedAndBooleanWindowMasks() throws {
        try lane()
        for dtype: DType in [.bfloat16,.float16] {
            for (count,prefix,window) in [(512,0,nil),(300,17,nil),(511,129,128),(512,127,128),(384,0,128)] as [(Int,Int,Int?)] {
                let q = values([1,4,count,192],salt:7,dtype:dtype)
                let k = values([1,2,count+prefix,192],salt:31,dtype:dtype)
                let v = values([1,2,count+prefix,128],salt:97,dtype:dtype)
                for useSinks in [false,true] {
                    let sinks = useSinks ? values([4],salt:11,dtype:dtype) : nil
                    _ = try compare(q:q,k:k,v:v,window:window,sinks:sinks)
                }
            }
        }
    }
    func testExactBytesWithNonunitHeadAndSequenceStrides() throws {
        try lane()
        let dtype: DType = .bfloat16, count = 384,prefix = 129
        let q = values([1,count,4,384],salt:2,dtype:dtype)
            .transposed(0,2,1,3)[.ellipsis,.stride(by:2)]
        let k = values([1,2,2*(count+prefix),384],salt:17,dtype:dtype)
            [0...,0...,MLXSlice.stride(by:2),MLXSlice.stride(by:2)]
        let v = values([1,2,2*(count+prefix),256],salt:43,dtype:dtype)
            [0...,0...,MLXSlice.stride(by:2),MLXSlice.stride(by:2)]
        _ = try compare(q:q,k:k,v:v,window:128,sinks:nil)
        _ = try compare(q:q,k:k,v:v,window:nil,sinks:nil)
    }
    func testOverflowAndNonfiniteFutureValuesNeverExpandFirstBlock() throws {
        try lane()
        let dtype: DType = .float16,count = 256
        let q = (MLXArray.ones([1,4,count,192],dtype:dtype)*2048).asType(dtype)
        let k = (MLXArray.ones([1,2,count,192],dtype:dtype)*(-2048)).asType(dtype)
        let overflow = matmul(q[0..<1,0..<1,0..<1,0...]*MLXArray(scale).asType(dtype),
                             k[0..<1,0..<1,0..<1,0...].swappedAxes(-1,-2))
        try withError { error in eval(overflow); try error.check() }
        XCTAssertTrue(overflow.asType(.float32).item(Float.self).isInfinite)
        for future: Float in [1,-2,.infinity,.nan] {
            let v = concatenated([MLXArray.zeros([1,2,128,128],dtype:dtype),
                MLXArray.ones([1,2,128,128],dtype:dtype)*future],axis:2).asType(dtype)
            let result = try compare(q:q,k:k,v:v,window:nil,sinks:nil)
            let first = result[0...,0...,0..<1,0...]
            XCTAssertTrue(first.asType(.float32).asArray(Float.self).allSatisfy { $0 == 0 })
        }
    }
    func testRawAdapterHasNoIssuedBankAndCannotInstallCallerBudget() throws {
        try lane()
        let (target,_) = try MiMoV26MTPChecks.fixture()
        let adapter = try MiMoV26CBv2Adapter(target:target)
        let budget = try MiMoV26BlockBatchBudget(engineID:UUID(),model:adapter,
            backend:NSObject(),cacheProvider:NSObject(),maximumQueries:512,layerCount:2,
            policy:XCTUnwrap(Memory.allocationFootprintPolicy()))
        XCTAssertFalse(adapter.cbv2TryInstallBlockBatchBudget(budget))
        XCTAssertFalse(MiMoV26BlockBatchAttention.matchesCurrentBudget(budget))
    }

    private final class Permit: MiMoV26SerialLoadReservation, Sendable {
        let request: MiMoV26SerialLoadRequest
        var reservedLoadBytes: UInt64 { request.requiredLoadBytes }
        init(_ request: MiMoV26SerialLoadRequest) { self.request = request }
        func validateActive(progress: MiMoV26SerialLoadProgress) throws {}
    }
    private struct Loader: TokenizerLoader {
        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
            let raw = try await AutoTokenizer.from(modelFolder:directory)
            return #adaptHuggingFaceTokenizer(raw)
        }
    }
    private struct Built: Sendable {
        let engine: EngineV2
        let contract: CBv2NativeExecutionContract
    }
    func testActualStrictIssuedEngineMtpOffChargesOnceAndKeepsOriginalBinding() async throws {
        try lane()
        let env = ProcessInfo.processInfo.environment
        guard env["DARKBLOOM_MIMO_BLOCK_BATCH_PREFILL"] == "1",
              env["DARKBLOOM_MIMO_V26_NAX_ATTENTION"] == "1",
              let root = env["MIMO_V26_BLOCKBATCH_NATIVE_FIXTURE"] else {
            throw XCTSkip("Separate strict tiny192/128 Q64/window128 fixture and actual route flags required")
        }
        // Existing strict loader, real tensor payload and actual tokenizer.
        // This selector never substitutes a conformance marker or fake receipt.
        let directory = URL(fileURLWithPath:root)
        let p = try XCTUnwrap(JSONSerialization.jsonObject(with:
            Data(contentsOf:directory.appendingPathComponent("provenance.json"))) as? [String:String])
        let provenance = MiMoV26ConvertedProvenance(artifactID:try XCTUnwrap(p["artifactID"]),
            sourceRepository:try XCTUnwrap(p["sourceRepository"]),sourceRevision:try XCTUnwrap(p["sourceRevision"]),
            conversionManifestSHA256:try XCTUnwrap(p["conversionManifestSHA256"]))
        let plan = try MiMoV26FilesystemWeights.preflight(root:directory,provenance:provenance,
            limits:.init(maximumShardBytes:64 << 20,maximumTotalFileBytes:128 << 20))
        let c = plan.bundlePlan.configuration
        guard c.hiddenSize <= 64,c.numHiddenLayers <= 4,c.maxPositionEmbeddings >= 520,
              [c.fullAttention,c.slidingAttention].allSatisfy({
                  $0.queryHeads == 64 && [4,8].contains($0.keyValueHeads)
                    && $0.headDim == 192 && $0.valueHeadDim == 128
              }),c.slidingWindow == 128 else { throw Failure.fixture }
        let session = try MiMoV26SerialLoadSession(plan:plan)
        let prepared = try await MiMoV26ModelFactory.prepare(request:session.request,
            configuration:.init(directory:directory),tokenizerLoader:Loader())
        let work = NativeConstructionWork()
        let container = try await MiMoV26ModelFactory.loadContainer(session:session,
            reservation:Permit(session.request),prepared:prepared,retaining:work)
        var retired = false
        defer {
            if !retired {
                _ = Unmanaged.passRetained(work); _ = Unmanaged.passRetained(container)
            }
        }
        try await work.acknowledgeContainerAdoption(container)
        let built = try await MiMoV26ModelFactory.withNativeConstruction(container:container,retaining:work) { model,scope in
            let binding = try model.makeCBv2Binding(enableMTP:false)
            _ = try binding.adapter.probeNativeKVTypes(retaining:scope)
            let resources = try binding.adapter.makeNativeExecutionResources(bytesCapacity:1 << 30,retaining:scope)
            let base = 4096
            let engine = EngineV2(model:binding.adapter,layerKinds:binding.adapter.layerKinds,
                backend:resources.backend,cacheProvider:resources.cacheProvider,
                schedulerConfig:.init(maxConcurrentRequests:1,maxBatchedTokensPerStep:512,
                    prefillChunkSize:512,enablePrefixCache:false),
                admissionConfig:.init(fixedBytesPerRequest:base),
                nativeCompletionTracking:true,nativeExecutionContract:resources.contract)
            try scope.retainOwner(engine)
            XCTAssertNil(engine.mtpMetricsSnapshot())
            XCTAssertNil(engine.groupedPrefillInactiveReason)
            XCTAssertGreaterThan(engine.groupedPrefillScratchBytes,0)
            XCTAssertEqual(engine.resolvedFixedBytesPerRequest,base+engine.groupedPrefillScratchBytes)
            let incompatible = try MiMoV26BlockBatchBudget(engineID:UUID(),model:binding.adapter,
                backend:resources.backend,cacheProvider:resources.cacheProvider,
                maximumQueries:1024,layerCount:c.numHiddenLayers,
                policy:XCTUnwrap(Memory.allocationFootprintPolicy()))
            XCTAssertFalse(binding.adapter.cbv2TryInstallBlockBatchBudget(incompatible),
                "second binding never replaces the first actual engine's cache bound")
            return Built(engine:engine,contract:resources.contract)
        }
        defer { if !retired { _ = Unmanaged.passRetained(built.engine) } }
        guard case .completed(let completion) = work.snapshot.disposition else { throw Failure.nativeCompletion }
        try await work.sealForPublication(completion)
        let before = MiMoV26BlockBatchAttention.encodedDispatches()
        let request = CBv2Request(id:.init(301),promptTokens:Array(repeating:1,count:512),
            sampling:.init(temperature:0),maxTokens:2,stopTokens:[],prefixCacheEnabled:false)
        let submission = try built.engine.submitWithNativeRetirement(request)
        let result = await cbv2SchedCollect(submission.events); await submission.retirement.wait()
        XCTAssertEqual(result.tokens.count,2)
        XCTAssertGreaterThan(MiMoV26BlockBatchAttention.encodedDispatches(),before)
        guard case .quiescent(let receipt) = await built.engine.shutdownReportingNativeCompletion() else {
            throw Failure.nativeCompletion
        }
        XCTAssertEqual(receipt.executionContractID,built.contract.id)
        retired = true
    }
}
