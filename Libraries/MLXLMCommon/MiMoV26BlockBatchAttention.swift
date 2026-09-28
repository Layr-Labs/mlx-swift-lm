// Copyright © 2026 Eigen Labs.
// SPDX-License-Identifier: Apache-2.0 AND MIT
// Block-batch intent: oMLX #3972/#3994 and Fusion31/32. Arithmetic is the
// unchanged native-rounded attention_nax_bdv, not upstream online softmax.
import Foundation
import MLX
import MLXFast

/// Engine-installed bound only; no public caller may assert scratch admission.
/// Root integration must charge fixedRequestBytes on the SAME real per-request
/// admission before installing this immutable value in the MiMo cache path.
package final class MiMoV26BlockBatchBudget: Sendable {
    package let engineID: UUID
    package let modelIdentity, backendIdentity, cacheProviderIdentity: ObjectIdentifier
    package let maximumQueries: Int
    package let fixedRequestBytes: Int
    package init(engineID: UUID, model: AnyObject, backend: AnyObject, cacheProvider: AnyObject,
                 maximumQueries: Int, layerCount: Int, policy: AllocationFootprintPolicy) throws {
        self.engineID = engineID
        modelIdentity = ObjectIdentifier(model); backendIdentity = ObjectIdentifier(backend)
        cacheProviderIdentity = ObjectIdentifier(cacheProvider)
        self.maximumQueries = maximumQueries
        fixedRequestBytes = try MiMoV26BlockBatchAttention.fixedRequestScratchBytes(
            maximumQueries: maximumQueries, layerCount: layerCount, policy:policy)
    }
}

public enum MiMoV26BlockBatchAttention {
    public static let environmentKey = "DARKBLOOM_MIMO_BLOCK_BATCH_PREFILL"
    static let requested = ProcessInfo.processInfo.environment[environmentKey] == "1"
    static let blockSize = 128
    static let maximumBlocksPerDispatch = 4
    public enum Refusal: Error { case invalidBound, overflow }

    /// Conservative ADDITIVE workspace, not KV or a measured physical peak.
    /// Prices all layer graphs and two simultaneous forwards independently:
    /// native scaled-Q/output, original+packed masks, descriptors and page slack.
    /// No subtraction from the existing host activation/OS/KV reserve.
    public static func fixedRequestScratchBytes(maximumQueries q: Int, layerCount layers: Int,
                                                policy: AllocationFootprintPolicy) throws -> Int {
        guard let extra = policy.maximumExtraBytes else { throw Refusal.invalidBound }
        return try scratchBytes(maximumQueries:q,layerCount:layers,maximumExtraBytes:extra,
                                upperBound:policy.upperBound(byteCount:))
    }
    static func scratchBytes(maximumQueries q: Int,layerCount layers: Int,maximumExtraBytes: Int,
                             upperBound: (Int) -> Int?) throws -> Int {
        guard (256...8192).contains(q), (1...48).contains(layers) else { throw Refusal.invalidBound }
        func mul(_ a: Int,_ b: Int) throws -> Int {
            let r = a.multipliedReportingOverflow(by:b)
            guard a >= 0,b >= 0,!r.overflow else { throw Refusal.overflow }; return r.partialValue
        }
        func add(_ a: Int,_ b: Int) throws -> Int {
            let r = a.addingReportingOverflow(b)
            guard a >= 0,b >= 0,!r.overflow else { throw Refusal.overflow }; return r.partialValue
        }
        let blocks = (q+127)/128
        guard maximumExtraBytes >= 0 else { throw Refusal.invalidBound }
        // Bound all partitions: each actual allocation's excess over logical
        // size is <=maximumExtraBytes from the captured allocator policy.
        func project(_ totalLogical: Int,_ maximumAllocations: Int) throws -> Int {
            guard maximumAllocations > 0,let one = upperBound(totalLogical),one >= totalLogical else {
                throw Refusal.invalidBound
            }
            return try add(one,mul(maximumAllocations-1,maximumExtraBytes))
        }
        let scaledQ = try project(mul(mul(q,64),192*2),blocks)
        let outputs = try project(mul(mul(q,64),128*2*2),blocks+1)
        let masks = try project(mul(q,2*255),mul(blocks,2))
        let descriptors = try project(mul(blocks,18*4),blocks)
        let small = try project(mul(blocks,8*512),mul(blocks,8))
        let device = try add(try add(scaledQ,outputs),try add(masks,try add(descriptors,small)))
        let host = try add(mul(blocks,18*4*2),16_384)
        return try mul(mul(try add(device,host),layers),2)
    }

    struct Descriptor: Equatable, Sendable {
        let queryStart, queryCount, keyStart, keyCount: Int
        let causal: Bool
        var alignedQuery: Bool { queryCount % 64 == 0 }
        var alignedKey: Bool { keyCount % 32 == 0 }
        func sameVariant(_ other: Self) -> Bool {
            causal == other.causal && alignedQuery == other.alignedQuery && alignedKey == other.alignedKey
        }
    }
    struct Plan: Sendable {
        let queryCount, keyCount, heads, kvHeads: Int
        let window: Int?
        let groups: [[Descriptor]]
    }

    /// Scalar shape validation only; no lazy strides/data read or GPU query.
    static func layout(queries: Int, keys: Int, heads: Int, kvHeads: Int,
                       window: Int?, admittedMaximumQueries: Int) -> Plan? {
        guard (256...8192).contains(admittedMaximumQueries),
              queries > 128, queries <= admittedMaximumQueries,
              keys >= queries, keys <= 1_048_576,
              (1...64).contains(heads), kvHeads > 0, heads.isMultiple(of:kvHeads),
              window == nil || window == 128 else { return nil }
        let history = keys-queries
        var groups: [[Descriptor]] = [], offset = 0
        while offset < queries {
            let count = min(128,queries-offset)
            // <=8-row tails must retain their original stock fallback. The
            // whole call declines before allocation; no partially built result.
            guard count > 8 else { return nil }
            let end = history+offset+count
            let start = window.map { max(0,history+offset+1-$0) } ?? 0
            let d = Descriptor(queryStart:offset,queryCount:count,keyStart:start,
                keyCount:end-start,causal:window == nil || end-start <= window!)
            if let last = groups.last, last.count < maximumBlocksPerDispatch,
               last.last!.sameVariant(d) {
                groups[groups.count-1].append(d)
            } else { groups.append([d]) }
            offset += count
        }
        // At least one genuinely aggregated launch; not a renamed scalar loop.
        guard groups.contains(where: { $0.count > 1 }) else { return nil }
        return .init(queryCount:queries,keyCount:keys,heads:heads,kvHeads:kvHeads,window:window,groups:groups)
    }

    static func makePlan(queries q: MLXArray, keys k: MLXArray, values v: MLXArray,
        scale: Float, sinks: MLXArray?, window: Int?, maximumQueries: Int, production: Bool) -> Plan? {
        guard q.ndim == 4,k.ndim == 4,v.ndim == 4,q.dim(0) == 1,k.dim(0) == 1,v.dim(0) == 1,
              q.dtype == .bfloat16 || q.dtype == .float16,k.dtype == q.dtype,v.dtype == q.dtype,
              q.dim(3) == 192,k.dim(3) == 192,v.dim(3) == 128,
              Array(k.shape.prefix(3)) == Array(v.shape.prefix(3)),
              scale.isFinite,scale > 0 else { return nil }
        if production && (q.dim(1) != 64 || ![4,8].contains(k.dim(1))) { return nil }
        if let sinks, sinks.shape != [q.dim(1)] || sinks.dtype != q.dtype { return nil }
        return layout(queries:q.dim(2),keys:k.dim(2),heads:q.dim(1),kvHeads:k.dim(1),
                      window:window,admittedMaximumQueries:maximumQueries)
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var encodings = 0
    public static func encodedDispatches() -> Int { lock.withLock { encodings } }

    // A narrow synchronous graph-build binding, like existing position scopes.
    // The engine installs its exact immutable charged capability; no arrays,
    // tasks, receipts or lifecycle authority live in this scope.
    @TaskLocal private static var activeBudget: MiMoV26BlockBatchBudget?
    static func withBudget<Result>(_ budget: MiMoV26BlockBatchBudget?,
                                   _ body: () throws -> Result) rethrows -> Result {
        try $activeBudget.withValue(budget,operation:body)
    }
    static func matchesCurrentBudget(_ budget: MiMoV26BlockBatchBudget?) -> Bool {
        guard let budget else { return false }
        return activeBudget === budget
    }

    /// Only the root-composed authenticated MiMo cache passes a charged budget.
    /// Non-NAX, nil budget, changed global block width and every unsupported
    /// shape decline before grouped native work. No private stream/eval/fence.
    static func tryAttention(queries: MLXArray,keys: MLXArray,values: MLXArray,
        scale: Float,sinks: MLXArray?,window: Int?,queryBlockSize: Int,
        budget: MiMoV26BlockBatchBudget?,enabled: Bool = requested) -> MLXArray? {
        guard enabled,MiMoV26NAXAttention.requested,queryBlockSize == blockSize,let budget,
              matchesCurrentBudget(budget),
              let plan = makePlan(queries:queries,keys:keys,values:values,scale:scale,
                sinks:sinks,window:window,maximumQueries:budget.maximumQueries,production:true) else { return nil }
        let stream = StreamOrDevice.default
        guard MiMoV26NAXGatherQMM.gpuStream(stream),MiMoV26NAXGatherQMM.naxAvailable else { return nil }
        let value = launch(queries:queries,keys:keys,values:values,scale:scale,sinks:sinks,plan:plan,stream:stream)
        lock.withLock { encodings += plan.groups.count }
        return value
    }

    private static let kernel = MLXFast.metalKernel(
        name:"mimo_v26_exact_block_batch_attention",
        inputNames:["q","k","v","mask","sinks","descriptors"],outputNames:["out"],
        source:MiMoV26BlockBatchMetalSources.source,
        header:MiMoV26NAXMetalSources.mlxHeader+"\n"+MiMoV26NAXAttentionMetalSources.header+"\n",
        ensureRowContiguous:false)

    /// Numerical test seam. Native arrays remain under the caller's real
    /// evaluation owner; this function never reports completion or admission.
    static func launch(queries: MLXArray,keys: MLXArray,values: MLXArray,
        scale: Float,sinks: MLXArray?,plan: Plan,stream: StreamOrDevice = .default) -> MLXArray {
        let nativeSinks = sinks.map { contiguous($0,stream:stream) } ?? MLXArray.zeros([1],dtype:queries.dtype)
        var outputs: [MLXArray] = []
        for group in plan.groups {
            let first = group.first!, end = group.last!.queryStart+group.last!.queryCount
            let length = end-first.queryStart
            let scaled = queries[0...,0...,first.queryStart..<end,0...]
                * MLXArray(scale).asType(queries.dtype)
            var words: [Int32] = [], masks: [MLXArray] = [], maskOffset = 0
            for d in group {
                // Exact same mask builder and physical key length as the old loop.
                let mode = CBv2AttentionV1.maskMode(L:d.queryCount,kL:d.keyCount,window:plan.window)
                if case .array(let mask) = mode {
                    masks.append(mask.reshaped([-1]))
                }
                // Original56byte AttnParams followed by4offset words. K/V
                // offsets address ORIGINAL row.update arrays; no key gathering.
                words += [1,Int32(plan.heads),192,Int32(d.queryCount),Int32(d.keyCount),
                    Int32(plan.heads/plan.kvHeads),Int32(bitPattern:scale.bitPattern),
                    Int32((d.queryCount+63)/64),Int32((d.keyCount+31)/32),
                    Int32(d.queryCount/64),Int32(d.keyCount/32),Int32(d.queryCount%64),
                    Int32(d.keyCount%32),Int32(d.keyCount-d.queryCount),
                    Int32(d.queryStart-first.queryStart),Int32(d.keyStart),Int32(maskOffset),0]
                if !d.causal { maskOffset += d.queryCount*d.keyCount }
            }
            let mask = masks.isEmpty ? MLXArray.zeros([1],dtype:.bool)
                : (masks.count == 1 ? masks[0] : concatenated(masks))
            let result = kernel([scaled,keys,values,mask,nativeSinks,MLXArray(words)],
                template:[("T",queries.dtype),("ALIGN_Q",first.alignedQuery),("ALIGN_K",first.alignedKey),
                    ("HAS_MASK",!first.causal),("DO_CAUSAL",first.causal),("HAS_SINKS",sinks != nil)],
                grid:(2*128,plan.heads,group.count),threadGroup:(128,1,1),
                outputShapes:[[1,length,plan.heads,128]],outputDTypes:[queries.dtype],stream:stream)
            precondition(result.count == 1)
            outputs.append(result[0].transposed(0,2,1,3))
        }
        return outputs.count == 1 ? outputs[0] : concatenated(outputs,axis:2)
    }
}
