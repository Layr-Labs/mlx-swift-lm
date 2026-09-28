import Foundation
import MLX
import MLXFast
import MLXRandom
import XCTest
@testable import MLXLMCommon

/// SOURCE-PREPARED. No tolerance-only pass is a model/state/lossless claim.
/// Exact control tests keep their byte gate; random matrix diagnostics do not
/// invent a broader ULP threshold when QK reduction changes association.
final class MiMoV26SplitKeyAttentionTests: XCTestCase {
    private let scale: Float = 0.07216878
    private func requireMatrixGPU() throws -> Character {
        guard ProcessInfo.processInfo.environment["DARKBLOOM_TEST_MIMO_SPLITKEY_QK"] == "1",
              MiMoV26NAXGatherQMM.gpuStream(.default), MiMoV26NAXGatherQMM.naxAvailable else {
            throw XCTSkip("Requires exclusive native M5/full-FP32-matrix GPU lane")
        }
        let arch = GPU.deviceInfo().architecture
        guard arch.hasPrefix("applegpu_"), let device = arch.last else { throw XCTSkip("Unknown core block policy") }
        return device
    }
    private func selectedPlan(rows: Int, keys: Int, dtype: DType, sinks: Bool, device: Character) throws
        -> MiMoV26SplitKeyAttention.Plan {
        let text = ProcessInfo.processInfo.environment["MLX_SDPA_BLOCKS"] ?? ""
        guard text.isEmpty || Int(text) != nil else { throw XCTSkip("Unsupported block override") }
        return try XCTUnwrap(MiMoV26SplitKeyAttention.plan(rows: rows, keys: keys,
            elementBytes: dtype.size, hasSinks: sinks, deviceClass: device, blockOverride: Int(text) ?? 0))
    }
    private func fixture(rows: Int, keys: Int, dtype: DType) -> (MLXArray, MLXArray, MLXArray) {
        let q = (MLXRandom.normal([1, rows, 64, 192], key: MLXRandom.key(710)) * 0.75)
            .asType(dtype).transposed(0, 2, 1, 3)
        let k = (MLXRandom.normal([1, 4, keys + 17, 192], key: MLXRandom.key(711)) * 0.5)
            .asType(dtype)[0..., 0..., ..<keys, 0...]
        let v = (MLXRandom.normal([1, 4, keys + 17, 128], key: MLXRandom.key(712)) * 0.5)
            .asType(dtype)[0..., 0..., ..<keys, 0...]
        return (q, k, v)
    }
    private func reference(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
                           scale: Float, sinks: MLXArray?) -> MLXArray {
        let rows = q.dim(2), keys = k.dim(2)
        return concatenated((0..<rows).map { row in
            let end = keys - rows + row + 1
            return MLXFast.scaledDotProductAttention(
                queries: q[0..., 0..., row..<row + 1, 0...],
                keys: k[0..., 0..., ..<end, 0...], values: v[0..., 0..., ..<end, 0...],
                scale: scale, mask: .none, sinks: sinks)
        }, axis: 2)
    }
    private func exact(_ a: MLXArray, _ b: MLXArray, _ label: String) {
        eval(a, b)
        XCTAssertEqual(a.shape, b.shape, label); XCTAssertEqual(a.dtype, b.dtype, label)
        XCTAssertEqual(a.asData(access: .copy).data, b.asData(access: .copy).data, label)
    }
    private func diagnostics(_ a: MLXArray, _ b: MLXArray, _ label: String) {
        eval(a, b)
        let ab = a.view(dtype: .uint16).asArray(UInt16.self)
        let bb = b.view(dtype: .uint16).asArray(UInt16.self)
        let af = a.asType(.float32).asArray(Float.self), bf = b.asType(.float32).asArray(Float.self)
        var maxULP: UInt32 = 0, changed = 0
        var maxAbs: Float = 0
        func order(_ value: UInt16) -> UInt32 {
            value & 0x8000 == 0 ? UInt32(value) + 0x8000 : UInt32(~value)
        }
        for index in ab.indices {
            XCTAssertTrue(af[index].isFinite && bf[index].isFinite, label)
            let x = order(ab[index]), y = order(bb[index])
            maxULP = max(maxULP, x > y ? x - y : y - x)
            maxAbs = max(maxAbs, abs(af[index] - bf[index]))
            if ab[index] != bb[index] { changed += 1 }
        }
        print("MIMO_SPLITKEY_QK \(label) dtype=\(a.dtype) changed=\(changed) maxULP=\(maxULP) maxAbs=\(maxAbs)")
        // No new/widened numerical tolerance. This records a reduction-order
        // diagnostic; exact full-model greedy/state gates remain mandatory.
    }

    func testEveryScalarBlockBoundaryAndIndependentAllocationBounds() throws {
        typealias Kernel = MiMoV26SplitKeyAttention
        XCTAssertNil(Kernel.plan(rows: 0, keys: 4099, elementBytes: 2, hasSinks: false, deviceClass: "d"))
        XCTAssertNil(Kernel.plan(rows: 5, keys: 4099, elementBytes: 2, hasSinks: false, deviceClass: "d"))
        XCTAssertNil(Kernel.plan(rows: 1, keys: 1023, elementBytes: 2, hasSinks: false, deviceClass: "d"))
        XCTAssertNil(Kernel.plan(rows: 1, keys: 4095, elementBytes: 2, hasSinks: false, deviceClass: "g"))
        XCTAssertNil(Kernel.plan(rows: 1, keys: 4099, elementBytes: 4, hasSinks: false, deviceClass: "d"))
        for device: Character in ["d", "s", "g"] {
            for rows in 2...4 {
                for keys in [1024,1027,4096,4099,8192,8193,8196,16384,16387,32769,65536,65539,1_048_576] {
                    let a = Kernel.plan(rows: rows, keys: keys, elementBytes: 2, hasSinks: true, deviceClass: device)
                    let b = MiMoV26DecodeRows.serialBlocks(rows: rows, keys: keys, deviceClass: device)
                    XCTAssertEqual(a?.blocks, b)
                    if let a { XCTAssertEqual(a.visibleKeys, (0..<rows).map { keys - rows + $0 + 1 }) }
                }
            }
        }
        XCTAssertNil(Kernel.plan(rows: 1, keys: 4099, elementBytes: 2, hasSinks: true,
                                  deviceClass: "d", blockOverride: Int.max))
        let p = try XCTUnwrap(Kernel.plan(rows: 4, keys: 1_048_576, elementBytes: 2,
                                          hasSinks: true, deviceClass: "d"))
        XCTAssertEqual(p.blocks, 1024); XCTAssertEqual(p.threads, 256)
        XCTAssertEqual(p.threadgroupBytes, 7424)
        XCTAssertEqual(p.temporaryLogicalBytes, 136_314_880)
        var visited: [Int] = []
        let bounded = try p.boundedAllocationBytes { bytes in
            visited.append(bytes); return ((bytes + 4095) / 4096) * 4096
        }
        XCTAssertEqual(visited, p.allocationByteCounts)
        XCTAssertGreaterThanOrEqual(bounded, p.totalLogicalBytes)
        XCTAssertThrowsError(try p.boundedAllocationBytes { $0 - 1 })
        XCTAssertThrowsError(try p.boundedAllocationBytes { _ in Int.max })
    }

    func testShapeMaskAndActualCPUStreamRefusals() throws {
        try Device.withDefaultDevice(.cpu) {
            let q = MLXArray.zeros([1, 64, 3, 192], dtype: .bfloat16)
            let k = MLXArray.zeros([1, 4, 4099, 192], dtype: .bfloat16)
            let v = MLXArray.zeros([1, 4, 4099, 128], dtype: .bfloat16)
            func plan(_ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray,
                      _ mask: MLXFast.ScaledDotProductAttentionMaskMode, _ sinks: MLXArray? = nil)
                -> MiMoV26SplitKeyAttention.Plan? {
                MiMoV26SplitKeyAttention.makePlan(queries: queries, keys: keys, values: values,
                    scale: scale, mask: mask, sinks: sinks, deviceClass: "d")
            }
            XCTAssertNotNil(plan(q,k,v,.causal))
            XCTAssertNil(plan(q,k,v,.none))
            XCTAssertNil(plan(q,k,v,.array(MLXArray.ones([3,4099], dtype: .bool))))
            XCTAssertNil(plan(q,k,v,.causal,MLXArray.zeros([64], dtype: .float32)))
            XCTAssertNil(plan(q.asType(.float32),k.asType(.float32),v.asType(.float32),.causal))
            XCTAssertNil(plan(broadcast(q, to: [2,64,3,192]),k,v,.causal))
            XCTAssertNil(MiMoV26SplitKeyAttention.tryEncode(queries:q,keys:k,values:v,
                scale:scale,mask:.causal,sinks:nil,stream:.cpu))
        }
        // This negative actually exercises the TaskLocal stream gate while
        // the default Device remains GPU; no kernel is evaluated here.
        guard MiMoV26SplitKeyAttention.requested else {
            throw XCTSkip("Enable process-start candidate flag for TaskLocal stream negative")
        }
        try Device.withDefaultDevice(.gpu) {
            try Stream.withNewDefaultStream(device: .cpu) {
                let q = MLXArray.zeros([1,64,1,192],dtype:.bfloat16)
                let k = MLXArray.zeros([1,4,4099,192],dtype:.bfloat16)
                let v = MLXArray.zeros([1,4,4099,128],dtype:.bfloat16)
                XCTAssertNil(MiMoV26SplitKeyAttention.tryEncode(queries:q,keys:k,values:v,
                    scale:scale,mask:.none,sinks:nil))
            }
        }
    }

    func testScalarQKControlExactlyMatchesUnmodifiedCoreIncludingPartialPrecision() throws {
        let device = try requireMatrixGPU()
        for dtype in [DType.bfloat16,.float16] {
            for rows in 1...4 {
                for keys in [1100,4099,16400,65540] {
                    let (q,k,v) = fixture(rows:rows,keys:keys,dtype:dtype)
                    for hasSinks in [false,true] {
                        let sinks = hasSinks ? MLXArray((0..<64).map { Float($0 % 7) * 0.125 - 1.75 }).asType(dtype) : nil
                        let plan = try selectedPlan(rows:rows,keys:keys,dtype:dtype,sinks:hasSinks,device:device)
                        let e = MiMoV26SplitKeyAttention.encode(queries:q,keys:k,values:v,
                            scale:scale,sinks:sinks,plan:plan,matrixScores:false)
                        XCTAssertEqual(e.evaluationRoots.prefix(3).map(\.dtype), [.float32,.float32,.float32])
                        eval(e.evaluationRoots)
                        exact(e.output, reference(q,k,v,scale:scale,sinks:sinks), "scalar-control \(dtype)/\(rows)/\(keys)/\(hasSinks)")
                    }
                }
            }
        }
    }

    func testMatrixDyadicScoresKeepNativeScaleExpAndValueRecurrenceExactly() throws {
        let device = try requireMatrixGPU(), rows = 3, keys = 4101
        for dtype in [DType.bfloat16,.float16] {
            // Only three dyadic QK terms; exact sums discriminate dtype/scale
            // placement and nonlinear changes without demanding a new tolerance.
            var qValues = [Float](repeating:0,count:64 * rows * 192)
            for h in 0..<64 { for r in 0..<rows {
                qValues[(h * rows + r) * 192] = Float(h % 4 + 1) / 8
                qValues[(h * rows + r) * 192 + 95] = -Float(r + 1) / 16
                qValues[(h * rows + r) * 192 + 191] = 0.0625
            }}
            let q = MLXArray(qValues,[1,64,rows,192]).asType(dtype)
            let keyValues = (0..<(4 * keys * 192)).map { Float($0 % 13 - 6) / 32 }
            let k = MLXArray(keyValues,[1,4,keys,192]).asType(dtype)
            let v = (MLXRandom.normal([1,4,keys,128],key:MLXRandom.key(760)) * 0.75).asType(dtype)
            let sinks = MLXArray((0..<64).map { Float($0 % 5 - 2) / 4 }).asType(dtype)
            let p = try selectedPlan(rows:rows,keys:keys,dtype:dtype,sinks:true,device:device)
            let e = MiMoV26SplitKeyAttention.encode(queries:q,keys:k,values:v,scale:0.125,sinks:sinks,plan:p)
            eval(e.evaluationRoots)
            exact(e.output,reference(q,k,v,scale:0.125,sinks:sinks),"dyadic native authority")
        }
    }

    func testMatrixRowsAndStridedViewsAreBitExactWithOwnScalarPositions() throws {
        let device = try requireMatrixGPU()
        for dtype in [DType.bfloat16,.float16] {
            for rows in 2...4 {
                for keys in [4099,16400] {
                    let (q,k,v) = fixture(rows:rows,keys:keys,dtype:dtype)
                    let sinks = MLXArray((0..<128).map { Float($0 % 11 - 5) / 8 }).asType(dtype)[.stride(by:2)]
                    let p = try selectedPlan(rows:rows,keys:keys,dtype:dtype,sinks:true,device:device)
                    let e = MiMoV26SplitKeyAttention.encode(queries:q,keys:k,values:v,scale:scale,sinks:sinks,plan:p)
                    let qt = contiguous(q.transposed(0,1,3,2)).transposed(0,1,3,2)
                    let kt = contiguous(k.transposed(0,1,3,2)).transposed(0,1,3,2)
                    let vt = contiguous(v.transposed(0,1,3,2)).transposed(0,1,3,2)
                    let strided = MiMoV26SplitKeyAttention.encode(queries:qt,keys:kt,values:vt,scale:scale,sinks:sinks,plan:p)
                    exact(strided.output,e.output,"noncontiguous sequence/head/inner strides")
                    var individual: [MLXArray] = []
                    for r in 0..<rows {
                        let n = keys - rows + r + 1
                        let rp = try selectedPlan(rows:1,keys:n,dtype:dtype,sinks:true,device:device)
                        let one = MiMoV26SplitKeyAttention.encode(
                            queries:q[0...,0...,r..<r+1,0...],keys:k[0...,0...,..<n,0...],
                            values:v[0...,0...,..<n,0...],scale:scale,sinks:sinks,plan:rp)
                        individual.append(one.output)
                    }
                    exact(e.output,concatenated(individual,axis:2),"each native visible prefix")
                    let native = reference(q,k,v,scale:scale,sinks:sinks)
                    diagnostics(e.output,native,"matrix versus scalar authority \(rows)/\(keys)")
                    // An attention-local top1 witness, not vocabulary greedy proof.
                    XCTAssertEqual(argMax(e.output,axis:-1).asArray(Int32.self),argMax(native,axis:-1).asArray(Int32.self))
                }
            }
        }
    }

    func testFutureValuesNeverEnterEarlierRowsIncludingNaNAndFiniteExtremes() throws {
        let device = try requireMatrixGPU(), keys = 4101, rows = 4
        for dtype in [DType.bfloat16,.float16] {
            let q = (MLXArray.ones([1,64,rows,192],dtype:dtype) * 0.5).asType(dtype)
            let k = (MLXArray.ones([1,4,keys,192],dtype:dtype) * -0.25).asType(dtype)
            let prefix = MLXArray.zeros([1,4,keys-1,128],dtype:dtype)
            let p = try selectedPlan(rows:rows,keys:keys,dtype:dtype,sinks:false,device:device)
            for poison: Float in [32768,-32768,.nan] {
                let v = concatenated([prefix,MLXArray.full([1,4,1,128],values:MLXArray(poison),dtype:dtype)],axis:2)
                let e = MiMoV26SplitKeyAttention.encode(queries:q,keys:k,values:v,scale:scale,sinks:nil,plan:p)
                let first = e.output[0...,0...,..<(rows-1),0...]
                exact(first,MLXArray.zeros(first.shape,dtype:dtype),"future V structurally excluded")
                exact(first,reference(q,k,v,scale:scale,sinks:nil)[0...,0...,..<(rows-1),0...],"native prefix poison control")
            }
        }
    }

    func testEncodingDoesNotMutateFullKVAndEveryRollbackContinuesExactState() throws {
        let device = try requireMatrixGPU(), history = 1100, width = 4
        for dtype in [DType.bfloat16,.float16] {
            let kind = CBv2LayerKind(attention:.full,headDim:192,valueHeadDim:128,kvHeads:4,queryHeads:64)
            for kept in 1...width {
                let backend = CBv2ContiguousKVBackend(config:.init(bytesCapacity:64 << 20,kvDType:dtype))
                let aState = try backend.makeSequenceState(layerKinds:[kind],promptLength:0,maxLength:2048)
                let bState = try backend.makeSequenceState(layerKinds:[kind],promptLength:0,maxLength:2048)
                defer { backend.release(aState); backend.release(bState) }
                let a = try XCTUnwrap(aState[0]), b = try XCTUnwrap(bState[0])
                let (q,k,v) = fixture(rows:width,keys:history+width,dtype:dtype)
                _ = a.update(keys:k[0...,0...,..<history,0...],values:v[0...,0...,..<history,0...])
                _ = b.update(keys:k[0...,0...,..<history,0...],values:v[0...,0...,..<history,0...])
                let initialA = a.snapshot(), initialB = b.snapshot()
                eval(initialA.keys,initialA.values,initialB.keys,initialB.values)
                a.beginSpeculativeWrite()
                let (ak,av) = a.update(keys:k[0...,0...,history...,0...],values:v[0...,0...,history...,0...])
                let p = try selectedPlan(rows:width,keys:history+width,dtype:dtype,sinks:false,device:device)
                let e = MiMoV26SplitKeyAttention.encode(queries:q,keys:ak,values:av,scale:scale,sinks:nil,plan:p)
                XCTAssertEqual(a.absoluteOffset,history+width)
                XCTAssertEqual(e.evaluationRoots.count,6)
                eval(e.evaluationRoots)
                XCTAssertEqual(a.absoluteOffset,history+width)
                a.rollback(width-kept); a.commitSpeculativeWrite()
                _ = b.update(keys:k[0...,0...,history..<(history+kept),0...],
                             values:v[0...,0...,history..<(history+kept),0...])
                let sa = a.snapshot(), sb = b.snapshot()
                XCTAssertEqual(sa.offset,sb.offset); XCTAssertEqual(a.retainedCount,b.retainedCount)
                exact(sa.keys,sb.keys,"accepted target K"); exact(sa.values,sb.values,"accepted target V")
                let (_,nextK,nextV) = fixture(rows:1,keys:1,dtype:dtype)
                let aNext = a.update(keys:nextK,values:nextV), bNext = b.update(keys:nextK,values:nextV)
                let nextQ = q[0...,0...,0..<1,0...]
                let continuation = try selectedPlan(rows:1,keys:history+kept+1,dtype:dtype,sinks:false,device:device)
                let ea = MiMoV26SplitKeyAttention.encode(queries:nextQ,keys:aNext.0,values:aNext.1,
                    scale:scale,sinks:nil,plan:continuation)
                let eb = MiMoV26SplitKeyAttention.encode(queries:nextQ,keys:bNext.0,values:bNext.1,
                    scale:scale,sinks:nil,plan:continuation)
                exact(ea.output,eb.output,"candidate continuation after real row rollback")
            }
        }
        // This exercises genuine row storage, not an installed Engine native
        // work loan or admission transfer. Those remain separate open gates.
    }

    // A single active dimension removes any legal QK summation-order
    // ambiguity. Every array element is native-representable; only the
    // FP32 scale is deliberately not representable in BF16 or FP16.
    private func nonNativeScaleFixture(rows: Int, dtype: DType)
        -> (q: MLXArray, k: MLXArray, v: MLXArray, sinks: MLXArray) {
        let keys = 4099
        var q = [Float](repeating: 0, count: 64 * rows * 192)
        for head in 0..<64 { for row in 0..<rows {
            q[(head * rows + row) * 192] = 1
        }}
        var k = [Float](repeating: 0, count: 4 * keys * 192)
        var v = [Float](repeating: 0, count: 4 * keys * 128)
        for head in 0..<4 {
            for key in 0..<keys { k[(head * keys + key) * 192] = key == 0 ? 128 : -128 }
            for column in 0..<128 { v[head * keys * 128 + column] = 1 }
        }
        return (MLXArray(q, [1, 64, rows, 192]).asType(dtype),
                MLXArray(k, [1, 4, keys, 192]).asType(dtype),
                MLXArray(v, [1, 4, keys, 128]).asType(dtype),
                MLXArray(Array(repeating: Float(9.25), count: 64)).asType(dtype))
    }

    private final class ScaleOracleRoots {
        var arrays: [MLXArray] = []
    }

    /// Evaluate only actual arrays; on uncertain completion keep their owner
    /// until this isolated process exits. This is not a production work lease.
    private func assertScaleOracle(
        _ e: MiMoV26SplitKeyAttention.Encoding,
        fixture f: (q: MLXArray, k: MLXArray, v: MLXArray, sinks: MLXArray),
        roots: ScaleOracleRoots
    ) throws {
        try withError { errors in
            let native = reference(f.q, f.k, f.v, scale: scale, sinks: f.sinks)
            roots.arrays.append(native)
            let roundedScale = MLXArray(scale).asType(f.q.dtype).asType(.float32)
            roots.arrays.append(roundedScale)
            try errors.check()
            eval([roundedScale])
            try errors.check()
            let rounded = roundedScale.item(Float.self)
            XCTAssertNotEqual(rounded.bitPattern, scale.bitPattern,
                              "Scale must discriminate an early native-dtype cast")
            // Deliberately wrong precision, through the independent native
            // authority, not a production-code mutation or a new tolerance.
            let premature = reference(f.q, f.k, f.v, scale: rounded, sinks: f.sinks)
            roots.arrays.append(premature)
            try errors.check()
            eval(roots.arrays)
            try errors.check()
            let actualBytes = e.output.asData(access: .copy).data
            let nativeBytes = native.asData(access: .copy).data
            let roundedBytes = premature.asData(access: .copy).data
            try errors.check()
            XCTAssertEqual(actualBytes, nativeBytes,
                           "One-term FP32-scale matrix result must match native stored bytes")
            XCTAssertNotEqual(nativeBytes, roundedBytes,
                              "Fixture must reject premature Q-scale rounding for \(f.q.dtype)")
            print("MIMO_SPLITKEY_SCALE_ORACLE dtype=\(f.q.dtype) rows=\(f.q.dim(2))"
                + " scale=\(scale) prematureNativeScale=\(rounded)"
                + " nativeBytesExact=\(actualBytes == nativeBytes)"
                + " wrongPrecisionDiffers=\(nativeBytes != roundedBytes)")
        }
    }

    func testOneActiveDimensionKeepsNonNativeScaleExactlyAgainstNativeBytes() throws {
        let device = try requireMatrixGPU()
        for dtype in [DType.bfloat16, .float16] {
            for rows in [1, 3] {
                let roots = ScaleOracleRoots()
                var completed = false
                defer { if !completed { _ = Unmanaged.passRetained(roots) } }
                try withError { errors in
                    let f = nonNativeScaleFixture(rows: rows, dtype: dtype)
                    roots.arrays += [f.q, f.k, f.v, f.sinks]
                    let plan = try selectedPlan(rows: rows, keys: 4099, dtype: dtype,
                                                sinks: true, device: device)
                    let e = MiMoV26SplitKeyAttention.encode(queries: f.q, keys: f.k, values: f.v,
                        scale: scale, sinks: f.sinks, plan: plan)
                    roots.arrays += e.evaluationRoots + [e.output]
                    try errors.check()
                    try assertScaleOracle(e, fixture: f, roots: roots)
                    try errors.check()
                }
                completed = true
            }
        }
    }

    func testOptedInTryEncodeEvaluatesRealRootsAndMatchesNativeScaleOracle() throws {
        let device = try requireMatrixGPU()
        guard MiMoV26SplitKeyAttention.requested else {
            throw XCTSkip("Requires DARKBLOOM_MIMO_V26_SPLITKEY_QK=1 at process start")
        }
        for dtype in [DType.bfloat16, .float16] {
            for rows in [1, 3] {
                let roots = ScaleOracleRoots()
                var completed = false
                defer { if !completed { _ = Unmanaged.passRetained(roots) } }
                try withError { errors in
                    let f = nonNativeScaleFixture(rows: rows, dtype: dtype)
                    roots.arrays += [f.q, f.k, f.v, f.sinks]
                    let e = try XCTUnwrap(MiMoV26SplitKeyAttention.tryEncode(
                        queries: f.q, keys: f.k, values: f.v, scale: scale,
                        mask: rows == 1 ? .none : .causal, sinks: f.sinks, stream: .default),
                        "Eligible real opted-in path must not silently return nil")
                    roots.arrays += e.evaluationRoots + [e.output]
                    try errors.check()
                    let expectedPlan = try selectedPlan(rows: rows, keys: 4099, dtype: dtype,
                                                        sinks: true, device: device)
                    XCTAssertEqual(e.plan, expectedPlan)
                    XCTAssertEqual(e.output.shape, [1, 64, rows, 128])
                    XCTAssertEqual(e.output.dtype, dtype)
                    XCTAssertEqual(e.evaluationRoots.count, 6)
                    XCTAssertEqual(e.evaluationRoots.map(\.dtype),
                                   [.float32, .float32, .float32, dtype, .float32, dtype])
                    XCTAssertEqual(e.evaluationRoots.map(\.nbytes), e.plan.allocationByteCounts,
                                   "Logical geometry only, not materialized-byte credit")
                    try assertScaleOracle(e, fixture: f, roots: roots)
                    try errors.check()
                }
                completed = true
            }
        }
    }

    func testDefaultOffDeclinesOtherwiseEligibleGPUShape() throws {
        let device = try requireMatrixGPU()
        guard !MiMoV26SplitKeyAttention.requested else {
            throw XCTSkip("Run default-OFF control in a separate process without the candidate flag")
        }
        let f = nonNativeScaleFixture(rows: 1, dtype: .bfloat16)
        XCTAssertNotNil(MiMoV26SplitKeyAttention.makePlan(queries: f.q, keys: f.k, values: f.v,
            scale: scale, mask: .none, sinks: f.sinks, deviceClass: device))
        XCTAssertNil(MiMoV26SplitKeyAttention.tryEncode(queries: f.q, keys: f.k, values: f.v,
            scale: scale, mask: .none, sinks: f.sinks, stream: .default))
    }
}
