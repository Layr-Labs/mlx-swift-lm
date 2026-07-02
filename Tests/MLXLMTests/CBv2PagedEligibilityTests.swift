// CBv2PagedEligibilityTests.swift
//
// Static eligibility of attention shapes for the paged decode kernels.
// The Swift-side threadgroup-memory model (PagedAttentionKernel
// .partThreadgroupBytes / ineligibilityReason) must match the shader's
// allocations byte-for-byte, and PagedKVBackend must refuse over-budget
// shapes at construction: dispatching one is an UNCATCHABLE Metal fatal
// ("Threadgroup memory size (32832) exceeds the maximum (32768)"), not a
// thrown error. Regression anchor: Gemma-4 global layers (headDim 512,
// GQA 8) passed the old head-dim-only eligibility and killed the process
// on the first dispatch. No model weights or GPU dispatch required.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("CBv2PagedEligibility")
struct CBv2PagedEligibilityTests {

    // MARK: - Helpers

    private func fullKind(
        headDim: Int, kvHeads: Int, queryHeads: Int, sinks: Bool = false
    ) -> CBv2LayerKind {
        CBv2LayerKind(
            attention: .full, hasSinks: sinks, headDim: headDim, kvHeads: kvHeads,
            queryHeads: queryHeads)
    }

    private func poolConfig() -> PagedKVPoolConfig {
        PagedKVPoolConfig(
            capacityBytes: 8 << 20, maxPrefillChunk: 64,
            nominalMaxSequenceLength: 1024)
    }

    private func expectIneligible(
        _ kinds: [CBv2LayerKind], reasonContains fragment: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        do {
            _ = try PagedKVBackend(layerKinds: kinds, config: poolConfig())
            Issue.record(
                "expected backendIneligible for \(kinds)", sourceLocation: sourceLocation)
        } catch CBv2KVError.backendIneligible(let reason) {
            #expect(
                reason.contains(fragment),
                "reason \"\(reason)\" should mention \"\(fragment)\"",
                sourceLocation: sourceLocation)
        } catch {
            Issue.record(
                "unexpected error: \(error)", sourceLocation: sourceLocation)
        }
    }

    // MARK: - (c) Budget math vs the shader's constants

    @Test func budgetFormulaMatchesShaderAllocations() {
        // The shader allocates q_smem[GQA*D] + red_smem[NSG*GQA*(D+2)]
        // floats (see pagedattention.metal "single source of truth").
        // Exact bytes for the shape that fataled in the field: Gemma-4
        // global layers, (8*512 + 1*8*514) * 4 = 32,832 B — 64 B over the
        // 32,768 B Metal cap even at one simdgroup.
        #expect(
            PagedAttentionKernel.partThreadgroupBytes(headDim: 512, gqa: 8, simdgroups: 1)
                == 32832)
        #expect(PagedAttentionKernel.threadgroupMemoryLimit == 32768)
        #expect(PagedAttentionKernel.mergeRecordMetaFloats == 2)

        // General formula on a spread of shapes.
        for (d, g, n) in [(64, 8, 8), (128, 8, 4), (256, 2, 2), (512, 1, 8), (512, 4, 2)] {
            let expected = (g * d + n * g * (d + 2)) * 4
            #expect(
                PagedAttentionKernel.partThreadgroupBytes(headDim: d, gqa: g, simdgroups: n)
                    == expected,
                "d=\(d) gqa=\(g) nsg=\(n)")
        }
    }

    /// Drift guard: the generated kernel bodies must still allocate exactly
    /// the buffers the Swift budget models — two float threadgroup buffers
    /// in each pass-A variant (q_smem, red_smem with RSTRIDE = D + 2), none
    /// in pass B or the bulk-write kernel.
    @Test func shaderSourceStillMatchesBudgetModel() {
        for part in [PagedAttentionMSL.partBody, PagedAttentionMSL.partBodyNoWrite] {
            #expect(part.contains("threadgroup float q_smem[GQA * D];"))
            #expect(part.contains("threadgroup float red_smem[NSG * GQA * (D + 2)];"))
            #expect(
                part.components(separatedBy: "threadgroup float").count - 1 == 2,
                "part bodies must allocate exactly the two modeled buffers")
        }
        #expect(
            !PagedAttentionMSL.mergeBody.contains("threadgroup float"),
            "merge body must allocate no threadgroup memory")
        #expect(
            !PagedAttentionMSL.writeBody.contains("threadgroup"),
            "bulk-write body must allocate no threadgroup memory")
        // RSTRIDE in the .metal impl mirrors mergeRecordMetaFloats.
        #expect(PagedAttentionMSL.header.contains("RSTRIDE = D + 2"))
    }

    @Test func simdgroupSelectionRespectsBudget() {
        // Largest NSG whose staging + merge buffers fit 32 KB.
        #expect(PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: 64, gqa: 8) == 8)
        #expect(PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: 128, gqa: 8) == 4)
        #expect(PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: 256, gqa: 8) == 2)
        #expect(PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: 512, gqa: 4) == 2)
        // Gemma-4 global layers: over budget even at NSG=1 → nil.
        #expect(PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: 512, gqa: 8) == nil)

        // Whenever an NSG is returned it fits the budget, and the next
        // larger candidate would not (largest-fit invariant).
        let candidates = PagedAttentionKernel.simdgroupCandidates
        for d in PagedAttentionKernel.supportedHeadDims.sorted() {
            for g in [1, 2, 4, 8, 16] {
                guard
                    let n = PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: d, gqa: g)
                else { continue }
                let bytes = PagedAttentionKernel.partThreadgroupBytes(
                    headDim: d, gqa: g, simdgroups: n)
                #expect(
                    bytes <= PagedAttentionKernel.threadgroupMemoryLimit,
                    "d=\(d) gqa=\(g): selected NSG=\(n) over budget")
                if let i = candidates.firstIndex(of: n), i > 0 {
                    let bigger = PagedAttentionKernel.partThreadgroupBytes(
                        headDim: d, gqa: g, simdgroups: candidates[i - 1])
                    #expect(
                        bigger > PagedAttentionKernel.threadgroupMemoryLimit,
                        "d=\(d) gqa=\(g): NSG=\(candidates[i - 1]) would also fit")
                }
            }
        }
    }

    // MARK: - (a) Over-budget shapes are refused at construction

    @Test func gemma4GlobalLayerShapeIsIneligible() {
        let reason = PagedAttentionKernel.ineligibilityReason(headDim: 512, gqa: 8)
        #expect(reason?.contains("threadgroup memory") == true)
        #expect(reason?.contains("32832") == true)

        // f16 pool slabs are the default; PagedKVBackend.init must throw
        // backendIneligible before any kernel is built or dispatched.
        expectIneligible(
            [fullKind(headDim: 512, kvHeads: 1, queryHeads: 8)],
            reasonContains: "threadgroup memory")
    }

    /// Per-layer-kind: ONE over-budget layer makes the whole model
    /// ineligible, even when every other layer fits.
    @Test func anyOverBudgetLayerMakesModelIneligible() {
        expectIneligible(
            [
                fullKind(headDim: 128, kvHeads: 8, queryHeads: 64),
                fullKind(headDim: 512, kvHeads: 1, queryHeads: 8),
            ],
            reasonContains: "layer 1")
    }

    /// Unsupported head dims still refuse through the same kernel-level
    /// eligibility path.
    @Test func unsupportedHeadDimIsIneligible() {
        #expect(PagedAttentionKernel.ineligibilityReason(headDim: 96, gqa: 2) != nil)
        expectIneligible(
            [fullKind(headDim: 96, kvHeads: 2, queryHeads: 4)],
            reasonContains: "headDim 96")
    }

    // MARK: - (b) Fleet shapes remain eligible

    @Test func fleetShapesRemainEligible() throws {
        // d128 / GQA 8 (dense fleet shape).
        #expect(PagedAttentionKernel.ineligibilityReason(headDim: 128, gqa: 8) == nil)
        let dense = try PagedKVBackend(
            layerKinds: [fullKind(headDim: 128, kvHeads: 8, queryHeads: 64)],
            config: poolConfig())
        #expect(dense.layerKinds.count == 1)

        // d64 / GQA 8 with attention sinks (GPT-OSS shape). Sinks are a
        // kernel parameter folded in at merge time — they add no
        // threadgroup memory and must not affect eligibility.
        #expect(PagedAttentionKernel.ineligibilityReason(headDim: 64, gqa: 8) == nil)
        let sinks = try PagedKVBackend(
            layerKinds: [fullKind(headDim: 64, kvHeads: 8, queryHeads: 64, sinks: true)],
            config: poolConfig())
        #expect(sinks.layerKinds.count == 1)

        // d512 stays supported at small GQA (headDim inclusion alone was
        // never the problem).
        #expect(PagedAttentionKernel.ineligibilityReason(headDim: 512, gqa: 1) == nil)
    }
}
