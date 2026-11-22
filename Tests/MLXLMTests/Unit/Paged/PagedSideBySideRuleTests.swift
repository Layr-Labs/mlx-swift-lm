import Foundation
import Testing

@testable import MLXLMCommon

extension UnitTests {

    /// The sizing and the length rule of the side-by-side paged pass A
    /// (`paged_attention_part_side_impl`).
    @Suite
    struct PagedSideBySideRuleTests {
        @Test func sizingFollowsTheHeadSplit() throws {
            // Qwen 3.5 27B full attention: 3 heads per threadgroup, 2 head groups.
            #expect(PagedAttentionKernel.sideBySideHeadsPerRound(headDim: 256, gqa: 6) == 3)
            // Gemma 4 global layers: 4 head groups; one merge record head per round fits 32 KB.
            #expect(PagedAttentionKernel.sideBySideHeadsPerRound(headDim: 512, gqa: 8) == 1)
            #expect(PagedAttentionKernel.sideBySideHeadsPerRound(headDim: 128, gqa: 16) == 8)
            // A kv head's query heads already share one threadgroup.
            #expect(PagedAttentionKernel.sideBySideHeadsPerRound(headDim: 128, gqa: 8) == nil)
            #expect(PagedAttentionKernel.sideBySideHeadsPerRound(headDim: 64, gqa: 4) == nil)
            for (d, gqa) in [(256, 6), (512, 8), (128, 16)] {
                let hred = try #require(
                    PagedAttentionKernel.sideBySideHeadsPerRound(headDim: d, gqa: gqa))
                let nsg = try #require(
                    PagedAttentionKernel.simdgroupsPerThreadgroup(headDim: d, gqa: gqa))
                let hpt = PagedAttentionKernel.headsPerThreadgroup(headDim: d, gqa: gqa)
                #expect(32 * nsg * (gqa / hpt) <= PagedAttentionKernel.sideBySideMaxThreads)
                #expect(
                    PagedAttentionKernel.sideBySideThreadgroupBytes(
                        headDim: d, gqa: gqa, simdgroups: nsg, headsPerRound: hred)
                        <= PagedAttentionKernel.threadgroupMemoryLimit)
            }
        }

        /// `sideBySideThreadgroupBytes` mirrors the buffers the side-by-side
        /// bodies declare.
        @Test func sideBodiesDeclareTheSizedBuffers() {
            let bodies = [
                PagedAttentionMSL.partBodySide, PagedAttentionMSL.partBodySideNoWrite,
                PagedSegmentAttention.sideBody(bindings: 2),
            ]
            for body in bodies {
                #expect(body.contains("threadgroup float q_smem[GQA * D];"))
                #expect(body.contains("threadgroup float red_smem[NSG * HRED * (D + 2)];"))
                #expect(body.contains("paged_attention_part_side_impl"))
            }
        }

        @Test func minimumLengthComesFromTheEnvironment() {
            let key = PagedAttentionKernel.sideBySideEnvironmentKey
            let fallback = PagedAttentionKernel.sideBySideDefaultTokens
            #expect(PagedAttentionKernel.sideBySideMinimum(environment: [:]) == fallback)
            #expect(PagedAttentionKernel.sideBySideMinimum(environment: [key: "0"]) == 0)
            #expect(PagedAttentionKernel.sideBySideMinimum(environment: [key: "8192"]) == 8192)
            #expect(PagedAttentionKernel.sideBySideMinimum(environment: [key: "-1"]) == fallback)
            #expect(PagedAttentionKernel.sideBySideMinimum(environment: [key: "x"]) == fallback)
            let rule = PagedAttentionKernel.sideBySideForDispatch
            #expect(rule(256, 6, 100_000, 0) == nil)
            #expect(rule(256, 6, 100, 101) == nil)
            #expect(rule(256, 6, 101, 101) == 3)
            #expect(rule(128, 8, 100_000, 1) == nil)
        }
    }
}
