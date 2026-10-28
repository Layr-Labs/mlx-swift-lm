import Testing

@testable import MLXLMCommon

extension UnitTests {
    /// Host-only scheduling and projection proof for the end-of-prompt
    /// partition (`CBv2SchedulerConfig.demandedLatestCheckpointMinimumTokens`)
    /// on dense Qwen's serving geometry: 512-token chunks, a 4,096-token solo
    /// stripe, one partial prefill at a time, and its short-checkpoint
    /// partition beside it.
    struct CBv2DemandedLatestCheckpointTests {
        private func config(latest: Bool = true) -> CBv2SchedulerConfig {
            var value = CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 2_048, prefillChunkSize: 512,
                soloPrefillStripeTokens: 4_096, maxConcurrentPartialPrefills: 1,
                enablePrefixCache: true)
            value.demandedShortCheckpointMinimumTokens = 1_024
            value.demandedLatestCheckpointMinimumTokens = latest ? 1_024 : nil
            return value
        }

        private func donor(
            _ id: UInt64, prompt: Int, hint: Int?, salt: String? = "tenant-fixture"
        ) -> CBv2Request {
            CBv2Request(
                id: .init(id), promptTokens: Array(repeating: 1, count: prompt), maxTokens: 1,
                cacheSalt: salt, prefixCheckpointTargetTokens: hint)
        }

        /// Prefill ranges of `request` until its prompt is computed. With
        /// `company`, a one-token request decodes in every step.
        private func prefillRanges(
            _ config: CBv2SchedulerConfig, _ request: CBv2Request, company: Bool = false
        ) throws -> [Int] {
            let scheduler = SchedulerV2(config: config)
            let neighbour = CBv2Request(id: .init(9_999), promptTokens: [1], maxTokens: 512)
            func confirmNeighbour() {
                scheduler.markPendingSamples(ids: [neighbour.id])
                scheduler.recordSampled(id: neighbour.id, token: 2)
            }
            if company {
                try scheduler.enqueue(neighbour)
                #expect(scheduler.plan().assignments.map(\.numTokens) == [1])
                confirmNeighbour()
            }
            let record = try scheduler.enqueue(request)
            var ranges: [Int] = []
            while record.numComputedTokens < request.promptTokens.count, ranges.count < 256 {
                let plan = scheduler.plan()
                if company { confirmNeighbour() }
                if let count = plan.assignments.first(where: { $0.id == request.id })?.numTokens {
                    ranges.append(count)
                }
            }
            #expect(record.numComputedTokens == request.promptTokens.count)
            return ranges
        }

        private func ends(_ ranges: [Int]) -> [Int] {
            ranges.reduce(into: [Int]()) { $0.append(($0.last ?? 0) + $1) }
        }

        /// The conversation qualification's first turn: the last stripe ends at
        /// 28,672, and the next turn would otherwise re-prefill 1,492 tokens.
        @Test func soloDonorEndsARangeJustBelowItsPromptEnd() throws {
            let ranges = try prefillRanges(config(), donor(1, prompt: 30_164, hint: nil))
            #expect(ranges == Array(repeating: 4_096, count: 7) + [1_280, 212])
            #expect(ends(ranges).contains(29_952))
            let ordinary = try prefillRanges(
                config(latest: false), donor(2, prompt: 30_164, hint: nil))
            #expect(ordinary == Array(repeating: 4_096, count: 7) + [1_492])
        }

        /// Only the cut range runs outside packed prefill, which would disarm
        /// its capture; the stripes and the tail keep the packing policy.
        @Test func onlyTheCutRangeIsMarkedForCapture() throws {
            let scheduler = SchedulerV2(config: config())
            let request = donor(12, prompt: 30_164, hint: nil)
            let record = try scheduler.enqueue(request)
            var markedEnds: [Int] = []
            for _ in 0 ..< 16 where record.numComputedTokens < request.promptTokens.count {
                if scheduler.plan().demandedShortCheckpointRows.contains(request.id) {
                    markedEnds.append(record.numComputedTokens)
                }
            }
            #expect(record.numComputedTokens == request.promptTokens.count)
            #expect(markedEnds == [29_952])
        }

        /// The boundary keeps `demandedLatestPromptEndMarginTokens` below the
        /// prompt end, so a re-rendered generation prompt cannot diverge before
        /// it: 13,568 is aligned but only 32 tokens from the end of a
        /// 13,600-token prompt, so 13,312 is cut instead.
        @Test func boundaryKeepsAMarginBelowThePromptEnd() throws {
            #expect(CBv2SchedulerConfig.demandedLatestPromptEndMarginTokens == 64)
            let near = try prefillRanges(config(), donor(3, prompt: 13_600, hint: nil))
            #expect(ends(near) == [4_096, 8_192, 12_288, 13_312, 13_600])
            let clear = try prefillRanges(config(), donor(4, prompt: 13_700, hint: nil))
            #expect(ends(clear) == [4_096, 8_192, 12_288, 13_568, 13_700])
        }

        /// A cut that saves less than the floor over its range's start costs an
        /// extra prefill range and checkpoint for little reuse, so the tail
        /// stays one ordinary range.
        @Test func shortGainKeepsTheOrdinaryTail() throws {
            // The boundary 49,664 saves 512 tokens over the 49,152 stripe end.
            let small = try prefillRanges(config(), donor(5, prompt: 49_852, hint: nil))
            #expect(ends(small).suffix(2) == [49_152, 49_852])
            // 1,024 tokens is enough.
            let enough = try prefillRanges(config(), donor(6, prompt: 50_300, hint: nil))
            #expect(ends(enough).suffix(3) == [49_152, 50_176, 50_300])
        }

        /// Beside a decoding request the plain chunk is 512 tokens, so no cut can
        /// save the floor and the ordinary geometry stays.
        @Test func companyChunksAreNeverCut() throws {
            let busy = try prefillRanges(
                config(), donor(7, prompt: 9_500, hint: nil), company: true)
            let ordinary = try prefillRanges(
                config(latest: false), donor(8, prompt: 9_500, hint: nil), company: true)
            #expect(busy == ordinary)
        }

        /// The short-checkpoint partition still ends a range on the hinted fork
        /// target; the end-of-prompt boundary is cut after it.
        @Test func composesWithTheShortPartition() throws {
            let ranges = try prefillRanges(config(), donor(9, prompt: 3_500, hint: 2_048))
            #expect(ends(ranges) == [2_048, 3_328, 3_500])
            let shortOnly = try prefillRanges(
                config(latest: false), donor(10, prompt: 3_500, hint: 2_048))
            #expect(ends(shortOnly) == [2_048, 3_500])
        }

        /// A fleet-novel prompt (a hint below the floor) is not written, and an
        /// unscoped or cache-disabled request takes no checkpoint, so none of
        /// them pays an extra range.
        @Test func writeDemandAndScopeDecideWhetherARangeIsCut() {
            func tail(_ config: CBv2SchedulerConfig, hint: Int?, scoped: Bool = true) -> Int {
                config.demandedCheckpointRange(
                    promptTokens: 30_164, hintTokens: hint, computedTokens: 28_672, proposed: 1_492,
                    armedSoloStripeTokens: 4_096, hasPrefixReuse: false,
                    requestAllowsCheckpoint: scoped, continuation: nil
                ).count
            }
            #expect(tail(config(), hint: nil) == 1_280)
            #expect(tail(config(), hint: 1_024) == 1_280)
            for hint in [0, 512, 1_023] { #expect(tail(config(), hint: hint) == 1_492) }
            #expect(tail(config(), hint: nil, scoped: false) == 1_492)
            var off = config()
            off.enablePrefixCache = false
            #expect(tail(off, hint: nil) == 1_492)
            #expect(tail(config(latest: false), hint: nil) == 1_492)
        }

        /// An adopter restored at 29,952 is cut above its restore like a donor.
        @Test func adopterIsCutAboveItsRestore() {
            func cut(_ computed: Int, _ proposed: Int) -> Int {
                config().demandedCheckpointRange(
                    promptTokens: 35_279, hintTokens: 30_155, computedTokens: computed,
                    proposed: proposed, armedSoloStripeTokens: 4_096, hasPrefixReuse: true,
                    requestAllowsCheckpoint: true, continuation: nil
                ).count
            }
            #expect(cut(29_952, 4_096) == 4_096)
            #expect(cut(34_048, 1_231) == 1_024)
            #expect(cut(35_072, 207) == 207)
        }

        /// The capture-boundary partition (GLM) already ends ranges below the
        /// prompt end, so it is used unchanged when both are configured.
        @Test func captureBoundaryPartitionTakesPrecedence() {
            var both = config()
            both.demandedCaptureBoundaryMinimumTokens = 1_024
            var captureOnly = config(latest: false)
            captureOnly.demandedCaptureBoundaryMinimumTokens = 1_024
            for computed in stride(from: 0, to: 30_164, by: 512) {
                let proposed = min(4_096, 30_164 - computed)
                func count(_ config: CBv2SchedulerConfig) -> Int {
                    config.demandedCheckpointRange(
                        promptTokens: 30_164, hintTokens: 20_000, computedTokens: computed,
                        proposed: proposed, armedSoloStripeTokens: 4_096, hasPrefixReuse: false,
                        requestAllowsCheckpoint: true, continuation: nil
                    ).count
                }
                #expect(count(both) == count(captureOnly))
            }
        }

        /// The first-token projection charges the same extra range.
        @Test func deadlineProjectionCountsTheCutRange() throws {
            for latest in [false, true] {
                let scheduler = SchedulerV2(config: config(latest: latest))
                let request = donor(11, prompt: 30_164, hint: nil)
                try scheduler.enqueue(request)
                guard
                    case .bounded(let work, _) = scheduler.firstTokenWorkProjection(for: request.id)
                else {
                    Issue.record("text-work projection must be bounded")
                    return
                }
                #expect(work.prefillTokens == 30_164)
                #expect(work.scheduledSteps == (latest ? 9 : 8))
            }
        }
    }
}
