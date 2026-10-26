import Testing

@testable import MLXLMCommon

extension UnitTests {
    /// Host-only scheduling and projection proof for the capture-boundary
    /// partition (`CBv2SchedulerConfig.demandedCaptureBoundaryMinimumTokens`), on
    /// GLM's serving geometry: one 2,048-token step budget shared by the prefill
    /// chunk and decode rows, no solo stripe, one partial prefill at a time.
    struct CBv2DemandedCaptureBoundaryTests {
        private func config(enabled: Bool = true) -> CBv2SchedulerConfig {
            var value = CBv2SchedulerConfig(
                maxConcurrentRequests: 4, maxBatchedTokensPerStep: 2_048,
                prefillChunkSize: 2_048, maxConcurrentPartialPrefills: 1, enablePrefixCache: true)
            value.demandedCaptureBoundaryMinimumTokens = enabled ? 1_024 : nil
            return value
        }

        private func range(
            _ config: CBv2SchedulerConfig, prompt: Int = 8_233, hint: Int? = 8_192,
            computed: Int = 0, proposed: Int = 2_048, reused: Bool = false, scoped: Bool = true
        ) -> Int {
            config.demandedCheckpointRange(
                promptTokens: prompt, hintTokens: hint, computedTokens: computed,
                proposed: proposed, armedSoloStripeTokens: nil, hasPrefixReuse: reused,
                requestAllowsCheckpoint: scoped, continuation: nil
            ).count
        }

        private func donor(
            _ id: UInt64, prompt: Int, hint: Int?, salt: String? = "tenant-fixture"
        ) -> CBv2Request {
            CBv2Request(
                id: .init(id), promptTokens: Array(repeating: 1, count: prompt), maxTokens: 1,
                cacheSalt: salt, prefixCheckpointTargetTokens: hint)
        }

        /// Prefill ranges of `request` until its prompt is computed. With
        /// `company`, a one-token request decodes in every step, as a served
        /// neighbour does.
        private func prefillRanges(
            _ config: CBv2SchedulerConfig, _ request: CBv2Request, company: Bool
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
            while record.numComputedTokens < request.promptTokens.count, ranges.count < 64 {
                let plan = scheduler.plan()
                if company {
                    #expect(plan.assignments.first { $0.id == neighbour.id }?.numTokens == 1)
                    confirmNeighbour()
                }
                if let count = plan.assignments.first(where: { $0.id == request.id })?.numTokens {
                    ranges.append(count)
                }
            }
            #expect(record.numComputedTokens == request.promptTokens.count)
            #expect(record.plannedPrefillChunkSize == 2_048)
            return ranges
        }

        private func ends(_ ranges: [Int]) -> [Int] {
            ranges.reduce(into: [Int]()) { $0.append(($0.last ?? 0) + $1) }
        }

        /// The qualification's 32k prompt: solo ranges are 2,048-aligned, and the
        /// last one now ends on the hinted 32,512 instead of leaving 30,720 as the
        /// deepest boundary.
        @Test func soloDonorEndsARangeOnTheHintedLatestBoundary() throws {
            let ranges = try prefillRanges(
                config(), donor(1, prompt: 32_682, hint: 32_512), company: false)
            #expect(ranges == Array(repeating: 2_048, count: 15) + [1_792, 170])
            #expect(ends(ranges).contains(32_512))
            let ordinary = try prefillRanges(
                config(enabled: false), donor(2, prompt: 32_682, hint: 32_512), company: false)
            #expect(ordinary == Array(repeating: 2_048, count: 15) + [1_962])
        }

        /// Beside a decoding request every chunk is 2,047 tokens and no range end
        /// is aligned. The range that would stop 4 tokens short of 8,192 ends on
        /// the query grid at 8,064, so the range that reaches the boundary is one
        /// query block long.
        @Test func companyShortenedRangesStillEndOnTheBoundary() throws {
            let ranges = try prefillRanges(
                config(), donor(3, prompt: 8_233, hint: 8_192), company: true)
            #expect(ranges == [2_047, 2_047, 2_047, 1_923, 128, 41])
            #expect(ends(ranges).filter { $0.isMultiple(of: 256) } == [8_192])
            let ordinary = try prefillRanges(
                config(enabled: false), donor(4, prompt: 8_233, hint: 8_192), company: true)
            #expect(ordinary == [2_047, 2_047, 2_047, 2_047, 45])
            #expect(ends(ordinary).filter { $0 < 8_233 }.allSatisfy { !$0.isMultiple(of: 256) })
        }

        /// A hint below the latest boundary is the fork target: it gets its own
        /// range end, then the latest one does.
        @Test func forkTargetAndLatestEachEndARange() throws {
            let ranges = try prefillRanges(
                config(), donor(5, prompt: 9_500, hint: 7_000), company: false)
            #expect(ranges == [2_048, 2_048, 2_048, 768, 2_048, 512, 28])
            #expect(
                ends(ranges).filter { $0.isMultiple(of: 256) && $0 >= 6_144 }
                    == [6_144, 6_912, 8_960, 9_472])
            let company = try prefillRanges(
                config(), donor(6, prompt: 9_500, hint: 7_000), company: true)
            #expect(ends(company).contains(6_912))
            #expect(ends(company).contains(9_472))
        }

        /// Without a coordinator the store writes every donation, so the latest
        /// boundary is cut. A fleet-novel prompt (a hint below the floor) is not
        /// written, so it keeps the ordinary geometry and pays no extra range.
        @Test func writeDemandDecidesWhetherAnyRangeIsCut() {
            func tail(_ hint: Int?) -> Int {
                range(config(), prompt: 32_682, hint: hint, computed: 30_720, proposed: 1_962)
            }
            #expect(tail(nil) == 1_792)
            for hint in [0, 512, 1_023] { #expect(tail(hint) == 1_962) }
            #expect(tail(1_024) == 1_792)
        }

        @Test func unscopedDisabledAndShortPromptsKeepTheirRange() {
            #expect(range(config(), computed: 6_144, proposed: 2_089, scoped: false) == 2_089)
            #expect(range(config(enabled: false), computed: 6_144, proposed: 2_089) == 2_089)
            var off = config()
            off.enablePrefixCache = false
            #expect(range(off, computed: 6_144, proposed: 2_089) == 2_089)
            // The deepest boundary of a 1,024-token prompt (768) is below the floor.
            #expect(range(config(), prompt: 1_024, hint: nil, proposed: 1_024) == 1_024)
            // A 1,500-token prompt gains a 1,280 boundary the 2,048 grid never had.
            #expect(range(config(), prompt: 1_500, hint: nil, proposed: 1_500) == 1_280)
            // A range already ending on the boundary, or starting on the latest
            // one, is unchanged.
            #expect(range(config(), computed: 6_144, proposed: 2_048) == 2_048)
            #expect(range(config(), computed: 8_192, proposed: 41) == 41)
        }

        /// A chunk-agnostic adopter captures above its restore like a cold donor:
        /// the fork target (11,776), then the latest (12,032).
        @Test func chunkAgnosticAdopterIsCutAboveItsRestore() {
            func cut(_ computed: Int, _ proposed: Int) -> Int {
                range(
                    config(), prompt: 12_100, hint: 12_000, computed: computed,
                    proposed: proposed, reused: true)
            }
            #expect(cut(8_192, 2_048) == 2_048)
            #expect(cut(10_240, 1_860) == 1_536)
            #expect(cut(11_776, 324) == 256)
            #expect(cut(12_032, 68) == 68)
        }

        /// A shape-bound adopter (a forced `recurrentChunkSize`) never qualifies,
        /// so no boundary can shorten the chunk it must run whole.
        @Test func shapeBoundAdopterIsNeverCut() throws {
            let scheduler = SchedulerV2(config: config())
            let record = try scheduler.enqueue(donor(7, prompt: 8_233, hint: 8_192))
            #expect(record.canScheduleDemandedShortCheckpoint)
            var plan = try #require(
                CBv2PrefixReuseCapability.derive(
                    layerKinds: CBv2SchedFixtures.tinyLayerKinds(), backend: .contiguousUnquantized
                ).plan(matchedBoundary: 6_144))
            plan.recurrentChunkSize = 2_048
            plan.recurrentPromptLength = 8_233
            record.prefixReusePlan = plan
            #expect(!record.canScheduleDemandedShortCheckpoint)
            plan.recurrentChunkSize = nil
            record.prefixReusePlan = plan
            #expect(record.canScheduleDemandedShortCheckpoint)
        }

        /// The first-token projection charges the same extra ranges.
        @Test func deadlineProjectionCountsTheCutRanges() throws {
            for enabled in [false, true] {
                let scheduler = SchedulerV2(config: config(enabled: enabled))
                let request = donor(8, prompt: 32_682, hint: 32_512)
                try scheduler.enqueue(request)
                guard
                    case .bounded(let work, _) = scheduler.firstTokenWorkProjection(for: request.id)
                else {
                    Issue.record("text-work projection must be bounded")
                    return
                }
                #expect(work.prefillTokens == 32_682)
                #expect(work.scheduledSteps == (enabled ? 17 : 16))
            }
        }
    }
}
